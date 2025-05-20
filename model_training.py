import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm.auto import tqdm
import csv
from typing import Dict, Tuple, List, Optional, Any, Union

def evaluate_model_accuracy(model, data_loader, target_vocab, source_vocab, device):
    
    model.eval()
    correct_count = incorrect_count = 0
    
    # Lists for storing prediction details
    successful_sources = []
    successful_targets = []
    successful_predictions = []
    
    failed_sources = []
    failed_targets = []
    failed_predictions = []
    
    with torch.no_grad():
        for source_batch, length_batch, target_batch in data_loader:
            # Move batches to device
            source_batch = source_batch.to(device)
            length_batch = length_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Generate predictions
            predictions = model.generate_greedy(
                source_batch, 
                length_batch, 
                target_vocab, 
                max_length=target_batch.size(1)
            )
            
            # Check each prediction in the batch
            for idx in range(source_batch.size(0)):
                # Convert to strings for comparison
                pred_text = target_vocab.detokenize(predictions[idx].cpu().tolist())
                gold_text = target_vocab.detokenize(target_batch[idx, 1:].cpu().tolist())  # Skip start token
                source_text = source_vocab.detokenize(source_batch[idx].cpu().tolist())
                
                # Check if prediction matches target
                is_match = (pred_text == gold_text)
                
                # Update counters and store results
                if is_match:
                    correct_count += 1
                    successful_sources.append(source_text)
                    successful_targets.append(gold_text)
                    successful_predictions.append(pred_text)
                else:
                    incorrect_count += 1
                    failed_sources.append(source_text)
                    failed_targets.append(gold_text)
                    failed_predictions.append(pred_text)
    
    # Calculate accuracy
    total_count = correct_count + incorrect_count
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    # Return accuracy and detailed results
    return (
        accuracy, 
        (successful_sources, successful_targets, successful_predictions),
        (failed_sources, failed_targets, failed_predictions)
    )

def save_results_to_csv(sources, targets, predictions, filename):
    """Save prediction results to a CSV file for analysis"""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Source', 'Target', 'Prediction'])
        writer.writerows(zip(sources, targets, predictions))
    
    return filename

def train_transliteration_model(
    model, 
    data_loaders, 
    source_vocab, 
    target_vocab, 
    device,
    config,
    model_save_path=None,
    enable_wandb_logging=True
):
   
    # Loss function ignores padding tokens
    criterion = nn.CrossEntropyLoss(ignore_index=target_vocab.pad_index)
    
    # Initialize optimizer based on config
    if config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer.lower() == 'nadam':
        optimizer = optim.NAdam(model.parameters(), lr=config.lr)
    else:
        # Default to Adam
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # Track best validation performance
    best_validation_accuracy = 0.0
    
    # Mapping for data loader keys
    loader_keys = {
        'train': 'train',
        'dev': 'validation', 
        'test': 'test'
    }
    
    # Training loop
    for epoch in tqdm(range(1, config.epochs + 1), desc="Training Progress"):
        model.train()
        epoch_loss = 0.0
        
        # Process training batches with progress tracking
        train_iterator = tqdm(
            data_loaders[loader_keys['train']], 
            desc=f"Epoch {epoch}/{config.epochs}", 
            leave=False
        )
        
        for source_batch, length_batch, target_batch in train_iterator:
            # Move batches to device
            source_batch = source_batch.to(device)
            length_batch = length_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass with teacher forcing
            output_logits = model(
                source_batch, 
                length_batch, 
                target_batch, 
                teacher_forcing_ratio=config.teacher_forcing
            )
            
            # Calculate loss
            batch_loss = criterion(
                output_logits.reshape(-1, output_logits.size(-1)),
                target_batch[:, 1:].reshape(-1)  # Shift target by 1 (skip start token)
            )
            
            # Backward pass
            batch_loss.backward()
            
            # Clip gradients to prevent explosion
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += batch_loss.item()
            
            # Update progress bar
            train_iterator.set_postfix(loss=batch_loss.item())
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / len(data_loaders[loader_keys['train']])
        
        # Validation loss
        model.eval()
        validation_loss = 0.0
        
        with torch.no_grad():
            for source_batch, length_batch, target_batch in data_loaders[loader_keys['dev']]:
                # Move batches to device
                source_batch = source_batch.to(device)
                length_batch = length_batch.to(device)
                target_batch = target_batch.to(device)
                
                # Forward pass without teacher forcing
                output_logits = model(
                    source_batch, 
                    length_batch, 
                    target_batch, 
                    teacher_forcing_ratio=0.0
                )
                
                # Calculate loss
                batch_loss = criterion(
                    output_logits.reshape(-1, output_logits.size(-1)),
                    target_batch[:, 1:].reshape(-1)
                )
                
                # Accumulate loss
                validation_loss += batch_loss.item()
        
        # Calculate average validation loss
        avg_validation_loss = validation_loss / len(data_loaders[loader_keys['dev']])
        
        # Compute accuracy on training and validation sets
        train_results = evaluate_model_accuracy(
            model, 
            data_loaders[loader_keys['train']], 
            target_vocab, 
            source_vocab, 
            device
        )
        train_accuracy = train_results[0]
        
        validation_results = evaluate_model_accuracy(
            model, 
            data_loaders[loader_keys['dev']], 
            target_vocab, 
            source_vocab, 
            device
        )
        validation_accuracy = validation_results[0]
        
        # Save model if it's the best so far
        if validation_accuracy > best_validation_accuracy and model_save_path:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f" New best model saved with validation accuracy: {validation_accuracy:.4f}")
            
            # Save detailed prediction analysis at milestones
            if epoch == config.epochs or epoch % 5 == 0:
                correct_data = validation_results[1]
                incorrect_data = validation_results[2]
                
                # Save correct predictions
                save_results_to_csv(
                    correct_data[0], 
                    correct_data[1], 
                    correct_data[2],
                    f"correct_predictions_epoch_{epoch}.csv"
                )
                
                # Save incorrect predictions
                save_results_to_csv(
                    incorrect_data[0], 
                    incorrect_data[1], 
                    incorrect_data[2],
                    f"incorrect_predictions_epoch_{epoch}.csv"
                )
        
        # Log metrics
        print(f"Epoch {epoch}/{config.epochs}:")
        print(f"  Train: Loss={avg_train_loss:.4f}, Accuracy={train_accuracy:.4f}")
        print(f"  Validation: Loss={avg_validation_loss:.4f}, Accuracy={validation_accuracy:.4f}")
        
        # Log to wandb if enabled
        if enable_wandb_logging:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'validation_loss': avg_validation_loss,
                'train_accuracy': train_accuracy * 100,  # As percentage
                'validation_accuracy': validation_accuracy * 100  # As percentage
            })
    
    # Final evaluation on test set
    test_results = evaluate_model_accuracy(
        model, 
        data_loaders[loader_keys['test']], 
        target_vocab, 
        source_vocab, 
        device
    )
    test_accuracy = test_results[0]
    
    print(f"Final test accuracy: {test_accuracy:.4f}")
    
    # Log test accuracy
    if enable_wandb_logging:
        wandb.log({'test_accuracy': test_accuracy * 100})  # As percentage
    
    # Save final prediction analysis
    correct_data = test_results[1]
    incorrect_data = test_results[2]
    
    # Save correct predictions
    save_results_to_csv(
        correct_data[0], 
        correct_data[1], 
        correct_data[2],
        "correct_predictions_final.csv"
    )
    
    # Save incorrect predictions
    save_results_to_csv(
        incorrect_data[0], 
        incorrect_data[1], 
        incorrect_data[2],
        "incorrect_predictions_final.csv"
    )
    
    return model, test_accuracy