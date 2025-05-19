import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm.auto import tqdm
import csv

def compute_detailed_accuracy(model, loader, tgt_vocab, src_vocab, device):
    """
    Enhanced accuracy function that returns:
    - Overall accuracy
    - Lists of correct and incorrect predictions for analysis
    """
    model.eval()
    correct = total = 0
    
    # Lists to store detailed results
    correct_srcs = []
    correct_tgts = []
    correct_preds = []
    
    incorrect_srcs = []
    incorrect_tgts = []
    incorrect_preds = []
    
    with torch.no_grad():
        for src, src_lens, tgt in loader:
            src, src_lens, tgt = (x.to(device) for x in (src, src_lens, tgt))
            pred = model.infer_greedy(src, src_lens, tgt_vocab, max_len=tgt.size(1))

            # iterate over the batch
            for b in range(src.size(0)):
                # Convert indices to strings
                pred_str = tgt_vocab.decode(pred[b].cpu().tolist())
                gold_str = tgt_vocab.decode(tgt[b, 1:].cpu().tolist())  # skip <sos>
                src_str = src_vocab.decode(src[b].cpu().tolist())
                
                # Check if prediction is correct
                is_correct = (pred_str == gold_str)
                correct += is_correct
                
                # Store detailed results
                if is_correct:
                    correct_srcs.append(src_str)
                    correct_tgts.append(gold_str)
                    correct_preds.append(pred_str)
                else:
                    incorrect_srcs.append(src_str)
                    incorrect_tgts.append(gold_str)
                    incorrect_preds.append(pred_str)
                    
            total += src.size(0)

    accuracy = correct / total if total else 0.0
    return (
        accuracy, 
        (correct_srcs, correct_tgts, correct_preds),
        (incorrect_srcs, incorrect_tgts, incorrect_preds)
    )

def save_predictions_to_csv(src_list, tgt_list, pred_list, file_name):
    """Save prediction details to CSV file for further analysis"""
    rows = zip(src_list, tgt_list, pred_list)
    
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Source', 'Target', 'Predicted'])
        writer.writerows(rows)
    
    return file_name

def train_model(
    model, 
    loaders, 
    src_vocab, 
    tgt_vocab, 
    device,
    config,
    save_path=None,
    log_to_wandb=True
):
    """
    Enhanced training function with:
    - Teacher forcing control
    - Detailed accuracy tracking
    - Progress bars
    - Optional WandB logging
    """
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
    
    # Select optimizer based on config
    if config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer.lower() == 'nadam':
        optimizer = optim.NAdam(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # Track best validation accuracy
    best_val_acc = 0.0
    
    # Main training loop
    for epoch in tqdm(range(1, config.epochs + 1), desc="Epochs", position=0):
        model.train()
        total_loss = 0.0

        # Training batches with progress bar
        train_loader = tqdm(loaders['train'], desc=f"Train {epoch}", leave=False, position=1)
        for src, src_lens, tgt in train_loader:
            src, src_lens, tgt = src.to(device), src_lens.to(device), tgt.to(device)

            optimizer.zero_grad()
            # Use teacher forcing ratio from config
            output = model(src, src_lens, tgt, teacher_forcing_ratio=config.teacher_forcing)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:,1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        train_loader.close()
        train_loss = total_loss / len(loaders['train'])

        # Validation loss
        val_loss = 0.0
        val_loader = tqdm(loaders['dev'], desc=f"Val {epoch}", leave=False, position=1)
        model.eval()
        with torch.no_grad():
            for src, src_lens, tgt in val_loader:
                src, src_lens, tgt = src.to(device), src_lens.to(device), tgt.to(device)
                output = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)  # No teacher forcing during validation
                val_loss += criterion(output.reshape(-1, output.size(-1)),
                                    tgt[:,1:].reshape(-1)).item()
        val_loader.close()
        val_loss /= len(loaders['dev'])

        # Compute detailed accuracy metrics
        train_results = compute_detailed_accuracy(model, loaders['train'], tgt_vocab, src_vocab, device)
        train_acc = train_results[0]
        
        val_results = compute_detailed_accuracy(model, loaders['dev'], tgt_vocab, src_vocab, device)
        val_acc = val_results[0]
        
        # Save model if it's the best so far
        if val_acc > best_val_acc and save_path:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model with validation accuracy: {val_acc:.4f}")
            
            # Save prediction analysis CSVs for best model
            if epoch == config.epochs or epoch % 5 == 0:  # Save at last epoch or every 5 epochs
                correct_data = val_results[1]
                incorrect_data = val_results[2]
                
                save_predictions_to_csv(
                    correct_data[0], correct_data[1], correct_data[2],
                    f"correct_predictions_epoch_{epoch}.csv"
                )
                
                save_predictions_to_csv(
                    incorrect_data[0], incorrect_data[1], incorrect_data[2],
                    f"incorrect_predictions_epoch_{epoch}.csv"
                )

        # Log metrics
        print(f"Epoch {epoch}/{config.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if log_to_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc*100,
                'val_acc': val_acc*100
            })
    
    # Final evaluation on test set
    test_results = compute_detailed_accuracy(model, loaders['test'], tgt_vocab, src_vocab, device)
    test_acc = test_results[0]
    print(f"Final test accuracy: {test_acc:.4f}")
    
    if log_to_wandb:
        wandb.log({'test_acc': test_acc})
    
    # Save final prediction analysis
    correct_data = test_results[1]
    incorrect_data = test_results[2]
    
    save_predictions_to_csv(
        correct_data[0], correct_data[1], correct_data[2],
        "correct_predictions_final.csv"
    )
    
    save_predictions_to_csv(
        incorrect_data[0], incorrect_data[1], incorrect_data[2],
        "incorrect_predictions_final.csv"
    )
 
    
    return model, test_acc