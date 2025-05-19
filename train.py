from models import Encoder, Decoder, Seq2Seq, seed_everything
from data_loader import get_dataloaders
from model_training import train_model, compute_detailed_accuracy, save_predictions_to_csv
import argparse
import wandb
import torch
from pathlib import Path
import types

if __name__ == "__main__":
    # Added the default parameters of my best config
    parser = argparse.ArgumentParser(description="Train a seq2seq transliteration model")
    parser.add_argument("--wandb_entity", "-we", help="Wandb Entity", default="cs24m042-iit-madras-foundation")
    parser.add_argument("--wandb_project", "-wp", help="Project name", default="DA6401-Assignment-3")
    
    # Data parameters
    parser.add_argument("--language", "-l", type=str, default="te", help="Language code (e.g., 'te' for Telugu)")
    parser.add_argument("--output_dir", "-od", type=str, default="./output", help="Directory to save models and outputs")
    
    # Model architecture
    parser.add_argument("--emb_size", "-es", type=int, default=256, help="Embedding size")
    parser.add_argument("--hidden_size", "-hs", type=int, default=512, help="Hidden state size")
    parser.add_argument("--enc_layers", "-el", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--cell_type", "-ct", type=str, default="LSTM", choices=["RNN", "GRU", "LSTM"], help="RNN cell type")
    parser.add_argument("--bidirectional", "-bi", action="store_true", help="Use bidirectional encoder")
    parser.add_argument("--use_attention", "-att", type=str, default="true", choices=["true", "false"], help="Use attention mechanism")
    
    # Training parameters
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=15, help="Number of epochs")
    parser.add_argument("--lr", "-lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout", "-dp", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--teacher_forcing", "-tf", type=float, default=0.5, help="Teacher forcing ratio")
    parser.add_argument("--optimizer", "-o", type=str, default="adam", choices=["adam", "nadam"], help="Optimizer")
    parser.add_argument("--weight_decay", "-wd", type=float, default=0.0001, help="Weight decay for optimizer")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--device", "-d", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    # Convert args to config object expected by train_model function
    config = types.SimpleNamespace(**vars(args))
    
    # Set up paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set seeds for reproducibility
    seed_everything(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Convert string arguments to boolean
    use_attention = (args.use_attention == "true")
    
    # Create a descriptive run name
    run_name = f"{args.language}_{args.cell_type}_{args.enc_layers}l_{args.emb_size}emb_{args.hidden_size}hid_" \
               f"{'bid' if args.bidirectional else 'uni'}_{'attn' if use_attention else 'plain'}_" \
               f"{args.optimizer}_lr{args.lr}_tf{args.teacher_forcing}"
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args)
    )
    
    # Load datasets
    print(f"Loading {args.language} data...")
    loaders, src_vocab, tgt_vocab = get_dataloaders(
        language=args.language,
        batch_size=args.batch_size,
        device=device.type
    )
    
    # Print vocabulary stats
    print(f"Source vocabulary size: {src_vocab.size}")
    print(f"Target vocabulary size: {tgt_vocab.size}")
    print(f"Training examples: {len(loaders['train'].dataset)}")
    print(f"Validation examples: {len(loaders['dev'].dataset)}")
    print(f"Test examples: {len(loaders['test'].dataset)}")
    
    # Create model
    print("Building model...")
    
    # Create encoder
    encoder = Encoder(
        src_vocab.size, args.emb_size, args.hidden_size,
        args.enc_layers, args.cell_type, args.dropout, 
        bidirectional=args.bidirectional
    ).to(device)
    
    # Calculate encoder output dimension (doubled if bidirectional)
    enc_out_dim = args.hidden_size * 2 if args.bidirectional else args.hidden_size
    
    # Create decoder
    decoder = Decoder(
        tgt_vocab.size, args.emb_size, enc_out_dim, args.hidden_size,
        args.enc_layers, args.cell_type, args.dropout, 
        use_attn=use_attention
    ).to(device)
    
    # Create sequence-to-sequence model
    model = Seq2Seq(encoder, decoder, pad_idx=src_vocab.pad_idx, device=device).to(device)
    
    # Define model save path
    best_model_path = output_dir / f"{run_name}_best.pt"
    
    # Train the model using the existing train_model function
    print("Training model...")
    model, test_acc = train_model(
        model=model,
        loaders=loaders,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        config=config,
        save_path=best_model_path,
        log_to_wandb=True
    )
    
    # Save final model
    final_model_path = output_dir / f"{run_name}_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Generate final predictions
    print("Generating final predictions on test set...")
    test_results = compute_detailed_accuracy(model, loaders['test'], tgt_vocab, src_vocab, device)
    test_acc = test_results[0]
    
    correct_data = test_results[1]
    incorrect_data = test_results[2]
    
    # Save predictions to CSV files
    correct_csv = output_dir / f"{run_name}_correct.csv"
    incorrect_csv = output_dir / f"{run_name}_incorrect.csv"
    
    save_predictions_to_csv(
        correct_data[0], correct_data[1], correct_data[2],
        correct_csv
    )
    
    save_predictions_to_csv(
        incorrect_data[0], incorrect_data[1], incorrect_data[2],
        incorrect_csv
    )
    
    # Log final results
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Saved correct predictions to {correct_csv}")
    print(f"Saved incorrect predictions to {incorrect_csv}")
    
    # Create wandb tables for predictions if desired
    try:
        # Log correct predictions
        correct_table = wandb.Table(
            columns=["Source", "Target", "Predicted"],
            data=list(zip(correct_data[0], correct_data[1], correct_data[2]))
        )
        wandb.log({"Attention_Correct": correct_table})
        
        # Log incorrect predictions
        incorrect_table = wandb.Table(
            columns=["Source", "Target", "Predicted"],
            data=list(zip(incorrect_data[0], incorrect_data[1], incorrect_data[2]))
        )
        wandb.log({"Attention_Incorrect": incorrect_table})
    except:
        print("Warning: Could not create wandb tables")
    
    wandb.finish()
