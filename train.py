from models import SourceEncoder, TargetDecoder, Seq2SeqTransliterator, set_random_seeds
from data_loader import prepare_dataloaders
from model_training import train_transliteration_model, evaluate_model_accuracy, save_results_to_csv
import argparse
import wandb
import torch
from pathlib import Path
import types

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a seq2seq transliteration model")
    parser.add_argument("--wandb_entity", "-we", help="Wandb Entity", default="cs24m042-iit-madras-foundation")
    parser.add_argument("--wandb_project", "-wp", help="Project name", default="DA6401-Assignment-3")
    
    # Data parameters
    parser.add_argument("--language", "-l", type=str, default="te", help="Language code (e.g., 'te' for Telugu)")
    parser.add_argument("--base_dir", "-br", type=str, default="/kaggle/working/dakshina_dataset_v1.0/te/lexicons", help="Base directory containing the dataset")
    parser.add_argument("--output_dir", "-od", type=str, default="./output", help="Directory to save models and outputs")
    
    # Model architecture
    parser.add_argument("--emb_size", "-es", type=int, default=256, help="Embedding size")
    parser.add_argument("--hidden_size", "-hs", type=int, default=1024, help="Hidden state size")
    parser.add_argument("--enc_layers", "-el", type=int, default=1, help="Number of encoder layers")
    parser.add_argument("--cell_type", "-ct", type=str, default="GRU", choices=["RNN", "GRU", "LSTM"], help="RNN cell type")
    parser.add_argument("--bidirectional", "-bi", action="store_true", help="Use bidirectional encoder")
    parser.add_argument("--use_attention", "-att", type=str, default="True", choices=["true", "false"], help="Use attention mechanism")
    
    # Training parameters
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", "-lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--dropout", "-dp", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--teacher_forcing", "-tf", type=float, default=0.5, help="Teacher forcing ratio")
    parser.add_argument("--optimizer", "-o", type=str, default="nadam", choices=["adam", "nadam"], help="Optimizer")
    parser.add_argument("--weight_decay", "-wd", type=float, default=0.0001, help="Weight decay for optimizer")
    parser.add_argument("--seed", "-s", type=int, default=46, help="Random seed")
    parser.add_argument("--device", "-d", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert arguments to config namespace
    config = types.SimpleNamespace(
        lr=args.lr,
        epochs=args.epochs,
        teacher_forcing=args.teacher_forcing,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        dropout=args.dropout,
        seed=args.seed
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds for reproducibility
    set_random_seeds(args.seed)
    
    # Set device
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Parse boolean arguments
    use_attention = (args.use_attention.lower() == "true")
    
    # Create a descriptive run name
    run_name = (
        f"{args.language}_{args.cell_type}_{args.enc_layers}L_{args.emb_size}E_{args.hidden_size}H_"
        f"{'Bidirectional' if args.bidirectional else 'Unidirectional'}_"
        f"{'Attention' if use_attention else 'NoAttention'}_"
        f"{args.optimizer}_LR{args.lr}_TF{args.teacher_forcing}"
    )
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args)
    )
    
    # Load datasets
    print(f"Loading {args.language} data...")
    data_loaders, source_vocab, target_vocab = prepare_dataloaders(
        language_code=args.language,
        data_root=args.base_dir,
        batch_size=args.batch_size,
        device_type=device.type
    )
    
    # Print vocabulary statistics
    print(f"Source vocabulary size: {source_vocab.vocabulary_size}")
    print(f"Target vocabulary size: {target_vocab.vocabulary_size}")
    print(f"Training examples: {len(data_loaders['train'].dataset)}")
    print(f"Validation examples: {len(data_loaders['validation'].dataset)}")
    print(f"Test examples: {len(data_loaders['test'].dataset)}")
    
    # Create model components
    print("Building model architecture...")
    
    # Create encoder
    encoder = SourceEncoder(
        vocabulary_size=source_vocab.vocabulary_size,
        embedding_dim=args.emb_size,
        hidden_dim=args.hidden_size,
        num_layers=args.enc_layers,
        rnn_type=args.cell_type,
        dropout_rate=args.dropout,
        use_bidirectional=args.bidirectional
    ).to(device)
    
    # Calculate encoder output dimension
    encoder_output_dim = args.hidden_size * 2 if args.bidirectional else args.hidden_size
    
    # Create decoder
    decoder = TargetDecoder(
        vocabulary_size=target_vocab.vocabulary_size,
        embedding_dim=args.emb_size,
        encoder_dim=encoder_output_dim,
        decoder_dim=args.hidden_size,
        num_layers=args.enc_layers,
        rnn_type=args.cell_type,
        dropout_rate=args.dropout,
        enable_attention=use_attention
    ).to(device)
    
    # Create sequence-to-sequence model
    model = Seq2SeqTransliterator(
        encoder=encoder,
        decoder=decoder,
        pad_token_idx=source_vocab.pad_index,
        device_name=device
    ).to(device)
    
    # Print model structure summary
    print("Model architecture:")
    print(f"  Encoder: {args.cell_type}, {'Bidirectional' if args.bidirectional else 'Unidirectional'}, "
          f"{args.enc_layers} layers, {args.hidden_size} hidden units")
    print(f"  Decoder: {args.cell_type}, {'With' if use_attention else 'Without'} attention, "
          f"{args.enc_layers} layers, {args.hidden_size} hidden units")
    
    # Define model save path
    best_model_path = output_dir / f"{run_name}_best.pt"
    
    # Train the model
    print("Starting model training...")
    model, test_accuracy = train_transliteration_model(
        model=model,
        data_loaders=data_loaders,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        device=device,
        config=config,
        model_save_path=best_model_path,
        enable_wandb_logging=True
    )
    
    # Save final model
    final_model_path = output_dir / f"{run_name}_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Generate final predictions
    print("Generating final predictions on test set...")
    test_results = evaluate_model_accuracy(
        model, 
        data_loaders['test'], 
        target_vocab, 
        source_vocab, 
        device
    )
    test_accuracy = test_results[0]
    
    correct_data = test_results[1]
    incorrect_data = test_results[2]
    
    # Save predictions to CSV files
    correct_csv = output_dir / f"{run_name}_correct.csv"
    incorrect_csv = output_dir / f"{run_name}_incorrect.csv"
    
    save_results_to_csv(
        correct_data[0], 
        correct_data[1], 
        correct_data[2],
        correct_csv
    )
    
    save_results_to_csv(
        incorrect_data[0], 
        incorrect_data[1], 
        incorrect_data[2],
        incorrect_csv
    )
    
    # Log final results
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Saved correct predictions to {correct_csv}")
    print(f"Saved incorrect predictions to {incorrect_csv}")
    
    # Create and log WandB tables for predictions
    try:
        # Log correct predictions
        correct_table = wandb.Table(
            columns=["Source", "Target", "Prediction"],
            data=list(zip(correct_data[0], correct_data[1], correct_data[2]))
        )
        wandb.log({"Correct_Predictions": correct_table})
        
        # Log incorrect predictions
        incorrect_table = wandb.Table(
            columns=["Source", "Target", "Prediction"],
            data=list(zip(incorrect_data[0], incorrect_data[1], incorrect_data[2]))
        )
        wandb.log({"Incorrect_Predictions": incorrect_table})
    except Exception as e:
        print(f"Warning: Could not create WandB tables: {e}")
    
    # Finish WandB logging
    wandb.finish()