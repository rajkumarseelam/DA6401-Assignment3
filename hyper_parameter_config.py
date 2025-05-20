import wandb
import torch


from models import SourceEncoder, TargetDecoder, Seq2SeqTransliterator, set_random_seeds
from model_training import train_transliteration_model
from data_loader import prepare_dataloaders

def hyperparameter_search():
    """Run hyperparameter search using Weights & Biases Sweep"""
    # Initialize a new WandB run
    run = wandb.init()
    cfg = run.config
    
    # Set seeds for reproducibility
    set_random_seeds(cfg.seed if hasattr(cfg, 'seed') else 42)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create a descriptive run name based on hyperparameters
    run_name = (
        f"{cfg.rnn_type}_{cfg.num_layers}L_{cfg.embedding_size}E_{cfg.hidden_size}H_"
        f"{'Bi' if cfg.bidirectional else 'Uni'}_"
        f"D{cfg.dropout}_TF{cfg.teacher_forcing}_{cfg.optimizer}"
    )
    wandb.run.name = run_name
    
    # Load data with appropriate dataloaders
    data_loaders, source_vocab, target_vocab = prepare_dataloaders(
        language_code='te',  # Telugu language
        batch_size=cfg.batch_size,
        device_type=device
    )
    
    # Create encoder
    encoder = SourceEncoder(
        vocabulary_size=source_vocab.vocabulary_size,
        embedding_dim=cfg.embedding_size,
        hidden_dim=cfg.hidden_size,
        num_layers=cfg.num_layers,
        rnn_type=cfg.rnn_type,
        dropout_rate=cfg.dropout,
        use_bidirectional=cfg.bidirectional
    ).to(device)
    
    # Calculate encoder output dimension (doubled if bidirectional)
    encoder_output_dim = cfg.hidden_size * 2 if cfg.bidirectional else cfg.hidden_size
    
    # Create decoder (without attention for this search)
    decoder = TargetDecoder(
        vocabulary_size=target_vocab.vocabulary_size,
        embedding_dim=cfg.embedding_size,
        encoder_dim=encoder_output_dim,
        decoder_dim=cfg.hidden_size,
        num_layers=cfg.num_layers,
        rnn_type=cfg.rnn_type,
        dropout_rate=cfg.dropout,
        enable_attention=False  # No attention for this search you can modify it to true..
    ).to(device)
    
    # Create full seq2seq model
    model = Seq2SeqTransliterator(
        encoder=encoder,
        decoder=decoder,
        pad_token_idx=source_vocab.pad_index,
        device_name=device
    ).to(device)
    
    # Train the model
    best_model_path = f"best_model_{run_name}.pt"
    _, test_accuracy = train_transliteration_model(
        model=model,
        data_loaders=data_loaders,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        device=device,
        config=cfg,
        model_save_path=best_model_path,
        enable_wandb_logging=True
    )
    
    # Log final test accuracy as summary metric
    wandb.run.summary['test_accuracy'] = test_accuracy * 100  # as percentage
    
    # Finish the run
    wandb.finish()

if __name__ == "__main__":
    # Define the hyperparameter search space
    sweep_configuration = {
        'method': 'bayes',  # Bayesian optimization for better search efficiency
        'name': 'Transliteration_NoAttention_Sweep',
        'metric': {'name': 'validation_accuracy', 'goal': 'maximize'},
        'parameters': {
            # Model architecture parameters
            'embedding_size': {'values': [128, 256, 512]},
            'hidden_size': {'values': [128, 256, 512, 1024]},
            'num_layers': {'values': [1, 2, 3, 4]},
            'rnn_type': {'values': ['RNN', 'GRU', 'LSTM']},
            'bidirectional': {'values': [True, False]},
            
            # Training parameters
            'dropout': {'values': [0.0, 0.1, 0.2, 0.3, 0.5]},
            'learning_rate': {'values': [1e-4, 2e-4, 5e-4, 8e-4, 1e-3]},
            'batch_size': {'values': [32, 64, 128]},
            'epochs': {'values': [10, 15, 20]},
            'teacher_forcing': {'values': [0.3, 0.5, 0.7, 1.0]},
            'optimizer': {'values': ['Adam', 'NAdam']},
            
            # Reproducibility parameter
            'seed': {'values': [42, 43, 44, 45, 46]},
        }
    }

    # Initialize the hyperparameter sweep
    sweep_id = wandb.sweep(
        sweep_configuration,
        entity='cs24m042-iit-madras-foundation', 
        project='DA6401-Assignment-3'
    )
    
    # Run the sweep agent
    wandb.agent(sweep_id, function=hyperparameter_search, count=1)