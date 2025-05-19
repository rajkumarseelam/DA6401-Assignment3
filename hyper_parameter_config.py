
import wandb
import torch

# Import enhanced modules
from models import Encoder, Decoder, Seq2Seq, seed_everything
from model_training import train_model
from data_loader import get_dataloaders

def objective():
    # Initialize WandB run
    run = wandb.init()
    cfg = run.config
    
    # Set seeds for reproducibility
    seed_everything(cfg.seed if hasattr(cfg, 'seed') else 42)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create a unique run name based on config
    run_name = f"{cfg.cell}_{cfg.enc_layers}l_{cfg.emb_size}e_{cfg.hidden_size}h_" \
               f"{'bid' if cfg.bidirectional else 'uni'}_{cfg.dropout}d_" \
               f"{cfg.teacher_forcing}tf_{cfg.optimizer}"
    wandb.run.name = run_name
    
    # Load data
    # I have choosen telugu language
    loaders, src_vocab, tgt_vocab = get_dataloaders(
        'te',
        batch_size=cfg.batch_size,
        device=device
    )
    
    # Create model components
    enc = Encoder(
        src_vocab.size, cfg.emb_size, cfg.hidden_size,
        cfg.enc_layers, cfg.cell, cfg.dropout, 
        bidirectional=cfg.bidirectional
    ).to(device)
    
    # Calculate encoder output dimension (doubled if bidirectional)
    enc_out_dim = cfg.hidden_size * 2 if cfg.bidirectional else cfg.hidden_size
    
    #Currently attention is set to false , To enable just pass use_attn=True
    dec = Decoder(
        tgt_vocab.size, cfg.emb_size, enc_out_dim, cfg.hidden_size,
        cfg.enc_layers, cfg.cell, cfg.dropout
    ).to(device) #use_attn=True as last parameter.
    
    model = Seq2Seq(enc, dec, pad_idx=src_vocab.pad_idx, device=device).to(device)
    
    # Train the model
    best_model_path = f"model_{run_name}.pt"
    _, test_acc = train_model(
        model=model,
        loaders=loaders,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        config=cfg,
        save_path=best_model_path,
        log_to_wandb=True
    )
    
    # Log final test accuracy as summary metric
    wandb.run.summary['test_accuracy'] = test_acc
    
    # Finish the run
    wandb.finish()

if __name__ == "__main__":
    sweep_cfg = {
        
        'method': 'bayes',  # Use Bayesian optimization
        'name':'Transliteration_without_Attention',
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': {
            
            # Model architecture
            'emb_size': {'values': [128, 256, 512]},
            'hidden_size': {'values': [128, 256, 512, 1024]},
            'enc_layers': {'values': [1, 2, 3, 4]},
            'cell': {'values': ['RNN', 'GRU', 'LSTM']},  
            'bidirectional': {'values': [True, False]},  # Bidirectional encode
            
            # Training parameters
            'dropout': {'values': [0.0, 0.1, 0.2, 0.3, 0.5]},
            'lr': {'values': [1e-4, 2e-4, 5e-4, 8e-4, 1e-3]},
            'batch_size': {'values': [32, 64, 128]},
            'epochs': {'values': [10, 15, 20]},
            'teacher_forcing': {'values': [0.3, 0.5, 0.7, 1.0]},  # Explicit teacher forcing
            'optimizer': {'values': ['Adam', 'NAdam']},  # Added optimizer options
            # Reproducibility
            'seed': {'values': [42, 43, 44, 45, 46]},  # Different seeds for robustness
        }
    }

    # Start the sweep
    sweep_id = wandb.sweep(
        sweep_cfg,
        entity='cs24m042-iit-madras-foundation', 
        project='DA6401-Assignment-3'
    )
    
    # Run the sweep agent
    wandb.agent(sweep_id, function=objective, count=1)