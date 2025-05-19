# Transliteration Model with Sequence-to-Sequence Architecture

## Assignment Outline

- In this assignment, I have trained a Sequence-to-Sequence (Seq2Seq) model with and without attention for transliteration tasks.
- The model is built to transliterate between Latin script and various Indic scripts.
- Attention mechanism is incorporated to improve transliteration quality.

## How to Load the Data?

**File:** `data_loader.py`

```python
loaders, src_vocab, tgt_vocab = get_dataloaders(
    language="te",
    batch_size=64,
    device="cuda"
)
```

- The `get_dataloaders` function loads data and creates dataloaders for training, validation, and testing
- The function returns dataloaders, source vocabulary, and target vocabulary objects
- Parameters include language code, batch size, and device type.

## How to Create a Model?

**File:** `models.py`

```python
# Create encoder
encoder = Encoder(
    src_vocab.size, args.emb_size, args.hidden_size,
    args.enc_layers, args.cell_type, args.dropout, 
    bidirectional=args.bidirectional
).to(device)

# Calculate encoder output dimension
enc_out_dim = args.hidden_size * 2 if args.bidirectional else args.hidden_size

# Create decoder
decoder = Decoder(
    tgt_vocab.size, args.emb_size, enc_out_dim, args.hidden_size,
    args.enc_layers, args.cell_type, args.dropout, 
    use_attn=use_attention
).to(device)

# Create sequence-to-sequence model
model = Seq2Seq(encoder, decoder, pad_idx=src_vocab.pad_idx, device=device).to(device)
```

- To create a model, you need to instantiate an Encoder, a Decoder, and a Seq2Seq model
- The Encoder accepts vocabulary size, embedding size, hidden size, number of layers, cell type, dropout rate, and bidirectional flag
- The Decoder additionally accepts an attention flag (`use_attn`)
- The Seq2Seq model connects the encoder and decoder, managing the information flow between them.

## How to Train the Model?

**File:** `model_trainer.py`

```python
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
```

- The `train_model` function takes the Seq2Seq model, data loaders, vocabularies, device, configuration, and logging options
- It handles the complete training process, including validation and saving the best model
- The function returns the trained model and the test accuracy
- It also logs metrics to Weights & Biases for experiment tracking

```text
train.py is the file you need to run to train the model. It supports various command-line arguments for flexible configuration.

Note: Download the data_loader.py,vocab.py, models.py, and model_trainer.py files before running train.py.
```

## Command-Line Arguments

Below are the command-line arguments supported by the script, specifying default values and the inputs it will take:

- `--wandb_entity`, `-we`  
  **Description:** WandB entity used to track experiments.  
  **Type:** `str`  
  **Default:** `cs24m042-iit-madras-foundation`

- `--wandb_project`, `-wp`  
  **Description:** Project name used in WandB for organizing experiment logs.  
  **Type:** `str`  
  **Default:** `DA6401-Assignment-3`

- `--language`, `-l`  
  **Description:** Language code for transliteration (e.g., 'te' for Telugu).  
  **Type:** `str`  
  **Default:** `te`


- `--output_dir`, `-od`  
  **Description:** Directory to save models and outputs.  
  **Type:** `str`  
  **Default:** `./output`

- `--emb_size`, `-es`  
  **Description:** Size of the embedding layer.  
  **Type:** `int`  
  **Default:** `256`

- `--hidden_size`, `-hs`  
  **Description:** Size of the hidden layers in encoder and decoder.  
  **Type:** `int`  
  **Default:** `512`

- `--enc_layers`, `-el`  
  **Description:** Number of layers in the encoder.  
  **Type:** `int`  
  **Default:** `2`

- `--cell_type`, `-ct`  
  **Description:** Type of RNN cell to use.  
  **Choices:** `RNN`, `GRU`, `LSTM`  
  **Default:** `LSTM`

- `--bidirectional`, `-bi`  
  **Description:** Use bidirectional encoder.  
  **Action:** `store_true`  
  **Default:** `False`

- `--use_attention`, `-att`  
  **Description:** Use attention mechanism in the decoder.  
  **Choices:** `true`, `false`  
  **Default:** `true`

- `--batch_size`, `-b`  
  **Description:** Batch size for training.  
  **Type:** `int`  
  **Default:** `64`

- `--epochs`, `-e`  
  **Description:** Number of epochs to train.  
  **Type:** `int`  
  **Default:** `15`

- `--lr`, `-lr`  
  **Description:** Learning rate for the optimizer.  
  **Type:** `float`  
  **Default:** `0.001`

- `--dropout`, `-dp`  
  **Description:** Dropout rate for regularization.  
  **Type:** `float`  
  **Default:** `0.3`

- `--teacher_forcing`, `-tf`  
  **Description:** Teacher forcing ratio for training.  
  **Type:** `float`  
  **Default:** `0.5`

- `--optimizer`, `-o`  
  **Description:** Optimizer for training.  
  **Choices:** `adam`, `nadam`  
  **Default:** `adam`

- `--weight_decay`, `-wd`  
  **Description:** Weight decay for optimizer.  
  **Type:** `float`  
  **Default:** `0.0001`

- `--seed`, `-s`  
  **Description:** Random seed for reproducibility.  
  **Type:** `int`  
  **Default:** `42`

- `--device`, `-d`  
  **Description:** Device to use for training.  
  **Choices:** `cuda`, `cpu`  
  **Default:** `cuda`

## How to do a sample run with default parameters?

```bash
python train.py --language te
```

For a more customized run:

```bash
python train.py --language te --cell_type GRU --enc_layers 3 --emb_size 256 --hidden_size 512 --batch_size 64 --lr 0.001 --teacher_forcing 0.5 --optimizer adam
```

Note: Better login first into wandb before running train.py.

```python
import wandb
wandb.login(key='Your key')
```

## Model Visualization

The training script automatically logs metrics and visualizations to Weights & Biases, including:

- Training and validation loss
- Word-level accuracy
- Correct and incorrect prediction tables
- Attention heatmaps (when using attention)

You can view these in your WandB dashboard to analyze model performance and attention patterns.
