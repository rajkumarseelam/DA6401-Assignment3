# Transliteration Model with Sequence-to-Sequence Architecture

## Assignment Outline

- In this assignment, I've implemented a Sequence-to-Sequence (Seq2Seq) model with and without attention for transliteration tasks.
- The model transliterates between Latin script and various Indic scripts (such as Telugu).
- The implementation features customizable encoder-decoder architecture with optional attention mechanism.


## Wandb Report Link
You can find the Report on [WandB Report](https://wandb.ai/cs24m042-iit-madras-foundation/DA6401-Assignment-3/reports/Assignment-3--VmlldzoxMjgyMDUwMw).

## Data Loading

**File:** `data_loader.py`

```python
data_loaders, source_vocab, target_vocab = prepare_dataloaders(
    language_code="te",
    format_type="tsv",
    batch_size=64,
    device_type="cuda"
)
```

- The `prepare_dataloaders` function loads transliteration data and creates DataLoader objects
- Returns data loaders for train/validation/test splits, along with source and target vocabularies
- Supports various parameters including language code, batch size, and device type
- The function can cache vocabularies for faster loading in subsequent runs

## Model Architecture

**File:** `models.py`

```python
# Create encoder
encoder = SourceEncoder(
    vocabulary_size=source_vocab.vocabulary_size,
    embedding_dim=args.emb_size,
    hidden_dim=args.hidden_size,
    num_layers=args.enc_layers,
    rnn_type=args.cell_type,
    dropout_rate=args.dropout,
    use_bidirectional=args.bidirectional
)

# Create decoder with optional attention
decoder = TargetDecoder(
    vocabulary_size=target_vocab.vocabulary_size,
    embedding_dim=args.emb_size,
    encoder_dim=encoder_output_dim,
    decoder_dim=args.hidden_size,
    num_layers=args.enc_layers,
    rnn_type=args.cell_type,
    dropout_rate=args.dropout,
    enable_attention=use_attention
)

# Create sequence-to-sequence model
model = Seq2SeqTransliterator(
    encoder=encoder,
    decoder=decoder,
    pad_token_idx=source_vocab.pad_index,
    device_name=device
)
```

- The model consists of three main components: `SourceEncoder`, `TargetDecoder`, and `Seq2SeqTransliterator`
- The encoder supports different RNN types (LSTM, GRU, RNN) and can be bidirectional
- The decoder includes an optional attention mechanism
- The Seq2SeqTransliterator connects the encoder and decoder components

## Training the Model

**File:** `model_training.py`

```python
model, test_accuracy = train_transliteration_model(
    model=model,
    data_loaders=data_loaders,
    source_vocab=source_vocab,
    target_vocab=target_vocab,
    device=device,
    config=config,
    model_save_path="best_model.pt",
    enable_wandb_logging=True
)
```

- The `train_transliteration_model` function handles the entire training process
- It performs training with teacher forcing, validation, and saves the best model
- Provides detailed metrics including accuracy and loss tracking
- Supports logging to Weights & Biases for experiment tracking
- Includes evaluation with detailed error analysis in CSV reports

## Running the Training Script

**File:** `train.py`

To train a model with default parameters:

```bash
python train.py --language te
```

For a customized training run:

```bash
python train.py --language te --cell_type GRU --enc_layers 2 --emb_size 256 --hidden_size 1024 --batch_size 32 --lr 0.0002 --teacher_forcing 0.5 --optimizer nadam --use_attention true
```

## Command-Line Arguments

The training script supports the following arguments:

- **Data parameters:**
  - `--language`, `-l`: Language code (default: `te` for Telugu)
  - `--base_dir`, `-br`: Base directory for dataset files
  - `--output_dir`, `-od`: Directory to save model outputs (default: `./output`)

- **Model architecture:**
  - `--emb_size`, `-es`: Embedding size (default: `256`)
  - `--hidden_size`, `-hs`: Hidden layer size (default: `1024`)
  - `--enc_layers`, `-el`: Number of encoder layers (default: `1`)
  - `--cell_type`, `-ct`: RNN cell type (`RNN`, `GRU`, `LSTM`) (default: `GRU`)
  - `--bidirectional`, `-bi`: Use bidirectional encoder (flag)
  - `--use_attention`, `-att`: Use attention mechanism (`true`, `false`) (default: `True`)

- **Training parameters:**
  - `--batch_size`, `-b`: Batch size (default: `32`)
  - `--epochs`, `-e`: Number of training epochs (default: `20`)
  - `--lr`, `-lr`: Learning rate (default: `0.0002`)
  - `--dropout`, `-dp`: Dropout rate (default: `0.2`)
  - `--teacher_forcing`, `-tf`: Teacher forcing ratio (default: `0.5`)
  - `--optimizer`, `-o`: Optimizer type (`adam`, `nadam`) (default: `nadam`)
  - `--seed`, `-s`: Random seed for reproducibility (default: `46`)
  - `--device`, `-d`: Device to use (`cuda`, `cpu`) (default: `cuda`)

- **Logging parameters:**
  - `--wandb_entity`, `-we`: WandB entity name
  - `--wandb_project`, `-wp`: WandB project name (default: `DA6401-Assignment-3`)

## Vocabulary Management

**File:** `vocab.py`

The project uses a specialized `SequenceVocabulary` class to handle character tokenization:

```python
# Create vocabularies from text data
source_vocab = SequenceVocabulary.create_from_corpus(source_texts)
target_vocab = SequenceVocabulary.create_from_corpus(target_texts)

# Convert text to token indices
token_indices = source_vocab.tokenize(text, add_start=True, add_end=True)

# Convert token indices back to text
text = target_vocab.detokenize(token_indices)
```

- Handles tokenization/detokenization for both source and target languages
- Manages special tokens (`<pad>`, `<bos>`, `<eos>`, `<unk>`)
- Supports serialization and caching for faster loading

## Evaluation and Metrics

The model evaluation provides:

- Word-level accuracy (correct transliterations / total examples)
- Detailed CSV reports of correct and incorrect predictions
- Support for greedy decoding during inference

## WandB Integration

The project integrates with Weights & Biases for experiment tracking:

```python
# Log into WandB before training
import wandb
wandb.login(key='Your key')
```

Metrics tracked include:
- Training and validation loss
- Training and validation accuracy
- Test accuracy
- Prediction samples (correct and incorrect)