import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from typing import Dict, Tuple, List, Optional, Callable, Any, Iterator


def parse_transliteration_file(filepath: str) -> Iterator[Tuple[str, str]]:
    with open(filepath, encoding='utf-8') as file:
        for line in file:
            items = line.strip().split('\t')
            if len(items) >= 2:
                # Return native and latin text in expected order
                yield items[1], items[0]

def parse_csv_file(filepath: str, source_col: str = 'src', target_col: str = 'trg') -> Iterator[Tuple[str, str]]:
    dataframe = pd.read_csv(filepath)
    for _, row in dataframe.iterrows():
        yield row[source_col], row[target_col]

class TransliterationCorpus(Dataset):
    
    def __init__(self, filepath: str, source_vocab, target_vocab, filetype: str = 'tsv'):
        
        self.samples = []
        self.filetype = filetype
        
        # Parse the data file based on format
        if filetype == 'tsv':
            parser = parse_transliteration_file
        elif filetype == 'csv':
            parser = parse_csv_file
        else:
            raise ValueError(f"Unsupported file type: {filetype}. Use 'tsv' or 'csv'")
        
        # Process each source-target pair
        for source_text, target_text in parser(filepath):
            # Tokenize both source and target text with boundary tokens
            source_indices = source_vocab.tokenize(source_text, add_start=True, add_end=True)
            target_indices = target_vocab.tokenize(target_text, add_start=True, add_end=True)
            
            # Convert to tensors and store
            self.samples.append((
                torch.tensor(source_indices, dtype=torch.long),
                torch.tensor(target_indices, dtype=torch.long)
            ))

    def __len__(self) -> int:
        """Get the number of samples in the dataset"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample by index"""
        return self.samples[idx]


def create_batch_tensors(batch_data, source_vocab, target_vocab) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # Separate sources and targets
    sources, targets = zip(*batch_data)
    
    # Pad sequences to the same length
    padded_sources = pad_sequence(sources, batch_first=True, padding_value=source_vocab.pad_index)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=target_vocab.pad_index)
    
    # Calculate original lengths for packing
    source_lengths = torch.tensor([len(s) for s in sources], dtype=torch.long)
    
    return padded_sources, source_lengths, padded_targets


def prepare_dataloaders(
        language_code: str = 'te',
        format_type: str = 'tsv',
        data_root: Optional[str] = None,
        batch_size: int = 64,
        device_type: str = 'cpu',
        worker_count: int = 2,
        prefetch_multiplier: int = 4,
        keep_workers_alive: bool = True,
        vocabulary_cache_dir: str = './vocab_cache',
        use_cached_vocabulary: bool = True
    ) -> Tuple[Dict[str, DataLoader], Any, Any]:
   
    # Set up the data directory
    if data_root is None:
        data_root = os.path.join(
            '/kaggle/working/dakshina_dataset_v1.0',
            language_code, 'lexicons'
        )
    
    # Prepare vocabulary cache
    if use_cached_vocabulary:
        os.makedirs(vocabulary_cache_dir, exist_ok=True)
        vocab_cache_path = os.path.join(
            vocabulary_cache_dir, 
            f"{language_code}_{format_type}_vocabulary.pkl"
        )
    
    # Load or build vocabularies
    if use_cached_vocabulary and os.path.exists(vocab_cache_path):
        print(f"Loading vocabularies from cache: {vocab_cache_path}")
        with open(vocab_cache_path, 'rb') as f:
            source_vocab, target_vocab = pickle.load(f)
    else:
        # Build vocabularies from training and validation data
        all_source_texts, all_target_texts = [], []
        
        for split in ['train', 'dev']:
            filepath = os.path.join(data_root, f"{language_code}.translit.sampled.{split}.tsv")
            for src_text, tgt_text in parse_transliteration_file(filepath):
                all_source_texts.append(src_text)
                all_target_texts.append(tgt_text)
        
        # Create vocabularies from the collected texts
        from vocab import SequenceVocabulary
        source_vocab = SequenceVocabulary.create_from_corpus(all_source_texts)
        target_vocab = SequenceVocabulary.create_from_corpus(all_target_texts)
        
        # Cache vocabularies if requested
        if use_cached_vocabulary:
            with open(vocab_cache_path, 'wb') as f:
                pickle.dump((source_vocab, target_vocab), f)
    
    # Common DataLoader settings
    loader_settings = {
        'batch_size': batch_size,
        'num_workers': worker_count,
        'prefetch_factor': prefetch_multiplier,
        'persistent_workers': keep_workers_alive and worker_count > 0,
        'pin_memory': device_type == 'cuda'
    }
    
    # Create data loaders for each split
    data_loaders = {}
    
    # Define the splits and their corresponding file names
    split_mapping = {
        'train': 'train', 
        'validation': 'dev', 
        'test': 'test'
    }
    
    # Create a data loader for each split
    for loader_name, file_suffix in split_mapping.items():
        filepath = os.path.join(data_root, f"{language_code}.translit.sampled.{file_suffix}.tsv")
        
        # Create the dataset
        dataset = TransliterationCorpus(
            filepath, 
            source_vocab, 
            target_vocab, 
            filetype=format_type
        )
        
        # Create the data loader with appropriate settings
        data_loaders[loader_name] = DataLoader(
            dataset,
            shuffle=(loader_name == 'train'),  # Only shuffle training data
            collate_fn=lambda b: create_batch_tensors(b, source_vocab, target_vocab),
            **loader_settings
        )
    
    return data_loaders, source_vocab, target_vocab