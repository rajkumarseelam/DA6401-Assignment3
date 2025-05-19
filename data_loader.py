
import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from vocab import CharVocab

class TransliterationDataset(Dataset):
    """A flexible dataset class that can handle  Dakshina  dataset"""
    
    def __init__(self, path, src_vocab, tgt_vocab, format='dakshina'):
       
        self.examples = []
        self.format = format
        
        if format == 'dakshina':
            # Dakshina format: tab-separated without header
            for src, tgt in read_tsv(path):
                src_ids = src_vocab.encode(src, add_sos=True, add_eos=True)
                tgt_ids = tgt_vocab.encode(tgt, add_sos=True, add_eos=True)
                self.examples.append((
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long)
                ))
        
        else:
            raise ValueError(f"Unknown format: {format}. Use 'dakshina'")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def read_tsv(path):
    """Read a tab-separated file with source and target text"""
    with open(path, encoding='utf-8') as f:
        for ln in f:
            parts = ln.strip().split('\t')
            if len(parts) >= 2:
                yield parts[1], parts[0]  # Dakshina format has target, source


def read_csv(path, src_col='src', tgt_col='trg'):
    """Read a CSV file with source and target columns"""
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        yield row[src_col], row[tgt_col]


def collate_fn(batch, src_vocab, tgt_vocab):
    """Collate function to handle variable-length sequences"""
    srcs, tgts = zip(*batch)
    srcs_p = pad_sequence(srcs, batch_first=True, padding_value=src_vocab.pad_idx)
    tgts_p = pad_sequence(tgts, batch_first=True, padding_value=tgt_vocab.pad_idx)
    src_lens = torch.tensor([len(s) for s in srcs], dtype=torch.long)
    return srcs_p, src_lens, tgts_p


def get_dataloaders(
        language='te', 
        dataset_format='dakshina',
        base_path=None,
        batch_size=64,
        device='cpu',
        num_workers=2,
        prefetch_factor=4,
        persistent_workers=True,
        cache_dir='./cache',
        use_cached_vocab=True
    ):
    
    # Set up paths based on dataset format
    if base_path is None:
        base_path = os.path.join(
            '/kaggle/working/dakshina_dataset_v1.0',
            language, 'lexicons'
        )

    # Create cache directory if it doesn't exist
    if use_cached_vocab:
        os.makedirs(cache_dir, exist_ok=True)
        vocab_cache_path = os.path.join(cache_dir, f"{language}_{dataset_format}_vocab.pkl")
    
    # Try to load cached vocabularies
    if use_cached_vocab and os.path.exists(vocab_cache_path):
        print(f"Loading cached vocabularies from {vocab_cache_path}")
        with open(vocab_cache_path, 'rb') as f:
            src_vocab, tgt_vocab = pickle.load(f)
    else:
        # Build vocabularies from data
        all_src, all_tgt = [], []
        
        for split in ['train', 'dev']:
            path = os.path.join(base_path, f"{language}.translit.sampled.{split}.tsv")
            for s, t in read_tsv(path):
                all_src.append(s)
                all_tgt.append(t)
        
        
        # Build vocabularies
        src_vocab = CharVocab.build_from_texts(all_src)
        tgt_vocab = CharVocab.build_from_texts(all_tgt)
        
        # Cache vocabularies
        if use_cached_vocab:
            with open(vocab_cache_path, 'wb') as f:
                pickle.dump((src_vocab, tgt_vocab), f)
    
    # Common DataLoader arguments
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers and num_workers > 0,
        pin_memory=(device == 'cuda')
    )
    
    # Create data loaders for each split
    loaders = {}
    
    
    splits = {'train': 'train', 'dev': 'dev', 'test': 'test'}
    for split_name, file_split in splits.items():
        path = os.path.join(base_path, f"{language}.translit.sampled.{file_split}.tsv")
        ds = TransliterationDataset(path, src_vocab, tgt_vocab, format='dakshina')
        loaders[split_name] = DataLoader(
            ds,
            shuffle=(split_name == 'train'),
            collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab),
            **loader_kwargs
        )
    
    
    return loaders, src_vocab, tgt_vocab