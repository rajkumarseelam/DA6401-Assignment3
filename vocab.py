# Enhanced vocabulary handling

import json

class CharVocab:
    def __init__(self, tokens=None, specials=['<pad>','<sos>','<eos>','<unk>']):
        self.specials = specials
        self.idx2char = list(specials) + (tokens or [])
        self.char2idx = {ch:i for i,ch in enumerate(self.idx2char)}

    @classmethod
    def build_from_texts(cls, texts):
        """Build vocabulary from a list of texts"""
        chars = sorted({c for line in texts for c in line})
        return cls(tokens=chars)
    
    @classmethod
    def build_from_file(cls, file_path, src_col='src', tgt_col='trg', is_csv=True):
        """
        Build vocabulary from a data file (CSV or TSV)
        
        Args:
            file_path (str): Path to the data file
            src_col (str): Name of the source column (for CSV)
            tgt_col (str): Name of the target column (for CSV)
            is_csv (bool): Whether the file is CSV (True) or TSV (False)
        """
        if is_csv:
            import pandas as pd
            df = pd.read_csv(file_path, header=None, names=[src_col, tgt_col])
            texts = df[src_col].dropna().tolist() + df[tgt_col].dropna().tolist()
        else:
            texts = []
            with open(file_path, encoding='utf-8') as f:
                for ln in f:
                    parts = ln.strip().split('\t')
                    if len(parts) >= 2:
                        texts.extend([parts[0], parts[1]])
        
        return cls.build_from_texts(texts)

    def save(self, path):
        """Save vocabulary to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.idx2char, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        """Load vocabulary from JSON file"""
        with open(path, encoding='utf-8') as f:
            idx2char = json.load(f)
        
        inst = cls(tokens=[])
        inst.idx2char = idx2char
        inst.char2idx = {c:i for i,c in enumerate(idx2char)}
        return inst

    def encode(self, text, add_sos=False, add_eos=False):
        
        seq = []
        if add_sos: seq.append(self.char2idx['<sos>'])
        for c in text:
            seq.append(self.char2idx.get(c, self.char2idx['<unk>']))
        if add_eos: seq.append(self.char2idx['<eos>'])
        return seq

    def decode(self, idxs, strip_specials=True, join=True):
       
        # Convert tensor to list if needed
        if hasattr(idxs, 'tolist'):
            idxs = idxs.tolist()
            
        # Convert indices to characters
        chars = [self.idx2char[i] for i in idxs if i < len(self.idx2char)]
        
        # Remove special tokens if requested
        if strip_specials:
            chars = [c for c in chars if c not in self.specials]
            
        # Return as string or list
        return ''.join(chars) if join else chars
    
    def batch_decode(self, batch_idxs, strip_specials=True):
        
        return [self.decode(seq, strip_specials=strip_specials) for seq in batch_idxs]
    
    def get_stats(self):
        """Get vocabulary statistics"""
        return {
            'size': len(self.idx2char),
            'num_specials': len(self.specials),
            'num_chars': len(self.idx2char) - len(self.specials)
        }
    
    def __len__(self):
        return len(self.idx2char)

    @property
    def pad_idx(self): return self.char2idx['<pad>']
    
    @property
    def sos_idx(self): return self.char2idx['<sos>']
    
    @property
    def eos_idx(self): return self.char2idx['<eos>']
    
    @property
    def unk_idx(self): return self.char2idx['<unk>']
    
    @property
    def size(self): return len(self.idx2char)