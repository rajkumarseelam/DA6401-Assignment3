import json
from typing import List, Dict, Set, Optional, Union, Tuple

class SequenceVocabulary:
    """Character-level vocabulary for sequence-to-sequence tasks"""
    
    # Standard special tokens
    PAD = '<pad>'
    SOS = '<bos>' 
    EOS = '<eos>'
    UNK = '<unk>'
    
    def __init__(self, character_set: Optional[List[str]] = None, 
                 special_tokens: Optional[List[str]] = None):
        """Initialize vocabulary with characters and special tokens
        
        Args:
            character_set: List of characters to include in vocabulary
            special_tokens: List of special tokens (defaults to standard tokens if None)
        """
        # Set special tokens with defaults
        self.special_tokens = special_tokens or [self.PAD, self.SOS, self.EOS, self.UNK]
        
        # Initialize mappings
        self.token_to_index: Dict[str, int] = {}
        self.index_to_token: List[str] = []
        
        # Add special tokens first
        for token in self.special_tokens:
            self._add_token(token)
        
        # Add regular characters if provided
        if character_set:
            for char in character_set:
                self._add_token(char)
    
    def _add_token(self, token: str) -> int:
        """Add a token to the vocabulary"""
        if token not in self.token_to_index:
            index = len(self.index_to_token)
            self.token_to_index[token] = index
            self.index_to_token.append(token)
            return index
        return self.token_to_index[token]
    
    @classmethod
    def create_from_corpus(cls, text_samples: List[str]) -> 'SequenceVocabulary':
        """Build vocabulary from a collection of text samples"""
        # Extract unique characters from all texts
        all_chars: Set[str] = set()
        for text in text_samples:
            all_chars.update(text)
        
        # Sort characters for deterministic ordering
        char_list = sorted(list(all_chars))
        return cls(character_set=char_list)
    
    def tokenize(self, text: str, add_start: bool = False, 
                add_end: bool = False) -> List[int]:
        """Convert text to token indices"""
        token_indices = []
        
        # Add start token if requested
        if add_start:
            token_indices.append(self.token_to_index[self.SOS])
        
        # Convert each character to its index
        for char in text:
            if char in self.token_to_index:
                token_indices.append(self.token_to_index[char])
            else:
                token_indices.append(self.token_to_index[self.UNK])
        
        # Add end token if requested
        if add_end:
            token_indices.append(self.token_to_index[self.EOS])
        
        return token_indices
    
    def detokenize(self, indices: List[int], remove_special: bool = True) -> str:
        """Convert token indices back to text"""
        if hasattr(indices, 'tolist'):  # Convert tensor to list if needed
            indices = indices.tolist()
        
        tokens = []
        for idx in indices:
            if idx < len(self.index_to_token):
                token = self.index_to_token[idx]
                # Skip special tokens if requested
                if remove_special and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        return ''.join(tokens)
    
    def serialize(self, file_path: str) -> None:
        """Save vocabulary to a file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.index_to_token, f, ensure_ascii=False)
    
    @classmethod
    def deserialize(cls, file_path: str) -> 'SequenceVocabulary':
        """Load vocabulary from a file"""
        vocab = cls(character_set=[])  # Create empty vocabulary
        with open(file_path, 'r', encoding='utf-8') as f:
            vocab.index_to_token = json.load(f)
        
        # Rebuild the token to index mapping
        vocab.token_to_index = {token: idx for idx, token in enumerate(vocab.index_to_token)}
        return vocab
    
    @property
    def pad_index(self) -> int:
        """Get index of pad token"""
        return self.token_to_index[self.PAD]
    
    @property
    def start_index(self) -> int:
        """Get index of start token"""
        return self.token_to_index[self.SOS]
    
    @property
    def end_index(self) -> int:
        """Get index of end token"""
        return self.token_to_index[self.EOS]
    
    @property
    def unknown_index(self) -> int:
        """Get index of unknown token"""
        return self.token_to_index[self.UNK]
    
    @property
    def vocabulary_size(self) -> int:
        """Get total size of vocabulary"""
        return len(self.index_to_token)