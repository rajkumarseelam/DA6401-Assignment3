import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from typing import Dict, Tuple, List, Optional, Union, Any, Callable

def set_random_seeds(seed_value: int = 42) -> None:
    """Set random seeds for all libraries to ensure reproducibility"""
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SourceEncoder(nn.Module):
    """Encoder module for sequence-to-sequence transliteration models"""
    
    def __init__(self, 
                 vocabulary_size: int, 
                 embedding_dim: int, 
                 hidden_dim: int, 
                 num_layers: int = 1,
                 rnn_type: str = 'LSTM', 
                 dropout_rate: float = 0.0,
                 use_bidirectional: bool = False):
        
        super().__init__()
        
        # Character embedding layer
        self.char_embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        
        # Configuration
        self.use_bidirectional = use_bidirectional
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Calculate output dimensions
        self.output_dim = hidden_dim * 2 if use_bidirectional else hidden_dim
        
        # Select RNN type
        rnn_classes = {
            'LSTM': nn.LSTM, 
            'GRU': nn.GRU, 
            'RNN': nn.RNN
        }
        
        if rnn_type not in rnn_classes:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Use 'LSTM', 'GRU', or 'RNN'")
        
        rnn_class = rnn_classes[rnn_type]
        
        # Create the RNN
        self.sequence_encoder = rnn_class(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=use_bidirectional
        )

    def forward(self, source_tokens: torch.Tensor, 
                token_lengths: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        
        # Get character embeddings
        embedded = self.char_embeddings(source_tokens)  # [B, T, E]
        
        # Pack sequences for efficiency
        packed_input = pack_padded_sequence(
            embedded, 
            token_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Process through RNN
        packed_output, final_states = self.sequence_encoder(packed_input)
        
        # Unpack sequences
        encoded_sequences, _ = pad_packed_sequence(
            packed_output, 
            batch_first=True
        )  # [B, T, H*dirs]
        
        # Process final states for bidirectional models
        if self.use_bidirectional:
            if self.rnn_type == 'LSTM':
                # For LSTM we have hidden and cell states
                hidden_states, cell_states = final_states
                
                # Combine forward/backward by averaging
                fwd_hidden = hidden_states[0:self.num_layers]
                bwd_hidden = hidden_states[self.num_layers:]
                merged_hidden = (fwd_hidden + bwd_hidden) / 2
                
                fwd_cell = cell_states[0:self.num_layers]
                bwd_cell = cell_states[self.num_layers:]
                merged_cell = (fwd_cell + bwd_cell) / 2
                
                final_states = (merged_hidden, merged_cell)
            else:
                # For GRU/RNN we only have hidden states
                fwd_hidden = final_states[0:self.num_layers]
                bwd_hidden = final_states[self.num_layers:]
                final_states = (fwd_hidden + bwd_hidden) / 2
                
        return encoded_sequences, final_states


class AttentionMechanism(nn.Module):
    """Neural attention mechanism for focusing on relevant encoder states"""
    
    def __init__(self, encoder_dim: int, decoder_dim: int):
        
        super().__init__()
        
        # Layers for attention computation
        self.attention_transform = nn.Linear(encoder_dim + decoder_dim, decoder_dim)
        self.attention_scorer = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, decoder_state: torch.Tensor, 
                encoder_outputs: torch.Tensor,
                input_mask: torch.Tensor) -> torch.Tensor:
       
        # Get dimensions
        batch_size, seq_len, _ = encoder_outputs.size()
        
        # Expand decoder state to match encoder outputs
        expanded_state = decoder_state.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Compute attention scores
        concat_states = torch.cat((expanded_state, encoder_outputs), dim=2)
        attention_features = torch.tanh(self.attention_transform(concat_states))
        attention_scores = self.attention_scorer(attention_features).squeeze(2)
        
        # Apply mask to ignore padding tokens
        attention_scores = attention_scores.masked_fill(~input_mask, -1e9)
        
        # Apply softmax to get attention weights
        return torch.softmax(attention_scores, dim=1)


class TargetDecoder(nn.Module):
    """Decoder for generating target sequences with optional attention"""
    
    def __init__(self, 
                 vocabulary_size: int, 
                 embedding_dim: int,
                 encoder_dim: int, 
                 decoder_dim: int,
                 num_layers: int = 1, 
                 rnn_type: str = 'LSTM', 
                 dropout_rate: float = 0.0,
                 enable_attention: bool = True):
       
        super().__init__()
        
        # Configuration
        self.enable_attention = enable_attention
        self.rnn_type = rnn_type
        
        # Character embedding layer
        self.char_embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        
        # Attention mechanism
        if enable_attention:
            self.attention = AttentionMechanism(encoder_dim, decoder_dim)
            rnn_input_dim = embedding_dim + encoder_dim
            projection_input_dim = decoder_dim + encoder_dim + embedding_dim
        else:
            rnn_input_dim = embedding_dim
            projection_input_dim = decoder_dim + embedding_dim
        
        # Select RNN type
        rnn_classes = {
            'LSTM': nn.LSTM, 
            'GRU': nn.GRU, 
            'RNN': nn.RNN
        }
        
        if rnn_type not in rnn_classes:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Use 'LSTM', 'GRU', or 'RNN'")
        
        rnn_class = rnn_classes[rnn_type]
        
        # Create the RNN
        self.sequence_decoder = rnn_class(
            input_size=rnn_input_dim,
            hidden_size=decoder_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Output projection layer
        self.output_projection = nn.Linear(projection_input_dim, vocabulary_size)

    def forward(self, 
                token: torch.Tensor,
                hidden_state: Any,
                encoder_outputs: torch.Tensor,
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
        
        # Get character embeddings for current token
        embedded = self.char_embeddings(token).unsqueeze(1)  # [B, 1, E]
        
        # Apply attention if enabled
        if self.enable_attention:
            # Extract the top layer hidden state for attention
            if self.rnn_type == 'LSTM':
                # For LSTM, hidden state is a tuple (h_n, c_n)
                top_hidden = hidden_state[0][-1]
            else:
                # For GRU/RNN, hidden state is just h_n
                top_hidden = hidden_state[-1]
            
            # Compute attention weights
            attention_weights = self.attention(top_hidden, encoder_outputs, attention_mask)  # [B, T]
            
            # Create context vector using attention weights
            context = torch.bmm(
                attention_weights.unsqueeze(1),  # [B, 1, T]
                encoder_outputs                  # [B, T, H]
            )  # [B, 1, H]
            
            # Concatenate embedding and context for RNN input
            rnn_input = torch.cat((embedded, context), dim=2)  # [B, 1, E+H]
        else:
            context = None
            attention_weights = None
            rnn_input = embedded  # [B, 1, E]
        
        # Process through RNN
        output, new_hidden = self.sequence_decoder(rnn_input, hidden_state)
        output = output.squeeze(1)  # [B, H]
        
        # Prepare inputs for final projection
        if self.enable_attention:
            context = context.squeeze(1)  # [B, H]
            embedded = embedded.squeeze(1)  # [B, E]
            projection_input = torch.cat((output, context, embedded), dim=1)  # [B, H+H+E]
        else:
            embedded = embedded.squeeze(1)  # [B, E]
            projection_input = torch.cat((output, embedded), dim=1)  # [B, H+E]
        
        # Project to vocabulary size
        logits = self.output_projection(projection_input)  # [B, V]
        
        return logits, new_hidden, attention_weights


class Seq2SeqTransliterator(nn.Module):
    """Complete sequence-to-sequence model for transliteration"""
    
    def __init__(self, encoder: SourceEncoder, decoder: TargetDecoder, 
                 pad_token_idx: int, device_name: str = 'cpu'):
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_token_idx = pad_token_idx
        self.device = device_name

    def forward(self, 
                source: torch.Tensor, 
                source_lengths: torch.Tensor,
                target: torch.Tensor, 
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        
        # Encode source sequences
        encoder_outputs, hidden_state = self.encoder(source, source_lengths)
        
        # Create mask for attention
        attention_mask = (source != self.pad_token_idx)
        
        # Get batch size and sequence length
        batch_size, target_len = target.size()
        
        # Initialize outputs tensor
        outputs = torch.zeros(
            batch_size, 
            target_len - 1, 
            self.decoder.output_projection.out_features, 
            device=self.device
        )
        
        # Start with first token (SOS)
        current_token = target[:, 0]
        
        # Process sequence one token at a time
        for t in range(1, target_len):
            # Get output for current step
            step_output, hidden_state, _ = self.decoder(
                current_token, 
                hidden_state, 
                encoder_outputs, 
                attention_mask
            )
            
            # Store output
            outputs[:, t-1] = step_output
            
            # Decide whether to use teacher forcing for next input
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                # Use actual target token as next input
                current_token = target[:, t]
            else:
                # Use predicted token as next input
                current_token = step_output.argmax(1)
                
        return outputs

    def generate_greedy(self, 
                         source: torch.Tensor, 
                         source_lengths: torch.Tensor,
                         target_vocab: Any, 
                         max_length: int = 50) -> torch.Tensor:
        
        # Encode source sequences
        encoder_outputs, hidden_state = self.encoder(source, source_lengths)
        
        # Create mask for attention
        attention_mask = (source != self.pad_token_idx)
        
        # Get batch size
        batch_size = source.size(0)
        
        # Start with start-of-sequence token
        current_token = torch.full(
            (batch_size,), 
            target_vocab.start_index, 
            device=self.device, 
            dtype=torch.long
        )
        
        # Store generated tokens
        generated_tokens = []
        
        # Generate sequence token by token
        for _ in range(max_length):
            # Get output for current step
            step_output, hidden_state, _ = self.decoder(
                current_token, 
                hidden_state, 
                encoder_outputs, 
                attention_mask
            )
            
            # Get most likely token
            current_token = step_output.argmax(1)
            
            # Add to generated sequence
            generated_tokens.append(current_token.unsqueeze(1))
            
            # Stop if all sequences have end token
            if (current_token == target_vocab.end_index).all():
                break
                
        # Concatenate all tokens
        return torch.cat(generated_tokens, dim=1)