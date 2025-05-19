import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random

# For reproducibility
def seed_everything(seed=42):
    """Set random seed for all major libraries"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, layers=1, cell='LSTM', dropout=0.0, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bidirectional = bidirectional
        self.cell_type = cell
        self.layers = layers
        self.hidden_size = hid_size
        
        # Output size will be doubled if bidirectional
        self.output_size = hid_size * 2 if bidirectional else hid_size
        
        rnn_cls = {'LSTM': nn.LSTM, 'GRU': nn.GRU, 'RNN': nn.RNN}[cell]
        self.rnn = rnn_cls(emb_size,
                         hid_size,
                         num_layers=layers,
                         dropout=dropout if layers>1 else 0.0,
                         batch_first=True,
                         bidirectional=bidirectional)

    def forward(self, src, lengths):
        # src: [B, T], lengths: [B]
        embedded = self.embedding(src)  # [B, T, E]
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, T, H*dirs]
        
        # If bidirectional, we need to process hidden state properly
        if self.bidirectional:
            if self.cell_type == 'LSTM':
                # For LSTM we have both hidden and cell states
                h_n, c_n = hidden
                # Combine forward and backward states by averaging
                h_n = torch.add(h_n[0:self.layers], h_n[self.layers:]) / 2
                c_n = torch.add(c_n[0:self.layers], c_n[self.layers:]) / 2
                hidden = (h_n, c_n)
            else:
                # For GRU/RNN we only have hidden state
                hidden = torch.add(hidden[0:self.layers], hidden[self.layers:]) / 2
                
        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid, dec_hid):
        super().__init__()
        self.attn = nn.Linear(enc_hid + dec_hid, dec_hid)
        self.v = nn.Linear(dec_hid, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [B, H], encoder_outputs: [B, T, H], mask: [B, T]
        B, T, H = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, T, 1)               # [B, T, H]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [B, T, H]
        scores = self.v(energy).squeeze(2)                        # [B, T]
        scores = scores.masked_fill(~mask, -1e9)
        return torch.softmax(scores, dim=1)                       # [B, T]


class Decoder(nn.Module):
    """
    One class, two modes:
        • use_attn=True  – Bahdanau attention (default)
        • use_attn=False – Plain RNN decoder (no attention)

    Forward always returns (logits, hidden, attn_weights_or_None),
    so Seq2Seq code stays unchanged.
    """
    def __init__(self, vocab_size, emb_size, enc_hid, dec_hid,
                 layers=1, cell="LSTM", dropout=0.0, use_attn=False):
        super().__init__()
        self.use_attn = use_attn
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.cell_type = cell

        # ----- dimensions depend on whether we concatenate context -----
        if use_attn:
            self.attention = BahdanauAttention(enc_hid, dec_hid)
            rnn_input_dim = emb_size + enc_hid            # [E ⊕ Henc]
            fc_input_dim  = dec_hid + enc_hid + emb_size  # [Hdec ⊕ Henc ⊕ E]
        else:
            rnn_input_dim = emb_size
            fc_input_dim  = dec_hid + emb_size

        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}[cell]
        self.rnn = rnn_cls(rnn_input_dim, dec_hid,
                           num_layers=layers,
                           dropout=dropout if layers > 1 else 0.0,
                           batch_first=True)
        self.fc = nn.Linear(fc_input_dim, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs, mask):
        """
        input_token : [B]
        hidden      : tuple|tensor  initial state for this step
        encoder_outputs : [B, Tenc, Henc]
        mask        : [B, Tenc]  (ignored when use_attn=False)
        """
        emb = self.embedding(input_token).unsqueeze(1)     # [B,1,E]

        if self.use_attn:
            # ---- additive attention ----
            if self.cell_type == 'LSTM':
                dec_h = hidden[0][-1]
            else:
                dec_h = hidden[-1]
                
            attn_w = self.attention(dec_h, encoder_outputs, mask)          # [B,Tenc]
            ctx    = torch.bmm(attn_w.unsqueeze(1), encoder_outputs)        # [B,1,Henc]
            rnn_in = torch.cat((emb, ctx), dim=2)                           # [B,1,E+Henc]
        else:
            ctx = None
            attn_w = None
            rnn_in = emb                                                    # [B,1,E]

        out, hidden = self.rnn(rnn_in, hidden)       # [B,1,Hdec]
        out = out.squeeze(1)                         # [B,Hdec]
        emb = emb.squeeze(1)                         # [B,E]

        if self.use_attn:
            ctx = ctx.squeeze(1)                     # [B,Henc]
            logits = self.fc(torch.cat((out, ctx, emb), dim=1))
        else:
            logits = self.fc(torch.cat((out, emb), dim=1))

        return logits, hidden, attn_w


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device='cpu'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        """
        Enhanced forward with explicit teacher forcing ratio control
        """
        enc_out, hidden = self.encoder(src, src_lens)
        mask = (src != self.pad_idx)
        B, T = tgt.size()
        outputs = torch.zeros(B, T-1, self.decoder.fc.out_features, device=self.device)
        input_tok = tgt[:, 0]  # <sos>
        
        for t in range(1, T):
            out, hidden, _ = self.decoder(input_tok, hidden, enc_out, mask)
            outputs[:, t-1] = out
            
            # Teacher forcing: with probability, use ground truth as next input
            # Otherwise use predicted token
            teacher_force = random.random() < teacher_forcing_ratio
            if teacher_force:
                input_tok = tgt[:, t]
            else:
                input_tok = out.argmax(1)
                
        return outputs

    def infer_greedy(self, src, src_lens, tgt_vocab, max_len=50):
        enc_out, hidden = self.encoder(src, src_lens)
        mask = (src != self.pad_idx)
        B = src.size(0)
        input_tok = torch.full((B,), tgt_vocab.sos_idx, device=self.device, dtype=torch.long)
        generated = []
        
        for _ in range(max_len):
            out, hidden, _ = self.decoder(input_tok, hidden, enc_out, mask)
            input_tok = out.argmax(1)
            generated.append(input_tok.unsqueeze(1))
            if (input_tok == tgt_vocab.eos_idx).all():
                break
                
        return torch.cat(generated, dim=1)