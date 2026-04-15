import torch
import torch.nn as nn
import math


def get_batch(batch_size, seq_len, vocab_size):
    """
    Generates a batch of random integer sequences and their exact reverse.
    We reserve 0 for Padding and 1 for the Start-Of-Sequence (SOS) token.
    """
    x = torch.randint(2, vocab_size, (batch_size, seq_len))
    y = torch.flip(x, dims=[1])
    
    return x, y

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers, 
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, tgt_mask=None):
        src = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.out(output)