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
