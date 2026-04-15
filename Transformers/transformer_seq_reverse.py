import torch
import torch.nn as nn
import math

# --- 1. Synthetic Data Generation ---

def get_batch(batch_size, seq_len, vocab_size):
    """
    Generates a batch of random integer sequences and their exact reverse.
    We reserve 0 for Padding and 1 for the Start-Of-Sequence (SOS) token.
    """
    # Generate random tokens in the range [2, vocab_size)
    x = torch.randint(2, vocab_size, (batch_size, seq_len))
    
    # The target is the exact reverse of the input sequence
    y = torch.flip(x, dims=[1])
    
    return x, y

# --- Quick Test ---
if __name__ == "__main__":
    vocab_size = 50
    seq_len = 6
    batch_size = 2
    
    x_sample, y_sample = get_batch(batch_size, seq_len, vocab_size)
    print("Input Batch (x):\n", x_sample)
    print("Target Batch (y):\n", y_sample)