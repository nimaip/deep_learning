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
    

def generate_square_subsequent_mask(sz):
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    sz: The length of the sequence
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


vocab_size = 20 
d_model = 64
nhead = 4
num_layers = 2
dim_feedforward = 128
batch_size = 16
seq_len = 8

model = TransformerModel(vocab_size, d_model, nhead, num_layers, dim_feedforward)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")
for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    
    src, y = get_batch(batch_size, seq_len, vocab_size)
    
    sos = torch.ones((batch_size, 1), dtype=torch.long)
    tgt_input = torch.cat((sos, y[:, :-1]), dim=1)
    tgt_mask = generate_square_subsequent_mask(seq_len)
    output = model(src, tgt_input, tgt_mask=tgt_mask)
    loss = criterion(output.view(-1, vocab_size), y.view(-1))
    
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("Training Complete!")



def reverse_sequence(model, input_seq, vocab_size, max_len):
    model.eval()
    with torch.no_grad():
        src = input_seq.unsqueeze(0) 
        
        tgt_input = torch.tensor([[1]], dtype=torch.long)
        
        for _ in range(max_len):
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1))
            output = model(src, tgt_input, tgt_mask=tgt_mask)
            next_token = output[0, -1, :].argmax().item()
            
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            tgt_input = torch.cat((tgt_input, next_token_tensor), dim=1)
            
            if tgt_input.size(1) > max_len:
                break
                
        return tgt_input[0, 1:] 
test_input = torch.randint(2, vocab_size, (seq_len,))
result = reverse_sequence(model, test_input, vocab_size, seq_len)

print("\n--- Final Test ---")
print(f"Original: {test_input.tolist()}")
print(f"Expected: {test_input.flip(0).tolist()}")
print(f"Predicted: {result.tolist()}")