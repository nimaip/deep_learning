import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(42)
time_steps = np.linspace(0, 100, 500)

# general upward trend of data with some noise
prices = time_steps * 0.5 + np.sin(time_steps) * 2 + np.random.normal(0, 0.5, 500)

prices_normalized = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))


def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X, y = create_sequences(prices_normalized, seq_length)


X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) 
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

split = int(0.8 * len(X_tensor))
X_train, X_test = X_tensor[:split], X_tensor[split:]
y_train, y_test = y_tensor[:split], y_tensor[split:]

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# print(f"X_train shape: {X_train.shape}")
# print(f"y_train shape: {y_train.shape}") 


class StockRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(StockRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, hn = self.rnn(x, h0)
        final_time_step_out = out[:, -1, :]
        prediction = self.fc(final_time_step_out)
        return prediction

model = StockRNN(input_size=1, hidden_size=32, num_layers=1, output_size=1)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005) # lr is the learning ratel
num_epochs = 100

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")