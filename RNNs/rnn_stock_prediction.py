import torch
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