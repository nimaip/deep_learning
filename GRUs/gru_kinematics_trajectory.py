import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

def generate_spiral_data(num_points=1000, noise_std=0.5):
    t = np.linspace(0, 20, num_points)
    x = t * np.cos(t)
    y = t * np.sin(t)
    
    x += np.random.normal(0, noise_std, num_points)
    y += np.random.normal(0, noise_std, num_points)
    
    data = np.stack([x, y], axis=1)
    return torch.tensor(data, dtype=torch.float32)

trajectory = generate_spiral_data()


def create_sequences(data, seq_length):
    xs = []
    ys = []
    
    for i in range(len(data) - seq_length):
        x_seq = data[i:(i + seq_length)]
        y_target = data[i + seq_length]
        
        xs.append(x_seq)
        ys.append(y_target)
        
    return torch.stack(xs), torch.stack(ys)

SEQ_LENGTH = 20
X, y = create_sequences(trajectory, SEQ_LENGTH)

train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

class KinematicLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1):
        super(KinematicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        lstm_out, (hidden_state, cell_state) = self.lstm(x)
        final_time_step_out = lstm_out[:, -1, :]
        prediction = self.fc(final_time_step_out)
        
        return prediction

model = KinematicLSTM(input_size=2, hidden_size=64)
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(model)