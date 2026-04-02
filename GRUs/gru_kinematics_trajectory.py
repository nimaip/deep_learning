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
trajectory_deltas = trajectory[1:] - trajectory[:-1]

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
X, y = create_sequences(trajectory_deltas, SEQ_LENGTH)

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
        lstm_out, _ = self.lstm(x)
        final_time_step_out = lstm_out[:, -1, :]
        prediction = self.fc(final_time_step_out)
        
        return prediction

model = KinematicLSTM(input_size=2, hidden_size=64)
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 150
train_losses = []
print("Starting Training...")
model.train()

for epoch in range(epochs):
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    train_losses.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}')

print("Training Complete!")

print("Evaluating Model...")
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test)
    print(f"Test MSE Loss: {test_loss.item():.4f}")

test_base_positions = trajectory[train_size + SEQ_LENGTH : -1].numpy()
test_preds_absolute_np = test_base_positions + test_predictions.numpy()
y_test_absolute_np = trajectory[train_size + SEQ_LENGTH + 1 : ].numpy()

y_train_absolute_np = trajectory[SEQ_LENGTH + 1 : train_size + SEQ_LENGTH + 1].numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(train_losses, color='red', linewidth=2)
ax1.set_title("Training Loss (MSE) over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)

ax2.plot(y_train_absolute_np[:, 0], y_train_absolute_np[:, 1], label="Training Data (Seen)", color='gray', linestyle='--', alpha=0.5)
ax2.plot(y_test_absolute_np[:, 0], y_test_absolute_np[:, 1], label="True Test Trajectory", color='blue', alpha=0.5, linewidth=4)
ax2.plot(test_preds_absolute_np[:, 0], test_preds_absolute_np[:, 1], label="LSTM Predictions", color='orange', linewidth=2)

ax2.set_title("Kinematic Tracking on Unseen Trajectory (Delta Model)")
ax2.set_xlabel("X Coordinate")
ax2.set_ylabel("Y Coordinate")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()