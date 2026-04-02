# Kinematic Trajectory Tracking with RNNs

### Task:
Predict the next $(x, y)$ coordinate in a continuous, noisy 2D sequence (a spiral trajectory) using a history of previous coordinates.

### Why an LSTM/GRU?
A standard Feedforward Neural Network (MLP) treats every input independently. If given a single $(x, y)$ point, an MLP has no mathematical context of the object's momentum, direction, or angular velocity. An LSTM or GRU, however, maintains a hidden state vector $h_t$ that carries this temporal context forward. By updating its internal memory at each time step via its gating mechanisms, the network inherently learns the underlying physical dynamics and sequence of the trajectory rather than just mapping static inputs to outputs.