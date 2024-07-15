import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=2):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define multiple RNN layers
        self.rnn_layers = nn.ModuleList([nn.RNN(input_size, hidden_size, batch_first=True)])
        for _ in range(num_layers - 1):
            self.rnn_layers.append(nn.RNN(hidden_size, hidden_size, batch_first=True))

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden states for each layer
        h = [torch.zeros(1, x.size(0), self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        out = x
        for i, layer in enumerate(self.rnn_layers):
            out, h[i] = layer(out, h[i])

        # Selecting the output from the last time step
        out = self.fc(out[:, -1, :])
        return out
