from torch import nn
import torch.nn.functional as F
import torch


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
        self.c0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = self.h0.expand(-1, x.size(0), -1).contiguous()
        c0 = self.c0.expand(-1, x.size(0), -1).contiguous()

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        out = F.softplus(out, beta=35, threshold=1)
        return out


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_size

        # Number of hidden layers
        self.num_layers = num_layers

        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, input_dim)
        # batch_dim = number of samples per batch
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_dim).requires_grad_()

        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        out, hn = self.rnn(x, h0.detach())

        # Index hidden state of last time step
        # out.size() --> 100, 28, 10
        # out[:, -1, :] --> 100, 10 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        out = F.softplus(out, beta=35, threshold=1)
        # out.size() --> 100, 10
        return out
