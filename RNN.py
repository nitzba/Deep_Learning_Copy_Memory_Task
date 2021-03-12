import torch
import torch.nn as nn
import math


class RNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, nonlinearity="tanh"):
        super().__init__()

        self.nonlinearity_dictionary = {"relu": torch.relu, "tanh": torch.tanh}
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nonlinearity = self.nonlinearity_dictionary[nonlinearity]

        self.input_weights = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.hidden_weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.init_weights()

    def init_weights(self):
        K = 1.0 / math.sqrt(self.hidden_dim)
        self.input_weights.data.uniform_(-K, K)
        self.hidden_weights.data.uniform_(-K, K)

    def forward(self, features, hidden_state=None):
        batch, seq_sz = features.size()

        if hidden_state is None:
            h_t = torch.zeros(batch, self.hidden_dim).to(features.device)
        else:
            h_t = hidden_state

        i_w_x = features.matmul(self.input_weights.t())
        h_w_h_t = h_t.matmul(self.hidden_weights.t())

        o_t = i_w_x + h_w_h_t
        b_t = self.nonlinearity(o_t)

        return b_t
