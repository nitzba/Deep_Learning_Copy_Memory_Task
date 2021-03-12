import torch
import torch.nn as nn
import math


class LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W = nn.Parameter(torch.randn(4 * hidden_dim, input_dim))
        self.U = nn.Parameter(torch.randn(4 * hidden_dim, hidden_dim))

        self.init_weights()

    def init_weights(self):
        K = 1.0 / math.sqrt(self.hidden_dim)
        self.W.data.uniform_(-K, K)
        self.U.data.uniform_(-K, K)

    def forward(self, features,  hidden_states=None):
        batch, feature_size = features.size()

        if hidden_states is None:
            h_t = torch.zeros(batch, self.hidden_dim).to(features.device)
            c_t = torch.zeros(batch, self.hidden_dim).to(features.device)
        else:
            h_t, c_t = hidden_states

        HS = self.hidden_dim

        gates = features.matmul(self.W.t()) + h_t.matmul(self.U.t())
        # input gate
        i_t = torch.sigmoid(gates[:, :HS])
        # forget gate
        f_t = torch.sigmoid(gates[:, HS:HS*2])
        # state gate
        s_t = torch.sigmoid(gates[:, HS*2:HS*3])
        # output gate
        o_t = torch.sigmoid(gates[:, HS*3:])

        c_t = f_t * c_t + i_t * s_t
        h_t = torch.tanh(c_t) * o_t

        return h_t, c_t
