import torch
import torch.nn as nn
import math

# this is mlp layer

class MLPLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.init_weights()

    def init_weights(self):
        K = 1.0 / math.sqrt(self.out_features)
        self.weights.data.uniform_(-K, K)

    def forward(self, features):
        output = features @ self.weights.t()
        return output

