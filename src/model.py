import torch
import torch.nn as nn


class ReadmittanceModel(nn.Module):
    def __init__(self, n_features, n_hidden=(100, 50)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden[0]),
            nn.ReLU(),
            nn.Linear(n_hidden[0], n_hidden[1]),
            nn.ReLU(),
            nn.Linear(n_hidden[1], 1),
        )

    def forward(self, X):
        return self.net(X).squeeze(-1)
