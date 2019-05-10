import torch.nn as nn


class LinearRegressor(nn.Module):

    def __init__(self, n_features):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)
