from sklearn.datasets import make_regression

import torch
from torch.utils.data import Dataset


class RegressionDataset(Dataset):

    def __init__(self, n_samples, n_features, noise=0.):
        super(RegressionDataset, self).__init__()
        self.n_features = n_features
        self.n_samples = n_samples
        x, y = make_regression(n_samples, n_features, noise=noise)
        self.samples = torch.from_numpy(x)
        self.targets = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]

    def __len__(self):
        return self.n_samples
