import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_regression


class RegressionDataset(Dataset):

    def __init__(self):
        self.x, self.y = self.get_dataset()

    def get_dataset(self):
        raise NotImplementedError

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


class RandomRegressionDataset(RegressionDataset):

    def __init__(self, n_samples, n_features, bias):
        self.n_samples = n_samples
        self.n_features = n_features
        self.bias = bias
        super(RandomRegressionDataset, self).__init__()

    def get_dataset(self):
        x, y = make_regression(self.n_samples, n_features=self.n_features, bias=self.bias)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y


class FileRegressionDataset(RegressionDataset):

    def __init__(self, filename):
        self.filename = filename
        super(FileRegressionDataset, self).__init__()

    def get_dataset(self):
        with np.load(self.filename) as data:
            data = np.load(self.filename)
            x = torch.from_numpy(data['x'])
            y = torch.from_numpy(data['y'])
            return x, y
