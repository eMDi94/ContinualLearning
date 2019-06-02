import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.datasets import make_classification, make_regression

from globals import device
from distillation import ClassificationDistillationModule


class MyModule(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, 10)
        self.linear2 = nn.Linear(10, 5)
        self.linear3 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x


class ClassificationDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, i):
        return self.features[i], self.labels[i]

    def __len__(self):
        return self.labels.size()[0]


def main():
    x = torch.randn(10, 5, 3, requires_grad=True)
    y = torch.randn(10)
    w = torch.randn(5, 3, requires_grad=True)
    x_c = x.reshape(10, -1)
    w_c = w.reshape(-1)
    out = torch.matmul((x_c ** 2), w_c)
    l = out.mean()
    l.backward()
    print(x_c.grad)
    print(x.grad)


main()
