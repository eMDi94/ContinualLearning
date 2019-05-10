import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from .dataset import RegressionDataset
from .linear_regressor import LinearRegressor
from globals import DEBUG

from distillation import distill_dataset

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main_regression():
    n_features = 2
    n_samples = 100

    dset = RegressionDataset(n_samples, n_features, 20.)
    if DEBUG is True:
        if n_features == 1:
            plt.scatter(dset.samples, dset.targets)
            plt.show()
        elif n_features == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(dset.samples[:, 0], dset.samples[:, 1], dset.targets)
            plt.show()

    regressor = LinearRegressor(n_features)
    sampler = RandomSampler(TensorDataset(torch.arange(n_samples)), True)
    data_loader = DataLoader(dset, 5, sampler=sampler)

    criterion = nn.MSELoss()
    number_of_distilled_examples = 50
    step_size = 0.0001
    lr0 = 0.0001
    optimization_iterations = 300
    weights_batch_size = 10

    results = distill_dataset(regressor, number_of_distilled_examples, (n_features, ), step_size, optimization_iterations,
                    lr0, data_loader, criterion, weights_batch_size)
    distilled_data, distilled_target, lr, records = results

    for record in records:
        numpy_values = record.numpy()
        plt.plot(np.arange(len(numpy_values)), numpy_values)
    plt.show()
