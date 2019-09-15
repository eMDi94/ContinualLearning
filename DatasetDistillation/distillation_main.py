import argparse

import torch
import torch.nn as nn

from distillation.distillation_trainer import DistillationTrainer
from networks.le_net import LeNet
from datasets.mnist import get_mnist_data_loader
from utils.weights import create_weights_init_fn


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--weights-batch-size', type=int)
    parser.add_argument('--training-data-batch-size', type=int)
    parser.add_argument('--training-data-batches', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--log-img-after', type=int, default=10)
    parser.add_argument('--log-loss-after', type=int, default=10)
    args = parser.parse_args()
    return args


def main():
    args = parse()

    device = torch.device(args.device)
    mnist_loader = get_mnist_data_loader('./data/', args.training_data_batch_size, training=True)
    network = LeNet()
    loss_fn = nn.CrossEntropyLoss()
    weights_init_fn = create_weights_init_fn(nn.init.normal_, mean=0, std=0.1)

    distillation_trainer = DistillationTrainer(network, device, loss_fn)
    distillation_trainer.classification_distillation(args.learning_rate, args.alpha, mnist_loader,
                                                     args.iterations, (1, 28, 28), 10, 3,
                                                     DistillationTrainer.CONSTANT_WEIGHTS_INIT, weights_init_fn,
                                                     args.weights_batch_size, args.training_data_batches,
                                                     save_data_after=args.log_img_after,
                                                     log_loss_after=args.log_loss_after)


if __name__ == '__main__':
    main()
