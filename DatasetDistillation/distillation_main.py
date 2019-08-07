from __future__ import print_function

from networks import LeNet
from datasets import get_mnist_training_data_loader
from utils import device, create_weights_init_fn, FlatTransform
from utils.savings import save_le_net_img
from distillation.distillation_trainer import DistillationTrainer

import torch
import torchvision.transforms as T


def parse():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--number_of_iterations', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weights_batch_size', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.001)

    return parser.parse_args()


def distillation():
    net = LeNet()

    init_fn = create_weights_init_fn(torch.nn.init.normal_, mean=0, std=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    args = parse()
    trainer = DistillationTrainer(net, args.number_of_iterations, (784,),
                                  init_fn, args.learning_rate, args.weights_batch_size, loss_fn,
                                  args.alpha, device)

    mnist = get_mnist_training_data_loader('./data/', 64)

    mnist.dataset.transform = T.Compose([
        mnist.dataset.transform,
    ])

    trainer.classification_distillation(mnist, 10, 1)
    # Once sure that the distillation works I'll test the saving phase
    # trainer.save_distilled_data('./output', save_le_net_img)


if __name__ == '__main__':
    distillation()
