from json_configuration import JsonConfiguration
from parsers import distillation_parser

import torch
import torch.nn as nn

from distillation.distillation_trainer import DistillationTrainer
# from networks.le_net import LeNet
from networks.le_net_cifar_10 import LeNet
from datasets.mnist import get_mnist_data_loader
from datasets.cifar10 import get_cifar10_data_loader
from utils.weights import create_weights_init_fn
from utils.savings import save_distilled_log_data, save_distilled_data


def main():
    args = JsonConfiguration().parse(distillation_parser())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # mnist_loader = get_mnist_data_loader('./data/', args.training_data_batch_size, training=True)
    cifar_loader = get_cifar10_data_loader('./data/', args.training_data_batch_size)
    network = LeNet()
    loss_fn = nn.CrossEntropyLoss()
    weights_init_fn = create_weights_init_fn(nn.init.xavier_normal_, gain=1.0)

    distillation_trainer = DistillationTrainer([network], device, loss_fn)
    distilled_data = distillation_trainer.distill(args.distill_epochs, args.epochs, args.distilled_batches, args.n_classes,
                                 args.examples_per_class, (3, 32, 32), args.distill_lr, args.optimizer_name, args.alpha,
                                 cifar_loader, weights_init_fn, save_log_fn=save_distilled_log_data,
                                 log_img_after=args.log_img_after, log_epoch=args.log_epoch)
    save_distilled_data(distilled_data, './output/')


if __name__ == '__main__':
    main()
