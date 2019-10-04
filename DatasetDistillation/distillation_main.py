import argparse
import sys
from json_configuration import import_json_configuration

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from distillation.distillation_trainer import DistillationTrainer
from networks.le_net import LeNet
from datasets.mnist import get_mnist_data_loader
from utils.weights import create_weights_init_fn
from utils.folder import create_folder_if_not_exists, append_separator_if_needed


def parse_arguments(source=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--weights-batch-size', type=int)
    parser.add_argument('--training-data-batch-size', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--log-img-after', type=int, default=10)
    parser.add_argument('--log-loss-after', type=int, default=10)
    parser.add_argument('--output-directory', type=str, default='./output/')
    args = parser.parse_args(source)
    return args


def parse_infos():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str)
    return parser.parse_args()


def save_distilled_data(distilled_data, targets, eta, output_directory):
    create_folder_if_not_exists(append_separator_if_needed(output_directory))
    labels = torch.unique(targets)
    pil_t = T.ToPILImage()
    for label in labels:
        label_folder = output_directory + str(label.item()) + '/'
        create_folder_if_not_exists(label_folder)
        for index, tensor_img in enumerate(distilled_data[targets == label]):
            img = pil_t(tensor_img)
            img.save(label_folder + str(index) + '.jpg')
    np.savez(append_separator_if_needed(output_directory) + 'lr', eta=eta.item())


def main():
    input_file = parse_infos().config_file
    args = parse_arguments(import_json_configuration(input_file))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mnist_loader = get_mnist_data_loader('./data/', args.training_data_batch_size, training=True)
    network = LeNet()
    loss_fn = nn.CrossEntropyLoss()
    weights_init_fn = create_weights_init_fn(nn.init.xavier_normal_, gain=1.0)

    distillation_trainer = DistillationTrainer(network, device, loss_fn)
    distillation_trainer.distill(args.iterations, args.weights_batch_size, weights_init_fn, mnist_loader,
                                 10, 3, args.alpha, args.learning_rate, loss_fn)
    # save_distilled_data(distilled_data, targets, eta, args.output_directory)


if __name__ == '__main__':
    main()
