from parsers import distillation_parser

import torch
import torch.nn as nn

from distillation.distillation_trainer import DistillationTrainer
from utils.weights import create_weights_init_fn
from utils.savings import save_distilled_log_data, save_distilled_data
from networks import get_networks, TRAINABLE_NETWORKS
from datasets import select_dataset


def main():
    args = distillation_parser().parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset, channels, squared_dim = select_dataset(args.dataset, './data/', args.training_data_batch_size, True)

    nets_args = {}
    for net_name in args.networks:
        if net_name in TRAINABLE_NETWORKS:
            nets_args[net_name] = {
                'input_channels': channels,
                'num_classes': args.n_classes,
                'input_dims': squared_dim
            }

    networks = get_networks(args.networks, args.pretrained, nets_args)
    loss_fn = nn.CrossEntropyLoss()
    weights_init_fn = create_weights_init_fn(nn.init.xavier_normal_, gain=1.0)

    distillation_trainer = DistillationTrainer(networks, device, args.distill_epochs, args.epochs,
                                               args.distilled_batches, args.n_classes, args.examples_per_class,
                                               (channels, squared_dim, squared_dim), args.distill_lr, args.optimizer_name,
                                               args.alpha, loss_fn, weights_init_fn, dataset,
                                               save_log_fn=save_distilled_log_data)
    distilled_data = distillation_trainer.distill()
    save_distilled_data(distilled_data, './output/')


if __name__ == '__main__':
    main()
