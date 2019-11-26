import argparse

from distillation.distillation_trainer import DistillationTrainer
from networks import TRAINABLE_NETWORKS
from datasets import AVAILABLE_DATASET


def distillation_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data-batch-size', type=int, default=1024)
    parser.add_argument('--distill-epochs', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--distilled-batches', type=int, default=3)
    parser.add_argument('--n-classes', type=int)
    parser.add_argument('--examples-per-class', type=int, default=10)
    parser.add_argument('--distill-lr', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--optimizer-name', type=str, default='adam', choices=DistillationTrainer.OPTIMIZERS)
    parser.add_argument('--log-img-after', type=int, default=30)
    parser.add_argument('--log-epoch', type=int, default=1)
    parser.add_argument('--networks', type=str, nargs='+')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--dataset', type=str, choices=AVAILABLE_DATASET)
    return parser


def training_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-set', type=str)
    return parser


def model_training_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainable-model', type=str, choices=TRAINABLE_NETWORKS)
    parser.add_argument('--train-dataset', type=str, choices=AVAILABLE_DATASET)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--training-batch-size', type=int)
    parser.add_argument('--training-loss', type=str, choices=('Cross-Entropy',))
    parser.add_argument('--input-size', type=int)
    parser.add_argument('--channels', type=int)
    parser.add_argument('--num-classes', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--save-path', type=str)
    return parser
