import argparse

from distillation.distillation_trainer import DistillationTrainer


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
    return parser


def training_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-set', type=str)
    return parser
