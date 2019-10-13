import argparse
import os
import glob

import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image

from networks.le_net import LeNet
from distillation.distillation_trainer import DistillationTrainer
from datasets.mnist import get_mnist_data_loader


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--input-directory', type=str)
    parser.add_argument('--validation-batch-size', default=20)
    args = parser.parse_args()
    return args


def evaluate(model, training_data_loader, device):
    it = iter(training_data_loader)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in it:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def main():
    args = parse()
    model = LeNet()

    device = torch.device(args.device)
    loss_fn = torch.nn.CrossEntropyLoss()
    trainer = DistillationTrainer(model, device, loss_fn)

    targets = torch.tensor([], dtype=torch.long)
    data = torch.empty((0, 1, 28, 28), dtype=torch.float)
    to_tensor = T.ToTensor()
    for label_directory in os.listdir(args.input_directory):
        if label_directory == 'lr.npz':
            continue
        label = int(label_directory)
        g = glob.glob(args.input_directory + '/' + label_directory + '/*.jpg')
        labels = torch.tensor([label], dtype=torch.long).repeat(len(g))
        targets = torch.cat([targets, labels])
        for img_name in g:
            img = Image.open(img_name)
            t_img = to_tensor(img)
            data = torch.cat([data, t_img.unsqueeze(0)], dim=0)

    with np.load(args.input_directory + '/lr.npz') as file_data:
        eta = torch.from_numpy(file_data['eta'])

    mnist = get_mnist_data_loader('./data', args.validation_batch_size, False)

    # Accuracy without training
    print('Accuracy without training: ', evaluate(model, mnist, device))

    data, targets = data.to(device), targets.to(device)
    trainer.train(data, targets, eta, 10, 50)

    print('Accuracy after training: ', evaluate(trainer.model, mnist, device))


if __name__ == '__main__':
    main()
