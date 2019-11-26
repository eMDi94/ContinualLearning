import torch
import torch.optim as optim
import torch.nn as nn


from networks import LeNet
from datasets import select_dataset
from parsers import model_training_parser
from json_configuration import JsonConfiguration


def select_loss(loss_name):
    if loss_name == 'Cross-Entropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError('Loss not recognized')


def main():
    args = JsonConfiguration().parse(model_training_parser())
    model = LeNet(args.channels, args.input_size, args.num_classes)
    dataset = select_dataset(args.train_dataset, './data/', args.training_batch_size, True)
    loss_fn = select_loss(args.training_loss)
    optimizer = optim.Adam(model.parameters(), args.lr)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    for epoch in range(args.epochs):
        running_loss = 0.
        for i, (data, labels) in enumerate(iter(dataset)):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            out = model(data)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Train ended')
    print('Saving model...')
    torch.save(model.state_dict(), args.save_path)
    print('End')


if __name__ == '__main__':
    main()
