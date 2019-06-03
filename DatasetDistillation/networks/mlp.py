import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_features, output_classes):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_features, 100)
        self.linear2 = nn.Linear(100, output_classes)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        return self.softmax(x)
