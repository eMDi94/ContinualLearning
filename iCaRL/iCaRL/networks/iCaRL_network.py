import torch
import torch.nn as nn
import torch.nn.functional as F


class _SigmoidsOutputLayer(nn.Module):

    def __init__(self, n_initial_classes, feature_vector_size):
        super(_SigmoidsOutputLayer, self).__init__()
        self.__number_of_classes = n_initial_classes
        self.feature_vector_size = feature_vector_size
        self.weights = nn.Parameter(torch.randn(n_initial_classes, feature_vector_size, dtype=torch.float,
                                                requires_grad=True))
        self.biases = nn.Parameter(torch.randn(n_initial_classes, dtype=torch.float, requires_grad=True))

    @property
    def number_of_classes(self):
        return self.__number_of_classes

    def add_classes(self, new_classes):
        if new_classes < 0:
            raise ValueError("It's not possible to reduce the number of classes.")
        self.__number_of_classes += new_classes
        new_weights = torch.cat((self.weights,
                                 torch.randn(new_classes, self.feature_vector_size, dtype=torch.float,
                                            requires_grad=True, device=self.weights.device)), dim=0)
        self.weights = nn.Parameter(new_weights)
        new_biases = torch.cat((self.biases,
                                torch.randn(new_classes, dtype=torch.float, requires_grad=True,
                                            device=self.biases.device)), dim=0)
        self.biases = nn.Parameter(new_biases)

    def forward(self, x):
        x = F.linear(x, self.weights, self.biases)
        return torch.sigmoid(x)


class iCaRLNetwork(nn.Module):

    def __init__(self, feature_network, n_initial_classes, feature_vector_size):
        super(iCaRLNetwork, self).__init__()
        self.feature_net = feature_network
        self.out_layer = _SigmoidsOutputLayer(n_initial_classes, feature_vector_size)

    @property
    def number_of_classes(self):
        return self.out_layer.number_of_classes

    def add_classes(self, new_classes):
        self.out_layer.add_classes(new_classes)

    def forward(self, x):
        x = self.feature_net(x)
        # L2 normalization
        norm = torch.norm(x)
        x /= norm

        x = self.out_layer(x)
        return x
