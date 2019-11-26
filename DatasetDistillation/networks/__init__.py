import torch

from .le_net import LeNet
from .alex_cifar_net import AlexCifarNet


TRAINABLE_NETWORKS = ('LeNet', 'AlexCifarNet',)
NETWORKS = ('LeNet', 'AlexCifarNet',)

TRAINABLE_NETWORKS_ARGUMENTS = {
    'LeNet': ('input_channels', 'input_dims', 'num_classes',),
    'AlexCifarNet': ('input_channels', 'num_classes',)
}


def select_params(net_name, net_params):
    params = {}
    for p_name in net_params.keys():
        if p_name in TRAINABLE_NETWORKS_ARGUMENTS[net_name]:
            params[p_name] = net_params[p_name]
    return params


def select_network(net_name, pretrained, net_params):
    if net_name == TRAINABLE_NETWORKS[0]:
        params = select_params(net_name, net_params)
        net = LeNet(**params)
        if pretrained is True:
            net.load_state_dict(torch.load('./weights/LeNet.pth'))
    elif net_name == TRAINABLE_NETWORKS[1]:
        params = select_params(net_name, net_params)
        net = AlexCifarNet(**params)
        if pretrained is True:
            net.load_state_dict(torch.load('./weights/AlexCifarNet.pth'))
    else:
        raise ValueError('Network name not recognized')
    return net


def get_networks(network_names, pretrained, net_params=None):
    net_params = net_params if net_params is not None else {}
    return tuple([select_network(net_name, pretrained, net_params.get(net_name, {})) for net_name in network_names])
