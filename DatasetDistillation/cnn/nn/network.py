import torchvision.models as models


def get_nn():
    nn = models.vgg19()
    return nn
