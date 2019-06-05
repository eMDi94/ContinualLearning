import torch

from .meta_model_utils import MetaModelUtils
from .weights import create_weights_init_fn
from .transforms import FlatTransform


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
