import torch

from .meta_model_utils import MetaModelUtils


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
