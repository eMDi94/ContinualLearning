import torch.nn as nn

from .representation import iCaRL_update_representation
from .exemplar_sets import iCaRL_construct_exemplar_set, iCaRL_reduce_exemplar_set
from .classification import iCaRL_classify
from iCaRL.networks import iCaRLNetwork


class iCaRL(nn.Module):

    def __init__(self, feature_net, feature_vector_size):
        super(iCaRL, self).__init__()
        self.iCaRL_net = iCaRLNetwork(feature_net, 0, feature_vector_size)
        self.exemplars = []
        self.__memory = 0

    @property
    def number_of_classes(self):
        return self.iCaRL_net.number_of_classes

    @property
    def number_of_exemplars_per_class(self):
        if len(self.exemplars) > 0:
            return int(self.__memory / len(self.exemplars))
        return None

    def incremental_train(self, new_classes_exemplars, memory_size):
        self.iCaRL_net.add_classes(len(new_classes_exemplars))
        iCaRL_update_representation(self, new_classes_exemplars, self.exemplars)

        m = int(memory_size / self.number_of_classes)
        if self.number_of_exemplars_per_class is not None and self.number_of_exemplars_per_class < m:
            raise ValueError("It's not possible to increase the dimension of the sets for each class.")

        new_exemplars = []
        for exemplars_t in self.exemplars:
            new_exemplars.append(iCaRL_reduce_exemplar_set(exemplars_t, m))
        for exemplars_t in new_classes_exemplars:
            new_exemplars.append(iCaRL_construct_exemplar_set(self.iCaRL_net.feature_net, m, exemplars_t))
        self.exemplars = new_exemplars

    def forward(self, x):
        return self.iCaRL_net(x)

    def classify(self, input_tensor):
        return iCaRL_classify(self, input_tensor, self.exemplars)
