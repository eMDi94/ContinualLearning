import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .representation import iCaRL_update_representation
from .exemplar_sets import iCaRL_construct_exemplar_set, iCaRL_reduce_exemplar_set
from .classification import iCaRL_classify
from iCaRL.networks import iCaRLNetwork


class iCaRL(nn.Module):

    def __init__(self, feature_net, feature_vector_size, lr=0.001, training_epochs=100,
                 batch_size=12):
        super(iCaRL, self).__init__()
        self.iCaRL_net = iCaRLNetwork(feature_net, 0, feature_vector_size)
        self.exemplars = []
        self.__memory = 0
        self.lr = lr
        self.training_epochs = training_epochs
        self.batch_size = batch_size

    @property
    def number_of_classes(self):
        return self.iCaRL_net.number_of_classes

    @property
    def number_of_exemplars_per_class(self):
        if len(self.exemplars) > 0:
            return int(self.__memory / len(self.exemplars))
        return None

    def train_function(self, dataset, n_of_exemplars, n_of_new_data):
        """
        I provided a default training function.
        If you need your own, inherit from iCaRL and override this function. It should be everything you need to
        do
        :param dataset:
        :param n_of_exemplars:
        :param n_of_new_data:
        :return:
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        for _ in range(self.training_epochs):
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for training_set, labels in data_loader:
                optimizer.zero_grad()
                with torch.enable_grad():
                    # Run to obtain the scores for the new classes
                    final_output = self.__call__(training_set)

                    # Scores of the already known classes
                    output = final_output[:, n_of_exemplars]

                    # Scores of the new classes
                    new_scores = final_output[:, -n_of_new_data:]

                    # Compute the loss function
                    # First the distillation term with is simpler
                    distillation_term = torch.mm(output, output.log()) + torch.mm((output - 1).neg(), (output - 1).neg().log())

                    # This is the classification term
                    out = (new_scores - 1).neg().log()
                    mask = torch.zeros_like(out, dtype=torch.long, device=out.device)
                    mask[:, labels - n_of_exemplars] = 1
                    out[mask] = new_scores[mask].log()
                    classification_term = torch.sum(out, dim=1)

                    # Compute the final loss
                    loss = classification_term + distillation_term

                # Make the backward pass
                loss.backward()
                optimizer.step()

    def incremental_train(self, new_classes_exemplars, memory_size):
        self.iCaRL_net.add_classes(len(new_classes_exemplars))
        iCaRL_update_representation(self, new_classes_exemplars, self.exemplars, self.train_function)

        m = int(memory_size / self.number_of_classes)
        if self.number_of_exemplars_per_class is not None and self.number_of_exemplars_per_class < m:
            raise ValueError("It's not possible to increase the dimension of the sets for each class.")
        self.__memory = memory_size

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
