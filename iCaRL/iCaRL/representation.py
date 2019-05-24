import torch
from torch.utils.data import Dataset


class iCaRLDataset(Dataset):

    def __init__(self, exemplars, labels):
        if exemplars.size()[0] != labels.size()[0]:
            raise ValueError('The number of elements in the set and the label must correspond.')
        self.exemplars = exemplars
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.exemplars[idx], self.labels[idx]


def get_merged_and_counts(values):
    counts = torch.empty(len(values), dtype=torch.int)
    new_set = torch.empty(0)
    for idx, ex in enumerate(values):
        counts[idx] = len(ex)
        new_set = torch.cat((new_set, ex), dim=0)
    return new_set, counts


def get_labels(exemplar_sets, training_examples):
    new_set = [*exemplar_sets]
    new_set.extend(training_examples)
    labels = torch.empty(0, dtype=torch.long)
    for label, s in enumerate(new_set):
        current_labels = torch.tensor([label]).repeat(len(s))
        labels = torch.cat((labels, current_labels), dim=0)
    return labels


def iCaRL_update_representation(model, training_examples, exemplar_sets, training_function):
    """
    iCaRL function to update the representation of new classes
    :param model: The NN
    :param training_examples: Python Iterable which contains the new classes' samples
    :param exemplar_sets: Python Iterable which contain the stored examples
    :return: the updated model (for convenience only)
    """

    # Form combined training sets
    training_exs, training_counts = get_merged_and_counts(training_examples)
    exemplars, exemplars_counts = get_merged_and_counts(exemplar_sets)
    training_set = torch.cat((exemplars, training_exs), dim=0)

    # After this add the new classes to the model
    model.add_classes(len(training_examples))
    labels = get_labels(exemplar_sets, training_examples)

    dataset = iCaRLDataset(training_set, labels)

    training_function(dataset, len(exemplar_sets), len(training_examples))

    return model
