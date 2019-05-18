import torch


def iCaRL_update_representation(model, training_examples, exemplar_sets):
    """
    iCaRL function to update the representation of each class
    :param model: The NN
    :param training_examples: Python Dictionary in the form (class: torch.Tensor)
    :param exemplar_sets: Python Iterable which contain the stored examples
    :return: the updated model (for convenie   nce only)
    """

    return model
