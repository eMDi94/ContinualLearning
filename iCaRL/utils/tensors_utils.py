import torch


class MetaModelUtils(object):

    @classmethod
    def cat_tensors(cls, sets, requires_labels=True):
        tensor = torch.cat(sets)
        if requires_labels is True:
            labels = []
            for label in range(len(sets)):
                labels.append(torch.empty(len(sets[label])).fill_(label))
            labels = torch.cat(labels)
        else:
            labels = None
        return tensor, labels
