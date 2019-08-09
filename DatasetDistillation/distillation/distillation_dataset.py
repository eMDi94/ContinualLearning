from torch.utils.data import Dataset


class DistillationDataset(Dataset):

    def __init__(self, tensor, targets):
        self.tensor = tensor
        self.targets = targets

    def __getitem__(self, index):
        return self.tensor[index], self.targets[index]

    def __len__(self):
        return self.tensor.size(0)
