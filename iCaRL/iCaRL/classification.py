import torch


def iCaRL_classify(model, input_image, exemplar_sets):
    """
    iCaRL function used for classification
    :param model: torch.nn.Module subclass used to extract feature maps
    :param input_image: The image to be passed to the model
    :param exemplar_sets: Iterable of torch.Tensor
    :return: the class label
    """
    input_image = input_image.unsqueeze(dim=0)
    x = model(input_image)
    x = x.squeeze(dim=0)
    x = x.view(-1)
    mean_tensor_size = len(exemplar_sets) + x.size()
    means = torch.zeros(mean_tensor_size, dtype=torch.float64, device=input_image.device)
    for idx, ten in enumerate(exemplar_sets):
        mean = ten.mean(dim=0)
        mean = mean.view(-1)
        means[idx] = mean
    diff = x - means
    diff = torch.sqrt(torch.mul(diff, diff))
    predicted_class = torch.argmin(diff)
    return predicted_class
