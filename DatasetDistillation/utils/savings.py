import os

import torch
import torchvision.transforms as T


def save_distilled_log_data(distilled_data, targets, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    labels = torch.unique(targets)
    pil_t = T.ToPILImage()
    for label in labels:
        label_folder = output_directory + str(label.item()) + '/'
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        for index, tensor_img in enumerate(distilled_data[targets == label]):
            img = pil_t(tensor_img)
            img.save(label_folder + str(index) + '.jpg')


def save_distilled_data(data_steps, out_directory):
    os.makedirs(out_directory)
    data_steps = [(d.cpu(), l.cpu(), eta.cpu()) for (d, l, eta) in data_steps]
    torch.save(data_steps, out_directory + 'distillation')
