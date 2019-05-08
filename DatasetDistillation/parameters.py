import torch.nn as nn


weights_distribution = nn.init.xavier_normal_
weights_batch_size = 100
distilled_dataset_size = 50
step_size = 0.001
optimization_iterations = 300
lr0 = 0.0001
criterion = nn.Softmax()
batch_size = 100
num_workers = 2
