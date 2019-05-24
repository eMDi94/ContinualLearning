import torch


def iCaRL_reduce_exemplar_set(exemplar_set, n_of_exemplars):
    return exemplar_set[:n_of_exemplars]


def iCaRL_construct_exemplar_set(feature_net, n_of_exemplars, current_exemplars):
    feature_vectors = feature_net(current_exemplars)
    mean = torch.mean(feature_vectors, dim=1)
    new_vectors = []
    for k in range(1, n_of_exemplars + 1):
        feat_vecs_summation = torch.sum(feature_vectors[:(k - 1)], dim=0)
        f_vec = (1 / k)*(feature_vectors + feat_vecs_summation)
        ps = (mean - f_vec).norm(dim=1)
        mean_x_idx = torch.argmin(ps)
        new_vectors.append(current_exemplars[mean_x_idx])
    new_vectors = torch.tensor(new_vectors, dtype=torch.float, device=current_exemplars.device,
                               requires_grad=current_exemplars.requires_grad)
    return new_vectors
