import torch
from torch import nn
from torch.nn import functional as F
import pdb
from torchvision import models


def compute_pairwise(z):
    return torch.norm(z[:, :, None, :] - z[:, None, :, :], p=2, dim=3)


def compute_pair_distance(z, weight=None):
    if weight is not None:
        z_pair_dist = compute_pairwise(weight) * compute_pairwise(z)
    else:
        z_pair_dist = compute_pairwise(z)
    norm_vec_z = torch.sum(z_pair_dist, dim=2)
    z_pair_dist = z_pair_dist / norm_vec_z[..., None].detach()
    return z_pair_dist


def compute_pair_unnormal_distance(z, weight=None):
    if weight is not None:
        z_pair_dist = compute_pairwise(weight) * compute_pairwise(z)
    else:
        z_pair_dist = compute_pairwise(z)
    norm_vec_z = torch.sum(z_pair_dist, dim=2)
    z_pair_dist = z_pair_dist
    return z_pair_dist


# z: N x S x C
# x: N x S x C x H x W


def compute_pairwise_divergence(recodes, codes):
    N, k, _ = codes.size()
    z_delta = compute_pair_distance(torch.squeeze(codes).view(N, k, -1))
    x_delta = compute_pair_distance(torch.squeeze(recodes).view(N, k, -1))
    div = F.relu(z_delta * 0.8 - x_delta)
    return div.sum()
