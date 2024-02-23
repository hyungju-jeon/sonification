import einops
import torch
import torch.nn as nn
import filter.utils as utils
import torch.nn.functional as Fn

from functools import partial
from filter.linalg_utils import bmv, bop


class LocalEncoderLRMvn(nn.Module):
    def __init__(self, enc_size, hidden_size, n_latents, rank, likelihood_pdf, device='cpu', dropout=0.0):
        super(LocalEncoderLRMvn, self).__init__()
        self.device = device

        self.rank = rank
        self.enc_size = enc_size
        self.n_latents = n_latents
        self.likelihood_pdf = likelihood_pdf

        self.mlp = nn.Sequential(nn.Linear(enc_size, hidden_size, device=device),
                                 nn.SiLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_size, (rank + 1) * n_latents, device=device)).to(device)


    def forward(self, y):
        h_log_J = self.mlp(y)
        h = h_log_J[..., :self.n_latents]
        L_vec = h_log_J[..., self.n_latents:]
        L = L_vec.view(y.shape[0], y.shape[1], self.n_latents, -1)

        return h, L

