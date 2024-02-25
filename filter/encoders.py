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

class BackwardEncoderLRMvn(nn.Module):
    def __init__(self, n_neurons_enc, hidden_size, n_latents, rank_y, rank_b, network_type='rnn', device='cpu', dropout=0.1):
        super(BackwardEncoderLRMvn, self).__init__()
        self.device = device
        self.network_type = network_type
        self.dropout = dropout

        self.rank_y = rank_y
        self.rank_b = rank_b
        self.n_latents = n_latents

        # self.rnn = torch.nn.GRU(input_size=n_latents + (rank_y * n_latents), hidden_size=hidden_size,
        #                         batch_first=True, bidirectional=False, device=device)
        self.rnn = torch.nn.GRU(input_size=2*n_latents, hidden_size=hidden_size,
                                batch_first=True, bidirectional=False, device=device)


        self.projection = torch.nn.Sequential(
                                              # nn.SiLU(),
                                              # nn.Dropout(dropout),
                                              torch.nn.Linear(hidden_size, (rank_b + 1) * n_latents, device=device),
                                              # torch.nn.Identity(),
                                             )
        # self.projection = torch.nn.Linear(hidden_size, (rank_b + 1) * n_latents, device=device)

    def forward(self, h_y, L_y):
        L_y_vec = torch.sum(L_y * L_y, dim=-1)
        nat_y_hat = torch.concat([h_y, L_y_vec], dim=-1)
        w_flip, _ = self.rnn(nat_y_hat)

        w = torch.flip(w_flip, dims=[1])
        h_log_J = self.projection(w)

        h = h_log_J[..., :self.n_latents]
        L_vec = h_log_J[..., self.n_latents:]
        L = L_vec.view(h_y.shape[0], h_y.shape[1], self.n_latents, self.rank_b)

        h_out = torch.concat([h[:, 1:], h[:, -1:] * 0.], dim=1)
        L_out = torch.concat([L[:, 1:], L[:, -1:] * 0.], dim=1)
        return h_out, L_out
