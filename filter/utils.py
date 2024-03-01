import os
import math
import torch
import scipy
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as Fn
import quantities as pq

from torch.distributions import Bernoulli
from filter.linalg_utils import bmv, bip, bop
from sklearn.metrics import r2_score
from tqdm import tqdm
from einops import rearrange
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import FactorAnalysis


class SeqDataLoader:
    def __init__(self, data_tuple, batch_size):
        """
        Constructor for fast data loader
        :param data_tuple: a tuple of matrices, where element i is an (trial x time x features) vector
        :param batch_size: batch size
        """
        self.data_tuple = data_tuple

        self.batch_size = batch_size
        self.dataset_len = self.data_tuple[0].shape[0]

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        r = torch.randperm(self.dataset_len)
        self.indices = [r[j * self.batch_size: (j * self.batch_size) + self.batch_size] for j in range(self.n_batches)]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n_batches:
            raise StopIteration
        idx = self.indices[self.i]
        batch = tuple([x[idx, :, :] for x in self.data_tuple])
        self.i += 1
        return batch

    def __len__(self):
        return self.n_batches


class MaskedLinear(torch.nn.Module):
    def __init__(self, dim_in, dim_out, device='cpu', p_mask=0.):
        super(MaskedLinear, self).__init__()
        self.linear = torch.nn.Linear(dim_in, dim_out, device=device)

        self.device = device
        self.p_mask = p_mask
        self.dim_out = dim_out
        self.dim_in = dim_in

    def forward(self, x):
        if self.training:
            if self.p_mask > 0.:
                scale_f = 1 / (1 - self.p_mask)
                mask = torch.rand(self.dim_out, self.dim_in, device=self.device) < self.p_mask
                y = nn.functional.linear(x, self.linear.weight * ~mask, self.linear.bias)
                return scale_f * y
            else:
                return self.linear(x)
        else:
            return self.linear(x)


class DynamicsGRU(torch.nn.Module):
    def __init__(self, hidden_dim, latent_dim, device, dropout=0.0):
        super(DynamicsGRU, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.gru_cell = nn.GRUCell(0, hidden_dim, device=device).to(device)
        self.h_to_z = nn.Linear(hidden_dim, latent_dim, device=device).to(device)
        self.z_to_h = nn.Linear(latent_dim, hidden_dim, device=device).to(device)
        # self.h_to_z = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, latent_dim, device=device).to(device))
        # self.z_to_h = nn.Sequential(nn.Linear(latent_dim, hidden_dim, device=device).to(device), nn.SiLU(), nn.Dropout(dropout))


    def forward(self, z):
        h_in = self.z_to_h(z)
        h_in_shape = list(h_in.shape)[:-1]
        h_in = h_in.reshape((-1, self.hidden_dim))

        empty_vec = torch.empty((h_in.shape[0], 0), device=z.device)
        h_out = self.gru_cell(empty_vec, h_in)
        h_out = h_out.reshape(h_in_shape + [self.hidden_dim])
        z_out = self.h_to_z(h_out)
        return z_out


class DynamicsLocallyLinear(torch.nn.Module):
    def __init__(self, hidden_dim, latent_dim, device):
        super(DynamicsLocallyLinear, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.gru_cell = nn.GRUCell(0, hidden_dim, device=device).to(device)
        self.z_to_h = nn.Linear(latent_dim, hidden_dim, device=device).to(device)
        self.h_to_z = nn.Linear(hidden_dim, latent_dim**2 + latent_dim, device=device).to(device)

        self.A = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, z, get_A=False):
        h_in = self.z_to_h(z)
        h_in_shape = list(h_in.shape)[:-1]
        h_in = h_in.reshape((-1, self.hidden_dim))

        empty_vec = torch.empty((h_in.shape[0], 0), device=self.device)
        h_out = self.gru_cell(empty_vec, h_in)
        h_out = h_out.reshape(h_in_shape + [self.hidden_dim])
        A_flat_out = self.h_to_z(h_out)
        s_out = Fn.sigmoid(A_flat_out[..., :self.latent_dim])
        A_out = einops.rearrange(A_flat_out[..., self.latent_dim:], '... (i j) -> ... i j', i=self.latent_dim)
        U, S, Vh = torch.linalg.svd(A_out)
        A_hat_out = U @ torch.diag_embed(s_out) @ Vh
        z_out = bmv(A_hat_out, z)

        if get_A:
            return self.A(z), self.A.weight * torch.ones(list(z.shape)+[z.shape[-1]])
        else:
            return self.A(z)


class ResNet(torch.nn.Module):
    def __init__(self, module, device):
        super().__init__()
        self.module = module
        self.device = device

    def forward(self, z):
        return z + self.module(z)


class LinearResNet(torch.nn.Module):
    def __init__(self, mlp, linear):
        super().__init__()
        self.mlp = mlp
        self.linear = linear

    def forward(self, z):
        return self.mlp(z) + self.linear(z)


class ReadoutLatentMask(torch.nn.Module):
    def __init__(self, n_latents, n_latents_read, device='cpu'):
        super().__init__()

        self.device = device
        self.n_latents = n_latents
        self.n_latents_read = n_latents_read

    def forward(self, z):
        return z[..., :self.n_latents_read]

    def get_matrix_repr(self):
        H = torch.zeros((self.n_latents_read, self.n_latents), device=self.device)
        H[torch.arange(self.n_latents_read), torch.arange(self.n_latents_read)] = 1.0
        return H


def gaussian_m_p_to_nat(m, P):
    P_chol = torch.linalg.cholesky(P)
    J = torch.cholesky_inverse(P_chol)
    h = torch.cholesky_solve(m.unsqueeze(-1), P_chol).squeeze(-1)
    J_vec = einops.rearrange(J, '... i j -> ...(i j)')
    nat_params = torch.cat([h, J_vec], dim=-1)

    return nat_params


def build_gru_dynamics_function(dim_input, dim_hidden, d_type=torch.float32, device='cpu', dropout=0.0):
    gru_dynamics = DynamicsGRU(dim_hidden, dim_input, device, dropout=dropout)
    return gru_dynamics


def build_resnet_dynamics_function(dim_input, dim_hidden, device):
    mlp = torch.nn.Sequential(nn.Linear(dim_input, dim_hidden, bias=False, device=device),
                              nn.Sigmoid(),
                              nn.Linear(dim_hidden, dim_input, bias=False, device=device))
    resnet = ResNet(mlp, device)
    return resnet


def build_locally_linear_dynamics_function(dim_input, dim_hidden, d_type=torch.float32, device='cpu'):
    gru_dynamics = DynamicsLocallyLinear(dim_hidden, dim_input, device)
    return gru_dynamics


def softplus_inv(x):
    return torch.log(torch.exp(x) - 1 + 1e-10)


def build_mlp_function(dim_input, dim_hidden_layers, dim_output, nonlinearity_fn, d_type=torch.float64, device='cpu'):
    nn_modules = []
    device = torch.device(device)

    for dx, hidden_layer_dim in enumerate(dim_hidden_layers):
        if dx == 0:
            dim_in = dim_input
            dim_out = hidden_layer_dim
        else:
            dim_in = dim_hidden_layers[dx - 1]
            dim_out = hidden_layer_dim

        nn_modules.append(
            torch.nn.Sequential(
                torch.nn.Linear(dim_in, dim_out, dtype=d_type).to(device),
                nonlinearity_fn()
            )
        )

    if len(dim_hidden_layers) > 0:
        nn_modules.append(torch.nn.Linear(dim_hidden_layers[-1], dim_output, dtype=d_type).to(device))
    else:
        nn_modules.append(torch.nn.Linear(dim_input, dim_output, dtype=d_type).to(device))

    mlp = torch.nn.Sequential(*nn_modules)
    mlp.apply(init_mlp_weights)
    return mlp


def build_resnet_mlp_function(dim_input, dim_hidden_layers, dim_output, nonlinearity_fn, d_type=torch.float64, device='cpu'):
    mlp = build_mlp_function(dim_input, dim_hidden_layers, dim_output, nonlinearity_fn, d_type, device)
    resnet_mlp = ResNet(mlp, device)

    return resnet_mlp


def build_masked_mlp_function(dim_input, dim_input_readout, dim_hidden_layers, dim_output, nonlinearity_fn, last_layer_fn,
                              d_type=torch.float32, device='cpu'):
    mlp = build_mlp_function(dim_input_readout, dim_hidden_layers, dim_output, nonlinearity_fn, d_type, device)
    mlp.apply(init_mlp_weights)

    H = torch.nn.Linear(dim_input, dim_input_readout, bias=False)
    H.weight.data = torch.zeros(H.weight.shape, dtype=d_type).to(device)
    H.weight.data[range(dim_input_readout), range(dim_input_readout)] = 1.
    H.requires_grad_(False)

    return torch.nn.Sequential(H, mlp, last_layer_fn())


def init_mlp_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        # torch.nn.init.orthogonal_(m.weight)
        # torch.nn.init.sparse_(m.weight, sparsity=0.5)
        try:
            m.bias.data.fill_(0.0)
        except:
            pass


def get_r2(y_true, x_hat, n_samples):
    """residuals train inference"""
    y_true_rpt = einops.repeat(y_true, 'b t -> s b t', s=n_samples).unsqueeze(-1)
    y_true_rpt_flat = einops.rearrange(y_true_rpt, 's b t d -> (s b t) d')
    y_true_rpt_flat = (y_true_rpt_flat - y_true_rpt_flat.mean()) / y_true_rpt_flat.std()

    x_hat_flat = einops.rearrange(x_hat, 's b t d -> (s b t) d')
    x_hat_flat = (x_hat_flat - x_hat_flat.mean(dim=0)) / x_hat_flat.std(dim=0)

    lstsq_result = torch.linalg.lstsq(x_hat_flat, y_true_rpt_flat)
    y_hat_flat = bmv(lstsq_result.solution.mT, x_hat_flat)
    r2 = r2_score(y_true_rpt_flat.detach().numpy(), y_hat_flat.detach().numpy())

    return r2


def get_ols_n_samples_per_trial(y, x):
    n_samples, n_trials, n_time_bins, n_latents = x.shape
    x_stacked = einops.rearrange(x, 's b t l -> (s b t) l')
    y_rpt = einops.repeat(y, 'b t n -> (s b t) n', s=n_samples)

    qp = x_stacked.T @ x_stacked
    qp_chol = torch.linalg.cholesky(qp)
    w = torch.cholesky_inverse(qp_chol) @ x_stacked.T @ y_rpt

    y_hat = x_stacked @ w
    y_hat_out = einops.rearrange(y_hat, '(s b t) n -> s b t n', s=n_samples, b=n_trials, t=n_time_bins)
    r2 = r2_score(y_rpt.reshape(-1).detach().numpy(), y_hat.reshape(-1).detach().numpy())

    return y_hat_out, r2, w


def propagate_latent_k_steps(z, dynamics_mod, k_steps):
    n_samples, n_trials, n_latents = z.shape
    Q = Fn.softplus(dynamics_mod.log_Q)
    mean_fn = dynamics_mod.mean_fn

    z_out = torch.zeros((n_samples, n_trials, k_steps + 1, n_latents), dtype=z.dtype).to(z.device)
    z_out[:, :, 0] = z

    for k in range(1, k_steps+1):
        z_out[:, :, k] = mean_fn(z_out[:, :, k - 1]) \
                         + torch.sqrt(Q) * torch.randn_like(z_out[:, :, k - 1]).to(z.device)

    return z_out


# def gpfa_array_to_spiketrains(array, bin_size):
#     """Convert B x T x N spiking array to list of list of SpikeTrains for GPFA"""
#     stList = []
#     for trial in range(len(array)):
#         trialList = []
#         for channel in range(array.shape[2]):
#             times = np.nonzero(array[trial, :, channel])[0]
#             counts = array[trial, times, channel].astype(int)
#             times = np.repeat(times, counts)
#             st = neo.SpikeTrain(times*bin_size*pq.ms, t_stop=array.shape[1]*bin_size*pq.ms)
#             trialList.append(st)
#         stList.append(trialList)
#     return stList


def get_linearized_eig(mean_fn, m):
    if m.dim() == 2:
        A = torch.vmap(torch.vmap(torch.func.jacfwd(mean_fn)))(m.unsqueeze(0)).squeeze(0)
    elif m.dim() == 3:
        A = torch.vmap(torch.vmap(torch.vmap(torch.func.jacfwd(mean_fn))))(m.unsqueeze(0)).squeeze(0)

    v = mean_fn(m) - m
    P = torch.eye(m.shape[-1]) - bop(v, v) / bip(v, v).unsqueeze(-1).unsqueeze(-1)
    F = P @ A @ P

    eigvals = torch.linalg.eigvals(F)
    return eigvals


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matrix_index_select(A, indices):
    # useful for batching over index selection
    return A[indices]


def sample_lds(u, A, B, C, b, Q_diag, R_diag, m_0, Q_0_diag, device='cpu'):
    # input dim: (trial x time x n_inputs)
    # z_t = A @ z_{t-1} + B @ u_t + v_t, v_t ~ N(0, Q)
    # y_t = C @ z_t + b + w_t, w_t ~ N(0, R)
    n_trials, n_time_bins, n_inputs = u.shape
    n_neurons, n_latents = C.shape

    y = torch.zeros((n_trials, n_time_bins, n_neurons), device=device)
    z = torch.zeros((n_trials, n_time_bins, n_latents), device=device)

    for t in range(n_time_bins):
        if t == 0:
            z[:, 0] = m_0 + torch.sqrt(Q_0_diag) * torch.randn_like(z[:, 0]) + bmv(B, u[:, 0])
        else:
            z[:, t] = bmv(A, z[:, t-1]) + torch.sqrt(Q_diag) * torch.randn_like(z[:, t-1]) + bmv(B, u[:, t])

        y[:, t] = bmv(C, z[:, t]) + b + torch.sqrt(R_diag) * torch.randn_like(y[:, t])

    return y, z


def make_2d_rotation_matrix(theta, device='cpu'):
    A = torch.tensor([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]], device=device)

    return A
