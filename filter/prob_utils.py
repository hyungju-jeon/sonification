import math
import torch
import einops
import itertools
import numpy as np

from einops import rearrange
from scipy.ndimage import gaussian_filter1d
from torch.nn.functional import poisson_nll_loss
from sklearn.decomposition import FactorAnalysis

from filter.linalg_utils import bip, bop, bmv, bqp, chol_bmv_solve


def kl_diagonal_gaussian(nat, nat_p, param_type='nat'):
    n_latents = nat_p.shape[-1] // 2
    h, J = nat[..., :n_latents], nat[..., n_latents:]
    h_p, J_p = nat_p[..., :n_latents], nat_p[..., n_latents:]

    P = 1 / J
    m = P * h

    P_p = 1 / J_p
    m_p = P_p * h_p

    kl = 0.5 * torch.log(P_p / P) + 0.5 * (P + (m - m_p) ** 2) / P_p - 0.5
    kl = kl.sum(dim=-1)
    return kl


def kl_dense_gaussian(nat_s, nat_p, param_type='nat'):
    # 0.5 * [(m2 - m1) J2 (m2 - m1) + tr(J2 J1^{-1}) + log |J1| - log |J2|]
    n_nat_params = nat_s.shape[-1]
    n_latents = int(-1 + math.sqrt(1 + 4 * n_nat_params)) // 2
    h1, J1_vec = nat_s[..., :n_latents], nat_s[..., n_latents:]
    h2, J2_vec = nat_p[..., :n_latents], nat_p[..., n_latents:]
    J1 = einops.rearrange(J1_vec, '... (i j) -> ... i j', i=n_latents)
    J2 = einops.rearrange(J2_vec, '... (i j) -> ... i j', i=n_latents)

    # TODO
    # J1_chol, _ = torch.linalg.cholesky_ex(J1)
    # J2_chol, _ = torch.linalg.cholesky_ex(J2)
    try:
        J1_chol = torch.linalg.cholesky(J1)
        J2_chol = torch.linalg.cholesky(J2)
    except:
        U, S, V = torch.linalg.svd(J1)
        J1_chol = torch.linalg.cholesky((U*(S+1e-2).unsqueeze(-1)) @ U.mT)
        U, S, V = torch.linalg.svd(J2)
        J2_chol = torch.linalg.cholesky((U*(S+1e-2).unsqueeze(-1)) @ U.mT)

    m1 = torch.cholesky_solve(h1.unsqueeze(-1), J1_chol).squeeze(-1)
    m2 = torch.cholesky_solve(h2.unsqueeze(-1), J2_chol).squeeze(-1)

    qp = torch.einsum('...i, ...ij, ...j -> ...', m2 - m1, J2, m2 - m1)
    tr = torch.einsum('...ii -> ...', torch.cholesky_solve(J2.mT, J1_chol).mT)
    logdet1 = 2 * torch.sum(torch.log(torch.diagonal(J1_chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)
    logdet2 = 2 * torch.sum(torch.log(torch.diagonal(J2_chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)

    kl = 0.5 * (qp + tr + logdet1 - logdet2 - h1.shape[-1])
    return kl

def kl_dense_gaussian_canon(m_s, P_s, m_p, P_p_chol):
    m_diff = m_s - m_p
    qp = bip(m_diff, torch.cholesky_solve(m_diff.unsqueeze(-1), P_p_chol).squeeze(-1))
    tr = torch.einsum('...ii -> ...', torch.cholesky_solve(P_s, P_p_chol))

    P_s_chol = torch.linalg.cholesky(P_s)
    logdet1 = -2 * torch.sum(torch.log(torch.diagonal(P_s_chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)
    logdet2 = -2 * torch.sum(torch.log(torch.diagonal(P_p_chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)

    kl = 0.5 * (qp + tr + logdet1 - logdet2 - m_s.shape[-1])
    return kl

def fisher_dense_gaussian(nat_params):
    n_nat_params = nat_params.shape[-1]
    n_latents = int(-1 + math.sqrt(1 + 4 * n_nat_params)) // 2

    h, J_vec = nat_params[..., :n_latents], nat_params[..., n_latents:]
    J = einops.rearrange(J_vec, '... (i j) -> ... i j', i=n_latents)
    J_chol = torch.linalg.cholesky(J)

    m = torch.cholesky_solve(h.unsqueeze(-1), J_chol).squeeze(-1)
    P = torch.cholesky_inverse(J_chol)

    eye = torch.ones(list(m.shape)[:-1] + [n_latents, n_latents]) * torch.eye(n_latents)

    if nat_params.dim() == 3:
        m_kron_eye = torch.vmap(torch.vmap(torch.kron))(m.unsqueeze(-2), eye)
        P_kron_P = torch.vmap(torch.vmap(torch.kron))(P, P)
    elif nat_params.dim() == 2:
        m_kron_eye = torch.vmap(torch.kron)(m.unsqueeze(-2), eye)
        P_kron_P = torch.vmap(torch.kron)(P, P)
    elif nat_params.dim() == 1:
        m_kron_eye = torch.kron(m.unsqueeze(-2), eye)
        P_kron_P = torch.kron(P, P)

    ul_analytic = P
    ur_analytic = -P @ m_kron_eye
    lr_analytic = 0.5 * P_kron_P + m_kron_eye.mT @ P @ m_kron_eye

    fisher_shape = list(m.shape)[:-1] + [n_nat_params, n_nat_params]
    block_fisher = torch.zeros(fisher_shape)
    block_fisher[..., :n_latents, :n_latents] = ul_analytic
    block_fisher[..., :n_latents, n_latents:] = ur_analytic
    block_fisher[..., n_latents:, :n_latents] = ur_analytic.mT
    block_fisher[..., n_latents:, n_latents:] = lr_analytic

    return block_fisher


def fisher_inv_dense_gaussian(nat_params):
    n_nat_params = nat_params.shape[-1]
    n_latents = int(-1 + math.sqrt(1 + 4 * n_nat_params)) // 2

    h, J_vec = nat_params[..., :n_latents], nat_params[..., n_latents:]
    J = einops.rearrange(J_vec, '... (i j) -> ... i j', i=n_latents)
    J_chol = torch.linalg.cholesky(J)

    m = torch.cholesky_solve(h.unsqueeze(-1), J_chol).squeeze(-1)

    if nat_params.dim() == 3:
        h_kron_J = torch.vmap(torch.vmap(torch.kron))(h.unsqueeze(-2), J.contiguous())
        J_kron_J = torch.vmap(torch.vmap(torch.kron))(J, J)
    elif nat_params.dim() == 2:
        h_kron_J = torch.vmap(torch.kron)(h.unsqueeze(-2), J.contiguous())
        J_kron_J = torch.vmap(torch.kron)(J, J)
    elif nat_params.dim() == 1:
        h_kron_J = torch.kron(h.unsqueeze(-2), J.contiguous())
        J_kron_J = torch.kron(J, J)

    ul_analytic = (1 + 2 * bip(m, h)).unsqueeze(-1).unsqueeze(-1) * J
    ur_analytic = 2 * h_kron_J
    lr_analytic = 2 * J_kron_J

    fisher_inv_shape = list(m.shape)[:-1] + [n_nat_params, n_nat_params]
    block_fisher_inv = torch.zeros(fisher_inv_shape)
    block_fisher_inv[..., :n_latents, :n_latents] = ul_analytic
    block_fisher_inv[..., :n_latents, n_latents:] = ur_analytic
    block_fisher_inv[..., n_latents:, :n_latents] = ur_analytic.mT
    block_fisher_inv[..., n_latents:, n_latents:] = lr_analytic

    return block_fisher_inv


def gaussian_nat_to_m_p(nat_params):
    n_nat_params = nat_params.shape[-1]
    n_latents = int(-1 + math.sqrt(1 + 4 * n_nat_params)) // 2

    h = nat_params[..., :n_latents]
    J_vec = nat_params[..., n_latents:]
    batch_dims = J_vec.size()[:-1]
    J = J_vec.reshape(*batch_dims, n_latents, n_latents)

    J_chol = torch.linalg.cholesky(J)
    m = torch.cholesky_solve(h.unsqueeze(-1), J_chol).squeeze(-1)
    P = torch.cholesky_inverse(J_chol)

    return m, P


def gaussian_m_p_to_nat(m, P):
    P_chol = torch.linalg.cholesky(P)
    J = torch.cholesky_inverse(P_chol)
    h = torch.cholesky_solve(m.unsqueeze(-1), P_chol).squeeze(-1)
    J_vec = einops.rearrange(J, '... i j -> ...(i j)')
    nat_params = torch.cat([h, J_vec], dim=-1)

    return nat_params


def estimate_readout_matrix_fa(Y, n_latents, smoothing_sigma=2, y_gaussian=False):
    '''
    Y in (B x T x N) -> Y_smooth in (T x N) after Gaussian smoothing and trial averaging
    Y_smooth_avg \approx PCA_transform(Y_smooth_avg) @ C.T where C in (N x D)

    :param Y:
    :param smoothing_sigma:
    :return:
    '''

    if not y_gaussian:
        Y_smooth = gaussian_filter1d(Y.cpu().data.numpy(), sigma=smoothing_sigma, axis=0)
        Y_smooth_avg = rearrange(Y_smooth, 'b t n -> (b t) n')
    else:
        Y_smooth_avg = rearrange(Y.cpu().data.numpy(), 'b t n -> (b t) n')

    fa = FactorAnalysis(n_components=n_latents, svd_method='lapack')
    fa.fit(Y_smooth_avg)

    C_hat = fa.components_.T
    C_hat = torch.tensor(C_hat, dtype=Y.dtype)

    b_hat = fa.mean_
    b_hat = torch.tensor(b_hat, dtype=Y.dtype)

    R_hat = fa.noise_variance_
    R_hat = torch.tensor(R_hat, dtype=Y.dtype)

    return C_hat, b_hat, R_hat


def estimate_poisson_rate_bias(y, time_delta):
    if isinstance(y, torch.Tensor):
        bias_hat = torch.log(torch.mean(y, dim=[0, 1]) / time_delta + 1e-12)

    elif isinstance(y, torch.utils.data.DataLoader):
        full_batch_mean = torch.zeros(y.shape[-1], device=y.device)

        for y_mb in y:
            full_batch_mean += torch.mean(y_mb, dim=[0, 1])

        full_batch_mean /= len(y)
        bias_hat = torch.log(full_batch_mean / time_delta + 1e-12)
    else:
        raise TypeError('pass in tensor or dataloader')

    return bias_hat


def kalman_filter(y, C, b, R_diag, F, Q_diag, m_0, Q_0_diag):
    device = C.device
    n_latents = C.shape[1]
    n_trials, n_time_bins, n_neurons = y.shape

    R_inv_diag = 1 / R_diag
    Q_0 = torch.diag(Q_0_diag)
    Q = torch.diag(Q_diag)

    m_p = []
    m_f = []
    P_f = []
    P_p = []

    for t in range(n_time_bins):
        if t == 0:
            m_p.append(m_0 * torch.ones([n_trials, n_latents], device=device))
            P_p.append(Q_0 * torch.ones([n_trials, n_latents, n_latents], device=device))
        else:
            m_p.append(bmv(F, m_f[t-1]))
            P_p.append(F @ P_f[t-1] @ F.T + Q)

        P_p_chol = torch.linalg.cholesky(P_p[t])
        h_p = torch.cholesky_solve(m_p[t].unsqueeze(-1), P_p_chol).squeeze(-1)
        h_f_t = h_p + bmv(C.mT, R_inv_diag * (y[:, t] - b))
        J_f_t = torch.cholesky_inverse(P_p_chol) + (C.mT * R_inv_diag) @ C
        J_f_t_chol = torch.linalg.cholesky(J_f_t)
        m_f_t = torch.cholesky_solve(h_f_t.unsqueeze(-1), J_f_t_chol).squeeze(-1)
        P_f_t = torch.cholesky_inverse(J_f_t_chol)

        m_f.append(m_f_t)
        P_f.append(P_f_t)

    m_f = torch.stack(m_f, dim=1)
    P_f = torch.stack(P_f, dim=1)
    m_p = torch.stack(m_p, dim=1)
    P_p = torch.stack(P_p, dim=1)

    return m_f, P_f, m_p, P_p


def rts_smoother(m_p, P_p, m_f, P_f, F):
    device = m_p.device
    n_trials, n_time_bins, n_latents = m_p.shape

    m_s = [None] * n_time_bins
    P_s = [None] * n_time_bins

    m_s[-1] = m_f[:, -1]
    P_s[-1] = P_f[:, -1]

    for t in range(n_time_bins - 2, -1, -1):
        P_p_chol = torch.linalg.cholesky(P_p[:, t+1])
        G = P_f[:, t] @ torch.cholesky_solve(F, P_p_chol).mT

        m_s[t] = m_f[:, t] + bmv(G, m_s[t+1] - m_p[:, t+1])
        P_s[t] = P_f[:, t] + G @ (P_s[t+1] - P_p[:, t+1]) @ G.mT

    m_s = torch.stack(m_s, dim=1)
    P_s = torch.stack(P_s, dim=1)

    return m_s, P_s


def lgssm_log_p_y_filtering(y, m_p, P_p, C, b, R_diag):
    n_trials, n_time_bins, n_neurons = y.shape

    diff = y - bmv(C, m_p) - b
    P_y = C @ P_p @ C.T + torch.diag(R_diag)
    P_y_chol = torch.linalg.cholesky(P_y)

    qp = bip(diff, chol_bmv_solve(P_y_chol, diff))
    logdet = 2 * torch.sum(torch.log(torch.diagonal(P_y_chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)
    const = n_neurons * math.log(2 * math.pi)

    log_p_y = -0.5 * (qp + logdet + const)
    log_p_y = log_p_y.sum(dim=-1)
    return log_p_y


def lgssm_log_p_y_smoothing(y, m, P, F, Q_diag, C, b, R_diag, m_0, Q_0_diag):
    n_trials, n_time_bins, n_neurons = y.shape
    log_p_y = 0.

    R = torch.diag(R_diag)
    Q = torch.diag(Q_diag)
    Q_0 = torch.diag(Q_0_diag)

    for t in range(n_time_bins):
        if t == 0:
            mvn_pri = torch.distributions.MultivariateNormal(m_0, Q_0)
        else:
            mvn_pri = torch.distributions.MultivariateNormal(bmv(F, m[:, t-1]), Q)

        mvn_ptr = torch.distributions.MultivariateNormal(m[:, t], P[:, t])
        mvn_lik = torch.distributions.MultivariateNormal(bmv(C, m[:, t]) + b, R)

        log_ptr = mvn_ptr.log_prob(m[:, t])
        log_pri = mvn_pri.log_prob(m[:, t])
        log_lik = mvn_lik.log_prob(y[:, t])

        log_p_y += (log_pri + log_lik - log_ptr)

    return log_p_y


def linear_gaussian_ell(y, C, b, R_diag, m, P):
    R_inv_diag = 1 / R_diag
    diff = y - bmv(C, m) - b

    qp = bip(diff, R_inv_diag * diff)
    logdet = torch.sum(torch.log(R_diag))
    tr = torch.einsum('...ii -> ...', (C.mT * R_inv_diag) @ C @ P)
    const = y.shape[-1] * math.log(2 * math.pi)

    ell = -0.5 * (qp + tr + logdet + const)

    return ell


def gaussian_log_prob(y, m, P):
    P_chol = torch.linalg.cholesky(P)
    diff = y - m

    qp = bip(diff, chol_bmv_solve(P_chol, diff))
    logdet = 2 * torch.sum(torch.log(torch.diagonal(P_chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)
    const = y.shape[-1] * math.log(2 * math.pi)

    log_prob = -0.5 * (qp + logdet + const)

    return log_prob


# source: https://github.com/arsedler9/lfads-torch
def bits_per_spike(preds, targets):
    """
    Computes BPS for n_samples x n_timesteps x n_neurons arrays.
    Preds are logrates and targets are binned spike counts.
    """
    nll_model = poisson_nll_loss(preds, targets, full=True, reduction="sum")
    nll_null = poisson_nll_loss(
        torch.mean(targets, dim=(0, 1), keepdim=True),
        targets,
        log_input=False,
        full=True,
        reduction="sum",
    )
    return (nll_null - nll_model) / torch.nansum(targets) / math.log(2)


# source: https://github.com/arsedler9/lfads-torch
def r2_score(preds, targets):
    if preds.ndim > 2:
        preds = preds.reshape(-1, preds.shape[-1])
    if targets.ndim > 2:
        targets = targets.reshape(-1, targets.shape[-1])
    target_mean = torch.mean(targets, dim=0)
    ss_tot = torch.sum((targets - target_mean) ** 2, dim=0)
    ss_res = torch.sum((targets - preds) ** 2, dim=0)
    return torch.mean(1 - ss_res / ss_tot)