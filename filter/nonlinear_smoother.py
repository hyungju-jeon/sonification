import math
import torch
import einops
import random
import numpy as np
import torch.nn as nn
import lightning as lightning
import torch.nn.functional as Fn
import filter.prob_utils as prob_utils
import filter.linalg_utils as linalg_utils

from torch.jit.annotations import Tuple, List, Dict
from filter.linalg_utils import bmv, bip, chol_bmv_solve, bop, bqp


class FullRankNonlinearStateSpaceModel(nn.Module):
    def __init__(self, dynamics_mod, approximation_pdf, likelihood_pdf,
                 initial_c_pdf, ell_grad, nl_filter, device='cpu'):
        super(FullRankNonlinearStateSpaceModel, self).__init__()

        self.device = device
        self.nl_filter = nl_filter
        self.ell_grad = ell_grad
        self.dynamics_mod = dynamics_mod
        self.initial_c_pdf = initial_c_pdf
        self.likelihood_pdf = likelihood_pdf
        self.approximation_pdf = approximation_pdf

    @torch.jit.export
    def forward(self,
                y,
                n_samples: int,
                p_mask: float=0.0):

        z_s, stats = self.fast_smooth_1_to_T(y, n_samples, p_mask)
        kl = full_rank_mvn_kl(stats['m_f'], stats['P_f_chol'], stats['m_p'], stats['P_p_chol'])
        ell = self.likelihood_pdf.get_ell(y, z_s).mean(dim=0)

        loss = kl - ell
        loss = loss.sum(dim=-1).mean()
        return loss, z_s, stats

    def fast_filter_1_to_T(self,
                           y,
                           n_samples: int,
                           p_mask: float=0.0):

        k_y, K_y = self.ell_grad(y)
        z_s, stats = self.nl_filter(k_y, K_y, n_samples, p_mask=p_mask)
        return z_s, stats

    def fast_smooth_1_to_T(self,
                           y,
                           n_samples: int,
                           p_mask: float=0.0):

        h_y, K_y = self.ell_grad(y)
        h_b, K_b = self.encoder(h_y, K_y)
        K = torch.concat([K_b, K_y], dim=-1)
        k = h_b + h_y

        z_s, stats = self.nl_filter(k, K, n_samples, p_mask=p_mask)
        return z_s, stats


class FullRankNonlinearStateSpaceModelFilter(nn.Module):
    def __init__(self, dynamics_mod, approximation_pdf, likelihood_pdf, B,
                 initial_c_pdf, ell_grad, nl_filter, device='cpu'):
        super(FullRankNonlinearStateSpaceModelFilter, self).__init__()

        self.device = device

        self.B = B
        self.nl_filter = nl_filter
        self.ell_grad = ell_grad
        self.dynamics_mod = dynamics_mod
        self.initial_c_pdf = initial_c_pdf
        self.likelihood_pdf = likelihood_pdf
        self.approximation_pdf = approximation_pdf

    @torch.jit.export
    def forward(self,
                y,
                n_samples: int,
                u=None,
                p_mask: float=0.0):

        u_in = self.B(u)
        z_s, stats = self.fast_filter_1_to_T(y, u=u_in, n_samples=n_samples, p_mask=p_mask)
        kl = full_rank_mvn_kl(stats['m_f'], stats['P_f_chol'], stats['m_p'], stats['P_p_chol'])
        ell = self.likelihood_pdf.get_ell(y, z_s).mean(dim=0)

        loss = kl - ell
        loss = loss.sum(dim=-1).mean()
        return loss, z_s, stats

    def step_t(self,
               y_t,
               u_t,
               n_samples,
               z_tm1
               ):

        u_in = self.B(u_t)
        k_y, K_y = self.ell_grad(y_t[:, None, :])
        k_y, K_y = k_y.squeeze(1), K_y.squeeze(1)

        Q_diag = Fn.softplus(self.dynamics_mod.log_Q)
        m_fn_z_tm1 = self.dynamics_mod.mean_fn(z_tm1).movedim(0, -1) + u_in.unsqueeze(-1)
        z_f_t, m_p_t, m_f_t, P_f_chol_t, P_p_chol_t = filter_step_t(m_fn_z_tm1, k_y, K_y, Q_diag, False)

        stats = {}
        stats['m_p'] = m_p_t
        stats['m_f'] = m_f_t
        stats['P_f_chol'] = P_f_chol_t
        stats['P_p_chol'] = P_p_chol_t

        return stats, z_f_t

    def step_0(self,
               y_t,
               u_t,
               n_samples):

        u_in = self.B(u_t)
        k_y, K_y = self.ell_grad(y_t[:, None, :])
        k_y, K_y = k_y.squeeze(1), K_y.squeeze(1)

        Q_0_diag = Fn.softplus(self.initial_c_pdf.log_v0)
        m_0 = self.initial_c_pdf.m0

        m_fn_z_tm1 = torch.ones((y_t.shape[0], k_y.shape[-1], n_samples), device=y_t.device) * m_0[None, :, None]
        m_fn_z_tm1 += u_in.unsqueeze(-1)

        z_f_t, m_p_t, m_f_t, P_f_chol_t, P_p_chol_t = filter_step_t(m_fn_z_tm1, k_y, K_y, Q_0_diag, False)

        stats = {}
        stats['m_p'] = m_p_t
        stats['m_f'] = m_f_t
        stats['P_f_chol'] = P_f_chol_t
        stats['P_p_chol'] = P_p_chol_t

        return stats, z_f_t

    def fast_filter_1_to_T(self,
                           y,
                           u,
                           n_samples: int,
                           p_mask: float=0.0):

        k_y, K_y = self.ell_grad(y)
        z_s, stats = self.nl_filter(k_y, K_y, n_samples=n_samples, u=u, p_mask=p_mask)
        return z_s, stats



class NonlinearFilter(nn.Module):
    def __init__(self, dynamics_mod, initial_c_pdf, device):
        super(NonlinearFilter, self).__init__()

        self.device = device
        self.dynamics_mod = dynamics_mod
        self.initial_c_pdf = initial_c_pdf

    def forward(self,
                k: torch.Tensor,
                K: torch.Tensor,
                n_samples: int,
                p_mask: float=0.0,
                u=None):

        # mask data, 0: data available, 1: data missing
        n_trials, n_time_bins, n_latents, rank = K.shape
        Q_diag = Fn.softplus(self.dynamics_mod.log_Q)
        t_mask = torch.rand(n_time_bins) < p_mask

        z_f = []
        m_p = []
        m_f = []
        P_p_chol = []
        P_f_chol = []
        stats = {}

        for t in range(n_time_bins):
            if t == 0:
                m_0 = self.initial_c_pdf.m0
                P_0_diag = Fn.softplus(self.initial_c_pdf.log_v0)
                z_f_t, m_f_t, P_f_chol_t, P_p_chol_t = filter_step_0(m_0, k[:, 0], K[:, 0], P_0_diag, n_samples)
                m_p.append(m_0 * torch.ones(n_trials, n_latents, device=k[:, 0].device))
            else:
                if u is None:
                    m_fn_z_tm1 = self.dynamics_mod.mean_fn(z_f[t-1]).movedim(0, -1)
                else:
                    m_fn_z_tm1 = self.dynamics_mod.mean_fn(z_f[t - 1]).movedim(0, -1) + u[:, t].unsqueeze(-1)

                z_f_t, m_p_t, m_f_t, P_f_chol_t, P_p_chol_t = filter_step_t(m_fn_z_tm1, k[:, t], K[:, t], Q_diag, t_mask[t])
                m_p.append(m_p_t)

            z_f.append(z_f_t)
            m_f.append(m_f_t)
            P_f_chol.append(P_f_chol_t)
            P_p_chol.append(P_p_chol_t)

        z_f = torch.stack(z_f, dim=2)
        stats['m_f'] = torch.stack(m_f, dim=1)
        stats['m_p'] = torch.stack(m_p, dim=1)
        stats['P_f_chol'] = torch.stack(P_f_chol, dim=1)
        stats['P_p_chol'] = torch.stack(P_p_chol, dim=1)

        return z_f, stats

def fast_bmv_P_p(M_c_p, Q_diag, v):
    u_1 = bmv(M_c_p, bmv(M_c_p.mT, v))
    u_2 = Q_diag * v
    u = u_1 + u_2
    return u

def fast_bmv_P_p_inv(Q_diag, M_c_p, Psi_p, v):
    Q_inv_diag = 1 / Q_diag

    u_1 = Q_inv_diag * v
    u_2 = Q_inv_diag * bmv(M_c_p, bmv(Psi_p, bmv(Psi_p.mT, bmv(M_c_p.mT, u_1))))
    u = u_1 - u_2
    return u

def full_rank_mvn_kl(m_f, P_f_chol, m_p, P_p_chol):
    tr = torch.einsum('...ii -> ...', torch.cholesky_solve(P_f_chol @ P_f_chol.mT, P_p_chol))
    logdet1 = 2 * torch.sum(torch.log(torch.diagonal(P_f_chol, dim1=-2, dim2=-1)), dim=-1)
    logdet2 = 2 * torch.sum(torch.log(torch.diagonal(P_p_chol, dim1=-2, dim2=-1)), dim=-1)
    qp = bip(m_f - m_p, chol_bmv_solve(P_p_chol, m_f - m_p))
    kl = 0.5 * (tr + qp + logdet2 - logdet1 - m_f.shape[-1])

    return kl

def predict_step_t(m_theta_z_tm1, Q_diag):
    M = -0.5 * (torch.diag(Q_diag) + bop(m_theta_z_tm1, m_theta_z_tm1))

    m_p = m_theta_z_tm1.mean(dim=0)
    M_p = M.mean(dim=0)
    P_p = -2 * M_p - bop(m_p, m_p)
    return m_p, P_p

def filter_step_t(m_theta_z_tm1, k, K, Q_diag, t_mask):
    device = m_theta_z_tm1.device
    n_trials, n_latents, rank = K.shape
    n_samples = m_theta_z_tm1.shape[-1]
    batch_sz = [n_trials]

    w_f = torch.randn([n_samples] + batch_sz + [n_latents], device=device)
    m_p, P_p = predict_step_t(m_theta_z_tm1.movedim(-1, 0), Q_diag)
    P_p_chol = torch.linalg.cholesky(P_p)

    if not t_mask:
        h_p = chol_bmv_solve(P_p_chol, m_p)
        h_f = h_p + k

        J_p = torch.cholesky_inverse(P_p_chol)
        J_f = J_p + K @ K.mT
        J_f_chol = torch.linalg.cholesky(J_f)
        P_f_chol = linalg_utils.triangular_inverse(J_f_chol).mT
        m_f = chol_bmv_solve(J_f_chol, h_f)
    else:
        m_f = m_p
        P_f_chol = P_p_chol

    z_f = m_f + bmv(P_f_chol, w_f)

    return z_f, m_p, m_f, P_f_chol, P_p_chol

def filter_step_0(m_0: torch.Tensor, k: torch.Tensor, K: torch.Tensor, P_0_diag: torch.Tensor, n_samples: int):
    n_trials, n_latents, rank = K.shape
    batch_sz = [n_trials]

    J_0_diag = 1 / P_0_diag
    h_0 = J_0_diag * m_0
    J_f = torch.diag(J_0_diag) + K @ K.mT
    J_f_chol = torch.linalg.cholesky(J_f)
    P_f_chol = linalg_utils.triangular_inverse(J_f_chol).mT

    h_f = h_0 + k
    m_f = chol_bmv_solve(J_f_chol, h_f)

    P_p_chol = torch.diag(torch.sqrt(P_0_diag)) + torch.zeros_like(P_f_chol, device=m_0.device)
    w_f = torch.randn([n_samples] + batch_sz + [n_latents]).to(m_0.device)
    z_f = m_f + bmv(P_f_chol, w_f)

    return z_f, m_f, P_f_chol, P_p_chol


