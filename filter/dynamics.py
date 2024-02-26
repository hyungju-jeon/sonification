import torch
import einops
import torch.nn as nn
import filter.utils as utils
import torch.nn.functional as Fn

from filter.linalg_utils import bip, bop, bmv


class DenseGaussianNonlinearDynamics(nn.Module):
    def __init__(self, mean_fn, n_latents, pdf, Q=None, device='cpu'):
        super(DenseGaussianNonlinearDynamics, self).__init__()
        self.device = device

        self.pdf = pdf
        self.mean_fn = mean_fn
        self.n_latents = n_latents
        self.n_nat_params = n_latents + n_latents**2

        if Q is None:
            self.log_Q = torch.nn.Parameter(utils.softplus_inv(torch.ones(n_latents).to(device)))
        else:
            self.log_Q = torch.nn.Parameter(utils.softplus_inv(Q))

    def sample_expected_mean_params(self, nat_params, n_samples):
        z_s = self.pdf.sample(nat_params, n_samples)
        mean_params_out = self.samples_to_mean_params(z_s)
        mean_params_out = mean_params_out.mean(dim=0)

        return mean_params_out

    def sample_expected_mean_params_from_mean_params(self, mean_params, n_samples):
        z_s = self.pdf.sample_from_mean_params(mean_params, n_samples)
        mean_params_out = self.samples_to_mean_params(z_s)
        mean_params_out = mean_params_out.mean(dim=0)

        return mean_params_out

    def samples_to_mean_params(self, z):
        m_out = self.mean_fn(z)
        P_out_diag_vec = Fn.softplus(self.log_Q) * torch.ones_like(m_out)
        P_out = torch.diag_embed(P_out_diag_vec)
        M_out = -0.5 * (P_out + bop(m_out, m_out))
        mean_params_out = self.pdf.pack_params(m_out, M_out)

        return mean_params_out

    def samples_to_nat_params(self, z):
        m_out = self.mean_fn(z)
        h_out = (1 / Fn.softplus(self.log_Q)) * m_out
        J_out_diag_vec = (1 / Fn.softplus(self.log_Q)) * torch.ones_like(m_out)
        J_out = torch.diag_embed(J_out_diag_vec)
        nat_params_out = self.pdf.pack_params(h_out, J_out)

        return nat_params_out

    def get_jvp_expected_mean_params(self, nat_params, lagrange_mult):
        mean_params = self.pdf.natural_to_mean(nat_params)
        m, M = self.pdf.unpack_params(mean_params)

        if nat_params.dim() == 3:
            A = torch.vmap(torch.vmap(torch.vmap(torch.func.jacfwd(self.mean_fn))))(m.unsqueeze(0)).squeeze(0)
        elif nat_params.dim() == 2:
            A = torch.vmap(torch.vmap(torch.func.jacfwd(self.mean_fn)))(m.unsqueeze(0)).squeeze(0)

        l, L = self.pdf.unpack_params(lagrange_mult)
        jvp1 = bmv(A.mT, l)
        jvp2 = A.mT @ L @ A
        jvp = self.pdf.pack_params(jvp1, jvp2)
        return jvp


class DiagonalGaussianNonlinearDynamics(nn.Module):
    def __init__(self, mean_fn, n_latents, pdf, Q=None, device='cpu'):
        super(DiagonalGaussianNonlinearDynamics, self).__init__()
        self.device = device

        self.pdf = pdf
        self.mean_fn = mean_fn
        self.n_latents = n_latents
        self.n_nat_params = n_latents + n_latents

        if Q is None:
            self.log_Q = torch.nn.Parameter(utils.softplus_inv(torch.ones(n_latents).to(device)))
        else:
            self.log_Q = torch.nn.Parameter(utils.softplus_inv(Q))

    def sample_expected_mean_params(self, nat_params, n_samples):
        z_s = self.pdf.sample(nat_params, n_samples)
        mean_params_out = self.samples_to_mean_params(z_s)
        mean_params_out = mean_params_out.mean(dim=0)

        return mean_params_out

    def sample_expected_mean_params_from_mean_params(self, mean_params, n_samples):
        z_s = self.pdf.sample_from_mean_params(mean_params, n_samples)
        mean_params_out = self.samples_to_mean_params(z_s)
        mean_params_out = mean_params_out.mean(dim=0)

        return mean_params_out

    def samples_to_mean_params(self, z):
        m_out = self.mean_fn(z)
        P_out = Fn.softplus(self.log_Q) * torch.ones_like(m_out)
        M_out = -0.5 * (P_out + m_out**2)
        mean_params_out = self.pdf.pack_params(m_out, M_out)

        return mean_params_out

    def samples_to_nat_params(self, z):
        m_out = self.mean_fn(z)
        J_out = (1 / Fn.softplus(self.log_Q)) * torch.ones_like(m_out)
        h_out = J_out * m_out
        nat_params_out = self.pdf.pack_params(h_out, J_out)

        return nat_params_out

class DenseGaussianInitialCondition(nn.Module):
    def __init__(self, n_latents, m0, v0=None, device='cpu'):
        super(DenseGaussianInitialCondition, self).__init__()
        self.device = device
        self.n_latents = n_latents
        self.m0 = torch.nn.Parameter(m0).to(self.device)

        if v0 is None:
            self.log_v0 = torch.nn.Parameter(utils.softplus_inv(torch.ones(n_latents).to(device)))
        else:
            self.log_v0 = torch.nn.Parameter(utils.softplus_inv(v0)).to(device)

    def get_nat_params(self):
        v0 = Fn.softplus(self.log_v0)
        h0 = (1 / v0) * self.m0
        J0 = (1 / v0) * torch.eye(self.n_latents, device=self.device)

        nat_params_out = self.pack_params(h0, J0)
        return nat_params_out

    def get_canon_params(self, get_v0_sqrt=False):
        v0 = Fn.softplus(self.log_v0)
        P0 = v0 * torch.eye(self.n_latents, device=self.device)

        if not get_v0_sqrt:
            return self.m0, P0
        else:
            P0_sqrt = torch.sqrt(v0) * torch.eye(self.n_latents, device=self.device)
            return self.m0, P0, P0_sqrt

    def pack_params(self, a, B):
        B_vec = einops.rearrange(B, '... i j -> ... (i j)')
        return torch.concat([a, B_vec], dim=-1)


class DiagonalGaussianInitialCondition(nn.Module):
    def __init__(self, n_latents, m0, v0=None, device='cpu'):
        super(DiagonalGaussianInitialCondition, self).__init__()
        self.device = device
        self.n_latents = n_latents
        self.m0 = torch.nn.Parameter(m0).to(device)

        if v0 is None:
            self.log_v0 = torch.nn.Parameter(utils.softplus_inv(torch.ones(n_latents).to(device)))
        else:
            self.log_v0 = torch.nn.Parameter(utils.softplus_inv(v0)).to(device)

    def get_nat_params(self):
        v0 = Fn.softplus(self.log_v0)
        J0 = (1 / v0)
        h0 = J0 * self.m0

        nat_params_out = self.pack_params(h0, J0)
        return nat_params_out

    def pack_params(self, a, B):
        return torch.concat([a, B], dim=-1)