import utils
import torch
import einops
import torch.nn as nn
import torch.nn.functional as Fn
import torch.distributions as dist

from functools import partial
from filter.linalg_utils import bip, bop, bqp, bmv, triangular_inverse


class DenseGaussianApproximations(nn.Module):
    def __init__(self, n_latents, device='cpu'):
        super(DenseGaussianApproximations, self).__init__()
        self.device = device
        self.n_latents = n_latents
        self.n_nat_params = n_latents + n_latents**2
        # self.tril_indices = torch.tril_indices(n_latents, n_latents, offset=0).to(device).tolist()
        # self.get_tril_entries = partial(utils.matrix_index_select, indices=self.tril_indices)

    def log_prob(self, nat_params, z):
        h, J = self.unpack_params(nat_params)
        J_chol = torch.linalg.cholesky(J)
        m = torch.cholesky_solve(h.unsqueeze(-1), J_chol).squeeze(-1)

        q_pdf = dist.MultivariateNormal(m, precision_matrix=J)
        log_prob = q_pdf.log_prob(z)
        return log_prob

    def sample(self, nat_params, n_samples):
        h, J = self.unpack_params(nat_params)
        J_chol = torch.linalg.cholesky(J)
        P_sqrt = triangular_inverse(J_chol).mT
        m = torch.cholesky_solve(h.unsqueeze(-1), J_chol).squeeze(-1)
        z_samples = m + bmv(P_sqrt, torch.randn([n_samples] + list(m.shape)).to(nat_params.device))

        return z_samples

    def sample_from_mean_params(self, mean_params, n_samples):
        m, M = self.unpack_params(mean_params)
        P = -2 * M - bop(m, m)

        q_pdf = dist.MultivariateNormal(m, P)
        z_samples = q_pdf.rsample([n_samples])
        return z_samples

    def natural_to_mean(self, nat_params):
        h, J = self.unpack_params(nat_params)
        J_chol = torch.linalg.cholesky(J)

        P = torch.cholesky_inverse(J_chol)
        m = torch.cholesky_solve(h.unsqueeze(-1), J_chol).squeeze(-1)
        M = -0.5 * (P + bop(m, m))
        mean_params = self.pack_params(m, M)

        return mean_params

    def mean_to_natural(self, mean_params):
        m, M = self.unpack_params(mean_params)
        P = -2 * M - torch.einsum('...i, ...j -> ...ij', m, m)

        try:
            P_chol = torch.linalg.cholesky(P)
            h = torch.cholesky_solve(m.unsqueeze(-1), P_chol).squeeze(-1)
            J = torch.cholesky_inverse(P_chol)
        except:
            U, S, V = torch.svd(P)# + 1e-2 * torch.eye(P.shape[-1]))
            J = (U * (1 / (S + 1e-2)).unsqueeze(-1)) @ U.mT
            h = bmv(J, m)

        nat_params = self.pack_params(h, J)

        return nat_params

    def kl_divergence(self, nat_params_1, nat_params_2):
        # 0.5 * [(m2 - m1) J2 (m2 - m1) + tr(J2 J1^{-1}) + log |J1| - log |J2|]
        h1, J1 = self.unpack_params(nat_params_1)
        h2, J2 = self.unpack_params(nat_params_2)

        J1_chol = torch.linalg.cholesky(J1)
        J2_chol = torch.linalg.cholesky(J2)

        m1 = torch.cholesky_solve(h1.unsqueeze(-1), J1_chol).squeeze(-1)
        m2 = torch.cholesky_solve(h2.unsqueeze(-1), J2_chol).squeeze(-1)

        qp = bqp(J2, m2 - m1)
        tr = torch.einsum('...ii -> ...', torch.cholesky_solve(J2.mT, J1_chol).mT)
        logdet1 = 2 * torch.sum(torch.log(torch.diagonal(J1_chol, dim1=-2, dim2=-1)+1e-8), dim=-1)
        logdet2 = 2 * torch.sum(torch.log(torch.diagonal(J2_chol, dim1=-2, dim2=-1)+1e-8), dim=-1)

        kl = 0.5 * (qp + tr + logdet1 - logdet2 - h1.shape[-1])
        return kl

    def kl_divergence_nat_mean(self, nat_params_1, mean_params_2):
        # 0.5 * [(m2 - m1) J2 (m2 - m1) + tr(J2 J1^{-1}) + log |J1| - log |J2|]
        h1, J1 = self.unpack_params(nat_params_1)
        m2, M2 = self.unpack_params(mean_params_2)
        P2 = -2 * M2 - bop(m2, m2)

        J1_chol = torch.linalg.cholesky(J1)
        P2_chol = torch.linalg.cholesky(P2)

        m1 = torch.cholesky_solve(h1.unsqueeze(-1), J1_chol).squeeze(-1)
        J2 = torch.cholesky_inverse(P2_chol)

        qp = bqp(J2, m2 - m1)
        tr = torch.einsum('...ii -> ...', torch.cholesky_solve(J2.mT, J1_chol).mT)
        logdet1 = 2 * torch.sum(torch.log(torch.diagonal(J1_chol, dim1=-2, dim2=-1)+1e-8), dim=-1)
        logdet2 = -2 * torch.sum(torch.log(torch.diagonal(P2_chol, dim1=-2, dim2=-1)+1e-8), dim=-1)

        kl = 0.5 * (qp + tr + logdet1 - logdet2 - h1.shape[-1])
        return kl

    def log_Z(self, nat_params):
        # 0.5 * m^T J m - 0.5 * log |J|
        h, J = self.unpack_params(nat_params)
        J_chol = torch.linalg.cholesky(J)
        P = torch.cholesky_inverse(J_chol)

        log_Z = 0.5 * bqp(P, h) - 0.5 * torch.logdet(J)
        return log_Z

    def to_sufficient_statistics(self, z):
        z_outer = torch.einsum('...i, ...j -> ...ij', z, z)
        t_z = self.pack_params(z, z_outer)
        return t_z

    def fisher_inv_from_nat_params(self, nat_params):
        raise NotImplementedError

        h, J = self.unpack_params(nat_params)

        # TODO: use dev
        fisher_inv = utils.mvn_fisher_inv_wrt_nat(h, J)

    def fisher_from_nat_params(self, nat_params):
        raise NotImplementedError

        mean_params = self.natural_to_mean(nat_params)
        m, M = self.unpack_params(mean_params)
        P = -2 * M - bop(m, m)

        # TODO: use dev
        fisher = utils.mvn_fisher_wrt_nat(m, P)

    def get_standard_nat_params(self):
        h = torch.zeros(self.n_latents).to(self.device)
        J = torch.eye(self.n_latents).to(self.device)
        nat_params = self.pack_params(h, J)

        return nat_params

    def pack_params(self, a, B):
        B_vec = einops.rearrange(B, '... i j -> ... (i j)')
        return torch.concat([a, B_vec], dim=-1)

    def unpack_params(self, params):
        a = params[..., :self.n_latents]
        B_vec = params[..., self.n_latents:]
        B = einops.rearrange(B_vec, '... (i j) -> ... i j', i=self.n_latents, j=self.n_latents)

        return a, B