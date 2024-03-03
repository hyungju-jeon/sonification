import h5py
import torch
import random
import numpy as np
import filter.utils as utils
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib import cm
from sklearn.linear_model import Ridge
from lightning.pytorch.loggers import CSVLogger
from filter.dynamics import DenseGaussianInitialCondition
from filter.dynamics import DenseGaussianNonlinearDynamics
from filter.approximations import DenseGaussianApproximations
from filter.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from filter.likelihoods import PoissonLikelihood, GaussianLikelihood, LinearPolarToCartesian

from filter.nonlinear_smoother import NonlinearFilter, FullRankNonlinearStateSpaceModelFilter


def main():
    bin_sz = 20e-3
    device = 'cpu'
    data_device = 'cpu'
    bin_sz_ms = int(bin_sz * 1e3)

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    default_dtype = torch.float32
    torch.set_default_dtype(default_dtype)

    """hyperparameters"""
    n_inputs = 1
    n_latents = 2
    n_hidden_current_obs = 128
    n_samples = 25
    rank_y = 2

    batch_sz = 256
    n_epochs = 250
    blues = cm.get_cmap("Blues", n_samples)

    """data params"""
    n_trials = 1000
    n_neurons = 100
    n_time_bins = 25
    theta = np.pi / 4

    A = torch.nn.Linear(n_latents, n_latents, bias=False, device=device).requires_grad_(False)
    A.weight.data = 0.9 * utils.make_2d_rotation_matrix(theta, device=device)
    B = torch.nn.Linear(n_inputs, n_latents, bias=False, device=device).requires_grad_(False)
    C = torch.nn.Linear(n_latents, n_neurons, device=device).requires_grad_(False)
    Q_0_diag = torch.ones(n_latents, device=device).requires_grad_(False)
    Q_diag = torch.ones(n_latents, device=device).requires_grad_(False)
    R_diag = torch.ones(n_neurons, device=device).requires_grad_(False)
    m_0 = torch.zeros(n_latents, device=device).requires_grad_(False)

    """generate input and latent/observations"""
    u = torch.rand((n_trials, n_time_bins, n_inputs), device=device)
    y_gt, z_gt = utils.sample_lds(u, A.weight.data, B.weight.data, C.weight.data, C.bias.data,
                                  Q_diag, R_diag, m_0, Q_0_diag, device=device)

    y_train_dataset = torch.utils.data.TensorDataset(y_gt, u, z_gt,)
    train_dataloader = torch.utils.data.DataLoader(y_train_dataset, batch_size=batch_sz, shuffle=True)

    """approximation pdf"""
    approximation_pdf = DenseGaussianApproximations(n_latents, device)

    """likelihood pdf"""
    C = LinearPolarToCartesian(n_latents, n_neurons, 4, device=device)
    C.linear.weight.data = None
    likelihood_pdf = PoissonLikelihood(C, n_neurons, bin_sz, device=device)

    """dynamics module"""
    # dynamics_fn = utils.build_gru_dynamics_function(n_latents, n_hidden_dynamics, device=device)
    dynamics_fn = A
    dynamics_mod = DenseGaussianNonlinearDynamics(dynamics_fn, n_latents, approximation_pdf, Q_diag, device=device)

    """initial condition"""
    initial_condition_pdf = DenseGaussianInitialCondition(n_latents, m_0, Q_0_diag, device=device)

    """local/backward encoder"""
    observation_to_nat = LocalEncoderLRMvn(n_neurons, n_hidden_current_obs, n_latents,
                                           likelihood_pdf=likelihood_pdf, rank=rank_y, device=device)
    nl_filter = NonlinearFilter(dynamics_mod, initial_condition_pdf, device)

    """sequence vae"""
    ssm = FullRankNonlinearStateSpaceModelFilter(dynamics_mod, approximation_pdf, likelihood_pdf, B,
                                                 initial_condition_pdf, observation_to_nat, nl_filter, device=device)

    # ssm.likelihood_pdf.readout_fn[-1].bias.data = prob_utils.estimate_poisson_rate_bias(y_train, bin_sz)
    """train model"""
    opt = torch.optim.Adam(ssm.parameters(), lr=1e-3, weight_decay=1e-6)

    for t in (p_bar := tqdm(range(n_epochs), position=0, leave=True)):
        avg_loss = 0.

        print(f'epoch: {t}')
        for dx, (y_tr, u_tr, z_tr) in enumerate(train_dataloader):
            ssm.train()
            opt.zero_grad()
            loss, z_s, stats = ssm(y_tr, n_samples, u_tr)
            avg_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(ssm.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            p_bar.set_description(f'loss: {loss.item()}')

        avg_loss /= len(train_dataloader)

        with torch.no_grad():
            if t % 10 == 0:
                torch.save(ssm.state_dict(), f'results/ssm_state_dict_epoch_{t}.pt')
                fig, axs = plt.subplots(1, n_latents)
                [axs[i].plot(z_s[j, 0, :, i], color=blues(j), alpha=0.5)
                 for i in range(n_latents) for j in range(n_samples)]
                [axs[i].plot(z_tr[0, :, i], color='black', alpha=0.7, label='true') for i in range(n_latents)]
                [axs[i].set_box_aspect(1.0) for i in range(n_latents)]
                [axs[i].set_title(f'dim {i}') for i in range(n_latents)]
                plt.show()

    torch.save(ssm.state_dict(), f'results/ssm_state_dict_epoch_{n_epochs}.pt')

    """real-time test"""
    z_f = []

    for t in range(n_time_bins):
        if t == 0:
            stats_t, z_f_t = ssm.step_0(y_gt[:, t], u[:, t], n_samples)
        else:
            stats_t, z_f_t = ssm.step_t(y_gt[:, t], u[:, t], n_samples, z_f[t-1])

        z_f.append(z_f_t)

    z_f = torch.stack(z_f, dim=2)

    with torch.no_grad():
        fig, axs = plt.subplots(1, n_latents)
        [axs[i].plot(z_f[j, 0, :, i], color=blues(j), alpha=0.5)
         for i in range(n_latents) for j in range(n_samples)]
        [axs[i].plot(z_gt[0, :, i], color='black', alpha=0.7, label='true') for i in range(n_latents)]
        [axs[i].set_box_aspect(1.0) for i in range(n_latents)]
        [axs[i].set_title(f'dim {i}') for i in range(n_latents)]
        plt.show()

if __name__ == '__main__':
    main()
