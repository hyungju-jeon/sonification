import h5py
import utils
import torch
import random
import nlb_tools
from nlb_tools.evaluation import velocity_decoding
import numpy as np
import lightning as lightning
import matplotlib.pyplot as plt
import filter.prob_utils as prob_utils

from tqdm import tqdm
from matplotlib import cm
from sklearn.linear_model import Ridge
from lightning.pytorch.loggers import CSVLogger
from filter.likelihoods import PoissonLikelihood
from filter.dynamics import DenseGaussianInitialCondition
from filter.dynamics import DenseGaussianNonlinearDynamics
from filter.approximations import DenseGaussianApproximations
from filter.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn

from filter.nonlinear_smoother import NonlinearFilter, FullRankNonlinearStateSpaceModel


import matplotlib


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
    n_latents = 40
    n_latents_read = 40
    n_hidden_dynamics = 128
    n_hidden_embedding = 128
    n_hidden_current_obs = 256
    rank_y, rank_b = 20, 5
    n_samples = 40

    n_samples_plt = 50
    blues = cm.get_cmap("Blues", n_samples_plt)

    batch_sz = 128
    n_epochs = 1000

    """data"""
    y_val_dataset = torch.utils.data.TensorDataset(y_val,)
    y_train_dataset = torch.utils.data.TensorDataset(y_train,)
    valid_dataloader = torch.utils.data.DataLoader(y_val_dataset, batch_size=batch_sz, shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(y_train_dataset, batch_size=batch_sz, shuffle=True)

    """approximation pdf"""
    approximation_pdf = DenseGaussianApproximations(n_latents, device)

    """likelihood pdf"""
    H = utils.ReadoutLatentMask(n_latents, n_latents_read)
    C = torch.nn.Linear(n_latents_read, n_neurons_obs, device=device)
    readout_fn = torch.nn.Sequential(H, C)
    likelihood_pdf = PoissonLikelihood(readout_fn, n_neurons_obs, bin_sz, device=device)

    """dynamics module"""
    Q = torch.ones(n_latents, device=device)
    dynamics_fn = utils.build_gru_dynamics_function(n_latents, n_hidden_dynamics, device=device)
    dynamics_mod = DenseGaussianNonlinearDynamics(dynamics_fn, n_latents, approximation_pdf, Q, device=device)

    """initial condition"""
    m0 = torch.zeros(n_latents, device=device)
    Q0 = 1.0 * torch.ones(n_latents, device=device)
    initial_condition_pdf = DenseGaussianInitialCondition(n_latents, m0, Q0, device=device)

    """local/backward encoder"""
    encoder = BackwardEncoderLRMvn(n_neurons_enc, n_hidden_embedding, n_latents, rank_y=rank_y, rank_b=rank_b, device=device, dropout=0.0)
    observation_to_nat = LocalEncoderLRMvn(n_neurons_enc, n_hidden_current_obs, n_latents, likelihood_pdf=likelihood_pdf, rank=rank_y, device=device, dropout=0.0)
    nl_filter = NonlinearFilter(dynamics_mod, initial_condition_pdf, device)

    """sequence vae"""
    ssm = FullRankNonlinearStateSpaceModel(dynamics_mod, approximation_pdf, likelihood_pdf,
                                           initial_condition_pdf, encoder, observation_to_nat, nl_filter, device=device)

    ssm.likelihood_pdf.readout_fn[-1].bias.data = prob_utils.estimate_poisson_rate_bias(y_train_obs, bin_sz)

    opt = torch.optim.Adam(ssm.parameters(), lr=1e-3, weight_decay=1e-6)

    for t in (p_bar := tqdm(range(n_epochs), position=0, leave=True)):
        avg_loss = 0.

        print(f'epoch: {t}')
        for dx, (y_obs_tr,) in enumerate(train_dataloader):

            ssm.train()
            opt.zero_grad()
            loss, z_s, stats = ssm(y_obs_tr, n_samples)
            avg_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(ssm.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            p_bar.set_description(f'loss: {loss.item()}')

        avg_loss /= len(train_dataloader)



if __name__ == '__main__':
    main()
