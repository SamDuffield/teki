########################################################################################################################
# Compare TEKI and ABC techniques for Gaussian Mixture model (d dimensional)
########################################################################################################################

import os
from typing import Union

from jax import numpy as jnp, random, vmap

import mocat
from mocat import abc

import utils

save_dir = f'./simulations/mixture'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

########################################################################################################################
# Simulation parameters
simulation_params = mocat.cdict()

# Number repeated simulations per algorithm
simulation_params.n_repeats = 5

# EKI ##################################################################################################################
# Vary n_samps
simulation_params.vary_n_samps_eki = jnp.asarray(10 ** jnp.linspace(2.2, 3.7, 6), dtype='int32')

# Fixed n_samps
simulation_params.fix_n_samps_eki = 500

# Fixed n_steps
simulation_params.fix_n_steps = 100

# Vary number of eki steps, fix n_samps
simulation_params.vary_n_steps_eki = jnp.array([1, 10, 100, 1000])

# max sd for optimisation
simulation_params.optim_max_sd_eki = 0.1

# max temp
simulation_params.max_temp_eki = 50.

# ABC MCMC #############################################################################################################
# N max
simulation_params.n_samps_abc_mcmc = int(1e6)

# N pre-run
simulation_params.n_pre_run_abc_mcmc = int(1e3)

# N cut_off for stepsize
simulation_params.pre_run_ar_abc_mcmc = 0.1

# RM params
simulation_params.rm_stepsize_scale_mcmc = 1.
simulation_params.rm_stepsize_neg_exponent = 2 / 3

# ABC SMC ##############################################################################################################
# Vary n_samps
# simulation_params.vary_n_samps_abc_smc = jnp.asarray(10 ** jnp.linspace(2.5, 4, 5), dtype='int32')
simulation_params.vary_n_samps_abc_smc = simulation_params.vary_n_samps_eki

# Maximum iterations
simulation_params.max_iter_abc_smc = 1000

# Retain threshold parameter
simulation_params.threshold_quantile_retain_abc_smc = 0.9

# Resample threshold parameter
simulation_params.threshold_quantile_resample_abc_smc = 0.5

# MCMC acceptance rate to terminate
simulation_params.termination_alpha = 0.015

########################################################################################################################

simulation_params.save(save_dir + '/sim_params', overwrite=True)


class MixtureModel(mocat.Scenario):
    name: str = 'Mixture Model'
    data: jnp.ndarray
    prior_mins: Union[float, jnp.ndarray] = -20.
    prior_maxs: Union[float, jnp.ndarray] = 40.

    def __init__(self,
                 dim: int,
                 w: float = 0.3,
                 rho: float = 0.7,
                 **kwargs):
        self.dim = dim
        self.w = w
        self.rho = rho

        self.likelihood_covariance = jnp.eye(self.dim) * (1 - self.rho) + self.rho
        self.likelihood_covariance_sqrt = jnp.linalg.cholesky(self.likelihood_covariance)
        self.likelihood_precision_sqrt = jnp.linalg.inv(self.likelihood_covariance_sqrt)
        self.likelihood_precision_det = 1 / jnp.linalg.det(self.likelihood_covariance)

        binary_combinations = jnp.array(jnp.meshgrid(*jnp.tile(jnp.arange(2), (dim, 1)))).T.reshape(-1, dim)
        self.weight_mat = jnp.where(binary_combinations, 1 - self.w, self.w).prod(1)
        self.mean_scalar_mat = 1 - 2 * binary_combinations
        super().__init__(**kwargs)

    def prior_sample(self,
                     random_key: jnp.ndarray) -> Union[float, jnp.ndarray]:
        return random.uniform(random_key, shape=(self.dim,)) * (self.prior_maxs - self.prior_mins) + self.prior_mins

    def prior_potential(self,
                        x: jnp.ndarray,
                        random_key: jnp.ndarray = None) -> float:
        return jnp.where(jnp.all(x > self.prior_mins) * jnp.all(x < self.prior_maxs), 0., jnp.inf)

    def likelihood_sample(self,
                          x: jnp.ndarray,
                          random_key: jnp.ndarray) -> jnp.ndarray:
        bern_key, norm_key = random.split(random_key)
        bs = random.bernoulli(bern_key, 1 - self.w, shape=(self.dim,))
        mean = x * (1 - 2 * bs)
        return mean + self.likelihood_covariance_sqrt @ random.normal(norm_key, shape=(self.dim,))

    def likelihood_dens(self,
                        x: jnp.ndarray,
                        random_key: jnp.ndarray = None) -> float:
        mean_mat = x * self.mean_scalar_mat
        gauss_pot_evals = mocat.utils.gaussian_potential(mean_mat,
                                                         self.data,
                                                         sqrt_prec=self.likelihood_precision_sqrt,
                                                         det_prec=self.likelihood_precision_det)
        return (self.weight_mat * jnp.exp(-gauss_pot_evals)).sum()

    def likelihood_potential(self,
                             x: jnp.ndarray,
                             random_key: jnp.ndarray = None) -> float:
        return -jnp.log(self.likelihood_dens(x, random_key))

    def distance_function(self,
                          simulated_data: jnp.ndarray) -> Union[float, jnp.ndarray]:
        return jnp.sqrt(jnp.square(simulated_data - self.data).sum())


dim = 2
mixture_scenario = MixtureModel(dim=dim)
mixture_scenario.data = jnp.ones(dim) * 5
