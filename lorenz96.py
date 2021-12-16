########################################################################################################################
# Compare TEKI and ABC techniques for Lorenz 96 scenario (high dimensional)
########################################################################################################################

import os
from typing import Any, Union

from jax import numpy as jnp, random, vmap
from jax.lax import scan

import mocat
from mocat import abc
from mocat import ssm

import utils

save_dir = f'./simulations/l96'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

########################################################################################################################
# Simulation parameters
simulation_params = mocat.cdict()

# Number repeated simulations per algorithm
simulation_params.n_repeats = 20

# EKI ##################################################################################################################
# Vary n_samps
simulation_params.vary_n_samps_eki = jnp.asarray(10 ** jnp.linspace(2.366, 3.699, 5), dtype='int32')
# simulation_params.vary_n_samps_eki = jnp.array([200, 1000, 5000])
# simulation_params.vary_n_samps_eki = jnp.array([200])

# # Fixed n_samps
# simulation_params.fix_n_samps_eki = 500
#
# # Fixed n_steps
# simulation_params.fix_n_steps = 50
#
# # Vary number of eki steps, fix n_samps
# simulation_params.vary_n_steps_eki = jnp.array([1, 10, 100])

# max sd for optimisation
simulation_params.optim_max_sd_eki = 0.1

# max temp
simulation_params.max_temp_eki = 10.

# ess_threshold for automatic temperatures
simulation_params.ess_threshold = 1/3

# ABC MCMC #############################################################################################################
# N max
simulation_params.n_samps_abc_mcmc = int(1e4)

# N pre-run
simulation_params.n_pre_run_abc_mcmc = int(1e3)

# N cut_off for stepsize
simulation_params.pre_run_ar_abc_mcmc = 0.1

# RM params
simulation_params.rm_stepsize_scale_mcmc = 1.
simulation_params.rm_stepsize_neg_exponent = 2 / 3

# ABC SMC ##############################################################################################################
# Vary n_samps
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


class L96(abc.ABCScenario):
    name: str = 'Lorenz 96'

    times: jnp.ndarray = None
    data: jnp.ndarray = None

    prior_covariance_sqrt: jnp.ndarray = None
    prior_precision_sqrt: jnp.ndarray = None
    prior_precision_det: float = None
    transition_covariance_sqrt: jnp.ndarray = None
    transition_precision_sqrt: jnp.ndarray = None
    transition_precision_det: float = None
    likelihood_covariance_sqrt: jnp.ndarray = None
    likelihood_precision_sqrt: jnp.ndarray = None
    likelihood_precision_det: float = None

    def __init__(self,
                 dim: int = 40,
                 forcing_constant: float = 8.,
                 prior_mean: Union[float, jnp.ndarray] = None,
                 prior_covariance: Union[float, jnp.ndarray] = 1.,
                 transition_covariance: Union[float, jnp.ndarray] = 1.,
                 likelihood_matrix: jnp.ndarray = 1.,
                 likelihood_covariance: Union[float, jnp.ndarray] = 1.,
                 likelihood_times: jnp.ndarray = jnp.array([1.]),
                 integrator_stepsize: float = 1e-3,
                 name: str = None):
        self.dim = dim
        super().__init__(name=name)
        self.forcing_constant = forcing_constant

        if prior_mean is None:
            prior_mean = jnp.zeros(self.dim) + forcing_constant
        self.prior_mean = prior_mean

        if isinstance(prior_covariance, float):
            prior_covariance = jnp.eye(self.dim) * prior_covariance
        self.prior_covariance = prior_covariance

        if isinstance(transition_covariance, float):
            transition_covariance = jnp.eye(self.dim) * transition_covariance
        self.transition_covariance = transition_covariance

        if isinstance(likelihood_matrix, float):
            likelihood_matrix = jnp.eye(self.dim) * likelihood_matrix
        self.dim_obs = likelihood_matrix.shape[0]
        self.likelihood_matrix = likelihood_matrix

        if isinstance(likelihood_covariance, float):
            likelihood_covariance = jnp.eye(self.dim_obs) * likelihood_covariance
        self.likelihood_covariance = likelihood_covariance

        self.likelihood_times = likelihood_times
        self.integrator_stepsize = integrator_stepsize
        self.integration_times = jnp.arange(0., self.likelihood_times.max() + 1e-5, self.integrator_stepsize)
        self.lik_inds = (self.integration_times[1:, None] == self.likelihood_times).argmax(axis=0)

    def __setattr__(self,
                    key: str,
                    value: Any):
        self.__dict__[key] = value
        if key[-10:] == 'covariance':
            mocat.utils.reset_covariance(self, key, value)

    def prior_sample(self,
                     random_key: jnp.ndarray) -> jnp.ndarray:
        return self.prior_mean + self.prior_covariance_sqrt @ random.normal(random_key, (self.dim,))

    def prior_potential(self,
                        x: jnp.ndarray,
                        random_key: jnp.ndarray = None) -> float:
        return mocat.utils.gaussian_potential(x, self.prior_mean,
                                              sqrt_prec=self.prior_precision_sqrt, det_prec=self.prior_precision_det)

    def likelihood_sample(self,
                          x: jnp.ndarray,
                          random_key: jnp.ndarray) -> jnp.ndarray:

        random_keys = random.split(random_key, len(self.integration_times) + len(self.likelihood_times))

        def body_fun(x_int, i):
            x_new = x_int + self.integrator_stepsize * ssm.scenarios.lorenz96_dynamics(x_int, 0, self.forcing_constant) \
                    + jnp.sqrt(self.integrator_stepsize) * self.transition_covariance_sqrt @ random.normal(
                random_keys[i], (self.dim,))
            return x_new, x_new

        _, trajectories = scan(body_fun, x, jnp.arange(len(self.integration_times) - 1))

        lik_rks = random_keys[len(self.integration_times):]
        trim_traj = trajectories[self.lik_inds]

        return jnp.concatenate(vmap(lambda a, rk: self.likelihood_matrix @ a
                                                  + self.likelihood_covariance_sqrt @ random.normal(rk,
                                                                                                    (self.dim_obs,))) \
                                   (trim_traj, lik_rks))


# dim: int = 40
# lik_mat = jnp.eye(dim)[jnp.arange(0, 40, 2)]
# lik_times = jnp.array([1., 2.])
# forcing_constant: float = 8.
# prior_cov: float = 3.
# transit_cov: float = 3.
# lik_cov: float = 0.1

# dim: int = 40
# lik_mat = jnp.eye(dim)[jnp.arange(0, dim, 2)]
# lik_times = jnp.arange(1., 6.)
# forcing_constant: float = 8.
# prior_cov: float = 5.
# transit_cov: float = 1.
# lik_cov: float = 0.1

dim: int = 40
lik_mat = jnp.eye(dim)[jnp.arange(0, dim, 2)]
lik_times = jnp.arange(1., 6.)
forcing_constant: float = 8.
prior_cov: float = 5.
transit_cov: float = 1.
lik_cov: float = .1


l96_scen = L96(dim=dim, forcing_constant=forcing_constant,
               prior_mean=jnp.zeros(dim) + forcing_constant, prior_covariance=prior_cov,
               transition_covariance=transit_cov,
               likelihood_matrix=lik_mat, likelihood_covariance=lik_cov, likelihood_times=lik_times)

true_params = l96_scen.prior_sample(random.PRNGKey(1))

random_key = random.PRNGKey(0)
repeat_sim_data_keys = random.split(random_key, simulation_params.n_repeats + 1)
random_key = repeat_sim_data_keys[0]

each_data = vmap(l96_scen.likelihood_sample, (None, 0))(true_params, repeat_sim_data_keys[1:])

########################################################################################################################

# # Run EKI
utils.run_eki(l96_scen, save_dir, random_key, repeat_data=each_data)

# # Run MCMC ABC
utils.run_abc_mcmc(l96_scen, save_dir, random_key, repeat_summarised_data=each_data)
#
# # Run AMC SMC
utils.run_abc_smc(l96_scen, save_dir, random_key, repeat_summarised_data=each_data)

# ########################################################################################################################

param_names = (r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$')
plot_ranges = ([-5, 15.],) * dim


# # Plot densities
utils.plot_comp_densities(l96_scen, save_dir, plot_ranges, dim_inds=jnp.arange(4),
                          true_params=true_params, param_names=param_names,
                          repeat_ind=3, n_ind=1)

# # Plot optim box plots
utils.plot_optim_box_plots(l96_scen, save_dir, dim_inds=jnp.arange(4),
                           true_params=true_params, param_names=param_names)

# Plot RMSE
utils.plot_rmse(l96_scen, save_dir, true_params)

