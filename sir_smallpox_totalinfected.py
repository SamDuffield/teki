########################################################################################################################
# Compare TEKI and ABC techniques for SIR model (2 dimensional) - Abakaliki Smallpox Outbreak (1975)
########################################################################################################################

import os

from jax import numpy as jnp, random

import mocat
from mocat import abc

import utils

save_dir = f'./simulations/sir_smallpox_ti'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

########################################################################################################################
# Simulation parameters
simulation_params = mocat.cdict()

# Number repeated simulations per algorithm
simulation_params.n_repeats = 1

# EKI ##################################################################################################################
# Vary n_samps
simulation_params.vary_n_samps_eki = jnp.asarray(10 ** jnp.linspace(2.2, 3.5, 5), dtype='int32')

# Fixed n_samps
simulation_params.fix_n_samps_eki = 500

# Fixed n_steps
simulation_params.fix_n_steps = 50

# Vary number of eki steps, fix n_samps
simulation_params.vary_n_steps_eki = jnp.array([1, 10, 100, 1000])

# max sd for optimisation
simulation_params.optim_max_sd_eki = 0.1

# max temp
simulation_params.max_temp_eki = 5.

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
# simulation_params.vary_n_samps_abc_smc = jnp.asarray(10 ** jnp.linspace(2.2, 4, 5), dtype='int32')
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


class TSIRTotalInfected(abc.scenarios.TransformedSIR):
    initial_si = jnp.array([119, 1])
    data = jnp.array([30])

    prior_rates = jnp.ones(2) * 0.1

    def likelihood_sample(self,
                          x: jnp.ndarray,
                          random_key: jnp.ndarray) -> jnp.ndarray:
        sim_times, sim_si = self.simulate_times_and_si(x, random_key)
        pop_size = self.initial_si.sum()
        return jnp.array([pop_size - sim_si[jnp.argmin(sim_si[:, 0]), 0]])

    def distance_function(self,
                          simulated_data: jnp.ndarray) -> float:
        diff_data = simulated_data - self.data
        return jnp.abs(diff_data)


sir_scenario = TSIRTotalInfected()

random_key = random.PRNGKey(0)

# # Run EKI
# utils.run_eki(sir_scenario, save_dir, random_key)
#
# # Run RWMH ABC
# utils.run_abc_mcmc(sir_scenario, save_dir, random_key)
#
# # Run AMC SMC
# utils.run_abc_smc(sir_scenario, save_dir, random_key)
#

param_names = (r'$\lambda$', r'$\gamma$')
plot_ranges = [[0., 3.], [0, 3.]]

# # Plot EKI
# utils.plot_eki(sir_scenario, save_dir, plot_ranges, param_names=param_names, bp_widths=0.7,
#                optim_ranges=[[0., 3.], [0, 3.]],
#                rmse_temp_round=0)
#
# # # Plot ABC-MCMC
# # utils.plot_abc_mcmc(sir_scenario, save_dir, plot_ranges,  param_names=param_names)
# #
#
# # Plot ABC-SMC
# utils.plot_abc_smc(sir_scenario, save_dir, plot_ranges,  param_names=param_names,
#                    rmse_temp_round=0, legend_loc='upper right', legend_ax=1, legend_size=8)
#
# # Plot distances
# utils.plot_dists(sir_scenario, save_dir)

# Plot resampled distances
n_resamps = 10
utils.plot_res_dists(sir_scenario, save_dir, n_resamps)

