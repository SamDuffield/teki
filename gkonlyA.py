########################################################################################################################
# Compare TEKI and ABC techniques for g-and-k distribution (1 dimensional)
########################################################################################################################

import os

from jax import numpy as jnp, random, vmap

import mocat
from mocat import abc

import utils

save_dir = f'./simulations/GKonlyA'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

########################################################################################################################
# Simulation parameters
simulation_params = mocat.cdict()

# Number of simulations from true data
simulation_params.n_data = int(1e4)

# Number of evenly spaced quantiles to summarise data
simulation_params.n_quantiles = 20

# Number repeated simulations per algorithm
simulation_params.n_repeats = 1

# EKI ##################################################################################################################
# Number of samples to generate
simulation_params.n_samps_eki = jnp.array([500])

# Number of eki steps
simulation_params.n_steps_eki = jnp.array([1, 10, 100])

# max sd for optimisation
simulation_params.optim_max_sd_eki = 0.

# max temp
simulation_params.max_temp_eki = 1.

# ABC MCMC #############################################################################################################
simulation_params.n_samps_rwmh = int(1e5)

# N pre-run
simulation_params.n_abc_pre_run = int(1e2)

# ABC SMC ##############################################################################################################
# Number of samples to generate
simulation_params.n_samps_abc_smc = jnp.array([1000, 10000])

# Maximum iterations
simulation_params.max_iter_abc_smc = 100

# Retain threshold parameter
simulation_params.threshold_quantile_retain_abc_smc = 0.95

########################################################################################################################

simulation_params.save(save_dir + '/sim_params', overwrite=True)


# class GKThinOrder(abc.scenarios.GKTransformedUniformPrior):
#     num_thin: int = 100
#     n_unsummarised_data: int = simulation_params.n_data
#
#     def summarise_data(self,
#                        data: jnp.ndarray):
#         order_stats = data.sort()
#         thin_inds = jnp.linspace(0, len(data), self.num_thin, endpoint=False, dtype='int32')
#         return order_stats[thin_inds]
#
# gk_scenario = GKThinOrder()


class GKQuantile(abc.scenarios.GKOnlyATransformedUniformPrior):
    quantiles = jnp.linspace(0, 1, simulation_params.n_quantiles + 1, endpoint=False)[1:]
    n_unsummarised_data: int = simulation_params.n_data

    def summarise_data(self,
                       data: jnp.ndarray):
        return jnp.quantile(data, self.quantiles)


gk_scenario = GKQuantile()


true_constrained_params = jnp.array([3.])
true_unconstrained_params = gk_scenario.unconstrain(true_constrained_params)

random_key = random.PRNGKey(0)
random_key, subkey = random.split(random_key)
repeat_sim_data_keys = random.split(subkey, simulation_params.n_repeats)

each_data = vmap(gk_scenario.likelihood_sample, (None, 0))(true_unconstrained_params, repeat_sim_data_keys)

########################################################################################################################

# Run EKI
# utils.run_eki(gk_scenario, save_dir, random_key, repeat_data=each_data)
#
# # Run MCMC ABC
# utils.run_abc_mcmc(gk_scenario, save_dir, random_key, repeat_summarised_data=each_summary_statistic)
#
# # Run AMC SMC
# utils.run_abc_smc(gk_scenario, save_dir, random_key, repeat_summarised_data=each_summary_statistic)
#
# ########################################################################################################################

param_names = (r'$A$',)
# ranges = ([0., 10.], [0., 5.], [0., 10.], [0., 5.])
plot_ranges = ([1., 4.],)

# # Plot EKI
utils.plot_eki(gk_scenario, save_dir, plot_ranges, true_params=true_constrained_params, param_names=param_names,
               y_range_mult2=0.25,
               rmse_temp_round=0)

# # Plot ABC-MCMC
# utils.plot_abc_mcmc(gk_scenario, save_dir, plot_ranges, true_params=true_constrained_params, param_names=param_names)
#
# # Plot ABC-SMC
# utils.plot_abc_smc(gk_scenario, save_dir, plot_ranges, true_params=true_constrained_params, param_names=param_names,
#                    rmse_temp_round=0)
