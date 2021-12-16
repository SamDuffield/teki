########################################################################################################################
# Compare TEKI and ABC techniques for g-and-k distribution (4 dimensional)
########################################################################################################################

import os

from jax import numpy as jnp, random, vmap

import mocat
from mocat import abc

import utils

save_dir = f'./simulations/GK'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

########################################################################################################################
# Simulation parameters
simulation_params = mocat.cdict()

# Number of simulations from true data
simulation_params.n_data = int(1e3)

# Number repeated simulations per algorithm
simulation_params.n_repeats = 20

# EKI ##################################################################################################################
# Vary n_samps
simulation_params.vary_n_samps_eki = jnp.asarray(10 ** jnp.linspace(2.366, 3.699, 5), dtype='int32')
# simulation_params.vary_n_samps_eki = jnp.array([200, 1000, 5000])

# # Fixed n_samps
# simulation_params.fix_n_samps_eki = 500

# # Fixed n_steps
# simulation_params.fix_n_steps = 50

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


class GKThinOrder(abc.scenarios.GKTransformedUniformPrior):
    num_thin: int = 100
    n_unsummarised_data: int = simulation_params.n_data

    def summarise_data(self,
                       data: jnp.ndarray):
        order_stats = data.sort()
        thin_inds = jnp.linspace(0, len(data), self.num_thin, endpoint=False, dtype='int32')
        return order_stats[thin_inds]


gk_scenario = GKThinOrder()

true_constrained_params = jnp.array([3., 1., 2., 0.5])
true_unconstrained_params = gk_scenario.unconstrain(true_constrained_params)

random_key = random.PRNGKey(0)
repeat_sim_data_keys = random.split(random_key, simulation_params.n_repeats + 1)
random_key = repeat_sim_data_keys[0]

each_data = vmap(gk_scenario.likelihood_sample, (None, 0))(true_unconstrained_params, repeat_sim_data_keys[1:])

########################################################################################################################

# # Run EKI
utils.run_eki(gk_scenario, save_dir, random_key, repeat_data=each_data)

# # Run MCMC ABC
utils.run_abc_mcmc(gk_scenario, save_dir, random_key, repeat_summarised_data=each_data)
#
# # Run AMC SMC
utils.run_abc_smc(gk_scenario, save_dir, random_key, repeat_summarised_data=each_data)

# ########################################################################################################################

param_names = (r'$A$', r'$B$', r'$g$', r'$k$')
plot_ranges = ([2.5, 3.5], [0., 2.], [0., 10.], [0., 3.])

# # Plot densities
utils.plot_comp_densities(gk_scenario, save_dir, plot_ranges,
                          true_params=true_constrained_params, param_names=param_names,
                          repeat_ind=3, n_ind=1)

# # Plot optim box plots
utils.plot_optim_box_plots(gk_scenario, save_dir,
                           true_params=true_constrained_params, param_names=param_names)


# Plot RMSE
utils.plot_rmse(gk_scenario, save_dir, true_constrained_params)
