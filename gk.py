########################################################################################################################
# Compare TEKI and ABC techniques for g-and-k distribution (4 dimensional)
########################################################################################################################

import os

from jax import numpy as np, random, vmap

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
simulation_params.n_repeats = 3

# EKI ##################################################################################################################
# Number of samples to generate
simulation_params.n_samps_eki = np.array([200, 1000, 5000])

# threshold param
simulation_params.eki_max_temp = 4.0

# ABC MCMC #############################################################################################################
simulation_params.n_samps_rwmh = int(1e5)

# N pre-run
simulation_params.n_abc_pre_run = int(1e3)

# ABC distance thresholds
simulation_params.abc_thresholds = np.array([3, 7, 15, 50])

# RWMH stepsizes
simulation_params.rwmh_stepsizes = np.array([1e-2, 1e-1, 1e-0])


# ABC SMC ##############################################################################################################
# Number of samples to generate
simulation_params.n_samps_abc_smc = np.array([200, 1000, 5000])


# Number of intermediate MCMC steps to take
simulation_params.n_mcmc_steps_abc_smc = 10

# Maximum iterations
simulation_params.max_iter_abc_smc = 100

# Retain threshold parameter
simulation_params.threshold_quantile_retain_abc_smc = 0.75

########################################################################################################################

simulation_params.save(save_dir + '/sim_params', overwrite=True)


class GKThinOrder(abc.scenarios.GKTransformedUniformPrior):
    num_thin: int = 100
    threshold = 5

    def simulate_data(self,
                      x: np.ndarray,
                      random_key: np.ndarray) -> np.ndarray:
        data_keys = random.split(random_key, simulation_params.n_data)
        return vmap(self.likelihood_sample, (None, 0))(x, data_keys)

    def summarise_data(self,
                       data: np.ndarray):
        order_stats = data.sort()
        thin_inds = np.linspace(0, len(data), self.num_thin, endpoint=False, dtype='int32')
        return order_stats[thin_inds]

    def distance_function(self,
                          summarised_simulated_data: np.ndarray) -> float:
        return np.sqrt(np.square(summarised_simulated_data - self.summary_statistic).sum())


gk_scenario = GKThinOrder()

true_constrained_params = np.array([3., 1., 2., 0.5])
true_unconstrained_params = gk_scenario.unconstrain(true_constrained_params)

random_key = random.PRNGKey(0)
random_key, subkey = random.split(random_key)
repeat_sim_data_keys = random.split(subkey, simulation_params.n_repeats)


def generate_data(rkey):
    sim_data_keys = random.split(rkey, simulation_params.n_data)
    data = vmap(gk_scenario.likelihood_sample, (None, 0))(true_unconstrained_params, sim_data_keys)
    return data


each_data = vmap(generate_data)(repeat_sim_data_keys)
each_summary_statistic = vmap(gk_scenario.summarise_data)(each_data)

########################################################################################################################
# Run EKI
########################################################################################################################

# Run EKI
# utils.run_eki(gk_scenario, save_dir, random_key, repeat_data=each_summary_statistic)

# Run RWMH ABC
# utils.run_abc_mcmc(gk_scenario, save_dir, random_key, repeat_summarised_data=each_summary_statistic)

# Run AMC SMC
# utils.run_abc_smc(gk_scenario, save_dir, random_key, repeat_summarised_data=each_summary_statistic)

param_names = (r'$A$', r'$B$', r'$g$', r'$k$')
# ranges = ([0., 10.], [0., 5.], [0., 10.], [0., 5.])
plot_ranges = ([2., 4.], [0., 3.2], [0., 10.], [0., 5.])

# Plot EKI
utils.plot_eki(gk_scenario, save_dir, plot_ranges, true_params=true_constrained_params, param_names=param_names,
               y_range_mult2=1.0,
               rmse_temp_round=0)

# Plot ABC
utils.plot_abc_mcmc(gk_scenario, save_dir, plot_ranges, true_params=true_constrained_params, param_names=param_names)


