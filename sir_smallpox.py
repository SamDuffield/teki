########################################################################################################################
# Compare TEKI and ABC techniques for SIR model (2 dimensional)
# Abakaliki Smallpox Outbreak (1975)
########################################################################################################################

import os

from jax import numpy as np, random

import mocat
from mocat import abc

import utils

save_dir = f'./simulations/sir_smallpox'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

########################################################################################################################
# Simulation parameters
simulation_params = mocat.CDict()

# Number repeated simulations per algorithm
simulation_params.n_repeats = 20

# EKI
# Number of samples to generate
simulation_params.n_samps_eki = np.array([500, 1000, 2000])

# threshold param
simulation_params.eki_max_temp = 3.0

# RWMH
simulation_params.n_samps_rwmh = int(1e6)

# N pre-run
simulation_params.n_abc_pre_run = int(1e5)

# ABC distance thresholds
simulation_params.abc_thresholds = np.array([5, 10, 15])

# RWMH stepsizes
simulation_params.rwmh_stepsizes = np.array([1e-2, 1e-1, 1e-0])
########################################################################################################################

simulation_params.save(save_dir + '/sim_params', overwrite=True)


class TSIRRemovalTimes(abc.scenarios.TransformedSIR):
    initial_si = np.array([119, 1])
    times = np.array([0., 13., 26., 39., 52., 65., 78., np.inf])
    data = np.array([2, 6, 3, 7, 8, 4, 0, 76])
    summary_statistic = data

    prior_rates = np.ones(2) * 0.1

    def simulate(self,
                 x: np.ndarray,
                 random_key: np.ndarray) -> np.ndarray:
        sim_times, sim_si = self.likelihood_sample(x, random_key)
        max_time = np.max(sim_times)
        sim_times = np.where(sim_times == 0, np.inf, sim_times)

        pop_size = self.initial_si.sum()

        active_pop_size = sim_si.sum(1)
        final_active_pop_size = np.min(active_pop_size)
        active_pop_size = np.where(sim_times == np.inf, final_active_pop_size, active_pop_size)

        time_inds = np.searchsorted(sim_times, self.times[1:]) - 1
        active_pop_size_times = active_pop_size[time_inds]

        active_pop_size_previous_times = np.append(pop_size, active_pop_size_times[:-1])

        return np.append(active_pop_size_previous_times - active_pop_size_times, max_time)

    def distance_function(self,
                          summarised_simulated_data: np.ndarray) -> float:
        diff_data = summarised_simulated_data - self.summary_statistic
        return np.sqrt(np.square(diff_data[:-1]) + (diff_data[-1] / 50) ** 2)


sir_scenario = TSIRRemovalTimes()

random_key = random.PRNGKey(0)

# eki_temp_1 = mocat.run_tempered_ensemble_kalman_inversion(sir_scenario,
#                                                           np.max(simulation_params.n_samps_eki),
#                                                           random_key,
#                                                           max_temp=1.)
#
# eki_temp_optim = mocat.run_tempered_ensemble_kalman_inversion(sir_scenario,
#                                                               np.max(simulation_params.n_samps_eki),
#                                                               random_key,
#                                                               max_temp=simulation_params.eki_max_temp)

# Run EKI
# utils.run_eki(sir_scenario, save_dir, random_key)

# Run RWMH ABC
# utils.run_abc(sir_scenario, save_dir, random_key)

param_names = (r'$\lambda$', r'$\gamma$')

# Plot EKI
plot_ranges_eki = [[0., 5.], [0, 0.3]]
# utils.plot_eki(sir_scenario, save_dir, plot_ranges_eki, param_names=param_names, y_range_mult2=0.5,
#                rmse_temp_round=0)

# Plot ABC
plot_ranges_abc = [[0., 5.], [0, 0.3]]
utils.plot_abc(sir_scenario, save_dir, plot_ranges_abc, param_names=param_names, y_range_mult2=1.0)


