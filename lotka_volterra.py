########################################################################################################################
# Compare TEKI and ABC techniques for stochastic Lotka-Volterra (3 dimensional)
########################################################################################################################

import os

from jax import numpy as np, random

import mocat
from mocat import abc

import utils

save_dir = f'./simulations/lotka_volterra'
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
simulation_params.eki_max_temp = 30.0

# RWMH
simulation_params.n_samps_rwmh = int(1e5)

# N pre-run
simulation_params.n_abc_pre_run = int(1e3)

# ABC distance thresholds
simulation_params.abc_thresholds = np.array([1, 2, 5])

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


class TLotkaVolterraDist(abc.scenarios.TransformedLotkaVolterra):
    initial_prey_pred = np.array([50, 100])
    times = np.arange(11, dtype='float32')
    data = np.log(np.array([88, 165, 274, 268, 114, 46, 32, 36, 53, 92]))
    summary_statistic = data

    prior_rates = np.ones(3)

    def distance_function(self,
                          summarised_simulated_data: np.ndarray) -> float:
        summarised_simulated_data = np.where(summarised_simulated_data == -np.inf, -1e5, summarised_simulated_data)
        return np.abs(summarised_simulated_data - self.data)


lv_scenario = TLotkaVolterraDist()

true_constrained_params = np.array([1., 0.005, 0.6])
true_unconstrained_params = lv_scenario.unconstrain(true_constrained_params)

random_key = random.PRNGKey(0)

# eki_temp_1 = mocat.run_tempered_ensemble_kalman_inversion(lv_scenario,
#                                                           np.max(simulation_params.n_samps_eki),
#                                                           random_key,
#                                                           max_temp=1.)

# eki_temp_20 = mocat.run_tempered_ensemble_kalman_inversion(lv_scenario,
#                                                            np.max(simulation_params.n_samps_eki),
#                                                            random_key,
#                                                            max_temp=simulation_params.eki_max_temp)


# Run EKI
# utils.run_eki(lv_scenario, save_dir, random_key)

# Run RWMH ABC
# utils.run_abc_mcmc(lv_scenario, save_dir, random_key)

# Run AMC SMC
utils.run_abc_smc(lv_scenario, save_dir, random_key)


param_names = (r'$\theta_1$', r'$\theta_2$', r'$\theta_3$')
plot_ranges = [[0., 3.], [0, 0.05], [0, 3.]]

# # Plot EKI
# utils.plot_eki(lv_scenario, save_dir, plot_ranges, true_params=true_constrained_params, param_names=param_names,
#                y_range_mult2=1.0,
#                rmse_temp_round=0)
#
# # Plot ABC
# utils.plot_abc_mcmc(lv_scenario, save_dir, plot_ranges, true_params=true_constrained_params, param_names=param_names,
#                     y_range_mult2=1.0)
