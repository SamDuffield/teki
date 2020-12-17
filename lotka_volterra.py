########################################################################################################################
# Compare TEKI and ABC techniques for stochastic Lotka-Volterra (3 dimensional)
########################################################################################################################

import os
from typing import Tuple
from jax import numpy as np, random

import mocat
from mocat import abc

import utils

save_dir = f'./simulations/lotka_volterra'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

########################################################################################################################
# Simulation parameters
simulation_params = mocat.cdict()

# Number repeated simulations per algorithm
simulation_params.n_repeats = 20

# EKI
# Number of samples to generate
simulation_params.n_samps_eki = np.array([500, 1000, 2000])

# threshold param
simulation_params.eki_optim_max_sd = 0.2

# ABC MCMC #############################################################################################################
simulation_params.n_samps_rwmh = int(1e5)

# N pre-run
simulation_params.n_abc_pre_run = int(1e4)

# ABC distance thresholds
simulation_params.abc_thresholds = np.array([120, 170, 250])

# RWMH stepsizes
simulation_params.rwmh_stepsizes = np.array([1e-2, 1e-1, 1e-0])


# ABC SMC ##############################################################################################################
# Number of samples to generate
simulation_params.n_samps_abc_smc = np.array([500, 2000, 5000])


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
    data = np.array([88, 165, 274, 268, 114, 46, 32, 36, 53, 92])
    summary_statistic = data

    prior_rates = np.ones(3)

    def distance_function(self,
                          summarised_simulated_data: np.ndarray) -> float:
        return np.max(np.abs(summarised_simulated_data - self.data))


lv_scenario = TLotkaVolterraDist()

true_constrained_params = np.array([1., 0.005, 0.6])
true_unconstrained_params = lv_scenario.unconstrain(true_constrained_params)

zero_dist = lv_scenario.distance_function(np.zeros_like(lv_scenario.data))


def generate_init_state_extra(abc_scenario: abc.ABCScenario,
                              n_samps: int,
                              random_key: np.ndarray,
                              max_iter: int = None) -> Tuple[mocat.cdict, mocat.cdict, int]:

    if max_iter is None:
        max_iter = n_samps * 100

    def body_fun(previous_carry, num_accepted_and_rkey):
        state, extra = previous_carry
        num_accepted, rkey = num_accepted_and_rkey
        rkey, val_key, data_key = random.split(rkey, 3)
        state.value = abc_scenario.prior_sample(val_key)
        extra.simulated_summary = abc_scenario.summarise_data(abc_scenario.simulate_data(state.value, data_key))
        state.distance = abc_scenario.distance_function(extra.simulated_summary)

        accept_state = state.distance < zero_dist
        num_accepted = np.where(accept_state, num_accepted + 1,  num_accepted)
        return (state, extra), (num_accepted, rkey)

    init_state = mocat.cdict(value=np.zeros(abc_scenario.dim), distance=zero_dist)
    init_extra = mocat.cdict(simulated_summary=np.zeros_like(abc_scenario.summary_statistic))

    all_states, all_extras = mocat.utils.while_loop_stacked(lambda carry, num_a_rkey: num_a_rkey[0] < n_samps,
                                                            body_fun,
                                                            ((init_state, init_extra), (0, random_key)),
                                                            max_iter)

    keep_inds = all_states.distance < zero_dist
    return all_states[keep_inds], all_extras[keep_inds], len(all_states.value)


random_key = random.PRNGKey(0)

# Run EKI
utils.run_eki(lv_scenario, save_dir, random_key)

# Run RWMH ABC
utils.run_abc_mcmc(lv_scenario, save_dir, random_key)

# Run AMC SMC
utils.run_abc_smc(lv_scenario, save_dir, random_key, initial_state_extra_generator=generate_init_state_extra)


param_names = (r'$\theta_1$', r'$\theta_2$', r'$\theta_3$')
plot_ranges = [[0., 3.], [0, 0.05], [0, 3.]]

# Plot EKI
utils.plot_eki(lv_scenario, save_dir, plot_ranges, true_params=true_constrained_params, param_names=param_names,
               y_range_mult2=1.0,
               rmse_temp_round=0)

# Plot ABC-MCMC
utils.plot_abc_mcmc(lv_scenario, save_dir, plot_ranges, true_params=true_constrained_params, param_names=param_names)


# Plot ABC-SMC
utils.plot_abc_smc(lv_scenario, save_dir, plot_ranges, true_params=true_constrained_params, param_names=param_names,
                   rmse_temp_round=0)

