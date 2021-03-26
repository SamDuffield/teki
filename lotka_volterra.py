########################################################################################################################
# Compare TEKI and ABC techniques for stochastic Lotka-Volterra (3 dimensional)
########################################################################################################################

import os
from typing import Tuple
from jax import numpy as jnp, random

import mocat
from mocat import abc

import utils

save_dir = f'./simulations/lotka_volterra'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

########################################################################################################################
# Simulation parameters
simulation_params = mocat.cdict()

# Number of simulations from true data
simulation_params.n_data = int(1e3)

# Number repeated simulations per algorithm
simulation_params.n_repeats = 1

# EKI ##################################################################################################################
# Vary n_samps
simulation_params.vary_n_samps_eki = jnp.asarray(10 ** jnp.linspace(2.3, 3.7, 6), dtype='int32')

# Fixed n_samps
simulation_params.fix_n_samps_eki = 300

# Fixed n_steps
simulation_params.fix_n_steps = 50

# Vary number of eki steps, fix n_samps
simulation_params.vary_n_steps_eki = jnp.array([1, 10, 100])

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
simulation_params.vary_n_samps_abc_smc = jnp.asarray(10 ** jnp.linspace(2.5, 4, 5), dtype='int32')

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


class TLotkaVolterraDist(abc.scenarios.TransformedLotkaVolterra):
    initial_prey_pred = jnp.array([50, 100])
    times = jnp.arange(11, dtype='float32')
    data = jnp.array([88, 165, 274, 268, 114, 46, 32, 36, 53, 92])
    summary_statistic = data

    prior_rates = jnp.array([1., 100., 1.])

    def distance_function(self,
                          summarised_simulated_data: jnp.ndarray) -> float:
        return jnp.max(jnp.abs(jnp.log(summarised_simulated_data) - jnp.log(self.data)))


lv_scenario = TLotkaVolterraDist()

true_constrained_params = jnp.array([1., 0.005, 0.6])
true_unconstrained_params = lv_scenario.unconstrain(true_constrained_params)

zero_dist = lv_scenario.distance_function(jnp.zeros_like(lv_scenario.data))


def generate_init_state_extra(abc_scenario: abc.ABCScenario,
                              n_samps: int,
                              random_key: jnp.ndarray,
                              max_iter: int = None) -> Tuple[mocat.cdict, mocat.cdict, int]:

    if max_iter is None:
        max_iter = n_samps * 100

    def body_fun(previous_carry, num_accepted_and_rkey):
        state, extra = previous_carry
        num_accepted, rkey = num_accepted_and_rkey
        rkey, val_key, data_key = random.split(rkey, 3)
        state.value = abc_scenario.prior_sample(val_key)
        extra.simulated_data = abc_scenario.likelihood_sample(state.value, data_key)
        state.distance = abc_scenario.distance_function(extra.simulated_data)

        accept_state = state.distance < zero_dist
        num_accepted = jnp.where(accept_state, num_accepted + 1,  num_accepted)
        return (state, extra), (num_accepted, rkey)

    init_state = mocat.cdict(value=jnp.zeros(abc_scenario.dim), distance=zero_dist)
    init_extra = mocat.cdict(simulated_data=jnp.zeros_like(abc_scenario.data))

    all_states, all_extras = mocat.utils.while_loop_stacked(lambda carry, num_a_rkey: num_a_rkey[0] < n_samps,
                                                            body_fun,
                                                            ((init_state, init_extra), (0, random_key)),
                                                            max_iter)

    keep_inds = all_states.distance < zero_dist
    return all_states[keep_inds], all_extras[keep_inds], len(all_states.value)


random_key = random.PRNGKey(0)

# Run EKI
# utils.run_eki(lv_scenario, save_dir, random_key)

# Run RWMH ABC
# utils.run_abc_mcmc(lv_scenario, save_dir, random_key)

# # Run AMC SMC
# utils.run_abc_smc(lv_scenario, save_dir, random_key, initial_state_extra_generator=generate_init_state_extra)


param_names = (r'$\theta_1$', r'$\theta_2$', r'$\theta_3$')
plot_ranges = [[0., 2.], [0, 0.025], [0, 2.]]
optim_ranges = [[0., 1.5], [0, 0.01], [0, 1.]]


# # Plot EKI
utils.plot_eki(lv_scenario, save_dir, plot_ranges, true_params=true_constrained_params, param_names=param_names,
               optim_ranges=optim_ranges, bp_widths=0.7,
               rmse_temp_round=0)

# # # Plot ABC-MCMC
# utils.plot_abc_mcmc(lv_scenario, save_dir, plot_ranges, true_params=true_constrained_params, param_names=param_names)
#
#
# # Plot ABC-SMC
# utils.plot_abc_smc(lv_scenario, save_dir, plot_ranges, true_params=true_constrained_params, param_names=param_names,
#                    rmse_temp_round=0, legend_loc='upper right', legend_ax=1, legend_size=8)
#
#
# # Plot RMSE
# utils.plot_rmse(lv_scenario, save_dir, true_constrained_params)
