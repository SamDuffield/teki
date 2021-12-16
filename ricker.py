########################################################################################################################
# Compare TEKI and ABC techniques for Ricker model (3 dimensional)
########################################################################################################################

import os
from typing import Union

from jax import numpy as jnp, random, vmap

import mocat
from mocat import abc

import utils

save_dir = f'./simulations/ricker'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

########################################################################################################################
# Simulation parameters
simulation_params = mocat.cdict()

# Number repeated simulations per algorithm
simulation_params.n_repeats = 1

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


class RickerTransformedUniform(abc.scenarios.Ricker):

    num_steps: int = 100
    observation_inds: jnp.ndarray = jnp.arange(51, 101)

    prior_mins: float = jnp.array([0., jnp.log(0.1), jnp.log(0.1)])
    prior_maxs: float = jnp.array([5., 0, jnp.log(15.)])

    def constrain(self,
                  unconstrained_x: jnp.ndarray):
        return jnp.exp(unconstrained_x)

    def unconstrain(self,
                    constrained_x: jnp.ndarray):
        return jnp.log(constrained_x)

    def prior_potential(self,
                        x: jnp.ndarray,
                        random_key: jnp.ndarray = None) -> Union[float, jnp.ndarray]:
        out = jnp.where(jnp.all(x > self.prior_mins), 1., jnp.inf)
        out = jnp.where(jnp.all(x < self.prior_maxs), out, jnp.inf)
        return out

    def prior_sample(self,
                     random_key: jnp.ndarray) -> Union[float, jnp.ndarray]:
        return self.prior_mins + random.uniform(random_key, (self.dim,)) * (self.prior_maxs - self.prior_mins)

    def likelihood_sample(self,
                          x: jnp.ndarray,
                          random_key: jnp.ndarray) -> jnp.ndarray:
        return super().likelihood_sample(self.constrain(x), random_key)


ricker_scenario = RickerTransformedUniform()

true_constrained_params = jnp.array([jnp.exp(3.8), 0.3, 10.])
true_unconstrained_params = ricker_scenario.unconstrain(true_constrained_params)

random_key = random.PRNGKey(0)
repeat_sim_data_keys = random.split(random_key, simulation_params.n_repeats + 1)
random_key = repeat_sim_data_keys[-1]

each_data = vmap(ricker_scenario.likelihood_sample, (None, 0))(true_unconstrained_params, repeat_sim_data_keys[:-1])

########################################################################################################################

# # # Run EKI
# utils.run_eki(ricker_scenario, save_dir, random_key, repeat_data=each_data)
#
# # # Run MCMC ABC
# utils.run_abc_mcmc(ricker_scenario, save_dir, random_key, repeat_summarised_data=each_data)
# #
# # # Run AMC SMC
# utils.run_abc_smc(ricker_scenario, save_dir, random_key, repeat_summarised_data=each_data)

# ########################################################################################################################

param_names = (r'$r$', r'$\sigma_e$', r'$\phi$')
plot_ranges = ([1., jnp.exp(5)], [0., 1.], [0., 15.])

# Plot EKI
utils.plot_eki(ricker_scenario, save_dir, plot_ranges, true_params=true_constrained_params, param_names=param_names,
               # y_range_mult=0.75
               bp_widths=0.1,
               rmse_temp_round=0)

# # Plot ABC-MCMC
# utils.plot_abc_mcmc(ricker_scenario, save_dir, plot_ranges, true_params=true_constrained_params, param_names=param_names)

# Plot ABC-SMC
utils.plot_abc_smc(ricker_scenario, save_dir, plot_ranges, true_params=true_constrained_params, param_names=param_names,
                   trim_thresholds=10,
                   rmse_temp_round=0)


# Plot RMSE
utils.plot_rmse(ricker_scenario, save_dir, true_constrained_params)

# Plot distances
utils.plot_dists(ricker_scenario, save_dir, repeat_summarised_data=each_data)

# Plot resampled distances
n_resamps = 100
utils.plot_res_dists(ricker_scenario, save_dir, n_resamps, repeat_summarised_data=each_data)

