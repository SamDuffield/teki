########################################################################################################################
# Compare TEKI and ABC techniques for ecological dynamical system (6 dimensional)
########################################################################################################################

import os
from typing import Union, Tuple

from jax import numpy as jnp, random, vmap
from jax.ops import index_update
from jax.lax import scan, cond

import mocat
from mocat import abc

import utils

save_dir = f'./simulations/ecological'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

########################################################################################################################
# Simulation parameters
simulation_params = mocat.cdict()

# Number repeated simulations per algorithm
simulation_params.n_repeats = 1

# EKI ##################################################################################################################
# Vary n_samps
# simulation_params.vary_n_samps_eki = jnp.array([200, 1000, 5000])
simulation_params.vary_n_samps_eki = jnp.array([200, 1000])

# Fixed n_samps
simulation_params.fix_n_samps_eki = 200

# Fixed n_steps
simulation_params.fix_n_steps = 50

# Vary number of eki steps, fix n_samps
simulation_params.vary_n_steps_eki = jnp.array([1, 10, 100])

# max sd for optimisation
simulation_params.optim_max_sd_eki = 0.1

# max temp
simulation_params.max_temp_eki = 10.

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


def ecological_simulate(initial_n: float,
                        p: float,
                        sigma_1: float,
                        sigma_2: float,
                        lag: int,
                        delta: float,
                        num_steps: int,
                        random_key: jnp.ndarray) -> jnp.ndarray:
    random_keys = random.split(random_key, num_steps * 2)
    keys_1 = random_keys[:num_steps]
    keys_2 = random_keys[num_steps:]

    sigma_1_sq = sigma_1 ** 2
    sigma_2_sq = sigma_2 ** 2

    init_ns = jnp.zeros(num_steps + 1, dtype='float32') + initial_n

    def body_fun(ns: jnp.ndarray, i: int) -> Tuple[jnp.ndarray, None]:
        n_min1 = ns[i - 1]
        n_minlag = jnp.where(i - 1 - lag >= 0, ns[i - 1 - lag], initial_n)

        new_n = p * n_minlag * jnp.exp(-n_minlag / initial_n) * sigma_1_sq * random.gamma(keys_1[i],
                                                                                          1 / sigma_1_sq) \
                + n_min1 * jnp.exp(-delta * sigma_2_sq * random.gamma(keys_2[i], 1 / sigma_2_sq))

        return index_update(ns, i, new_n), None

    ns, _ = scan(body_fun, init_ns, jnp.arange(1, num_steps + 1))
    return ns


# class EcologicalTransformedUniform(abc.ABCScenario):
#     name: str = 'Blowfly'
#     dim: int = 6
#
#     prior_mins: float = -5.
#     prior_maxs: float = 2.
#
#     num_steps: int = 180 + 50
#     observation_inds: jnp.ndarray = jnp.arange(num_steps - 180, num_steps)
#
#     def constrain(self,
#                   unconstrained_x: jnp.ndarray):
#         return jnp.exp(unconstrained_x)
#
#     def unconstrain(self,
#                     constrained_x: jnp.ndarray):
#         return jnp.log(constrained_x)
#
#     def prior_potential(self,
#                         x: jnp.ndarray,
#                         random_key: jnp.ndarray = None) -> Union[float, jnp.ndarray]:
#         out = jnp.where(jnp.all(x > self.prior_mins), 1., jnp.inf)
#         out = jnp.where(jnp.all(x < self.prior_maxs), out, jnp.inf)
#         return out
#
#     def prior_sample(self,
#                      random_key: jnp.ndarray) -> Union[float, jnp.ndarray]:
#         return self.prior_mins + random.uniform(random_key, (self.dim,)) * (self.prior_maxs - self.prior_mins)
#
#     def likelihood_sample(self,
#                           x: jnp.ndarray,
#                           random_key: jnp.ndarray) -> jnp.ndarray:
#         params = self.constrain(x)
#         lag = jnp.ceil(params[4]).astype('int32')
#         ns = ecological_simulate(*params[:4], lag, params[-5], self.num_steps, random_key)
#         return self.summarise_data(ns[self.observation_inds])
#
#     # def summarise_data(self, ns: jnp.ndarray) -> jnp.ndarray:
#     #     ns_mean = ns.mean()
#     #     return jnp.array([ns_mean, jnp.median(ns) - ns_mean, ns.argmax(), jnp.log(ns.max())])
#
#     def summarise_data(self, ns: jnp.ndarray) -> jnp.ndarray:
#         return ns
#
#
# ecological_scenario = EcologicalTransformedUniform()


def peakdet(x: jnp.ndarray, delta: float):

    def body_fun(carry, i):
        live_min, live_max, live_min_pos, live_max_pos, lookformax = carry
        current_val = x[i]
        new_live_min_bool = current_val < live_min
        live_min = jnp.where(new_live_min_bool, current_val, live_min)
        live_min_pos = jnp.where(new_live_min_bool, i, live_min_pos)
        new_live_max_bool = current_val > live_max
        live_max = jnp.where(new_live_max_bool, current_val, live_max)
        live_max_pos = jnp.where(new_live_max_bool, i, live_max_pos)

        new_min_bool = jnp.logical_and(~lookformax, current_val > (live_min + delta))
        ret_mintab_pos = jnp.where(new_min_bool, live_min_pos, -1)
        live_max = jnp.where(new_min_bool, current_val, live_max)
        live_max_pos = jnp.where(new_min_bool, i, live_max_pos)
        lookformax = jnp.where(new_min_bool, True, lookformax)

        new_max_bool = jnp.logical_and(lookformax, current_val < (live_max - delta))
        ret_maxtab_pos = jnp.where(new_max_bool, live_max_pos, -1)
        live_min = jnp.where(new_max_bool, current_val, live_min)
        live_min_pos = jnp.where(new_max_bool, i, live_min_pos)
        lookformax = jnp.where(new_max_bool, False, lookformax)

        return (live_min, live_max, live_min_pos, live_max_pos, lookformax), (ret_mintab_pos, ret_maxtab_pos)

    return scan(body_fun, (jnp.inf, -jnp.inf, 0, 0, True), jnp.arange(len(x)))[1]


class EcologicalTransformedGaussian(abc.ABCScenario):
    name: str = 'Blowfly'
    dim: int = 6

    prior_means: jnp.ndarray = jnp.array([6., 2., -0.5, -0.75, 2.7, -1.8])
    prior_sds: jnp.ndarray = jnp.array([0.5, 2., 1., 1., 0.1, 0.4])

    num_steps: int = 180 + 50
    observation_inds: jnp.ndarray = jnp.arange(num_steps - 180, num_steps)

    # Params x = (N_0, P, sigma_1, sigma_2, lag, delta)

    def constrain(self,
                  unconstrained_x: jnp.ndarray):
        return jnp.exp(unconstrained_x)

    def unconstrain(self,
                    constrained_x: jnp.ndarray):
        return jnp.log(constrained_x)

    def prior_potential(self,
                        x: jnp.ndarray,
                        random_key: jnp.ndarray = None) -> Union[float, jnp.ndarray]:
        return 0.5 * jnp.square((x - self.prior_means) / self.prior_sds).sum()

    def prior_sample(self,
                     random_key: jnp.ndarray) -> Union[float, jnp.ndarray]:
        return self.prior_means + self.prior_sds * random.normal(random_key, shape=(self.dim,))

    def likelihood_sample(self,
                          x: jnp.ndarray,
                          random_key: jnp.ndarray) -> jnp.ndarray:
        params = self.constrain(x)
        lag = jnp.ceil(params[4]).astype('int32')
        ns = ecological_simulate(*params[:4], lag, params[-5], self.num_steps, random_key)
        return self.summarise_data(ns[self.observation_inds])

    # def summarise_data(self, ns: jnp.ndarray) -> jnp.ndarray:
    #     ns_mean = ns.mean()
    #     return jnp.array([ns_mean, jnp.median(ns) - ns_mean, ns.argmax(), jnp.log(ns.max() + 1e-10)])

    def summarise_data(self, ns: jnp.ndarray) -> jnp.ndarray:
        ns_sorted = ns.sort()
        ns_diff_sorted = jnp.diff(ns).sort()
        n = len(self.observation_inds)

        peaks0point5m_inds = peakdet(ns, 0.5)[1]
        num_peaks0point5m = (peaks0point5m_inds >= 0).size
        peaks1point5m_inds = peakdet(ns, 1.5)[1]
        num_peaks1point5m = (peaks1point5m_inds >= 0).size

        return jnp.array([jnp.log(ns_sorted[:n//4].mean() / 1000. + 1e-10),
                          jnp.log(ns_sorted[n//4:n//2].mean() / 1000. + 1e-10),
                          jnp.log(ns_sorted[n//2:(3*n)//4].mean() / 1000. + 1e-10),
                          jnp.log(ns_sorted[(3*n)//4:].mean() / 1000. + 1e-10),
                          ns_diff_sorted[:n//4].mean(),
                          ns_diff_sorted[n//4:n//2].mean(),
                          ns_diff_sorted[n//2:(3*n)//4].mean(),
                          ns_diff_sorted[(3*n)//4:].mean(),
                          num_peaks0point5m,
                          num_peaks1point5m])


ecological_scenario = EcologicalTransformedGaussian()


### data from https://rdrr.io/cran/gamair/man/blowfly.html
with open(save_dir + '/blowflydata.csv', 'r') as datafile:
    blowfly_data = jnp.array([int(a.split(',')[0]) for i, a in enumerate(datafile) if i > 0])

ecological_scenario.data = ecological_scenario.summarise_data(blowfly_data)

random_key = random.PRNGKey(0)

# from teki import TemperedEKI
# # t_1 = 1e-4
# # n = 1000
# # beta = jnp.exp(-jnp.log(t_1) / (n - 2))
# # sched = jnp.append(0, t_1 * beta ** jnp.arange(n - 1))
# # eki_samps = mocat.run(ecological_scenario, TemperedEKI(temperature_schedule=sched), 200, random_key)
# eki_samps = mocat.run(ecological_scenario, TemperedEKI(max_temperature=10.), 200, random_key)


########################################################################################################################

# # Run EKI
# utils.run_eki(ecological_scenario, save_dir, random_key)
#
# # # Run MCMC ABC
# utils.run_abc_mcmc(ecological_scenario, save_dir, random_key)
# #
# # # Run AMC SMC
# utils.run_abc_smc(ecological_scenario, save_dir, random_key)

# ########################################################################################################################

# Params x = (N_0, P, sigma_1, sigma_2, lag, delta)
param_names = (r'$N_0$', r'$P$', r'$\sigma_1$', r'$\sigma_2$', r'$\tau$', r'$\delta$')
plot_ranges = ([0., 3000.], [0., 10.], [0., 5.], [0., 5.], [10., 25.], [0., 1.])

# Plot EKI
utils.plot_eki(ecological_scenario, save_dir, plot_ranges, param_names=param_names,
               # y_range_mult=0.75
               bp_widths=0.1,
               rmse_temp_round=0)

# # Plot ABC-MCMC
# utils.plot_abc_mcmc(ecological_scenario, save_dir, plot_ranges, true_params=true_constrained_params, param_names=param_names)

# Plot ABC-SMC
utils.plot_abc_smc(ecological_scenario, save_dir, plot_ranges,
                   param_names=param_names,
                   trim_thresholds=10,
                   rmse_temp_round=0)

#
# # Plot distances
# utils.plot_dists(ecological_scenario, save_dir)

# Plot resampled distances
n_resamps = 20
utils.plot_res_dists(ecological_scenario, save_dir, n_resamps)
