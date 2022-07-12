########################################################################################################################
# Simulate true g-and-k posterior using numerical inversion + autodiff + standard MCMC
########################################################################################################################

import os
from typing import Any, Union, Callable, Tuple

import jax.lax
from jax import numpy as jnp, random, vmap, grad, jit
from jax.scipy.stats import norm

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

c = 0.8
true_constrained_params = jnp.array([3., 1., 2., 0.5])
true_unconstrained_params = gk_scenario.unconstrain(true_constrained_params)

random_key = random.PRNGKey(0)
repeat_sim_data_keys = random.split(random_key, simulation_params.n_repeats + 1)
random_key = repeat_sim_data_keys[0]

n_ind = 1

full_data = gk_scenario.full_data_sample(true_unconstrained_params, repeat_sim_data_keys[1:][n_ind])
summary_data = gk_scenario.summarise_data(full_data)

param_names = (r'$A$', r'$B$', r'$g$', r'$k$')
plot_ranges = ([2.5, 3.5], [0., 2.], [0., 10.], [0., 3.])


def quantile(u, x):
    z = norm.ppf(u)
    expmingz = jnp.exp(-x[2] * z)
    return x[0] \
           + x[1] * (1 + c * (1 - expmingz) / (1 + expmingz)) \
           * z * (1 + z ** 2) ** x[3]


def bisect_scan(fun: Callable,
                bounds: Union[list, jnp.ndarray],
                max_iter: int = 1000) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    evals = jnp.array([fun(bounds[0]), fun(bounds[1])])
    increasing_bool = evals[1] > evals[0]

    def body_func(int_state: Tuple[jnp.ndarray, jnp.ndarray, int]) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        int_bounds, int_evals, iter_ind = int_state

        new_pos = int_bounds[0] - int_evals[0] * (int_bounds[1] - int_bounds[0]) / (int_evals[1] - int_evals[0])
        new_eval = fun(new_pos)

        replace_upper = jnp.where(increasing_bool, new_eval > 0, new_eval < 0)

        out_bounds = jnp.where(replace_upper, jnp.array([int_bounds[0], new_pos]), jnp.array([new_pos, int_bounds[1]]))
        out_evals = jnp.where(replace_upper, jnp.array([int_evals[0], new_eval]), jnp.array([new_eval, int_evals[1]]))

        return out_bounds, out_evals, iter_ind + 1

    fin_bounds, fin_evals, fin_iter = jax.lax.scan(lambda s, _: (body_func(s), None),
                                                   (bounds, evals, 0),
                                                   jnp.arange(max_iter))[0]

    return fin_bounds, fin_evals, fin_iter


def cdf(y, x):
    # Solve u such that y = quantile(u, x)
    b, e, it = bisect_scan(lambda w: quantile(w, x) - y,
                           jnp.array([1e-5, 1 - 1e-6]))
    return b[e.argmin()]


post_data = summary_data

pdf = grad(cdf)
pdf_all = lambda x: vmap(pdf, in_axes=(0, None))(post_data, x)


def log_pdf(x):
    lp = jnp.log(vmap(pdf, in_axes=(0, None))(post_data, x))
    return jnp.where(lp <= -10, -10, lp).sum()


log_pdf = jit(log_pdf)


class GKInv(mocat.Scenario):
    dim = 4

    def prior_potential(self,
                        x: jnp.ndarray,
                        random_key: jnp.ndarray = None) -> float:
        return jnp.where(jnp.any(x < 0) | jnp.any(x > 10), jnp.inf, 0)

    def likelihood_potential(self,
                             x: jnp.ndarray,
                             random_key: jnp.ndarray = None) -> float:
        return - log_pdf(x)


gk_inv_scen = GKInv()

### Run MCMC
samps = mocat.run(gk_inv_scen,
                  mocat.RandomWalk(stepsize=0.01),
                  100000,
                  random.PRNGKey(0),
                  correction=mocat.RMMetropolis(),
                  initial_state=mocat.cdict(value=true_constrained_params))

print(samps.alpha.mean())
print(samps.value.mean(0))
print(samps.time)

samps.save(save_dir + '/rwmh_samps', overwrite=True)


### Plot

# Plot densities
utils.plot_comp_densities_rwmh(gk_scenario, save_dir, plot_ranges,
                               true_params=true_constrained_params, param_names=param_names,
                               repeat_ind=3, n_ind=n_ind)
