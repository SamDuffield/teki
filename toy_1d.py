########################################################################################################################
# TEKI and ABC posteriors for linear Gaussian example (1 dimensional)
########################################################################################################################

import os
import pickle

from jax import numpy as jnp, random
from jax.scipy.stats import norm
import matplotlib.pyplot as plt

import mocat
from mocat import abc
from teki import TemperedEKI
from utils import plot_kde, dens_clean_ax

save_dir = './simulations/toy_1d'
samples_dir = save_dir + '/samples'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

########################################################################################################################
# Simulation parameters
simulation_params = mocat.cdict()

# EKI
# Number of samples to generate
simulation_params.n_samps_eki = int(1e5)

# Stepsizes for EKI
simulation_params.stepsizes_eki = [0.01]

# ABC
# Number of samples to generate
simulation_params.n_samps_abc = int(1e7)

# Distance thresholds
simulation_params.abc_thresholds = jnp.array([0.5, 1, 5])

simulation_params.save(save_dir + '/simulation_params', overwrite=True)

prior_col = 'darkorange'
lik_col = 'red'
post_col = 'saddlebrown'

########################################################################################################################

lw = 3.
alp = 0.3
random_key = random.PRNGKey(0)


def ensure_len_n(val, n):
    val = jnp.asarray(val)
    if val.size != n:
        val = jnp.repeat(val, n)
    return val


def plot_sample_densities(samps, params, param_title=None, alpha=None, xlim=(-7, 7), y_range_mult=1.,
                          **kwargs):
    n_lines = len(samps)
    if alpha is None:
        alpha = jnp.linspace(alp, 1., n_lines)
    alpha = ensure_len_n(alpha, n_lines)
    fig, ax = plt.subplots()
    for s, p, a in zip(samps, params, alpha):
        plot_kde(ax, s, alpha=a, label=str(p), xlim=xlim, **kwargs)

    yl = ax.get_ylim()
    ax.set_ylim(yl[0] * y_range_mult, yl[1] * y_range_mult)
    dens_clean_ax(ax)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], title=param_title, frameon=False)
    return fig, ax


def run_teki(scen, xlim):
    scen_name = scen.name.replace(' ', '_')

    keys = random.split(random_key, len(simulation_params.stepsizes_eki))
    teki_samps = []
    for s, rk in zip(simulation_params.stepsizes_eki, keys):
        ts = jnp.arange(0, 1. + s, s)
        teki_samps.append(mocat.run(scen,
                                    TemperedEKI(temperature_schedule=ts),
                                    n=simulation_params.n_samps_eki,
                                    random_key=rk))

    with open(samples_dir + f'/{scen_name}_teki', 'wb') as file:
        pickle.dump(teki_samps, file)

    fig, ax = plot_sample_densities([samp.value[-1, :, 0] for samp in teki_samps],
                                    simulation_params.stepsizes_eki, 'Stepsize',
                                    color='royalblue',
                                    linewidth=lw, xlim=xlim)
    fig.savefig(save_dir + f'/{scen_name}_teki', dpi=300)


def run_abc(scen, xlim):
    scen_name = scen.name.replace(' ', '_')

    keys = random.split(random_key, len(simulation_params.abc_thresholds))
    abc_samps = [mocat.run(scen,
                           abc.VanillaABC(threshold=thresh),
                           n=simulation_params.n_samps_abc,
                           random_key=rk) for thresh, rk in zip(simulation_params.abc_thresholds, keys)]
    with open(samples_dir + f'/{scen_name}_abc', 'wb') as file:
        pickle.dump(abc_samps, file)

    fig, ax = plot_sample_densities([samp.value[samp.log_weight > -jnp.inf, 0] for samp in abc_samps],
                                    simulation_params.abc_thresholds, 'Threshold',
                                    color='forestgreen',
                                    linewidth=lw, xlim=xlim)
    fig.savefig(save_dir + f'/{scen_name}_abc', dpi=300)


########################################################################################################################
# Gaussian Prior
# Linear Gaussian Likelihood

prior_mean = 0.
prior_sd = jnp.sqrt(5)
likelihood_mat = 1.
likelihood_sd = 1.
lg_data = 3.
posterior_mean = 5 / 2
posterior_sd = jnp.sqrt(5 / 6)

xlim = (-7, 7)

lg_scenario = abc.scenarios.LinearGaussian(prior_mean=jnp.ones(1) * prior_mean,
                                           prior_covariance=jnp.eye(1) * prior_sd ** 2,
                                           likelihood_matrix=jnp.eye(1) * likelihood_mat,
                                           likelihood_covariance=jnp.eye(1) * likelihood_sd)
lg_scenario.data = lg_data

# Plot true densities
linsp = jnp.linspace(xlim[0], xlim[1], 1000)
fig, ax = plt.subplots()
prior_dens = norm.pdf(linsp, prior_mean, prior_sd)
ax.plot(linsp, prior_dens, color=prior_col, zorder=1, linewidth=lw, alpha=alp, label='Prior')
lik = norm.pdf(linsp * likelihood_mat, lg_data, likelihood_sd)
ax.plot(linsp, lik, color=lik_col, zorder=1, linewidth=lw, alpha=alp, label='Likelihood')
post_dens = norm.pdf(linsp, posterior_mean, posterior_sd)
ax.plot(linsp, post_dens, color=post_col, zorder=1, linewidth=lw, alpha=alp, label='Posterior')
dens_clean_ax(ax)
plt.legend(frameon=False)
fig.savefig(save_dir + f'/{lg_scenario.name.replace(" ", "_")}_truth', dpi=300)

# TEKI
if True:
    run_teki(lg_scenario, xlim=xlim)

# Vanilla ABC
if True:
    run_abc(lg_scenario, xlim=xlim)


########################################################################################################################
# Multi-modal Prior
# Linear Gaussian Likelihood

class OneDimMMPriorLGLik(abc.ABCScenario):
    name = 'Multi-modal Prior'

    dim = 1
    prior_means = jnp.array([-1., 1.]) * 7
    prior_sd = jnp.sqrt(3)
    likelihood_mat = 1.
    likelihood_sd = 10.
    data = 0.

    def prior_sample(self,
                     random_key: jnp.ndarray) -> jnp.ndarray:
        norm_key, uniform_key = random.split(random_key)
        return self.prior_means[random.choice(norm_key, 2)] \
               + self.prior_sd * random.normal(random_key, (self.dim,))

    def prior_potential(self,
                        x: jnp.ndarray,
                        random_key: jnp.ndarray = None) -> float:
        return - jnp.log(jnp.exp(-0.5 * jnp.square((x - self.prior_means) / self.prior_sd)).sum())

    def likelihood_sample(self,
                          x: jnp.ndarray,
                          random_key: jnp.ndarray) -> jnp.ndarray:
        return self.likelihood_mat * x + self.likelihood_sd * random.normal(random_key)

    def likelihood_potential(self,
                             x: jnp.ndarray,
                             random_key: jnp.ndarray = None) -> float:
        return 0.5 * jnp.square((self.data - self.likelihood_mat * x) / self.likelihood_sd)


mmprior_lg_lik_scenario = OneDimMMPriorLGLik()
xlim = (-12, 12)

# Plot truth
linsp = jnp.linspace(xlim[0], xlim[1], 1000)
fig, ax = plt.subplots()
prior_dens = norm.pdf(linsp, mmprior_lg_lik_scenario.prior_means[0], mmprior_lg_lik_scenario.prior_sd) * 0.5 \
             + norm.pdf(linsp, mmprior_lg_lik_scenario.prior_means[1], mmprior_lg_lik_scenario.prior_sd) * 0.5
ax.plot(linsp, prior_dens, color=prior_col, zorder=1, linewidth=lw, alpha=alp, label='Prior')
lik = norm.pdf(mmprior_lg_lik_scenario.likelihood_mat * linsp, mmprior_lg_lik_scenario.data,
               mmprior_lg_lik_scenario.likelihood_sd)
ax.plot(linsp, lik, color=lik_col, zorder=1, linewidth=lw, alpha=alp, label='Likelihood')
post_dens = prior_dens * lik * 30
ax.plot(linsp, post_dens, color=post_col, zorder=1, linewidth=lw, alpha=alp, label='Posterior')
dens_clean_ax(ax)
plt.legend(frameon=False)
fig.savefig(save_dir + f'/{mmprior_lg_lik_scenario.name.replace(" ", "_")}_truth', dpi=300)

# TEKI
if True:
    run_teki(mmprior_lg_lik_scenario, xlim=xlim)

# Vanilla ABC
if True:
    run_abc(mmprior_lg_lik_scenario, xlim=xlim)


########################################################################################################################
# Gaussian Prior
# Multi-modal Likelihood


class OneDimGPriorMMLik(abc.ABCScenario):
    name = 'Multi-modal Likelihood'

    dim = 1
    prior_mean = 0.
    prior_sd = jnp.sqrt(5)
    likelihood_sd = 3.
    data = 5.

    def prior_sample(self,
                     random_key: jnp.ndarray) -> jnp.ndarray:
        return self.prior_mean + self.prior_sd * random.normal(random_key, (self.dim,))

    def prior_potential(self,
                        x: jnp.ndarray,
                        random_key: jnp.ndarray = None) -> float:
        return 0.5 * jnp.square((x - self.prior_mean) / self.prior_sd).sum()

    def likelihood_sample(self,
                          x: jnp.ndarray,
                          random_key: jnp.ndarray) -> jnp.ndarray:
        return jnp.abs(x) + self.likelihood_sd * random.normal(random_key)

    def likelihood_potential(self,
                             x: jnp.ndarray,
                             random_key: jnp.ndarray = None) -> float:
        return 0.5 * jnp.square((self.data - jnp.abs(x)) / self.likelihood_sd)


mm_lik_scenario = OneDimGPriorMMLik()
xlim = (-12, 12)

linsp = jnp.linspace(xlim[0], xlim[1], 1000)
fig, ax = plt.subplots()
prior_dens = norm.pdf(linsp, mm_lik_scenario.prior_mean, mm_lik_scenario.prior_sd)
ax.plot(linsp, prior_dens, color=prior_col, zorder=1, linewidth=lw, alpha=alp, label='Prior')
lik = norm.pdf(jnp.abs(linsp), mm_lik_scenario.data, mm_lik_scenario.likelihood_sd)
ax.plot(linsp, lik, color=lik_col, zorder=1, linewidth=lw, alpha=alp, label='Likelihood')
post_dens = prior_dens * lik * 20
ax.plot(linsp, post_dens, color=post_col, zorder=1, linewidth=lw, alpha=alp, label='Posterior')
dens_clean_ax(ax)
plt.legend(frameon=False)
fig.savefig(save_dir + f'/{mm_lik_scenario.name.replace(" ", "_")}_truth', dpi=300)


# TEKI
if True:
    run_teki(mm_lik_scenario, xlim=xlim)

# Vanilla ABC
if True:
    run_abc(mm_lik_scenario, xlim=xlim)

