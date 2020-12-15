########################################################################################################################
# TEKI and ABC posteriors for linear Gaussian example (1 dimensional)
########################################################################################################################

import os

from jax import numpy as np, random
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as onp

import mocat
from mocat import abc

from utils import plot_kde, dens_clean_ax

save_dir = f'./simulations/toy_1d'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

########################################################################################################################
# Simulation parameters
simulation_params = mocat.cdict()

# EKI
# Number of samples to generate
simulation_params.n_samps_eki = int(1e5)

# RWMH
# Number of samples to generate
simulation_params.n_samps_rwmh = int(1e7)

# ABC distance thresholds
simulation_params.abc_thresholds = np.array([0.5, 1, 5])

# RWMH stepsizes
simulation_params.rwmh_stepsize = 1e-0

simulation_params.save(save_dir + '/simulation_params', overwrite=True)

prior_col = 'darkorange'
lik_col = 'brown'
post_col = 'red'

########################################################################################################################

lw = 3.
alp = 0.5


def plot_eki_dens(temp1_samps, optim_temp_samps, optim_temp, xlim=(-7., 7.), y_range_mult=1.):
    fig, ax = plt.subplots()
    plot_kde(ax, temp1_samps, xlim=xlim, color='royalblue', zorder=4, alpha=0.5, linewidth=lw,
             label='1.0')
    plot_kde(ax, optim_temp_samps, xlim=xlim, color='royalblue', zorder=4, alpha=0.9, linewidth=lw,
             label=f'{float(optim_temp):.1f}')

    yl = ax.get_ylim()
    ax.set_ylim(yl[0] * y_range_mult, yl[1] * y_range_mult)
    dens_clean_ax(ax)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], title='Temperatures', frameon=False)
    return fig, ax


def plot_abc_dens(abc_samps, xlim=(-7., 7.), y_range_mult=1.):
    fig, ax = plt.subplots()
    for j in range(len(simulation_params.abc_thresholds)):
        plot_kde(ax, abc_samps[j].value[:, 0], xlim=xlim, color='forestgreen', zorder=0,
                 alpha=0.3 + 0.7 * j / len(simulation_params.abc_thresholds), linewidth=lw,
                 label=f'{float(simulation_params.abc_thresholds[j]):.1f}')
    yl = ax.get_ylim()
    ax.set_ylim(yl[0] * y_range_mult, yl[1] * y_range_mult)
    dens_clean_ax(ax)
    plt.legend(title='Threshold', frameon=False)
    return fig, ax


random_key = random.PRNGKey(0)
random_key, eki_sim_key, abc_sim_key = random.split(random_key, 3)
abc_sim_keys = random.split(abc_sim_key, len(simulation_params.abc_thresholds))

########################################################################################################################
# Gaussian Prior
# Linear Gaussian Likelihood

prior_mean = 0.
prior_sd = np.sqrt(5)
likelihood_mat = 1.
likelihood_sd = 1.
lg_data = 3.
posterior_mean = 5 / 2
posterior_sd = np.sqrt(5 / 6)


class OneDimLG(abc.scenarios.LinearGaussian):
    def distance_function(self,
                          summarised_simulated_data: np.ndarray) -> float:
        return np.sqrt(np.square(summarised_simulated_data - self.summary_statistic).sum())


lg_scenario = OneDimLG(prior_mean=np.ones(1) * prior_mean,
                       prior_covariance=np.eye(1) * prior_sd ** 2,
                       likelihood_matrix=np.eye(1) * likelihood_mat,
                       likelihood_covariance=np.eye(1) * likelihood_sd)
lg_scenario.summary_statistic = lg_data

eki_max_temp = 5.


eki_samps = mocat.run_tempered_ensemble_kalman_inversion(lg_scenario,
                                                         simulation_params.n_samps_eki,
                                                         eki_sim_key,
                                                         data=lg_data)

eki_samps_optim = mocat.run_tempered_ensemble_kalman_inversion(lg_scenario,
                                                               simulation_params.n_samps_eki,
                                                               eki_sim_key,
                                                               data=lg_data,
                                                               max_temp=eki_max_temp)

xlim = (-7, 7)
fig, ax = plot_eki_dens(eki_samps.value[-1, :, 0], eki_samps_optim.value[-1, :, 0],
                        eki_samps_optim.temperature_schedule[-1], xlim=xlim)
fig.savefig(save_dir + '/LG_EKI_densities', dpi=300)

linsp = np.linspace(xlim[0], xlim[1], 1000)
fig, ax = plt.subplots()
prior_dens = norm.pdf(linsp, prior_mean, prior_sd)
ax.plot(linsp, prior_dens, color=prior_col, zorder=1, linewidth=lw, alpha=alp, label='Prior')
lik = norm.pdf(linsp * likelihood_mat, lg_data, likelihood_sd)
ax.plot(linsp, lik, color=lik_col, zorder=1, linewidth=lw, alpha=alp, label='Likelihood')
post_dens = norm.pdf(linsp, posterior_mean, posterior_sd)
ax.plot(linsp, post_dens, color=post_col, zorder=1, linewidth=lw, alpha=alp, label='Posterior')
dens_clean_ax(ax)
plt.legend(frameon=False)
fig.savefig(save_dir + '/LG_true_densities', dpi=300)

abc_samps = onp.zeros(len(simulation_params.abc_thresholds), dtype='object')
abc_sampler = abc.RandomWalkABC(stepsize=simulation_params.rwmh_stepsize)
for j in range(len(simulation_params.abc_thresholds)):
    lg_scenario.threshold = float(simulation_params.abc_thresholds[j])
    abc_samps[j] = mocat.run_mcmc(lg_scenario,
                                  abc_sampler,
                                  simulation_params.n_samps_rwmh,
                                  abc_sim_keys[j],
                                  initial_state=mocat.cdict(value=np.array([posterior_mean])))

fig, ax = plot_abc_dens(abc_samps, xlim)
fig.savefig(save_dir + '/LG_ABC_densities', dpi=300)


times = mocat.cdict(eki=[[eki_samps.temperature_schedule[-1], eki_samps.time],
                         [eki_samps_optim.temperature_schedule[-1], eki_samps_optim.time]],
                    ABC=[[simulation_params.abc_thresholds[i], abc_samps[i].time]
                         for i in range(len(simulation_params.abc_thresholds))])
times.save(save_dir + '/LG_times', overwrite=True)


########################################################################################################################
# Multi-modal Prior
# Linear Gaussian Likelihood

class OneDimMMPriorLGLik(abc.ABCScenario):
    name = 'Multi-modal Prior'

    dim = 1
    prior_means = np.array([-1., 1.]) * 7
    prior_sd = np.sqrt(3)
    likelihood_mat = 1.
    likelihood_sd = 10.
    summary_statistic = 0.

    def prior_sample(self,
                     random_key: np.ndarray) -> np.ndarray:
        norm_key, uniform_key = random.split(random_key)
        return self.prior_means[random.choice(norm_key, 2)] \
               + self.prior_sd * random.normal(random_key, (self.dim,))

    def prior_potential(self,
                        x: np.ndarray) -> float:
        return - np.log(np.exp(-0.5 * np.square((x - self.prior_means) / self.prior_sd)).sum())

    def likelihood_sample(self,
                          x: np.ndarray,
                          random_key: np.ndarray) -> np.ndarray:
        return self.likelihood_mat * x + self.likelihood_sd * random.normal(random_key)

    def likelihood_potential(self,
                             x: np.ndarray,
                             y: np.ndarray) -> float:
        return 0.5 * np.square((y - self.likelihood_mat * x) / self.likelihood_sd)

    def summarise_data(self,
                       data: np.ndarray) -> np.ndarray:
        return data

    def simulate(self,
                 x: np.ndarray,
                 random_key: np.ndarray) -> np.ndarray:
        return self.likelihood_sample(x, random_key)

    def distance_function(self,
                          summarised_simulated_data: np.ndarray) -> float:
        return np.sqrt(np.square(summarised_simulated_data - self.summary_statistic).sum())


mmprior_lg_lik_scenario = OneDimMMPriorLGLik()

eki_max_temp = 5.

eki_samps = mocat.run_tempered_ensemble_kalman_inversion(mmprior_lg_lik_scenario,
                                                         simulation_params.n_samps_eki,
                                                         eki_sim_key,
                                                         data=mmprior_lg_lik_scenario.summary_statistic)

eki_samps_optim = mocat.run_tempered_ensemble_kalman_inversion(mmprior_lg_lik_scenario,
                                                               simulation_params.n_samps_eki,
                                                               eki_sim_key,
                                                               data=mmprior_lg_lik_scenario.summary_statistic,
                                                               max_temp=eki_max_temp)

xlim = (-12, 12)
fig, ax = plot_eki_dens(eki_samps.value[-1, :, 0], eki_samps_optim.value[-1, :, 0],
                        eki_samps_optim.temperature_schedule[-1], xlim=xlim)
fig.savefig(save_dir + '/MMprior_LG_lik_EKI_densities', dpi=300)

linsp = np.linspace(xlim[0], xlim[1], 1000)
fig, ax = plt.subplots()
prior_dens = norm.pdf(linsp, mmprior_lg_lik_scenario.prior_means[0], mmprior_lg_lik_scenario.prior_sd) * 0.5 \
             + norm.pdf(linsp, mmprior_lg_lik_scenario.prior_means[1], mmprior_lg_lik_scenario.prior_sd) * 0.5
ax.plot(linsp, prior_dens, color=prior_col, zorder=1, linewidth=lw, alpha=alp, label='Prior')
lik = norm.pdf(mmprior_lg_lik_scenario.likelihood_mat * linsp, mmprior_lg_lik_scenario.summary_statistic, mmprior_lg_lik_scenario.likelihood_sd)
ax.plot(linsp, lik, color=lik_col, zorder=1, linewidth=lw, alpha=alp, label='Likelihood')
post_dens = prior_dens * lik * 30
ax.plot(linsp, post_dens, color='red', zorder=1, linewidth=lw, alpha=alp, label='Posterior')
dens_clean_ax(ax)
plt.legend(frameon=False)
fig.savefig(save_dir + '/MMprior_LG_lik_true_densities', dpi=300)

abc_samps = onp.zeros(len(simulation_params.abc_thresholds), dtype='object')
abc_sampler = abc.RandomWalkABC(stepsize=simulation_params.rwmh_stepsize)
for j in range(len(simulation_params.abc_thresholds)):
    mmprior_lg_lik_scenario.threshold = float(simulation_params.abc_thresholds[j])
    abc_samps[j] = mocat.run_mcmc(mmprior_lg_lik_scenario,
                                  abc_sampler,
                                  simulation_params.n_samps_rwmh,
                                  abc_sim_keys[j],
                                  initial_state=mocat.cdict(value=np.array([0.])))

fig, ax = plot_abc_dens(abc_samps, xlim)
fig.savefig(save_dir + '/MMprior_LG_lik_ABC_densities', dpi=300)


times = mocat.cdict(eki=[[eki_samps.temperature_schedule[-1], eki_samps.time],
                         [eki_samps_optim.temperature_schedule[-1], eki_samps_optim.time]],
                    ABC=[[simulation_params.abc_thresholds[i], abc_samps[i].time]
                         for i in range(len(simulation_params.abc_thresholds))])
times.save(save_dir + '/MMprior_LG_lik_times', overwrite=True)


########################################################################################################################
# Gaussian Prior
# Multi-modal Likelihood

class OneDimGPriorMMLik(abc.ABCScenario):
    name = 'Multi-modal Likelihood'

    dim = 1
    prior_mean = 0.
    prior_sd = np.sqrt(5)
    likelihood_sd = 3.
    summary_statistic = 5.

    def prior_sample(self,
                     random_key: np.ndarray) -> np.ndarray:
        return self.prior_mean + self.prior_sd * random.normal(random_key, (self.dim,))

    def prior_potential(self,
                        x: np.ndarray) -> float:
        return 0.5 * np.square((x - self.prior_mean) / self.prior_sd).sum()

    def likelihood_sample(self,
                          x: np.ndarray,
                          random_key: np.ndarray) -> np.ndarray:
        return np.abs(x) + self.likelihood_sd * random.normal(random_key)

    def likelihood_potential(self,
                             x: np.ndarray,
                             y: np.ndarray) -> float:
        return 0.5 * np.square((y - np.abs(x)) / self.likelihood_sd)

    def summarise_data(self,
                       data: np.ndarray) -> np.ndarray:
        return data

    def simulate(self,
                 x: np.ndarray,
                 random_key: np.ndarray) -> np.ndarray:
        return self.likelihood_sample(x, random_key)

    def distance_function(self,
                          summarised_simulated_data: np.ndarray) -> float:
        return np.sqrt(np.square(summarised_simulated_data - self.summary_statistic).sum())


mm_lik_scenario = OneDimGPriorMMLik()

eki_max_temp = 20.

eki_samps = mocat.run_tempered_ensemble_kalman_inversion(mm_lik_scenario,
                                                         simulation_params.n_samps_eki,
                                                         eki_sim_key,
                                                         data=mm_lik_scenario.summary_statistic)

eki_samps_optim = mocat.run_tempered_ensemble_kalman_inversion(mm_lik_scenario,
                                                               simulation_params.n_samps_eki,
                                                               eki_sim_key,
                                                               data=mm_lik_scenario.summary_statistic,
                                                               max_temp=eki_max_temp)

xlim = (-12, 12)
fig, ax = plot_eki_dens(eki_samps.value[-1, :, 0], eki_samps_optim.value[-1, :, 0],
                        eki_samps_optim.temperature_schedule[-1], xlim=xlim)
fig.savefig(save_dir + '/MM_lik_EKI_densities', dpi=300)

linsp = np.linspace(xlim[0], xlim[1], 1000)
fig, ax = plt.subplots()
prior_dens = norm.pdf(linsp, mm_lik_scenario.prior_mean, mm_lik_scenario.prior_sd)
ax.plot(linsp, prior_dens, color=prior_col, zorder=1, linewidth=lw, alpha=alp, label='Prior')
lik = norm.pdf(np.abs(linsp), mm_lik_scenario.summary_statistic, mm_lik_scenario.likelihood_sd)
ax.plot(linsp, lik, color=lik_col, zorder=1, linewidth=lw, alpha=alp, label='Likelihood')
post_dens = prior_dens * lik * 20
ax.plot(linsp, post_dens, color='red', zorder=1, linewidth=lw, alpha=alp, label='Posterior')
dens_clean_ax(ax)
plt.legend(frameon=False)
fig.savefig(save_dir + '/MM_lik_true_densities', dpi=300)

abc_samps = onp.zeros(len(simulation_params.abc_thresholds), dtype='object')
abc_sampler = abc.RandomWalkABC(stepsize=simulation_params.rwmh_stepsize)
for j in range(len(simulation_params.abc_thresholds)):
    mm_lik_scenario.threshold = float(simulation_params.abc_thresholds[j])
    abc_samps[j] = mocat.run_mcmc(mm_lik_scenario,
                                  abc_sampler,
                                  simulation_params.n_samps_rwmh,
                                  abc_sim_keys[j],
                                  initial_state=mocat.cdict(value=np.array([0.])))

fig, ax = plot_abc_dens(abc_samps, xlim)
fig.savefig(save_dir + '/MM_lik_ABC_densities', dpi=300)


times = mocat.cdict(eki=[[eki_samps.temperature_schedule[-1], eki_samps.time],
                         [eki_samps_optim.temperature_schedule[-1], eki_samps_optim.time]],
                    ABC=[[simulation_params.abc_thresholds[i], abc_samps[i].time]
                         for i in range(len(simulation_params.abc_thresholds))])
times.save(save_dir + '/MM_lik_times', overwrite=True)

