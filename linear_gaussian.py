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

save_dir = f'./simulations/linear_gaussian'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

########################################################################################################################
# Simulation parameters
simulation_params = mocat.CDict()

simulation_params.prior_mean = 0.
simulation_params.prior_sd = np.sqrt(5)
simulation_params.likelihood_mat = 1.
simulation_params.likelihood_sd = 1.
simulation_params.data = 3.
simulation_params.posterior_mean = 5 / 2
simulation_params.posterior_sd = np.sqrt(5 / 6)

# EKI
# Number of samples to generate
simulation_params.n_samps_eki = int(1e5)

# RWMH
simulation_params.n_samps_rwmh = int(1e6)

# ABC distance thresholds
simulation_params.abc_thresholds = np.array([1, 5])

# RWMH stepsizes
simulation_params.rwmh_stepsize = 1e-1


########################################################################################################################


class OneDimLG(abc.scenarios.LinearGaussian):
    def distance_function(self,
                          summarised_simulated_data: np.ndarray) -> float:
        return np.sqrt(np.square(summarised_simulated_data - self.summarised_data).sum())


lg_scenario = OneDimLG(prior_mean=np.ones(1) * simulation_params.prior_mean,
                       prior_covariance=np.eye(1) * simulation_params.prior_sd ** 2,
                       likelihood_matrix=np.eye(1) * simulation_params.likelihood_mat,
                       likelihood_covariance=np.eye(1) * simulation_params.likelihood_sd)
lg_scenario.summarised_data = simulation_params.data

random_key = random.PRNGKey(0)
random_key, eki_sim_key = random.split(random_key)

eki_samps = mocat.run_tempered_ensemble_kalman_inversion(lg_scenario,
                                                         simulation_params.n_samps_eki,
                                                         eki_sim_key,
                                                         data=simulation_params.data)

fig, ax = plt.subplots()
# xlim = [float(np.min(eki_samps.value)), float(np.max(eki_samps.value))]
lw = 3.
alp = 0.5
xlim = [-7., 7.]
linsp = np.linspace(xlim[0], xlim[1], 1000)
prior_dens = norm.pdf(linsp, simulation_params.prior_mean, simulation_params.prior_sd)
ax.plot(linsp, prior_dens, color='darkorange', zorder=1, linewidth=lw, alpha=alp)
for i in range(len(eki_samps.temperature_schedule)-1, -1, -1):
    plot_kde(ax, eki_samps.value[i, :, 0], xlim=xlim, color='royalblue', zorder=0, alpha=alp, linewidth=lw,
             label=f'{float(eki_samps.temperature_schedule[i]):.2f}')
post_dens = norm.pdf(linsp, simulation_params.posterior_mean, simulation_params.posterior_sd)
ax.plot(linsp, post_dens, color='red', zorder=1, linewidth=lw, alpha=alp)
dens_clean_ax(ax)
plt.legend(title='Temperatures', frameon=False, handlelength=0, handletextpad=0)
fig.savefig(save_dir + '/EKI_densities', dpi=300)


abc_samps = onp.zeros(len(simulation_params.abc_thresholds), dtype='object')
abc_sampler = abc.RandomWalkABC(stepsize=simulation_params.rwmh_stepsize)
random_key, abc_sim_key = random.split(random_key)
abc_sim_keys = random.split(abc_sim_key, len(simulation_params.abc_thresholds))
for j in range(len(simulation_params.abc_thresholds)):
    lg_scenario.threshold = float(simulation_params.abc_thresholds[j])
    abc_samps[j] = mocat.run_mcmc(lg_scenario,
                                  abc_sampler,
                                  simulation_params.n_samps_rwmh,
                                  abc_sim_keys[j],
                                  initial_state=mocat.CDict(value=np.array([simulation_params.posterior_mean])))

fig, ax = plt.subplots()
ax.plot(linsp, prior_dens, color='darkorange', zorder=1, linewidth=lw, alpha=alp)
ax.plot(linsp, post_dens, color='red', zorder=1, linewidth=lw, alpha=alp)
for j in range(len(simulation_params.abc_thresholds)):
    plot_kde(ax, abc_samps[j].value[:, 0], xlim=xlim, color='forestgreen', zorder=0,
             alpha=0.3 + 0.7*j/len(simulation_params.abc_thresholds), linewidth=lw,
             label=f'{int(simulation_params.abc_thresholds[j])}')
dens_clean_ax(ax)
plt.legend(title='Threshold', frameon=False)

