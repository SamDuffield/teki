########################################################################################################################
# Compare TEKI and ABC techniques for g-and-k distribution (4 dimensional)
########################################################################################################################

import os
import pickle

from jax import numpy as np, random, vmap
import matplotlib.pyplot as plt
import numpy as onp

import mocat
from mocat import abc

from utils import plot_kde, dens_clean_ax

save_dir = f'./simulations/GK'
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
# Number of samples to generate
simulation_params.n_samps_eki = np.array([200, 1000, 5000])

# threshold param
simulation_params.eki_max_temp = 4.0

# ABC MCMC #############################################################################################################
simulation_params.n_samps_rwmh = int(1e5)

# N pre-run
simulation_params.n_abc_pre_run = int(1e3)

# ABC distance thresholds
simulation_params.abc_thresholds = np.array([3, 7, 15, 50])

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


#
# class GKThinOrder(abc.scenarios.GKTransformedUniformPrior):
#     num_thin: int = 100
#     threshold = 5
#     min_ind: int = 0
#     max_ind: int = None
#     max_power: int = 1
#
#     def __setattr__(self, key, value):
#         super().__setattr__(key, value)
#         if key == 'data' and self.max_ind is None:
#             self.max_ind = len(value)
#
#     def summarise_data(self,
#                        data: np.ndarray):
#         order_stats = data.sort()
#         thin_inds = np.linspace(self.min_ind, self.max_ind, self.num_thin, endpoint=False, dtype='int32')
#         thinned_order_stats = order_stats[thin_inds]
#         return np.concatenate(vmap(lambda expo: thinned_order_stats ** expo)(np.arange(1, self.max_power + 1)))
#
#     def distance_function(self,
#                           summarised_simulated_data: np.ndarray) -> float:
#         return np.sqrt(np.square(summarised_simulated_data - self.summary_statistic).sum())


# class GKThinOrder(abc.scenarios.GKTransformedUniformPrior):
#     threshold = 10
#     num_thin: int = 50
#
#     def summarise_data(self,
#                        data: np.ndarray):
#         return np.array([data.mean(),
#                          np.sqrt(np.cov(data)),
#                          np.quantile(data, 0.25),
#                          np.quantile(data, 0.5),
#                          np.quantile(data, 0.75)])
#
#     def distance_function(self,
#                           summarised_simulated_data: np.ndarray) -> float:
#         return np.sqrt(np.square(summarised_simulated_data - self.summary_statistic).sum())


class GKThinOrder(abc.scenarios.GKTransformedUniformPrior):
    num_thin: int = 100
    threshold = 5

    def summarise_data(self,
                       data: np.ndarray):
        order_stats = data.sort()
        thin_inds = np.linspace(0, len(data), self.num_thin, endpoint=False, dtype='int32')
        return order_stats[thin_inds]

    def distance_function(self,
                          summarised_simulated_data: np.ndarray) -> float:
        return np.sqrt(np.square(summarised_simulated_data - self.summary_statistic).sum())

    # def distance_function(self,
    #                       summarised_simulated_data: np.ndarray) -> float:
    #     return np.abs(summarised_simulated_data - self.summary_statistic).sum()


gk_scenario = GKThinOrder()

true_constrained_params = np.array([3., 1., 2., 0.5])
true_unconstrained_params = gk_scenario.unconstrain(true_constrained_params)

random_key = random.PRNGKey(0)
random_key, subkey = random.split(random_key)
repeat_sim_data_keys = random.split(subkey, simulation_params.n_repeats)


def generate_data(rkey):
    sim_data_keys = random.split(rkey, simulation_params.n_data)
    data = vmap(gk_scenario.likelihood_sample, (None, 0))(true_unconstrained_params, sim_data_keys)
    return data


each_data = vmap(generate_data)(repeat_sim_data_keys)
each_summary_statistic = vmap(gk_scenario.summarise_data)(each_data)

########################################################################################################################
# Run EKI
########################################################################################################################
eki_samps_all = onp.zeros((simulation_params.n_repeats,
                           len(simulation_params.n_samps_eki)), dtype='object')
for i in range(simulation_params.n_repeats):
    gk_scenario.data = each_data[i]
    gk_scenario.summary_statistic = each_summary_statistic[i]
    for j, n_eki_single in enumerate(simulation_params.n_samps_eki):
        print(f'Iter={i} - EKI - N={n_eki_single}')
        random_key, _ = random.split(random_key)
        eki_samps = mocat.run_tempered_ensemble_kalman_inversion(gk_scenario,
                                                                 n_eki_single,
                                                                 random_key,
                                                                 data=each_summary_statistic[i],
                                                                 max_temp=simulation_params.eki_max_temp)
        eki_samps.repeat_ind = i
        print(f'time = {eki_samps.time}')

        eki_samps_all[i][j] = eki_samps.deepcopy()
with open(save_dir + '/eki_samps', 'wb') as file:
    pickle.dump(eki_samps_all, file)

# run single for sampling
gk_scenario.data = each_data[0]
gk_scenario.summary_statistic = each_summary_statistic[0]
eki_temp_1 = mocat.run_tempered_ensemble_kalman_inversion(gk_scenario,
                                                          np.max(simulation_params.n_samps_eki),
                                                          random_key,
                                                          data=each_summary_statistic[0],
                                                          max_temp=1.)

########################################################################################################################
# Run RWMH ABC
########################################################################################################################
rwmh_abc_samps_all = onp.zeros((simulation_params.n_repeats,
                                len(simulation_params.abc_thresholds),
                                len(simulation_params.rwmh_stepsizes)), dtype='object')
pre_run_sampler = abc.VanillaABC()
abc_sampler = abc.RandomWalkABC()
for i in range(simulation_params.n_repeats):
    gk_scenario.data = each_data[i]
    gk_scenario.summary_statistic = each_summary_statistic[i]
    for j, abc_threshold_single in enumerate(simulation_params.abc_thresholds):
        gk_scenario.threshold = abc_threshold_single
        for k, abc_stepsize in enumerate(simulation_params.rwmh_stepsizes):
            print(f'Iter={i} - ABC RMWH - d={abc_threshold_single} - stepsize={abc_stepsize}')
            random_key, pre_run_key = random.split(random_key)

            pre_run_samps = mocat.run_mcmc(gk_scenario,
                                           pre_run_sampler, simulation_params.n_abc_pre_run, pre_run_key)

            abc_sampler.parameters.stepsize = abc_stepsize
            abc_samps = mocat.run_mcmc(gk_scenario,
                                       abc_sampler, simulation_params.n_samps_rwmh, random_key,
                                       initial_state=mocat.cdict(
                                           value=pre_run_samps.value[np.argmin(pre_run_samps.distance)]))
            abc_samps.repeat_ind = i
            abc_samps.threshold = abc_threshold_single
            print(f'time = {abc_samps.time}')

            rwmh_abc_samps_all[i][j][k] = abc_samps.deepcopy()
with open(save_dir + '/abc_rwmh_samps', 'wb') as file:
    pickle.dump(rwmh_abc_samps_all, file)

########################################################################################################################
# Plot densities for varying parameters

param_names = (r'$A$', r'$B$', r'$g$', r'$k$')
# ranges = ([0., 10.], [0., 5.], [0., 10.], [0., 5.])
ranges = ([2., 4.], [0., 3.2], [0., 10.], [0., 5.])

# Plot EKI densities up to max_temp
y_range_mult = 1.0
fig_eki, axes_eki = plt.subplots(2, 2)
rav_axes_eki = np.ravel(axes_eki)
for i in range(4):
    rav_axes_eki[i].set_yticks([])
    rav_axes_eki[i].set_xlabel(param_names[i])
    for j in range(len(eki_samps_all[0, -1].temperature_schedule) - 1, -1, -1):
        samps = gk_scenario.constrain(eki_samps_all[0, -1].value[j, :, i])
        plot_kde(rav_axes_eki[i], samps, ranges[i], linewidth=2.,
                 color='blue',
                 alpha=0.3,
                 # alpha=1- 0.7 * j / len(eki_samps_all[0, -1].temperature_schedule),
                 # label=f'{float(eki_samps_all[0, -1].temperature_schedule[j]):.1f}'
                 )
        rav_axes_eki[i].axvline(true_constrained_params[i], c='red')
    yl = rav_axes_eki[i].get_ylim()
    rav_axes_eki[i].set_ylim(yl[0], yl[1] * y_range_mult)
    dens_clean_ax(rav_axes_eki[i])
leg = rav_axes_eki[1].legend([f'{float(a):.1f}' for a in eki_samps_all[0, -1].temperature_schedule[::-1]],
                             frameon=False, handlelength=0, handletextpad=0,
                             prop={'size': 8})
leg.set_title(title='Temperatures', prop={'size': 8})
fig_eki.tight_layout()
fig_eki.savefig(save_dir + '/EKI_densities', dpi=300)

# Plot EKI densities at 1 and max_temp
y_range_mult = 0.2
fig_eki, axes_eki = plt.subplots(2, 2)
rav_axes_eki = np.ravel(axes_eki)
for i in range(4):
    rav_axes_eki[i].set_yticks([])
    rav_axes_eki[i].set_xlabel(param_names[i])
    plot_kde(rav_axes_eki[i], gk_scenario.constrain(eki_temp_1.value[-1, :, i]), ranges[i], linewidth=2.,
             color='blue',
             alpha=0.3,
             label=f'{1.0:.1f}'
             )

    plot_kde(rav_axes_eki[i], gk_scenario.constrain(eki_samps_all[0, -1].value[-1, :, i]), ranges[i], linewidth=2.,
             color='blue',
             alpha=1.0,
             label=f'{float(simulation_params.eki_max_temp):.1f}'
             )
    rav_axes_eki[i].axvline(true_constrained_params[i], c='red')
    yl = rav_axes_eki[i].get_ylim()
    rav_axes_eki[i].set_ylim(yl[0] * y_range_mult, yl[1] * y_range_mult)
    dens_clean_ax(rav_axes_eki[i])
plt.legend(title='Temperature', frameon=False)
fig_eki.tight_layout()
fig_eki.savefig(save_dir + '/EKI_densities_1_mt', dpi=300)


# ABC alpha matrix
abc_alphas = np.array([[onp.mean([a.alpha.mean() for a in b]) for b in c]
                       for c in rwmh_abc_samps_all.transpose((2, 1, 0))])
fig_abc_alpmat, ax_abc_alpmat = plt.subplots()
ax_abc_alpmat.imshow(abc_alphas, interpolation=None, cmap='Greens', origin='lower')
ax_abc_alpmat.set_xticks(np.arange(len(simulation_params.abc_thresholds)))
ax_abc_alpmat.set_xticklabels(simulation_params.abc_thresholds)
ax_abc_alpmat.set_xlabel('Threshold')
ax_abc_alpmat.set_yticks(np.arange(len(simulation_params.rwmh_stepsizes)))
ax_abc_alpmat.set_yticklabels(simulation_params.rwmh_stepsizes)
ax_abc_alpmat.set_ylabel('Stepsize')
for j in range(len(simulation_params.abc_thresholds)):
    for i in range(len(simulation_params.rwmh_stepsizes)):
        ax_abc_alpmat.text(j, i, f'{abc_alphas[i, j]:.2f}',
                           ha="center", va="center", color="w" if abc_alphas[i, j] > 0.5 else 'darkgreen')
fig_abc_alpmat.tight_layout()
fig_abc_alpmat.savefig(save_dir + '/ABC_alpha_mat', dpi=300)

# Plot ABC densities
num_thresh_plot = np.minimum(4, len(simulation_params.abc_thresholds))
for ss_ind in range(len(simulation_params.rwmh_stepsizes)):
    ss = round(float(simulation_params.rwmh_stepsizes[ss_ind]), 4)
    fig_abci, axes_abci = plt.subplots(2, 2)
    rav_axes_abci = np.ravel(axes_abci)
    for i in range(4):
        rav_axes_abci[i].set_yticks([])
        rav_axes_abci[i].set_xlabel(param_names[i])
        for thresh_ind in range(num_thresh_plot):
            samps = gk_scenario.constrain(rwmh_abc_samps_all[0, thresh_ind, ss_ind].value[:, i])
            plot_kde(rav_axes_abci[i], samps, ranges[i], color='green',
                     alpha=0.3 + 0.7 * thresh_ind / len(simulation_params.rwmh_stepsizes),
                     label=str(simulation_params.abc_thresholds[thresh_ind]))
            rav_axes_abci[i].axvline(true_constrained_params[i], c='red')
        dens_clean_ax(rav_axes_abci[i])
    fig_abci.suptitle(f'Stepsize: {ss}')
    plt.legend(title='Threshold', frameon=False)
    fig_abci.tight_layout()
    fig_abci.savefig(save_dir + f'/abc_densities_stepsize{ss}'.replace('.', '_'), dpi=300)

plt.show()
########################################################################################################################
# Plot convergence in RMSE

marker_types = ('o', 'X', 'D', '^', 'P', 'p')
line_types = ('-', '--', ':', '-.')

# EKI convergence
num_simulations = onp.zeros((len(simulation_params.n_samps_eki), simulation_params.n_repeats), dtype='object')
rmses = onp.zeros_like(num_simulations)
temp_scheds = onp.zeros_like(num_simulations)

linsp = np.linspace(0., simulation_params.eki_max_temp, 100)
rmse_kdes = onp.zeros((len(simulation_params.n_samps_eki), simulation_params.n_repeats, len(linsp)))

for n_samp_ind in range(len(simulation_params.n_samps_eki)):
    for repeat_ind in range(simulation_params.n_repeats):
        samps = eki_samps_all[repeat_ind, n_samp_ind]
        ts = samps.temperature_schedule
        num_sims_single = onp.zeros_like(ts)
        rmses_single = onp.zeros_like(ts)
        for temp_ind in range(len(ts)):
            num_sims_single[temp_ind] = (temp_ind + 1) * n_samp_ind
            rmses_single[temp_ind] = np.sqrt(np.square(gk_scenario.constrain(samps.value[temp_ind])
                                                       - true_constrained_params).mean())
        num_simulations[n_samp_ind, repeat_ind] = num_sims_single.copy()
        rmses[n_samp_ind, repeat_ind] = rmses_single.copy()
        temp_scheds[n_samp_ind, repeat_ind] = onp.array(ts).copy()

fig, ax = plt.subplots()
for n_samp_ind in range(len(simulation_params.n_samps_eki)):
    all_ts = onp.concatenate([list(a) for a in temp_scheds[n_samp_ind]])
    all_rmses = onp.concatenate([list(a) for a in rmses[n_samp_ind] if not np.all(a == 0)])
    keep_inds = ~np.isnan(all_rmses)
    all_ts = all_ts[keep_inds]
    all_rmses = all_rmses[keep_inds]

    all_ts_round = np.round(all_ts, 1)
    all_ts_round_unique = np.unique(all_ts_round)
    all_rmse_round = np.array([all_rmses[np.where(all_ts_round == a)].mean() for a in all_ts_round_unique])

    ax.plot(all_ts_round_unique, all_rmse_round, color='blue', linestyle=line_types[n_samp_ind],
            linewidth=3, alpha=0.6,
            label=str(int(simulation_params.n_samps_eki[n_samp_ind])))

ax.set_xlabel('Temperature')
ax.set_ylabel('RMSE')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(frameon=False, title='N')
fig.tight_layout()
fig.savefig(save_dir + '/EKI_rmseconv', dpi=300)

# RWMH ABC convergence
n_samps_rwmh_range = simulation_params.n_samps_rwmh / 10 ** np.arange(3)

rmses = onp.zeros((len(simulation_params.rwmh_stepsizes),
                   len(n_samps_rwmh_range),
                   simulation_params.n_repeats,
                   len(simulation_params.abc_thresholds)))

for stepsize_int in range(len(simulation_params.rwmh_stepsizes)):
    ss = round(float(simulation_params.rwmh_stepsizes[stepsize_int]), 4)
    fig, ax = plt.subplots()
    for n_samp_ind in range(len(n_samps_rwmh_range)):
        for thresh_ind in range(len(simulation_params.abc_thresholds)):
            for repeat_ind in range(simulation_params.n_repeats):
                rmses[stepsize_int, n_samp_ind, repeat_ind, thresh_ind] \
                    = np.sqrt(np.square(gk_scenario.constrain(
                    rwmh_abc_samps_all[repeat_ind,
                                       thresh_ind,
                                       stepsize_int].value[:int(n_samps_rwmh_range[n_samp_ind])])
                                        - true_constrained_params).mean())

        ax.plot(simulation_params.abc_thresholds, rmses[stepsize_int, n_samp_ind].mean(0),
                color='green', linestyle=line_types[n_samp_ind],
                linewidth=3, alpha=0.6,
                label=str(int(n_samps_rwmh_range[n_samp_ind])))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('RMSE')
    fig.suptitle(f'Stepsize: {ss}')
    plt.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_dir + f'/abc_rmseconv_stepsize{ss}'.replace('.', '_'), dpi=300)

plt.show()
