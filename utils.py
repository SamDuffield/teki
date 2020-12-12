from jax import random, numpy as np
from matplotlib import pyplot as plt
from scipy.stats.kde import gaussian_kde
import pickle
import numpy as onp

import mocat
from mocat import abc

marker_types = ('o', 'X', 'D', '^', 'P', 'p')
line_types = ('-', '--', ':', '-.')


def get_subplot_config(dim):
    subplots_config_by_dim = ((1,),
                              (2,),
                              (3,),
                              (2, 2))
    return subplots_config_by_dim[min(dim, 4) - 1]


def plot_kde(ax, data, xlim, resolution=1000, **kwargs):
    if xlim is None:
        xlim = [np.min(data), np.max(data)]
    linsp = np.linspace(*xlim, resolution)
    dens = gaussian_kde(data)
    ax.plot(linsp, dens(linsp), **kwargs)


def dens_clean_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.xaxis.set_ticks_position('none')
    plt.tight_layout()


def run_eki(scenario, save_dir, random_key, repeat_data=None, simulation_params=None):
    if simulation_params is None:
        simulation_params = mocat.load_CDict(save_dir + '/sim_params.CDict')

    eki_samps_all = onp.zeros((simulation_params.n_repeats,
                               len(simulation_params.n_samps_eki)), dtype='object')
    for i in range(simulation_params.n_repeats):
        int_data = scenario.data if repeat_data is None else repeat_data[i]

        for j, n_eki_single in enumerate(simulation_params.n_samps_eki):
            print(f'Iter={i} - EKI - N={n_eki_single}')
            random_key, _ = random.split(random_key)
            eki_samps = mocat.run_tempered_ensemble_kalman_inversion(scenario,
                                                                     n_eki_single,
                                                                     random_key,
                                                                     data=int_data,
                                                                     max_temp=simulation_params.eki_max_temp)
            eki_samps.repeat_ind = i
            print(f'time = {eki_samps.time}')

            eki_samps_all[i][j] = eki_samps.deepcopy()

    with open(save_dir + '/eki_samps', 'wb') as file:
        pickle.dump(eki_samps_all, file)

    # run single EKI for max_temp = 1
    print(f'EKI posterior - N={np.max(simulation_params.n_samps_eki)}')
    eki_temp_1 = mocat.run_tempered_ensemble_kalman_inversion(scenario,
                                                              np.max(simulation_params.n_samps_eki),
                                                              random_key,
                                                              data=int_data,
                                                              max_temp=1.)

    eki_temp_1.save(save_dir + '/eki_posterior', overwrite=True)


def run_abc_mcmc(scenario, save_dir, random_key, repeat_summarised_data=None, simulation_params=None):
    if simulation_params is None:
        simulation_params = mocat.load_CDict(save_dir + '/sim_params.CDict')

    rwmh_abc_samps_all = onp.zeros((simulation_params.n_repeats,
                                    len(simulation_params.abc_thresholds),
                                    len(simulation_params.rwmh_stepsizes)), dtype='object')
    pre_run_sampler = abc.VanillaABC()
    abc_sampler = abc.RandomWalkABC()
    for i in range(simulation_params.n_repeats):
        if repeat_summarised_data is not None:
            scenario.summary_statistic = repeat_summarised_data[i]
        for j, abc_threshold_single in enumerate(simulation_params.abc_thresholds):
            scenario.threshold = abc_threshold_single
            for k, abc_stepsize in enumerate(simulation_params.rwmh_stepsizes):
                print(
                    f'Iter={i} - ABC RMWH - N={simulation_params.n_samps_rwmh} - dist={abc_threshold_single} - stepsize={abc_stepsize}')
                random_key, pre_run_key = random.split(random_key)

                pre_run_samps = mocat.run_mcmc(scenario,
                                               pre_run_sampler, simulation_params.n_abc_pre_run, pre_run_key)

                abc_sampler.parameters.stepsize = abc_stepsize

                pre_run_distances_concat = pre_run_samps.distance.mean(1)\
                    if pre_run_samps.distance.ndim == 2 else pre_run_samps.distance

                abc_samps = mocat.run_mcmc(scenario,
                                           abc_sampler, simulation_params.n_samps_rwmh, random_key,
                                           initial_state=mocat.CDict(
                                               value=pre_run_samps.value[np.argmin(pre_run_distances_concat)]))
                abc_samps.repeat_ind = i
                abc_samps.threshold = abc_threshold_single
                print(f'time={abc_samps.time}')
                print(f'AR={abc_samps.alpha.mean()}')

                rwmh_abc_samps_all[i][j][k] = abc_samps.deepcopy()
    with open(save_dir + '/abc_rwmh_samps', 'wb') as file:
        pickle.dump(rwmh_abc_samps_all, file)


def run_abc_smc(scenario, save_dir, random_key, repeat_summarised_data=None, simulation_params=None):
    if simulation_params is None:
        simulation_params = mocat.load_CDict(save_dir + '/sim_params.CDict')

    abc_smc_samps_all = onp.zeros((simulation_params.n_repeats,
                                    len(simulation_params.n_samps_abc_smc)), dtype='object')
    for i in range(simulation_params.n_repeats):
        if repeat_summarised_data is not None:
            scenario.summary_statistic = repeat_summarised_data[i]
        for j, n_samps_single in enumerate(simulation_params.n_samps_abc_smc):
            print(f'Iter={i} - ABC SMC - N={n_samps_single}')
            random_key, _ = random.split(random_key)

            abc_samps = mocat.abc.run_abc_smc_sampler(scenario, n_samps_single, random_key,
                                                      mcmc_steps=simulation_params.n_mcmc_steps_abc_smc,
                                                      max_iter=simulation_params.max_iter_abc_smc,
                                                      threshold_quantile_retain=simulation_params.threshold_quantile_retain_abc_smc)
            abc_samps.repeat_ind = i
            print(f'time={abc_samps.time}')
            print(f'{len(abc_samps.threshold_schedule)} schedules, terminating at {abc_samps.threshold_schedule[-1]}')

            abc_smc_samps_all[i][j] = abc_samps.deepcopy()
    with open(save_dir + '/abc_smc_samps', 'wb') as file:
        pickle.dump(abc_smc_samps_all, file)


def plot_eki(scenario, save_dir, ranges, true_params=None, param_names=None, y_range_mult2=None, rmse_temp_round=1):
    with open(save_dir + '/eki_samps', 'rb') as file:
        eki_samps_all = pickle.load(file)

    simulation_params = mocat.load_CDict(save_dir + '/sim_params.CDict')

    subplots_config = get_subplot_config(scenario.dim)

    # Plot EKI densities up to max_temp
    eki_samps_single = eki_samps_all[0, -1]
    fig_eki, axes_eki = plt.subplots(*subplots_config)
    rav_axes_eki = np.ravel(axes_eki)
    for i in range(min(scenario.dim, 4)):
        rav_axes_eki[i].set_yticks([])
        if param_names is not None:
            rav_axes_eki[i].set_xlabel(param_names[i])
        for j in range(len(eki_samps_all[0, -1].temperature_schedule) - 1, -1, -1):
            samps = scenario.constrain(eki_samps_single.value[j, :, i]) \
                if hasattr(scenario, 'constrain') else eki_samps_single.value[j, :, i]
            plot_kde(rav_axes_eki[i], samps, ranges[i], linewidth=2.,
                     color='blue',
                     alpha=0.3,
                     )
            if true_params is not None:
                rav_axes_eki[i].axvline(true_params[i], c='red')
        dens_clean_ax(rav_axes_eki[i])
    leg = rav_axes_eki[1].legend([f'{float(a):.1f}' for a in eki_samps_single.temperature_schedule[::-1]],
                                 frameon=False, handlelength=0, handletextpad=0,
                                 prop={'size': 8})
    leg.set_title(title='Temperatures', prop={'size': 8})
    fig_eki.tight_layout()
    fig_eki.savefig(save_dir + '/EKI_densities', dpi=300)

    # Plot EKI densities at 1 and max_temp
    post_samps = mocat.load_CDict(save_dir + '/eki_posterior.CDict')
    fig_eki, axes_eki = plt.subplots(*subplots_config)
    rav_axes_eki = np.ravel(axes_eki)
    for i in range(min(scenario.dim, 4)):
        rav_axes_eki[i].set_yticks([])
        if param_names is not None:
            rav_axes_eki[i].set_xlabel(param_names[i])
        post_samps_i = scenario.constrain(post_samps.value[-1, :, i]) \
            if hasattr(scenario, 'constrain') else post_samps.value[-1, :, i]
        plot_kde(rav_axes_eki[i], post_samps_i, ranges[i], linewidth=2.,
                 color='blue',
                 alpha=0.3,
                 label=f'{1.0:.1f}'
                 )

        optim_samps_i = scenario.constrain(eki_samps_single.value[-1, :, i]) \
            if hasattr(scenario, 'constrain') else eki_samps_single.value[-1, :, i]
        plot_kde(rav_axes_eki[i], optim_samps_i, ranges[i], linewidth=2.,
                 color='blue',
                 alpha=1.0,
                 label=f'{float(simulation_params.eki_max_temp):.1f}'
                 )
        if true_params is not None:
            rav_axes_eki[i].axvline(true_params[i], c='red')
        yl = rav_axes_eki[i].get_ylim()
        rav_axes_eki[i].set_ylim(yl[0] * y_range_mult2, yl[1] * y_range_mult2)
        dens_clean_ax(rav_axes_eki[i])
    plt.legend(title='Temperature', frameon=False)
    fig_eki.tight_layout()
    fig_eki.savefig(save_dir + '/EKI_densities_post', dpi=300)

    if true_params is not None:
        # Plot convergence in RMSE
        num_simulations = onp.zeros((len(simulation_params.n_samps_eki), simulation_params.n_repeats), dtype='object')
        rmses = onp.zeros_like(num_simulations)
        temp_scheds = onp.zeros_like(num_simulations)

        for n_samp_ind in range(len(simulation_params.n_samps_eki)):
            for repeat_ind in range(simulation_params.n_repeats):
                samps = eki_samps_all[repeat_ind, n_samp_ind]
                ts = samps.temperature_schedule
                num_sims_single = onp.zeros_like(ts)
                rmses_single = onp.zeros_like(ts)
                for temp_ind in range(len(ts)):
                    num_sims_single[temp_ind] = (temp_ind + 1) * n_samp_ind
                    rmses_single[temp_ind] = np.sqrt(np.square(scenario.constrain(samps.value[temp_ind])
                                                               - true_params).mean())
                num_simulations[n_samp_ind, repeat_ind] = num_sims_single.copy()
                rmses[n_samp_ind, repeat_ind] = rmses_single.copy()
                temp_scheds[n_samp_ind, repeat_ind] = onp.array(ts).copy()

        fig, ax = plt.subplots()
        for n_samp_ind in range(len(simulation_params.n_samps_eki)):
            all_ts = onp.concatenate([list(a) for a in temp_scheds[n_samp_ind]])
            all_rmses = onp.concatenate([list(a) for a in rmses[n_samp_ind] if not np.all(a == 0)])
            keep_inds = ~onp.isnan(all_rmses)
            all_ts = all_ts[keep_inds]
            all_rmses = all_rmses[keep_inds]

            all_ts_round = np.round(all_ts, rmse_temp_round)
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


def plot_abc_mcmc(scenario, save_dir, ranges, true_params=None, param_names=None):
    with open(save_dir + '/abc_rwmh_samps', 'rb') as file:
        rwmh_abc_samps_all = pickle.load(file)

    simulation_params = mocat.load_CDict(save_dir + '/sim_params.CDict')

    subplots_config = get_subplot_config(scenario.dim)

    # Plot alpha matrix
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
        fig_abci, axes_abci = plt.subplots(*subplots_config)
        rav_axes_abci = np.ravel(axes_abci)
        for i in range(min(scenario.dim, 4)):
            rav_axes_abci[i].set_yticks([])
            rav_axes_abci[i].set_xlabel(param_names[i])
            for thresh_ind in range(num_thresh_plot):
                samps = scenario.constrain(rwmh_abc_samps_all[0, thresh_ind, ss_ind].value[:, i]) \
                    if hasattr(scenario, 'constrain') else rwmh_abc_samps_all[0, thresh_ind, ss_ind].value[:, i]
                plot_kde(rav_axes_abci[i], samps, ranges[i], color='green',
                         alpha=0.3 + 0.7 * thresh_ind / len(simulation_params.rwmh_stepsizes),
                         label=str(simulation_params.abc_thresholds[thresh_ind]))
                if true_params is not None:
                    rav_axes_abci[i].axvline(true_params[i], c='red')
            dens_clean_ax(rav_axes_abci[i])
        fig_abci.suptitle(f'Stepsize: {ss}')
        plt.legend(title='Threshold', frameon=False)
        fig_abci.tight_layout()
        fig_abci.savefig(save_dir + f'/abc_densities_stepsize{ss}'.replace('.', '_'), dpi=300)

    if true_params is not None:
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
                        samps = rwmh_abc_samps_all[repeat_ind,
                                                   thresh_ind,
                                                   stepsize_int].value[:int(n_samps_rwmh_range[n_samp_ind])]
                        samps = scenario.constrain(samps) \
                            if hasattr(scenario, 'constrain') else samps

                        rmses[stepsize_int, n_samp_ind, repeat_ind, thresh_ind] \
                            = np.sqrt(np.square(samps - true_params).mean())

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


# def plot_abc_smc(scenario, save_dir, ranges, true_params=None, param_names=None, y_range_mult2=None):
#     with open(save_dir + '/abc_smc_samps', 'rb') as file:
#         abc_smc_samps_all = pickle.load(file)
#
#     simulation_params = mocat.load_CDict(save_dir + '/sim_params.CDict')
#
#     subplots_config = get_subplot_config(scenario.dim)
#
#     # Plot ABC densities
#     num_thresh_plot = np.minimum(4, len(abc_smc_samps_all[0, -1].threshold_schedule))
#     for thresh_ind in range(num_thresh_plot):
#
#         ss = round(float(simulation_params.rwmh_stepsizes[ss_ind]), 4)
#         fig_abci, axes_abci = plt.subplots(*subplots_config)
#         rav_axes_abci = np.ravel(axes_abci)
#         for i in range(min(scenario.dim, 4)):
#             rav_axes_abci[i].set_yticks([])
#             rav_axes_abci[i].set_xlabel(param_names[i])
#             for thresh_ind in range(num_thresh_plot):
#                 samps = scenario.constrain(rwmh_abc_samps_all[0, thresh_ind, ss_ind].value[:, i]) \
#                     if hasattr(scenario, 'constrain') else rwmh_abc_samps_all[0, thresh_ind, ss_ind].value[:, i]
#                 plot_kde(rav_axes_abci[i], samps, ranges[i], color='green',
#                          alpha=0.3 + 0.7 * thresh_ind / len(simulation_params.rwmh_stepsizes),
#                          label=str(simulation_params.abc_thresholds[thresh_ind]))
#                 if true_params is not None:
#                     rav_axes_abci[i].axvline(true_params[i], c='red')
#             dens_clean_ax(rav_axes_abci[i])
#         fig_abci.suptitle(f'Stepsize: {ss}')
#         plt.legend(title='Threshold', frameon=False)
#         fig_abci.tight_layout()
#         fig_abci.savefig(save_dir + f'/abc_densities_stepsize{ss}'.replace('.', '_'), dpi=300)
#
#     if true_params is not None:
#         # RWMH ABC convergence
#         n_samps_rwmh_range = simulation_params.n_samps_rwmh / 10 ** np.arange(3)
#
#         rmses = onp.zeros((len(simulation_params.rwmh_stepsizes),
#                            len(n_samps_rwmh_range),
#                            simulation_params.n_repeats,
#                            len(simulation_params.abc_thresholds)))
#
#         for stepsize_int in range(len(simulation_params.rwmh_stepsizes)):
#             ss = round(float(simulation_params.rwmh_stepsizes[stepsize_int]), 4)
#             fig, ax = plt.subplots()
#             for n_samp_ind in range(len(n_samps_rwmh_range)):
#                 for thresh_ind in range(len(simulation_params.abc_thresholds)):
#                     for repeat_ind in range(simulation_params.n_repeats):
#                         samps = rwmh_abc_samps_all[repeat_ind,
#                                                    thresh_ind,
#                                                    stepsize_int].value[:int(n_samps_rwmh_range[n_samp_ind])]
#                         samps = scenario.constrain(samps) \
#                             if hasattr(scenario, 'constrain') else samps
#
#                         rmses[stepsize_int, n_samp_ind, repeat_ind, thresh_ind] \
#                             = np.sqrt(np.square(samps - true_params).mean())
#
#                 ax.plot(simulation_params.abc_thresholds, rmses[stepsize_int, n_samp_ind].mean(0),
#                         color='green', linestyle=line_types[n_samp_ind],
#                         linewidth=3, alpha=0.6,
#                         label=str(int(n_samps_rwmh_range[n_samp_ind])))
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#             ax.set_xlabel('Threshold')
#             ax.set_ylabel('RMSE')
#             fig.suptitle(f'Stepsize: {ss}')
#             plt.legend(frameon=False)
#             fig.tight_layout()
#             fig.savefig(save_dir + f'/abc_rmseconv_stepsize{ss}'.replace('.', '_'), dpi=300)

