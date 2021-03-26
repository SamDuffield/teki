from typing import Union
from jax import random, numpy as jnp, vmap
from matplotlib import pyplot as plt
from scipy.stats.kde import gaussian_kde
import pickle
import numpy as np

import mocat
from mocat import abc, cdict
from teki import TemperedEKI

marker_types = ('o', 'X', 'D', '^', 'P', 'p')
line_types = ('-', '--', ':', '-.')
num_plot_optim_temps = 10


# def run_eki(scenario, save_dir, random_key, repeat_data=None, simulation_params=None):
#     if simulation_params is None:
#         simulation_params = mocat.load_cdict(save_dir + '/sim_params.cdict')
#
#     eki_samps_all = np.zeros((simulation_params.n_repeats,
#                               len(simulation_params.n_samps_eki),
#                               len(simulation_params.n_steps_eki)), dtype='object')
#
#     eki_optim_all = np.zeros_like(eki_samps_all)
#
#     for i in range(simulation_params.n_repeats):
#         if repeat_data is not None:
#             scenario.data = repeat_data[i]
#
#         for j, n_eki_single in enumerate(simulation_params.n_samps_eki):
#             random_key, samp_key, optim_key = random.split(random_key, 3)
#             for k, n_steps in enumerate(simulation_params.n_steps_eki):
#                 print(f'Iter={i} - EKI - N={n_eki_single}  - n_steps={n_steps}')
#
#                 gamm = 2 ** (1 / n_steps)
#
#                 next_temp = lambda state, extra: jnp.round(gamm ** extra.iter - 1, 4)
#
#                 # temp_mult = 1 + scenario.dim ** -0.5
#                 # first_temp = (1 - temp_mult) / (1 - temp_mult ** n_steps)
#                 #
#                 # next_temp = lambda state, extra: state.temperature + first_temp * temp_mult ** (extra.iter - 1)
#
#                 eki_samp = mocat.run(scenario,
#                                       TemperedEKI(next_temperature=next_temp,
#                                                   max_temperature=1.0),
#                                       n_eki_single,
#                                       samp_key)
#                 eki_samp.repeat_ind = i
#                 print(f'Time = {eki_samp.time} - Nans = {jnp.any(jnp.isnan(eki_samp.value[-1]))}')
#
#                 eki_samps_all[i, j, k] = eki_samp.deepcopy()
#
#                 eki_optim = mocat.run(scenario,
#                                       TemperedEKI(next_temperature=next_temp,
#                                                   term_max_sd=simulation_params.optim_max_sd_eki,
#                                                   max_temperature=simulation_params.max_temp_eki),
#                                       n_eki_single,
#                                       samp_key)
#                 eki_optim.repeat_ind = i
#                 print(f'Time = {eki_optim.time} - Final temperature = {eki_optim.temperature[-1]}'
#                       f'- Nans = {jnp.any(jnp.isnan(eki_optim.value[-1]))}')
#
#                 eki_optim_all[i, j, k] = eki_optim.deepcopy()
#
#     with open(save_dir + '/eki_samp', 'wb') as file:
#         pickle.dump(eki_samps_all, file)
#
#     with open(save_dir + '/eki_optim', 'wb') as file:
#         pickle.dump(eki_optim_all, file)

def run_eki(scenario, save_dir, random_key, repeat_data=None, simulation_params=None, optim_n_samps=False):
    if simulation_params is None:
        simulation_params = mocat.load_cdict(save_dir + '/sim_params.cdict')

    n_samps = simulation_params.vary_n_samps_eki

    eki_samps_n_vary = np.zeros((simulation_params.n_repeats,
                                 len(n_samps)), dtype='object')

    eki_optim_n_vary = np.zeros((simulation_params.n_repeats,
                                 len(n_samps)), dtype='object')

    eki_samps_nsteps_vary = np.zeros(len(simulation_params.vary_n_steps_eki), dtype='object')

    # Vary n
    for i in range(simulation_params.n_repeats):
        if repeat_data is not None:
            scenario.data = repeat_data[i]

        gamm = 2 ** (1 / simulation_params.fix_n_steps)
        next_temp = lambda state, extra: jnp.round(gamm ** extra.iter - 1, 4)
        for j, n_single in enumerate(n_samps):
            random_key, samp_key = random.split(random_key)
            print(f'Iter={i} - EKI samp - N={n_single}')

            eki_samp = mocat.run(scenario,
                                 TemperedEKI(next_temperature=next_temp,
                                             max_temperature=1.0),
                                 n_single,
                                 samp_key)
            eki_samp.repeat_ind = i
            print(f'Time = {eki_samp.time} - Nans = {jnp.any(jnp.isnan(eki_samp.value[-1]))}')
            eki_samps_n_vary[i, j] = eki_samp.deepcopy()

            random_key, samp_key = random.split(random_key)
            print(f'Iter={i} - EKI optim - N={n_single}')

            eki_optim = mocat.run(scenario,
                                  TemperedEKI(next_temperature=next_temp,
                                              term_max_sd=simulation_params.optim_max_sd_eki,
                                              max_temperature=simulation_params.max_temp_eki),
                                  n_single,
                                  samp_key)
            eki_optim.repeat_ind = i
            print(
                f'Time = {eki_optim.time} Final temp = {eki_optim.temperature[-1]}- Nans = {jnp.any(jnp.isnan(eki_optim.value[-1]))}')
            eki_optim_n_vary[i, j] = eki_optim.deepcopy()

    with open(save_dir + '/eki_samps_n_vary', 'wb') as file:
        pickle.dump(eki_samps_n_vary, file)

    with open(save_dir + '/eki_optim_n_vary', 'wb') as file:
        pickle.dump(eki_optim_n_vary, file)

    # Vary n_steps
    for k, n_steps in enumerate(simulation_params.vary_n_steps_eki):
        random_key, samp_key = random.split(random_key)
        print(f'EKI - N={simulation_params.fix_n_samps_eki}  - n_steps={n_steps}')
        gamm = 2 ** (1 / n_steps)
        next_temp = lambda state, extra: jnp.round(gamm ** extra.iter - 1, 4)
        eki_samp = mocat.run(scenario,
                             TemperedEKI(next_temperature=next_temp,
                                         max_temperature=1.0),
                             simulation_params.fix_n_samps_eki,
                             samp_key)
        print(f'Time = {eki_samp.time} - Nans = {jnp.any(jnp.isnan(eki_samp.value[-1]))}')
        eki_samps_nsteps_vary[k] = eki_samp.deepcopy()

    with open(save_dir + '/eki_samps_nsteps_vary', 'wb') as file:
        pickle.dump(eki_samps_nsteps_vary, file)

    # # Optimisation
    # random_key, samp_key = random.split(random_key)
    # print(f'EKI optim - N={simulation_params.fix_n_samps_eki}  - n_steps={simulation_params.fix_n_steps}')
    # gamm = 2 ** (1 / simulation_params.fix_n_steps)
    # next_temp = lambda state, extra: jnp.round(gamm ** extra.iter - 1, 4)
    # eki_optim = mocat.run(scenario,
    #                       TemperedEKI(next_temperature=next_temp,
    #                                   term_max_sd=simulation_params.optim_max_sd_eki,
    #                                   max_temperature=simulation_params.max_temp_eki),
    #                       simulation_params.fix_n_samps_eki,
    #                       samp_key)
    # print(f'Time = {eki_optim.time} - Final temperature = {eki_optim.temperature[-1]}'
    #       f'- Nans = {jnp.any(jnp.isnan(eki_optim.value[-1]))}')
    #
    # with open(save_dir + '/eki_optim', 'wb') as file:
    #     pickle.dump(eki_optim, file)


def run_abc_mcmc(scenario, save_dir, random_key, repeat_summarised_data=None, simulation_params=None):
    if simulation_params is None:
        simulation_params = mocat.load_cdict(save_dir + '/sim_params.cdict')

    rwmh_abc_samps_all = np.zeros(simulation_params.n_repeats, dtype='object')
    pre_run_sampler = abc.VanillaABC()
    abc_sampler = abc.RandomWalkABC()

    n_max = simulation_params.n_samps_abc_mcmc
    for i in range(simulation_params.n_repeats):
        if repeat_summarised_data is not None:
            scenario.data = repeat_summarised_data[i]

        print(f'Iter={i} - ABC RMWH - N={n_max}')
        random_key, pre_run_key = random.split(random_key)

        pre_run_samps = mocat.run(scenario, pre_run_sampler, simulation_params.n_pre_run_abc_mcmc, pre_run_key)

        cut_off_dist = jnp.quantile(pre_run_samps.distance, simulation_params.pre_run_ar_abc_mcmc)
        pre_run_accepted_samps = pre_run_samps.value[pre_run_samps.distance < cut_off_dist]

        abc_sampler.parameters.stepsize = vmap(jnp.cov, (1,))(pre_run_accepted_samps) / scenario.dim * 2.38 ** 2
        abc_sampler.parameters.threshold = cut_off_dist

        abc_samps = mocat.run(scenario,
                              abc_sampler,
                              n_max,
                              random_key,
                              initial_state=mocat.cdict(
                                  value=pre_run_samps.value[jnp.argmin(pre_run_samps.distance)]),
                              correction=abc.RMMetropolisDiagStepsize(
                                  rm_stepsize_scale=simulation_params.rm_stepsize_scale_mcmc,
                                  rm_stepsize_neg_exponent=simulation_params.rm_stepsize_neg_exponent))
        abc_samps.repeat_ind = i
        abc_samps.num_sims = simulation_params.n_pre_run_abc_mcmc + n_max
        print(f'time={abc_samps.time} - AR={abc_samps.alpha.mean()}')

        rwmh_abc_samps_all[i] = abc_samps.deepcopy()

    with open(save_dir + '/abc_mcmc_samps', 'wb') as file:
        pickle.dump(rwmh_abc_samps_all, file)


def run_abc_smc(scenario, save_dir, random_key, repeat_summarised_data=None, simulation_params=None,
                initial_state_extra_generator=None):
    if simulation_params is None:
        simulation_params = mocat.load_cdict(save_dir + '/sim_params.cdict')

    n_samps = simulation_params.vary_n_samps_abc_smc

    abc_smc_samps_all = np.zeros((simulation_params.n_repeats,
                                  len(n_samps)), dtype='object')
    for i in range(simulation_params.n_repeats):
        if repeat_summarised_data is not None:
            scenario.data = repeat_summarised_data[i]
        for j, n_samps_single in enumerate(n_samps):
            print(f'Iter={i} - ABC SMC - N={n_samps_single}')
            random_key, init_sim_key, samp_key = random.split(random_key, 3)

            if initial_state_extra_generator is None:
                initial_state = None
                initial_extra = None
                extra_sims = 0
            else:
                initial_state, initial_extra, extra_sims = initial_state_extra_generator(scenario, n_samps_single,
                                                                                         init_sim_key)
                extra_sims = extra_sims - n_samps_single

            abc_samps = mocat.run(scenario,
                                  abc.MetropolisedABCSMCSampler(max_iter=simulation_params.max_iter_abc_smc,
                                                                ess_threshold_retain=simulation_params.threshold_quantile_retain_abc_smc,
                                                                ess_threshold_resample=simulation_params.threshold_quantile_resample_abc_smc,
                                                                termination_alpha=simulation_params.termination_alpha),
                                  n_samps_single,
                                  samp_key,
                                  initial_state=initial_state, initial_extra=initial_extra)

            alive_particles = (abc_samps.log_weight[:-1] > -jnp.inf).sum(1)
            num_sims = jnp.where(
                alive_particles > (n_samps_single * simulation_params.threshold_quantile_resample_abc_smc),
                alive_particles, n_samps_single)

            abc_samps.repeat_ind = i
            abc_samps.num_sims = extra_sims + num_sims.sum()
            print(f'time={abc_samps.time}')
            print(f'{len(abc_samps.threshold)} thresholds, terminating at {abc_samps.threshold[-1]}')
            print(f'Num Simulations={abc_samps.num_sims}')

            abc_smc_samps_all[i][j] = abc_samps.deepcopy()
    with open(save_dir + '/abc_smc_samps', 'wb') as file:
        pickle.dump(abc_smc_samps_all, file)


#########################################
def get_subplot_config(dim):
    subplots_config_by_dim = ((1,),
                              (2,),
                              (3,),
                              (2, 2))
    return subplots_config_by_dim[min(dim, 4) - 1]


def plot_kde(ax, data, xlim=None, resolution=1000, **kwargs):
    if xlim is None:
        xlim = [jnp.min(data), jnp.max(data)]
    linsp = jnp.linspace(*xlim, resolution)
    dens = gaussian_kde(data)
    ax.plot(linsp, dens(linsp), **kwargs)


def dens_clean_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    # ax.xaxis.set_ticks_position('none')
    plt.tight_layout()


def plot_densities(scenario, vals, true_params, param_names, ranges, y_range_mult, alpha, color, labels,
                   **kwargs):
    subplots_config = get_subplot_config(scenario.dim)

    n_lines = len(vals)

    if np.array(alpha).size == 1:
        alpha = np.repeat(alpha, n_lines)

    if np.array(color).size == 1:
        color = np.repeat(color, n_lines)

    if np.array(labels).size == 1:
        labels = np.repeat(labels, n_lines)

    fig, axes = plt.subplots(*subplots_config)
    rav_axes = np.ravel(axes)
    for i in range(min(scenario.dim, 4)):
        rav_axes[i].set_yticks([])
        if param_names is not None:
            rav_axes[i].set_xlabel(param_names[i])
        for j in range(n_lines):
            if isinstance(vals[j], cdict):
                samps = vals[j].value
            else:
                samps = vals[j]

            if hasattr(scenario, 'constrain'):
                samps = scenario.constrain(samps)

            plot_kde(rav_axes[i], samps[:, i], ranges[i], color=color[j], alpha=float(alpha[j]), label=str(labels[j]),
                     **kwargs)
            if true_params is not None:
                rav_axes[i].axvline(true_params[i], c='red')
        if y_range_mult is not None:
            yl = rav_axes[i].get_ylim()
            rav_axes[i].set_ylim(yl[0] * y_range_mult, yl[1] * y_range_mult)
        dens_clean_ax(rav_axes[i])

    fig.tight_layout()
    return fig, axes


def plot_joint_scatters(vals, param_names, true_params=None, lims=None, color='black', **kwargs):
    d = vals.shape[-1]
    fig, axes = plt.subplots(d, d)
    for i in range(d):
        for j in range(d):
            if i > j:
                axes[i, j].scatter(vals[:, i], vals[:, j], color=color, **kwargs)
                if true_params is not None:
                    axes[i, j].scatter(true_params[i], true_params[j], color='red', marker='x')
                if lims is not None:
                    axes[i, j].set_xlim(lims[i])
                    axes[i, j].set_ylim(lims[j])
                axes[i, j].set_xlabel(param_names[i])
                axes[i, j].set_ylabel(param_names[j])

            elif i == j:
                axes[i, j].hist(vals[:, i], bins=50, color=color)
                axes[i, j].set_yticks([])
                axes[i, j].spines['left'].set_visible(False)
                if true_params is not None:
                    axes[i, j].axvline(true_params[i], color='red')
                if lims is not None:
                    axes[i, j].set_xlim(lims[i])
                axes[i, j].set_xlabel(param_names[i])

            axes[i, j].spines['top'].set_visible(False)
            axes[i, j].spines['right'].set_visible(False)

            if i < j:
                axes[i, j].spines['bottom'].set_visible(False)
                axes[i, j].spines['left'].set_visible(False)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
    fig.tight_layout()
    return fig, axes


def plot_joint_contours(vals, param_names, true_params=None, lims=None, cmap='jet', hist_color='black', **kwargs):
    d = vals.shape[-1]
    fig, axes = plt.subplots(d, d)
    lims_not_none = lims is not None
    if not lims_not_none:
        lims = [[vals[:, i].min(), vals[:, i].max()] for i in range(d)]

    for i in range(d):
        for j in range(d):
            if i > j:
                xmin, xmax = lims[i]
                ymin, ymax = lims[j]

                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([vals[:, i], vals[:, j]])
                kernel = gaussian_kde(values)
                f = np.reshape(kernel(positions).T, xx.shape)

                axes[i, j].contourf(xx, yy, f, cmap=cmap, **kwargs)
                if true_params is not None:
                    axes[i, j].scatter(true_params[i], true_params[j], color='red', marker='x')
                if lims_not_none:
                    axes[i, j].set_xlim(lims[i])
                    axes[i, j].set_ylim(lims[j])
                axes[i, j].set_xlabel(param_names[i])
                axes[i, j].set_ylabel(param_names[j])
            elif i == j:
                # axes[i, j].hist(vals[:, i], bins=50, color=hist_color)

                linsp = jnp.linspace(*lims[i], 100)
                dens = gaussian_kde(vals[:, i])
                axes[i, j].plot(linsp, dens(linsp), color=hist_color)
                if true_params is not None:
                    axes[i, j].axvline(true_params[i], color='red')
                axes[i, j].set_yticks([])
                axes[i, j].spines['left'].set_visible(False)
                if lims_not_none:
                    axes[i, j].set_xlim(lims[i])
                axes[i, j].set_xlabel(param_names[i])

            axes[i, j].spines['top'].set_visible(False)
            axes[i, j].spines['right'].set_visible(False)

            if i < j:
                axes[i, j].spines['bottom'].set_visible(False)
                axes[i, j].spines['left'].set_visible(False)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
    fig.tight_layout()
    return fig, axes


def plot_eki(scenario, save_dir, ranges, true_params=None, param_names=None, y_range_mult=1.0, rmse_temp_round=1,
             optim_ranges=None, bp_widths=1.0,
             legend_size=10, legend_ax=0):
    with open(save_dir + '/eki_samps_n_vary', 'rb') as file:
        eki_samp_all = pickle.load(file)

    with open(save_dir + '/eki_samps_nsteps_vary', 'rb') as file:
        eki_samps_nsteps_vary = pickle.load(file)

    with open(save_dir + '/eki_optim_n_vary', 'rb') as file:
        eki_optim_all = pickle.load(file)

    if optim_ranges is None:
        optim_ranges = ranges
    simulation_params = mocat.load_cdict(save_dir + '/sim_params.cdict')

    # Plot sampling densities by n_steps
    n_steps_lines = len(eki_samps_nsteps_vary)
    eki_samp_vals = [a.value[-1] for a in eki_samps_nsteps_vary]

    samp_steps_fig, samp_steps_axes = plot_densities(scenario, eki_samp_vals,
                                                     true_params=true_params, param_names=param_names,
                                                     ranges=ranges, y_range_mult=y_range_mult,
                                                     labels=simulation_params.vary_n_steps_eki,
                                                     color='blue',
                                                     # alpha=0.2 + 0.8 * np.arange(n_steps_lines) / n_steps_lines,
                                                     alpha=0.8 - 0.7 * np.arange(n_steps_lines) / n_steps_lines,
                                                     linewidth=2.5)
    leg_ax = samp_steps_axes if scenario.dim == 1 else samp_steps_axes.ravel()[legend_ax]
    handles, labels = leg_ax.get_legend_handles_labels()
    leg = leg_ax.legend(handles[::-1], labels[::-1], frameon=False, prop={'size': legend_size})
    leg.set_title(title='Number of Steps', prop={'size': legend_size})
    samp_steps_fig.savefig(save_dir + '/EKI_temp1_varynsteps', dpi=300)

    # Plot optimisation by temperature
    # Boxplot
    eki_optim_single = eki_optim_all[0, -1]
    n_optim_temps = 7
    n_optim_inds = np.linspace(0., len(eki_optim_single.temperature) - 1, n_optim_temps, endpoint=True, dtype='int')
    temps = eki_optim_single.temperature[n_optim_inds]

    optim_vals = eki_optim_single.value[n_optim_inds]
    if hasattr(scenario, 'constrain'):
        optim_vals = scenario.constrain(optim_vals)

    optim_bp_fig, optim_bp_ax = plt.subplots(scenario.dim)
    optim_bp_ax_rav = np.ravel(optim_bp_ax)
    for i in range(min(scenario.dim, 4)):
        optim_bp_ax_rav[i].boxplot(optim_vals[:, :, i], positions=np.round(temps, 1), sym='', zorder=1,
                                   widths=bp_widths)
        if true_params is not None:
            optim_bp_ax_rav[i].axhline(true_params[i], color='red', zorder=0, alpha=0.5)
        if i != min(scenario.dim, 4) - 1:
            optim_bp_ax_rav[i].set_xticks([])
            optim_bp_ax_rav[i].spines['top'].set_visible(False)
            optim_bp_ax_rav[i].spines['right'].set_visible(False)
            optim_bp_ax_rav[i].spines['bottom'].set_visible(False)
        optim_bp_ax_rav[i].set_ylabel(param_names[i])
        optim_bp_ax_rav[i].yaxis.set_tick_params(labelsize=8)
        optim_bp_ax_rav[i].set_xlim(temps[0] - bp_widths, temps[-1] + bp_widths)

    optim_bp_ax_rav[-1].xaxis.set_tick_params(labelsize=8)
    optim_bp_ax_rav[-1].set_xlabel('Temperature')
    optim_bp_ax_rav[-1].spines['top'].set_visible(False)
    optim_bp_ax_rav[-1].spines['right'].set_visible(False)

    optim_bp_fig.tight_layout()
    optim_bp_fig.savefig(save_dir + '/EKI_temp1_varytemps_bp', dpi=300)

    # Densities
    # n_temps = len(eki_optim_single.temperature)
    temp_labs = [f'{float(a):.1f}' for a in temps]
    samp_temps_fig, samp_temps_axes = plot_densities(scenario, eki_optim_single.value[n_optim_inds],
                                                     true_params=true_params, param_names=param_names,
                                                     ranges=optim_ranges, y_range_mult=1.0,
                                                     labels=temp_labs,
                                                     color='blue',
                                                     # alpha=0.3 + 0.7 * np.arange(n_temps) / n_temps,
                                                     alpha=0.8 - 0.7 * np.arange(n_optim_temps) / n_optim_temps,
                                                     linewidth=2)
    leg_ax = samp_temps_axes if scenario.dim == 1 else samp_temps_axes.ravel()[legend_ax]
    handles, labels = leg_ax.get_legend_handles_labels()
    leg = samp_temps_fig.legend(handles[::-1], labels[::-1], frameon=False, loc="center right",
                                borderaxespad=0.1)
    leg.set_title(title='Temperature')
    plt.subplots_adjust(right=0.85)
    samp_temps_fig.savefig(save_dir + '/EKI_temp1_varytemps', dpi=300)

    # Joint dims
    samps = eki_samp_all[0, -1].value[-1]
    if np.any(np.isnan(samps)):
        samps = eki_samp_all[0, -2].value[-1]
    if hasattr(scenario, 'constrain'):
        samps = scenario.constrain(samps)
    fig_joint, ax_joint = plot_joint_scatters(samps, param_names, true_params=true_params, color='blue', s=0.5,
                                              lims=ranges)
    fig_joint.savefig(save_dir + '/EKI_joint_dens', dpi=300)
    fig_joint_cont, ax_joint_cont = plot_joint_contours(samps, param_names, true_params=true_params,
                                                        cmap='Blues', hist_color='blue', lims=ranges)
    fig_joint_cont.savefig(save_dir + '/EKI_joint_dens_contour', dpi=300)

    #
    #
    #
    #
    #
    # # Plot EKI densities up to max_temp
    #
    # eki_optim_single = eki_optim_all[0, -1, 0]
    # num_optim_temps = len(eki_optim_single.temperature)
    # opt_inds = np.ceil(np.linspace(0, num_optim_temps - 1, num_plot_optim_temps)).astype('int32')
    #
    # fig_eki, axes_eki = plt.subplots(*subplots_config)
    # rav_axes_eki = jnp.ravel(axes_eki)
    # for i in range(min(scenario.dim, 4)):
    #     rav_axes_eki[i].set_yticks([])
    #     if param_names is not None:
    #         rav_axes_eki[i].set_xlabel(param_names[i])
    #     for j in opt_inds:
    #         samps = scenario.constrain(eki_optim_single.value[j, :, i]) \
    #             if hasattr(scenario, 'constrain') else eki_optim_single.value[j, :, i]
    #         plot_kde(rav_axes_eki[i], samps, ranges[i], linewidth=2.,
    #                  color='blue',
    #                  alpha=0.3,
    #                  )
    #         if true_params is not None:
    #             rav_axes_eki[i].axvline(true_params[i], c='red')
    #     dens_clean_ax(rav_axes_eki[i])
    # leg = rav_axes_eki[1].legend([f'{float(a):.1f}' for a in eki_optim_single.temperature[opt_inds][::-1]],
    #                              frameon=False, handlelength=0, handletextpad=0,
    #                              prop={'size': 8})
    # leg.set_title(title='Temperatures', prop={'size': 8})
    # fig_eki.tight_layout()
    # fig_eki.savefig(save_dir + '/EKI_densities', dpi=300)
    #
    # # Plot EKI densities at 1 and max_temp
    # post_samps = eki_samps_all[0, -1, 0]
    # fig_eki, axes_eki = plt.subplots(*subplots_config)
    # rav_axes_eki = jnp.ravel(axes_eki)
    # for i in range(min(scenario.dim, 4)):
    #     rav_axes_eki[i].set_yticks([])
    #     if param_names is not None:
    #         rav_axes_eki[i].set_xlabel(param_names[i])
    #     post_samps_i = scenario.constrain(post_samps.value[-1, :, i]) \
    #         if hasattr(scenario, 'constrain') else post_samps.value[-1, :, i]
    #     plot_kde(rav_axes_eki[i], post_samps_i, ranges[i], linewidth=2.,
    #              color='blue',
    #              alpha=0.3,
    #              label=f'{1.0:.1f}'
    #              )
    #
    #     optim_samps_i = scenario.constrain(eki_optim_single.value[-1, :, i]) \
    #         if hasattr(scenario, 'constrain') else eki_optim_single.value[-1, :, i]
    #     plot_kde(rav_axes_eki[i], optim_samps_i, ranges[i], linewidth=2.,
    #              color='blue',
    #              alpha=1.0,
    #              label=f'{float(eki_optim_single.temperature[-1]):.1f}')
    #     if true_params is not None:
    #         rav_axes_eki[i].axvline(true_params[i], c='red')
    #     yl = rav_axes_eki[i].get_ylim()
    #     rav_axes_eki[i].set_ylim(yl[0] * y_range_mult2, yl[1] * y_range_mult2)
    #     dens_clean_ax(rav_axes_eki[i])
    # plt.legend(title='Temperature', frameon=False)
    # fig_eki.tight_layout()
    # fig_eki.savefig(save_dir + '/EKI_densities_post', dpi=300)
    #
    # # Plot EKI sampling densities for varying stepsize
    # post_samps_stepsizes = eki_samps_all[0, -1]
    # fig_eki, axes_eki = plt.subplots(*subplots_config)
    # rav_axes_eki = jnp.ravel(axes_eki)
    # for i in range(min(scenario.dim, 4)):
    #     rav_axes_eki[i].set_yticks([])
    #     if param_names is not None:
    #         rav_axes_eki[i].set_xlabel(param_names[i])
    #     for j in range(len(post_samps_stepsizes)):
    #         post_samps_i_j = scenario.constrain(post_samps_stepsizes[j].value[-1, :, i]) \
    #             if hasattr(scenario, 'constrain') else post_samps_stepsizes[j].value[-1, :, i]
    #         plot_kde(rav_axes_eki[i], post_samps_i_j, ranges[i], linewidth=2.,
    #                  color='blue',
    #                  alpha=0.3 + j / len(post_samps_stepsizes),
    #                  label=f'{simulation_params.eki_stepsizes[j]:.2f}'
    #                  )
    #         dens_clean_ax(rav_axes_eki[i])
    #     if true_params is not None:
    #         rav_axes_eki[i].axvline(true_params[i], c='red')
    #
    # plt.legend(title='Stepsize', frameon=False)
    # fig_eki.tight_layout()
    # fig_eki.savefig(save_dir + '/EKI_densities_post_stepsizes', dpi=300)
    #
    # if true_params is not None:
    #     # Plot convergence in RMSE
    #     num_simulations = np.zeros((len(simulation_params.n_samps_eki), simulation_params.n_repeats), dtype='object')
    #     rmses = np.zeros_like(num_simulations)
    #     temp_scheds = np.zeros_like(num_simulations)
    #
    #     for n_samp_ind in range(len(simulation_params.n_samps_eki)):
    #         for repeat_ind in range(simulation_params.n_repeats):
    #             samps = eki_optim_all[repeat_ind, n_samp_ind]
    #             ts = samps.temperature_schedule
    #             num_sims_single = np.zeros_like(ts)
    #             rmses_single = np.zeros_like(ts)
    #             for temp_ind in range(len(ts)):
    #                 num_sims_single[temp_ind] = (temp_ind + 1) * n_samp_ind
    #                 rmses_single[temp_ind] \
    #                     = jnp.sqrt(jnp.square(scenario.constrain(samps.value[temp_ind]) - true_params).mean()) \
    #                     if hasattr(scenario, 'constrain') else \
    #                     jnp.sqrt(jnp.square(samps.value[temp_ind] - true_params).mean())
    #             num_simulations[n_samp_ind, repeat_ind] = num_sims_single.copy()
    #             rmses[n_samp_ind, repeat_ind] = rmses_single.copy()
    #             temp_scheds[n_samp_ind, repeat_ind] = np.array(ts).copy()
    #
    #     fig, ax = plt.subplots()
    #     for n_samp_ind in range(len(simulation_params.n_samps_eki)):
    #         all_ts = np.concatenate([list(a) for a in temp_scheds[n_samp_ind]])
    #         all_rmses = np.concatenate([list(a) for a in rmses[n_samp_ind] if not jnp.all(a == 0)])
    #         keep_inds = ~np.isnan(all_rmses)
    #         all_ts = all_ts[keep_inds]
    #         all_rmses = all_rmses[keep_inds]
    #
    #         all_ts_round = jnp.round(all_ts, rmse_temp_round)
    #         all_ts_round_unique = jnp.unique(all_ts_round)
    #         all_rmse_round = jnp.array([all_rmses[jnp.where(all_ts_round == a)].mean() for a in all_ts_round_unique])
    #
    #         ax.plot(all_ts_round_unique, all_rmse_round, color='blue', linestyle=line_types[n_samp_ind],
    #                 linewidth=3, alpha=0.6,
    #                 label=str(int(simulation_params.n_samps_eki[n_samp_ind])))
    #
    #     ax.set_xlabel('Temperature')
    #     ax.set_ylabel('RMSE')
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     plt.legend(frameon=False, title='N')
    #     fig.tight_layout()
    #     fig.savefig(save_dir + '/EKI_rmseconv', dpi=300)


def plot_abc_mcmc(scenario, save_dir, ranges, true_params=None, param_names=None):
    with open(save_dir + '/abc_mcmc_samps', 'rb') as file:
        rwmh_abc_samps_all = pickle.load(file)

    simulation_params = mocat.load_cdict(save_dir + '/sim_params.cdict')

    subplots_config = get_subplot_config(scenario.dim)

    # Plot alpha matrix
    abc_alphas = jnp.array([[np.mean([a.alpha.mean() for a in b]) for b in c]
                            for c in rwmh_abc_samps_all.transpose((2, 1, 0))])
    fig_abc_alpmat, ax_abc_alpmat = plt.subplots()
    ax_abc_alpmat.imshow(abc_alphas, interpolation=None, cmap='Greens', origin='lower')
    ax_abc_alpmat.set_xticks(jnp.arange(len(simulation_params.abc_thresholds)))
    ax_abc_alpmat.set_xticklabels(simulation_params.abc_thresholds)
    ax_abc_alpmat.set_xlabel('Threshold')
    ax_abc_alpmat.set_yticks(jnp.arange(len(simulation_params.rwmh_stepsizes)))
    ax_abc_alpmat.set_yticklabels(simulation_params.rwmh_stepsizes)
    ax_abc_alpmat.set_ylabel('Stepsize')
    for j in range(len(simulation_params.abc_thresholds)):
        for i in range(len(simulation_params.rwmh_stepsizes)):
            ax_abc_alpmat.text(j, i, f'{abc_alphas[i, j]:.2f}',
                               ha="center", va="center", color="w" if abc_alphas[i, j] > 0.5 else 'darkgreen')
    fig_abc_alpmat.tight_layout()
    fig_abc_alpmat.savefig(save_dir + '/ABC_alpha_mat', dpi=300)

    # Plot ABC densities
    num_thresh_plot = jnp.minimum(4, len(simulation_params.abc_thresholds))
    for ss_ind in range(len(simulation_params.rwmh_stepsizes)):
        ss = round(float(simulation_params.rwmh_stepsizes[ss_ind]), 4)
        fig_abci, axes_abci = plt.subplots(*subplots_config)
        rav_axes_abci = jnp.ravel(axes_abci)
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
        fig_abci.savefig(save_dir + f'/abc_mcmc_densities_stepsize{ss}'.replace('.', '_'), dpi=300)

    if true_params is not None:
        # RWMH ABC convergence
        n_samps_rwmh_range = simulation_params.n_samps_rwmh / 10 ** jnp.arange(3)

        rmses = np.zeros((len(simulation_params.rwmh_stepsizes),
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
                            = jnp.sqrt(jnp.square(samps - true_params).mean())

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
            fig.savefig(save_dir + f'/abc_mcmc_rmseconv_stepsize{ss}'.replace('.', '_'), dpi=300)


def plot_abc_smc(scenario, save_dir, ranges, true_params=None, param_names=None,
                 rmse_temp_round=1, thresh_cut_off=200,
                 y_range_mult=1, trim_thresholds=0, joint_dims=(0, 1),
                 legend_size=10, legend_ax=0, legend_loc='upper left'):
    with open(save_dir + '/abc_smc_samps', 'rb') as file:
        abc_smc_samps_all = pickle.load(file)

    simulation_params = mocat.load_cdict(save_dir + '/sim_params.cdict')

    subplots_config = get_subplot_config(scenario.dim)

    # Plot ABC densities
    single_abc_samps = abc_smc_samps_all[0, -1]

    # Plot threshold
    fig_thresh, ax_thresh = plt.subplots()
    ax_thresh.plot(single_abc_samps.threshold)
    ax_thresh.set_yscale('log')
    fig_thresh.savefig(save_dir + '/abc_smc_threshold', dpi=300)

    # Plot ESS
    fig_ess, ax_ess = plt.subplots()
    ax_ess.plot(single_abc_samps.ess)
    fig_ess.savefig(save_dir + '/abc_smc_ess', dpi=300)

    # Plot acceptance rate
    fig_alpha, ax_alpha = plt.subplots()
    ax_alpha.plot(single_abc_samps.alpha[1:].mean(1))
    fig_alpha.savefig(save_dir + '/abc_smc_alpha', dpi=300)

    threshs = single_abc_samps.threshold[trim_thresholds:]
    num_thresh_plot = min(4, len(threshs))
    thresh_sched_inds = trim_thresholds + np.linspace(0, threshs.size - 1, num_thresh_plot, dtype='int32')

    vals = [single_abc_samps.value[i] for i in thresh_sched_inds]

    thresh_labs = [f'{float(a):.0f}' for a in single_abc_samps.threshold[thresh_sched_inds]] \
        if single_abc_samps.threshold[0] > 10. \
        else [f'{float(a):.1f}' for a in single_abc_samps.threshold[thresh_sched_inds]]

    thresh_dens_fig, thresh_dens_axes = plot_densities(scenario, vals,
                                                       true_params=true_params, param_names=param_names,
                                                       ranges=ranges, y_range_mult=y_range_mult,
                                                       labels=thresh_labs,
                                                       color='green',
                                                       alpha=0.8 - 0.7 * np.arange(num_thresh_plot) / num_thresh_plot,
                                                       linewidth=2.5)
    leg_ax = thresh_dens_axes if scenario.dim == 1 else thresh_dens_axes.ravel()[legend_ax]
    handles, labels = leg_ax.get_legend_handles_labels()
    leg = leg_ax.legend(handles[::-1], labels[::-1], frameon=False, prop={'size': legend_size},
                        loc=legend_loc)
    leg.set_title(title='Threshold', prop={'size': legend_size})
    thresh_dens_fig.savefig(save_dir + '/abc_smc_densities', dpi=300)

    # Joint dims
    samps_cdict = abc_smc_samps_all[0, -1]
    samps = samps_cdict.value[-1][samps_cdict.log_weight[-1] > -jnp.inf]
    if hasattr(scenario, 'constrain'):
        samps = scenario.constrain(samps)

    fig_joint, ax_joint = plot_joint_scatters(samps, param_names, true_params=true_params, color='green', s=0.5,
                                              lims=ranges)
    fig_joint.savefig(save_dir + '/abc_smc_joint_dens', dpi=300)

    fig_joint_cont, ax_joint_cont = plot_joint_contours(samps, param_names, true_params=true_params,
                                                        cmap='Greens', hist_color='green', lims=ranges)
    fig_joint_cont.savefig(save_dir + '/abc_smc_joint_dens_contour', dpi=300)

    #
    # if true_params is not None:
    #     # Plot convergence in RMSE
    #     num_simulations = np.zeros((len(simulation_params.n_samps_abc_smc), simulation_params.n_repeats),
    #                                dtype='object')
    #     rmses = np.zeros_like(num_simulations)
    #     temp_scheds = np.zeros_like(num_simulations)
    #
    #     for n_samp_ind in range(len(simulation_params.n_samps_abc_smc)):
    #         for repeat_ind in range(simulation_params.n_repeats):
    #             samps = abc_smc_samps_all[repeat_ind, n_samp_ind]
    #             ts = samps.threshold_schedule
    #             num_sims_single = np.zeros_like(ts)
    #             rmses_single = np.zeros_like(ts)
    #             for temp_ind in range(len(ts)):
    #                 num_sims_single[temp_ind] = samps.num_sims - (len(ts) - temp_ind - 1) * n_samp_ind
    #                 rmses_single[temp_ind] \
    #                     = jnp.sqrt(jnp.square(scenario.constrain(samps.value[temp_ind]) - true_params).mean()) \
    #                     if hasattr(scenario, 'constrain') else \
    #                     jnp.sqrt(jnp.square(samps.value[temp_ind] - true_params).mean())
    #             num_simulations[n_samp_ind, repeat_ind] = num_sims_single.copy()
    #             rmses[n_samp_ind, repeat_ind] = rmses_single.copy()
    #             temp_scheds[n_samp_ind, repeat_ind] = np.array(ts).copy()
    #
    #     fig, ax = plt.subplots()
    #     for n_samp_ind in range(len(simulation_params.n_samps_abc_smc)):
    #         all_ts = np.concatenate([list(a) for a in temp_scheds[n_samp_ind]])
    #         all_rmses = np.concatenate([list(a) for a in rmses[n_samp_ind] if not jnp.all(a == 0)])
    #         keep_inds = ~np.isnan(all_rmses)
    #         all_ts = all_ts[keep_inds]
    #         all_rmses = all_rmses[keep_inds]
    #
    #         all_ts_round = jnp.round(all_ts, rmse_temp_round)
    #         all_ts_round_unique = jnp.unique(all_ts_round)
    #         all_rmse_round = jnp.array([all_rmses[jnp.where(all_ts_round == a)].mean() for a in all_ts_round_unique])
    #
    #         keep_inds = all_ts_round_unique < thresh_cut_off
    #         all_ts_round_unique = all_ts_round_unique[keep_inds]
    #         all_rmse_round = all_rmse_round[keep_inds]
    #
    #         ax.plot(all_ts_round_unique, all_rmse_round, color='green', linestyle=line_types[n_samp_ind],
    #                 linewidth=3, alpha=0.6,
    #                 label=str(int(simulation_params.n_samps_abc_smc[n_samp_ind])))
    #
    #     ax.set_xlabel('Threshold')
    #     ax.set_ylabel('RMSE')
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     plt.legend(frameon=False, title='N')
    #     fig.tight_layout()
    #     fig.savefig(save_dir + '/abc_smc_rmseconv', dpi=300)


def calculate_rmse(value, true_params, axis=None):
    return jnp.sqrt(jnp.square(value - true_params).mean(axis))


def plot_rmse(scenario, save_dir, true_params):
    simulation_params = mocat.load_cdict(save_dir + '/sim_params.cdict')

    with open(save_dir + '/eki_samps_n_vary', 'rb') as file:
        eki_samp_all = pickle.load(file)

    with open(save_dir + '/eki_optim_n_vary', 'rb') as file:
        eki_optim_all = pickle.load(file)

    with open(save_dir + '/abc_mcmc_samps', 'rb') as file:
        abc_mcmc_samps = pickle.load(file)

    with open(save_dir + '/abc_smc_samps', 'rb') as file:
        abc_smc_samps_all = pickle.load(file)

    n_repeats = simulation_params.n_repeats

    transform = scenario.constrain if hasattr(scenario, 'constrain') else lambda x: x

    eki_rmses = np.array([[calculate_rmse(transform(eki_samp_all[i, j].value[-1]), true_params)
                           for j in range(len(simulation_params.vary_n_samps_eki))]
                          for i in range(n_repeats)])
    eki_num_sims = np.array([[j * simulation_params.fix_n_steps
                              for j in simulation_params.vary_n_samps_eki]
                             for _ in range(n_repeats)])

    eki_optim_rmses = np.array([[calculate_rmse(transform(eki_optim_all[i, j].value[-1]), true_params)
                                 for j in range(len(simulation_params.vary_n_samps_eki))]
                                for i in range(n_repeats)])
    eki_optim_num_sims = np.array([[simulation_params.vary_n_samps_eki[j] * (eki_optim_all[i, j].temperature.size - 1)
                                    for j in range(len(simulation_params.vary_n_samps_eki))]
                                   for i in range(n_repeats)])

    abc_smc_rmses = np.array([[calculate_rmse(transform(abc_smc_samps_all[i, j].value[-1]), true_params)
                               for j in range(len(simulation_params.vary_n_samps_abc_smc))]
                              for i in range(n_repeats)])
    abc_smc_sims = np.array([[abc_smc_samps_all[i, j].num_sims
                              for j in range(len(simulation_params.vary_n_samps_abc_smc))]
                             for i in range(n_repeats)])

    n_min = np.min([simulation_params.vary_n_samps_eki.min(), simulation_params.vary_n_samps_abc_smc.min()])
    n_max = np.max([simulation_params.vary_n_samps_eki.max(), simulation_params.vary_n_samps_abc_smc.max()])
    n_min_log10 = np.log(n_min) / np.log(10)
    n_max_log10 = np.log(n_max) / np.log(10)
    num_points_mcmc = np.maximum(len(simulation_params.vary_n_samps_eki), len(simulation_params.vary_n_samps_abc_smc))
    n_mcmc = np.asarray(10 ** np.linspace(n_min_log10, n_max_log10, num_points_mcmc, endpoint=True), dtype='int')

    abc_mcmc_n_rmses = np.array([[calculate_rmse(transform(abc_mcmc_samps[i].value[:j]), true_params)
                                  for j in n_mcmc]
                                 for i in range(n_repeats)])

    eki_n_samps_repeat = jnp.repeat(simulation_params.vary_n_samps_eki[jnp.newaxis], n_repeats, 0)
    mcmc_n_samps_repeat = jnp.repeat(n_mcmc[jnp.newaxis], n_repeats, 0)
    smc_n_samps_repeat = jnp.repeat(simulation_params.vary_n_samps_abc_smc[jnp.newaxis], n_repeats, 0)

    fig_n, ax_n = plt.subplots()
    ax_n.scatter(eki_n_samps_repeat, eki_rmses, color='blue', label='TEKI')
    ax_n.scatter(eki_n_samps_repeat, eki_optim_rmses, color='aqua', label='TEKI Optim')
    ax_n.scatter(mcmc_n_samps_repeat, abc_mcmc_n_rmses, color='orange', label='ABC-MCMC')
    ax_n.scatter(smc_n_samps_repeat, abc_smc_rmses, color='green', label='ABC-SMC')
    ax_n.set_xscale('log')
    ax_n.set_xlim([10 ** np.floor(n_min_log10), 10 ** np.ceil(n_max_log10)])
    ax_n.spines['top'].set_visible(False)
    ax_n.spines['right'].set_visible(False)
    ax_n.set_xlabel(r'$N$')
    ax_n.set_ylabel('RMSE')
    ax_n.legend(frameon=False)
    fig_n.savefig(save_dir + '/rmse_n_scatter', dpi=300)

    nsim_min = np.min([eki_num_sims.min(), abc_smc_sims.min()])
    nsim_max = np.max([eki_num_sims.max(), abc_smc_sims.max()])
    nsim_max = np.minimum(nsim_max, simulation_params.n_samps_abc_mcmc)
    nsim_min_log10 = np.log(nsim_min) / np.log(10)
    nsim_max_log10 = np.log(nsim_max) / np.log(10)
    nsim_mcmc = np.asarray(10 ** np.linspace(nsim_min_log10, nsim_max_log10, num_points_mcmc, endpoint=True),
                           dtype='int')

    abc_mcmc_nsims_rmses = np.array(
        [[calculate_rmse(transform(abc_mcmc_samps[i].value[:(j - simulation_params.n_pre_run_abc_mcmc)]), true_params)
          for j in nsim_mcmc]
         for i in range(n_repeats)])

    abc_mcmc_sims = np.array([[j + simulation_params.n_pre_run_abc_mcmc
                               for j in nsim_mcmc]
                              for _ in range(n_repeats)])

    fig_nsims, ax_nsims = plt.subplots()
    ax_nsims.scatter(eki_num_sims, eki_rmses, color='blue', label='TEKI')
    ax_nsims.scatter(eki_optim_num_sims, eki_optim_rmses, color='aqua', label='TEKI Optim')
    ax_nsims.scatter(abc_mcmc_sims, abc_mcmc_nsims_rmses, color='orange', label='ABC-MCMC')
    ax_nsims.scatter(abc_smc_sims, abc_smc_rmses, color='green', label='ABC-SMC')
    ax_nsims.set_xscale('log')
    ax_nsims.set_xlim([10 ** np.floor(nsim_min_log10), 10 ** np.ceil(nsim_max_log10)])
    ax_nsims.spines['top'].set_visible(False)
    ax_nsims.spines['right'].set_visible(False)
    ax_nsims.set_xlabel('Likelihood Simulations')
    ax_nsims.set_ylabel('RMSE')
    ax_nsims.legend(frameon=False)
    fig_nsims.savefig(save_dir + '/rmse_nsims_scatter', dpi=300)


def plot_dists(scenario, save_dir, repeat_summarised_data=None):
    simulation_params = mocat.load_cdict(save_dir + '/sim_params.cdict')

    with open(save_dir + '/eki_samps_n_vary', 'rb') as file:
        eki_samp_all = pickle.load(file)

    with open(save_dir + '/abc_mcmc_samps', 'rb') as file:
        abc_mcmc_samps = pickle.load(file)

    with open(save_dir + '/abc_smc_samps', 'rb') as file:
        abc_smc_samps_all = pickle.load(file)

    n_repeats = simulation_params.n_repeats

    eki_dists = np.zeros((n_repeats, len(simulation_params.vary_n_samps_eki)))
    for i in range(n_repeats):
        if repeat_summarised_data is not None:
            scenario.data = repeat_summarised_data[i]
        for j in range(len(simulation_params.vary_n_samps_eki)):
            if not jnp.any(jnp.isnan(eki_samp_all[i, j].simulated_data[-1])):
                eki_dists[i, j] = vmap(scenario.distance_function)(eki_samp_all[i, j].simulated_data[-1]).mean()
            else:
                eki_dists[i, j] = np.nan

    eki_num_sims = np.array([[j * simulation_params.fix_n_steps
                              for j in simulation_params.vary_n_samps_eki]
                             for _ in range(n_repeats)])

    abc_smc_dists = np.array([[abc_smc_samps_all[i, j].distance[-1].mean()
                               for j in range(len(simulation_params.vary_n_samps_abc_smc))]
                              for i in range(n_repeats)])
    abc_smc_sims = np.array([[abc_smc_samps_all[i, j].num_sims
                              for j in range(len(simulation_params.vary_n_samps_abc_smc))]
                             for i in range(n_repeats)])

    n_min = np.min([simulation_params.vary_n_samps_eki.min(), simulation_params.vary_n_samps_abc_smc.min()])
    n_max = np.max([simulation_params.vary_n_samps_eki.max(), simulation_params.vary_n_samps_abc_smc.max()])
    n_min_log10 = np.log(n_min) / np.log(10)
    n_max_log10 = np.log(n_max) / np.log(10)
    num_points_mcmc = np.maximum(len(simulation_params.vary_n_samps_eki), len(simulation_params.vary_n_samps_abc_smc))
    n_mcmc = np.asarray(10 ** np.linspace(n_min_log10, n_max_log10, num_points_mcmc, endpoint=True), dtype='int')

    abc_mcmc_n_dists = np.array([[abc_mcmc_samps[i].distance[:j].mean()
                                  for j in n_mcmc]
                                 for i in range(n_repeats)])

    eki_n_samps_repeat = jnp.repeat(simulation_params.vary_n_samps_eki[jnp.newaxis], n_repeats, 0)
    mcmc_n_samps_repeat = jnp.repeat(n_mcmc[jnp.newaxis], n_repeats, 0)
    smc_n_samps_repeat = jnp.repeat(simulation_params.vary_n_samps_abc_smc[jnp.newaxis], n_repeats, 0)

    fig_n, ax_n = plt.subplots()
    ax_n.scatter(eki_n_samps_repeat, eki_dists, color='blue', label='TEKI')
    ax_n.scatter(mcmc_n_samps_repeat, abc_mcmc_n_dists, color='orange', label='ABC-MCMC')
    ax_n.scatter(smc_n_samps_repeat, abc_smc_dists, color='green', label='ABC-SMC')
    ax_n.set_xscale('log')
    ax_n.spines['top'].set_visible(False)
    ax_n.spines['right'].set_visible(False)
    ax_n.set_xlabel(r'$N$')
    ax_n.set_ylabel('Distance')
    ax_n.legend(frameon=False)
    fig_n.savefig(save_dir + '/distance_n_scatter', dpi=300)

    nsim_min = np.min([eki_num_sims.min(), abc_smc_sims.min()])
    nsim_max = np.max([eki_num_sims.max(), abc_smc_sims.max()])
    nsim_min_log10 = np.log(nsim_min) / np.log(10)
    nsim_max_log10 = np.log(nsim_max) / np.log(10)
    nsim_mcmc = np.asarray(10 ** np.linspace(nsim_min_log10, nsim_max_log10, num_points_mcmc, endpoint=True),
                           dtype='int')

    abc_mcmc_nsims_dists = np.array(
        [[abc_mcmc_samps[i].distance[:(j - simulation_params.n_pre_run_abc_mcmc)].mean()
          for j in nsim_mcmc]
         for i in range(n_repeats)])

    abc_mcmc_sims = np.array([[j + simulation_params.n_pre_run_abc_mcmc
                               for j in nsim_mcmc]
                              for _ in range(n_repeats)])

    fig_nsims, ax_nsims = plt.subplots()
    ax_nsims.scatter(eki_num_sims, eki_dists, color='blue', label='TEKI')
    ax_nsims.scatter(abc_mcmc_sims, abc_mcmc_nsims_dists, color='orange', label='ABC-MCMC')
    ax_nsims.scatter(abc_smc_sims, abc_smc_dists, color='green', label='ABC-SMC')
    ax_nsims.set_xscale('log')
    ax_nsims.spines['top'].set_visible(False)
    ax_nsims.spines['right'].set_visible(False)
    ax_nsims.set_xlabel('Likelihood Simulations')
    ax_nsims.set_ylabel('Distance')
    ax_nsims.legend(frameon=False)
    fig_nsims.savefig(save_dir + '/distance_nsims_scatter', dpi=300)


def plot_res_dists(scenario, save_dir, n_resamps, repeat_summarised_data=None):
    simulation_params = mocat.load_cdict(save_dir + '/sim_params.cdict')

    with open(save_dir + '/eki_samps_n_vary', 'rb') as file:
        eki_samp_all = pickle.load(file)

    with open(save_dir + '/abc_mcmc_samps', 'rb') as file:
        abc_mcmc_samps = pickle.load(file)

    with open(save_dir + '/abc_smc_samps', 'rb') as file:
        abc_smc_samps_all = pickle.load(file)

    n_repeats = simulation_params.n_repeats

    random_key = random.PRNGKey(0)

    def estimate_dist(vals, rk):
        vals = jnp.asarray(vals)
        n_vals = len(vals)
        rks = random.split(rk, n_vals * n_resamps).reshape(n_vals, n_resamps, 2)
        return vmap(lambda i: vmap(lambda j:
                                   scenario.distance_function(scenario.likelihood_sample(vals[i], rks[i, j])
                                                              ))(jnp.arange(n_resamps)).mean()) \
            (jnp.arange(n_vals)).mean()

    eki_dists = np.zeros((n_repeats, len(simulation_params.vary_n_samps_eki)))
    for i in range(n_repeats):
        print(f'EKI dist sims - Iter {i}')
        if repeat_summarised_data is not None:
            scenario.data = repeat_summarised_data[i]
        for j in range(len(simulation_params.vary_n_samps_eki)):
            if not jnp.any(jnp.isnan(eki_samp_all[i, j].simulated_data[-1])):
                random_key, subkey = random.split(random_key)
                eki_dists[i, j] = estimate_dist(eki_samp_all[i, j].value[-1], subkey)
            else:
                eki_dists[i, j] = np.nan

    with open(save_dir + '/eki_est_dists', 'wb') as file:
        pickle.dump(eki_dists, file)

    eki_num_sims = np.array([[j * simulation_params.fix_n_steps
                              for j in simulation_params.vary_n_samps_eki]
                             for _ in range(n_repeats)])

    abc_smc_dists = np.zeros((n_repeats, len(simulation_params.vary_n_samps_abc_smc)))
    for i in range(n_repeats):
        print(f'SMC dist sims - Iter {i}')
        if repeat_summarised_data is not None:
            scenario.data = repeat_summarised_data[i]
        for j in range(len(simulation_params.vary_n_samps_abc_smc)):
            if not jnp.any(jnp.isnan(abc_smc_samps_all[i, j].simulated_data[-1])):
                random_key, subkey = random.split(random_key)
                abc_smc_dists[i, j] = estimate_dist(abc_smc_samps_all[i, j].value[-1], subkey)
            else:
                abc_smc_dists[i, j] = np.nan

    with open(save_dir + '/c', 'wb') as file:
        pickle.dump(abc_smc_dists, file)

    abc_smc_sims = np.array([[abc_smc_samps_all[i, j].num_sims
                              for j in range(len(simulation_params.vary_n_samps_abc_smc))]
                             for i in range(n_repeats)])

    n_min = np.min([simulation_params.vary_n_samps_eki.min(), simulation_params.vary_n_samps_abc_smc.min()])
    n_max = np.max([simulation_params.vary_n_samps_eki.max(), simulation_params.vary_n_samps_abc_smc.max()])
    n_min_log10 = np.log(n_min) / np.log(10)
    n_max_log10 = np.log(n_max) / np.log(10)
    num_points_mcmc = np.maximum(len(simulation_params.vary_n_samps_eki), len(simulation_params.vary_n_samps_abc_smc))
    n_mcmc = np.asarray(10 ** np.linspace(n_min_log10, n_max_log10, num_points_mcmc, endpoint=True), dtype='int')

    # abc_mcmc_n_dists = np.zeros((n_repeats, num_points_mcmc))
    # for i in range(n_repeats):
    #     print(f'MCMC dist sims - Iter {i}')
    #     if repeat_summarised_data is not None:
    #         scenario.data = repeat_summarised_data[i]
    #     sum_dist_ests = 0.
    #     for j in range(num_points_mcmc):
    #         n_prev = n_mcmc[j - 1] if j > 0 else 0
    #         n_new = n_mcmc[j]
    #         n_int = n_new - n_prev
    #         vals_int = abc_mcmc_samps[i].value[n_prev:n_new]
    #         random_key, subkey = random.split(random_key)
    #         sum_dist_ests += estimate_dist(vals_int, subkey) * n_int
    #         abc_mcmc_n_dists[i, j] = sum_dist_ests / n_new
    #
    # with open(save_dir + '/abc_mcmc_est_dists', 'wb') as file:
    #     pickle.dump(abc_mcmc_n_dists, file)

    eki_n_samps_repeat = jnp.repeat(simulation_params.vary_n_samps_eki[jnp.newaxis], n_repeats, 0)
    mcmc_n_samps_repeat = jnp.repeat(n_mcmc[jnp.newaxis], n_repeats, 0)
    smc_n_samps_repeat = jnp.repeat(simulation_params.vary_n_samps_abc_smc[jnp.newaxis], n_repeats, 0)

    # est_dist_y_max = np.maximum(abc_mcmc_n_dists.max(), abc_smc_dists.max())
    est_dist_y_max = abc_smc_dists.max()

    fig_n, ax_n = plt.subplots()
    ax_n.scatter(eki_n_samps_repeat, eki_dists, color='blue', label='TEKI')
    # ax_n.scatter(mcmc_n_samps_repeat, abc_mcmc_n_dists, color='orange', label='ABC-MCMC')
    ax_n.scatter(smc_n_samps_repeat, abc_smc_dists, color='green', label='ABC-SMC')
    ax_n.set_ylim([0, est_dist_y_max])
    ax_n.set_xscale('log')
    ax_n.spines['top'].set_visible(False)
    ax_n.spines['right'].set_visible(False)
    ax_n.set_xlabel(r'$N$')
    ax_n.set_ylabel('Distance')
    ax_n.legend(frameon=False)
    fig_n.savefig(save_dir + '/est_distance_n_scatter', dpi=300)

    abc_mcmc_sims = mcmc_n_samps_repeat + simulation_params.n_pre_run_abc_mcmc

    fig_nsims, ax_nsims = plt.subplots()
    ax_nsims.scatter(eki_num_sims, eki_dists, color='blue', label='TEKI')
    # ax_nsims.scatter(abc_mcmc_sims, abc_mcmc_n_dists, color='orange', label='ABC-MCMC')
    ax_nsims.scatter(abc_smc_sims, abc_smc_dists, color='green', label='ABC-SMC')
    ax_nsims.set_ylim([0, est_dist_y_max])
    ax_nsims.set_xscale('log')
    ax_nsims.spines['top'].set_visible(False)
    ax_nsims.spines['right'].set_visible(False)
    ax_nsims.set_xlabel('Likelihood Simulations')
    ax_nsims.set_ylabel('Distance')
    ax_nsims.legend(frameon=False)
    fig_nsims.savefig(save_dir + '/est_distance_nsims_scatter', dpi=300)
