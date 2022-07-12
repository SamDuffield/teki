from typing import Union
from jax import random, numpy as jnp, vmap
from matplotlib import pyplot as plt
from scipy.stats.kde import gaussian_kde
from scipy.interpolate import make_interp_spline
import pickle
import numpy as np

import mocat
from mocat import abc, cdict
from teki import TemperedEKI, AdaptiveTemperedEKI

plt.style.use('thesis')

marker_types = ('o', 'X', 'D', '^', 'P', 'p')
line_types = ('-', '--', ':', '-.')
num_plot_optim_temps = 10


# def run_eki(scenario, save_dir, random_key, repeat_data=None, simulation_params=None, optim_n_samps=False):
#     if simulation_params is None:
#         simulation_params = mocat.load_cdict(save_dir + '/sim_params.cdict')
#
#     n_samps = simulation_params.vary_n_samps_eki
#
#     eki_samps_n_vary = np.zeros((simulation_params.n_repeats,
#                                  len(n_samps)), dtype='object')
#
#     eki_optim_n_vary = np.zeros((simulation_params.n_repeats,
#                                  len(n_samps)), dtype='object')
#
#     eki_samps_nsteps_vary = np.zeros(len(simulation_params.vary_n_steps_eki), dtype='object')
#
#     vary_n_key = random_key.copy()
#
#     # Vary n
#     for i in range(simulation_params.n_repeats):
#         if repeat_data is not None:
#             scenario.data = repeat_data[i]
#
#         gamm = 2 ** (1 / simulation_params.fix_n_steps)
#         next_temp = lambda state, extra: jnp.round(gamm ** extra.iter - 1, 4)
#         for j, n_single in enumerate(n_samps):
#             vary_n_key, samp_key = random.split(vary_n_key)
#             print(f'Iter={i} - EKI samp - N={n_single}')
#
#             eki_samp = mocat.run(scenario,
#                                  TemperedEKI(next_temperature=next_temp,
#                                              max_temperature=1.0),
#                                  n_single,
#                                  samp_key)
#             eki_samp.repeat_ind = i
#             print(f'Time = {eki_samp.time} - Nans = {jnp.any(jnp.isnan(eki_samp.value[-1]))}')
#             eki_samps_n_vary[i, j] = eki_samp.deepcopy()
#
#             random_key, samp_key = random.split(random_key)
#             print(f'Iter={i} - EKI optim - N={n_single}')
#
#             eki_optim = mocat.run(scenario,
#                                   TemperedEKI(next_temperature=next_temp,
#                                               term_std=simulation_params.optim_max_sd_eki,
#                                               max_temperature=simulation_params.max_temp_eki),
#                                   n_single,
#                                   samp_key)
#             eki_optim.repeat_ind = i
#             print(
#                 f'Time = {eki_optim.time} Final temp = {eki_optim.temperature[-1]}- Nans = {jnp.any(jnp.isnan(eki_optim.value[-1]))}')
#             eki_optim_n_vary[i, j] = eki_optim.deepcopy()
#
#     with open(save_dir + '/eki_samps_n_vary', 'wb') as file:
#         pickle.dump(eki_samps_n_vary, file)
#
#     with open(save_dir + '/eki_optim_n_vary', 'wb') as file:
#         pickle.dump(eki_optim_n_vary, file)
#
#     # Vary n_steps
#     if repeat_data is not None:
#         scenario.data = repeat_data[0]
#
#     vary_nsteps_key = random_key.copy()
#     for k, n_steps in enumerate(simulation_params.vary_n_steps_eki):
#         vary_nsteps_key, samp_key = random.split(vary_nsteps_key)
#         print(f'EKI - N={simulation_params.fix_n_samps_eki}  - n_steps={n_steps}')
#         gamm = 2 ** (1 / n_steps)
#         next_temp = lambda state, extra: jnp.round(gamm ** extra.iter - 1, 4)
#         eki_samp = mocat.run(scenario,
#                              TemperedEKI(next_temperature=next_temp,
#                                          max_temperature=1.0),
#                              simulation_params.fix_n_samps_eki,
#                              samp_key)
#         print(f'Time = {eki_samp.time} - Nans = {jnp.any(jnp.isnan(eki_samp.value[-1]))}')
#         eki_samps_nsteps_vary[k] = eki_samp.deepcopy()
#
#     with open(save_dir + '/eki_samps_nsteps_vary', 'wb') as file:
#         pickle.dump(eki_samps_nsteps_vary, file)
#
#     # Optimisation
#     if repeat_data is not None:
#         scenario.data = repeat_data[0]
#     optim_single_key = random_key.copy()
#     print(f'EKI optim - N={simulation_params.fix_n_samps_eki}  - n_steps={simulation_params.fix_n_steps}')
#     gamm = 2 ** (1 / simulation_params.fix_n_steps)
#     next_temp = lambda state, extra: jnp.round(gamm ** extra.iter - 1, 4)
#     eki_optim = mocat.run(scenario,
#                           TemperedEKI(next_temperature=next_temp,
#                                       term_std=simulation_params.optim_max_sd_eki,
#                                       max_temperature=simulation_params.max_temp_eki),
#                           simulation_params.fix_n_samps_eki,
#                           optim_single_key)
#     print(f'Time = {eki_optim.time} - Final temperature = {eki_optim.temperature[-1]}'
#           f'- Nans = {jnp.any(jnp.isnan(eki_optim.value[-1]))}')
#
#     with open(save_dir + '/eki_optim_single', 'wb') as file:
#         pickle.dump(eki_optim, file)


def run_eki(scenario, save_dir, random_key, repeat_data=None, simulation_params=None):
    if simulation_params is None:
        simulation_params = mocat.load_cdict(save_dir + '/sim_params.cdict')

    n_samps = simulation_params.vary_n_samps_eki

    eki_samps_n_vary = np.zeros((simulation_params.n_repeats,
                                 len(n_samps)), dtype='object')

    eki_optim_n_vary = np.zeros((simulation_params.n_repeats,
                                 len(n_samps)), dtype='object')

    vary_n_key = random_key.copy()

    # Vary n
    for i in range(simulation_params.n_repeats):
        if repeat_data is not None:
            scenario.data = repeat_data[i]

        for j, n_single in enumerate(n_samps):
            vary_n_key, samp_key = random.split(vary_n_key)
            print(f'Iter={i} - EKI samp - N={n_single}')

            eki_samp = mocat.run(scenario,
                                 AdaptiveTemperedEKI(ess_threshold=simulation_params.ess_threshold,
                                                     max_temperature=1.0),
                                 n_single,
                                 samp_key)
            eki_samp.repeat_ind = i
            print(f'Time = {eki_samp.time} - Num temps = {eki_samp.temperature.size}'
                  f' - Nans = {jnp.any(jnp.isnan(eki_samp.value[-1]))}')
            eki_samps_n_vary[i, j] = eki_samp.deepcopy()

            random_key, samp_key = random.split(random_key)
            print(f'Iter={i} - EKI optim - N={n_single}')

            eki_optim = mocat.run(scenario,
                                  AdaptiveTemperedEKI(ess_threshold=simulation_params.ess_threshold,
                                                      term_std=simulation_params.optim_max_sd_eki,
                                                      max_temperature=simulation_params.max_temp_eki),
                                  n_single,
                                  samp_key)
            eki_optim.repeat_ind = i
            print(f'Time = {eki_optim.time} - Final temp = {eki_optim.temperature[-1]}'
                  f' - Num temps = {eki_optim.temperature.size}'
                  f' - Nans = {jnp.any(jnp.isnan(eki_optim.value[-1]))}')
            eki_optim_n_vary[i, j] = eki_optim.deepcopy()

    with open(save_dir + '/eki_samps_n_vary', 'wb') as file:
        pickle.dump(eki_samps_n_vary, file)

    with open(save_dir + '/eki_optim_n_vary', 'wb') as file:
        pickle.dump(eki_optim_n_vary, file)


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


d_max = 6


def plot_joint_scatters(vals, param_names, true_params=None, lims=None, color='black', **kwargs):
    d = vals.shape[-1]
    if d > d_max:
        d = d_max
        vals = vals[:, :d_max]
    fig, axes = plt.subplots(d, d)
    for i in range(d):
        for j in range(d):
            if i > j:
                axes[i, j].scatter(vals[:, j], vals[:, i], color=color, **kwargs)
                if true_params is not None:
                    axes[i, j].scatter(true_params[j], true_params[i], color='red', marker='x')
                if lims is not None:
                    axes[i, j].set_xlim(lims[j])
                    axes[i, j].set_ylim(lims[i])
                if len(param_names) == d:
                    axes[i, j].set_xlabel(param_names[j])
                    axes[i, j].set_ylabel(param_names[i])

            elif i == j:
                axes[i, j].hist(vals[:, i], bins=50, color=color)
                axes[i, j].set_yticks([])
                axes[i, j].spines['left'].set_visible(False)
                if true_params is not None:
                    axes[i, j].axvline(true_params[i], color='red')
                if lims is not None:
                    axes[i, j].set_xlim(lims[i])
                if len(param_names) == d:
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
    if d > d_max:
        d = d_max
        vals = vals[:, :d_max]
    fig, axes = plt.subplots(d, d)
    lims_not_none = lims is not None
    if not lims_not_none:
        lims = [[vals[:, i].min(), vals[:, i].max()] for i in range(d)]

    for i in range(d):
        for j in range(d):
            if i > j:
                xmin, xmax = lims[j]
                ymin, ymax = lims[i]

                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([vals[:, j], vals[:, i]])
                kernel = gaussian_kde(values)
                f = np.reshape(kernel(positions).T, xx.shape)

                axes[i, j].contourf(xx, yy, f, cmap=cmap, **kwargs)
                if true_params is not None:
                    axes[i, j].scatter(true_params[j], true_params[i], color='red', marker='x')
                if lims_not_none:
                    axes[i, j].set_xlim(lims[j])
                    axes[i, j].set_ylim(lims[i])
                if len(param_names) == d:
                    axes[i, j].set_xlabel(param_names[j])
                    axes[i, j].set_ylabel(param_names[i])
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
                if len(param_names) == d:
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


def plot_comp_densities(scenario, save_dir, ranges, true_params=None, param_names=None,
                        dim_inds=None, n_ind=None, repeat_ind=0):
    with open(save_dir + '/eki_samps_n_vary', 'rb') as file:
        eki_samp_all = pickle.load(file)

    with open(save_dir + '/abc_smc_samps', 'rb') as file:
        smc_samps_all = pickle.load(file)

    d = eki_samp_all[repeat_ind, 0].value.shape[-1]

    if dim_inds is None:
        dim_inds = jnp.arange(min(4, d))

    if n_ind is None:
        n_ind = 0 if len(eki_samp_all[repeat_ind]) == 1 else 1

    eki_samp_vals = eki_samp_all[repeat_ind, n_ind].value[-1][:, dim_inds]
    smc_vals = smc_samps_all[repeat_ind, n_ind].value[-1][:, dim_inds]
    smc_vals = smc_vals[smc_samps_all[repeat_ind, n_ind].log_weight[-1] > -jnp.inf]

    fig, axes = plot_densities(scenario, [eki_samp_vals, smc_vals],
                               true_params=true_params, param_names=param_names,
                               ranges=ranges, y_range_mult=1.,
                               labels=['EKI Sampling', 'ABC-SMC'],
                               color=['blue', 'green'],
                               alpha=1.,
                               linewidth=2)

    leg_ax = axes if scenario.dim == 1 else axes.ravel()[-1]
    leg_ax.legend(frameon=False, prop={'size': 10})
    fig.savefig(save_dir + '/densities', dpi=300)


def plot_comp_densities_rwmh(scenario, save_dir, ranges, true_params=None, param_names=None,
                        dim_inds=None, n_ind=None, repeat_ind=0):
    with open(save_dir + '/eki_samps_n_vary', 'rb') as file:
        eki_samp_all = pickle.load(file)

    with open(save_dir + '/abc_smc_samps', 'rb') as file:
        smc_samps_all = pickle.load(file)

    rwmh_samps = mocat.load_cdict(save_dir + '/rwmh_samps.cdict').value
    if hasattr(scenario, 'unconstrain'):
        rwmh_samps = scenario.unconstrain(rwmh_samps)

    d = eki_samp_all[repeat_ind, 0].value.shape[-1]

    if dim_inds is None:
        dim_inds = jnp.arange(min(4, d))

    if n_ind is None:
        n_ind = 0 if len(eki_samp_all[repeat_ind]) == 1 else 1

    eki_samp_vals = eki_samp_all[repeat_ind, n_ind].value[-1][:, dim_inds]
    smc_vals = smc_samps_all[repeat_ind, n_ind].value[-1][:, dim_inds]
    smc_vals = smc_vals[smc_samps_all[repeat_ind, n_ind].log_weight[-1] > -jnp.inf]

    fig, axes = plot_densities(scenario, [rwmh_samps, eki_samp_vals, smc_vals],
                               true_params=true_params, param_names=param_names,
                               ranges=ranges, y_range_mult=1.,
                               labels=['True Posterior', 'EKI Sampling', 'ABC-SMC'],
                               color=['grey', 'blue', 'green'],
                               alpha=[0.75, 1., 1.],
                               linewidth=2)

    leg_ax = axes if scenario.dim == 1 else axes.ravel()[-1]
    leg_ax.legend(frameon=False, prop={'size': 10})
    fig.savefig(save_dir + '/densities_w_rwmh', dpi=300)


def plot_optim_box_plots(scenario, save_dir, true_params=None, param_names=None,
                         dim_inds=None, n_ind=None, repeat_ind=0, bp_widths=0.1, n_optim_temps=7):
    with open(save_dir + '/eki_optim_n_vary', 'rb') as file:
        eki_optim_all = pickle.load(file)

    d = eki_optim_all[repeat_ind, 0].value.shape[-1]

    if dim_inds is None:
        dim_inds = jnp.arange(min(4, d))
    if n_ind is None:
        n_ind = 0 if len(eki_optim_all[repeat_ind]) == 1 else 1

    eki_optim_single = eki_optim_all[repeat_ind, n_ind]

    n_optim_inds = np.linspace(0., len(eki_optim_single.temperature) - 1, n_optim_temps, endpoint=True, dtype='int')
    temps = eki_optim_single.temperature[n_optim_inds]

    optim_vals = eki_optim_single.value[n_optim_inds]
    if hasattr(scenario, 'constrain'):
        optim_vals = scenario.constrain(optim_vals)

    optim_bp_fig, optim_bp_ax = plt.subplots(len(dim_inds))
    optim_bp_ax_rav = np.ravel(optim_bp_ax)
    for i in range(len(dim_inds)):
        int_dim = dim_inds[i]
        optim_bp_ax_rav[i].boxplot([b for b in optim_vals[:, :, int_dim]], positions=np.round(temps, 1),
                                   sym='', zorder=1, widths=bp_widths)
        if true_params is not None:
            optim_bp_ax_rav[i].axhline(true_params[i], color='red', zorder=0, alpha=1.)
        if i != len(dim_inds) - 1:
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
    optim_bp_fig.savefig(save_dir + '/EKI_temp_varytemps_bp', dpi=300)


def plot_eki(scenario, save_dir, ranges, true_params=None, param_names=None, y_range_mult=1.0,
             optim_ranges=None, bp_widths=1.0,
             legend_size=10, legend_ax=0):
    with open(save_dir + '/eki_samps_n_vary', 'rb') as file:
        eki_samp_all = pickle.load(file)

    # with open(save_dir + '/eki_samps_nsteps_vary', 'rb') as file:
    #     eki_samps_nsteps_vary = pickle.load(file)

    with open(save_dir + '/eki_optim_n_vary', 'rb') as file:
        eki_optim_all = pickle.load(file)

    # with open(save_dir + '/eki_optim_single', 'rb') as file:
    #     eki_optim_single = pickle.load(file)

    single_ind = 1 if eki_samp_all.shape[1] > 1 else 0

    eki_optim_single = eki_optim_all[0, single_ind]

    if optim_ranges is None:
        optim_ranges = ranges
    simulation_params = mocat.load_cdict(save_dir + '/sim_params.cdict')

    # # Plot sampling densities by n_steps
    # n_steps_lines = len(eki_samps_nsteps_vary)
    # eki_samp_vals = [a.value[-1] for a in eki_samps_nsteps_vary]
    #
    # samp_steps_fig, samp_steps_axes = plot_densities(scenario, eki_samp_vals,
    #                                                  true_params=true_params, param_names=param_names,
    #                                                  ranges=ranges, y_range_mult=y_range_mult,
    #                                                  labels=simulation_params.vary_n_steps_eki,
    #                                                  color='blue',
    #                                                  # alpha=0.2 + 0.8 * np.arange(n_steps_lines) / n_steps_lines,
    #                                                  alpha=0.7 - 0.7 * np.arange(n_steps_lines) / n_steps_lines,
    #                                                  linewidth=2.5)
    # leg_ax = samp_steps_axes if scenario.dim == 1 else samp_steps_axes.ravel()[legend_ax]
    # handles, labels = leg_ax.get_legend_handles_labels()
    # leg = leg_ax.legend(handles[::-1], labels[::-1], frameon=False, prop={'size': legend_size})
    # leg.set_title(title='Number of Steps', prop={'size': legend_size})
    # samp_steps_fig.savefig(save_dir + '/EKI_temp1_varynsteps', dpi=300)

    # Plot optimisation by temperature
    # Boxplot
    n_optim_temps = 7
    n_optim_inds = np.linspace(0., len(eki_optim_single.temperature) - 1, n_optim_temps, endpoint=True, dtype='int')
    temps = eki_optim_single.temperature[n_optim_inds]

    optim_vals = eki_optim_single.value[n_optim_inds]
    if hasattr(scenario, 'constrain'):
        optim_vals = scenario.constrain(optim_vals)

    optim_bp_fig, optim_bp_ax = plt.subplots(min(scenario.dim, 4))
    optim_bp_ax_rav = np.ravel(optim_bp_ax)
    for i in range(min(scenario.dim, 4)):
        optim_bp_ax_rav[i].boxplot([b for b in optim_vals[:, :, i]], positions=np.round(temps, 1),
                                   sym='', zorder=1, widths=bp_widths)
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
    temp_labs = [f'{float(a):.1f}' for a in temps]
    samp_temps_fig, samp_temps_axes = plot_densities(scenario, eki_optim_single.value[n_optim_inds],
                                                     true_params=true_params, param_names=param_names,
                                                     ranges=optim_ranges, y_range_mult=1.0,
                                                     labels=temp_labs,
                                                     color='blue',
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
    # samps = eki_samps_nsteps_vary[-1].value[-1]
    samps = eki_samp_all[0, single_ind].value[-1]
    if np.any(np.isnan(samps)):
        samps = eki_samp_all[0, -2].value[-1]
    if hasattr(scenario, 'constrain'):
        samps = scenario.constrain(samps)

    # Scatter
    fig_joint, ax_joint = plot_joint_scatters(samps, param_names, true_params=true_params, color='blue', s=0.5,
                                              lims=ranges)
    fig_joint.savefig(save_dir + '/EKI_joint_dens', dpi=300)

    # Contours
    fig_joint_cont, ax_joint_cont = plot_joint_contours(samps, param_names, true_params=true_params,
                                                        cmap='Blues', hist_color='blue', lims=ranges)
    fig_joint_cont.savefig(save_dir + '/EKI_joint_dens_contour', dpi=300)


def plot_abc(scenario, save_dir, ranges, true_params=None, param_names=None, legend_size=10):
    with open(save_dir + '/abc_mcmc_samps', 'rb') as file:
        mcmc_samps_all = pickle.load(file)

    with open(save_dir + '/abc_smc_samps', 'rb') as file:
        smc_samps_all = pickle.load(file)

    single_ind = 1 if smc_samps_all.shape[1] > 1 else 0

    smc_vals = smc_samps_all[0, single_ind].value[-1]
    mcmc_vals = mcmc_samps_all[0].value

    fig, axes = plot_densities(scenario, [mcmc_vals, smc_vals],
                               true_params=true_params, param_names=param_names,
                               ranges=ranges, y_range_mult=1.,
                               labels=['ABC-MCMC', 'ABC-SMC'],
                               color=['orange', 'green'],
                               alpha=1.,
                               linewidth=2)
    leg_ax = axes if scenario.dim == 1 else axes.ravel()[-1]
    leg_ax.legend(frameon=False, prop={'size': legend_size})
    fig.savefig(save_dir + '/abc_densities', dpi=300)


def calculate_rmse(value, true_params, axis=None):
    return jnp.sqrt(jnp.square(value - true_params).mean(axis))


# Use splines to smooth trajectories
def smooth(t, x, k=2):
    t_new = jnp.linspace(t.min(), t.max(), 200)
    spl = make_interp_spline(t, x, k=k)
    y_smooth = spl(t_new)
    return t_new, y_smooth


def smooth_bars(xs, ys, ax, color, label, nstds=1):
    xs_mean = xs.mean(0)
    ys_mean = np.array([x[~np.isnan(x)].mean() for x in ys.T])
    ys_std = np.array([np.std(x[~np.isnan(x)], ddof=1) for x in ys.T])
    xs_smooth, ys_mean_smooth = smooth(xs_mean, ys_mean)
    _, ys_top_smooth = smooth(xs_mean, ys_mean + nstds * ys_std)
    _, ys_bottom_smooth = smooth(xs_mean, ys_mean - nstds * ys_std)
    ax.plot(xs_smooth, ys_mean_smooth, color=color, label=label, zorder=5)
    ax.fill_between(xs_smooth, ys_bottom_smooth, ys_top_smooth,
                    color=color, alpha=0.25, zorder=4)


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

    n_repeats = len(eki_samp_all)

    transform = scenario.constrain if hasattr(scenario, 'constrain') else lambda x: x

    vary_n_mcmc = simulation_params.n_samps_abc_mcmc * np.array([0.01, 0.1, 1.])
    vary_n_mcmc = np.array(vary_n_mcmc, dtype='int')

    eki_rmses = np.array([[calculate_rmse(transform(eki_samp_all[i, j].value[-1]), true_params)
                           for j in range(len(simulation_params.vary_n_samps_eki))]
                          for i in range(n_repeats)])
    eki_num_sims = np.array([[j * (eki_samp_all[0, 0].temperature.size - 1)
                              for j in simulation_params.vary_n_samps_eki]
                             for _ in range(n_repeats)])

    eki_optim_rmses = np.array(
        [[jnp.sqrt(jnp.square(transform(eki_optim_all[i, j].value[-1]).mean(0) - true_params).mean())
          for j in range(len(simulation_params.vary_n_samps_eki))]
         for i in range(n_repeats)])
    eki_optim_num_sims = np.array([[simulation_params.vary_n_samps_eki[j] * (eki_optim_all[i, j].temperature.size - 1)
                                    for j in range(len(simulation_params.vary_n_samps_eki))]
                                   for i in range(n_repeats)])

    abc_mcmc_rmses = np.array([[calculate_rmse(transform(abc_mcmc_samps[i].value[:j]), true_params)
                                for j in vary_n_mcmc]
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
    ax_n.scatter(mcmc_n_samps_repeat, abc_mcmc_n_rmses, color='orange', label='ABC-MCMC')
    ax_n.scatter(smc_n_samps_repeat, abc_smc_rmses, color='green', label='ABC-SMC')
    ax_n.scatter(eki_n_samps_repeat, eki_rmses, color='blue', label='EKI Sampling')
    ax_n.scatter(eki_n_samps_repeat, eki_optim_rmses, color='aqua', label='EKI Optimisation')
    ax_n.set_xscale('log')
    ax_n.set_xlim([10 ** n_min_log10 * 0.9, 10 ** n_max_log10 * 1.1])
    ax_n.spines['top'].set_visible(False)
    ax_n.spines['right'].set_visible(False)
    ax_n.set_xlabel(r'$N$')
    ax_n.set_ylabel('RMSE')
    ax_n.legend(frameon=False)
    fig_n.savefig(save_dir + '/rmse_n_scatter', dpi=300)

    fig_n_s, ax_n_s = plt.subplots()
    smooth_bars(mcmc_n_samps_repeat, abc_mcmc_n_rmses, ax_n_s, color='orange', label='ABC-MCMC')
    smooth_bars(smc_n_samps_repeat, abc_smc_rmses, ax_n_s, color='green', label='ABC-SMC')
    smooth_bars(eki_n_samps_repeat, eki_rmses, ax_n_s, color='blue', label='EKI Sampling')
    smooth_bars(eki_n_samps_repeat, eki_optim_rmses, ax_n_s, color='aqua', label='EKI Optimisation')
    ax_n_s.set_xscale('log')
    ax_n_s.set_xlim([10 ** n_min_log10 * 0.9, 10 ** n_max_log10 * 1.1])
    ax_n_s.spines['top'].set_visible(False)
    ax_n_s.spines['right'].set_visible(False)
    ax_n_s.set_xlabel(r'$N$')
    ax_n_s.set_ylabel('RMSE')
    ax_n_s.legend(frameon=False)
    fig_n_s.savefig(save_dir + '/rmse_n', dpi=300)

    nsim_min = np.min([eki_num_sims.min(), eki_optim_num_sims.min(), abc_smc_sims.min()])
    nsim_max = np.max([eki_num_sims.max(), eki_optim_num_sims.max(), abc_smc_sims.max()])
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
    ax_nsims.scatter(abc_mcmc_sims, abc_mcmc_nsims_rmses, color='orange', label='ABC-MCMC')
    ax_nsims.scatter(abc_smc_sims, abc_smc_rmses, color='green', label='ABC-SMC')
    ax_nsims.scatter(eki_num_sims, eki_rmses, color='blue', label='EKI Sampling')
    ax_nsims.scatter(eki_optim_num_sims, eki_optim_rmses, color='aqua', label='EKI Optimisation')
    ax_nsims.set_xscale('log')
    ax_nsims.set_xlim([10 ** nsim_min_log10 * 0.9, 10 ** nsim_max_log10 * 1.1])
    ax_nsims.spines['top'].set_visible(False)
    ax_nsims.spines['right'].set_visible(False)
    ax_nsims.set_xlabel(r'Likelihood Simulations (from varying $N$)')
    ax_nsims.set_ylabel('RMSE')
    ax_nsims.legend(frameon=False)
    fig_nsims.savefig(save_dir + '/rmse_nsims_scatter', dpi=300)

    fig_nsims_s, ax_nsims_s = plt.subplots()
    smooth_bars(abc_mcmc_sims, abc_mcmc_n_rmses, ax_nsims_s, color='orange', label='ABC-MCMC')
    smooth_bars(abc_smc_sims, abc_smc_rmses, ax_nsims_s, color='green', label='ABC-SMC')
    smooth_bars(eki_num_sims, eki_rmses, ax_nsims_s, color='blue', label='EKI Sampling')
    smooth_bars(eki_optim_num_sims, eki_optim_rmses, ax_nsims_s, color='aqua', label='EKI Optimisation')
    ax_nsims_s.set_xscale('log')
    ax_nsims_s.set_xlim([10 ** nsim_min_log10 * 0.9, 10 ** nsim_max_log10 * 1.1])
    ax_nsims_s.spines['top'].set_visible(False)
    ax_nsims_s.spines['right'].set_visible(False)
    ax_nsims_s.set_xlabel(r'Likelihood Simulations (from varying $N$)')
    ax_nsims_s.set_ylabel('RMSE')
    ax_nsims_s.legend(frameon=False)
    fig_nsims_s.savefig(save_dir + '/rmse_nsims', dpi=300)

    num_n_eki = len(simulation_params.vary_n_samps_eki)
    num_n_mcmc = len(vary_n_mcmc)
    num_n_smc = len(simulation_params.vary_n_samps_abc_smc)

    eki_times = np.array([[eki_samp_all[i, j].time
                           for j in range(len(simulation_params.vary_n_samps_eki))]
                          for i in range(n_repeats)])
    eki_optim_times = np.array([[eki_optim_all[i, j].time
                                 for j in range(len(simulation_params.vary_n_samps_eki))]
                                for i in range(n_repeats)])
    abc_smc_times = np.array([[abc_smc_samps_all[i, j].time
                               for j in range(len(simulation_params.vary_n_samps_eki))]
                              for i in range(n_repeats)])

    table_fig, table_axes \
        = plt.subplots(1 + 2 * num_n_eki + num_n_mcmc + num_n_smc, 5)

    for axi in table_axes.ravel():
        axi.set_axis_off()

    def add_text(ax, s):
        ax.text(0.1, 0.1, s)

    add_text(table_axes[0, 1], 'N')
    add_text(table_axes[0, 2], 'RMSE')
    add_text(table_axes[0, 3], 'Lik Sims')
    add_text(table_axes[0, 4], 'Time (s)')

    row_ind = 1
    for i in range(num_n_eki):
        add_text(table_axes[row_ind + i, 0], 'EKI')
        add_text(table_axes[row_ind + i, 1], simulation_params.vary_n_samps_eki[i])

        rmse_med = np.median(eki_rmses[:, i])
        add_text(table_axes[row_ind + i, 2], f'{rmse_med:.3f}')

        lik_sim_med = np.median(eki_num_sims[:, i])
        add_text(table_axes[row_ind + i, 3], f'{lik_sim_med:.0f}')

        time_med = np.median(eki_times[:, i])
        add_text(table_axes[row_ind + i, 4], f'{time_med:.1f}')
    row_ind += num_n_eki

    for i in range(num_n_eki):
        add_text(table_axes[row_ind + i, 0], 'EKI Optim')
        add_text(table_axes[row_ind + i, 1], simulation_params.vary_n_samps_eki[i])

        rmse_med = np.median(eki_optim_rmses[:, i])
        add_text(table_axes[row_ind + i, 2], f'{rmse_med:.3f}')

        lik_sim_med = np.median(eki_optim_num_sims[:, i])
        add_text(table_axes[row_ind + i, 3], f'{lik_sim_med:.0f}')

        time_med = np.median(eki_optim_times[:, i])
        add_text(table_axes[row_ind + i, 4], f'{time_med:.1f}')
    row_ind += num_n_eki

    for i in range(num_n_mcmc):
        add_text(table_axes[row_ind + i, 0], 'ABC-MCMC')
        add_text(table_axes[row_ind + i, 1], vary_n_mcmc[i])

        rmse_med = np.median(abc_mcmc_rmses[:, i])
        add_text(table_axes[row_ind + i, 2], f'{rmse_med:.3f}')

        lik_sim_med = np.median(abc_mcmc_sims[:, i])
        add_text(table_axes[row_ind + i, 3], f'{lik_sim_med:.0f}')

    time_med = np.median(np.array([a.time for a in abc_mcmc_samps]))
    add_text(table_axes[row_ind + num_n_mcmc - 1, 4], f'{time_med:.1f}')
    row_ind += num_n_mcmc

    for i in range(num_n_smc):
        add_text(table_axes[row_ind + i, 0], 'ABC-SMC')
        add_text(table_axes[row_ind + i, 1], simulation_params.vary_n_samps_abc_smc[i])

        rmse_med = np.median(abc_smc_rmses[:, i])
        add_text(table_axes[row_ind + i, 2], f'{rmse_med:.3f}')

        lik_sim_med = np.median(abc_smc_sims[:, i])
        add_text(table_axes[row_ind + i, 3], f'{lik_sim_med:.0f}')

        time_med = np.median(abc_smc_times[:, i])
        add_text(table_axes[row_ind + i, 4], f'{time_med:.1f}')
    row_ind += num_n_eki

    table_fig.tight_layout()
    table_fig.savefig(save_dir + '/rmse_table', dpi=300)


# def plot_rmse(scenario, save_dir, true_params):
#     simulation_params = mocat.load_cdict(save_dir + '/sim_params.cdict')
#
#     with open(save_dir + '/eki_samps_n_vary', 'rb') as file:
#         eki_samp_all = pickle.load(file)
#
#     with open(save_dir + '/eki_optim_n_vary', 'rb') as file:
#         eki_optim_all = pickle.load(file)
#
#     with open(save_dir + '/abc_mcmc_samps', 'rb') as file:
#         abc_mcmc_samps = pickle.load(file)
#
#     with open(save_dir + '/abc_smc_samps', 'rb') as file:
#         abc_smc_samps_all = pickle.load(file)
#
#     n_repeats = simulation_params.n_repeats
#
#     transform = scenario.constrain if hasattr(scenario, 'constrain') else lambda x: x
#
#     vary_n_mcmc = simulation_params.n_samps_abc_mcmc * np.array([0.01, 0.1, 1.])
#     vary_n_mcmc = np.array(vary_n_mcmc, dtype='int')
#
#     eki_rmses = np.array([[calculate_rmse(transform(eki_samp_all[i, j].value[-1]), true_params)
#                            for j in range(len(simulation_params.vary_n_samps_eki))]
#                           for i in range(n_repeats)])
#     eki_num_sims = np.array([[j * simulation_params.fix_n_steps
#                               for j in simulation_params.vary_n_samps_eki]
#                              for _ in range(n_repeats)])
#
#     eki_optim_rmses = np.array(
#         [[jnp.sqrt(jnp.square(transform(eki_optim_all[i, j].value[-1]).mean(0) - true_params).mean())
#           for j in range(len(simulation_params.vary_n_samps_eki))]
#          for i in range(n_repeats)])
#     eki_optim_num_sims = np.array([[simulation_params.vary_n_samps_eki[j] * (eki_optim_all[i, j].temperature.size - 1)
#                                     for j in range(len(simulation_params.vary_n_samps_eki))]
#                                    for i in range(n_repeats)])
#
#     abc_mcmc_rmses = np.array([[calculate_rmse(transform(abc_mcmc_samps[i].value[:j]), true_params)
#                                 for j in vary_n_mcmc]
#                                for i in range(n_repeats)])
#     abc_mcmc_sims = np.array([[simulation_params.n_pre_run_abc_mcmc + j
#                                for j in vary_n_mcmc]
#                               for i in range(n_repeats)])
#
#     abc_smc_rmses = np.array([[calculate_rmse(transform(abc_smc_samps_all[i, j].value[-1]), true_params)
#                                for j in range(len(simulation_params.vary_n_samps_abc_smc))]
#                               for i in range(n_repeats)])
#     abc_smc_sims = np.array([[abc_smc_samps_all[i, j].num_sims
#                               for j in range(len(simulation_params.vary_n_samps_abc_smc))]
#                              for i in range(n_repeats)])
#
#     n_min = np.min([simulation_params.vary_n_samps_eki.min(), simulation_params.vary_n_samps_abc_smc.min()])
#     n_max = np.max([simulation_params.vary_n_samps_eki.max(), simulation_params.vary_n_samps_abc_smc.max()])
#     n_min_log10 = np.log(n_min) / np.log(10)
#     n_max_log10 = np.log(n_max) / np.log(10)
#     num_points_mcmc = np.maximum(len(simulation_params.vary_n_samps_eki), len(simulation_params.vary_n_samps_abc_smc))
#     n_mcmc = np.asarray(10 ** np.linspace(n_min_log10, n_max_log10, num_points_mcmc, endpoint=True), dtype='int')
#
#     abc_mcmc_n_rmses = np.array([[calculate_rmse(transform(abc_mcmc_samps[i].value[:j]), true_params)
#                                   for j in n_mcmc]
#                                  for i in range(n_repeats)])
#
#     eki_n_samps_repeat = jnp.repeat(simulation_params.vary_n_samps_eki[jnp.newaxis], n_repeats, 0)
#     mcmc_n_samps_repeat = jnp.repeat(n_mcmc[jnp.newaxis], n_repeats, 0)
#     smc_n_samps_repeat = jnp.repeat(simulation_params.vary_n_samps_abc_smc[jnp.newaxis], n_repeats, 0)
#
#     fig_n, ax_n = plt.subplots()
#     ax_n.scatter(eki_n_samps_repeat, eki_rmses, color='blue', label='TEKI')
#     ax_n.scatter(eki_n_samps_repeat, eki_optim_rmses, color='aqua', label='TEKI Optim')
#     ax_n.scatter(mcmc_n_samps_repeat, abc_mcmc_n_rmses, color='orange', label='ABC-MCMC')
#     ax_n.scatter(smc_n_samps_repeat, abc_smc_rmses, color='green', label='ABC-SMC')
#     ax_n.set_xscale('log')
#     ax_n.set_xlim([10 ** np.floor(n_min_log10), 10 ** np.ceil(n_max_log10)])
#     ax_n.spines['top'].set_visible(False)
#     ax_n.spines['right'].set_visible(False)
#     ax_n.set_xlabel(r'$N$')
#     ax_n.set_ylabel('RMSE')
#     ax_n.legend(frameon=False)
#     fig_n.savefig(save_dir + '/rmse_n_scatter', dpi=300)
#
#     nsim_min = np.min([eki_num_sims.min(), abc_smc_sims.min()])
#     nsim_max = np.max([eki_num_sims.max(), abc_smc_sims.max()])
#     nsim_max = np.minimum(nsim_max, simulation_params.n_samps_abc_mcmc)
#     nsim_min_log10 = np.log(nsim_min) / np.log(10)
#     nsim_max_log10 = np.log(nsim_max) / np.log(10)
#     nsim_mcmc = np.asarray(10 ** np.linspace(nsim_min_log10, nsim_max_log10, num_points_mcmc, endpoint=True),
#                            dtype='int')
#
#     # abc_mcmc_nsims_rmses = np.array(
#     #     [[calculate_rmse(transform(abc_mcmc_samps[i].value[:(j - simulation_params.n_pre_run_abc_mcmc)]), true_params)
#     #       for j in nsim_mcmc]
#     #      for i in range(n_repeats)])
#
#     # abc_mcmc_sims = np.array([[j + simulation_params.n_pre_run_abc_mcmc
#     #                            for j in nsim_mcmc]
#     #                           for _ in range(n_repeats)])
#     #
#     # fig_nsims, ax_nsims = plt.subplots()
#     # ax_nsims.scatter(eki_num_sims, eki_rmses, color='blue', label='TEKI')
#     # ax_nsims.scatter(eki_optim_num_sims, eki_optim_rmses, color='aqua', label='TEKI Optim')
#     # ax_nsims.scatter(abc_mcmc_sims, abc_mcmc_nsims_rmses, color='orange', label='ABC-MCMC')
#     # ax_nsims.scatter(abc_smc_sims, abc_smc_rmses, color='green', label='ABC-SMC')
#     # ax_nsims.set_xscale('log')
#     # ax_nsims.set_xlim([10 ** np.floor(nsim_min_log10), 10 ** np.ceil(nsim_max_log10)])
#     # ax_nsims.spines['top'].set_visible(False)
#     # ax_nsims.spines['right'].set_visible(False)
#     # ax_nsims.set_xlabel('Likelihood Simulations')
#     # ax_nsims.set_ylabel('RMSE')
#     # ax_nsims.legend(frameon=False)
#     # fig_nsims.savefig(save_dir + '/rmse_nsims_scatter', dpi=300)
#
#     num_n_eki = len(simulation_params.vary_n_samps_eki)
#     num_n_mcmc = len(vary_n_mcmc)
#     num_n_smc = len(simulation_params.vary_n_samps_abc_smc)
#
#     eki_times = np.array([[eki_samp_all[i, j].time
#                            for j in range(len(simulation_params.vary_n_samps_eki))]
#                           for i in range(n_repeats)])
#     eki_optim_times = np.array([[eki_optim_all[i, j].time
#                                  for j in range(len(simulation_params.vary_n_samps_eki))]
#                                 for i in range(n_repeats)])
#     abc_smc_times = np.array([[abc_smc_samps_all[i, j].time
#                                for j in range(len(simulation_params.vary_n_samps_eki))]
#                               for i in range(n_repeats)])
#
#     table_fig, table_axes \
#         = plt.subplots(1 + 2 * num_n_eki + num_n_mcmc + num_n_smc, 5)
#
#     for axi in table_axes.ravel():
#         axi.set_axis_off()
#
#     def add_text(ax, s):
#         ax.text(0.1, 0.1, s)
#
#     add_text(table_axes[0, 1], 'N')
#     add_text(table_axes[0, 2], 'RMSE')
#     add_text(table_axes[0, 3], 'Lik Sims')
#     add_text(table_axes[0, 4], 'Time (s)')
#
#     row_ind = 1
#     for i in range(num_n_eki):
#         add_text(table_axes[row_ind + i, 0], 'EKI')
#         add_text(table_axes[row_ind + i, 1], simulation_params.vary_n_samps_eki[i])
#
#         rmse_med = np.median(eki_rmses[:, i])
#         add_text(table_axes[row_ind + i, 2], f'{rmse_med:.3f}')
#
#         lik_sim_med = np.median(eki_num_sims[:, i])
#         add_text(table_axes[row_ind + i, 3], f'{lik_sim_med:.0f}')
#
#         time_med = np.median(eki_times[:, i])
#         add_text(table_axes[row_ind + i, 4], f'{time_med:.1f}')
#     row_ind += num_n_eki
#
#     for i in range(num_n_eki):
#         add_text(table_axes[row_ind + i, 0], 'EKI Optim')
#         add_text(table_axes[row_ind + i, 1], simulation_params.vary_n_samps_eki[i])
#
#         rmse_med = np.median(eki_optim_rmses[:, i])
#         add_text(table_axes[row_ind + i, 2], f'{rmse_med:.3f}')
#
#         lik_sim_med = np.median(eki_optim_num_sims[:, i])
#         add_text(table_axes[row_ind + i, 3], f'{lik_sim_med:.0f}')
#
#         time_med = np.median(eki_optim_times[:, i])
#         add_text(table_axes[row_ind + i, 4], f'{time_med:.1f}')
#     row_ind += num_n_eki
#
#     for i in range(num_n_mcmc):
#         add_text(table_axes[row_ind + i, 0], 'ABC-MCMC')
#         add_text(table_axes[row_ind + i, 1], vary_n_mcmc[i])
#
#         rmse_med = np.median(abc_mcmc_rmses[:, i])
#         add_text(table_axes[row_ind + i, 2], f'{rmse_med:.3f}')
#
#         lik_sim_med = np.median(abc_mcmc_sims[:, i])
#         add_text(table_axes[row_ind + i, 3], f'{lik_sim_med:.0f}')
#
#     time_med = np.median(np.array([a.time for a in abc_mcmc_samps]))
#     add_text(table_axes[row_ind + num_n_mcmc - 1, 4], f'{time_med:.1f}')
#     row_ind += num_n_mcmc
#
#     for i in range(num_n_smc):
#         add_text(table_axes[row_ind + i, 0], 'ABC-SMC')
#         add_text(table_axes[row_ind + i, 1], simulation_params.vary_n_samps_abc_smc[i])
#
#         rmse_med = np.median(abc_smc_rmses[:, i])
#         add_text(table_axes[row_ind + i, 2], f'{rmse_med:.3f}')
#
#         lik_sim_med = np.median(abc_smc_sims[:, i])
#         add_text(table_axes[row_ind + i, 3], f'{lik_sim_med:.0f}')
#
#         time_med = np.median(abc_smc_times[:, i])
#         add_text(table_axes[row_ind + i, 4], f'{time_med:.1f}')
#     row_ind += num_n_eki
#
#     table_fig.tight_layout()
#     table_fig.savefig(save_dir + '/rmse_table', dpi=300)


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

    with open(save_dir + '/eki_optim_n_vary', 'rb') as file:
        eki_optim_all = pickle.load(file)

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
    eki_optim_dists = np.zeros_like(eki_dists)
    for i in range(n_repeats):
        print(f'EKI dist sims - Iter {i}')
        if repeat_summarised_data is not None:
            scenario.data = repeat_summarised_data[i]
        for j in range(len(simulation_params.vary_n_samps_eki)):
            print(f'N = {simulation_params.vary_n_samps_eki[j]}')
            if not jnp.any(jnp.isnan(eki_samp_all[i, j].simulated_data[-1])):
                random_key, subkey = random.split(random_key)
                eki_dists[i, j] = estimate_dist(eki_samp_all[i, j].value[-1], subkey)
            else:
                eki_dists[i, j] = np.nan

            if not jnp.any(jnp.isnan(eki_optim_all[i, j].simulated_data[-1])):
                random_key, subkey = random.split(random_key)
                eki_optim_dists[i, j] = estimate_dist(eki_samp_all[i, j].value[-1].mean(0)[jnp.newaxis], subkey)
            else:
                eki_optim_dists[i, j] = np.nan

    with open(save_dir + '/eki_est_dists', 'wb') as file:
        pickle.dump(eki_dists, file)

    with open(save_dir + '/eki_optim_est_dists', 'wb') as file:
        pickle.dump(eki_optim_dists, file)

    eki_num_sims = np.array([[j * simulation_params.fix_n_steps
                              for j in simulation_params.vary_n_samps_eki]
                             for _ in range(n_repeats)])

    eki_optim_num_sims = np.array([[simulation_params.vary_n_samps_eki[j] * (eki_optim_all[i, j].temperature.size - 1)
                                    for j in range(len(simulation_params.vary_n_samps_eki))]
                                   for i in range(n_repeats)])

    abc_smc_dists = np.zeros((n_repeats, len(simulation_params.vary_n_samps_abc_smc)))
    for i in range(n_repeats):
        print(f'SMC dist sims - Iter {i}')
        if repeat_summarised_data is not None:
            scenario.data = repeat_summarised_data[i]
        for j in range(len(simulation_params.vary_n_samps_abc_smc)):
            print(f'N = {simulation_params.vary_n_samps_abc_smc[j]}')
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
    # est_dist_y_max = abc_smc_dists.max()

    fig_n, ax_n = plt.subplots()
    ax_n.scatter(eki_n_samps_repeat, eki_dists, color='blue', label='TEKI')
    ax_n.scatter(eki_n_samps_repeat, eki_optim_dists, color='aqua', label='TEKI Optim')
    # ax_n.scatter(mcmc_n_samps_repeat, abc_mcmc_n_dists, color='orange', label='ABC-MCMC')
    ax_n.scatter(smc_n_samps_repeat, abc_smc_dists, color='green', label='ABC-SMC')
    # ax_n.set_ylim([0, est_dist_y_max])
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
    ax_n.scatter(eki_optim_num_sims, eki_optim_dists, color='aqua', label='TEKI Optim')
    # ax_nsims.scatter(abc_mcmc_sims, abc_mcmc_n_dists, color='orange', label='ABC-MCMC')
    ax_nsims.scatter(abc_smc_sims, abc_smc_dists, color='green', label='ABC-SMC')
    # ax_nsims.set_ylim([0, est_dist_y_max])
    ax_nsims.set_xscale('log')
    ax_nsims.spines['top'].set_visible(False)
    ax_nsims.spines['right'].set_visible(False)
    ax_nsims.set_xlabel('Likelihood Simulations')
    ax_nsims.set_ylabel('Distance')
    ax_nsims.legend(frameon=False)
    fig_nsims.savefig(save_dir + '/est_distance_nsims_scatter', dpi=300)
