########################################################################################################################
# Compare TEKI and ABC techniques for g-and-k distribution (4 dimensional)
########################################################################################################################


from jax import numpy as np, random, vmap
import matplotlib.pyplot as plt

import mocat
from mocat import abc

n_data = int(1e3)


class GKThinOrder(abc.scenarios.GKTransformedUniformPrior):
    num_thin: int = 100
    threshold = 5

    def simulate_data(self,
                      x: np.ndarray,
                      random_key: np.ndarray) -> np.ndarray:
        data_keys = random.split(random_key, n_data)
        return vmap(self.likelihood_sample, (None, 0))(x, data_keys)

    def summarise_data(self,
                       data: np.ndarray):
        order_stats = data.sort()
        thin_inds = np.linspace(0, len(data), self.num_thin, endpoint=False, dtype='int32')
        return order_stats[thin_inds]

    def distance_function(self,
                          summarised_simulated_data: np.ndarray) -> float:
        return np.sqrt(np.square(summarised_simulated_data - self.summary_statistic).sum())


gk_scenario = GKThinOrder()

true_constrained_params = np.array([3., 1., 2., 0.5])
true_unconstrained_params = gk_scenario.unconstrain(true_constrained_params)

random_key = random.PRNGKey(0)
random_key, subkey = random.split(random_key)

sim_data_keys = random.split(subkey, n_data)
data = vmap(gk_scenario.likelihood_sample, (None, 0))(true_unconstrained_params, sim_data_keys)

summary_statistic = gk_scenario.summarise_data(data)

n_samps = int(1e3)

eki_one_step = mocat.run_tempered_ensemble_kalman_inversion(gk_scenario, n_samps, random_key, summary_statistic,
                                                            temperature_schedule=[0, 1])

ng_sched_mod = 11
eki_multig_mod_step = mocat.run_tempered_ensemble_kalman_inversion(gk_scenario, n_samps, random_key, summary_statistic,
                                                                   temperature_schedule=np.linspace(0, 1, ng_sched_mod))

ng_sched_large = 51
eki_multig_large_step = mocat.run_tempered_ensemble_kalman_inversion(gk_scenario, n_samps, random_key,
                                                                     summary_statistic,
                                                                     temperature_schedule=np.linspace(0, 1,
                                                                                                      ng_sched_large))

fig, axes = plt.subplots(2, 2)
axes_rav = np.ravel(axes)
alph = 0.7
for i in range(4):
    axes_rav[i].hist(gk_scenario.constrain(eki_one_step.value[-1, :, i]), bins=50, density=True,
                     label='1 Step', alpha=alph)
    axes_rav[i].hist(gk_scenario.constrain(eki_multig_mod_step.value[-1, :, i]), bins=50, density=True,
                     label=f'{ng_sched_mod - 1} Step', alpha=alph)
    axes_rav[i].hist(gk_scenario.constrain(eki_multig_large_step.value[-1, :, i]), bins=50, density=True,
                     label=f'{ng_sched_large - 1} Step', alpha=alph)
    axes_rav[i].axvline(true_constrained_params[i], label='True Value', c='red')

axes_rav[0].set_xlim(0.5, 4.0)
axes_rav[0].set_ylim(0., 2.0)

fig.suptitle(f'Geometric Schedule TEKI, g-and-k, N={n_samps}')
plt.tight_layout()
plt.legend(frameon=False)

plt.show()

fig.savefig(f'/Users/samddd/Desktop/geometric_teki_N={n_samps}', dpi=300)


