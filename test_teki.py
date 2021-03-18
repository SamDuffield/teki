########################################################################################################################
# Module: tests/test_teki.py
# Description: Tests for tempered ensemble Kalman inversion.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

import unittest

import jax.numpy as jnp
from jax.random import PRNGKey

from mocat.src.tests.test_abc_linear_gaussian import TestLinearGaussianABC
from mocat.src.abc.scenarios.gk import GKTransformedUniformPrior, GKOnlyATransformedUniformPrior
from mocat.src.sample import run
from teki import TemperedEKI


class TestTEKIGaussian(TestLinearGaussianABC):
    n = 1000

    def test_pre_threshold(self):
        preschedule = jnp.arange(0., 1.1, 0.1)
        sample = run(self.scenario, TemperedEKI(temperature_schedule=preschedule), n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[-1])
        self._test_cov(sample.value[-1])

    def test_func_temp(self):
        n_steps = 100
        gamm = 2 ** (1 / n_steps)
        next_temp = lambda state, extra: jnp.round(gamm ** extra.iter - 1, 4)
        sample = run(self.scenario, TemperedEKI(next_temperature=next_temp), n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[-1])
        self._test_cov(sample.value[-1])


class GKQuantile(GKTransformedUniformPrior):
    quantiles = jnp.linspace(0, 1, 50, endpoint=False)[1:]
    n_unsummarised_data: int = 1000

    def summarise_data(self,
                       data: jnp.ndarray):
        return jnp.quantile(data, self.quantiles)


class GKOnlyAQuantile(GKOnlyATransformedUniformPrior):
    quantiles = jnp.linspace(0, 1, 50, endpoint=False)[1:]
    n_unsummarised_data: int = 1000

    def summarise_data(self,
                       data: jnp.ndarray):
        return jnp.quantile(data, self.quantiles)


class TestGKQuantile(unittest.TestCase):

    n = int(1e5)
    onlyA = False

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        if self.onlyA:
            self.scenario = GKOnlyAQuantile()
            self.true_constrained_params = jnp.array([3.])
        else:
            self.scenario = GKQuantile()
            self.true_constrained_params = jnp.array([3., 1., 2., 0.5])
        self.true_unconstrained_params = self.scenario.unconstrain(self.true_constrained_params)

        self.scenario.data = self.scenario.likelihood_sample(self.true_unconstrained_params,
                                                             random_key=PRNGKey(0))

    def _test_mean(self,
                   vals: jnp.ndarray,
                   precision: float = 3.0):
        samp_mean = vals.mean(axis=0)
        self.assertLess(jnp.abs(self.true_unconstrained_params - samp_mean).sum(), precision)


class TestTEKIGK(TestGKQuantile):
    n = 300

    def test_pre_threshold(self):
        preschedule = jnp.arange(0., 1.1, 0.1)
        sample = run(self.scenario, TemperedEKI(temperature_schedule=preschedule), n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[-1])

    def test_func_temp(self):
        n_steps = 100
        gamm = 2 ** (1 / n_steps)
        next_temp = lambda state, extra: jnp.round(gamm ** extra.iter - 1, 4)

        sample = run(self.scenario, TemperedEKI(next_temperature=next_temp), n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[-1])


class TestTEKIGKOnlyA(TestGKQuantile):
    n = 300
    onlyA = True

    def test_pre_threshold(self):
        preschedule = jnp.arange(0., 1.1, 0.1)
        sample = run(self.scenario, TemperedEKI(temperature_schedule=preschedule), n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[-1])

    def test_func_temp(self):
        n_steps = 100
        gamm = 2 ** (1 / n_steps)
        next_temp = lambda state, extra: jnp.round(gamm ** extra.iter - 1, 4)

        sample = run(self.scenario, TemperedEKI(next_temperature=next_temp), n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[-1])


if __name__ == '__main__':
    unittest.main()
