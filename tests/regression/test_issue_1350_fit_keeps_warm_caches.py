# SPDX-License-Identifier: BSD-3-Clause
"""A default fit must not de-warm the process (#1350).

``Fitter.run``'s default "iterate" cache policy promises *"fit() keeps your warm
caches"*, but it called :func:`clear_shared_caches` with the inherited
``drop_xla=True``, so every fit ended in ``jax.clear_caches()`` — wiping JAX's
process-wide executables. The tengri-side entries it deliberately preserved were
left hollow (Python objects kept, compiles gone), and the caller's own
``predict_photometry`` was de-warmed too (measured 2152x in #1350).

**These tests are load-immune by construction.** They do not time anything —
wall-clock on a shared runner is worthless here (the same comparison measured
0.97x on one machine and 1.99x on another). Instead a canary ``jax.jit``
function counts its own *traces*: its Python body runs only when JAX traces it,
so a re-trace after a fit is exactly the signal that the process was de-warmed.

The sweep policy (Catalog / the retired ``lean=True``) must keep dropping
executables — it exists for memory relief and the notebook-OOM class — so that
half of the contract is pinned too.
"""

import jax
import numpy as np
import pytest


def _canary():
    """A jitted function that counts how many times JAX traced it."""
    traces = []

    @jax.jit
    def fn(x):
        traces.append(1)  # Python-side: runs on trace, not on cached execution
        return x * 2.0

    fn(1.0)  # first trace
    assert len(traces) == 1
    return fn, traces


@pytest.mark.regression_bug
class TestFitKeepsWarmCaches:
    """The default fit policy must leave the process's JAX caches intact."""

    @staticmethod
    def _model_and_data(ssp_data_wne, synthetic_tophat_obs):
        from tengri import FIXED, FREE, SEDModel

        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust={"law": "power_law", "type": "two_component", "all_params": FIXED},
            redshift=0.05,
        )
        key = jax.random.PRNGKey(0)
        flux = np.asarray(model.predict_photometry(model.spec.sample(key)), dtype=np.float64)
        noise = np.maximum(0.05 * np.abs(flux), np.max(np.abs(flux)) * 1e-8)
        return model, flux, noise, key

    def test_default_fit_does_not_retrace_the_process(self, ssp_data_wne, synthetic_tophat_obs):
        """LOAD-BEARING: a default fit leaves an unrelated warm jit untouched.

        Neuter: restore ``drop_xla=True`` on the iterate branch in ``Fitter.run``
        and the canary re-traces, failing this test.
        """
        model, flux, noise, key = self._model_and_data(ssp_data_wne, synthetic_tophat_obs)
        fn, traces = _canary()

        model.fit(flux, noise, method="map", key=key, n_steps=50, verbose=False)

        fn(1.0)  # would re-trace only if the process's JAX caches were cleared
        assert len(traces) == 1, (
            f"a default fit() de-warmed the process: the canary re-traced "
            f"({len(traces)} traces). jax.clear_caches() must not fire on the "
            f"iterate policy — see #1350."
        )

    def test_sweep_policy_still_releases_executables(self, ssp_data_wne, synthetic_tophat_obs):
        """The memory-relief path must keep dropping XLA executables.

        Guards the other half of the trade-off: sweep exists for the notebook-OOM
        class, so the #1350 fix must not silently keep 5-6 GB scan bodies alive.
        """
        model, flux, noise, key = self._model_and_data(ssp_data_wne, synthetic_tophat_obs)
        fn, traces = _canary()

        # _cache_policy="sweep" is the private kwarg Catalog uses; the retired
        # lean=True maps onto it as well.
        model.fit(
            flux, noise, method="map", key=key, n_steps=50, verbose=False, _cache_policy="sweep"
        )

        fn(1.0)
        assert len(traces) == 2, (
            "the sweep policy must still call jax.clear_caches() (memory relief for "
            "the notebook-OOM class); the canary should have re-traced exactly once."
        )
