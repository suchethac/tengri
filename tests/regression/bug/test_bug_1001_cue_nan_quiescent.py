# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #1001 — Cue all-NaN SED for quiescent/old SFHs.

Root cause: SSP grids that include an age-0 anchor bin carry
``log10(age) = -inf`` as the first entry of the age grid. Bilinear
interpolation in :func:`interpolate_ionizing_params` then computes the
cell fraction ``fa = (x - (-inf)) / (edge - (-inf)) = inf/inf = NaN``
for any target in the first cell, poisoning the interpolated ionizing
parameters. Downstream, ``argmax(weighted_qh)`` selects the NaN bin
(NaN wins any comparison) whenever the age-0 bin has nonzero SFH
weight — dexp/dpl at low redshift — feeding an all-NaN ionizing
spectrum into the Cue emulator and producing a silent all-NaN SED.

https://github.com/suchethac/tengri/issues/1001
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.nebular.ionizing_spectrum import interpolate_ionizing_params

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def inf_edged_tables():
    """Tiny tables on a grid whose first age bin is log10(0) = -inf."""
    ssp_lgmet = jnp.array([-3.0, -2.0, -1.0])
    ssp_log_age_yr = jnp.array([-jnp.inf, 5.1, 6.0, 7.0])
    n_met, n_age = ssp_lgmet.shape[0], ssp_log_age_yr.shape[0]
    # Distinct, finite values so node identity is checkable exactly.
    ionspec = jnp.arange(n_met * n_age * 7, dtype=jnp.float64).reshape(n_met, n_age, 7) / 10.0
    logqion = jnp.arange(n_met * n_age, dtype=jnp.float64).reshape(n_met, n_age) + 40.0
    return ionspec, logqion, ssp_lgmet, ssp_log_age_yr


class TestInfAgeEdgeInterpolation:
    """#1001: the -inf age edge must never produce NaN."""

    def test_target_at_minus_inf_returns_age_zero_node(self, inf_edged_tables):
        """Interpolating AT the age-0 bin returns exactly that node's values."""
        ionspec, logqion, lgmet, log_age = inf_edged_tables
        i7, q = interpolate_ionizing_params(
            ionspec, logqion, lgmet, log_age, float(lgmet[0]), -np.inf
        )
        assert np.all(np.isfinite(np.asarray(i7))), f"NaN ionspec at -inf target: {i7}"
        assert np.isfinite(float(q))
        assert_allclose(np.asarray(i7), np.asarray(ionspec[0, 0]), rtol=1e-12)
        assert_allclose(float(q), float(logqion[0, 0]), rtol=1e-12)

    def test_target_at_first_finite_age_returns_that_node(self, inf_edged_tables):
        """A target on the first finite grid age lands in the degenerate
        [-inf, age1] cell; it must return the age1 node, not NaN."""
        ionspec, logqion, lgmet, log_age = inf_edged_tables
        i7, q = interpolate_ionizing_params(
            ionspec, logqion, lgmet, log_age, float(lgmet[0]), float(log_age[1])
        )
        assert np.all(np.isfinite(np.asarray(i7))), f"NaN ionspec at age grid[1]: {i7}"
        assert_allclose(np.asarray(i7), np.asarray(ionspec[0, 1]), rtol=1e-12)
        assert_allclose(float(q), float(logqion[0, 1]), rtol=1e-12)

    def test_all_grid_ages_finite_under_vmap(self, inf_edged_tables):
        """The vmap over the full age grid (the _compute_weighted_cue_params
        pattern) must be finite in every row — this is the exact call shape
        that poisoned the Cue emulator inputs in #1001."""
        ionspec, logqion, lgmet, log_age = inf_edged_tables
        i7_all, q_all = jax.vmap(
            lambda a: interpolate_ionizing_params(ionspec, logqion, lgmet, log_age, -2.5, a)
        )(log_age)
        assert np.all(np.isfinite(np.asarray(i7_all))), (
            f"NaN rows: {np.where(np.isnan(np.asarray(i7_all)).any(axis=-1))[0]}"
        )
        assert np.all(np.isfinite(np.asarray(q_all)))

    def test_interior_cells_unchanged(self, inf_edged_tables):
        """The guard must not perturb ordinary finite-cell interpolation:
        midway between age nodes 1 and 2 gives the exact average."""
        ionspec, logqion, lgmet, log_age = inf_edged_tables
        mid = 0.5 * (float(log_age[1]) + float(log_age[2]))
        i7, q = interpolate_ionizing_params(ionspec, logqion, lgmet, log_age, float(lgmet[0]), mid)
        expected = 0.5 * (np.asarray(ionspec[0, 1]) + np.asarray(ionspec[0, 2]))
        assert_allclose(np.asarray(i7), expected, rtol=1e-12)
        assert_allclose(float(q), 0.5 * (float(logqion[0, 1]) + float(logqion[0, 2])), rtol=1e-12)


class TestQuiescentCueParamsFinite:
    """#1001 end-to-end at the backend level (data-gated on the bc03 grid)."""

    def test_weighted_cue_params_finite_with_age_zero_weight(self, ssp_data_bc03):
        """Positive weight in the age-0 bin must not select a NaN ionizing
        spectrum — the exact failure mode of quiescent_z0 in #1001."""
        from tengri.components.nebular import _DEFAULT_CUE_WEIGHTS_PATH
        from tengri.components.nebular.cue import CueBackend

        backend = CueBackend(str(_DEFAULT_CUE_WEIGHTS_PATH), ssp_data=ssp_data_bc03)
        n_age = backend._ssp_log_age_yr.shape[0]
        weights = jnp.full((n_age,), 1e-5)  # every bin weighted, incl. age 0
        params = backend._compute_weighted_cue_params(
            weights, jnp.asarray(backend._ssp_log_age_yr), log_z=-1.79
        )
        vals = np.array([float(params[k]) for k in sorted(params)])
        assert np.all(np.isfinite(vals)), f"non-finite Cue params: {params}"
