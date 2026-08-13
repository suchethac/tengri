# SPDX-License-Identifier: BSD-3-Clause
"""Regression: dense CIC integrand degenerates on age=0 SSP templates (#1030).

#964 evaluates parametric SFHs on a dense log-spaced integrand grid spanning
the SSP template ages, built as ``10 ** linspace(log10(ages[0]),
log10(ages[-1]))``. A leading age = 0 template (``lg = -inf``, e.g. the BC03
Padova-1994 + STELIB grid shipped in the public catalog as
``bc03_pdva_stelib_chabrier``) makes ``log10(ages[0]) = -inf``, so the
"dense" grid collapses to ``[0, 0, ..., 0, agemax]``. The SFH shape is then
renormalized on that degenerate grid and cloud-in-cell assigns ~100% of the
formed mass to the youngest SSP node — for every SFH type. Total mass is
conserved, so the failure is silent: an old skew-normal SFH (peak 10 Gyr
ago) produced an optical SED 87x too bright in the ProSpect-R physics
reproduction while ``log_mstar_formed`` stayed exact.

The sibling CIC helpers already guard this edge (``_cic_parcels`` falls back
to linear-in-age interpolation weights; ``_youngest_bin_lookback_multiplier``
is a documented no-op) — only the grid builder missed it. The fix spans the
dense grid from the smallest *positive* template age; the ``[0, age_1]``
sliver is covered by ``_cic_parcels``' lookback-0 extension (#538 contract).
Same failure class as the -inf age-grid edges of #1001/#1017.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel
from tengri.components.stellar.component import _refine_sfh_table_ages
from tengri.components.stellar.sps.dsps_wrapper import SSPData

pytestmark = pytest.mark.regression_bug


def _synthetic_ssp(prepend_age_zero: bool = False):
    """20-bin synthetic SSP (CI-safe); optionally prepend an age=0 template."""
    lg = jnp.linspace(-3.5, 1.1, 20)  # 0.32 Myr .. ~12.6 Gyr, no age=0
    n_met, n_wave = 3, 100
    flux = (
        jnp.abs(jax.random.normal(jax.random.PRNGKey(123), (n_met, lg.shape[0], n_wave))) * 1e-3
        + 1e-5
    )
    if prepend_age_zero:
        lg = jnp.concatenate([jnp.array([-jnp.inf]), lg])
        flux = jnp.concatenate([jnp.zeros((n_met, 1, n_wave)), flux], axis=1)
    return SSPData(
        ssp_wave=jnp.linspace(3000.0, 10000.0, n_wave),
        ssp_flux=flux,
        ssp_lg_age_gyr=lg,
        ssp_lgmet=jnp.array([-1.5, -0.5, 0.0]),
    )


def _old_snorm_model(ssp):
    """All-Fixed skew-normal SFH peaking 10 Gyr ago (the ProSpect-R fiducial)."""
    return SEDModel.build(
        ssp,
        sfh={
            "type": "snorm",
            "peak_lbt_gyr": Fixed(10.0),
            "width_gyr": Fixed(2.0),
            "skew": Fixed(0.5),
            "log_total_mass": Fixed(10.0),
            "*": FIXED,
        },
        met={"logzsol": Fixed(0.0), "*": FIXED},
        dust={
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "*": FIXED,
        },
        redshift=Fixed(0.0),
    )


class TestRefinedGrid:
    def test_finite_and_increasing_on_age0_grid(self):
        """A leading age=0 template must not degenerate the dense grid."""
        ages_yr = jnp.concatenate([jnp.zeros(1), 10.0 ** jnp.linspace(5.5, 10.1, 20)])
        grid = np.asarray(_refine_sfh_table_ages(ages_yr))
        assert np.all(np.isfinite(grid))
        assert np.all(np.diff(grid) > 0.0), "dense grid must be strictly increasing"
        assert grid[0] == pytest.approx(10.0**5.5, rel=1e-12)  # smallest POSITIVE age
        assert grid[-1] == pytest.approx(10.0**10.1, rel=1e-12)

    def test_unchanged_without_age0(self):
        """Grids without an age=0 template keep the exact original span."""
        ages_yr = 10.0 ** jnp.linspace(5.5, 10.1, 20)
        grid = np.asarray(_refine_sfh_table_ages(ages_yr))
        assert grid[0] == pytest.approx(float(ages_yr[0]), rel=1e-12)
        assert grid[-1] == pytest.approx(float(ages_yr[-1]), rel=1e-12)
        assert np.all(np.diff(grid) > 0.0)

    def test_jit_safe_on_traced_ages(self):
        """The guard must trace (SSP-as-JIT-parameter path): no host branching."""
        ages_yr = jnp.concatenate([jnp.zeros(1), 10.0 ** jnp.linspace(5.5, 10.1, 20)])
        grid = jax.jit(_refine_sfh_table_ages)(ages_yr)
        assert np.all(np.isfinite(np.asarray(grid)))
        assert np.all(np.diff(np.asarray(grid)) > 0.0)


class TestOldSFHOnAge0Grid:
    def test_mass_stays_old(self):
        """Peak-10-Gyr snorm: <5% of the mass may sit at template ages <100 Myr.

        Pre-fix this fraction was 1.0000 — every parcel landed on the age=0
        node because the SFH was evaluated on the ``[0, ..., 0, agemax]`` grid.
        """
        ssp = _synthetic_ssp(prepend_age_zero=True)
        state = _old_snorm_model(ssp).predict_state({})
        aw = np.asarray(state.derived["age_weights"])
        ages_yr = np.asarray(10.0 ** (ssp.ssp_lg_age_gyr + 9.0))  # -inf -> 0
        young = np.nansum(np.where(ages_yr < 1e8, aw, 0.0))
        assert young / aw.sum() < 0.05

    def test_age0_grid_matches_clean_grid(self):
        """Same SFH on the age=0 grid vs the grid without that template.

        Templates older than the smallest positive age must carry identical
        weights. The ``[0, age_1]`` sliver differs by construction — the
        age=0 grid CIC-splits it between the age=0 node and its neighbor,
        the clean grid assigns it wholly to the youngest node (#538) — so
        the young end is compared as a sum. Either way the sliver is ~1e-5
        of the total for this SFH.
        """
        aw0 = np.asarray(
            _old_snorm_model(_synthetic_ssp(prepend_age_zero=True))
            .predict_state({})
            .derived["age_weights"]
        )
        aw1 = np.asarray(
            _old_snorm_model(_synthetic_ssp(prepend_age_zero=False))
            .predict_state({})
            .derived["age_weights"]
        )
        np.testing.assert_allclose(aw0[2:], aw1[1:], rtol=1e-3, atol=0.0)
        assert aw0[0] + aw0[1] == pytest.approx(aw1[0], rel=1e-3)
        assert aw0[0] < 1e-4 * aw0.sum()  # the age=0 template holds ~no mass
