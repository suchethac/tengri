# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #1016 — zero stellar mass scale on age-0-anchored SSPs.

Root cause: ``_refine_sfh_table_ages`` builds its dense CIC integrand grid as
``10 ** linspace(log10(ssp_ages_yr[0]), log10(ssp_ages_yr[-1]))``. SSP grids
with an age-0 anchor template (bc03 stelib: ``ssp_ages_yr[0] = 0``) make
``log10(0) = -inf``, collapsing the refined grid to ``[0, 0, ..., age_max]``.
The SFR evaluated there is zero everywhere with support, so
``_age_weights_cic`` returns ``total_mass = 0`` and all-zero age weights —
the entire stellar SED silently vanishes (photometry ~45 dex low), while the
published ``sfr_history`` (computed on the healthy lookback grid) still
integrates to exactly ``10**log_total_mass``.

Same failure family as #1001 (the age-0 anchor's -inf breaking log-space
arithmetic); ``_cic_parcels`` already special-cases the anchor, this
refinement helper was missed.

https://github.com/suchethac/tengri/issues/1016
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.component import _age_weights_cic, _refine_sfh_table_ages
from tengri.components.stellar.sfh.mean_sfh import delayed_exponential

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def bc03_like_ages():
    """SSP age grid with a leading age-0 anchor template (bc03 stelib shape)."""
    return jnp.concatenate([jnp.zeros(1), 10.0 ** jnp.linspace(5.1, 10.301, 106)])


class TestRefinedGridOnAgeZeroAnchor:
    """#1016: the dense integrand grid must survive an age-0 first template."""

    def test_refined_grid_finite_positive_ascending(self, bc03_like_ages):
        """The refined grid must be finite, positive past the floor, strictly
        ascending, and still span up to the oldest template age."""
        fine = _refine_sfh_table_ages(bc03_like_ages)
        f = np.asarray(fine)
        assert np.all(np.isfinite(f)), "refined grid contains non-finite ages"
        assert np.all(np.diff(f) > 0.0), "refined grid is not strictly ascending"
        assert f[-1] == pytest.approx(float(bc03_like_ages[-1]), rel=1e-10)
        # The grid must actually resolve the SSP age span — the bug collapsed
        # every interior node to zero.
        assert np.sum((f > 1e5) & (f < 1e10)) > 100, "interior of the age span unresolved"

    def test_refined_grid_unchanged_without_anchor(self):
        """Grids without an age-0 anchor are byte-identical to the old
        behavior (the guard must be a no-op there)."""
        ages = 10.0 ** jnp.linspace(5.1, 10.301, 107)
        fine = _refine_sfh_table_ages(ages)
        n = ages.shape[0]
        expected = 10.0 ** jnp.linspace(jnp.log10(ages[0]), jnp.log10(ages[-1]), (n - 1) * 16 + 1)
        np.testing.assert_allclose(np.asarray(fine), np.asarray(expected), rtol=1e-12)

    def test_cic_mass_conserved_on_age_zero_anchor(self, bc03_like_ages):
        """The CIC weights on the refined grid must recover the SFH mass.

        Conservation: ``total_mass = trapezoid(SFR, t)`` on the dense grid
        must equal ``10**log_total_mass`` (the stellar-normalization
        contract) — the bug returned exactly 0.
        """
        fine = _refine_sfh_table_ages(bc03_like_ages)
        sfr = delayed_exponential(fine, log_total_mass=7.153, tau=1.368e9, start=0.0)
        weights, total_mass = _age_weights_cic(fine, sfr, bc03_like_ages, 13.16)
        np.testing.assert_allclose(float(total_mass), 10.0**7.153, rtol=1e-2)
        np.testing.assert_allclose(float(jnp.sum(weights)), 1.0, rtol=1e-10)
        assert np.all(np.isfinite(np.asarray(weights)))


class TestQuiescentMassScaleEndToEnd:
    """#1016 end-to-end (data-gated on the bc03 grid)."""

    def test_quiescent_z0_mass_formed_honors_contract(self, ssp_data_bc03, synthetic_tophat_obs):
        """log_mstar_formed must equal the sampled log_total_mass (the mass
        contract), and the photometry must sit at a physical flux scale —
        the bug gave log_mstar_formed = -30 and ~1e-72 fluxes."""
        from tengri import SEDModel, recipes

        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            observation=synthetic_tophat_obs,
            **recipes.quiescent_z0(),
        )
        params = model.spec.sample(jax.random.PRNGKey(0))
        state = model.predict_state(params)
        log_mstar_formed = float(state.derived["log_mstar_formed"])
        expected = float(params["sfh_dexp_log_total_mass"])
        np.testing.assert_allclose(log_mstar_formed, expected, atol=0.05)

        # Surviving mass: the age-0 anchor also poisoned the synthesized
        # mass-remaining table (NaN at lg_age = -inf), making log_mstar NaN
        # for every bc03 build. Finite, and bounded by the formed mass.
        log_mstar = float(state.derived["log_mstar"])
        assert np.isfinite(log_mstar), "surviving log_mstar is not finite"
        assert log_mstar <= log_mstar_formed + 1e-6, "surviving mass exceeds formed mass"

        flux = np.asarray(model.predict_photometry(params))
        assert np.all(np.isfinite(flux))
        # Generous physical window for a 10^6-10^9 Msun galaxy at z=0.05:
        # catches scale collapses (the bug: ~1e-72) without pinning exact SED
        # values on a specific SSP file.
        assert 1e-35 < flux.max() < 1e-15, f"photometry at unphysical scale: {flux}"
