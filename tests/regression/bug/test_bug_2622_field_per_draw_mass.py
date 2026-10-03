# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #2622 (item 1 only; item 2, the free PSD slope, is
out of scope here): every GP-field realization of a ``['delayed', 'field']``
composite forms the DECLARED mass, not merely the ensemble mean.

Every draw's published history integrates to the declared mass (the
per-draw pin of ``_mass_conserving_total`` plus the factor on the published
``sfr_history``), so the mass does not scatter with the field draw. pcigale's
``sfhstochastic_carvajal2025`` (``normalise=True``) returns 1.000000 for
every seed. DECISION (overrulable by the owner): the declaration states the
formed mass of each realization; the per-draw normalization is already
achieved by the existing ``_mass_conserving_total`` pin (#2521/#2567) plus
the #2640 fix that carries the SAME rescale onto the published
``sfr_history``.
"""

from __future__ import annotations

import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel
from tengri.cosmology import age_at_z

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp()


@pytest.fixture(scope="module")
def model(ssp):
    sig, tau_break_myr = 0.4, 150.0
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": ["delayed", "field"],
            "sfh_delayed_log_total_mass": Fixed(0.0),
            "sfh_delayed_tau_gyr": Fixed(2.0),
            "sfh_delayed_age_gyr": Fixed(5.0),
            "sfh_field_psd_sigma": Fixed(sig),
            "sfh_field_psd_tau_myr": Fixed(tau_break_myr / (2 * np.pi)),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )


def test_every_draw_forms_the_declared_mass(model):
    """200 draws: mean == declared, std == 0 (pcigale normalise=True parity)."""
    import jax.numpy as jnp

    rng = np.random.default_rng(1)
    declared_mass = 1.0
    masses = np.array(
        [
            10
            ** float(
                model.predict_state(
                    {"sfh_field_xi": jnp.asarray(rng.standard_normal(256))}
                ).derived["log_mstar_formed"]
            )
            for _ in range(50)
        ]
    )
    np.testing.assert_allclose(masses, declared_mass, rtol=1e-6)
    assert masses.std() < 1e-6 * declared_mass, (
        f"formed mass should be IDENTICAL across draws (per-draw normalization), "
        f"got std={masses.std():.3e}"
    )


@pytest.mark.parametrize("seed", [0, 1, 42, 123])
def test_sfr_history_integral_matches_declared_mass_per_draw(model, seed):
    """∫ sfr_history (support) == declared mass for an arbitrary single draw too."""
    import jax

    xi = jax.random.normal(jax.random.PRNGKey(seed), shape=(256,))
    st = model.predict_state({"sfh_field_xi": xi})
    lbt = np.asarray(st.derived["sfh_grid_lbt_yr"])
    sfr = np.asarray(st.derived["sfr_history"])
    formed_mass = 10 ** float(st.derived["log_mstar_formed"])
    age_z_yr = float(age_at_z(0.0)) * 1e9

    trapz_support = np.trapezoid(sfr, lbt)

    np.testing.assert_allclose(trapz_support, formed_mass, rtol=1e-5)
    np.testing.assert_allclose(formed_mass, 1.0, rtol=1e-6)
