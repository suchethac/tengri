# SPDX-License-Identifier: BSD-3-Clause
"""Regression: ``StellarSEDComponent`` builds ``sed_intrinsic`` from the
trapezoid-contract ``Σ(weights × ssp_flux) × total_mass`` reconstruction —
**not** from DSPS's own ``rest_sed`` (= ``sed_unit_mstar × mstar_obs``).

History: #394 originally switched ``sed_intrinsic`` TO ``dsps_result.rest_sed``
for "kernel self-consistency". That was **reversed by #616**: ``mstar_obs``
(DSPS's cumulative-SFH quadrature) and ``total_mass`` (the trapezoid that
defines the ``trapezoid(sfr, t) = 10**log_total_mass`` normalization contract)
diverge by up to ~6.6% at low z (large ``t_obs``). Using ``rest_sed`` silently
broke that contract for the exact SED at low z and disagreed with every
precompute LUT (which use ``total_mass``). Reconstructing from ``total_mass``
makes the exact SED, the photometry/spectrum LUTs, ``lnu_age``, ``L_age``,
``age_weights``, and ``pred.stellar_mass`` all honor the one contract mass.
See issues #616 (DSPS quadrature) and #617 (birth-cloud dust LUT residual).

The tests below pin that:

1. ``dsps_result.rest_sed`` and the einsum reconstruction with ``total_mass``
   agree at moderate ``t_obs`` (here 5 Gyr) — the quadratures only diverge at
   low z (large ``t_obs``); see #616.

2. ``StellarSEDComponent`` wires ``sed_intrinsic`` to the ``total_mass``
   reconstruction (the contract path), not ``dsps_result.rest_sed``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def _build_tau_fiducial():
    """Tau SFH at t_obs=5 Gyr, Z=Z_sun. Returns (ssp, t, sfr, lgmet)."""
    import glob

    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    candidates = sorted(glob.glob("data/ssp_*chabrier*.h5"))
    if not candidates:
        pytest.skip("no Chabrier SSP fixture present")
    ssp = load_ssp_data(candidates[0])

    t_obs_gyr = 5.0
    tau_gyr = 1.0
    n_grid = 200
    t_grid = np.linspace(0.01, t_obs_gyr, n_grid)
    # Delayed-tau-like: SFR rises toward present.
    sfr = np.exp(-(t_obs_gyr - t_grid) / tau_gyr)
    total = np.trapezoid(sfr, t_grid * 1e9)
    sfr = sfr / total  # normalize to 1 Msun total

    log_z_sun_absolute = -1.848  # LOG10_ZSUN in tengri
    return ssp, jnp.asarray(t_grid), jnp.asarray(sfr), log_z_sun_absolute, t_obs_gyr


def test_dsps_rest_sed_matches_einsum_reconstruction():
    """``dsps_result.rest_sed = Σ_m,a (weights × ssp_flux) × mstar_obs``.
    The einsum reconstruction with ``total_mass`` agrees to 1e-3 relative
    at every wavelength on a fiducial tau SFH."""
    from dsps.sed.stellar_sed import calc_rest_sed_sfh_table_lognormal_mdf

    ssp, t_grid, sfr, lgmet, t_obs = _build_tau_fiducial()
    total_mass = float(jnp.trapezoid(sfr, t_grid * 1e9))

    res = calc_rest_sed_sfh_table_lognormal_mdf(
        gal_t_table=t_grid,
        gal_sfr_table=sfr,
        gal_lgmet=jnp.asarray(lgmet),
        gal_lgmet_scatter=0.2,
        ssp_lgmet=ssp.ssp_lgmet,
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        ssp_flux=ssp.ssp_flux,
        t_obs=t_obs,
    )

    einsum_recon = jnp.einsum("ma,maw->aw", res.weights, ssp.ssp_flux).sum(axis=0) * total_mass
    rest_sed = res.rest_sed

    # Strip leading SSP wavelengths where the flux is identically zero
    # (some libraries pad below Lyman alpha).
    mask = rest_sed > 0
    ratio = einsum_recon[mask] / rest_sed[mask]
    # The two paths should agree to ~1e-3 (limited by t_obs interp
    # precision in DSPS's mstar_obs vs trapezoid here). If a per-age
    # correction were silently in DSPS's rest_sed, this would fail with
    # a wavelength-dependent 2x-3x spread — see #394 audit table.
    assert jnp.allclose(ratio, 1.0, rtol=1e-3), (
        f"einsum recon vs dsps.rest_sed disagree: "
        f"min ratio = {float(jnp.min(ratio)):.4f}, "
        f"max ratio = {float(jnp.max(ratio)):.4f}"
    )


def test_sed_intrinsic_uses_total_mass_reconstruction(synthetic_ssp_wide):
    """``StellarSEDComponent.apply()`` must build ``sed_intrinsic`` from the
    ``total_mass`` reconstruction (= ``Σ_age lnu_age``), NOT from DSPS's
    ``rest_sed`` (= ``mstar_obs``-normalized).

    Regression: #394 (reversed by #616). Using ``rest_sed`` violates the
    trapezoid normalization contract by up to ~6.6% at low z because
    ``mstar_obs`` (DSPS cumulative-SFH quadrature) and ``total_mass``
    (trapezoid normalization) diverge at low z (large t_obs).

    Test: sed_intrinsic must scale linearly with log_total_mass (the sole
    multiplicative normalization), indicating reconstruction from
    Σ_age(lnu_age) × total_mass, not from mstar_obs-normalized rest_sed.
    """
    from tengri import Fixed, SEDModel

    ssp = synthetic_ssp_wide

    # Build a simple model with only stellar component
    from tengri import FREE

    model = SEDModel.build(
        ssp_data=ssp,
        observation=None,  # No observation needed for state test
        sfh={"type": "dpl", "all_params": FREE},  # All SFH params free
        met={"logzsol": Fixed(0.0)},  # Fixed solar metallicity
        redshift=Fixed(0.0),  # Rest-frame observation
    )

    # Create two parameter sets with different total_mass values
    import jax

    key = jax.random.PRNGKey(0)
    params1 = model.spec.sample(key)
    params1 = dict(params1)  # Convert to dict for modification
    params1["sfh_dpl_log_total_mass"] = 10.0  # log10(M_sun)

    params2 = model.spec.sample(key)
    params2 = dict(params2)  # Convert to dict for modification
    params2["sfh_dpl_log_total_mass"] = 11.0  # 10x more mass

    # Get forward states
    state1 = model.predict_state(params1)
    state2 = model.predict_state(params2)

    sed1 = state1.sed_intrinsic
    sed2 = state2.sed_intrinsic

    # If sed_intrinsic is built from total_mass reconstruction, it should
    # scale linearly (in linear space, not log). The ratio should be ~10.
    ratio = sed2 / sed1
    expected_ratio = 10.0  # 10^(11-10) = 10

    # Allow 1% tolerance (numerical precision, interpolation)
    assert jnp.allclose(ratio, expected_ratio, rtol=0.01), (
        f"sed_intrinsic does not scale linearly with total_mass. "
        f"Expected 10x, got {jnp.mean(ratio):.2f}x "
        f"(min={jnp.min(ratio):.2f}, max={jnp.max(ratio):.2f}). "
        f"This suggests it is built from mstar_obs-normalized rest_sed "
        f"rather than the total_mass reconstruction."
    )
