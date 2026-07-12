# SPDX-License-Identifier: BSD-3-Clause
"""Regression: ``StellarSEDComponent`` builds ``sed_intrinsic`` from the
trapezoid-contract ``Σ(weights × ssp_flux) × total_mass`` reconstruction —
**not** from DSPS's own ``rest_sed`` (= ``sed_unit_mstar × mstar_obs``).

History: #394 originally switched ``sed_intrinsic`` TO ``dsps_result.rest_sed()``
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

1. ``dsps_result.rest_sed()`` and the einsum reconstruction with ``total_mass``
   agree at moderate ``t_obs`` (here 5 Gyr) — the quadratures only diverge at
   low z (large ``t_obs``); see #616.

2. ``StellarSEDComponent`` wires ``sed_intrinsic`` to the ``total_mass``
   reconstruction (the contract path), not ``dsps_result.rest_sed()``.
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
    """``dsps_result.rest_sed() = Σ_m,a (weights × ssp_flux) × mstar_obs``.
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
    rest_sed = res.rest_sed()

    # Strip leading SSP wavelengths where the flux is identically zero
    # (some libraries pad below Lyman alpha).
    mask = rest_sed > 0
    ratio = einsum_recon[mask] / rest_sed[mask]
    # The two paths should agree to ~1e-3 (limited by t_obs interp
    # precision in DSPS's mstar_obs vs trapezoid here). If a per-age
    # correction were silently in DSPS's rest_sed, this would fail with
    # a wavelength-dependent 2x-3x spread — see #394 audit table.
    assert jnp.allclose(ratio, 1.0, rtol=1e-3), (
        f"einsum recon vs dsps.rest_sed() disagree: "
        f"min ratio = {float(jnp.min(ratio)):.4f}, "
        f"max ratio = {float(jnp.max(ratio)):.4f}"
    )


def test_sed_intrinsic_uses_total_mass_reconstruction():
    """``StellarSEDComponent.apply()`` must build ``sed_intrinsic`` from the
    ``total_mass`` reconstruction (= ``Σ_age lnu_age``), NOT from DSPS's
    ``rest_sed`` (= ``mstar_obs``-normalized). Pin the wire by re-reading the
    source so a future change can't silently re-introduce the #394 path that
    breaks the trapezoid normalization contract at low z (#616)."""
    import inspect

    from tengri.components.stellar import component

    src = inspect.getsource(component)
    assert "sed_intrinsic = lnu_age.sum(axis=0)" in src, (
        "StellarSEDComponent no longer builds sed_intrinsic from the total_mass "
        "reconstruction (Σ_age lnu_age) — this would re-introduce the #394 "
        "dsps_result.rest_sed() path that violates the trapezoid normalization "
        "contract by up to ~6.6% at low z (see #616)."
    )
    assert "sed_intrinsic = dsps_result.rest_sed()" not in src, (
        "sed_intrinsic must not use DSPS's mstar_obs-normalized rest_sed (reversed in #616)."
    )
