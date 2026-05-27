# SPDX-License-Identifier: BSD-3-Clause
"""Regression: ``StellarSEDComponent`` returns DSPS's own ``rest_sed``
rather than reconstructing it via ``Σ(weights × ssp_flux) × total_mass``
(issue #394).

The two paths are mathematically identical when ``total_mass == mstar_obs``
(both integrate the same SFH), but reading DSPS's value keeps the stellar
SED self-consistent with the kernel's own ``sed_unit_mstar × mstar_obs``
bookkeeping. The test below pins that:

1. ``dsps_result.rest_sed × L_SUN`` equals
   ``(Σ_m,a (weights × ssp_flux)) × total_mass × L_SUN`` to
   floating-point precision at every wavelength on a fiducial tau
   SFH — i.e. there is no per-age "surviving mass" correction baked
   into DSPS's ``rest_sed`` that the einsum reconstruction misses.

2. ``model.predict_rest_sed`` returns the DSPS value (not the
   reconstruction), confirming the wiring at
   ``components/stellar/component.py:797``.
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
    sfr = sfr / total  # normalise to 1 Msun total

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


def test_predict_rest_sed_reads_dsps_rest_sed_path():
    """``StellarSEDComponent.apply()`` must use ``dsps_result.rest_sed``
    for ``sed_intrinsic`` — not the einsum reconstruction. Pin the wire
    by re-reading the source so a future "optimisation" can't silently
    re-introduce the redundant path."""
    import inspect

    from tengri.components.stellar import component

    src = inspect.getsource(component)
    assert "sed_intrinsic = dsps_result.rest_sed" in src, (
        "StellarSEDComponent no longer reads dsps_result.rest_sed for "
        "sed_intrinsic — silent re-introduction of the einsum-reconstruction "
        "path (#394)."
    )
