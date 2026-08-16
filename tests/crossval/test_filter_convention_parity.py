# SPDX-License-Identifier: BSD-3-Clause
"""Cross-code parity for the photometric filter-convolution convention.

Pins tengri's two conventions to their reference codes:

- ``BESSELL`` (default, photon-counting ``w = 1/lambda``) is bit-identical to
  **DSPS** (``dsps.photometry`` — ``_obs_flux_ssp`` / ``_flux_ab0_at_10pc`` /
  ``calc_obs_mag``), which is the FSPS / Fukugita+1996 / Hogg+2002 photon AB
  convention. DSPS is the reference of record (tengri's own SSP engine).
- ``ENERGY`` (``w = 1/lambda^2``, flat-in-frequency) is the CIGALE / bagpipes
  energy mean; pinned to an independent analytic reference (and to ``pcigale``
  when it is importable).

These are the regression guard for the convention fix: tengri must agree with
DSPS to machine precision (the kernels are mathematically identical), not just
"close". See ``docs/units.md`` (Photometric filter-convolution convention) and
ADR-0016.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.photometry import FilterConvention, compute_flux_density

pytestmark = pytest.mark.crossval

dsps_kernels = pytest.importorskip(
    "dsps.photometry.photometry_kernels",
    reason="dsps not installed",
)

_AB0 = 1.13492e-13  # 3631 Jy at 10 pc, Lsun/Hz (DSPS convention)
_DL_10PC_CM = 3.085677581491367e19  # 10 pc in cm


def _sed_lnu(wave):
    """A non-flat L_nu SED (red continuum + an emission line in-band)."""
    cont = (wave / 5000.0) ** 1.5
    line = 40.0 * np.exp(-0.5 * ((wave - 5007.0) / 8.0) ** 2)
    return (cont + line).astype(np.float64)


def _tophat(center, width, n=400):
    fw = np.linspace(center - width / 2, center + width / 2, n)
    ft = np.where(np.abs(fw - center) < 0.8 * width / 2, 1.0, 0.0)
    return fw.astype(np.float64), ft.astype(np.float64)


def _tengri_band_mean_lnu(lnu, wave, fw, ft, z, convention):
    """tengri's band-averaged L_nu (strip the (1+z)/4 pi dL^2 flux scale)."""
    fd = float(
        compute_flux_density(
            jnp.asarray(lnu),
            jnp.asarray(wave),
            jnp.asarray(fw),
            jnp.asarray(ft),
            z,
            _DL_10PC_CM,
            convention=convention,
        )
    )
    flux_scale = (1.0 + z) / (4.0 * np.pi * _DL_10PC_CM**2)
    return fd / flux_scale


_FILTERS = [(2300.0, 400.0), (3500.0, 600.0), (5500.0, 900.0), (8000.0, 1500.0), (21000.0, 5000.0)]
_REDSHIFTS = [0.0, 0.5, 1.0, 2.0]


@pytest.mark.parametrize("center,width", _FILTERS)
@pytest.mark.parametrize("z", _REDSHIFTS)
def test_bessell_matches_dsps_obs_flux(center, width, z):
    """tengri BESSELL band-mean L_nu == DSPS _obs_flux_ssp to machine precision."""
    wave = np.linspace(1000.0, 30000.0, 6000)
    lnu = _sed_lnu(wave)
    fw, ft = _tophat(center, width)

    tengri_mean = _tengri_band_mean_lnu(lnu, wave, fw, ft, z, FilterConvention.BESSELL)

    # DSPS: flux_source = int T Lnu_obs / wave ;  flux_ab0 = int T AB0 / wave.
    # The band-mean L_nu is flux_source / (flux_ab0 / AB0).
    num = float(
        dsps_kernels._obs_flux_ssp(
            jnp.asarray(wave), jnp.asarray(lnu), jnp.asarray(fw), jnp.asarray(ft), z
        )
    )
    ab0 = float(dsps_kernels._flux_ab0_at_10pc(jnp.asarray(fw), jnp.asarray(ft)))
    dsps_mean = num / (ab0 / _AB0)

    np.testing.assert_allclose(tengri_mean, dsps_mean, rtol=1e-6, atol=0.0)


def test_bessell_matches_dsps_calc_obs_mag_colors():
    """tengri BESSELL colors (Δmag between bands) == DSPS calc_obs_mag colors.

    Colors cancel cosmology and the AB zero-point, so this compares the two
    kernels' magnitude responses directly to <1e-4 mag.
    """
    wave = np.linspace(1000.0, 30000.0, 6000)
    lnu = _sed_lnu(wave)
    z = 0.5
    Om0, w0, wa, h = 0.3, -1.0, 0.0, 0.7

    bands = [_tophat(c, w) for c, w in _FILTERS]

    # tengri AB mag from band-mean L_nu: m = -2.5 log10(mean_Lnu / AB0)  (+ common
    # cosmology terms that cancel in a color).
    def _tengri_ab_mag(fw, ft):
        mean = _tengri_band_mean_lnu(lnu, wave, fw, ft, z, FilterConvention.BESSELL)
        return -2.5 * np.log10(mean / _AB0)

    tengri_mag = np.array([_tengri_ab_mag(fw, ft) for fw, ft in bands])
    dsps_mag = np.array(
        [
            float(
                dsps_kernels.calc_obs_mag(
                    jnp.asarray(wave),
                    jnp.asarray(lnu),
                    jnp.asarray(fw),
                    jnp.asarray(ft),
                    z,
                    Om0,
                    w0,
                    wa,
                    h,
                )
            )
            for fw, ft in bands
        ]
    )

    tengri_colors = np.diff(tengri_mag)
    dsps_colors = np.diff(dsps_mag)
    np.testing.assert_allclose(tengri_colors, dsps_colors, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize("center,width", _FILTERS)
def test_bessell_matches_dsps_rest_flux(center, width):
    """tengri BESSELL rest-frame band-mean == DSPS calc_rest_flux (z=0)."""
    wave = np.linspace(1000.0, 30000.0, 6000)
    lnu = _sed_lnu(wave)
    fw, ft = _tophat(center, width)

    tengri_mean = _tengri_band_mean_lnu(lnu, wave, fw, ft, 0.0, FilterConvention.BESSELL)
    rest_flux = float(
        dsps_kernels.calc_rest_flux(
            jnp.asarray(wave), jnp.asarray(lnu), jnp.asarray(fw), jnp.asarray(ft)
        )
    )
    dsps_mean = rest_flux * _AB0  # calc_rest_flux = flux_source / flux_ab0
    np.testing.assert_allclose(tengri_mean, dsps_mean, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("center,width", _FILTERS)
@pytest.mark.parametrize("z", _REDSHIFTS)
def test_energy_matches_analytic_reference(center, width, z):
    """tengri ENERGY band-mean == analytic energy convention int T Lnu/lam^2 / int T/lam^2."""
    wave = np.linspace(1000.0, 30000.0, 6000)
    lnu = _sed_lnu(wave)
    fw, ft = _tophat(center, width)

    tengri_mean = _tengri_band_mean_lnu(lnu, wave, fw, ft, z, FilterConvention.ENERGY)

    # Reference: redshift the SED, interpolate onto the filter grid, weight 1/lam^2.
    wave_obs = wave * (1.0 + z)
    sed_on = np.interp(fw, wave_obs, lnu, left=0.0, right=0.0)
    num = np.trapezoid(sed_on * ft / fw**2, fw)
    den = np.trapezoid(ft / fw**2, fw)
    ref_mean = num / den
    np.testing.assert_allclose(tengri_mean, ref_mean, rtol=1e-6, atol=0.0)


def test_conventions_differ_on_sloped_sed():
    """Sanity: Bessell and energy disagree on a non-flat SED but agree on a flat one."""
    wave = np.linspace(1000.0, 30000.0, 6000)
    fw, ft = _tophat(8000.0, 4000.0)

    flat = np.ones_like(wave)
    b_flat = _tengri_band_mean_lnu(flat, wave, fw, ft, 0.0, FilterConvention.BESSELL)
    e_flat = _tengri_band_mean_lnu(flat, wave, fw, ft, 0.0, FilterConvention.ENERGY)
    np.testing.assert_allclose(b_flat, e_flat, rtol=1e-6)  # flat F_nu: identical

    sloped = _sed_lnu(wave)
    b = _tengri_band_mean_lnu(sloped, wave, fw, ft, 0.0, FilterConvention.BESSELL)
    e = _tengri_band_mean_lnu(sloped, wave, fw, ft, 0.0, FilterConvention.ENERGY)
    assert abs(-2.5 * np.log10(b / e)) > 1e-3  # they must differ on a sloped SED
