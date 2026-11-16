# SPDX-License-Identifier: BSD-3-Clause
"""#3541: SpectrumPrecomp matches the exact spectrum at an off-node metallicity.

Since #3528 the stellar component reads the native SSP flux across metallicity by
monotone PCHIP, but the spectrum table ``ssp_on_pixels`` is resampled to the pixels and
then PCHIP-read over its own stored values. PCHIP's slope limiter is nonlinear in the
data, so the two differ (2.7e-2 here; 6e-15 before #3528). The fix (owner ruling
2026-11-15, "Exact: native slopes") is on hold, so this case is an expected failure
until it lands.

The model is the synthetic three-node grid at absolute log Z -1.848 (between the -2
and -1 nodes): ``met_logzsol`` is solar-relative, so 0.0 gives that value.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    SpectrumPrecomp,
    Uniform,
    WavePrecomp,
)
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation import Spectroscopy
from tengri.observation.line_list import LineList

pytestmark = pytest.mark.regression_bug

#: Relative tolerance of a table read against the exact read (round-off).
RTOL = 1e-12
_MET_LOGZSOL = 0.0  # absolute log Z -1.848, between the -2 and -1 nodes
_MET_LOGZSOL_HOSTED = -0.052  # absolute log Z -1.900, between the hosted -2.0 and -1.85387 nodes
_Z = 0.1
PHOT_BANDS = ["galex_fuv", "sdss_u", "sdss_r"]


def _synthetic_ssp():
    return SSPData(
        ssp_wave=jnp.linspace(3000.0, 10000.0, 100),
        ssp_flux=jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (3, 20, 100))) * 1e-3 + 1e-5,
        ssp_lg_age_gyr=jnp.linspace(-1.0, 1.14, 20),
        ssp_lgmet=jnp.array([-2.0, -1.0, 0.0]),
    )


def _spectro_obs():
    wave = np.linspace(5200.0, 7400.0, 2000)
    sp = Spectroscopy(
        wave_obs=jnp.asarray(wave),
        resolution=1500.0,
        eline_mode="marginalized",
        eline_catalog=LineList.from_names(["Hbeta", "OIII_4959", "OIII_5007", "Halpha"]),
        eline_prior_sigma=1.0e-14,
        # SpectrumPrecomp samples the model at the pixel centers, so the exact path it is
        # compared with must too. resample="auto" resolves to the flux-conserving pixel
        # integral on a fine-grained hosted grid, which the LUT refuses; this test is about the
        # metallicity read, not the resample, so it states the one both routes share.
        resample="point",
    )
    return Observation(spectroscopy=sp)


def _sfh():
    return {
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_frac": Fixed(0.6),
        "log_total_mass": Fixed(10.0),
        "all_params": Fixed(DEFAULT),
        "age_kernel": "cic",
    }


def _build(*, approx, ssp=None, met_logzsol=_MET_LOGZSOL):
    return SEDModel.build(
        ssp_data=_synthetic_ssp() if ssp is None else ssp,
        observation=_spectro_obs(),
        approx=approx,
        redshift=Fixed(0.1),
        sfh=_sfh(),
        met={"type": "delta", "met_logzsol": Fixed(met_logzsol), "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
    )


def _max_rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def test_spectrum_lut_matches_exact_at_off_node_z():
    exact = np.asarray(_build(approx=None).predict_spectrum({}))
    lut = np.asarray(_build(approx=SpectrumPrecomp()).predict_spectrum({}))
    assert _max_rel(lut, exact) <= RTOL


# ── (b) WavePrecomp photometry at an off-node Z, fixed and free redshift ─────


def _phot_obs():
    return Observation(photometry=Photometry.from_names(PHOT_BANDS))


def _build_phot(ssp, *, approx, met_logzsol, redshift=None):
    redshift = Fixed(_Z) if redshift is None else redshift
    return SEDModel.build(
        ssp_data=ssp,
        observation=_phot_obs(),
        approx=approx,
        redshift=redshift,
        sfh=_sfh(),
        met={"type": "delta", "met_logzsol": Fixed(met_logzsol), "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
    )


def test_wave_photometry_matches_exact_at_off_node_z_fixed_z():
    ssp = _synthetic_ssp()
    exact = np.asarray(
        _build_phot(ssp, approx=None, met_logzsol=_MET_LOGZSOL).predict_photometry({})
    )
    lut = np.asarray(
        _build_phot(ssp, approx=WavePrecomp(), met_logzsol=_MET_LOGZSOL).predict_photometry({})
    )
    assert _max_rel(lut, exact) <= RTOL


def test_wave_photometry_matches_exact_at_off_node_z_free_z():
    """Free z: exact at a z-table node, and within the z-interpolation floor between nodes.

    The z-table is read in z by a cubic between its nodes, which is an approximation of the
    exact photometry with no metallicity read at all (a one-metallicity control gives the same
    1.0e-6 residual between nodes). So the metallicity contract is exact at the nodes, where
    the z read is exact, and the off-node bound sits above that floor and far below the
    1.5e-3 of the bug.
    """
    ssp = _synthetic_ssp()
    free = Uniform(0.05, 0.15)
    exact_model = _build_phot(ssp, approx=None, met_logzsol=_MET_LOGZSOL, redshift=free)
    lut_model = _build_phot(ssp, approx=WavePrecomp(), met_logzsol=_MET_LOGZSOL, redshift=free)
    z_grid = np.asarray(_lut_state(lut_model).ssp_phot_ztable.z_grid)
    inside = z_grid[(z_grid > 0.05) & (z_grid < 0.15)]
    z_node = float(inside[len(inside) // 2])
    for z, bound in [(z_node, RTOL), (0.09, 1e-5)]:
        params = {"redshift": jnp.asarray(z)}
        exact = np.asarray(exact_model.predict_photometry(params))
        lut = np.asarray(lut_model.predict_photometry(params))
        assert _max_rel(lut, exact) <= bound, f"z={z}"


def _lut_state(model):
    return next(
        c for c in model._build_component_chain() if type(c).__name__.startswith("Stellar")
    )._state


# ── (c) the hosted default grid, off-node Z ──────────────────────────────────


@pytest.mark.slow
def test_hosted_grid_spectrum_lut_matches_exact_at_off_node_z():
    import tengri

    ssp = tengri.load_ssp()
    exact = np.asarray(
        _build(approx=None, ssp=ssp, met_logzsol=_MET_LOGZSOL_HOSTED).predict_spectrum({})
    )
    lut = np.asarray(
        _build(
            approx=SpectrumPrecomp(), ssp=ssp, met_logzsol=_MET_LOGZSOL_HOSTED
        ).predict_spectrum({})
    )
    assert _max_rel(lut, exact) <= RTOL


# ── (d), (e): the met_pchip contract ─────────────────────────────────────────


def _nodes(n):
    return jnp.linspace(-2.0, 0.0, n) if n > 1 else jnp.array([-2.0])


@pytest.mark.parametrize("n_met", [1, 2, 3, 4, 5, 9])
def test_hermite_read_equals_pchip_read(n_met):
    from tengri.components.stellar.sps import met_pchip as mp

    x = _nodes(n_met)
    rng = np.random.default_rng(n_met)
    table = jnp.asarray(np.abs(rng.normal(size=(n_met, 3, 4))) + 0.1)
    slopes = mp.pchip_node_slopes(table, x)
    for z in np.linspace(-2.2, 0.2, 37):
        ref = mp.read_met(table, x, jnp.asarray(z))
        got = mp.read_met_hermite(table, slopes, x, jnp.asarray(z))
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=RTOL, atol=0.0)
    for k in range(n_met):
        got = mp.read_met_hermite(table, slopes, x, x[k])
        np.testing.assert_array_equal(np.asarray(got), np.asarray(table[k]))


@pytest.mark.parametrize("n_met", [2, 5])
def test_hermite_per_age_read_equals_pchip_per_age_read(n_met):
    from tengri.components.stellar.sps import met_pchip as mp

    x = _nodes(n_met)
    rng = np.random.default_rng(10 + n_met)
    table = jnp.asarray(np.abs(rng.normal(size=(n_met, 6, 4))) + 0.1)  # (met, age, wave)
    slopes = mp.pchip_node_slopes(table, x)
    z_age = jnp.linspace(-1.9, -0.1, 6)
    ref = mp.read_met_per_age(table, x, z_age, age_axis=1)
    got = mp.read_met_hermite_per_age(table, slopes, x, z_age, age_axis=1)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=RTOL, atol=0.0)
