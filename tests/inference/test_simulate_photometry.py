# SPDX-License-Identifier: BSD-3-Clause
"""Tests for simulate.photometry_from_sfh filter handling."""

import chex
import jax.numpy as jnp
import pytest

from tengri import load_filter_set
from tengri.analysis.simulate import photometry_from_sfh, sed_from_sfh

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def sdss_r_only():
    return load_filter_set(["sdss_r"])


def test_sed_from_sfh_runs(synthetic_ssp):
    t = jnp.linspace(0.5, 10.0, 50)
    sfr = jnp.exp(-t / 3.0) + 0.01
    out = sed_from_sfh(t, sfr, synthetic_ssp, log_z=-0.3)
    assert "sed" in out and out["sed"].shape == synthetic_ssp.ssp_wave.shape


def test_photometry_accepts_load_filter_set_tuple(synthetic_ssp, sdss_r_only):
    t = jnp.linspace(0.5, 10.0, 50)
    sfr = jnp.exp(-t / 3.0) + 0.01
    out = photometry_from_sfh(
        t,
        sfr,
        synthetic_ssp,
        sdss_r_only,
        log_z=-0.3,
        redshift=0.1,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        apply_igm=False,
    )
    assert out["flux"].shape == (1,)


def test_photometry_accepts_filter_curve_list(synthetic_ssp, sdss_r_only):
    t = jnp.linspace(0.5, 10.0, 50)
    sfr = jnp.exp(-t / 3.0) + 0.01
    curves = sdss_r_only[2]
    out = photometry_from_sfh(
        t,
        sfr,
        synthetic_ssp,
        curves,
        log_z=-0.3,
        redshift=0.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        apply_igm=False,
    )
    assert out["flux"].shape == (1,)


def test_filter_waves_invalid_type_raises(synthetic_ssp):
    """_filter_waves_and_trans raises TypeError for non-FilterCurve input."""
    from tengri.analysis.simulate import photometry_from_sfh

    t = jnp.linspace(0.5, 10.0, 20)
    sfr = jnp.ones(20)
    with pytest.raises(TypeError, match="filters must be"):
        photometry_from_sfh(t, sfr, synthetic_ssp, "not_a_filter")


def test_sed_from_sfh_array_metallicity(synthetic_ssp):
    """sed_from_sfh works with an array log_z history."""
    from tengri.analysis.simulate import sed_from_sfh

    t = jnp.linspace(0.5, 10.0, 50)
    sfr = jnp.exp(-t / 3.0) + 0.01
    # Linearly enriching metallicity history
    log_z_arr = jnp.linspace(-2.0, -0.3, 50)
    out = sed_from_sfh(t, sfr, synthetic_ssp, log_z=log_z_arr)
    assert "sed" in out
    assert out["sed"].shape == synthetic_ssp.ssp_wave.shape
    chex.assert_tree_all_finite(out["sed"])


def test_sed_from_sfh_with_dust(synthetic_ssp):
    """sed_from_sfh applies dust attenuation when tau_bc > 0."""
    from tengri.analysis.simulate import sed_from_sfh

    t = jnp.linspace(0.5, 10.0, 50)
    sfr = jnp.exp(-t / 3.0) + 0.01
    out_no_dust = sed_from_sfh(t, sfr, synthetic_ssp, log_z=-0.3, dust_tau_bc=0.0)
    out_dust = sed_from_sfh(t, sfr, synthetic_ssp, log_z=-0.3, dust_tau_bc=1.0, dust_tau_diff=0.5)
    # Dust reduces UV flux
    chex.assert_tree_all_finite(out_dust["sed"])
    assert float(jnp.sum(out_dust["sed"])) < float(jnp.sum(out_no_dust["sed"]))


def test_photometry_with_igm(synthetic_ssp, sdss_r_only):
    """photometry_from_sfh with apply_igm=True runs without error at nonzero z."""
    out = photometry_from_sfh(
        jnp.linspace(0.5, 10.0, 50),
        jnp.exp(-jnp.linspace(0.5, 10.0, 50) / 3.0) + 0.01,
        synthetic_ssp,
        sdss_r_only,
        log_z=-0.3,
        redshift=0.5,
        apply_igm=True,
    )
    assert out["flux"].shape == (1,)
    chex.assert_tree_all_finite(out["flux"])


def test_spectrum_from_sfh_basic(synthetic_ssp):
    """spectrum_from_sfh returns observed spectrum on requested wavelength grid."""
    from tengri.analysis.simulate import spectrum_from_sfh

    t = jnp.linspace(0.5, 10.0, 50)
    sfr = jnp.exp(-t / 3.0) + 0.01
    wave_obs = jnp.linspace(3500.0, 9500.0, 200)
    out = spectrum_from_sfh(t, sfr, synthetic_ssp, wave_obs, log_z=-0.3, redshift=0.0)
    assert "flux" in out
    assert out["flux"].shape == (200,)
    chex.assert_tree_all_finite(out["flux"])


def test_spectrum_from_sfh_with_igm(synthetic_ssp):
    """spectrum_from_sfh applies IGM at z>0."""
    from tengri.analysis.simulate import spectrum_from_sfh

    t = jnp.linspace(0.5, 10.0, 50)
    sfr = jnp.exp(-t / 3.0) + 0.01
    wave_obs = jnp.linspace(4000.0, 9000.0, 150)
    out = spectrum_from_sfh(t, sfr, synthetic_ssp, wave_obs, log_z=-0.3, redshift=0.3)
    assert out["flux"].shape == (150,)
    chex.assert_tree_all_finite(out["flux"])
