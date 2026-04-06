"""Tests for simulate.photometry_from_sfh filter handling."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tengri import load_filter_set
from tengri.models.sps.dsps_wrapper import SSPData
from tengri.simulate import photometry_from_sfh, sed_from_sfh


@pytest.fixture(scope="module")
def synthetic_ssp():
    """Minimal synthetic SSP (matches other unit tests)."""
    n_met, n_age, n_wave = 3, 20, 100
    wave = jnp.linspace(3000.0, 10000.0, n_wave)
    ages_gyr = jnp.linspace(-1.0, 1.14, n_age)
    key = jax.random.PRNGKey(123)
    flux = jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-5
    lgmet = jnp.array([-1.5, -0.5, 0.0])
    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux,
        ssp_lg_age_gyr=ages_gyr,
        ssp_lgmet=lgmet,
    )


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
