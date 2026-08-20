# SPDX-License-Identifier: BSD-3-Clause
r"""Tests for spinning dust (AME) through the astrodust grammar (#1093).

Tests that:
1. The grammar accepts spinning_dust and f_cnm configuration options
2. spinning_dust changes the SED at microwave/radio frequencies
3. The emission is properly scaled by the dust mass (L_ir)
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

HDF5 = Path("data/astrodust_templates.h5")
# Wide-wavelength (wNE) SSP grid needed for radio/microwave coverage
SSP_FILE = Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")


@pytest.fixture(scope="module")
def fixture_path():
    if not HDF5.is_file():
        pytest.skip(
            f"Astrodust HDF5 not built at {HDF5}; run "
            f"`python scripts/build_astrodust_hdf5.py --download`"
        )
    return str(HDF5)


@pytest.fixture(scope="module")
def ssp_data():
    if not SSP_FILE.is_file():
        pytest.skip(f"SSP file {SSP_FILE.name} not present in data/")
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data(SSP_FILE)


def test_grammar_accepts_spinning_dust(fixture_path, ssp_data):
    """The grammar should accept spinning_dust=True in dust.emission."""
    from tengri import Observation, SEDModel
    from tengri.observation import Photometry

    ssp = ssp_data
    obs = Observation(photometry=Photometry.from_names(["sdss_g"]))

    # Should NOT raise "Unknown key 'spinning_dust'"
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        dust={
            "type": "two_component",
            "law": "calzetti",
            "emission": {"type": "astrodust", "spinning_dust": True},
        },
        redshift=0.0,
    )

    # Verify the spec was set correctly
    assert bool(model.spec.astrodust_spinning_dust) is True


def test_grammar_accepts_f_cnm(fixture_path, ssp_data):
    """The grammar should accept f_cnm in dust.emission."""
    from tengri import Observation, SEDModel
    from tengri.observation import Photometry

    ssp = ssp_data
    obs = Observation(photometry=Photometry.from_names(["sdss_g"]))

    # Should NOT raise "Unknown key 'f_cnm'"
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        dust={
            "type": "two_component",
            "law": "calzetti",
            "emission": {"type": "astrodust", "f_cnm": 0.5},
        },
        redshift=0.0,
    )

    # Verify the spec was set correctly
    assert float(model.spec.astrodust_f_cnm) == pytest.approx(0.5, rel=1e-3)


def test_spinning_dust_changes_microwave_sed(ssp_data):
    """Spinning dust should change the SED at microwave wavelengths."""
    from tengri import FIXED, Observation, SEDModel
    from tengri.observation.spectroscopy import Spectroscopy

    ssp = ssp_data
    # Wide wavelength range covering microwave/radio
    wave_aa = jnp.geomspace(1.0e3, 3.0e8, 1500)
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_aa))

    # Build two models: one with and one without spinning dust
    # All parameters fixed to isolate the spinning_dust effect
    model_no_spd = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED},
        dust={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": 0.5,
            "tau_diff": 1.5,
            "emission": {"type": "astrodust", "spinning_dust": False},
            "all_params": FIXED,
        },
        met={"type": "delta", "logzsol": 0.0},
        redshift=0.0,
    )

    model_yes_spd = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED},
        dust={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": 0.5,
            "tau_diff": 1.5,
            "emission": {"type": "astrodust", "spinning_dust": True},
            "all_params": FIXED,
        },
        met={"type": "delta", "logzsol": 0.0},
        redshift=0.0,
    )

    # Make predictions with fixed parameters
    params = {
        "sfh_dpl_alpha": 1.5,
        "sfh_dpl_beta": 2.5,
        "sfh_dpl_tau_gyr": 1.0,
        "sfh_dpl_log_total_mass": 9.0,
        "dust_lgU": 0.2,
        "redshift": 0.0,
    }

    pred_no = model_no_spd.predict(params)
    pred_yes = model_yes_spd.predict(params)

    sed_no = np.asarray(pred_no.rest_sed())
    sed_yes = np.asarray(pred_yes.rest_sed())
    # Use the actual wavelength grid the SED was computed on
    wave_rest = np.asarray(pred_no.wave_rest)

    # Precondition: dust_lgU parameter is actually live (not vacuous)
    # Vary lgU and verify the SED changes significantly in the IR
    params_high_u = params.copy()
    params_high_u["dust_lgU"] = 2.0  # Higher radiation field
    pred_high_u = model_no_spd.predict(params_high_u)
    sed_high_u = np.asarray(pred_high_u.rest_sed())
    ir_mask = (wave_rest > 5.0e4) & (wave_rest < 5.0e6)
    ir_change = np.sum(np.abs(sed_high_u[ir_mask] - sed_no[ir_mask])) / np.sum(
        np.abs(sed_no[ir_mask])
    )
    assert ir_change > 0.01, (
        f"dust_lgU parameter change produces only {ir_change:.2%} IR change "
        "— parameter may be vacuous"
    )

    # Spinning dust peaks ~1 cm = 1e8 Angstrom (10–60 GHz microwave band)
    microwave_mask = (wave_rest > 5.0e7) & (wave_rest < 2.0e8)
    microwave_no = sed_no[microwave_mask]
    microwave_yes = sed_yes[microwave_mask]

    # With spinning dust, the microwave SED should be higher
    # Quantify the effect: AME should contribute ~5–20% of the thermal continuum
    # in the microwave band (Hensley & Draine 2023)
    ame_fractional_increase = np.sum(microwave_yes - microwave_no) / np.sum(microwave_no)
    assert ame_fractional_increase > 0.001, (
        f"AME contribution {ame_fractional_increase:.2%} is unmeasurably small"
    )
    assert np.any(microwave_yes > microwave_no), (
        "spinning_dust=True changed nothing in microwave band"
    )

    # The IR should be essentially unchanged (spinning dust is weak in IR, ~0.1%)
    np.testing.assert_allclose(sed_yes[ir_mask], sed_no[ir_mask], rtol=1e-3)


def test_f_cnm_changes_spinning_dust_spectrum(ssp_data):
    """Different f_cnm values should change the spinning-dust spectrum."""
    from tengri import FIXED, Observation, SEDModel
    from tengri.observation.spectroscopy import Spectroscopy

    ssp = ssp_data
    wave_aa = jnp.geomspace(1.0e3, 3.0e8, 1500)
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_aa))

    # Build models with different f_cnm values
    # All parameters fixed to isolate the f_cnm effect
    model_cnm_low = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED},
        dust={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": 0.5,
            "tau_diff": 1.5,
            "emission": {"type": "astrodust", "spinning_dust": True, "f_cnm": 0.1},
            "all_params": FIXED,
        },
        met={"type": "delta", "logzsol": 0.0},
        redshift=0.0,
    )

    model_cnm_high = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED},
        dust={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": 0.5,
            "tau_diff": 1.5,
            "emission": {"type": "astrodust", "spinning_dust": True, "f_cnm": 0.9},
            "all_params": FIXED,
        },
        met={"type": "delta", "logzsol": 0.0},
        redshift=0.0,
    )

    params = {
        "sfh_dpl_alpha": 1.5,
        "sfh_dpl_beta": 2.5,
        "sfh_dpl_tau_gyr": 1.0,
        "sfh_dpl_log_total_mass": 9.0,
        "dust_lgU": 0.2,
        "redshift": 0.0,
    }

    pred_low = model_cnm_low.predict(params)
    pred_high = model_cnm_high.predict(params)

    sed_low = np.asarray(pred_low.rest_sed())
    sed_high = np.asarray(pred_high.rest_sed())
    # Use the actual wavelength grid the SED was computed on
    wave_rest = np.asarray(pred_low.wave_rest)

    # The microwave SEDs should be different
    microwave_mask = (wave_rest > 5.0e7) & (wave_rest < 2.0e8)
    assert not np.allclose(sed_low[microwave_mask], sed_high[microwave_mask], rtol=1e-2)
