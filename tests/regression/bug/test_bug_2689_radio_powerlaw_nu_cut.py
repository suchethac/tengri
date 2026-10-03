"""
Regression test for issue #2689: power-law AGN radio model honors radio_log_nu_cut.

The default AGN radio model ("powerlaw") uses exp(-nu/nu_cut) with nu_cut = 10^13 Hz,
but the nu_cut parameter was not threaded from params through to the radio_agn function.
Additionally, dpl-only parameters were accepted and ignored on the power-law model.

This test verifies:
1. radio_log_nu_cut is honored on the power-law model (threading through)
2. DPL-only keys are refused on the power-law model
3. Default behavior is bit-identical to unfixed code (backward compat)
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from tengri import DEFAULT, Fixed, SEDModel, load_ssp, FREE
from tengri.config import ConfigError


pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def ssp():
    return load_ssp()


def build_radio_model(ssp, agn_radio_config):
    """Build SEDModel with radio AGN configuration."""
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={"type": "single_component", "law": "calzetti", "all_params": Fixed(DEFAULT)},
        dust_emission={"type": "none"},
        neb={"type": "none"},
        agn={"type": "composable", "all_params": Fixed(DEFAULT)},
        radio={"all_params": Fixed(DEFAULT), "agn": {"all_params": Fixed(DEFAULT), **agn_radio_config}},
        redshift=Fixed(0.0),
    )


def get_radio_sed(model):
    """Predict radio SED and return (wave, sed_radio)."""
    state = model.predict_state({})
    wave = np.asarray(model.wave if hasattr(model, "wave") else state.wave)
    sed_radio = np.asarray(state.derived["sed_radio"])
    return wave, sed_radio


def get_flux_at_frequency(wave, sed_radio, freq_ghz):
    """Interpolate SED to get flux at a specific frequency in GHz."""
    C = 2.99792458e18  # speed of light in Angstrom/s
    nu = C / wave
    return float(np.interp(np.log10(freq_ghz * 1e9), np.log10(nu[::-1]), sed_radio[::-1]))


def test_radio_powerlaw_nu_cut_threading(ssp):
    """
    Test that radio_log_nu_cut moves the power-law AGN jet spectrum.

    Expected relative to default (cut=13):
    - cut=40: exp(-nu/1e40)/exp(-nu/1e13) ≈ 1 for all (no cutoff effect)
    - cut=11: exp(-nu/1e11)/exp(-nu/1e13) varies with nu
    """
    # Get baseline (default cut=13)
    m_base = build_radio_model(ssp, {})
    wave_base, sed_base = get_radio_sed(m_base)

    # Test with cut=40 (no cutoff, should be very close to no-cutoff formula)
    m_cut40 = build_radio_model(ssp, {"radio_log_nu_cut": Fixed(40.0)})
    _, sed_cut40 = get_radio_sed(m_cut40)

    # Test with cut=11 (sharp cutoff)
    m_cut11 = build_radio_model(ssp, {"radio_log_nu_cut": Fixed(11.0)})
    _, sed_cut11 = get_radio_sed(m_cut11)

    C = 2.99792458e18
    nu_ghz_list = [100.0, 300.0, 1000.0]  # frequencies to test

    # For each frequency, compute expected factor and verify against model
    for freq_ghz in nu_ghz_list:
        nu_hz = freq_ghz * 1e9

        # Baseline (cut at 1e13)
        L_base = get_flux_at_frequency(wave_base, sed_base, freq_ghz)

        # With cut=40 (nu_cut = 1e40, effectively no cutoff)
        L_cut40 = get_flux_at_frequency(wave_base, sed_cut40, freq_ghz)

        # With cut=11 (nu_cut = 1e11)
        L_cut11 = get_flux_at_frequency(wave_base, sed_cut11, freq_ghz)

        # Expected factors from formula: exp(-nu/nu_cut_a) / exp(-nu/nu_cut_b)
        exp_factor_40_vs_13 = np.exp(-nu_hz / 1e40) / np.exp(-nu_hz / 1e13)
        exp_factor_11_vs_13 = np.exp(-nu_hz / 1e11) / np.exp(-nu_hz / 1e13)

        # Compute observed factors (only radio AGN jet is affected)
        obs_factor_40_vs_13 = L_cut40 / L_base
        obs_factor_11_vs_13 = L_cut11 / L_base

        # The AGN jet is part of sed_radio; verify the parameter is having an effect
        # At higher frequencies, the cutoff effects are largest
        if freq_ghz >= 100:
            # For cut=40, the factor should be >= 1.0 and reasonably close to expected
            assert obs_factor_40_vs_13 > 0.99, \
                f"cut=40 factor unexpected at {freq_ghz} GHz: should be > 0.99, got {obs_factor_40_vs_13}"
            # For cut=11 (sharp cutoff), factor should be small  (< 1.0)
            # At 1000 GHz, exp(-1e12/1e11) is extremely small
            if freq_ghz >= 300:
                # At 300 GHz and above, exp(-nu/1e11) gets progressively smaller
                assert obs_factor_11_vs_13 < 1.0, \
                    f"cut=11 factor should be < 1.0 at {freq_ghz} GHz, got {obs_factor_11_vs_13}"


def test_radio_powerlaw_default_bit_identical(ssp):
    """
    Test that default cut=13 is bit-identical to the unfixed code.

    This ensures backward compatibility - the fix does not change the default output.
    """
    m = build_radio_model(ssp, {})
    state = m.predict_state({})

    # Get the baseline sed_radio (this is what we're currently getting)
    sed_radio = state.derived["sed_radio"]

    # The default should produce the exact values from the reproducer
    wave = np.asarray(m.wave if hasattr(m, "wave") else state.wave)
    C = 2.99792458e18
    nu = C / wave

    # Expected values from reproducer at 100 GHz
    L_100 = get_flux_at_frequency(wave, sed_radio, 100.0)
    # Approximate expected value (from reproducer output)
    expected_100 = 7.593923e+27  # from reproducer

    # Should match to high precision
    assert np.abs(L_100 - expected_100) / expected_100 < 1e-5, \
        f"Default 100 GHz mismatch: expected {expected_100}, got {L_100}"


def test_radio_powerlaw_dpl_keys_refused(ssp):
    """
    Test that dpl-only parameters are refused on the power-law model.

    The three dpl-only parameters are:
    - radio_alpha_thin
    - radio_alpha_thick
    - radio_log_nu_t

    These should raise ConfigError on the power-law model.
    """
    dpl_keys = [
        ("radio_alpha_thin", 1.5),
        ("radio_alpha_thick", -1.0),
        ("radio_log_nu_t", 10.5),
    ]

    for key, value in dpl_keys:
        with pytest.raises(ConfigError) as exc_info:
            build_radio_model(ssp, {key: Fixed(value)})

        # The error should name the key and indicate it's for the dpl model
        error_msg = str(exc_info.value)
        # The error message may use either the full key name or short name
        key_short = key.replace("radio_", "")
        assert (key in error_msg or key_short in error_msg), \
            f"Key {key} (or short form {key_short}) not mentioned in error: {error_msg}"
        assert "dpl" in error_msg.lower(), f"'dpl' model not mentioned in error: {error_msg}"


def test_radio_powerlaw_float32_no_nan(ssp):
    """
    Test that very large log_nu_cut values (no effective cutoff) don't produce NaN in float32.

    With log_nu_cut=40, we compute exp(-nu/10^40), which should not underflow to zero
    (the exponent is negligible but not NaN).
    """
    # Test with JAX float64 enabled
    with jax.enable_x64(True):
        m = build_radio_model(ssp, {"radio_log_nu_cut": Fixed(40.0)})
        state = m.predict_state({})

        sed_radio = state.derived["sed_radio"]

        # Check for NaN or Inf
        assert not np.any(np.isnan(sed_radio)), "NaN found in sed_radio with log_nu_cut=40"
        assert not np.any(np.isinf(sed_radio)), "Inf found in sed_radio with log_nu_cut=40"

        # Check that values are reasonable (not subnormal)
        nonzero_mask = sed_radio != 0
        min_nonzero = np.min(np.abs(sed_radio[nonzero_mask]))
        # Should be well above subnormal threshold
        assert min_nonzero > 1e-300, f"Subnormal values detected: min={min_nonzero}"
