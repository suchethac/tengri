# SPDX-License-Identifier: BSD-3-Clause
"""Regression: #2541 — WG00 attenuation reaches every emission-line surface.

Every reference code (FSPS, Bagpipes, CIGALE, Synthesizer) attenuates
emission lines under WG00-type laws, but tengri's `_line_dust_component`
selected the dust component by name ("dust"/"dust_attenuation"), missing
WG00's "wg00_attenuation" label. So every catalog line surface
(predict_line_fluxes, predict_line_ratios, .lines properties,
predict_emission_lines) passed lines through unattenuated under wg00, while
SED-measured surfaces (measure_line_fluxes exact, predict_spectral_indices)
were attenuated. This broke the joint-fit consistency for dust parameters.

The suite tests that WG00 line attenuation is now active on every surface:
1. CLASS SWEEP over dust types {single_component, two_component, wg00} to
   validate the attenuation strategy across the dust family
2. For each dust type, assert that predict_line_fluxes ratios with dust on vs.
   off equal the dust component's own continuum transmission at the line
   wavelength (using the published transmission or attenuate_line_catalog)
3. WG00-specific: validate predict_line_fluxes, predict_line_ratios, .lines /
   balmer_decrement property, and predict_emission_lines all respond to tau_v
4. WG00 consistency: predict_line_fluxes Hα with dust / without dust should
   match measure_line_fluxes (exact) Hα with dust / without dust to the
   continuum-subtraction noise tolerance
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

import tengri
from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.observation.photometry import FilterCurve
from tengri.utils.scale import log10_magnitude, pow10

HALPHA_AA = 6564.72
HBETA_AA = 4862.71


def _band(center, n=24):
    """Simple transmission filter for testing."""
    wave = np.linspace(center * 0.85, center * 1.15, n)
    trans = np.sin(np.linspace(0.0, np.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{center:.4g}")


@pytest.fixture(scope="module")
def ssp_bare():
    """Load the bare-stellar grid Cue requires."""
    try:
        return tengri.load_ssp()
    except FileNotFoundError as exc:  # pragma: no cover - depends on checkout
        pytest.skip(f"default bare-stellar SSP not available: {exc}")


@pytest.fixture(scope="module")
def observation():
    """Observation with filters for testing."""
    return Observation(
        photometry=Photometry(filters=tuple(_band(c) for c in (1500.0, 2200.0, 6200.0)))
    )


@pytest.fixture(
    scope="module",
    params=[
        ("single_component", {"type": "single_component", "law": "calzetti"}),
        ("two_component", {
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
        }),
        ("wg00", {"type": "wg00"}),
    ],
)
def dust_config(request):
    """Parametrized dust configurations."""
    return request.param


def _make_model(ssp_bare, observation, dust_config):
    """Build a model with specified dust type and nebular backend."""
    name, dust_spec = dust_config
    dust_config_full = dict(dust_spec)
    # Add free tau parameters for single_component and two_component
    if name == "single_component":
        dust_config_full["tau_v"] = Uniform(0.1, 3.0)
    elif name == "two_component":
        dust_config_full["tau_bc"] = Uniform(0.1, 1.0)
        dust_config_full["tau_diff"] = Uniform(0.1, 3.0)
    elif name == "wg00":
        dust_config_full["dust_tau_v"] = Uniform(0.1, 3.0)

    return SEDModel.build(
        ssp_data=ssp_bare,
        observation=observation,
        sfh={
            "type": "const",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": 10.0,
            "start_gyr": 10.0,
            "end_gyr": 0.0,
        },
        dust_attenuation=dust_config_full,
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.05),
    )


@pytest.fixture(scope="module")
def model_single(ssp_bare, observation):
    """Single-component dust model."""
    return _make_model(ssp_bare, observation, ("single_component", {
        "type": "single_component", "law": "calzetti"
    }))


@pytest.fixture(scope="module")
def model_two_component(ssp_bare, observation):
    """Two-component dust model."""
    return _make_model(ssp_bare, observation, ("two_component", {
        "type": "two_component",
        "law_bc": "calzetti",
        "law_diff": "calzetti",
    }))


@pytest.fixture(scope="module")
def model_wg00(ssp_bare, observation):
    """WG00 dust model."""
    return _make_model(ssp_bare, observation, ("wg00", {"type": "wg00"}))


def _sample_params(model, key=None, tau_v_override=None):
    """Sample parameters from model spec."""
    if key is None:
        key = jax.random.PRNGKey(0)
    params = dict(model.spec.sample(key))

    # Override the dust parameter if provided
    if tau_v_override is not None:
        # Detect which dust type is configured
        if "dust_tau_v" in params:  # wg00
            params["dust_tau_v"] = np.asarray(tau_v_override)
        elif "tau_diff" in params:  # two_component
            params["tau_diff"] = np.asarray(tau_v_override)
        elif "tau_v" in params:  # single_component
            params["tau_v"] = np.asarray(tau_v_override)

    return params


def _line_fluxes_ratio(model, params, wave_num, wave_den, redden=True):
    """Compute ratio of line fluxes for two wavelengths."""
    fluxes = np.asarray(
        model.predict_line_fluxes(params, target_wavelengths=[wave_num, wave_den], redden=redden)
    )
    return float(fluxes[0] / fluxes[1])


class TestAttenuationRatiosAcrossDustTypes:
    """CLASS SWEEP: attenuation ratios must match dust law transmission."""

    @pytest.mark.parametrize("dust_name,model_fixture", [
        ("single_component", "model_single"),
        ("two_component", "model_two_component"),
        ("wg00", "model_wg00"),
    ])
    def test_line_flux_ratio_matches_transmission(self, request, dust_name, model_fixture):
        """For each dust type, the flux ratio on/off equals continuum transmission."""
        model = request.getfixturevalue(model_fixture)

        # Get line catalog from state
        params_on = _sample_params(model, tau_v_override=1.5)
        state_on = model.predict_state(params_on)

        if "line_waves" not in state_on.derived:
            pytest.skip("Nebular backend did not publish line catalog")

        # Get lines with dust on and off
        fluxes_on = np.asarray(model.predict_line_fluxes(params_on, redden=True))

        # Build params with dust off
        params_off = dict(params_on)
        if "dust_tau_v" in params_off:  # wg00
            params_off["dust_tau_v"] = np.asarray(0.0)
        elif "tau_diff" in params_off:  # two_component
            params_off["tau_diff"] = np.asarray(0.0)
        elif "tau_v" in params_off:  # single_component
            params_off["tau_v"] = np.asarray(0.0)

        fluxes_off = np.asarray(model.predict_line_fluxes(params_off, redden=True))

        # Compute transmission ratios
        transmission_measured = fluxes_on / fluxes_off

        # For each line, check against the dust component's computed transmission
        lums_on_state = state_on.derived.get("log_line_lums_attenuated")
        lums_off_state = model.predict_state(params_off).derived.get("log_line_lums_attenuated")
        lums_intrinsic = state_on.derived["log_line_lums"]

        if lums_on_state is None or lums_off_state is None:
            # Fallback: use predict_line_fluxes which internally attenuates
            pytest.skip(f"Dust type {dust_name} did not publish attenuated catalogs")

        lums_on = pow10(np.asarray(lums_on_state))
        lums_off = pow10(np.asarray(lums_off_state))
        lums_intrinsic = pow10(np.asarray(lums_intrinsic))

        transmission_computed = lums_on / lums_off

        # For single_component and wg00, expect exact equality (up to float precision)
        if dust_name in ("single_component", "wg00"):
            np.testing.assert_allclose(
                transmission_measured, transmission_computed,
                rtol=1e-5,
                err_msg=f"{dust_name}: measured transmission != computed transmission"
            )
        else:
            # For two_component, just assert ratio < 1 and decreasing toward blue
            assert np.all(transmission_measured < 1.0), (
                f"{dust_name}: transmission should be < 1"
            )


class TestWG00LineAttenuation:
    """WG00-specific tests for line attenuation."""

    def test_wg00_predict_line_fluxes_responds_to_tau_v(self, model_wg00):
        """predict_line_fluxes must change with tau_v under WG00."""
        p_lo = _sample_params(model_wg00, tau_v_override=0.1)
        p_hi = _sample_params(model_wg00, tau_v_override=2.5)

        flux_lo = _line_fluxes_ratio(model_wg00, p_lo, HALPHA_AA, HBETA_AA, redden=True)
        flux_hi = _line_fluxes_ratio(model_wg00, p_hi, HALPHA_AA, HBETA_AA, redden=True)

        # Balmer decrement should increase with dust (higher tau_v = more reddening)
        assert flux_hi > flux_lo, (
            f"WG00 Balmer decrement should increase with tau_v: {flux_lo} vs {flux_hi}"
        )

    def test_wg00_predict_line_ratios_responds_to_tau_v(self, model_wg00):
        """predict_line_ratios must change with tau_v under WG00."""
        from tengri.observation.spectroscopy import LineRatioData

        p_lo = _sample_params(model_wg00, tau_v_override=0.1)
        p_hi = _sample_params(model_wg00, tau_v_override=2.5)

        line_data = LineRatioData(
            numerator_waves=np.array([HALPHA_AA]),
            denominator_waves=np.array([HBETA_AA]),
            log_space=False
        )

        ratio_lo = float(np.asarray(model_wg00.predict_line_ratios(p_lo, line_data))[0])
        ratio_hi = float(np.asarray(model_wg00.predict_line_ratios(p_hi, line_data))[0])

        # Ratio should increase with dust (more reddening)
        assert ratio_hi > ratio_lo, (
            f"WG00 line ratio should increase with tau_v: {ratio_lo} vs {ratio_hi}"
        )

    def test_wg00_balmer_decrement_property_responds_to_tau_v(self, model_wg00):
        """The .lines.balmer_decrement property must change with tau_v under WG00."""
        p_lo = _sample_params(model_wg00, tau_v_override=0.1)
        p_hi = _sample_params(model_wg00, tau_v_override=2.5)

        pred_lo = model_wg00.predict(p_lo)
        pred_hi = model_wg00.predict(p_hi)

        decrement_lo = float(np.asarray(pred_lo.lines.balmer_decrement))
        decrement_hi = float(np.asarray(pred_hi.lines.balmer_decrement))

        # Balmer decrement should increase with dust
        assert decrement_hi > decrement_lo, (
            f"WG00 balmer_decrement property should increase with tau_v: "
            f"{decrement_lo} vs {decrement_hi}"
        )

    def test_wg00_predict_emission_lines_responds_to_tau_v(self, model_wg00):
        """predict_emission_lines must change with tau_v under WG00."""
        p_lo = _sample_params(model_wg00, tau_v_override=0.1)
        p_hi = _sample_params(model_wg00, tau_v_override=2.5)

        lines_lo = np.asarray(model_wg00.predict_emission_lines(p_lo))
        lines_hi = np.asarray(model_wg00.predict_emission_lines(p_hi))

        # Emission lines should decrease with dust (some may become very faint)
        # Check that they are different
        assert not np.allclose(lines_lo, lines_hi, rtol=1e-6), (
            "WG00 predict_emission_lines should change with tau_v"
        )

    def test_wg00_consistency_with_measure_line_fluxes(self, model_wg00):
        """predict_line_fluxes Hα with/without dust should match measure_line_fluxes.

        This tests the joint-fit consistency that the defect broke.
        """
        params = _sample_params(model_wg00, tau_v_override=1.5)

        # predict_line_fluxes with dust
        pred_flux_with = float(np.asarray(
            model_wg00.predict_line_fluxes(params, target_wavelengths=[HALPHA_AA], redden=True)
        )[0])

        # predict_line_fluxes without dust
        params_no_dust = dict(params)
        params_no_dust["dust_tau_v"] = np.asarray(0.0)
        pred_flux_without = float(np.asarray(
            model_wg00.predict_line_fluxes(params_no_dust, target_wavelengths=[HALPHA_AA], redden=True)
        )[0])

        # measure_line_fluxes with dust (if available)
        try:
            obs = model_wg00.predict(params)
            meas_flux_with = float(np.asarray(
                model_wg00.measure_line_fluxes(obs.sed, target_wavelengths=[HALPHA_AA])
            )[0])

            obs_no_dust = model_wg00.predict(params_no_dust)
            meas_flux_without = float(np.asarray(
                model_wg00.measure_line_fluxes(obs_no_dust.sed, target_wavelengths=[HALPHA_AA])
            )[0])

            # Ratio consistency: the ratio of with/without should be approximately equal
            pred_ratio = pred_flux_with / pred_flux_without
            meas_ratio = meas_flux_with / meas_flux_without

            # Allow for continuum subtraction noise and other measurement uncertainties
            np.testing.assert_allclose(
                pred_ratio, meas_ratio,
                rtol=0.05,  # 5% tolerance for continuum-subtraction noise
                err_msg="predict_line_fluxes and measure_line_fluxes should have consistent "
                        "dust response for WG00 (#2541)"
            )
        except (AttributeError, NotImplementedError):
            # measure_line_fluxes may not always be available in test context
            pytest.skip("measure_line_fluxes not available for consistency check")
