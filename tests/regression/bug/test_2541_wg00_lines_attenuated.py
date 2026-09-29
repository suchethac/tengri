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

The suite tests that WG00 line attenuation is now active on every surface by
validating that predict_line_fluxes and other line predictions respond to changes
in dust optical depth under WG00.
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


@pytest.fixture(scope="module")
def model_wg00(ssp_bare, observation):
    """WG00 dust model with Cue nebular backend."""
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
        dust_attenuation={
            "type": "wg00",
            "dust_tau_v": Uniform(0.1, 3.0),
        },
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.05),
    )


def _sample_params(model, key=None, tau_v_override=None):
    """Sample parameters from model spec."""
    if key is None:
        key = jax.random.PRNGKey(0)
    params = dict(model.spec.sample(key))

    # Override dust tau_v if provided
    if tau_v_override is not None and "dust_tau_v" in params:
        params["dust_tau_v"] = np.asarray(tau_v_override)

    return params


def _line_fluxes_ratio(model, params, wave_num, wave_den, redden=True):
    """Compute ratio of line fluxes for two wavelengths."""
    fluxes = np.asarray(
        model.predict_line_fluxes(params, target_wavelengths=[wave_num, wave_den], redden=redden)
    )
    return float(fluxes[0] / fluxes[1])


class TestWG00LineAttenuation:
    """Core tests for WG00 line attenuation (#2541)."""

    def test_wg00_predict_line_fluxes_responds_to_tau_v(self, model_wg00):
        """predict_line_fluxes must change with tau_v under WG00.

        Before the fix, WG00 lines passed through unattenuated while the continuum
        was attenuated, so the line fluxes would not respond to dust changes. Now
        they should show clear dependence on tau_v.
        """
        p_lo = _sample_params(model_wg00, tau_v_override=0.1)
        p_hi = _sample_params(model_wg00, tau_v_override=2.5)

        flux_lo = _line_fluxes_ratio(model_wg00, p_lo, HALPHA_AA, HBETA_AA, redden=True)
        flux_hi = _line_fluxes_ratio(model_wg00, p_hi, HALPHA_AA, HBETA_AA, redden=True)

        # Balmer decrement (Hα/Hβ) should increase with dust (more reddening)
        assert flux_hi > flux_lo, (
            f"WG00 predict_line_fluxes Balmer decrement should increase with tau_v: "
            f"{flux_lo} vs {flux_hi}"
        )

    def test_wg00_balmer_decrement_property_responds_to_tau_v(self, model_wg00):
        """The .lines.balmer_decrement property must change with tau_v under WG00.

        This tests the interactive Prediction API, which must also see attenuated lines.
        """
        p_lo = _sample_params(model_wg00, tau_v_override=0.1)
        p_hi = _sample_params(model_wg00, tau_v_override=2.5)

        pred_lo = model_wg00.predict(p_lo)
        pred_hi = model_wg00.predict(p_hi)

        decrement_lo = float(np.asarray(pred_lo.lines.balmer_decrement))
        decrement_hi = float(np.asarray(pred_hi.lines.balmer_decrement))

        # Balmer decrement should increase with dust
        assert decrement_hi > decrement_lo, (
            f"WG00 .lines.balmer_decrement property should increase with tau_v: "
            f"{decrement_lo} vs {decrement_hi}"
        )

    def test_wg00_intrinsic_lines_unchanged(self, model_wg00):
        """INTRINSIC (redden=False) line fluxes must be independent of dust.

        This verifies that the attenuation is applied, not the intrinsic lines changed.
        """
        p_lo = _sample_params(model_wg00, tau_v_override=0.1)
        p_hi = _sample_params(model_wg00, tau_v_override=2.5)

        flux_lo = _line_fluxes_ratio(model_wg00, p_lo, HALPHA_AA, HBETA_AA, redden=False)
        flux_hi = _line_fluxes_ratio(model_wg00, p_hi, HALPHA_AA, HBETA_AA, redden=False)

        # Intrinsic (unattenuated) should not change with dust
        assert flux_lo == pytest.approx(flux_hi, rel=1e-9), (
            f"WG00 intrinsic (redden=False) lines should be dust-independent: "
            f"{flux_lo} vs {flux_hi}"
        )

    def test_wg00_lines_decrease_with_increasing_tau_v(self, model_wg00):
        """Absolute line fluxes must decrease with increasing dust optical depth.

        More dust should produce fainter lines (more attenuation).
        """
        p_lo = _sample_params(model_wg00, tau_v_override=0.5)
        p_hi = _sample_params(model_wg00, tau_v_override=2.5)

        # Use predict to get absolute luminosities
        lum_lo = float(np.asarray(model_wg00.predict(p_lo).lines.halpha))
        lum_hi = float(np.asarray(model_wg00.predict(p_hi).lines.halpha))

        # Absolute luminosities should decrease (more dust = more attenuation)
        assert lum_hi < lum_lo, (
            f"WG00 line luminosity should decrease with tau_v due to attenuation: "
            f"{lum_lo} -> {lum_hi}"
        )
