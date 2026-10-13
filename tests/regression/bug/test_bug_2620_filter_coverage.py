# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #2620: a band beyond the model's wavelength axis is refused.

``photometry._filter_integral_union`` zero-fills the model SED outside its
observed-frame axis, but divides by the filter's weight over its whole support.
A band partly or fully beyond the axis therefore returns its covered fraction,
or zero, with no warning. The fix is a coverage check in the fitters: a band
whose uncovered weight exceeds ``FILTER_COVERAGE_TOLERANCE`` is refused unless
the data mark it absent. ``SEDModel`` only warns, because a model on its own
is not wrong, only unable to say anything about that band.

The fixture is the narrow optical SSP (3000-10000 A rest). Three bands at
z = 0.1 (observed axis 3300-11000 A): a fully covered band, a band straddling
the red edge, and a band wholly beyond it.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pytest

from tengri import DEFAULT, Fitter, Fixed, ForwardModel, Observation, SEDModel, Uniform
from tengri.observation.photometry_config import FilterCurve, Photometry

pytestmark = [pytest.mark.regression_bug]

Z = 0.1
_COVERED = FilterCurve(
    wave=np.linspace(4000.0, 5000.0, 60), trans=np.ones(60) * 0.5, name="covered"
)
_STRADDLE = FilterCurve(
    wave=np.linspace(9000.0, 13000.0, 60), trans=np.ones(60) * 0.5, name="straddle"
)
_BEYOND = FilterCurve(
    wave=np.linspace(20000.0, 30000.0, 60), trans=np.ones(60) * 0.5, name="beyond"
)
_BANDS = (_COVERED, _STRADDLE, _BEYOND)


def _model(ssp, filters):
    obs = Observation(photometry=Photometry(filters=tuple(filters)))
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
        },
        neb={"type": "none"},
        redshift=Fixed(Z),
    )
    return model, obs


def _fitter(model, obs, presence=None):
    flux = np.ones(len(obs.photometry.filters))
    noise = 0.1 * flux
    fwd = ForwardModel.build(sed=model, observation=obs)
    with contextlib.redirect_stdout(io.StringIO()):
        return Fitter(fwd, flux, noise, presence=presence)


class TestFilterCoverageFraction:
    def test_covered_straddling_and_beyond_bands(self):
        from tengri.observation.photometry import filter_coverage_fraction

        waves = [b.wave for b in _BANDS]
        trans = [b.trans for b in _BANDS]
        frac = filter_coverage_fraction(waves, trans, np.geomspace(3300.0, 11000.0, 4000))
        assert frac[0] == pytest.approx(0.0, abs=1e-9)
        assert 0.2 < frac[1] < 0.8
        assert frac[2] == pytest.approx(1.0, abs=1e-9)


class TestBuildWarnsOnUncoveredBand:
    def test_uncovered_band_warns_at_build(self, synthetic_ssp):
        from tengri.config.exceptions import FilterCoverageWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _model(synthetic_ssp, [_COVERED, _BEYOND])
        messages = [
            str(w.message) for w in caught if issubclass(w.category, FilterCoverageWarning)
        ]
        assert messages, "an out-of-axis band must warn at SEDModel.build"
        assert "beyond" in messages[0]

    def test_fully_covered_bands_are_silent(self, synthetic_ssp):
        from tengri.config.exceptions import FilterCoverageWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _model(synthetic_ssp, [_COVERED])
        assert not [w for w in caught if issubclass(w.category, FilterCoverageWarning)]


class TestFitterRefusesUncoveredBand:
    def test_unmasked_out_of_axis_band_is_refused(self, synthetic_ssp):
        from tengri.config.exceptions import ConfigError

        model, obs = _model(synthetic_ssp, _BANDS)
        with pytest.raises(ConfigError) as exc:
            _fitter(model, obs)
        message = str(exc.value)
        assert "beyond" in message
        assert "fraction" in message
        assert "z=" in message or "z =" in message or "redshift" in message

    def test_masked_out_of_axis_bands_are_accepted(self, synthetic_ssp):
        model, obs = _model(synthetic_ssp, _BANDS)
        fitter = _fitter(model, obs, presence=np.array([1.0, 0.0, 0.0]))
        assert fitter is not None

    def test_partly_masked_out_of_axis_band_is_still_refused(self, synthetic_ssp):
        from tengri.config.exceptions import ConfigError

        model, obs = _model(synthetic_ssp, _BANDS)
        with pytest.raises(ConfigError) as exc:
            _fitter(model, obs, presence=np.array([1.0, 1.0, 0.0]))
        assert "straddle" in str(exc.value)

    def test_fully_covered_band_is_untouched(self, synthetic_ssp):
        model, obs = _model(synthetic_ssp, [_COVERED])
        assert _fitter(model, obs) is not None


class TestFilterCoverageConstants:
    def test_tolerance_is_one_part_in_a_thousand(self):
        from tengri.observation.photometry import FILTER_COVERAGE_TOLERANCE

        assert FILTER_COVERAGE_TOLERANCE == 1e-3
