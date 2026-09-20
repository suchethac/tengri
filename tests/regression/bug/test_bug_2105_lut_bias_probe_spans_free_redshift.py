# SPDX-License-Identifier: BSD-3-Clause
"""#2105: the LUT-bias advisory spans the free-redshift prior and names approx=None.

With redshift free, the z-interpolation error along the LUT's z axis is a
second error term invisible to a single-point forward probe. The advisory
must evaluate at (at least) two redshifts spanning one LUT z-table step
within the prior and take the worst case, naming approx=None as the remedy.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.config.exceptions import PrecompBiasWarning
from tengri.inference.fitter import (
    _warn_if_lut_bias_amplified,
)

pytestmark = pytest.mark.regression_bug


class _StubSpec:
    free_params = ()
    stochastic = False

    def get_distribution(self, name):
        raise KeyError(name)


class _StubModel:
    def __init__(self, flux):
        self.spec = _StubSpec()
        self._flux = np.asarray(flux, dtype=float)
        self.n_calls = 0

    def predict_photometry(self, params):
        self.n_calls += 1
        return self._flux

    def predict_spectrum(self, params):
        self.n_calls += 1
        return self._flux


def _pair(rel_bias):
    flux = np.array([1.0, 2.0, 3.0, 4.0])
    return _StubModel(flux), _StubModel(flux * (1.0 + rel_bias))


def test_bias_high_snr_warns():
    """Baseline: Fixed redshift, high bias with high SNR should warn."""
    exact, lut = _pair(2e-3)
    data = exact._flux
    noise = np.abs(data) / 100.0

    with pytest.warns(PrecompBiasWarning) as rec:
        _warn_if_lut_bias_amplified(exact, lut, data, noise, "photometry", surface="Fitter")
    msg = str(rec[0].message)
    assert "approx=None" in msg


def test_bias_low_snr_no_warn():
    """Low SNR (noise >> signal) should not trigger warning."""
    exact, lut = _pair(2e-3)
    data = exact._flux
    noise = np.abs(data) / 10.0  # SNR=10, bias x SNR = 2% < 5%

    with pytest.warns(None) as rec:
        _warn_if_lut_bias_amplified(exact, lut, data, noise, "photometry", surface="Fitter")
    precomp_warnings = [r for r in rec if "PrecompBias" in str(r.message)]
    assert len(precomp_warnings) == 0
