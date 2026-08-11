# SPDX-License-Identifier: BSD-3-Clause
"""#1671/#1688: the LUT's SNR-amplified bias becomes operational, not advisory.

``WavePrecomp`` / ``SpectrumPrecomp`` carry a small forward bias (measured
0.13-0.26 % on photometry, #1671) that enters the posterior gradient
multiplied by SNR — 5 % wrong at SNR 30, ~50 % at SNR 300 — and moves the
mode. The forward error is constant in SNR, so no forward-model check can
see it, and #1641 made the LUT the resolved default for every fit. #1688
measured the spectroscopy sibling: a ~1-sigma posterior shift on a 50-pixel
5 %-noise fixture.

The fix here is #1671's action 2: at the point where a fit resolves the LUT
and has the data in hand, ONE exact-vs-LUT forward call prices the bias on
this very model, and ``max(bias x SNR)`` above threshold warns with the
number and the remedy. These tests pin the helper's contract and that every
LUT-resolving surface calls it.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from tengri.config.exceptions import PrecompBiasWarning
from tengri.inference.fitter import (
    _central_params,
    _lut_forward_bias,
    _warn_if_lut_bias_amplified,
)

pytestmark = pytest.mark.regression_bug


class _StubSpec:
    free_params = ()
    stochastic = False

    def get_distribution(self, name):  # pragma: no cover - no free params
        raise KeyError(name)


class _StubModel:
    """A model whose forward is a fixed vector — bias is set by construction."""

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
    """(exact, lut) stub pair with a known relative forward bias."""
    flux = np.array([1.0, 2.0, 3.0, 4.0])
    return _StubModel(flux), _StubModel(flux * (1.0 + rel_bias))


def test_high_snr_amplification_warns_with_the_numbers():
    """bias 0.2 % x SNR 100 = 20 % estimated gradient error — must warn.

    The warning must carry the estimate and cite #1671: an advisory that
    does not say how wrong or why is the invisible state this fix removes.
    """
    exact, lut = _pair(2e-3)
    data = exact._flux
    noise = np.abs(data) / 100.0  # SNR 100 in every band

    with pytest.warns(PrecompBiasWarning) as rec:
        _warn_if_lut_bias_amplified(exact, lut, data, noise, "photometry", surface="Fitter")
    msg = str(rec[0].message)
    assert "#1671" in msg, "the warning must cite the measurement it operationalizes"
    assert "20" in msg, "the warning must state the estimated gradient error"
    assert "approx=None" in msg, "the warning must name the remedy"


def test_low_snr_stays_silent():
    """bias 0.2 % x SNR 10 = 2 % — below threshold, no warning.

    The LUT is the deliberate default (#1641); pricing it must not nag every
    ordinary fit, only the ones where the amplified bias is material.
    """
    exact, lut = _pair(2e-3)
    data = exact._flux
    noise = np.abs(data) / 10.0

    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecompBiasWarning)
        _warn_if_lut_bias_amplified(exact, lut, data, noise, "photometry", surface="Fitter")


def test_the_spectroscopy_channel_is_covered():
    """#1688 is the spectroscopy sibling — a photometry-only guard misses it.

    A guard is only as wide as its census: the fix for the amplification
    must treat both channels, or the exact failure #1688 measured (a
    ~1-sigma posterior shift under SpectrumPrecomp) stays invisible.
    """
    exact, lut = _pair(2e-3)
    data = exact._flux
    noise = np.abs(data) / 100.0

    with pytest.warns(PrecompBiasWarning):
        _warn_if_lut_bias_amplified(exact, lut, data, noise, "spectroscopy", surface="Fitter")


def test_the_probe_never_breaks_a_fit():
    """The advisory is worth nothing if it can take the fit down with it.

    A model whose forward raises at the central parameters (missing data
    files, partial stubs, exotic components) must degrade to no-warning —
    the fit itself proceeds exactly as before the advisory existed.
    """

    class _Broken(_StubModel):
        def predict_photometry(self, params):
            raise RuntimeError("no forward at central params")

    exact = _Broken(np.ones(4))
    _, lut = _pair(2e-3)
    _warn_if_lut_bias_amplified(
        exact, lut, np.ones(4), np.ones(4) / 100.0, "photometry", surface="Fitter"
    )  # must not raise


def test_bias_is_computed_once_and_cached_on_the_lut_model():
    """One exact forward per (model, LUT) pair — not one per galaxy.

    A catalog fit constructs a Fitter per galaxy against the same resolved
    clone; the bias is a property of the model pair, so the probe must cache
    on the clone or a 1000-galaxy catalog pays 1000 exact compiles.
    """
    exact, lut = _pair(2e-3)
    _lut_forward_bias(exact, lut, "photometry")
    first = (exact.n_calls, lut.n_calls)
    _lut_forward_bias(exact, lut, "photometry")
    assert (exact.n_calls, lut.n_calls) == first, "second call must hit the cache"


def test_central_params_are_the_declared_prior_medians():
    """The probe point is u=0 through each prior's own pushforward.

    ``unstandardize(0)`` is the declared prior's median for every
    distribution class — the same single source of truth the hierarchical
    seam uses (#1651), so the probe point is defined without inventing a
    second defaults mechanism.
    """
    from tengri.parameters.priors import Uniform

    class _Spec:
        free_params = ("a",)
        stochastic = False

        def get_distribution(self, name):
            return {"a": Uniform(0.0, 10.0)}[name]

    params = _central_params(_Spec())
    assert np.isclose(float(params["a"]), 5.0), "Uniform(0,10) median is 5"


def test_every_lut_resolving_surface_calls_the_warning():
    """The census: single, catalog, and population fits all price the LUT.

    #1656's lesson — a policy applied at one resolver and not its mirror is
    how the channel matrix grows unwritten cells. Source-level pin: each
    surface where a LUT resolution meets fit data must call the shared
    warning helper.
    """
    from tengri.inference._hierarchical_flat import build_flat_problem
    from tengri.inference.catalog_fitter import _CatalogFitterOriginal
    from tengri.inference.fitter import Fitter

    # The check runs at fit time, not construction: merely building a fitter
    # must stay cheap (the probe costs one exact forward), so the hook lives
    # in run() — priced once per instance, only when a fit actually executes.
    assert "_warn_if_lut_bias_amplified(" in inspect.getsource(Fitter.run)
    assert "_warn_if_lut_bias_amplified(" in inspect.getsource(_CatalogFitterOriginal.run)
    # PopulationFitter defers LUT resolution to fit time, so its call site is
    # the flat problem builder — where the resolved model meets the data.
    assert "_warn_if_lut_bias_amplified(" in inspect.getsource(build_flat_problem)
