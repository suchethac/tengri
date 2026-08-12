# SPDX-License-Identifier: BSD-3-Clause
"""Photometry-only fits must attempt the feature LUT by default (#1596).

The controlled #1596 measurement (A/A floor 1.23x, nb10's 10-parameter Cue
model): removing the line channel made the same fit ~4x SLOWER on the default
path, because ``FeaturePrecomp`` — despite the name, the *nebular* precompute;
for Cue the per-Q_H grid replaces the emulator call itself — was only added
when a line channel was fit. ``WavePrecomp`` alone does not clear the noise
floor on a Cue model (1.07x). The batch surfaces gained the
attempt-and-fallback top-up in #1641; this pins the single-galaxy mirror.

Stubs keep it in the fast tier: the question is "which config does the auto
policy resolve", which needs no SSP grid. The real-model resolution is
exercised by the executed check in the PR.
"""

from __future__ import annotations

import pytest

from tengri.forward.sed_model import FeaturePrecomp, WavePrecomp
from tengri.inference.fitter import Fitter

pytestmark = pytest.mark.regression_bug


class _StubApprox:
    wave_precomp = False
    spectrum_precomp = False
    feature_precomp = False


class _StubModel:
    def __init__(self, *, features_tabulate):
        self.approx = _StubApprox()
        self.approx_configs = ()
        self.features_tabulate = features_tabulate
        self.received_cfg = None

    def _has_modern_approx(self):
        return False

    def with_approx(self, cfg):
        cfgs = cfg if isinstance(cfg, tuple) else (cfg,)
        if any(isinstance(c, FeaturePrecomp) for c in cfgs) and not self.features_tabulate:
            # The real contract: a backend with nothing to tabulate raises
            # (measured: neb='none' -> ValueError); the raise IS the detection.
            raise ValueError("no emission lines to tabulate")
        new = _StubModel(features_tabulate=self.features_tabulate)
        new.received_cfg = cfg
        return new


class _StubFitterSelf:
    """The minimum of ``Fitter`` that ``_resolve_fit_approx`` reads."""

    data_type = "photometry"
    _line_flux_override = None
    _eline_marginalize = False
    _eline_fitted = False

    _line_fluxes_for = Fitter._line_fluxes_for
    _fits_line_fluxes = Fitter._fits_line_fluxes
    _fits_lines = Fitter._fits_lines
    _auto_approx_config = Fitter._auto_approx_config
    _add_feature_precomp = Fitter._add_feature_precomp
    _warn_lines_without_lut = Fitter._warn_lines_without_lut
    _resolve_fit_approx = Fitter._resolve_fit_approx

    def __init__(self, model):
        self.model = model


def _resolved_cfgs(model):
    fitter = _StubFitterSelf(model)
    resolved = fitter._resolve_fit_approx(model, "auto")
    assert resolved is not model, "auto must clone onto the LUT path"
    cfg = resolved.received_cfg
    return cfg if isinstance(cfg, tuple) else (cfg,)


def test_photometry_auto_attempts_the_feature_lut():
    """A feature-tabulating backend (Cue) gets BOTH LUTs with no line channel."""
    cfgs = _resolved_cfgs(_StubModel(features_tabulate=True))
    assert any(isinstance(c, WavePrecomp) for c in cfgs)
    assert any(isinstance(c, FeaturePrecomp) for c in cfgs), (
        "photometry-only auto stayed WavePrecomp-only — the #1596 defect: "
        "~4x slower than the same fit WITH a line channel"
    )


def test_a_backend_that_cannot_tabulate_keeps_the_wave_lut():
    """The feature raise must cost nothing: fall back to WavePrecomp, not raw."""
    cfgs = _resolved_cfgs(_StubModel(features_tabulate=False))
    assert any(isinstance(c, WavePrecomp) for c in cfgs)
    assert not any(isinstance(c, FeaturePrecomp) for c in cfgs)


class _StubObs:
    photometry = object()
    line_fluxes = None
    line_ratios = None
    spectral_indices = None


def test_a_line_ratio_channel_disables_the_feature_topup():
    """The ratio term reads the backend's DISCRETE line catalog — which the
    feature-LUT path does not publish.

    Measured on main at the #1656 merge (594a60552):
    ``test_ratio_term_constrains_fit`` went red with "Configured nebular
    backend did not publish a discrete line catalog" because the top-up fired
    for an Observation carrying ``line_ratios`` — a channel ``_fits_lines``
    does not see. The channel matrix's unwritten cells strike again
    (#1460/#1480/#1599 lineage): the top-up may fire only when NO
    line-adjacent channel exists.
    """
    model = _StubModel(features_tabulate=True)
    obs = _StubObs()
    obs.line_ratios = object()
    model.observation = obs
    cfgs = _resolved_cfgs(model)
    assert any(isinstance(c, WavePrecomp) for c in cfgs)
    assert not any(isinstance(c, FeaturePrecomp) for c in cfgs), (
        "the feature top-up fired despite a line-ratio channel that needs "
        "the discrete catalog the LUT path does not publish"
    )


def test_spectral_indices_also_disable_the_feature_topup():
    """Unverified interaction stays OFF: indices are a channel the top-up has
    never been executed against, and the ratio channel proved 'plausibly
    orthogonal' is not evidence."""
    model = _StubModel(features_tabulate=True)
    obs = _StubObs()
    obs.spectral_indices = object()
    model.observation = obs
    cfgs = _resolved_cfgs(model)
    assert not any(isinstance(c, FeaturePrecomp) for c in cfgs)


class _StubModernModel(_StubModel):
    """A model BUILT with ``approx=WavePrecomp()``."""

    def _has_modern_approx(self):
        return True


def test_the_line_flux_override_counts_as_a_line_channel():
    """``line_flux_data=`` is a line-flux channel, so the predicate must see it.

    ``_build_data_args`` resolves the channel through ``_resolved_line_fluxes``
    -- the ``line_flux_data=`` override (#1599 per-galaxy values) *or* the
    Observation's own ``line_fluxes`` -- while the predicate read only the
    latter. So a fit supplying its line fluxes per galaxy publishes
    ``line_flux_waves`` and sets ``has_line_fluxes`` in the loss, yet classifies
    as photometry-only. The two must read the same sources or they disagree
    about whether the fit needs the line LUT.
    """
    model = _StubModel(features_tabulate=True)
    model.observation = _StubObs()  # line_fluxes stays None
    fitter = _StubFitterSelf(model)
    fitter._line_flux_override = object()  # LineFluxData stand-in

    assert fitter._fits_lines(model), (
        "a line-flux channel supplied via line_flux_data= was not counted as a "
        "line channel; the predicate censuses observation.line_fluxes only, "
        "while _build_data_args resolves both sources"
    )


def test_the_line_flux_override_gets_the_feature_topup():
    """The #1596 cliff must not reopen for the per-galaxy spelling.

    On a model built with ``approx=WavePrecomp()`` the ``"auto"`` policy tops up
    ``FeaturePrecomp`` only when ``_fits_lines`` is true. Spelled via
    ``observation.line_fluxes`` the same fit is topped up; spelled via
    ``line_flux_data=`` it was returned untouched -- no LUT and no warning,
    i.e. exactly the ~21x per-gradient cliff ``_add_feature_precomp`` exists
    to prevent, on the two public paths the docs recommend together.
    """
    model = _StubModernModel(features_tabulate=True)
    model.observation = _StubObs()  # line_fluxes stays None
    fitter = _StubFitterSelf(model)
    fitter._line_flux_override = object()

    resolved = fitter._resolve_fit_approx(model, "auto")
    assert resolved is not model, (
        "the model was returned untouched: a line-flux fit via line_flux_data= "
        "got no FeaturePrecomp top-up and no warning"
    )
    cfg = resolved.received_cfg
    cfgs = cfg if isinstance(cfg, tuple) else (cfg,)
    assert any(isinstance(c, FeaturePrecomp) for c in cfgs)
