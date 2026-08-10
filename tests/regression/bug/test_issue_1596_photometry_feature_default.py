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

    _fits_line_fluxes = staticmethod(Fitter._fits_line_fluxes)
    _fits_lines = Fitter._fits_lines
    # Reads observation.line_ratios / .spectral_indices, which the stub model's
    # observation does not define — so the LUT is offered here, as before. The
    # real exclusion it implements is covered on real Observations in
    # test_feature_precomp_channel_predicate.py.
    _needs_full_forward_state = staticmethod(Fitter._needs_full_forward_state)
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
