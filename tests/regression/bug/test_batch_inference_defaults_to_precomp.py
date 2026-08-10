# SPDX-License-Identifier: BSD-3-Clause
"""Population inference must default to the precompute LUT, like ``Fitter``.

Single-galaxy fits have routed through the LUT under the default
``approx="auto"`` policy since ``Fitter._resolve_fit_approx`` landed — but
every hierarchical runner consumed ``model_factory`` output raw, so population
fits silently ran the exact wave-grid path at a measured ~2-6x per-evaluation
premium, on the surface whose samplers evaluate the model the most. A
hierarchical fit that takes minutes instead of well under one is this bug's
symptom.

Stub models keep this in the fast tier: the policy under test is "which model
does the fit-time factory hand the runners", which needs no SSP grid. The
runtime effect is verified by the executed seam probes (see the PR).
"""

from __future__ import annotations

import warnings

import pytest

from tengri.forward.sed_model import SpectrumPrecomp, WavePrecomp
from tengri.inference.hierarchical import PopulationFitter

pytestmark = pytest.mark.regression_bug


class _StubSpec:
    free_params = ("log_total_mass", "sfh_field_psd_sigma", "sfh_field_psd_tau_myr")


class _StubApprox:
    def __init__(self, wave_precomp=False, spectrum_precomp=False):
        self.wave_precomp = wave_precomp
        self.spectrum_precomp = spectrum_precomp


class _StubModel:
    """Just enough surface for the ctor's spec read and the approx policy."""

    def __init__(
        self,
        *,
        wave_precomp=False,
        spectrum_precomp=False,
        via_with_approx=False,
        features_tabulate=False,
    ):
        self.spec = _StubSpec()
        self.approx = _StubApprox(wave_precomp, spectrum_precomp)
        self.approx_configs = ()
        self.via_with_approx = via_with_approx
        self.features_tabulate = features_tabulate
        self.received_cfg = None

    def with_approx(self, cfg):
        from tengri.forward.sed_model import FeaturePrecomp

        cfgs = cfg if isinstance(cfg, tuple) else (cfg,)
        if any(isinstance(c, FeaturePrecomp) for c in cfgs) and not self.features_tabulate:
            # Mirrors the real contract: a backend with nothing to tabulate
            # raises (measured: neb='none' -> ValueError "no emission lines
            # to tabulate"), and that raise IS the detection mechanism.
            raise ValueError("no emission lines to tabulate")
        new = _StubModel(wave_precomp=True, via_with_approx=True)
        new.received_cfg = cfg
        return new


def _fitter(factory, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return PopulationFitter(factory, [{"flux_obs": [1.0], "noise": [0.1]}], **kwargs)


def test_the_fit_time_factory_routes_through_the_lut_by_default():
    """No ``approx`` argument -> factory-built models carry WavePrecomp."""
    fitter = _fitter(lambda **kw: _StubModel())
    model = fitter.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)
    assert model.via_with_approx, "the fit-time factory handed the runners the raw model"
    cfgs = model.received_cfg if isinstance(model.received_cfg, tuple) else (model.received_cfg,)
    assert any(isinstance(c, WavePrecomp) for c in cfgs)


def test_approx_none_keeps_the_exact_path():
    """``approx=None`` is the documented opt-out and must stay exact."""
    fitter = _fitter(lambda **kw: _StubModel(), approx=None)
    model = fitter.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)
    assert not model.via_with_approx


def test_an_explicit_config_is_used_verbatim():
    """An explicit config means what it says, exactly as on ``Fitter``."""
    cfg = WavePrecomp(n_z=17)
    fitter = _fitter(lambda **kw: _StubModel(), approx=cfg)
    model = fitter.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)
    assert model.via_with_approx
    assert model.received_cfg is cfg


def test_a_model_already_carrying_the_lut_is_not_rewrapped():
    """Build-time ``approx=WavePrecomp()`` must not be cloned again."""
    fitter = _fitter(lambda **kw: _StubModel(wave_precomp=True))
    model = fitter.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)
    assert not model.via_with_approx


def test_spectroscopy_resolves_to_the_spectrum_lut():
    """``data_type="spectroscopy"`` -> SpectrumPrecomp, mirroring Fitter's auto map."""
    fitter = _fitter(lambda **kw: _StubModel(), data_type="spectroscopy")
    model = fitter.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)
    assert model.via_with_approx
    cfgs = model.received_cfg if isinstance(model.received_cfg, tuple) else (model.received_cfg,)
    assert any(isinstance(c, SpectrumPrecomp) for c in cfgs)


def test_the_spec_template_stays_raw():
    """The ctor's template exists to read the spec; it must not pay the LUT build."""
    fitter = _fitter(lambda **kw: _StubModel())
    assert not fitter._template.via_with_approx


def _catalog_fitter(model, **kwargs):
    from tengri.inference.catalog_fitter import _CatalogFitterOriginal

    return _CatalogFitterOriginal(model, [{"flux_obs": [1.0], "noise": [0.1]}], **kwargs)


def test_catalog_fits_route_through_the_lut_by_default():
    """CatalogFitter had the same gap: precomp existed only as advice text."""
    fitter = _catalog_fitter(_StubModel())
    assert fitter.model.via_with_approx, "the catalog fit holds the raw exact-path model"
    cfgs = (
        fitter.model.received_cfg
        if isinstance(fitter.model.received_cfg, tuple)
        else (fitter.model.received_cfg,)
    )
    assert any(isinstance(c, WavePrecomp) for c in cfgs)


def test_catalog_approx_none_keeps_the_exact_path():
    fitter = _catalog_fitter(_StubModel(), approx=None)
    assert not fitter.model.via_with_approx


def test_catalog_model_already_carrying_the_lut_is_not_rewrapped():
    fitter = _catalog_fitter(_StubModel(wave_precomp=True))
    assert not fitter.model.via_with_approx


def test_auto_tops_up_featureprecomp_when_the_backend_can_tabulate():
    """A feature-tabulating backend (Cue) gets BOTH LUTs under the default.

    Measured on the 2-galaxy Cue population MAP fit: +FeaturePrecomp is
    1.45x warm / 1.68x cold over WavePrecomp alone (A/A floor 1.17x), and
    ~7x per-gradient on the single-galaxy #1596 model — WavePrecomp alone
    does not clear the noise floor on a Cue model at all.
    """
    from tengri.forward.sed_model import FeaturePrecomp

    fitter = _fitter(lambda **kw: _StubModel(features_tabulate=True))
    model = fitter.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)
    assert model.via_with_approx
    cfgs = model.received_cfg if isinstance(model.received_cfg, tuple) else (model.received_cfg,)
    assert any(isinstance(c, WavePrecomp) for c in cfgs)
    assert any(isinstance(c, FeaturePrecomp) for c in cfgs)


def test_auto_falls_back_to_waveprecomp_when_features_cannot_tabulate():
    """A line-less backend must keep WavePrecomp — never fall back to raw.

    The naive spelling tries (WavePrecomp, FeaturePrecomp) atomically and
    loses BOTH when the feature half raises, silently regressing the #1641
    default for every stellar-only model.
    """
    from tengri.forward.sed_model import FeaturePrecomp

    fitter = _fitter(lambda **kw: _StubModel(features_tabulate=False))
    model = fitter.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)
    assert model.via_with_approx, "the feature raise must not cost the wave LUT"
    cfgs = model.received_cfg if isinstance(model.received_cfg, tuple) else (model.received_cfg,)
    assert any(isinstance(c, WavePrecomp) for c in cfgs)
    assert not any(isinstance(c, FeaturePrecomp) for c in cfgs)


def test_catalog_auto_also_tops_up_featureprecomp():
    from tengri.forward.sed_model import FeaturePrecomp

    fitter = _catalog_fitter(_StubModel(features_tabulate=True))
    cfgs = (
        fitter.model.received_cfg
        if isinstance(fitter.model.received_cfg, tuple)
        else (fitter.model.received_cfg,)
    )
    assert any(isinstance(c, FeaturePrecomp) for c in cfgs)
