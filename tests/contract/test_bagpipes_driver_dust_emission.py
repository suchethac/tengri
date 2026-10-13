# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the BAGPIPES driver returns attenuated starlight without its implicit DL07 emission.

``model_galaxy`` adds ``dust_flux * dust_emission.spectrum(qpah, umin, gamma)`` to every
model that carries a ``dust`` block. The driver removes the term by substituting a
zero-shaped emission model while a model is built. These tests pin that on a stand-in
``model_galaxy``: the emission-free spectrum equals the attenuated spectrum alone, the
default path keeps the emission, and the original class is restored.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import types

import numpy as np
import pytest

pytestmark = pytest.mark.contract

_WAVE = np.linspace(1000.0, 30000.0, 64)
_ATTENUATED = np.linspace(1.0, 2.0, _WAVE.size)
_DUST_FLUX = 3.0


class _FakeDustEmission:
    def __init__(self, wavelengths):
        self.n = len(wavelengths)

    def spectrum(self, qpah, umin, gamma):
        return np.full(self.n, qpah * umin + gamma)


class _FakeModelGalaxy:
    def __init__(self, components, spec_wavs=None):
        self.model_comp = components
        self.wavelengths = _WAVE
        self.dust_emission = sys.modules["bagpipes.models.model_galaxy"].dust_emission(_WAVE)
        spectrum = _ATTENUATED.copy()
        if "dust" in components:
            spectrum += _DUST_FLUX * self.dust_emission.spectrum(2.0, 1.0, 0.01)
        self.spectrum_full = spectrum


@pytest.fixture
def driver(monkeypatch):
    fake = types.ModuleType("bagpipes.models.model_galaxy")
    fake.model_galaxy = _FakeModelGalaxy
    fake.dust_emission = _FakeDustEmission
    for name in ("bagpipes", "bagpipes.models"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "bagpipes.models.model_galaxy", fake)
    mod = importlib.import_module("reproduction.bagpipes._drivers.bagpipes_driver")
    monkeypatch.setattr(mod, "assert_sampling", lambda *a, **k: 0.0)
    monkeypatch.setattr(mod, "_converged_sampling", lambda scale=1.0: contextlib.nullcontext())
    return mod, fake


def test_emission_removal_reproduces_attenuated_starlight(driver):
    mod, _ = driver
    comp = {"redshift": 0.0, "dust": {"type": "Calzetti", "Av": 1.0}}
    with mod._without_dust_emission():
        mg = mod._build_model(comp)
    np.testing.assert_allclose(mg.spectrum_full, _ATTENUATED, rtol=0, atol=0)


def test_default_build_keeps_the_emission(driver):
    mod, _ = driver
    comp = {"redshift": 0.0, "dust": {"type": "Calzetti", "Av": 1.0}}
    mg = mod._build_model(comp)
    expected = _ATTENUATED + _DUST_FLUX * (2.0 * 1.0 + 0.01)
    np.testing.assert_allclose(mg.spectrum_full, expected)


def test_original_emission_class_is_restored_after_an_error(driver):
    mod, fake = driver
    with pytest.raises(RuntimeError, match="boom"), mod._without_dust_emission():
        assert fake.dust_emission is not _FakeDustEmission
        raise RuntimeError("boom")
    assert fake.dust_emission is _FakeDustEmission
