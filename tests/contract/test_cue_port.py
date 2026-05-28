# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for CueNebularSEDComponent.

Validates the SEDModelComponent port of the Cue NN emulator:
- Registry and isinstance checks
- Parameter declarations flow correctly
- Parity with original Cue backend on line luminosities
- Graceful skip when weights are missing
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract


def test_cue_nebular_component_registered():
    """Cue component is registered in the global registry."""
    from tengri.components.sed_model_component import _REGISTRY

    assert "cue_emulator" in _REGISTRY, "cue_emulator not in component registry"
    assert _REGISTRY["cue_emulator"].__name__ == "CueNebularSEDComponent"


def test_cue_nebular_component_isinstance():
    """CueNebularSEDComponent inherits from SEDModelComponent."""
    from tengri.components.nebular.cue_model import CueNebularSEDComponent
    from tengri.components.sed_model_component import SEDModelComponent

    comp = CueNebularSEDComponent()
    assert isinstance(comp, SEDModelComponent)


def test_cue_nebular_declared_parameters():
    """Declared parameters include 14 Cue knobs with correct units."""
    from tengri.components.nebular.cue_model import CueNebularSEDComponent

    comp = CueNebularSEDComponent()
    decls = comp.declared_parameters()

    # Should have 14 parameters: 4 standard + 7 ionizing spectrum + 3 gas extras
    assert len(decls) == 14, f"Expected 14 params, got {len(decls)}"

    # Check names and units
    param_names = {d.name for d in decls}
    expected = {
        "neb_logU",
        "neb_logZ_gas",
        "neb_fesc",
        "neb_fesc_lya",
        "neb_ionspec_index1",
        "neb_ionspec_index2",
        "neb_ionspec_index3",
        "neb_ionspec_index4",
        "neb_ionspec_logLratio1",
        "neb_ionspec_logLratio2",
        "neb_ionspec_logLratio3",
        "neb_gas_logn",
        "neb_gas_logno",
        "neb_gas_logco",
    }

    # Note: the actual names will include the neb_ prefix
    assert param_names == expected, f"Param mismatch: {param_names ^ expected}"

    # Spot-check units
    for decl in decls:
        has_log = any(x in decl.name for x in ["logU", "logZ", "index", "ratio"])
        if has_log:
            assert decl.units in {
                "dex",
                "dimensionless",
            }, f"Units for {decl.name}: {decl.units}"


def test_cue_nebular_inputs_outputs():
    """Cross-component contract declares inputs and outputs."""
    from tengri.components.nebular.cue_model import CueNebularSEDComponent

    comp = CueNebularSEDComponent()
    inputs = comp.inputs()
    outputs = comp.outputs()

    # Inputs: ssp_ages_yr, age_weights
    input_names = {dk.name for dk in inputs}
    assert "ssp_ages_yr" in input_names
    assert "age_weights" in input_names

    # Outputs: line_waves, line_lums
    output_names = {dk.name for dk in outputs}
    assert "line_waves" in output_names
    assert "line_lums" in output_names


def test_cue_nebular_parity_vs_backend():
    """``CueNebularSEDComponent.predict`` and ``CueBackend._forward_lines``
    must return identical line luminosities for matching parameters.

    Regression for the silent normalisation bug where ``cue_model.py`` used
    ``gas_logq = logU`` instead of ``_logq_from_logu(logU, gas_logn)``: the
    pre-existing parity test was unconditionally ``skipif(True)``-skipped
    AND only asserted ``len(line_lums) > 0`` even when run, so two Cue
    forward paths diverged by ~51 dex without surfacing in CI. This rewrite
    skips only when the weights file is genuinely absent and compares the
    actual numerical outputs.
    """
    import os

    import jax.numpy as jnp

    from tengri.components.nebular.cue import CueBackend
    from tengri.components.nebular.cue_model import CueNebularSEDComponent

    weights_path = "data/cue_weights.npz"
    if not os.path.exists(weights_path):
        pytest.skip(f"Cue weights file not found at {weights_path}")

    backend = CueBackend(weights_path=weights_path, ssp_data=None)

    # Drive the SEDComponent path with the same backend instance so the only
    # difference between the two is the in-component ``gas_logq`` arithmetic.
    comp = CueNebularSEDComponent()
    comp.data = backend

    # Use values inside every prior so neither path clips on inputs.
    shared = {
        "logU": -3.0,
        "logZ_gas": -0.3,
        "ionspec_index1": 2.0,
        "ionspec_index2": 1.5,
        "ionspec_index3": 1.0,
        "ionspec_index4": 0.5,
        "ionspec_logLratio1": 1.0,
        "ionspec_logLratio2": 0.5,
        "ionspec_logLratio3": 0.5,
        "gas_logn": 2.0,
        "gas_logno": 0.0,
        "gas_logco": 0.0,
    }
    comp_params = {k: jnp.asarray(v) for k, v in shared.items()}

    backend_params = {f"gas_{k}" if k.startswith("log") else k: v for k, v in shared.items()}
    backend_params["gas_logu"] = backend_params.pop("gas_logU")
    backend_params["gas_logz"] = backend_params.pop("gas_logZ_gas")
    # Legacy path takes gas_logqion as an explicit param; match the SEDComponent
    # placeholder so the two paths line up on Q_H too.
    # Pin both paths to the same Q_H so the only thing under test is the
    # Strömgren-corrected gas_logq normalisation.
    legacy_logqion = 49.1
    backend_params["gas_logqion"] = legacy_logqion
    backend_params["neb_fesc"] = 0.0
    backend_params["neb_fesc_lya"] = 0.0

    wave = jnp.linspace(100.0, 10000.0, 1000)
    sed_in = jnp.zeros_like(wave)

    _sed_out, published = comp.predict(
        comp_params, sed_in, wave, nion=jnp.asarray(10.0**legacy_logqion)
    )

    legacy_wav, legacy_lum = backend._forward_lines(
        {k: jnp.asarray(v) for k, v in backend_params.items()},
    )

    # Match line ordering (both paths sort by wavelength internally).
    assert published["line_waves"].shape == legacy_wav.shape, (
        f"line count mismatch: {published['line_waves'].shape} vs {legacy_wav.shape}"
    )
    assert jnp.allclose(published["line_waves"], legacy_wav, rtol=0, atol=1e-3), (
        "line wavelengths diverged between the two Cue paths"
    )
    # Lines span many decades; compare in log space at a tight tolerance.
    log_comp = jnp.log10(jnp.maximum(published["line_lums"], 1e-300))
    log_legacy = jnp.log10(jnp.maximum(legacy_lum, 1e-300))
    assert jnp.allclose(log_comp, log_legacy, atol=1e-3), (
        "line luminosities diverge between CueNebularSEDComponent.predict "
        "and CueBackend._forward_lines — the two paths are out of sync."
    )


def test_cue_nebular_weights_missing_graceful():
    """Component returns empty outputs when weights file is missing.

    Simulates the case where data/cue_weights.npz is not available.
    """
    import jax.numpy as jnp

    from tengri.components.nebular.cue_model import CueNebularSEDComponent

    comp = CueNebularSEDComponent()

    # Don't load weights (self.data remains None)
    # This simulates the weights file being missing

    params = {
        "logU": -3.0,
        "logZ_gas": -0.3,
        "fesc": 0.0,
        "fesc_lya": 0.0,
        "ionspec_index1": 2.0,
        "ionspec_index2": 1.5,
        "ionspec_index3": 1.0,
        "ionspec_index4": 0.5,
        "ionspec_logLratio1": 1.0,
        "ionspec_logLratio2": 0.5,
        "ionspec_logLratio3": 0.5,
        "gas_logn": 2.0,
        "gas_logno": 0.0,
        "gas_logco": 0.0,
    }

    wave = jnp.linspace(100, 10000, 100)
    sed_in = jnp.zeros_like(wave)

    sed_out, published = comp.predict(params, sed_in, wave)

    # Should return empty arrays gracefully
    assert published["line_waves"].shape[0] == 0
    assert published["line_lums"].shape[0] == 0
    assert jnp.allclose(sed_out, sed_in)


def test_cue_sed_component_uses_stromgren_gas_logq(monkeypatch):
    """``CueNebularSEDComponent.predict`` must pass the Strömgren-corrected
    ``gas_logq`` (from :func:`_logq_from_logu`) to ``predict_all_lines``, not
    the raw ``logU`` parameter.

    Regression for the silent normalisation bug in cue_model.py: an earlier
    revision computed ``gas_logq = logU`` (~−3 dex) instead of the
    Strömgren-corrected ionising-photon rate
    ``logU + log(4π) + 2·log(R_S) + logn + log(c)`` (~+48 dex for typical HII
    region values). The legacy :meth:`CueBackend._forward_lines` path used the
    correct formula; the SEDComponent port silently disagreed by ~51 dex,
    which the ``±100``-dex clip in :func:`predict_all_lines` masked instead
    of surfacing.

    Weights-independent: we monkey-patch ``predict_all_lines`` to capture the
    ``gas_logq`` argument, so this test runs without ``data/cue_weights.npz``.

    References
    ----------
    Li+2024, ApJ 969, 28 — Cue training convention.
    """
    import jax.numpy as jnp

    from tengri.components.nebular import cue_model as cue_model_mod
    from tengri.components.nebular.cue import _logq_from_logu
    from tengri.components.nebular.cue_model import CueNebularSEDComponent

    captured = {}

    def _capture(*, nn_params, weights, gas_logq, gas_logqion):
        captured["gas_logq"] = float(jnp.asarray(gas_logq))
        captured["gas_logqion"] = float(jnp.asarray(gas_logqion))
        # Return shapes consistent with predict_all_lines so .predict survives.
        wav = jnp.array([6564.61, 5008.24, 4862.68], dtype=jnp.float32)
        lum = jnp.zeros(3, dtype=jnp.float32)
        return wav, lum

    def _capture_cont(**_kwargs):
        wav = jnp.linspace(100.0, 10000.0, 8, dtype=jnp.float32)
        return wav, jnp.zeros_like(wav)

    monkeypatch.setattr(cue_model_mod, "predict_all_lines", _capture)
    monkeypatch.setattr(cue_model_mod, "predict_continuum", _capture_cont)

    comp = CueNebularSEDComponent()

    # Provide a minimal stand-in backend so predict() does not return early on
    # the "weights missing" branch. We only need ``backend.weights`` to be a
    # truthy attribute — _capture ignores it.
    class _StubBackend:
        weights = object()

    comp.data = _StubBackend()

    logU = -3.0
    gas_logn = 2.0
    params = {
        "logU": jnp.asarray(logU),
        "logZ_gas": jnp.asarray(-0.3),
        "ionspec_index1": jnp.asarray(2.0),
        "ionspec_index2": jnp.asarray(1.5),
        "ionspec_index3": jnp.asarray(1.0),
        "ionspec_index4": jnp.asarray(0.5),
        "ionspec_logLratio1": jnp.asarray(1.0),
        "ionspec_logLratio2": jnp.asarray(0.5),
        "ionspec_logLratio3": jnp.asarray(0.5),
        "gas_logn": jnp.asarray(gas_logn),
        "gas_logno": jnp.asarray(0.0),
        "gas_logco": jnp.asarray(0.0),
    }
    wave = jnp.linspace(100.0, 10000.0, 16)
    sed_in = jnp.zeros_like(wave)

    # Q_H of a typical Milky-Way-ish SF galaxy — chosen far from the legacy
    # 49.1-dex placeholder so we can distinguish whether the wiring is live.
    nion_value = 1e53
    comp.predict(params, sed_in, wave, nion=jnp.asarray(nion_value))

    expected = float(_logq_from_logu(jnp.asarray(logU), jnp.asarray(gas_logn)))
    # Sanity-check: the Strömgren-corrected value is far from the buggy `logU`
    # value, so a passing assertion can only mean the fix is in place.
    assert expected > 40.0, f"_logq_from_logu sanity check failed: {expected} dex — expected ~48"
    assert captured["gas_logq"] == pytest.approx(expected, rel=1e-5), (
        f"CueNebularSEDComponent passed gas_logq={captured['gas_logq']} "
        f"to predict_all_lines, expected {expected} (Strömgren-corrected). "
        "Did cue_model.py revert `gas_logq = _logq_from_logu(logU, gas_logn)` "
        "to the buggy `gas_logq = logU`?"
    )


def test_cue_sed_component_uses_ssp_derived_qion(monkeypatch):
    """``CueNebularSEDComponent.predict`` must compute ``gas_logqion`` from
    the upstream ``nion`` input, not the legacy ``49.1`` placeholder.

    Regression for the second original-behaviour defect uncovered alongside
    the ``gas_logq`` fix: the SEDComponent port shipped with
    ``gas_logqion = jnp.asarray(49.1)`` since commit 075630b3, which is
    3-4 dex below the actual Q_H of any realistic SF galaxy and explains
    the residual "CIGALE stronger than tengri" gap after the Strömgren
    correction landed. Fitting the 7 ionising-spectrum-shape parameters is
    incoherent without consuming the matching amplitude from the SSP.

    Weights-independent: we monkey-patch ``predict_all_lines`` to capture
    the ``gas_logqion`` argument.
    """
    import jax.numpy as jnp

    from tengri.components.nebular import cue_model as cue_model_mod
    from tengri.components.nebular.cue_model import CueNebularSEDComponent

    captured = {}

    def _capture(*, nn_params, weights, gas_logq, gas_logqion):
        captured["gas_logqion"] = float(jnp.asarray(gas_logqion))
        wav = jnp.array([6564.61], dtype=jnp.float32)
        lum = jnp.zeros(1, dtype=jnp.float32)
        return wav, lum

    def _capture_cont(**_kwargs):
        wav = jnp.linspace(100.0, 10000.0, 4, dtype=jnp.float32)
        return wav, jnp.zeros_like(wav)

    monkeypatch.setattr(cue_model_mod, "predict_all_lines", _capture)
    monkeypatch.setattr(cue_model_mod, "predict_continuum", _capture_cont)

    comp = CueNebularSEDComponent()

    class _StubBackend:
        weights = object()

    comp.data = _StubBackend()

    nion_value = 5.7e52  # photons/s — well separated from 10**49.1
    params = {
        "logU": jnp.asarray(-3.0),
        "logZ_gas": jnp.asarray(-0.3),
        "ionspec_index1": jnp.asarray(2.0),
        "ionspec_index2": jnp.asarray(1.5),
        "ionspec_index3": jnp.asarray(1.0),
        "ionspec_index4": jnp.asarray(0.5),
        "ionspec_logLratio1": jnp.asarray(1.0),
        "ionspec_logLratio2": jnp.asarray(0.5),
        "ionspec_logLratio3": jnp.asarray(0.5),
        "gas_logn": jnp.asarray(2.0),
        "gas_logno": jnp.asarray(0.0),
        "gas_logco": jnp.asarray(0.0),
    }
    wave = jnp.linspace(100.0, 10000.0, 16)
    sed_in = jnp.zeros_like(wave)

    comp.predict(params, sed_in, wave, nion=jnp.asarray(nion_value))

    expected = float(jnp.log10(jnp.asarray(nion_value)))
    assert captured["gas_logqion"] == pytest.approx(expected, rel=1e-5), (
        f"CueNebularSEDComponent passed gas_logqion={captured['gas_logqion']} "
        f"to predict_all_lines, expected log10({nion_value:.3e}) ≈ {expected}. "
        "The SSP-derived Q_H wiring (`gas_logqion = log10(nion)`) was reverted "
        "to the legacy 49.1-dex placeholder."
    )


def test_cue_sed_component_requires_nion():
    """``CueNebularSEDComponent.predict`` must fail loudly when ``nion`` is
    not supplied — silent fallback to a placeholder is what got us into the
    49.1-dex undercount in the first place."""
    import jax.numpy as jnp

    from tengri.components.nebular.cue_model import CueNebularSEDComponent

    comp = CueNebularSEDComponent()

    class _StubBackend:
        weights = object()

    comp.data = _StubBackend()

    params = {
        "logU": jnp.asarray(-3.0),
        "logZ_gas": jnp.asarray(-0.3),
        "ionspec_index1": jnp.asarray(2.0),
        "ionspec_index2": jnp.asarray(1.5),
        "ionspec_index3": jnp.asarray(1.0),
        "ionspec_index4": jnp.asarray(0.5),
        "ionspec_logLratio1": jnp.asarray(1.0),
        "ionspec_logLratio2": jnp.asarray(0.5),
        "ionspec_logLratio3": jnp.asarray(0.5),
        "gas_logn": jnp.asarray(2.0),
        "gas_logno": jnp.asarray(0.0),
        "gas_logco": jnp.asarray(0.0),
    }
    wave = jnp.linspace(100.0, 10000.0, 16)
    sed_in = jnp.zeros_like(wave)

    with pytest.raises(KeyError, match="nion"):
        comp.predict(params, sed_in, wave)  # missing nion → KeyError


def test_cue_nebular_precompute_stores_backend():
    """Precompute() loads weights and stores on self.data."""
    import jax.numpy as jnp

    from tengri.components.nebular.cue_model import CueNebularSEDComponent

    comp = CueNebularSEDComponent()
    state = comp.precompute(wave_grid=jnp.linspace(100, 10000, 100))

    # Weights may or may not be loaded depending on file availability
    # State should be created regardless
    assert state.name == "cue_emulator"

    # If weights were loaded, self.data should be set
    if hasattr(comp, "data") and comp.data is not None:
        from tengri.components.nebular.cue import CueBackend

        assert isinstance(comp.data, CueBackend)
