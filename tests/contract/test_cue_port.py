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

    # Should have 15 parameters: 4 standard + 7 ionizing spectrum + 3 gas extras
    # + neb_fdust (CIGALE LyC dust-absorption fraction, added 2026-06).
    assert len(decls) == 15, f"Expected 15 params, got {len(decls)}"

    # Check names and units
    param_names = {d.name for d in decls}
    expected = {
        "neb_logU",
        "neb_logZ_gas",
        "neb_fesc",
        "neb_fesc_lya",
        "neb_fdust",
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

    import chex
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

    # ``cloudyfsps_only=False`` matches the SEDComponent path, which does
    # not currently apply the cloudyfsps subset filter. Without this both
    # paths report different line counts (128 vs 138) and the test fails
    # before any luminosity comparison — that was a bug in my own setup
    # in #477 surfaced once the weights file landed in CI.
    legacy_wav, legacy_lum = backend._forward_lines(
        {k: jnp.asarray(v) for k, v in backend_params.items()},
        cloudyfsps_only=False,
    )

    # Both paths sort lines by wavelength internally; same shape and waves.
    chex.assert_equal_shape([published["line_waves"], legacy_wav])
    chex.assert_trees_all_close(
        published["line_waves"],
        legacy_wav,
        atol=1e-3,
        custom_message="line wavelengths diverged between the two Cue paths",
    )
    # Lines span many decades; compare in log space at the crossval tolerance.
    log_comp = jnp.log10(jnp.maximum(published["line_lums"], 1e-300))
    log_legacy = jnp.log10(jnp.maximum(legacy_lum, 1e-300))
    chex.assert_trees_all_close(
        log_comp,
        log_legacy,
        atol=1e-3,
        custom_message=(
            "line luminosities diverge between CueNebularSEDComponent.predict "
            "and CueBackend._forward_lines — the two paths are out of sync."
        ),
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


@pytest.mark.parametrize(
    "nion_value, expected_logqion",
    [
        # Floor: a literal zero ionising-photon rate is unphysical. Clamp to
        # ``1e-300`` so ``log10`` stays finite for the JIT trace; the result
        # (≈ −300 dex) drives the ±50-clip in ``predict_all_lines`` into
        # uniform saturation — the same load-loud signature #480 added for
        # gas_logq normalisation bugs.
        (0.0, -300.0),
        # A small but positive ``nion`` should pass through to log10 cleanly
        # (no spurious flooring on physically-tiny but legal inputs).
        (1e-200, -200.0),
        # The CIGALE / typical SF-galaxy value as a sanity-check anchor.
        (1e53, 53.0),
    ],
)
def test_cue_sed_component_nion_clamp_handles_degenerate_inputs(
    monkeypatch, nion_value, expected_logqion
):
    """``CueNebularSEDComponent.predict`` must not produce NaN/inf or hide a
    degenerate ``nion`` from upstream behind a hardcoded floor.

    Regression: PR #477 introduced ``gas_logqion = log10(jnp.maximum(nion, 1.0))``
    which silently substituted ``log10(1.0) = 0`` when upstream published
    ``nion = 0`` (zero stellar mass, all-quiescent SFH, or a stellar-component
    bug). The follow-up uses a log-domain floor (``1e-300``) so the resulting
    ``gas_logqion ≈ -300`` saturates the ±50-clip uniformly — a visible bug
    signature rather than a near-physical silent fixup.
    """
    import chex
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

    chex.assert_tree_all_finite(jnp.asarray(captured["gas_logqion"]))
    assert captured["gas_logqion"] == pytest.approx(expected_logqion, rel=1e-5, abs=1e-5)


def test_cue_predict_all_lines_clip_bounded_at_50dex():
    """The exponent clip in ``predict_all_lines`` must not exceed ±50 dex.

    The ±100-dex bound shipped with the original Cue port was wide enough
    that the +51-dex ``gas_logq = logU`` bug (#477) produced saturated-but-
    near-physical line luminosities instead of obviously-broken output. ±50
    dex is the discipline: any normalisation slip ≥ 50 dex now hits the
    ceiling/floor uniformly and is visible in inspection.

    Source-pinning regression: re-loosening the clip would silently undo
    that defence. We grep the source rather than calling the function so
    the assertion fails immediately on any future widening.
    """
    import inspect

    from tengri.components.nebular import cue

    source = inspect.getsource(cue.predict_all_lines)
    assert "jnp.clip(exponent, -50.0, 50.0)" in source, (
        "predict_all_lines clip widened from ±50 dex; see #477 follow-up for "
        "why ±100 silently masked the gas_logq normalisation bug."
    )

    cont_source = inspect.getsource(cue.predict_continuum)
    assert "jnp.clip(exponent, -50.0, 50.0)" in cont_source, (
        "predict_continuum clip widened from ±50 dex; same rationale as the "
        "lines path — keep both clips in lockstep."
    )


def test_cue_predict_all_lines_clip_saturates_on_synthetic_bug():
    """A synthetic +51-dex error in ``gas_logq`` (the magnitude of the
    pre-#477 bug) must drive every line to the same saturated clip value,
    making the bug visually obvious rather than near-physical."""
    import os

    import chex
    import jax.numpy as jnp

    from tengri.components.nebular import _DEFAULT_CUE_WEIGHTS_PATH
    from tengri.components.nebular.cue import load_cue_weights, predict_all_lines

    if not os.path.exists(_DEFAULT_CUE_WEIGHTS_PATH):
        pytest.skip(f"Cue weights file not found at {_DEFAULT_CUE_WEIGHTS_PATH}")
    weights = load_cue_weights(str(_DEFAULT_CUE_WEIGHTS_PATH))

    # NN-ready 12-vector (the function broadcasts internally over the 16
    # batched sub-emulators). Values centred in their training ranges; the
    # test isn't about absolute luminosities, only about how the clip
    # responds to a synthetic normalisation error.
    nn_params = jnp.array(
        [19.7, 5.3, 1.6, 0.6, 3.9, 0.01, 0.2, 48.5, 2.0, 0.0, 0.0, 0.0],
        dtype=jnp.float32,
    )
    correct_gas_logq = jnp.asarray(48.5)
    bug_gas_logq = jnp.asarray(-3.0)  # the pre-#477 value
    gas_logqion = jnp.asarray(52.0)

    _wav, lum_correct = predict_all_lines(
        nn_params=nn_params,
        weights=weights,
        gas_logq=correct_gas_logq,
        gas_logqion=gas_logqion,
    )
    _wav, lum_buggy = predict_all_lines(
        nn_params=nn_params,
        weights=weights,
        gas_logq=bug_gas_logq,
        gas_logqion=gas_logqion,
    )

    chex.assert_equal_shape([lum_correct, lum_buggy])
    chex.assert_tree_all_finite(lum_correct)
    chex.assert_tree_all_finite(lum_buggy)

    log_correct = jnp.log10(jnp.maximum(lum_correct, 1e-300))
    log_buggy = jnp.log10(jnp.maximum(lum_buggy, 1e-300))

    # Correct path: a healthy line forest spans many decades.
    span_correct = float(log_correct.max() - log_correct.min())
    assert span_correct > 1.0, (
        f"correct-gas_logq line luminosities should span multiple decades, got {span_correct} dex"
    )

    # Buggy +51-dex path: the bulk of lines (>90%) hit the +50 clip ceiling
    # uniformly. A few intrinsically-faint lines may sit below the ceiling
    # but the bulk is heavily peaked at the saturation value — that is the
    # bug-detection signal the tight clip provides. Pre-#477 the ±100 clip
    # let the same lines emerge at ~+99 dex, which looked plausible.
    saturated_frac = float((log_buggy >= 49.9).mean())
    assert saturated_frac > 0.9, (
        f"buggy +51-dex gas_logq should saturate the bulk of lines at the "
        f"+50 clip ceiling; only {saturated_frac:.1%} reached it. The clip "
        f"may be too loose to catch this class of normalisation bug."
    )
    # And the median is right at the ceiling — the bug signature is a
    # degenerate distribution at the clip value.
    assert float(jnp.median(log_buggy)) >= 49.9


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
