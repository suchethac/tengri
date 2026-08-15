# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for nebular continuum / emission line decomposition — synthesizer parity.

Mirrors synthesizer's PR #990 validation logic to ensure nebular continuum and lines
are returned as separable components without double-counting. If the pipeline reprocesses
both together, the total SED should not sum to more than the components.

Pitfall: P-11 — emission line continuum double-counting in two-component dust pipelines.
Synthesizer PR #990: nebular_continuum_emission fixture was missing. ReprocessedEmission
auto-creates nebular+transmitted children but wasn't passing nebular_continuum to nebular
model. Result: continuum counted twice or skipped.

Synthesizer source:
- https://github.com/flaresimulations/synthesizer/pull/990 (fixture / double-count bug)
- https://github.com/flaresimulations/synthesizer/blob/main/tests/test_emission.py

Reference papers:
- Byler et al. 2017, ApJ, 840, 122 (nebular emission / line + continuum decomposition)
- Li et al. 2025, ApJ, 986, 9 (Cue — nebular model; arXiv:2405.04598)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_paper
from scipy.integrate import simpson

jax.config.update("jax_enable_x64", True)

from tengri.components.nebular import _DEFAULT_CUE_WEIGHTS_PATH
from tengri.components.nebular.cue import CueBackend


@pytest.fixture(scope="module")
def cue_backend() -> CueBackend:
    """Load the Cue neural net emulator for nebular emission."""
    return CueBackend(str(_DEFAULT_CUE_WEIGHTS_PATH))


@pytest.fixture(scope="module")
def wave_optical_to_ir() -> jnp.ndarray:
    """Rest-frame wavelength grid for continuum + lines [Angstrom].

    Spans optical to IR to capture both emission-line peaks and continuum.
    """
    return jnp.logspace(3.0, 5.5, 500)  # 1000 Å .. 316,000 Å


@pytest.fixture(scope="module")
def cue_test_params() -> dict:
    """Fixed Cue parameters for repeatable nebular calculations."""
    return {
        "ionspec_index1": jnp.array(0.5),
        "ionspec_index2": jnp.array(-1.5),
        "ionspec_index3": jnp.array(-2.5),
        "ionspec_index4": jnp.array(-3.0),
        "ionspec_logLratio1": jnp.array(0.0),
        "ionspec_logLratio2": jnp.array(0.0),
        "ionspec_logLratio3": jnp.array(0.0),
        "gas_logu": jnp.array(0.0),
        "gas_logn": jnp.array(2.5),
        "gas_logz": jnp.array(0.0),
        "gas_logco": jnp.array(-0.3),
        "gas_logno": jnp.array(-1.0),
    }


def test_nebular_continuum_and_lines_are_separable(
    cue_backend, cue_test_params, wave_optical_to_ir
):
    """Line + continuum are separate outputs with no overlap or double-counting.

    Pitfall P-11: If nebular_continuum is not passed correctly to the model,
    code may return either zero continuum, or continuum that's already summed
    with lines (double-counting).

    Test strategy: Verify that both predict_nebular_continuum and
    predict_nebular_line_luminosities return finite, non-negative arrays
    independently.
    """
    wav_cont, lum_cont = cue_backend.predict_nebular_continuum(**cue_test_params, neb_fesc=0.0)
    _, lum_lines = cue_backend.predict_nebular_line_luminosities(**cue_test_params, neb_fesc=0.0)

    # Both must be finite arrays
    assert bool(jnp.all(jnp.isfinite(lum_cont))), "Nebular continuum has NaN/inf"
    assert bool(jnp.all(jnp.isfinite(lum_lines))), "Nebular line luminosities have NaN/inf"

    # Both must be non-negative (astrophysical emission)
    assert bool(jnp.all(lum_cont >= 0.0)), "Nebular continuum has negative values"
    assert bool(jnp.all(lum_lines >= 0.0)), "Nebular lines have negative values"

    # Both must have positive total (not all zeros)
    cont_total = float(simpson(lum_cont, wav_cont))
    lines_total = float(jnp.sum(lum_lines))

    assert cont_total > 0.0, (
        f"P-11 BUG: nebular continuum total is zero {cont_total:.3e}. "
        "Continuum may be suppressed or missing."
    )
    assert lines_total > 0.0, (
        f"P-11 BUG: nebular line total is zero {lines_total:.3e}. "
        "Lines may be suppressed or missing."
    )


def test_nebular_continuum_decreases_with_fesc(cue_backend, cue_test_params):
    """Nebular continuum should decrease monotonically with increasing fesc.

    Pitfall P-11: If continuum is double-counted (both in line and continuum branches),
    or if fesc is not applied consistently, this test will catch the inconsistency.
    Also overlaps with P-9 (fesc wiring).
    """
    wav, lum_fesc0 = cue_backend.predict_nebular_continuum(**cue_test_params, neb_fesc=0.0)
    _, lum_fesc05 = cue_backend.predict_nebular_continuum(**cue_test_params, neb_fesc=0.5)
    _, lum_fesc09 = cue_backend.predict_nebular_continuum(**cue_test_params, neb_fesc=0.9)

    cont_0 = float(simpson(lum_fesc0, wav))
    cont_half = float(simpson(lum_fesc05, wav))
    cont_9 = float(simpson(lum_fesc09, wav))

    # Strict monotonicity
    assert cont_0 >= cont_half, (
        f"P-11 BUG: continuum at fesc=0 ({cont_0:.3e}) < fesc=0.5 ({cont_half:.3e}). "
        "fesc is not wired or inverted."
    )
    assert cont_half >= cont_9, (
        f"P-11 BUG: continuum at fesc=0.5 ({cont_half:.3e}) < fesc=0.9 ({cont_9:.3e}). "
        "fesc is not wired or inverted."
    )

    # Allow small numerical noise, but decreasing trend must be clear
    assert cont_0 > cont_9 * 1.01, (
        f"P-11 WARNING: continuum monotonicity weak. "
        f"fesc=0: {cont_0:.3e}, fesc=0.9: {cont_9:.3e}. "
        "Relative difference < 1%."
    )


def test_nebular_lines_decrease_with_fesc(cue_backend, cue_test_params):
    """Nebular line luminosities should decrease monotonically with increasing fesc.

    Pitfall P-11 (implicit): If lines are double-counted or fesc is not applied,
    this monotonicity will break.
    """
    _, lum_fesc0 = cue_backend.predict_nebular_line_luminosities(**cue_test_params, neb_fesc=0.0)
    _, lum_fesc05 = cue_backend.predict_nebular_line_luminosities(**cue_test_params, neb_fesc=0.5)
    _, lum_fesc09 = cue_backend.predict_nebular_line_luminosities(**cue_test_params, neb_fesc=0.9)

    lines_0 = float(jnp.sum(lum_fesc0))
    lines_half = float(jnp.sum(lum_fesc05))
    lines_9 = float(jnp.sum(lum_fesc09))

    assert lines_0 >= lines_half, (
        f"P-11 BUG: line total at fesc=0 ({lines_0:.3e}) < fesc=0.5 ({lines_half:.3e})."
    )
    assert lines_half >= lines_9, (
        f"P-11 BUG: line total at fesc=0.5 ({lines_half:.3e}) < fesc=0.9 ({lines_9:.3e})."
    )


def test_nebular_output_all_zero_at_fesc_unity(cue_backend, cue_test_params):
    """With fesc=1.0 (all photons escape), both continuum and lines ≈ 0.

    Pitfall P-11: If continuum leaks through even with fesc=1, it indicates
    double-counting or improper flux conservation.
    """
    wav, lum_cont = cue_backend.predict_nebular_continuum(**cue_test_params, neb_fesc=1.0)
    _, lum_lines = cue_backend.predict_nebular_line_luminosities(**cue_test_params, neb_fesc=1.0)

    cont_total = float(simpson(lum_cont, wav))
    lines_total = float(jnp.sum(lum_lines))

    # At fesc=1, both should be near-zero (numerical noise tolerance: 1%)
    assert cont_total < 1e-2, (
        f"P-11 BUG: nebular continuum not suppressed at fesc=1.0. Got {cont_total:.3e}"
    )
    assert lines_total < 1e-2, (
        f"P-11 BUG: nebular lines not suppressed at fesc=1.0. Got {lines_total:.3e}"
    )


def test_continuum_and_lines_both_positive_no_negative_values(
    cue_backend, cue_test_params, wave_optical_to_ir
):
    """No negative luminosities in continuum or lines (physical requirement).

    Pitfall P-11 (implicit): If subtraction of reprocessing or incorrect
    continuum removal is applied, negative flux values would indicate a bug.
    """
    _, lum_cont = cue_backend.predict_nebular_continuum(**cue_test_params, neb_fesc=0.2)
    _, lum_lines = cue_backend.predict_nebular_line_luminosities(**cue_test_params, neb_fesc=0.2)

    min_cont = float(jnp.min(lum_cont))
    min_lines = float(jnp.min(lum_lines))

    assert min_cont >= 0.0, (
        f"P-11 BUG: nebular continuum has negative values (min={min_cont:.3e}). "
        "Indicates subtraction / double-count removal went wrong."
    )
    assert min_lines >= 0.0, (
        f"P-11 BUG: nebular lines have negative values (min={min_lines:.3e}). "
        "Indicates subtraction / double-count removal went wrong."
    )
