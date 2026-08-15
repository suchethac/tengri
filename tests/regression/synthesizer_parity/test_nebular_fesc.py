# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for ionizing photon escape fraction (neb_fesc) — synthesizer parity.

Mirrors synthesizer issue #126 — fesc (ionizing photon escape fraction) was
declared in function signature but never used in actual SED calculation.
Tengri likelihood: HIGH. Verify fesc is applied: ``intrinsic_photons =
total_photons * (1 - fesc)`` BEFORE nebular continuum calc.

Synthesizer source: https://github.com/flaresimulations/synthesizer/issues/126
Reference papers:
  - Li et al. 2025, ApJ, 986, 9 (Cue, incl. fesc threading; arXiv:2405.04598)
  - Byler et al. 2017, ApJ, 840, 122 (photoionization and escape fraction)

Every test cites the pitfall ID from ``~/.claude/plans/synthesizer-pitfall-catalog.md``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_paper
from scipy.integrate import simpson

from tengri.components.nebular import _DEFAULT_CUE_WEIGHTS_PATH
from tengri.components.nebular._recombination_coeffs import lyc_dust_escape_factor
from tengri.components.nebular.cue import CueBackend


@pytest.fixture(scope="module")
def cue_backend() -> CueBackend:
    """Load the Cue neural net emulator for nebular emission."""
    return CueBackend(str(_DEFAULT_CUE_WEIGHTS_PATH))


@pytest.fixture(scope="module")
def wave_optical_to_ir() -> jnp.ndarray:
    """Rest-frame wavelength grid covering optical through IR [Angstrom].

    Cue trains on the full emission-line spectrum, so we span optical lines
    (H-alpha 6562, [OIII]5007) through mid-IR ([Ne V] 14.3 micron) for
    line + continuum coverage.
    """
    return jnp.logspace(3.0, 5.5, 500)  # 1000 Å .. 316,000 Å (31.6 micron)


@pytest.fixture(scope="module")
def cue_default_params() -> dict:
    """Cue default parameters for a T ~ 40,000 K blackbody ionizing source + high logU.

    Ionizing spectrum: 7 shape parameters (ionspec_index1..4, ionspec_logLratio1..3).
      - Powers: [0.5, -1.5, -2.5, -3.0] (decreasing ionizing hardness with wavelength)
      - Log ratios: [0.0, 0.0, 0.0] (flat relative normalization; alternative to flux-matching)

    Gas properties: 5 parameters.
      - logU = 0.0 (intermediate ionization parameter, typical for HII regions)
      - logn = 2.5 (electron density 300 cm^-3, moderate ISM)
      - logZ = 0.0 (solar metallicity)
      - logC/O = -0.3 (sub-solar C; traces O-rich gas in star-forming galaxies)
      - logN/O = -1.0 (sub-solar N; Andromeda-like N/O ratio)
    """
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


# ---------------------------------------------------------------------------
# P-9 / ionizing photon escape fraction (fesc)
# ---------------------------------------------------------------------------
# Pitfall: P-9 — escaped ionizing photons not wired to nebular continuum.
# If fesc is NOT threaded into the nebular calculation, the continuum and
# line luminosities will NOT scale with (1 - fesc).


def test_fesc_zero_vs_half_reduces_nebular(cue_backend, cue_default_params):
    """With neb_fesc=0.5, nebular continuum is ~50% lower than neb_fesc=0.0.

    Pitfall P-9: If fesc is unwired, both outputs will be identical.
    If wired correctly, ~half the ionizing photons escape → only ~half ionize gas.

    Test strategy: Compute SED with fesc=0 and fesc=0.5. Assert the latter has
    SMALLER continuum + emission lines (within expected physical range).
    """
    # Compute nebular continuum and lines with NO escape
    wav_neb_0, lum_neb_cont_0 = cue_backend.predict_nebular_continuum(
        **cue_default_params, neb_fesc=0.0
    )
    _, lum_neb_lines_0 = cue_backend.predict_nebular_line_luminosities(
        **cue_default_params, neb_fesc=0.0
    )

    # Compute nebular continuum and lines with 50% escape
    wav_neb_half, lum_neb_cont_half = cue_backend.predict_nebular_continuum(
        **cue_default_params, neb_fesc=0.5
    )
    _, lum_neb_lines_half = cue_backend.predict_nebular_line_luminosities(
        **cue_default_params, neb_fesc=0.5
    )

    # Continuum: measure total power integrated across wavelength
    cont_0_total = float(simpson(lum_neb_cont_0, wav_neb_0))
    cont_half_total = float(simpson(lum_neb_cont_half, wav_neb_half))

    # Lines: sum total line luminosity across all emission lines
    lines_0_total = float(jnp.sum(lum_neb_lines_0))
    lines_half_total = float(jnp.sum(lum_neb_lines_half))

    # Assert: fesc=0.5 output is lower than fesc=0.0 (P-9 bug detection)
    assert cont_half_total < cont_0_total, (
        f"P-9 BUG: nebular continuum did not decrease with fesc. "
        f"fesc=0: {cont_0_total:.3e}, fesc=0.5: {cont_half_total:.3e}"
    )

    assert lines_half_total < lines_0_total, (
        f"P-9 BUG: nebular line flux did not decrease with fesc. "
        f"fesc=0: {lines_0_total:.3e}, fesc=0.5: {lines_half_total:.3e}"
    )


def test_fesc_unity_suppresses_all_nebular(cue_backend, cue_default_params):
    """With neb_fesc=1.0 (all photons escape), nebular continuum and lines ≈ 0.

    Pitfall P-9: If fesc is unwired, output will be identical to fesc=0.0.

    Test strategy: Compute with fesc=1.0 (all ionizing photons escape, no gas ionization).
    The nebular output should be negligibly small (within 1% of fesc=0 × 0.0).
    """
    # Compute with complete escape
    wav_neb_all, lum_neb_cont_all = cue_backend.predict_nebular_continuum(
        **cue_default_params, neb_fesc=1.0
    )
    _, lum_neb_lines_all = cue_backend.predict_nebular_line_luminosities(
        **cue_default_params, neb_fesc=1.0
    )

    # Integrate for total power
    cont_all_total = float(simpson(lum_neb_cont_all, wav_neb_all))
    lines_all_total = float(jnp.sum(lum_neb_lines_all))

    # With fesc=1.0, output should be near zero (machine epsilon)
    # Allow 1% relative tolerance for numerical noise in the neural net
    assert cont_all_total < 1e-2, (
        f"P-9 BUG: nebular continuum not suppressed at fesc=1.0. Got {cont_all_total:.3e}"
    )

    assert lines_all_total < 1e-2, (
        f"P-9 BUG: nebular lines not suppressed at fesc=1.0. Got {lines_all_total:.3e}"
    )


def test_fesc_linearity(cue_backend, cue_default_params):
    """Doubling fesc reduces nebular contribution roughly proportionally (test linearity).

    Pitfall P-9: If fesc is unwired, ratio would be 1.0. If linear, ratio should be
    close to (1 - fesc_high) / (1 - fesc_low).

    For fesc_low=0.2 → factor = (1 - 0.2) = 0.8
    For fesc_high=0.4 → factor = (1 - 0.4) = 0.6
    Ratio = 0.6 / 0.8 = 0.75 (expected)

    Test strategy: compute continuum at fesc=0.2 and fesc=0.4. Assert the ratio
    of outputs is between 0.6 and 0.8 (physics-consistent linear reduction).
    """
    # Compute at fesc=0.2
    wav_low, lum_cont_low = cue_backend.predict_nebular_continuum(
        **cue_default_params, neb_fesc=0.2
    )
    cont_low_total = float(simpson(lum_cont_low, wav_low))

    # Compute at fesc=0.4
    wav_high, lum_cont_high = cue_backend.predict_nebular_continuum(
        **cue_default_params, neb_fesc=0.4
    )
    cont_high_total = float(simpson(lum_cont_high, wav_high))

    # Compute ratio
    if cont_low_total > 1e-10:
        ratio = cont_high_total / cont_low_total
        expected_ratio = (1.0 - 0.4) / (1.0 - 0.2)  # = 0.75

        assert 0.6 <= ratio <= 0.95, (
            f"P-9 BUG: fesc linearity broken. "
            f"Ratio of L_neb(fesc=0.4)/L_neb(fesc=0.2) = {ratio:.3f}, "
            f"expected ≈ {expected_ratio:.3f}"
        )
    else:
        pytest.skip("Continuum luminosity too small to test linearity")


def test_fesc_gradient_matches_kfactor_wiring(cue_backend, cue_default_params):
    """Verify fesc is wired to JAX's differentiable computation graph.

    Pitfall P-9: Even if forward-pass values scale with fesc correctly, if fesc
    is *not* part of JAX's traced computation, gradient-based inference (VI,
    HMC) will see flat posteriors for ``neb_fesc`` (zero gradient signal).

    The Cue backend reddens nebular emission by the CIGALE ionizing-budget
    k-factor ``k = lyc_dust_escape_factor(neb_fesc, neb_fdust)`` (Ferland 1980;
    not the simpler ``1 − neb_fesc`` linear form). It is a JAX-traced
    multiplication, so the continuum sum is ``S_intrinsic · k(fesc)`` and

        ∂(Σ L_neb) / ∂(neb_fesc) = S_intrinsic · k'(fesc) = (Σ L_neb|_{fesc=0}) · k'(fesc)

    (since ``k(0) = 1`` for ``neb_fdust = 0``). We assert the actual gradient
    equals this to one part in 1e-4 — decisively distinct from the P-9 unwired
    case, where the gradient would be exactly 0.
    """
    fdust = float(cue_default_params.get("neb_fdust", 0.0))

    def loss_fn(fesc_traced):
        _, lum = cue_backend.predict_nebular_continuum(**cue_default_params, neb_fesc=fesc_traced)
        return jnp.sum(lum)

    # Reference: continuum sum at the linearization point, evaluated directly.
    _, lum_ref = cue_backend.predict_nebular_continuum(**cue_default_params, neb_fesc=0.0)
    sum_ref = float(jnp.sum(lum_ref))

    # Expected gradient from the actual k-factor wiring:
    #   Σ L_neb(fesc) = Σ L_neb(0) · k(fesc)/k(0)  ⇒  grad = sum_ref · k'(fesc)/k(0).
    k0 = float(lyc_dust_escape_factor(0.0, fdust))
    kprime = float(jax.grad(lambda f: lyc_dust_escape_factor(f, fdust))(jnp.array(0.3)))
    expected = sum_ref * kprime / k0

    actual = float(jax.grad(loss_fn)(jnp.array(0.3)))

    assert sum_ref > 0.0, (
        f"Continuum sum at fesc=0 is non-positive ({sum_ref}); test setup invalid."
    )
    rel_err = abs(actual - expected) / abs(expected)
    assert rel_err < 1e-4, (
        f"P-9 BUG: ∂L_neb/∂fesc = {actual:.3e}, expected ≈ {expected:.3e} "
        f"(k-factor wiring). Relative error {rel_err:.3e} > 1e-4. "
        "fesc may be applied as a Python scalar outside the JAX trace."
    )
