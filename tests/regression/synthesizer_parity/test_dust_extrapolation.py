# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for dust attenuation curve extrapolation — synthesizer parity.

Mirrors the *shape* of synthesizer's ``tests/test_dust_attenuation.py`` assertions.
Synthesizer is GPL-3.0; we paraphrase the test structure but write our own
implementation. Every test cites the synthesizer source path it parallels and
the pitfall ID from ``~/.claude/plans/synthesizer-pitfall-catalog.md``.

These tests guard against extrapolation failures where attenuation curves are
evaluated outside their valid wavelength ranges, producing NaN, inf, or
unphysically steep/flat behavior. Pitfall P-5: Synthesizer PR #980 found that
Calzetti2000 extrapolation beyond 1 µm was invalid; fix used fill_value="extrapolate"
with hard wavelength caps. Tengri must avoid the same pitfall.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_paper

from tengri.components.dust.attenuation import (
    DUST_LAWS,
)
from tests._bounds import assert_non_negative
from tests._dust_laws import every_dust_law
from tests._jit_parity import assert_jit_matches_eager

# ---------------------------------------------------------------------------
# The law sweep
# ---------------------------------------------------------------------------
# The five property tests below each carried their own copy of the same
# hand-written list of law names. All five copies had 21 entries; ``DUST_LAWS``
# has 22, so ``reddy15`` had no finiteness, non-negativity, V-band
# normalization, far-IR or UV-slope coverage at all.
#
# The set is derived once in ``tests/_dust_laws.py`` — shared with
# tests/components/dust/test_dust_attenuation_laws.py, which had the same
# defect with a 20-name list — and guarded by
# tests/contract/test_dust_law_sweep_is_complete.py.

EVERY_DUST_LAW = every_dust_law()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wide_wavelength_grid() -> jnp.ndarray:
    """Wide rest-frame wavelength grid covering UV through MIR [Angstrom].

    Spans 100 Å (far-UV) to 100 µm (far-IR) = 1e6 Å, testing extrapolation
    well beyond typical grid limits (0.1-10 µm).
    """
    return jnp.logspace(2.0, 6.0, 500)  # 100 Å .. 100 µm


@pytest.fixture(scope="module")
def tau_v_reference() -> float:
    """Reference V-band optical depth for normalization tests.

    Most dust attenuation laws normalize to k(5500 Å) = 1, so tau(5500 A) = tau_V.
    """
    return 1.0


# ---------------------------------------------------------------------------
# P-5 / extrapolation parity
# ---------------------------------------------------------------------------
# Mirrors: synthesizer/tests/test_dust_attenuation.py
#   ::test_dust_curves_valid_beyond_grid_limits
# Pitfall: P-5 (dust attenuation curve extrapolation beyond grid limits)


class TestDustAttenuationFiniteness:
    """Ensure all dust laws produce finite, non-negative attenuation everywhere.

    Pitfall P-5 — Synthesizer PR #980 found that Calzetti2000 extrapolation
    beyond 1 µm produced NaN or invalid values. Tengri must guarantee all
    laws are finite (no NaN/inf) and non-negative (no negative transmission).
    """

    @pytest.mark.parametrize("law_name", EVERY_DUST_LAW)
    def test_attenuation_is_finite_everywhere(
        self, law_name, wide_wavelength_grid, tau_v_reference
    ):
        """Attenuation curve k(λ) must be finite (not NaN/inf) on wide grid.

        Mirrors: synthesizer/tests/test_dust_attenuation.py
        Pitfall: P-5 — Curves that fail beyond their design range produce NaN
        or inf, breaking likelihood evaluation.
        """
        law_func = DUST_LAWS[law_name].callable
        k = law_func(wide_wavelength_grid, **{})

        assert jnp.all(jnp.isfinite(k)), (
            f"Law {law_name!r} produced NaN or inf on wide grid: "
            f"finite count {jnp.sum(jnp.isfinite(k))}/{len(k)}"
        )

    @pytest.mark.parametrize("law_name", EVERY_DUST_LAW)
    def test_attenuation_is_nonnegative(self, law_name, wide_wavelength_grid, tau_v_reference):
        """Attenuation curve k(λ) ≥ 0 everywhere; transmission never exceeds 1.

        Mirrors: synthesizer/tests/test_dust_attenuation.py
        Pitfall: P-5 — Extrapolation can produce negative k(λ), violating
        exp(-tau) ≤ 1 and causing unphysical amplification.
        """
        law_func = DUST_LAWS[law_name].callable
        k = law_func(wide_wavelength_grid, **{})

        assert_non_negative(
            k,
            name="k",
            msg=f"Law {law_name!r} produced negative k(λ): "
            f"min={jnp.min(k):.6e}, count={jnp.sum(k < 0)}",
        )

    @pytest.mark.parametrize("law_name", EVERY_DUST_LAW)
    def test_v_band_normalization(self, law_name, tau_v_reference):
        """k(5500 Å) ≈ 1 (within 10%) — locks V-band normalization.

        Most dust laws normalize to k(V) = 1. Verify this normalization
        is correct; deviation >10% indicates a bug in the law implementation.

        Mirrors: synthesizer/tests/test_dust_attenuation.py
        Pitfall: P-5 — Broken normalization can cause amplitude miscalibration
        in the entire SED fitting pipeline.
        """
        law_func = DUST_LAWS[law_name].callable
        wave_v = jnp.array([5500.0])
        k_v = law_func(wave_v, **{})

        # Most laws normalize to k(5500 A) = 1, but allow some tolerance
        # for parametric laws or those with different definitions.
        rel_error = jnp.abs(k_v[0] - 1.0) / max(1.0, jnp.abs(k_v[0]))
        assert rel_error < 0.1, (
            f"Law {law_name!r} violates k(5500 A) ≈ 1: "
            f"got k(V)={k_v[0]:.6f} (rel error {rel_error:.1%})"
        )

    @pytest.mark.parametrize("law_name", EVERY_DUST_LAW)
    def test_far_ir_non_extrapolation(self, law_name, tau_v_reference):
        """k(λ ≥ 30 µm) ≤ 0.1 × k(V) — catches unphysical far-IR extrapolation.

        Dust attenuation decreases steeply into the IR. At 30+ µm (far-IR),
        attenuation should be much weaker than at V. This catches cases where
        extrapolation produces unphysically high or rising attenuation.

        Mirrors: synthesizer/tests/test_dust_attenuation.py
        Pitfall: P-5 — Broken grids or interpolation can cause attenuation to
        rise or plateau in the far-IR, violating physical expectations.
        """
        law_func = DUST_LAWS[law_name].callable
        wave_v = jnp.array([5500.0])
        wave_fir = jnp.array([300000.0])  # 30 µm = 3e5 Å

        k_v = law_func(wave_v, **{})[0]
        k_fir = law_func(wave_fir, **{})[0]

        # Far-IR should be much attenuated compared to V
        assert k_fir <= 0.1 * k_v, (
            f"Law {law_name!r} has unphysical far-IR: "
            f"k(30µm)={k_fir:.6f}, k(V)={k_v:.6f}; "
            f"ratio k(30µm)/k(V)={k_fir / k_v:.2f} > 0.1"
        )

    @pytest.mark.parametrize("law_name", EVERY_DUST_LAW)
    def test_uv_stronger_than_v(self, law_name):
        """k(1500 Å) > k(V) — UV attenuation exceeds V-band.

        All dust laws should show increased attenuation toward shorter
        wavelengths. UV (1500 Å) should be more attenuated than V (5500 Å).

        Mirrors: synthesizer/tests/test_dust_attenuation.py
        Pitfall: P-5 — Broken extrapolation or grid endpoints can invert
        the UV-to-V slope, producing unphysical reddening curves.
        """
        law_func = DUST_LAWS[law_name].callable
        wave_uv = jnp.array([1500.0])
        wave_v = jnp.array([5500.0])

        k_uv = law_func(wave_uv, **{})[0]
        k_v = law_func(wave_v, **{})[0]

        assert k_uv > k_v, (
            f"Law {law_name!r} has inverted UV slope: k(1500 A)={k_uv:.6f}, k(V)={k_v:.6f}"
        )


class TestDustAttenuationJitCompatibility:
    """Ensure dust laws remain JIT- and grad-compatible after extrapolation."""

    @pytest.mark.parametrize(
        "law_name",
        [
            "calzetti",
            "kriek_conroy",
            "cardelli",
            "smc",
            "salim",
        ],
    )
    def test_law_is_jit_compatible(self, law_name, wide_wavelength_grid):
        """Dust law must be JIT-compilable on wide wavelength grid.

        Mirrors: synthesizer's JIT-compilation smoke test.
        Pitfall: P-5 — If extrapolation uses non-JAX operations (e.g. scipy
        interp1d with mode='extrapolate'), JIT will fail.
        """
        law_func = DUST_LAWS[law_name].callable

        # Should compile and run without error
        k = assert_jit_matches_eager(law_func, wide_wavelength_grid)
        chex.assert_tree_all_finite(k)

    @pytest.mark.parametrize(
        "law_name",
        [
            "calzetti",
            "kriek_conroy",
            "cardelli",
            "smc",
            "salim",
        ],
    )
    def test_law_is_grad_compatible(self, law_name, wide_wavelength_grid):
        """Dust law gradient must be computable (no Tracer errors).

        Mirrors: synthesizer's automatic-differentiation smoke test.
        Pitfall: P-5 — Extrapolation with discontinuities or conditionals
        can break gradient flow, preventing inference.
        """
        law_func = DUST_LAWS[law_name].callable

        def loss_fn(wave):
            return jnp.sum(law_func(wave))

        # Should compute gradients without error
        grad_fn = jax.grad(loss_fn)
        grad_val = grad_fn(wide_wavelength_grid)
        chex.assert_tree_all_finite(grad_val)
