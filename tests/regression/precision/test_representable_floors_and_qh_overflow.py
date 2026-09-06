# SPDX-License-Identifier: BSD-3-Clause
"""Guards that must not evaporate in float32 (#1491, #1492).

Two defects with one shape: a protection written as a literal, which the
float32 dtype silently removes.

* **#1492** — float32's smallest subnormal is 1.4e-45, so a floor literal below
  that is *exactly* ``0.0`` there. ``jnp.maximum(x, 1e-50)`` reads as a guard
  and is ``jnp.maximum(x, 0.0)``. Perversely the smaller the literal the worse:
  ``1e-30`` survives float32, ``1e-100`` does not, and the latter looks more
  careful.
* **#1491** — ``where(isfinite(qh), qh, 0.0)`` is correct for an SSP grid with
  incomplete UV coverage, where non-finite honestly means "no ionizing flux".
  It is wrong when *healthy* input overflows: Q_H reaches ~1e47 photons/s and
  float32 tops out at 3.4e38, so the guard rewrote a real ionizing budget to
  zero for 861 of 1395 grid points — no nebular emission, silently.

Both fixes are float64-preserving by construction, and that is asserted here
rather than assumed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular._shared import QHTableOverflowError, sanitize_qh_table
from tengri.utils import sed_quantities as SQ
from tengri.utils.scale import representable_floor

pytestmark = pytest.mark.regression_bug


# ── #1492: floors must be representable in the working dtype ──────────


def test_literals_below_the_subnormal_floor_are_no_ops_in_float32():
    """The premise. If this fails, float32 grew a wider range and #1492 is moot."""
    with jax.enable_x64(False):
        for literal in (1e-50, 1e-60, 1e-100, 1e-300):
            assert float(jnp.asarray(literal, jnp.float32)) == 0.0, (
                f"{literal:g} is no longer flushed to zero in float32"
            )


def test_representable_floor_leaves_float64_untouched():
    """float64's tiny is 2.2e-308, below every floor the tree uses."""
    with jax.enable_x64(True):
        for literal in (1e-50, 1e-60, 1e-100, 1e-300):
            assert representable_floor(literal) == literal


def test_representable_floor_lifts_into_range_in_float32():
    with jax.enable_x64(False):
        floor = representable_floor(1e-50)
    assert floor == pytest.approx(float(np.finfo(np.float32).tiny))
    assert floor > 0.0
    with jax.enable_x64(False):
        assert float(jnp.asarray(floor, jnp.float32)) > 0.0, "the lifted floor must survive"


def _beta(sed, wave, x64):
    with jax.enable_x64(x64):
        dt = jnp.result_type(float)
        return float(SQ.compute_uv_slope_beta(jnp.asarray(sed, dt), jnp.asarray(wave, dt)))


def test_uv_slope_survives_a_zeroed_bin_in_float32():
    """The demonstrated failure: log(0) = -inf poisoned the regression.

    float64 degrades into a finite, obviously-wrong number a caller can catch.
    float32 returned NaN, because the floor guarding the ``log`` was not there.
    """
    wave = np.linspace(1200.0, 2700.0, 60)
    sed = np.full_like(wave, 1e-20)
    sed[10:14] = 0.0

    assert np.isfinite(_beta(sed, wave, x64=True)), "float64 was always finite here"
    assert np.any(_beta(sed, wave, x64=True) != 0.0), (
        "`_beta(sed, wave, x64=True)` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
    assert np.isfinite(_beta(sed, wave, x64=False)), (
        "float32 UV slope is non-finite — the 1e-50 floor is inert at this precision"
    )


def test_uv_slope_float64_is_unchanged():
    """Reference values recorded before the fix; the floor must not move float64."""
    wave = np.linspace(1200.0, 2700.0, 60)
    healthy = np.full_like(wave, 1e-20)
    zeroed = healthy.copy()
    zeroed[10:14] = 0.0

    assert _beta(healthy, wave, x64=True) == pytest.approx(-2.0, abs=1e-9)
    assert _beta(zeroed, wave, x64=True) == pytest.approx(24.3677, rel=1e-4)


# ── #1491: overflow must be loud, bad input must stay graceful ────────


def test_qh_overflow_raises_instead_of_zeroing():
    """A table whose survivors sit against the ceiling is overflow, and must raise.

    Built synthetically rather than from an SSP grid on purpose. An earlier
    version computed Q_H from ``ssp_bare`` and asserted it overflowed in
    float32; that passed against the real ``fsps_prsc_miles_chabrier.h5``
    (861/1395 entries overflow) and failed in CI, which carries the *synthetic*
    SSP fixture from #613 whose fluxes are small enough not to overflow at all.

    The assertion belongs on the sanitizer's rule, not on how large one data
    file's fluxes happen to be — the same error as pinning machine-dependent
    reference values instead of the invariant.
    """
    with jax.enable_x64(False):
        # inf entries, and survivors pressed against float32's 3.4e38 ceiling:
        # the signature of dtype overflow rather than missing UV coverage.
        qh = jnp.asarray(
            np.array([[1.0e38, np.inf, 2.0e38], [np.inf, 1.5e38, np.inf]], dtype=np.float32)
        )
        with pytest.raises(QHTableOverflowError, match="overflowed"):
            sanitize_qh_table(qh, backend_name="CloudyGridBackend")


def test_qh_float64_is_unchanged(ssp_bare):
    """float64 has no non-finite entries at all, so nothing may change.

    Data-independent: any SSP grid, real or the synthetic fixture, integrates to
    a finite Q_H in float64. The assertion is pass-through, not a magnitude.
    """
    from tengri.components.nebular.cloudy_grid import _compute_qh_grid

    with jax.enable_x64(True):
        qh = _compute_qh_grid(jnp.asarray(ssp_bare.ssp_wave), jnp.asarray(ssp_bare.ssp_flux))
        assert jnp.all(jnp.isfinite(qh)), "float64 Q_H must be finite for any sane SSP"
        assert jnp.any(qh != 0.0), (
            "`qh` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )
        out = sanitize_qh_table(qh, backend_name="CloudyGridBackend")
        assert jnp.array_equal(out, qh), "float64 Q_H table must pass through untouched"


def test_a_genuinely_patchy_grid_still_degrades_to_zero():
    """The guard's original purpose must survive.

    A grid with incomplete UV coverage loses a few bins while its survivors sit
    far below the dtype ceiling. That is not overflow and must not raise —
    zero is the honest answer there.
    """
    with jax.enable_x64(True):
        patchy = jnp.asarray(np.array([[1e47, np.inf, 1e46], [1e45, 1e44, np.nan]]))
        out = sanitize_qh_table(patchy, backend_name="X")

    assert int(jnp.sum(out == 0)) == 2
    assert jnp.all(jnp.isfinite(out))
