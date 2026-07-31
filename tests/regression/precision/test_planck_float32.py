# SPDX-License-Identifier: BSD-3-Clause
r"""The Planck function must not form :math:`\nu^3` (#1206).

.. math::

    B_\nu(T) = \frac{2 h \nu^3}{c^2} \frac{1}{\exp(h\nu / k_B T) - 1}

Written that way the intermediate :math:`\nu^3` reaches 2.7e49 on a UV-to-far-IR
grid — eleven decades past the float32 ceiling of 3.4e38 — even though
:math:`B_\nu` itself peaks at ~8e-12 and is perfectly representable.

Both implementations guarded this by casting to float64 first. Under
``jax.enable_x64(False)`` that cast **silently truncates back to float32** (JAX
emits a UserWarning and carries on), so the guard evaporates in exactly the
configuration Tier B targets, and 73% of the array goes non-finite.

Reassociating as :math:`2 h \nu (\nu/c)^2` never forms the cube: the largest
intermediate becomes :math:`(\nu/c)^2 \approx 1e12`. Algebraically identical,
and identical in float64 to machine epsilon.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

#: UV (100 A) to far-IR (1 mm) — the grid a panchromatic dust model runs on.
_WAVE_AA = np.logspace(2.0, 7.0, 1600)

#: Cold dust; the Wien tail genuinely underflows here, which is fine — those
#: values are ~1e-220 of the peak and contribute nothing.
_T_DUST = 35.0


def _dust_bnu(wave_aa, temperature):
    from tengri.components.dust.emission._physics import planck_bnu

    return planck_bnu(wave_aa, temperature)


def _agn_lnu(nu, temperature):
    from tengri.components.agn._phys import planck_lnu

    return planck_lnu(nu, temperature)


def test_dust_planck_is_finite_in_pure_float32():
    """``planck_bnu`` must be finite in float32 and match float64."""
    ref = np.asarray(_dust_bnu(jnp.asarray(_WAVE_AA), _T_DUST), dtype=np.float64)
    peak = float(np.abs(ref).max())
    assert np.all(np.isfinite(ref)), "setup: float64 Planck is not finite"
    assert 0.0 < peak < 3.4e38, f"setup: peak {peak:.3e} is not float32-representable"

    with jax.enable_x64(False):
        wave32 = jnp.asarray(_WAVE_AA, dtype=jnp.float32)
        assert wave32.dtype == jnp.float32  # precondition: genuinely pure float32
        got = np.asarray(_dust_bnu(wave32, _T_DUST))

    finite_fraction = float(np.isfinite(got).mean())
    assert finite_fraction == 1.0, (
        f"only {finite_fraction:.2%} of the float32 Planck function is finite — "
        "the nu**3 intermediate (2.7e49) overflowed float32"
    )
    error = float(np.abs(got.astype(np.float64) - ref).max() / peak)
    assert error < 1.0e-5, f"float32 Planck departs from float64 by {error:.3e} of peak"


def test_agn_planck_is_finite_in_pure_float32():
    """``planck_lnu`` carries the same nu**3 intermediate as the dust one."""
    nu = 2.998e18 / _WAVE_AA  # c [A/s] / lambda [A]
    ref = np.asarray(_agn_lnu(jnp.asarray(nu), 1.0e4), dtype=np.float64)
    peak = float(np.abs(ref).max())
    assert np.all(np.isfinite(ref)), "setup: float64 AGN Planck is not finite"
    assert 0.0 < peak < 3.4e38, f"setup: peak {peak:.3e} is not float32-representable"

    with jax.enable_x64(False):
        nu32 = jnp.asarray(nu, dtype=jnp.float32)
        got = np.asarray(_agn_lnu(nu32, 1.0e4))

    finite_fraction = float(np.isfinite(got).mean())
    assert finite_fraction == 1.0, (
        f"only {finite_fraction:.2%} of the float32 AGN Planck is finite — same "
        "nu**3 overflow as the dust implementation"
    )
    error = float(np.abs(got.astype(np.float64) - ref).max() / peak)
    assert error < 1.0e-5, f"float32 AGN Planck departs from float64 by {error:.3e} of peak"


@pytest.mark.parametrize("temperature", [10.0, 35.0, 100.0, 1000.0])
def test_dust_planck_float64_unchanged_across_temperatures(temperature):
    """Reassociating must not move the float64 answer anywhere.

    ``2 h nu**3 / c**2`` and ``2 h nu (nu/c)**2`` differ only in rounding, so
    the float64 result must stay put to ~machine epsilon at every temperature
    the model is used at.
    """
    got = np.asarray(_dust_bnu(jnp.asarray(_WAVE_AA), temperature), dtype=np.float64)

    # Independent reference, computed the textbook way in numpy float64.
    from tengri.components.dust.emission._physics import (
        _AA_TO_CM,
        _C_CGS,
        _H_PLANCK,
        _K_BOLTZMANN,
    )

    nu = _C_CGS / (_WAVE_AA * _AA_TO_CM)
    x = np.clip(_H_PLANCK * nu / (_K_BOLTZMANN * temperature), 1e-10, 500.0)
    expected = 2.0 * _H_PLANCK * nu**3 / _C_CGS**2 / np.expm1(x)

    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_dust_planck_gradient_is_finite_in_float32():
    """Gradients w.r.t. temperature stay finite under pure float32."""

    def total(temperature):
        return _dust_bnu(jnp.asarray(_WAVE_AA, dtype=jnp.float32), temperature).sum()

    with jax.enable_x64(False):
        grad = float(jax.grad(total)(jnp.float32(_T_DUST)))

    assert np.isfinite(grad), f"float32 Planck gradient is {grad}"


# ── The reverse pass under a caller's large cotangent (#1439) ─────────────
#
# A disc ring multiplies B_nu by its area, ~1e30. Autodiff's quotient rule on
# the primal's ``/(1 - e**-x)`` needs that denominator SQUARED, and with the
# cotangent arriving first the intermediate reaches 4.8e39 — past float32's
# 3.4e38 — for a true answer of 1.9e27. The two factors sit on opposite sides
# of the function boundary, so no rewrite of the expression reaches it; the
# derivative has to be stated. See ``utils/blackbody._planck_core_jvp``.

#: lambda = 1 mm, T = 1e6 K, cotangent 1e30 — the disc ring #1439 bisected to.
_RING_LAM_AA = 1e7
_RING_T = 1e6
_RING_AREA = 1e30


def _ring_sum(temperature):
    from tengri.utils.physics_constants import AA_TO_CM, C_CGS

    nu = jnp.asarray(C_CGS / (_RING_LAM_AA * AA_TO_CM), dtype=temperature.dtype)
    return jnp.sum(_RING_AREA * _agn_lnu(nu, temperature))


def test_planck_reverse_gradient_survives_a_large_cotangent_in_float32():
    """``d/dT`` through a ~1e30 ring area must match float64, not overflow."""
    with jax.enable_x64(True):
        want = float(jax.grad(_ring_sum)(jnp.float64(_RING_T)))
    assert np.isfinite(want) and want > 0.0, f"setup: float64 gradient is {want}"

    with jax.enable_x64(False):
        got = float(jax.grad(_ring_sum)(jnp.float32(_RING_T)))

    assert np.isfinite(got), (
        f"float32 reverse-mode Planck gradient is {got} under a {_RING_AREA:.0e} "
        "cotangent — the quotient rule re-formed the squared denominator (#1439)"
    )
    assert abs(got - want) / want < 1e-5, f"float32 {got:.6e} vs float64 {want:.6e}"


def test_planck_still_supports_forward_mode():
    """Forward mode must keep working — a ``custom_vjp`` here would raise.

    Not a formality. Forward mode computes this gradient *correctly* in pure
    float32 even where reverse mode overflowed, and geoVI and
    ``inference/preconditioning.py`` both differentiate forward. #1439
    prescribed a ``custom_vjp``, which is opaque to ``jvp`` — it would have
    turned a wrong-but-finite mode into a ``TypeError``. This is the same
    regression ``_mass_scale_lnu`` already had to undo, so it is guarded here
    rather than left to be rediscovered.
    """
    with jax.enable_x64(True):
        want = float(jax.jvp(_ring_sum, (jnp.float64(_RING_T),), (jnp.float64(1.0),))[1])

    with jax.enable_x64(False):
        # Would raise TypeError("can't apply forward-mode autodiff (jvp) to a
        # custom_vjp function") if the rule were ever respelled as a custom_vjp.
        got = float(jax.jvp(_ring_sum, (jnp.float32(_RING_T),), (jnp.float32(1.0),))[1])

    assert np.isfinite(got), f"float32 forward-mode Planck gradient is {got}"
    assert abs(got - want) / want < 1e-5, f"float32 {got:.6e} vs float64 {want:.6e}"


def test_planck_float64_gradient_is_unchanged_by_the_custom_rule():
    """The explicit rule must not move float64: value bit-identical, grad <= 1 ulp.

    Compared against autodiff of the *unmodified* expression, written out here
    so the comparison cannot drift with the implementation.
    """
    from tengri.utils.blackbody import _T_MIN, _X_MAX, _X_MIN
    from tengri.utils.physics_constants import C_CGS, H_PLANCK, K_BOLTZ

    def plain(nu, temperature):
        nu_w = jnp.asarray(nu, dtype=jnp.result_type(float))
        t_safe = jnp.maximum(jnp.asarray(temperature, dtype=jnp.result_type(float)), _T_MIN)
        x = jnp.clip((H_PLANCK / K_BOLTZ) * nu_w / t_safe, _X_MIN, _X_MAX)
        return 2.0 * H_PLANCK * nu_w * (nu_w / C_CGS) ** 2 * jnp.exp(-x) / -jnp.expm1(-x)

    nu = jnp.asarray(C_CGS / (_WAVE_AA * 1e-8))
    with jax.enable_x64(True):
        for temperature in (2.7, 35.0, 1500.0, 1e6):
            t = jnp.asarray(temperature)
            got_v = np.asarray(_agn_lnu(nu, t), dtype=np.float64)
            want_v = np.asarray(plain(nu, t), dtype=np.float64)
            assert np.array_equal(got_v, want_v), (
                f"float64 Planck value moved at T={temperature}: the custom rule "
                "must not touch the primal expression"
            )
            got_g = float(jax.grad(lambda tt: jnp.sum(_agn_lnu(nu, tt)))(t))
            want_g = float(jax.grad(lambda tt: jnp.sum(plain(nu, tt)))(t))
            assert abs(got_g - want_g) <= 4e-16 * abs(want_g), (
                f"float64 gradient moved at T={temperature}: {got_g!r} vs {want_g!r}"
            )
