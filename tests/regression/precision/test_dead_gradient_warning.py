# SPDX-License-Identifier: BSD-3-Clause
r"""The nthcomp kernel differentiates what it is asked to (#1206, #1822).

``agn_kt_warm`` is declared ``Uniform(0.1, 0.5)`` by the KD18 disc and reaches
the SED only through ``_nthcomp_lnu_interp``. Two defects lived there, and this
file pins the fix for both.

**Dead (#1822, part 1).** The ``custom_jvp`` supplied a ``gamma`` tangent and
discarded the ``kTe`` one, so ``d/d(agn_kt_warm)`` was exactly ``0.0`` against a
central difference of ~7e41. The consequence was not a wrong number but an
*unfalsifiable* one: every gradient backend left the parameter at its initial
value, so the posterior was the prior — which is exactly what an
unconstrained-but-honestly-fitted parameter also looks like. This file used to
assert the zero and warn about it; the rule now supplies the tangent.

**NaN (#1822, part 2).** The tangent believed to be working was worse. The
kernel returned float32 regardless of the caller's precision, so reverse mode
took the cotangent in float32 — and ``disc.py`` multiplies this kernel's output
by a ring luminosity, making that cotangent ~1e66. Representable in float64,
``inf`` in float32 (ceiling 3.4e38), and ``inf * fd_grad`` is NaN. Measured on
``kubota_done_disc``: ``jax.jvp`` gave 5.2e30 for ``d/d(agn_gamma_warm)`` while
``jax.grad`` gave **NaN** — and every gradient backend (MAP, NUTS, VI) is
reverse-mode. The rule's docstring had argued the old ``custom_vjp``'s overflow
rescaling was unnecessary because "forward mode never forms the cotangent
product"; true, and beside the point, because ``jax.grad`` transposes the jvp
and forms exactly that product.

**Why the existing anti-laziness suite cannot catch either.**
``tests/crossval/test_anti_laziness.py::test_kd_all_unique_params_matter``
asserts that ``agn_kt_warm`` changes the SED, and passes — the *forward*
sensitivity was always real (18.1x in ``sum(L_nu)`` across the declared prior).
It is the *derivative* that was missing. A parameter can be forward-live and
gradient-dead at the same time, and a suite that varies values can never
distinguish them, which is why this file differentiates instead.

**On finite-difference references.** A central difference is a measurement with
its own error budget, and here it is a poor ruler over much of the range: the
kernel interpolates in float32, so the disc total (~8e47) carries ~1e-7 relative
noise, and dividing that by ``2h`` puts the reference's noise floor near 4e43 at
``h=1e-3``. At ``kt_warm=0.25`` the true derivative is ~1e42 — *below its own
reference's noise floor*, and the "central difference" there swings over three
orders of magnitude and changes sign as ``h`` varies. The probes below are
therefore placed where the derivative is large enough to be resolved, and the
disc-level check asserts stability across several ``h`` rather than trusting one.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.config.exceptions import DeadGradientParameterWarning

pytestmark = pytest.mark.regression_bug

_SFH = {
    "type": "delayed",
    "all_params": Fixed(DEFAULT),
    "log_total_mass": Uniform(9.0, 11.0),
    "tau_gyr": 1.0,
    "age_gyr": 5.0,
}


def _build(ssp, *, free_kt_warm):
    """A KD18-disc AGN model, with ``kt_warm`` freed or pinned."""
    disc = {"type": "kubota_done", "all_params": Fixed(DEFAULT)}
    disc["kt_warm"] = Uniform(0.1, 0.5) if free_kt_warm else Fixed(0.2)
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(["sdss_r", "wise_w1"])),
        redshift=Fixed(0.1),
        sfh=_SFH,
        agn={
            "type": "composable",
            "all_params": Fixed(DEFAULT),
            "disc": disc,
            "log_lbol": Uniform(9.0, 12.0),
        },
    )


# ── 1. The kernel's kTe tangent ───────────────────────────────────


def test_kte_tangent_is_supplied_and_correct():
    """``d/d(kTe)`` is non-zero and matches a central difference.

    Asserting agreement, not merely non-zero: a rule that returned any
    non-zero constant would satisfy "not dead" while still being wrong, and
    that is the shape of failure a finiteness assertion also misses.
    """
    from tengri.components.agn._nthcomp import nthcomp_lnu_interp

    nu = jnp.asarray(np.logspace(14.5, 18.5, 200))

    def total(kte):
        return jnp.sum(nthcomp_lnu_interp(nu, jnp.asarray(2.37), kte, jnp.asarray(0.05)))

    x = jnp.asarray(0.2)
    _, tangent = jax.jvp(total, (x,), (jnp.asarray(1.0),))
    h = 1e-4
    central = float((total(x + h) - total(x - h)) / (2 * h))

    assert central != 0.0, "setup: the reference is zero, so it cannot judge the tangent"
    assert np.all(np.isfinite(central)), (
        "`central` is non-finite — non-zero is not enough, `nan != 0.0` is True "
        "and a NaN satisfies a non-zero assertion (#2178)"
    )
    assert float(tangent) != 0.0, (
        "d/d(kTe) is exactly zero — the nthcomp custom_jvp has stopped supplying "
        "the kTe tangent, which makes agn_kt_warm unfittable again (#1822)"
    )
    assert np.all(np.isfinite(float(tangent))), (
        "`float(tangent)` is non-finite — non-zero is not enough, `nan != 0.0` is True "
        "and a NaN satisfies a non-zero assertion (#2178)"
    )
    rel = abs(float(tangent) - central) / abs(central)
    assert rel < 0.05, f"d/d(kTe) = {float(tangent):.5e} vs central {central:.5e} ({rel:.1%})"


def test_forward_and_reverse_agree_on_the_kernel():
    """Both modes, because the NaN of #1822 appeared in only one of them.

    ``jax.jvp`` was correct while ``jax.grad`` returned NaN, so a suite that
    exercised forward mode alone reported a healthy kernel.
    """
    from tengri.components.agn._nthcomp import nthcomp_lnu_interp

    nu = jnp.asarray(np.logspace(14.5, 18.5, 200))

    def total(kte):
        return jnp.sum(nthcomp_lnu_interp(nu, jnp.asarray(2.37), kte, jnp.asarray(0.05)))

    x = jnp.asarray(0.2)
    _, fwd = jax.jvp(total, (x,), (jnp.asarray(1.0),))
    rev = jax.grad(total)(x)
    assert jnp.isfinite(rev), f"reverse-mode d/d(kTe) is {rev}"
    assert jnp.any(rev != 0.0), (
        "`rev` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
    np.testing.assert_allclose(float(rev), float(fwd), rtol=1e-10)


# ── 2. The reverse-mode cotangent overflow ────────────────────────


@pytest.mark.parametrize("param", ["agn_gamma_warm", "agn_kt_warm"])
def test_reverse_mode_survives_a_realistic_ring_luminosity(param):
    """The regression proper: ``jax.grad`` through the disc must be finite.

    ``disc.py`` multiplies the kernel's ~1e-19 shape by a ring luminosity, so
    the cotangent reverse mode hands back is ~1e66. While the kernel returned
    float32 that overflowed to ``inf`` and the gradient came back NaN — for
    ``agn_gamma_warm`` too, which nothing had flagged, because a NaN gradient is
    not a *dead* one and ``_DEAD_GRADIENT_PARAMS`` only tracked zeros.
    """
    from tengri.components.agn.disc import kubota_done_disc

    wave = jnp.geomspace(1.0, 1.0e5, 300)
    x0 = {"agn_gamma_warm": 2.5, "agn_kt_warm": 0.25}[param]

    def total(v):
        kw = {"agn_log_lbol": jnp.asarray(45.0), "agn_log_mbh": jnp.asarray(8.0), param: v}
        return jnp.sum(kubota_done_disc(wave, **kw))

    x = jnp.asarray(x0)
    grad = jax.grad(total)(x)
    assert jnp.isfinite(grad), (
        f"reverse-mode d/d({param}) through kubota_done_disc is {grad}. The nthcomp "
        "kernel is forcing a float32 output again, so the ~1e66 cotangent from the "
        "ring luminosity overflows float32's 3.4e38 ceiling (#1822)."
    )
    assert jnp.any(grad != 0.0), (
        "`grad` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
    _, fwd = jax.jvp(total, (x,), (jnp.asarray(1.0),))
    np.testing.assert_allclose(float(grad), float(fwd), rtol=1e-8)


def test_kernel_returns_the_callers_precision():
    """The mechanism, pinned directly.

    The table is float32 and is interpolated there on purpose. Returning
    float32 is what forced the caller's precision and broke reverse mode.
    """
    from tengri.components.agn._nthcomp import nthcomp_lnu_interp

    nu64 = jnp.asarray(np.logspace(14.5, 18.5, 64), dtype=jnp.float64)
    out64 = nthcomp_lnu_interp(nu64, jnp.asarray(2.37), jnp.asarray(0.2), jnp.asarray(0.05))
    assert out64.dtype == jnp.float64, (
        f"float64 inputs produced {out64.dtype} — the kernel is forcing the table's "
        "precision on the caller, which is the #1822 overflow"
    )

    nu32 = nu64.astype(jnp.float32)
    out32 = nthcomp_lnu_interp(
        nu32,
        jnp.asarray(2.37, jnp.float32),
        jnp.asarray(0.2, jnp.float32),
        jnp.asarray(0.05, jnp.float32),
    )
    assert out32.dtype == jnp.float32, "an all-float32 caller must still get float32 back"
    np.testing.assert_allclose(np.asarray(out64), np.asarray(out32), rtol=1e-6)


# ── 3. The warning must not outlive the defect ────────────────────


def test_freeing_kt_warm_no_longer_warns(ssp_bare):
    """The gradient is live, so announcing it as dead would now be the lie.

    This is the inverse of the assertion this file shipped before #1822.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeadGradientParameterWarning)
        _build(ssp_bare, free_kt_warm=True)


def test_pinning_kt_warm_does_not_warn(ssp_bare):
    """A pinned parameter is a normal, correct configuration."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeadGradientParameterWarning)
        _build(ssp_bare, free_kt_warm=False)


def test_the_dead_gradient_machinery_still_works():
    """Emptying the registry must not silently disable the check.

    ``_DEAD_GRADIENT_PARAMS`` is empty since #1822. Keeping the mechanism is
    deliberate — the class recurs — so assert it still fires for a name placed
    in it, rather than leaving a guard that cannot fail.
    """
    from tengri.forward import sed_model as sm

    assert sm._DEAD_GRADIENT_PARAMS == {}, (
        "a parameter was added back to _DEAD_GRADIENT_PARAMS; update this test "
        "and say which parameter and why in the PR"
    )

    class _Spec:
        free_params = ("made_up_param",)

    original = dict(sm._DEAD_GRADIENT_PARAMS)
    sm._DEAD_GRADIENT_PARAMS["made_up_param"] = "a test reason"
    try:
        with pytest.warns(DeadGradientParameterWarning, match="made_up_param"):
            sm._warn_dead_gradient_params(_Spec())
    finally:
        sm._DEAD_GRADIENT_PARAMS.clear()
        sm._DEAD_GRADIENT_PARAMS.update(original)
