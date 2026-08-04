# SPDX-License-Identifier: BSD-3-Clause
r"""A range-safety grouping is not binding on XLA — only a data dependency is (#1535).

:func:`~tengri.inference.likelihoods.gaussian.diag_gaussian_chi2` forms
``r = (d - mu) / sigma`` and squares *that*, rather than the algebraically equal
``(d - mu)**2 / sigma**2``, because ``1/sigma**2`` overflows float32 for the
``sigma`` ~ 1e-31 of a real photometric error. The source says so.

Under ``jax.jit``, when ``d`` and ``sigma`` arrive as **closure constants**, XLA
re-associates it back and constant-folds the reciprocal::

    %multiply.4  = multiply(%sub.0, %sub.0)          # (d-mu)^2 -> 0 in f32
    %constant.32 = f32[] constant(inf)               # 1/sigma^2 folded
    %mul.0       = multiply(%multiply.4, %broadcast) # 0 * inf = NaN

So the mitigation holds eagerly, holds in float64, and holds when the data are
traced — and fails in exactly one combination, which is the one a user writes
first when they wrap a likelihood for a sampler.

**Why this file exists rather than a fix.** tengri's own inference path is
immune: the compiled callable is ``val_and_grad(params_u, data_args)`` and takes
the data as a traced argument. So there is no live bug to fix here — there is a
trap to pin. These tests pin both halves: that the real path stays immune (a
regression in *that* would silently NaN every float32 fit), and that the closure
form is still broken, so whoever fixes #1535 is told where the record is.

The general rule, which is the reason this is a test and not a comment: a
numerical mitigation expressed as an association *order in source* is not binding
on the compiler. Any guard of the form "group the operations this way for range
safety" needs a jit arm, and the assertion belongs in the compiled HLO rather
than in the Python.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.likelihoods.gaussian import diag_gaussian_chi2

pytestmark = pytest.mark.regression_bug

#: Real photometric scale. ``sigma`` ~ 1e-31 puts ``1/sigma**2`` at ~1e62,
#: twenty-four decades past the float32 ceiling of 3.4e38.
_DATA = np.array([1.0e-30, 2.0e-30])
_SIGMA = np.array([1.0e-31, 2.0e-31])


def _f32_inputs():
    d = jnp.asarray(_DATA, dtype=jnp.float32)
    s = jnp.asarray(_SIGMA, dtype=jnp.float32)
    return d, s, d * jnp.asarray(1.01, dtype=jnp.float32)


def test_setup_the_reciprocal_really_is_out_of_range():
    """Guard the guard: if 1/sigma**2 fitted in float32 this file would be vacuous."""
    reciprocal = 1.0 / float(_SIGMA[0]) ** 2
    assert reciprocal > 3.4e38, (
        f"1/sigma**2 = {reciprocal:.3e} is inside the float32 window, so nothing here "
        "is testing an overflow — pick a smaller sigma"
    )


def test_the_traced_argument_form_is_finite_in_float32():
    """The path tengri actually uses. A regression here NaNs every float32 fit.

    ``InferenceContext.neg_log_posterior_fn`` returns
    ``val_and_grad(params_u, data_args)`` — data as a traced argument — which is
    what keeps the grouping intact through the compiler.
    """
    with jax.enable_x64(False):
        d, s, mu = _f32_inputs()
        assert d.dtype == jnp.float32, "precondition: genuinely pure float32"
        chi2 = float(jax.jit(diag_gaussian_chi2)(mu, d, s))
    assert np.isfinite(chi2), (
        f"chi2 is {chi2} with the data passed as traced arguments. This is the form the "
        "inference path uses, so if it has stopped being immune every pure-float32 fit "
        "now returns NaN (#1535)"
    )


def test_eager_float32_is_finite():
    """The mitigation does work — the compiler is what undoes it."""
    with jax.enable_x64(False):
        d, s, mu = _f32_inputs()
        assert np.isfinite(float(diag_gaussian_chi2(mu, d, s)))


def test_float64_is_immune_either_way():
    """Isolates the cause to precision, not to jit.

    Without this arm, "jit breaks it" would be an equally good explanation, and
    the fix would be aimed at the wrong thing.
    """
    d, s = jnp.asarray(_DATA), jnp.asarray(_SIGMA)
    mu = d * 1.01
    closed = float(jax.jit(lambda m: diag_gaussian_chi2(m, d, s))(mu))
    assert np.isfinite(closed), f"float64 closure form gave {closed}; it should be immune"


def test_the_closure_form_is_still_broken():
    """Pins the defect so #1535 landing is visible rather than silent.

    Asserts the *undesirable* behavior deliberately. When this fails, the trap is
    gone and this file should be rewritten to assert finiteness instead.
    """
    with jax.enable_x64(False):
        d, s, mu = _f32_inputs()
        chi2 = float(jax.jit(lambda m: diag_gaussian_chi2(m, d, s))(mu))
    assert np.isnan(chi2), (
        f"the float32 closure form now gives {chi2} rather than NaN. If #1535 was fixed, "
        "flip this file to assert finiteness and drop the warning on "
        "InferenceContext.neg_log_posterior_fn"
    )


def _hlo_op_count(compiled_text, op):
    return sum(f" {op}(" in line for line in compiled_text.splitlines())


def test_the_division_survives_only_in_the_traced_form():
    """The assertion belongs in the HLO, not in the Python — that is the lesson.

    The Python always writes ``(d - mu) / sigma``. Whether the *compiled module*
    still divides is the only thing that decides range safety, and a test reading
    values alone cannot distinguish "the grouping held" from "the grouping was
    rewritten and happened not to overflow on this input".

    **The signal is the divide count, not ``constant(inf)``.** A first version of
    this test grepped for a folded infinity and failed, because the traced form
    contains one too — a benign internal of ``hypot`` (visible in its metadata,
    consumed by an equality compare) rather than a folded reciprocal. Measured:

    ======================  ==========  ==========
    form                    ``divide``  result
    ======================  ==========  ==========
    closure constants                0  ``NaN``
    traced arguments                 2  0.0200
    ======================  ==========  ==========

    Zero divides is the grouping being destroyed: ``(d-mu)/sigma`` became
    ``(d-mu) * constant``, with the constant folded to ``inf``.
    """
    with jax.enable_x64(False):
        d, s, mu = _f32_inputs()
        closed_hlo = jax.jit(lambda m: diag_gaussian_chi2(m, d, s)).lower(mu).compile().as_text()
        traced_hlo = jax.jit(diag_gaussian_chi2).lower(mu, d, s).compile().as_text()

    assert _hlo_op_count(traced_hlo, "divide") > 0, (
        "the traced-argument form no longer divides by sigma in the compiled module — "
        "XLA has folded the reciprocal on the path the inference code actually uses, so "
        "every pure-float32 fit is now NaN (#1535)"
    )
    assert _hlo_op_count(closed_hlo, "divide") == 0, (
        "the closure form now keeps its division, so the constant folding this file "
        "documents no longer happens. If #1535 was fixed upstream, rewrite this file to "
        "assert both forms are finite and drop the warning on "
        "InferenceContext.neg_log_posterior_fn"
    )
