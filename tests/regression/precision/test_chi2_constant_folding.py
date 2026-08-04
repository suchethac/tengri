# SPDX-License-Identifier: BSD-3-Clause
r"""A range-safety grouping must be a data dependency, not a source order (#1535).

Every χ² in tengri divides before squaring, because at real photometric scales
:math:`\sigma \sim 10^{-31}` both :math:`\sigma^2` and :math:`(d-\mu)^2` underflow
float32 to zero while the ratio is O(1). Writing it in that order turned out not
to be enough.

Under ``jax.jit``, when the data are compile-time constants, XLA re-associated
:math:`((d-\mu)/\sigma)^2` back into :math:`(d-\mu)^2 \cdot (1/\sigma^2)` and
constant-folded the reciprocal::

    %multiply.4  = multiply(%sub.0, %sub.0)          # (d-mu)^2 -> 0 in f32
    %constant.32 = f32[] constant(inf)               # 1/sigma^2 folded
    %mul.0       = multiply(%multiply.4, %broadcast) # 0 * inf = NaN

The mitigation held eagerly, held in float64, and held when the data were traced
— and failed in exactly one combination, which is the one a user writes first
when wrapping a likelihood for a sampler.

:func:`~tengri.inference.likelihoods.gaussian.standardized_residual` fixes it
with ``jax.lax.optimization_barrier``, which makes the intended order a data
dependency the compiler must respect. It is semantically the identity: float64
values and gradients are bit-exact against the pre-fix expression, and the cost
is -1.6% against a 2.2% A/A noise floor, i.e. unmeasurable.

**Six sites, not one.** ``diag_gaussian_chi2`` was the one that was reported, but
the same grouping appeared open-coded in five more — line fluxes, line ratios and
spectral indices in ``loss_functions``, the NIFTy Hamiltonian in ``jit_engine``,
and the population χ² in ``hierarchical``. All were measured to NaN identically,
and all now route through the helper.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.likelihoods.gaussian import diag_gaussian_chi2, standardized_residual

pytestmark = pytest.mark.regression_bug

#: Real photometric scale. ``sigma`` ~ 1e-31 puts ``1/sigma**2`` at ~1e62,
#: twenty-four decades past the float32 ceiling of 3.4e38.
_DATA = np.array([1.0e-30, 2.0e-30])
_SIGMA = np.array([1.0e-31, 2.0e-31])
_EXPECTED = 0.0199999  # (0.01 d / 0.1 d)^2 summed over two bands


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


@pytest.mark.parametrize("form", ["eager", "jit_closure", "jit_traced"])
def test_chi2_is_finite_in_pure_float32_however_it_is_called(form):
    """The fix. ``jit_closure`` is the arm that returned NaN before #1535.

    Parametrized rather than split so the three are visibly one contract: how the
    data reach the function must not change the answer.
    """
    with jax.enable_x64(False):
        d, s, mu = _f32_inputs()
        assert d.dtype == jnp.float32, "precondition: genuinely pure float32"
        if form == "eager":
            chi2 = float(diag_gaussian_chi2(mu, d, s))
        elif form == "jit_closure":
            chi2 = float(jax.jit(lambda m: diag_gaussian_chi2(m, d, s))(mu))
        else:
            chi2 = float(jax.jit(diag_gaussian_chi2)(mu, d, s))

    assert np.isfinite(chi2), (
        f"chi2 is {chi2} for the {form!r} form. If this is jit_closure, the "
        "optimization_barrier in standardized_residual has stopped binding and XLA is "
        "folding 1/sigma**2 to inf again (#1535)"
    )
    assert chi2 == pytest.approx(_EXPECTED, rel=1e-4), (
        f"{form!r} gives {chi2}, not the expected {_EXPECTED} — finite but wrong is a "
        "worse failure than NaN"
    )


def test_the_barrier_is_semantically_the_identity_in_float64():
    """It must fix a compiler artifact without moving a single float64 bit.

    Bit-exact, not merely close: ``optimization_barrier`` reorders nothing, so
    any difference at all would mean it is doing something other than advertised.
    """
    d, s = jnp.asarray(_DATA), jnp.asarray(_SIGMA)
    mu = d * 1.01
    sigma_eff = jnp.hypot(s, 0.0 * d)

    with_barrier = float(jnp.sum(standardized_residual(d, mu, sigma_eff) ** 2))
    without = float(jnp.sum(((d - mu) / sigma_eff) ** 2))
    assert with_barrier == without, f"float64 moved: {with_barrier!r} vs {without!r}"

    grad_with = float(
        jax.grad(lambda m: jnp.sum(standardized_residual(d, m, sigma_eff) ** 2))(mu)[0]
    )
    grad_without = float(jax.grad(lambda m: jnp.sum(((d - m) / sigma_eff) ** 2))(mu)[0])
    assert grad_with == grad_without, f"gradient moved: {grad_with!r} vs {grad_without!r}"


def _folded_infinities(compiled_text):
    """Count ``constant(inf)`` literals that carry no metadata.

    Metadata-bearing ones are legitimate: ``hypot``'s internals emit an ``inf``
    constant and compare against it. A *bare* one is a folded reciprocal — the
    artifact #1535 is about.
    """
    return sum(
        "constant(inf)" in line and "metadata" not in line for line in compiled_text.splitlines()
    )


def test_the_folded_reciprocal_is_gone_from_the_compiled_module():
    """The assertion belongs in the HLO — that is the whole lesson of #1535.

    The Python always wrote ``(d - mu) / sigma``; only the compiled module ever
    disagreed. A test reading values alone cannot distinguish "the grouping held"
    from "the grouping was rewritten and happened not to overflow on this input",
    and a test reading the source cannot see the problem at all.

    Measured discriminator, fixed vs an open-coded control:

    ==========================  ==========  =========================
    form                        ``divide``  bare ``constant(inf)``
    ==========================  ==========  =========================
    via ``standardized_residual``        0  **0**
    open-coded                           0  **1**
    ==========================  ==========  =========================

    Note ``divide`` is zero in *both* compiled modules — the division is fused
    away regardless, so it does not discriminate here even though it does in the
    lowered form. The barrier itself is also absent by name: XLA consumes it
    during optimization. Only the folded constant distinguishes them, and only
    once the metadata-bearing ``hypot`` infinities are excluded.
    """
    with jax.enable_x64(False):
        d, s, mu = _f32_inputs()
        closed = jax.jit(lambda m: diag_gaussian_chi2(m, d, s))
        hlo = closed.lower(mu).compile().as_text()
        value = float(closed(mu))

    assert np.isfinite(value), "precondition: the closure form must be fixed first"
    assert _folded_infinities(hlo) == 0, (
        "the compiled closure-form module still contains a bare constant(inf) — XLA is "
        "folding 1/sigma**2 again, so standardized_residual's barrier is not binding "
        "(#1535). The value may still look finite on this input; it will not on others"
    )


@pytest.mark.parametrize(
    "site",
    [
        "loss_functions.line_fluxes",
        "loss_functions.line_ratios",
        "loss_functions.indices",
        "jit_engine.hamiltonian",
        "hierarchical.population_chi2",
    ],
)
def test_every_open_coded_chi2_site_was_converted(site):
    """The generalization: five more sites had the identical grouping.

    Measured vulnerable before the fix — a bare ``((d - p) / n) ** 2`` under a
    closure jit returns NaN in float32 exactly as ``diag_gaussian_chi2`` did.
    This asserts each source site now goes through the helper rather than
    re-deriving, because the next person to open-code it gets the bug back.
    """
    import inspect

    module_name, _ = site.split(".", 1)
    module = {
        "loss_functions": "tengri.inference.loss_functions",
        "jit_engine": "tengri.inference.jit_engine",
        "hierarchical": "tengri.inference.hierarchical",
    }[module_name]
    source = inspect.getsource(__import__(module, fromlist=["_"]))

    assert "standardized_residual" in source, (
        f"{module} no longer imports standardized_residual — its chi2 site has been "
        "open-coded again and is NaN in pure float32 under a closure jit (#1535)"
    )
    assert ") / noise) ** 2" not in source and ") / all_noise) ** 2" not in source, (
        f"{module} contains an open-coded divide-then-square chi2 again"
    )


def test_a_bare_open_coded_chi2_still_demonstrates_the_hazard():
    """Negative control: the bug is real and the helper is what avoids it.

    Without this, every assertion above would pass on a build where XLA simply
    stopped constant-folding, and the helper would look load-bearing when it was
    not. Pins the hazard the helper exists for.
    """
    with jax.enable_x64(False):
        d, s, mu = _f32_inputs()
        open_coded = jax.jit(lambda m: jnp.sum(((d - m) / s) ** 2))
        bare = float(open_coded(mu))
        folded = _folded_infinities(open_coded.lower(mu).compile().as_text())
    assert np.isnan(bare), (
        f"an open-coded ((d-mu)/sigma)**2 now gives {bare} rather than NaN. XLA has "
        "stopped folding the reciprocal, so standardized_residual may no longer be "
        "necessary — re-measure before removing it, and update #1535"
    )
    assert folded == 1, (
        f"the open-coded control compiles to {folded} bare constant(inf), not 1 — the "
        "HLO signal the test above keys on no longer identifies the defect"
    )
