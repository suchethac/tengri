# SPDX-License-Identifier: BSD-3-Clause
"""A denominator's guard floor must be sized for its DERIVATIVE (#1860).

``_filter_integral_union`` ended with ``num / jnp.maximum(den, 1e-30)``.
``pad_filters_to_bucket`` pads the filter-count axis with all-zero rows, so a
padded row arrives with ``num == den == 0``. Forward that is safe at any floor.
The quotient's VJP carries ``-num/den**2``, and ``(1e-30)**2`` flushes to
exactly ``0.0`` in float32 — so the reverse pass divided by zero and returned a
**NaN redshift gradient**. Redshift alone saw it: ``den`` integrates over a grid
that scales with ``(1+z)``, so z is the only parameter reaching the denominator.

These tests pin the **rule**, not the literal. A test asserting the source
string ``1e-30`` or ``1e-18`` would block the correct fix, which is what
happened to the Cue clip bound in #477.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.photometry import FILTER_COUNT_BUCKETS, _next_filter_bucket
from tengri.utils.scale import representable_denominator, representable_floor

pytestmark = pytest.mark.regression_bug


def _tiny(dtype):
    return float(np.finfo(dtype).tiny)


def test_denominator_floor_keeps_its_reciprocal_square_representable():
    """The bound is derived from finfo, not asserted as a magic number."""
    with jax.enable_x64(False):
        floor = representable_denominator(1e-30)
        # The defining property: 1/floor**2 must not overflow.
        f32 = np.float32
        assert np.isfinite(f32(1.0) / (f32(floor) * f32(floor))), (
            f"1/{floor}**2 overflows float32; the VJP of a quotient cannot use it"
        )
        assert np.any(f32(1.0) / (f32(floor) * f32(floor)) != 0.0), (
            "`f32(1.0) / (f32(floor) * f32(floor))` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )
        assert floor >= np.sqrt(_tiny(np.float32))


def test_value_sized_floor_is_not_derivative_sized():
    """``representable_floor`` passes 1e-30 through — that is the trap, not a bug in it.

    1e-30 is *above* float32's tiny (1.18e-38), so a floor census reports the
    site clean while its reverse pass divides by zero. The two helpers must
    therefore disagree here; if they ever agree, one of them has changed meaning.
    """
    with jax.enable_x64(False):
        assert representable_floor(1e-30) == 1e-30
        assert representable_denominator(1e-30) > representable_floor(1e-30)


def test_float64_is_untouched_at_and_above_its_own_derivative_bound():
    """float64 stays bit-identical for every literal at or above sqrt(tiny_f64).

    That is 1.5e-154, so the 1e-30 / 1e-40 floors this guards do not move.
    """
    with jax.enable_x64(True):
        for literal in (1e-30, 1e-40, 1e-100):
            assert representable_denominator(literal) == literal


def test_float64_below_its_own_bound_is_raised_and_that_is_correct():
    """A 1e-300 floor is derivative-unsafe in float64 too, so it is raised there.

    Not a float64 regression: ``(1e-300)**2`` is ``0.0`` in float64 (smallest
    normal 2.2e-308), so such a site's VJP divides by zero at *both* precisions.
    Pinned because it is the one case where this helper is not a float64 no-op,
    which makes it unsafe to apply blind.
    """
    with jax.enable_x64(True):
        raised = representable_denominator(1e-300)
        assert raised > 1e-300
        assert raised == pytest.approx(np.sqrt(_tiny(np.float64)))
        assert np.isfinite(np.float64(1.0) / (raised * raised))


@pytest.mark.parametrize("literal", [1e-30, 1e-40])
def test_a_value_sized_floor_nans_a_quotient_vjp(literal):
    """The mechanism, self-contained: same expression, two floors, two outcomes.

    This is also the neuter-check — the ``representable_floor`` arm must go NaN,
    or the fix is not what removes the NaN.
    """

    def quotient(z, floor):
        # An all-zero padded row: num == den == 0, denominator scales with (1+z).
        trans = jnp.zeros(32, dtype=jnp.float32)
        grid = jnp.linspace(3000.0, 4000.0, 32, dtype=jnp.float32) * (1.0 + z)
        return jnp.trapezoid(trans, grid) / jnp.maximum(jnp.trapezoid(trans, grid), floor)

    with jax.enable_x64(False):
        z0 = jnp.asarray(0.5, dtype=jnp.float32)
        unsafe = jax.grad(quotient)(z0, representable_floor(literal))
        safe = jax.grad(quotient)(z0, representable_denominator(literal))

    assert np.isnan(unsafe), (
        f"value-sized floor {literal} no longer NaNs the VJP — this test can no "
        "longer detect the regression it exists for"
    )
    # grad-assert: finite-only — the underflow is what this measures
    assert np.isfinite(safe), f"derivative-sized floor {literal} still NaNs the VJP"


# --- end-to-end ------------------------------------------------------------

_BANDS = [
    "galex_fuv",
    "galex_nuv",
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
]


def _redshift_gradient(ssp, n_bands, x64):
    from tengri import DEFAULT, FREE, Fixed, SEDModel
    from tengri.observation import Observation, Photometry
    from tengri.parameters.priors import Uniform

    obs = Observation(photometry=Photometry.from_names(_BANDS[:n_bands]))
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
        },
        redshift=Uniform(0.01, 2.0, "redshift"),
        approx=None,
    )
    params = model.spec.sample(jax.random.PRNGKey(0))
    with jax.enable_x64(x64):
        g = jax.grad(lambda p: jnp.sum(model.predict_photometry(p)))(params)
        return float(np.asarray(g["redshift"]))


# Cover a count that pads (the broken case) and one that lands on a bucket
# exactly (which was always clean, and must stay clean). Derived from the
# bucket table so a change to FILTER_COUNT_BUCKETS cannot silently make this
# test vacuous.
_PADS = next(n for n in range(2, 8) if _next_filter_bucket(n) != n)
_EXACT = next(n for n in range(2, 8) if n in FILTER_COUNT_BUCKETS)


@pytest.mark.parametrize("n_bands", sorted({_PADS, _EXACT}))
def test_free_redshift_gradient_is_finite_in_pure_float32(ssp_bare, n_bands):
    """No band count may produce a non-finite free-redshift gradient."""
    with jax.enable_x64(False):
        assert jnp.zeros(1).dtype == np.float32, "arm is not float32"
    g = _redshift_gradient(ssp_bare, n_bands, x64=False)
    pad_rows = _next_filter_bucket(n_bands) - n_bands
    assert np.isfinite(g), (
        f"{n_bands} bands ({pad_rows} all-zero padded row(s)) gave a "
        f"non-finite float32 redshift gradient: {g}"
    )
    assert np.any(g != 0.0), (
        "`g` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
