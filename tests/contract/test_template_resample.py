# SPDX-License-Identifier: BSD-3-Clause
"""Mathematical-contract tests for ``resample_template``.

Tabulated AGN torus and dust-emission templates are stored on coarse,
log-spaced wavelength grids (SKIRTOR v3 is 136 points, R ~ 7) and must be
resampled onto the much finer model wavelength grid. Interpolating linearly in
linear lambda and linear flux puts a straight chord across a curve that is a
power law in log-log, which biases the far-IR one-sided. These tests pin the
properties that motivated moving that resampling into log space.

The defining property is :func:`test_power_law_is_exact`: log-log linear
interpolation reproduces a power law exactly, at any sampling density. That is
the discriminator — a silent fallback to linear interpolation fails it, while
node-exactness and padding tests would still pass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.grid_interp import resample_template
from tests._bounds import assert_non_negative
from tests._jit_parity import assert_jit_matches_eager


@pytest.mark.limit
def test_power_law_is_exact():
    """A power law is reproduced exactly, however coarse the native grid.

    SED tails (Rayleigh-Jeans, modified blackbody, synchrotron) are power laws
    in log-log, so this is the property that removes the far-IR bias.
    """
    wave_in = np.logspace(3.0, 7.0, 12)  # deliberately coarse: R ~ 1
    slope = -2.7
    flux_in = 1e-3 * (wave_in / 1e4) ** slope

    wave_out = np.logspace(3.1, 6.9, 501)
    got = np.asarray(
        resample_template(jnp.asarray(wave_out), jnp.asarray(wave_in), jnp.asarray(flux_in))
    )
    want = 1e-3 * (wave_out / 1e4) ** slope

    rel = np.abs(got / want - 1.0)
    assert rel.max() < 1e-12, f"power law not reproduced: max rel err {rel.max():.3e}"


@pytest.mark.limit
def test_node_exact():
    """At every native node the tabulated value is returned."""
    wave_in = np.logspace(3.0, 6.0, 25)
    flux_in = np.abs(np.sin(np.log10(wave_in) * 3.0)) + 0.5

    got = np.asarray(
        resample_template(jnp.asarray(wave_in), jnp.asarray(wave_in), jnp.asarray(flux_in))
    )
    np.testing.assert_allclose(got, flux_in, rtol=1e-12)


@pytest.mark.bounds
def test_out_of_range_uses_padding():
    """Queries outside the native range return ``left`` / ``right`` unchanged."""
    wave_in = np.logspace(4.0, 6.0, 20)
    flux_in = np.linspace(1.0, 2.0, 20)
    wave_out = np.array([1e2, 1e3, 1e5, 1e7, 1e8])

    got = np.asarray(
        resample_template(
            jnp.asarray(wave_out), jnp.asarray(wave_in), jnp.asarray(flux_in), left=0.0, right=0.0
        )
    )
    assert got[0] == 0.0 and got[1] == 0.0, "below-range not padded with left"
    assert got[3] == 0.0 and got[4] == 0.0, "above-range not padded with right"
    assert got[2] > 0.0, "in-range value was padded"

    got1 = np.asarray(
        resample_template(
            jnp.asarray(wave_out), jnp.asarray(wave_in), jnp.asarray(flux_in), left=1.0, right=1.0
        )
    )
    assert got1[0] == 1.0 and got1[-1] == 1.0, "left/right=1.0 not honored"


@pytest.mark.bounds
def test_zeros_in_template_stay_finite():
    """A template containing exact zeros interpolates finitely.

    Templates are zero outside their support (e.g. a PAH-only component blueward
    of its onset). ``log(0)`` is -inf, so the log path must fall back on any
    interval with a non-positive endpoint.
    """
    wave_in = np.logspace(3.0, 6.0, 21)
    flux_in = np.zeros(21)
    flux_in[10:] = np.logspace(0.0, -3.0, 11)

    wave_out = np.logspace(3.05, 5.95, 401)
    got = np.asarray(
        resample_template(jnp.asarray(wave_out), jnp.asarray(wave_in), jnp.asarray(flux_in))
    )
    assert np.all(np.isfinite(got)), "non-finite output where the template has zeros"
    assert_non_negative(got, name="got", msg="negative flux produced from a non-negative template")


@pytest.mark.bounds
def test_no_overshoot_between_nodes():
    """The interpolant stays within the two bracketing node values."""
    wave_in = np.logspace(3.0, 6.0, 16)
    rng = np.random.default_rng(0)
    flux_in = 10.0 ** rng.uniform(-2.0, 2.0, 16)

    wave_out = np.logspace(3.001, 5.999, 2001)
    got = np.asarray(
        resample_template(jnp.asarray(wave_out), jnp.asarray(wave_in), jnp.asarray(flux_in))
    )
    idx = np.clip(np.searchsorted(wave_in, wave_out) - 1, 0, len(wave_in) - 2)
    lo = np.minimum(flux_in[idx], flux_in[idx + 1])
    hi = np.maximum(flux_in[idx], flux_in[idx + 1])
    assert np.all(got >= lo * (1 - 1e-12)), "undershoot below bracketing nodes"
    assert np.all(got <= hi * (1 + 1e-12)), "overshoot above bracketing nodes"


@pytest.mark.gradient
def test_gradient_finite_including_across_zeros():
    """``jax.grad`` w.r.t. the template values is finite, zeros included.

    The zero-handling must use the double-``where`` pattern: gating only the
    output leaves ``0 * inf = NaN`` in the VJP (the #892 trap).
    """
    wave_in = jnp.asarray(np.logspace(3.0, 6.0, 12))
    wave_out = jnp.asarray(np.logspace(3.1, 5.9, 50))

    def total(flux_in):
        return jnp.sum(resample_template(wave_out, wave_in, flux_in))

    for label, flux_in in (
        ("all positive", jnp.asarray(np.logspace(0.0, -3.0, 12))),
        ("leading zeros", jnp.asarray([0.0] * 5 + list(np.logspace(0.0, -2.0, 7)))),
        ("all zeros", jnp.zeros(12)),
    ):
        g = np.asarray(jax.grad(total)(flux_in))
        assert np.all(np.isfinite(g)), f"non-finite VJP ({label}): {g}"
        assert np.any(g != 0.0), (
            "`g` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )


@pytest.mark.limit
def test_log_flux_false_is_linear_in_flux():
    """``log_flux=False`` keeps flux linear but still uses log lambda.

    Needed for signed quantities, where a log-flux interpolation is undefined.
    """
    wave_in = np.array([1e4, 1e5])
    flux_in = np.array([-1.0, 3.0])  # changes sign: log-flux is meaningless
    wave_out = np.array([10.0**4.5])  # the geometric midpoint

    got = float(
        resample_template(
            jnp.asarray(wave_out), jnp.asarray(wave_in), jnp.asarray(flux_in), log_flux=False
        )[0]
    )
    assert abs(got - 1.0) < 1e-12, f"expected linear-in-flux midpoint 1.0, got {got}"


@pytest.mark.contract
def test_jit_and_vmap_safe():
    """The helper survives ``jit`` and ``vmap`` over a batch of templates."""
    wave_in = jnp.asarray(np.logspace(3.0, 6.0, 20))
    wave_out = jnp.asarray(np.logspace(3.1, 5.9, 64))
    batch = jnp.asarray(np.logspace(0.0, -3.0, 20) * np.arange(1, 6)[:, None])

    out = assert_jit_matches_eager(
        jax.vmap(lambda f: resample_template(wave_out, wave_in, f)), batch
    )
    assert out.shape == (5, 64)
    assert np.all(np.isfinite(np.asarray(out)))


@pytest.mark.limit
def test_loglog_integral_is_exact_for_a_power_law():
    """The integral matching ``resample_template``'s interpolant.

    Templates are normalized to unit frequency integral on their native grid
    and only then resampled. If that normalization uses ``trapezoid`` (linear
    in nu) while the resampling is log-log, the delivered SED no longer
    integrates to the normalized value and energy balance drifts. This is the
    integral of the same power-law-segment interpolant, so the two agree.
    """
    from tengri.utils.grid_interp import loglog_integral

    # f(x) = C x^s  ->  int_a^b = C (b^(s+1) - a^(s+1)) / (s+1)
    for s in (-2.7, -1.5, -0.3, 0.0, 1.4):
        x = np.logspace(0.0, 4.0, 9)  # deliberately coarse
        C = 3.0
        y = C * x**s
        got = float(loglog_integral(jnp.asarray(x), jnp.asarray(y)))
        want = C * (x[-1] ** (s + 1) - x[0] ** (s + 1)) / (s + 1)
        assert abs(got / want - 1.0) < 1e-12, f"s={s}: {got} vs {want}"


@pytest.mark.limit
def test_loglog_integral_handles_the_s_equals_minus_one_pole():
    """s = -1 makes the closed form 0/0; the stable branch must still be exact."""
    from tengri.utils.grid_interp import loglog_integral

    x = np.logspace(0.0, 3.0, 7)
    y = 5.0 / x  # s = -1 exactly: int = 5 ln(b/a)
    got = float(loglog_integral(jnp.asarray(x), jnp.asarray(y)))
    want = 5.0 * np.log(x[-1] / x[0])
    assert abs(got / want - 1.0) < 1e-12, f"{got} vs {want}"


@pytest.mark.gradient
def test_loglog_integral_finite_gradient_with_zeros():
    """Zeros fall back to trapezoid without poisoning the VJP."""
    from tengri.utils.grid_interp import loglog_integral

    x = jnp.asarray(np.logspace(0.0, 3.0, 8))

    def total(y):
        return loglog_integral(x, y)

    for y in (
        jnp.asarray(np.logspace(0.0, -2.0, 8)),
        jnp.asarray([0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0]),
        jnp.zeros(8),
    ):
        g = np.asarray(jax.grad(total)(y))
        assert np.all(np.isfinite(g)), f"non-finite VJP: {g}"
        assert np.any(g != 0.0), (
            "`g` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )
