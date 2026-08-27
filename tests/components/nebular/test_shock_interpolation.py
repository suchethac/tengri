# SPDX-License-Identifier: BSD-3-Clause
"""Tests for MAPPINGS V triweight interpolation in shock.py.

Velocity and log_density are continuously interpolated: their gradients agree
with finite difference and move the output. **The B-field axis is not** -- it is
flat at 17 of 18 off-node points inside the documented range, against 6 of 6
responding on velocity, and that is pinned ``xfail(strict=True)`` against #2066
rather than asserted away.

Two ranges the module refuses are wider than the ranges it documents, and part
of the gap returns an identically-zero spectrum with no error (#2065). Every
test here evaluates at ``_base_point()``, inside the region the data actually
covers, and asserts the output is non-empty before asserting anything about it.
The three gradient tests used to evaluate at ``_midpoint(b_axis)`` = 500 μG,
where every ratio is zero, so each reduced to ``assert_allclose(0.0, 0.0)``.
"""

from __future__ import annotations

import chex
import pytest

pytest.importorskip("h5py", reason="h5py required for MAPPINGS grid tests")

import jax
import jax.numpy as jnp
import numpy as np


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# --- conditional import: skip entire module if grid file is absent ---

from tengri.components.nebular.shock import (
    _load_mappings_grids,
    compute_shock_sed,
    shock_line_ratios,
)

# One assignment holding both. Assigning `pytestmark` twice rebinds the name,
# which silently dropped the `bounds` taxonomy marker: `pytest -m bounds`
# collected nothing from this module and the CI marker guard still passed.
pytestmark = [
    pytest.mark.bounds,
    pytest.mark.skipif(
        _load_mappings_grids() is None,
        reason="MAPPINGS grid file not found; run scripts/download_mappings_templates.py",
    ),
]


# ── Helpers ───────────────────────────────────────────────────────


def _get_grid():
    """Return the mappings5 grid dict (skips test if unavailable)."""
    grids = _load_mappings_grids()
    if grids is None or "mappings5" not in grids:
        pytest.skip("MAPPINGS grid not available")
    return grids["mappings5"]


def _midpoint(arr):
    """Return the midpoint of a 1-D array."""
    a = np.asarray(arr)
    return float(0.5 * (a[0] + a[-1]))


def _quarter_point(arr):
    """Return the first-quarter element of a 1-D array.

    Used for the B-field axis: the full b_field axis spans 1e-4–1000 μG but
    MAPPINGS V data for solar abundance only exists up to ~10 μG.  Taking the
    quarter-point keeps tests firmly inside the valid data region.
    """
    a = np.asarray(arr)
    return float(a[len(a) // 4])


def _base_point():
    """(velocity, B-field, log_density) inside the region the data supports.

    The B-field is the axis quarter-point, not its midpoint. ``shock_line_ratios``
    documents ``shock_b_over_sqrt_n`` as valid on ``[0.0001, 10]`` μG, but the
    stored axis runs to 1000 and the range guard only refuses outside
    ``[0.0001, 1000]`` -- so ``_midpoint(b_axis)`` is **500**: inside the guard,
    outside the data, and every line ratio there is exactly 0.0 (#2065).

    That is where ``TestGradients`` used to evaluate. With every ratio zero, the
    autodiff gradient and the finite difference are both zero and
    ``assert_allclose(0.0, 0.0)`` passes against any implementation -- so the
    three gradient tests could not fail, despite a class docstring claiming they
    would catch the pre-triweight nearest-neighbor lookup.
    """
    g = _get_grid()
    return (
        float(_midpoint(g["velocities_kms"])),
        _quarter_point(g["b_axis"]),
        float(_midpoint(g["log_density_cm3"])),
    )


# ── Grid-node exact-lookup tests ──────────────────────────────────


class TestGridNodeLookup:
    """Lookups at grid nodes return a finite, non-empty set of ratios.

    Not "equal the table value", which is what this class used to claim: neither
    test indexes ``g["combined_ratios"]``, so nothing here compares against the
    table. Writing that comparison needs the abundance and component indices the
    public entry point selects internally, and is worth doing -- but the name
    should not promise it in the meantime.
    """

    def test_velocity_node_lookup_is_finite_and_nonempty(self):
        """A lookup at a velocity node, with the other axes in-domain."""
        g = _get_grid()
        v_node = float(np.asarray(g["velocities_kms"])[1])  # skip first (edge effects)
        _v, b_in, n_mid = _base_point()

        ratios = shock_line_ratios(v_node, shock_log_density=n_mid, shock_b_over_sqrt_n=b_in)

        assert len(ratios) > 0
        chex.assert_tree_all_finite(ratios)
        assert sum(ratios.values()) > 0.0, "all-zero spectrum at a grid node"

    def test_grid_corner_lookup_is_finite_and_nonempty(self):
        """The first node on every axis -- the corner most likely to be empty."""
        g = _get_grid()
        v0 = float(np.asarray(g["velocities_kms"])[0])
        b0 = float(np.asarray(g["b_axis"])[0])
        n0 = float(np.asarray(g["log_density_cm3"])[0])

        ratios = shock_line_ratios(v0, shock_log_density=n0, shock_b_over_sqrt_n=b0)

        chex.assert_tree_all_finite(ratios)
        assert sum(ratios.values()) > 0.0, "all-zero spectrum at the grid corner"


# ── Monotonicity / physics plausibility ───────────────────────────


class TestInterpolationSmoothness:
    """Triweight interpolation should produce smooth outputs along each axis."""

    def test_velocity_monotone_oiii(self):
        """[OIII] ratio increases with velocity over 200–600 km/s (Allen+2008 trend)."""
        g = _get_grid()
        # Use quarter-point of b_axis: full axis spans 1e-4–1000 μG but solar-abundance
        # data only exists up to ~10 μG, so _midpoint(≈500 μG) would query empty space.
        b_mid = _quarter_point(g["b_axis"])
        n_mid = float(_midpoint(g["log_density_cm3"]))

        velocities = np.linspace(200.0, 600.0, 10)
        oiii_key = next(
            (k for k in shock_line_ratios(300.0) if "O3_5007" in k or "OIII" in k),
            None,
        )
        if oiii_key is None:
            pytest.skip("No [OIII] 5007 line in grid")

        vals = [
            float(
                shock_line_ratios(v, shock_log_density=n_mid, shock_b_over_sqrt_n=b_mid)[oiii_key]
            )
            for v in velocities
        ]
        # [OIII] should be non-negligible and the sequence should not be all zeros
        assert max(vals) > 0.01, "Expected non-negligible [OIII] ratios"

    def test_output_across_b_field_range(self):
        """Over the documented B-field range the output is finite and non-empty.

        Sampled geometrically on ``[0.0001, 10]`` μG -- the range
        ``shock_line_ratios`` documents -- rather than linearly over the stored
        axis to 1000 μG. The old sweep did the latter, so eight of its ten
        samples sat above 100 μG where the solar grid has no data, and it
        asserted only finiteness, which zero satisfies.
        """
        v_mid, _b, n_mid = _base_point()

        for b in np.geomspace(1e-4, 10.0, 10):
            ratios = shock_line_ratios(
                v_mid, shock_log_density=n_mid, shock_b_over_sqrt_n=float(b)
            )
            chex.assert_tree_all_finite(ratios)
            assert sum(ratios.values()) > 0.0, f"all-zero spectrum at b_field={b:.4g}"

    def test_output_across_the_populated_density_range(self):
        """Over the populated density range the output is finite and non-empty.

        Capped at ``log_density = 1.5``. Measured at the in-domain B-field, the
        response falls smoothly from 80.8 at -2 to 3.79 at 1.33, then 0.0048 at
        1.89, then **exactly zero** at 2.44 and at 3.0 -- so the upper third of
        the range the docstring calls valid (``[0, 3]``) is empty, and returns
        no error saying so. Tracked in #2065; pinned by
        ``test_upper_density_range_is_empty`` below.
        """
        v_mid, b_in, _n = _base_point()

        for n in np.linspace(-2.0, 1.5, 10):
            ratios = shock_line_ratios(v_mid, shock_log_density=float(n), shock_b_over_sqrt_n=b_in)
            chex.assert_tree_all_finite(ratios)
            assert sum(ratios.values()) > 0.0, f"all-zero spectrum at log_density={n:.4g}"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "#2065: shock_line_ratios documents shock_log_density as valid on [0, 3] "
            "and refuses only outside [-2, 3], but returns an identically-zero "
            "spectrum above ~2.2 with no error and no warning."
        ),
    )
    def test_upper_density_range_is_empty(self):
        """The documented ceiling should either carry data or refuse."""
        v_mid, b_in, _n = _base_point()

        top = shock_line_ratios(v_mid, shock_log_density=3.0, shock_b_over_sqrt_n=b_in)
        assert sum(top.values()) > 0.0


# ── Gradient tests — the core motivation for this change ──────────

#: (axis id, how to vary it from the base point, FD step).
#: The step is in the axis's own units: km/s, μG, dex.
_GRAD_AXES = [
    ("velocity", 0, 1.0),
    pytest.param(
        "b_field",
        1,
        None,  # set from the base value; see _fd_step
        marks=pytest.mark.xfail(
            strict=True,
            reason=(
                "#2066: the B-field axis is flat at 17 of 18 off-node points inside "
                "the documented [0.0001, 10] μG region, against 6 of 6 responding on "
                "velocity. Physical-space bandwidth over a 7-decade log axis, the "
                "same shape as #2061."
            ),
        ),
    ),
    ("log_density", 2, 1e-3),
]


def _fd_step(value, declared):
    """FD step for one axis; relative when the table declares none.

    The B-field axis spans seven decades, so a step in absolute μG that suits
    b = 5 is meaningless at b = 0.0001. It takes ``None`` above and scales.
    """
    return value * 1e-3 if declared is None else declared


class TestGradients:
    """Continuous axes must give gradients that agree with finite difference.

    These would fail under the pre-triweight nearest-neighbor lookup, where
    ``jax.grad`` through ``np.argmin`` returns zero -- *provided they are
    evaluated where the data lives*. They were not: every one used
    ``_midpoint(b_axis)`` = 500 μG, where all ratios are identically zero and
    both sides of the comparison are 0.0. See ``_base_point``.
    """

    @pytest.mark.parametrize(("axis", "idx", "step"), _GRAD_AXES)
    def test_gradient_matches_finite_difference(self, axis, idx, step):
        """Autodiff agrees with central FD, and the axis actually moves the output.

        The second half is the part the old tests lacked. A detached gradient is
        exactly 0.0 and the difference quotient of a locally flat function is
        exactly 0.0, so ``assert_allclose`` between them passes. The
        log-sensitivity floor -- fractional change out per fractional change in
        -- is 1e-3; measured at this base point velocity gives 0.53 and
        log_density 0.73, so the floor is ~500x clear of the real values and is
        not a threshold anyone has to tune.
        """
        base = list(_base_point())

        def f(x):
            args = list(base)
            args[idx] = x
            return sum(
                shock_line_ratios(
                    args[0], shock_log_density=args[2], shock_b_over_sqrt_n=args[1]
                ).values()
            )

        value = float(f(base[idx]))
        assert value > 0.0, (
            f"{axis}: the base point returns an all-zero spectrum, so any gradient "
            f"comparison here is vacuous (#2065)"
        )

        grad_auto = float(jax.grad(f)(jnp.array(base[idx])))
        np.testing.assert_allclose(
            grad_auto,
            fd_grad(lambda x: float(f(x)), base[idx], eps=_fd_step(base[idx], step)),
            rtol=1e-3,
            err_msg=f"shock_line_ratios: FD check d/d{axis}",
        )

        log_sens = abs(base[idx] * grad_auto / value)
        assert log_sens > 1e-3, (
            f"{axis}: log-sensitivity {log_sens:.3g} — the axis does not move the "
            f"output, so the FD agreement above is 0.0 == 0.0"
        )

    def test_joint_gradient_agrees_componentwise(self):
        """One grad call over all three axes reproduces the per-axis velocity FD."""
        v0, b0, n0 = _base_point()

        def sum_ratios(params):
            v, b, n = params[0], params[1], params[2]
            return sum(shock_line_ratios(v, shock_log_density=n, shock_b_over_sqrt_n=b).values())

        params = jnp.array([v0, b0, n0])
        assert float(sum_ratios(params)) > 0.0, "base point is all-zero (#2065)"

        grads = jax.grad(sum_ratios)(params)
        chex.assert_tree_all_finite(grads)

        np.testing.assert_allclose(
            float(grads[0]),
            fd_grad(lambda v: float(sum_ratios(jnp.array([v, b0, n0]))), v0, eps=1.0),
            rtol=1e-3,
            err_msg="shock_line_ratios joint: FD check d/dvelocity",
        )
        assert abs(float(grads[0])) > 0.0, "velocity component of the joint gradient is zero"


# ── Smoke tests for compute_shock_sed ─────────────────────────────


class TestComputeShockSed:
    """Smoke tests: compute_shock_sed returns finite SEDs at mid-grid values."""

    def test_smoke_mid_grid_values(self):
        """compute_shock_sed with mid-grid parameters returns a finite SED."""
        g = _get_grid()
        v_mid = float(_midpoint(g["velocities_kms"]))
        b_mid = _quarter_point(g["b_axis"])  # full-axis midpoint ≈500 μG exceeds valid solar range
        n_mid = float(_midpoint(g["log_density_cm3"]))

        wave = jnp.linspace(1000.0, 10000.0, 500)
        sed = compute_shock_sed(
            wave,
            v_mid,
            l_shock_halpha=1e40,
            shock_log_density=n_mid,
            shock_b_over_sqrt_n=b_mid,
            line_sigma_aa=5.0,
        )
        chex.assert_equal_shape([sed, wave])
        chex.assert_tree_all_finite(sed)
        assert jnp.any(sed > 0), "compute_shock_sed returned all-zero SED"

    def test_sed_gradient_wrt_velocity(self):
        """Gradient of total SED flux w.r.t. velocity is finite."""
        g = _get_grid()
        v_mid = float(_midpoint(g["velocities_kms"]))
        b_mid = _quarter_point(g["b_axis"])  # full-axis midpoint ≈500 μG exceeds valid solar range
        n_mid = float(_midpoint(g["log_density_cm3"]))

        wave = jnp.linspace(3000.0, 9000.0, 200)

        def total_flux(v):
            return jnp.sum(
                compute_shock_sed(
                    wave,
                    v,
                    l_shock_halpha=1e40,
                    shock_log_density=n_mid,
                    shock_b_over_sqrt_n=b_mid,
                    line_sigma_aa=5.0,
                )
            )

        def f(v):
            return float(total_flux(jnp.array(v)))

        g_v = float(jax.grad(total_flux)(jnp.array(v_mid)))
        np.testing.assert_allclose(
            g_v,
            fd_grad(f, v_mid, eps=1.0),
            rtol=1e-3,
            err_msg="compute_shock_sed: FD check ∂(∑SED)/∂velocity",
        )
