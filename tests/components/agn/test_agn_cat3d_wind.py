# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the CAT3D-Wind torus component.

Grid templates come from AGNfitter-rX (Zhuang et al. 2024) — a
three-parameter projection of Hönig & Kishimoto 2017.  See
``scripts/build_cat3d_wind_grid.py``.  Tests skip cleanly when the
HDF5 grid is absent so the unit suite stays green on CI images without
model data.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from tengri._data_setup import find_data
from tests._jit_parity import assert_jit_matches_eager

pytestmark = []

# parents[4] is one level above the repo root from tests/components/agn/, so
# this guard was permanently true and the tests below never ran (#1431).
_GRID_PATH = find_data("cat3d_wind_torus_grid.h5")
_has_grid = _GRID_PATH is not None

pytestmark = [
    pytest.mark.bounds,
    pytest.mark.skipif(
        not _has_grid,
        reason=(
            "CAT3D-Wind grid not built. Run: "
            "python scripts/build_cat3d_wind_grid.py "
            "--input /tmp/AGNfitter-rX/models/TORUS/CAT3D_mean_3p.pickle"
        ),
    ),
]


@pytest.fixture(scope="module")
def wavelength() -> jnp.ndarray:
    return jnp.geomspace(1e3, 1e6, 512)


@pytest.fixture(scope="module")
def torus_fn():
    from tengri.components.agn.cat3d_wind import create_cat3d_wind_from_grid

    return create_cat3d_wind_from_grid(str(_GRID_PATH))


def test_grid_metadata_sensible() -> None:
    import h5py

    with h5py.File(_GRID_PATH, "r") as f:
        g = f["cat3d_wind"]
        incl = g["incl_axis"][:]
        a = g["a_axis"][:]
        fwd = g["fwd_axis"][:]
        wave = g["wavelength"][:]
        tpl = g["template"][:]

    assert incl.ndim == 1 and incl.size >= 3
    assert a.ndim == 1 and a.size >= 2
    assert fwd.ndim == 1 and fwd.size >= 3
    assert jnp.all(a[1:] > a[:-1]), "a_axis must be ascending"
    assert jnp.all(fwd[1:] > fwd[:-1]), "fwd_axis must be ascending"
    assert wave.ndim == 1 and jnp.all(wave[1:] > wave[:-1])
    chex.assert_shape(tpl, (incl.size, a.size, fwd.size, wave.size))


def test_output_shape_and_finiteness(torus_fn, wavelength) -> None:
    sed = torus_fn(
        wavelength,
        agn_log_lbol=44.0,
        agn_cos_inc=0.5,
        agn_a_cat3d=-2.0,
        agn_fwd_cat3d=1.5,
        agn_torus_frac=0.5,
    )
    chex.assert_equal_shape([sed, wavelength])
    chex.assert_tree_all_finite(sed)
    assert float(sed.max()) > 0.0


def test_luminosity_scales_linearly_with_lbol(torus_fn, wavelength) -> None:
    sed_lo = torus_fn(wavelength, agn_log_lbol=44.0)
    sed_hi = torus_fn(wavelength, agn_log_lbol=45.0)
    mask = sed_lo > 0.0
    ratio = jnp.where(mask, sed_hi / jnp.where(mask, sed_lo, 1.0), 10.0)
    assert jnp.allclose(ratio[mask], 10.0, rtol=1e-5)


def test_each_axis_actually_interpolated(torus_fn, wavelength) -> None:
    """Endpoints span the real grid extent: cos(incl) ∈ [0, 1] (incl ∈ [0, 90°]),
    a ∈ [-3, -1.5], fwd ∈ [1.0, 2.25] — AGNfitter's rows-210+ sub-library (#1036)."""
    base = torus_fn(wavelength, agn_cos_inc=0.5, agn_a_cat3d=-2.0, agn_fwd_cat3d=1.0)
    variants = [
        torus_fn(wavelength, agn_cos_inc=0.0, agn_a_cat3d=-2.0, agn_fwd_cat3d=1.0),
        torus_fn(wavelength, agn_cos_inc=0.5, agn_a_cat3d=-3.0, agn_fwd_cat3d=1.0),
        torus_fn(wavelength, agn_cos_inc=0.5, agn_a_cat3d=-2.0, agn_fwd_cat3d=2.25),
    ]
    for i, v in enumerate(variants):
        diff = float(jnp.linalg.norm(v - base))
        norm = float(jnp.linalg.norm(base) + 1e-300)
        assert diff / norm > 1e-3, f"axis {i} appears inert"


def test_jit_compatible(torus_fn, wavelength) -> None:
    sed = assert_jit_matches_eager(
        lambda ci, a, fwd: torus_fn(wavelength, agn_cos_inc=ci, agn_a_cat3d=a, agn_fwd_cat3d=fwd),
        0.4,
        -1.5,
        0.3,
    )
    chex.assert_tree_all_finite(sed)


def _scalar_loss(torus_fn, wavelength):
    def loss(ci: float, a: float, fwd: float) -> float:
        sed = torus_fn(wavelength, agn_cos_inc=ci, agn_a_cat3d=a, agn_fwd_cat3d=fwd)
        return jnp.log1p(jnp.sum(sed))

    return loss


# Measured extent of the three grid axes: outside these, the interpolation
# clamps and the derivative along that axis is exactly zero (see
# test_gradient_is_exactly_zero_off_grid).  a in [-3.0, -1.5], fwd in
# (1.0, 2.25].  An interior point is required for any gradient assertion.
_INTERIOR = (0.5, -2.0, 1.5)


def test_grad_flows_through_all_axes(torus_fn, wavelength) -> None:
    """Gradient must flow through each of the three grid axes.

    The docstring's claim, now actually asserted.  This previously evaluated
    at ``fwd=0.2`` — below the grid's lower edge of 1.0, where the fwd
    interpolation is clamped and ``d/d(fwd)`` is identically zero.  The
    assertion was ``isfinite``, which zero satisfies, so the one axis the
    test names in its own title was dead and the test still passed.
    """
    grads = jax.grad(_scalar_loss(torus_fn, wavelength), argnums=(0, 1, 2))(*_INTERIOR)
    names = ("agn_cos_inc", "agn_a_cat3d", "agn_fwd_cat3d")
    for name, g in zip(names, grads, strict=True):
        assert jnp.isfinite(g), f"{name}: gradient is not finite ({g})"
        assert float(g) != 0.0, f"{name}: gradient is identically zero — axis is dead"


@pytest.mark.parametrize(
    ("axis", "point", "argnum"),
    [
        ("agn_a_cat3d below grid", (0.5, -3.5, 1.5), 1),
        ("agn_a_cat3d above grid", (0.5, -1.0, 1.5), 1),
        ("agn_fwd_cat3d below grid", (0.5, -2.0, 0.5), 2),
        ("agn_fwd_cat3d above grid", (0.5, -2.0, 2.5), 2),
    ],
)
def test_gradient_is_exactly_zero_off_grid(torus_fn, wavelength, axis, point, argnum) -> None:
    """Off the template grid the interpolation clamps, so the axis loses its gradient.

    Documented rather than asserted-around: a fitter whose sampler wanders
    outside the grid gets no restoring force along that axis and stalls
    there silently, because the value stays finite the whole time.  Pinning
    it means the trap is visible, and a future switch to an extrapolating
    kernel turns this red instead of changing fit behaviour unremarked.
    """
    g = jax.grad(_scalar_loss(torus_fn, wavelength), argnums=argnum)(*point)
    assert float(g) == 0.0, f"{axis}: expected a clamped (zero) gradient, got {float(g)}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "cat3d_wind: d/d(agn_cos_inc) is wrong at both ends of the declared "
        "Uniform(0, 1) prior, while correct in the interior. At cos_inc=0 "
        "(edge-on) it is exactly -0.0 against a limit of -3.941267e-01. At "
        "cos_inc=1 (face-on) it is exactly HALF the limit: -1.176157 vs "
        "-2.352332, ratio 2.000016 / 2.000037 / 2.000010 / 1.999975 at "
        "(a,fwd) = (-2.0,1.5) / (-2.5,2.0) / (-1.75,1.2) / (-3.0,1.1). "
        "Interior points are clean to 6 s.f., so this is an endpoint "
        "convention, not a general interpolation defect."
    ),
)
@pytest.mark.parametrize(
    ("cos_inc", "nudge"),
    [pytest.param(0.0, 1e-6, id="edge-on"), pytest.param(1.0, -1e-6, id="face-on")],
)
def test_cos_inc_gradient_is_right_at_the_prior_endpoints(
    torus_fn, wavelength, cos_inc, nudge
) -> None:
    """Both endpoints carry prior mass, so a broken derivative there is live.

    ``agn_cos_inc`` is declared ``Uniform(0.0, 1.0)`` and its description
    reads "0=edge-on, 1=face-on" — these are not academic corners but the
    two most natural inclinations a user pins by hand.  The SED value is
    finite at both, so nothing warns; only the derivative is wrong, and a
    fitter reads that as a flat (or half-strength) direction.
    """
    grad_fn = jax.grad(_scalar_loss(torus_fn, wavelength), argnums=0)
    at_endpoint = float(grad_fn(cos_inc, -2.0, 1.5))
    just_inside = float(grad_fn(cos_inc + nudge, -2.0, 1.5))
    assert at_endpoint == pytest.approx(just_inside, rel=1e-3), (
        f"cos_inc={cos_inc}: derivative {at_endpoint:.6e} disagrees with its "
        f"own one-sided limit {just_inside:.6e}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The unified dispatch wrapper turns the face-on endpoint into a "
        "non-finite gradient that the raw grid function does not have: "
        "d/d(agn_cos_inc) at cos_inc=1.0 is +inf for agn_lum_ratio in "
        "{0.1, 0.5, 1.0} and NaN for agn_lum_ratio=0.0, at every (a,fwd) "
        "tried. The SED value there is finite (loss 150.037) and the limit "
        "is well-behaved: -1.110820 (0.99), -1.055614 (0.999), -1.047113 "
        "(0.9999), -1.046148 (0.999999). The raw path returns a finite "
        "-1.176157 at the same point, so the wrapper introduces this."
    ),
)
@pytest.mark.parametrize("lum_ratio", [0.0, 0.1, 1.0])
def test_unified_face_on_gradient_is_finite(wavelength, lum_ratio) -> None:
    """A non-finite gradient at a sampled parameter value poisons the fit.

    Nothing upstream warns, because the forward value is finite — the NaN
    only appears once the sampler applies the update, far from its cause.
    """
    import warnings

    from tengri.components.agn.unified import resolve_agn_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fn = resolve_agn_model("cat3d_wind")

    def loss(ci: float) -> float:
        sed = fn(
            wavelength,
            agn_log_lbol=44.0,
            agn_lum_ratio=lum_ratio,
            agn_cos_inc=ci,
            agn_a_cat3d=-2.0,
            agn_fwd_cat3d=1.5,
        )
        return jnp.log1p(jnp.sum(sed))

    g = jax.grad(loss)(1.0)
    assert jnp.isfinite(g), f"lum_ratio={lum_ratio}: d/d(agn_cos_inc) = {float(g)}"


def test_unified_dispatch_registered() -> None:
    import warnings

    from tengri.components.agn.unified import resolve_agn_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fn = resolve_agn_model("cat3d_wind")
    assert callable(fn)
    wl = jnp.geomspace(1e3, 1e6, 128)
    sed = fn(
        wl,
        agn_log_lbol=44.0,
        agn_lum_ratio=0.1,
        agn_cos_inc=0.5,
        agn_a_cat3d=-2.0,
        agn_fwd_cat3d=1.5,
    )
    chex.assert_equal_shape([sed, wl])
    assert float(sed.max()) > 0.0
