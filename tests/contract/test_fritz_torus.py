# SPDX-License-Identifier: BSD-3-Clause
"""Conformance tests for Fritz et al. (2006) torus model.

Tests the fritz_torus_block, grid loading, interpolation, normalization,
and gradient safety.

Markers (see tests/TESTING.md)
------
- ``@pytest.mark.contract`` — firmware contracts (grid loads, output shapes/signs)
- ``@pytest.mark.gradient`` — gradient correctness
- ``@pytest.mark.contract`` — JIT-compiles-correctly smoke tests
"""

from __future__ import annotations

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._bounds import assert_non_negative
from tests._grad_parity import assert_grad_matches_fd
from tests._jit_parity import assert_jit_matches_eager

# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def fritz_grid_path() -> Path:
    """Locate the Fritz2006 grid file, or skip if missing."""
    base = Path(__file__).resolve().parents[2]
    grid_path = base / "data" / "fritz2006_torus_grid.h5"
    if not grid_path.is_file():
        pytest.skip(
            f"Fritz2006 grid not found at {grid_path}. Run: python scripts/build_fritz2006_grid.py"
        )
    return grid_path


@pytest.fixture
def wavelength_aa() -> jnp.ndarray:
    """Standard wavelength grid [Angstrom]."""
    return jnp.logspace(2, 5, 256)  # 100 Å to 100 µm


# ────────────────────────────────────────────────────────────────────────────
# Grid loading and structure


@pytest.mark.contract
def test_fritz_grid_loads(fritz_grid_path: Path) -> None:
    """Grid file loads without error."""
    import h5py

    with h5py.File(fritz_grid_path, "r") as f:
        assert "fritz2006" in f
        g = f["fritz2006"]
        assert "wavelength_aa" in g
        assert "dust" in g
        assert "disk" in g
        # Check axes
        assert "r_ratio_axis" in g
        assert "tau_axis" in g
        assert "beta_axis" in g
        assert "gamma_axis" in g
        assert "opening_angle_axis" in g
        assert "psy_axis" in g


@pytest.mark.contract
def test_fritz_grid_shape(fritz_grid_path: Path) -> None:
    """Grid has expected shape."""
    import h5py

    with h5py.File(fritz_grid_path, "r") as f:
        g = f["fritz2006"]
        axes_shapes = {
            "r_ratio_axis": (5,),
            "tau_axis": (8,),
            "beta_axis": (5,),
            "gamma_axis": (4,),
            "opening_angle_axis": (3,),
            "psy_axis": (10,),
        }
        for axis, expected_shape in axes_shapes.items():
            assert g[axis].shape == expected_shape, (
                f"{axis} has shape {g[axis].shape}, expected {expected_shape}"
            )

        # Check grid shape (all axes + wavelength)
        dust_shape = g["dust"].shape
        assert dust_shape == (5, 8, 5, 4, 3, 10, 178), (
            f"dust shape {dust_shape}, expected (5,8,5,4,3,10,178)"
        )


@pytest.mark.contract
def test_fritz_create_from_grid(fritz_grid_path: Path) -> None:
    """create_fritz_from_grid returns a callable."""
    from tengri.components.agn.fritz import create_fritz_from_grid

    fritz_fn = create_fritz_from_grid(str(fritz_grid_path))
    assert callable(fritz_fn)


@pytest.mark.contract
def test_fritz_analytic_basic(fritz_grid_path: Path, wavelength_aa: jnp.ndarray) -> None:
    """fritz_analytic returns finite positive output."""
    from tengri.components.agn.fritz import create_fritz_from_grid

    fritz_fn = create_fritz_from_grid(str(fritz_grid_path))
    L_nu = fritz_fn(
        wavelength_aa,
        agn_log_lbol=44.0,
        agn_torus_frac=0.5,
        agn_fritz_r_ratio=60.0,
        agn_fritz_tau=1.0,
        agn_fritz_beta=-0.5,
        agn_fritz_gamma=4.0,
        agn_fritz_oa=60.0,
        agn_fritz_psy=0.001,
    )

    assert L_nu.shape == wavelength_aa.shape
    assert jnp.all(jnp.isfinite(L_nu)), "Output contains NaN or Inf"
    assert_non_negative(L_nu, name="L_nu", msg="Output contains negative luminosities")


# ────────────────────────────────────────────────────────────────────────────
# Normalization and energy conservation


@pytest.mark.contract
def test_fritz_torus_frac_normalization(fritz_grid_path: Path, wavelength_aa: jnp.ndarray) -> None:
    """Output integrated luminosity ≈ agn_torus_frac × L_bol."""
    from tengri.components.agn._phys import wavelength_to_nu
    from tengri.components.agn.fritz import create_fritz_from_grid

    fritz_fn = create_fritz_from_grid(str(fritz_grid_path))

    # Test parameters
    log_lbol = 44.5
    torus_frac = 0.6
    L_bol_lsun = 10.0**log_lbol
    L_bol_erg_s = L_bol_lsun * 3.839e33  # L_sun in erg/s

    L_nu = fritz_fn(
        wavelength_aa,
        agn_log_lbol=log_lbol,
        agn_torus_frac=torus_frac,
        agn_fritz_r_ratio=60.0,
        agn_fritz_tau=1.0,
        agn_fritz_beta=-0.5,
        agn_fritz_gamma=4.0,
        agn_fritz_oa=60.0,
        agn_fritz_psy=0.001,
    )

    # Integrate over frequency (L_ν has units erg/s/Hz)
    nu = wavelength_to_nu(wavelength_aa)
    idx_sort = jnp.argsort(nu)
    integral_nu = jnp.trapezoid(L_nu[idx_sort], nu[idx_sort])

    expected_lum = torus_frac * L_bol_erg_s
    # Allow ~5% tolerance due to grid interpolation and wavelength discretization
    rel_error = jnp.abs(integral_nu - expected_lum) / expected_lum
    assert float(rel_error) < 0.05, f"Normalization error {float(rel_error):.2%}; expected <5%"


# ────────────────────────────────────────────────────────────────────────────
# Parameter variation


# ── the six grid axes, measured ───────────────────────────────────
#
# This replaces three ~35-line near-duplicates (tau, psy, opening angle) that
# each asserted only ``max_diff > 0`` -- a threshold one ULP of difference
# satisfies -- and covered three of the six axes. Two problems came with them:
#
#   * each of them passed ``agn_log_lbol=44.0``. That field is log10(L_bol/L_sun)
#     (NAMING_CONTRACT / CLAUDE.md), so 44.0 asks for 1e44 L_sun = 3.8e77 erg/s,
#     about 33 dex above any AGN, and the SED came back at 5e64 erg/s/Hz. The
#     model is exactly linear in luminosity -- +1 dex multiplies the sum by
#     10.000000 -- so no shape conclusion changed, but every number printed by a
#     failure was meaningless, and at that scale a float32 path would overflow.
#     The value that reads as "1e44 erg/s" is 10.4; these use 11.0.
#     ``test_fritz_analytic_basic`` and ``test_fritz_torus_frac_normalization``
#     still pass 44.0 and 44.5 and are left alone: both convert through
#     ``L_sun`` themselves, so they are self-consistent at whatever luminosity
#     they name, and the model is exactly linear in it.
#
#   * the opening-angle test compared oa=60 against oa=140 on a grid that spans
#     [20, 60]. Beyond the last node the model extrapolates to ~80 and is
#     clamped flat past that (oa = 80, 100, 140 and 1000 all return the same
#     SED), so the test was measuring extrapolation, not the opening angle.
#
#: axis -> (grid low, grid high, off-node evaluation point, measured max
#: relative SED change across the full axis). Endpoints are the real grid
#: bounds read from the HDF5 axes. The evaluation point is the midpoint of the
#: widest interior cell, which is off-node by construction -- a derivative
#: referenced against a central difference must not sit on a node.
_FRITZ_AXES = {
    "agn_fritz_tau": (0.1, 10.0, 8.0, 0.711),
    "agn_fritz_psy": (0.001, 89.99, 5.0505, 0.657),
    "agn_fritz_oa": (20.0, 60.0, 30.0, 0.127),
    "agn_fritz_beta": (-1.0, 0.0, -0.875, 0.574),
    "agn_fritz_gamma": (0.0, 6.0, 1.0, 0.270),
    "agn_fritz_r_ratio": (10.0, 150.0, 125.0, 0.178),
}

#: Fixed point in the other five dimensions. Physical luminosity, mid-grid
#: everywhere else.
_FRITZ_BASE = dict(
    agn_log_lbol=11.0,
    agn_torus_frac=0.5,
    agn_fritz_r_ratio=60.0,
    agn_fritz_tau=1.0,
    agn_fritz_beta=-0.5,
    agn_fritz_gamma=4.0,
    agn_fritz_oa=60.0,
    agn_fritz_psy=30.1,
)


@functools.lru_cache(maxsize=2)
def _fritz_engine(path: str):
    from tengri.components.agn.fritz import create_fritz_from_grid

    return create_fritz_from_grid(path)


def _fritz_sed(path: str, wave: jnp.ndarray, **overrides):
    return _fritz_engine(path)(wave, **{**_FRITZ_BASE, **overrides})


@pytest.mark.contract
@pytest.mark.parametrize("axis", sorted(_FRITZ_AXES))
def test_fritz_axis_moves_the_sed(
    axis: str, fritz_grid_path: Path, wavelength_aa: jnp.ndarray
) -> None:
    """Each grid axis measurably reshapes the SED across its own grid range."""
    lo, hi, _, expected = _FRITZ_AXES[axis]
    path = str(fritz_grid_path)

    sed_lo = _fritz_sed(path, wavelength_aa, **{axis: lo})
    sed_hi = _fritz_sed(path, wavelength_aa, **{axis: hi})
    rel = float(jnp.max(jnp.abs(sed_lo - sed_hi)) / jnp.max(jnp.abs(sed_lo)))

    assert rel > 0.5 * expected, (
        f"{axis} moves the SED by only {rel:.3e} across [{lo}, {hi}]; measured "
        f"2026-08-16 it moves it by {expected:.3f}. Anything near zero means "
        f"the axis stopped reaching the templates."
    )


# ────────────────────────────────────────────────────────────────────────────
# Gradient safety


@pytest.mark.gradient
@pytest.mark.parametrize("axis", sorted(_FRITZ_AXES))
def test_fritz_axis_is_differentiable(
    axis: str, fritz_grid_path: Path, wavelength_aa: jnp.ndarray
) -> None:
    """Each axis carries a non-zero gradient that matches a finite difference.

    This replaces two tests (tau and psy) that asserted only ``isfinite`` on
    ``jax.grad``. Zero is finite, and on this model that is not hypothetical:
    ``agn_fritz_tau`` has an identically zero derivative over two thirds of its
    range (see the ratchet below). The old tau test happened to differentiate
    at tau = 1.0, one of the few places the axis is alive, and could not have
    failed anywhere else.
    """
    _, _, x, _ = _FRITZ_AXES[axis]
    path = str(fritz_grid_path)

    def total(value):
        return jnp.sum(_fritz_sed(path, wavelength_aa, **{axis: value}))

    value = float(total(x))
    # atol=0.0: these sums are ~1e31 erg/s/Hz, so the helper's old fixed
    # absolute floor was meaningless here in the other direction too.
    grad = float(assert_grad_matches_fd(total, x, rtol=1e-3, atol=0.0))

    assert grad != 0.0, f"{axis} has an exactly zero gradient at {x}"
    sensitivity = abs(x * grad / value) if value else 0.0
    assert sensitivity > 1e-3, (
        f"{axis} is effectively flat at {x}: d ln(sum L_nu) / d ln {axis} = "
        f"{sensitivity:.3e}. A fit cannot move this parameter."
    )


@pytest.mark.gradient
@pytest.mark.contract
def test_fritz_tau_is_interpolated_like_every_other_axis(
    fritz_grid_path: Path, wavelength_aa: jnp.ndarray
) -> None:
    """Verify smooth interpolation across the non-uniform tau axis.

    The tau axis [0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0] is non-uniform.
    A 40-point uniform sweep should return 40 distinct SEDs with zero-gradient
    fraction at 0%, confirming that the triweight kernel is computing weights
    in index space (not physical space). This fixes #1851.
    """
    path = str(fritz_grid_path)
    lo, hi, _, _ = _FRITZ_AXES["agn_fritz_tau"]

    def total(value):
        return jnp.sum(_fritz_sed(path, wavelength_aa, agn_fritz_tau=value))

    grad_fn = jax.jit(jax.grad(total))
    xs = np.linspace(lo, hi, 40)
    values = np.array([float(total(float(x))) for x in xs])
    grads = np.array([float(grad_fn(float(x))) for x in xs])

    zero_fraction = float(np.mean(grads == 0.0))
    assert zero_fraction == 0.0, (
        f"{zero_fraction:.1%} of a uniform tau sweep has an exactly zero "
        f"gradient; the axis is a lookup, not an interpolation."
    )
    assert len(np.unique(values)) == len(xs), (
        f"a {len(xs)}-point tau sweep returns only {len(np.unique(values))} "
        f"distinct SEDs; the axis snaps to its {8} grid nodes."
    )


@pytest.mark.contract
def test_fritz_clamps_beyond_the_opening_angle_grid(
    fritz_grid_path: Path, wavelength_aa: jnp.ndarray
) -> None:
    """Past the opening-angle grid the model extrapolates, then freezes.

    Pinned as measured behavior, not endorsed. The grid spans [20, 60] deg;
    the model keeps varying to ~80 and returns a bit-identical SED for every
    value above that. The test this replaces compared oa = 60 against oa = 140
    and read the 1.9% difference as evidence that the opening angle works --
    it is evidence about extrapolation. A caller who fits oa freely will find
    it dead above 80.
    """
    path = str(fritz_grid_path)

    def total(oa):
        return float(jnp.sum(_fritz_sed(path, wavelength_aa, agn_fritz_oa=oa)))

    far, further, absurd = total(100.0), total(140.0), total(1000.0)
    assert far == further == absurd, (
        "the opening angle is no longer frozen above the grid; the clamp "
        "changed and the note above is stale."
    )
    assert total(60.0) != far, "oa is frozen at the last grid node, not beyond it"


# ────────────────────────────────────────────────────────────────────────────
# JIT smoke tests


@pytest.mark.contract
def test_fritz_jit_compiles(fritz_grid_path: Path, wavelength_aa: jnp.ndarray) -> None:
    """fritz_analytic is JIT-compatible."""
    from tengri.components.agn.fritz import create_fritz_from_grid

    fritz_fn = create_fritz_from_grid(str(fritz_grid_path))

    L_nu = assert_jit_matches_eager(
        fritz_fn,
        wavelength_aa,
        agn_log_lbol=44.0,
        agn_torus_frac=0.5,
        agn_fritz_r_ratio=60.0,
        agn_fritz_tau=1.0,
        agn_fritz_beta=-0.5,
        agn_fritz_gamma=4.0,
        agn_fritz_oa=60.0,
        agn_fritz_psy=0.001,
    )

    assert L_nu.shape == wavelength_aa.shape
    assert jnp.all(jnp.isfinite(L_nu))


# ────────────────────────────────────────────────────────────────────────────
# Block integration


@pytest.mark.contract
def test_fritz_torus_block_registered() -> None:
    """fritz torus block is registered and accessible."""
    import tengri.components.agn.blocks.torus  # noqa: F401  populate registry
    from tengri.components.agn.blocks._protocol import AGN_BLOCKS

    # The registry is populated at import time via @register_agn_block decorators
    assert "torus" in AGN_BLOCKS
    assert "fritz" in AGN_BLOCKS["torus"], (
        f"fritz not in torus blocks: {list(AGN_BLOCKS['torus'].keys())}"
    )


@pytest.mark.contract
def test_fritz_torus_block_callable(fritz_grid_path: Path, wavelength_aa: jnp.ndarray) -> None:
    """fritz_torus_block is callable and produces correct output shape."""
    from tengri.components.agn.blocks.torus import fritz_torus_block

    L_lambda = fritz_torus_block(
        wavelength_aa,
        agn_log_lbol=44.0,
        l5100_disc=None,  # ignored
        agn_fritz_r_ratio=60.0,
        agn_fritz_tau=1.0,
        agn_fritz_beta=-0.5,
        agn_fritz_gamma=4.0,
        agn_fritz_oa=60.0,
        agn_fritz_psy=0.001,
        agn_torus_frac=0.5,
    )

    assert L_lambda.shape == wavelength_aa.shape
    assert jnp.all(jnp.isfinite(L_lambda))
    assert_non_negative(L_lambda, name="L_lambda")


# ────────────────────────────────────────────────────────────────────────────
# Model building integration


@pytest.mark.contract
def test_fritz_in_seds_model_build(
    fritz_grid_path: Path, synthetic_ssp_wide, synthetic_tophat_obs
) -> None:
    """Fritz torus is reachable through ``SEDModel.build`` and the built model
    predicts a finite SED. Uses the synthetic SSP fixture so this runs on CI
    without the gitignored real SSP grids (#613)."""
    from tengri import SEDModel

    # A disc must be active for the composable runner's normalization chain to
    # be physically meaningful; the Fritz torus itself self-normalizes off
    # agn_log_lbol × agn_torus_frac.
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        agn={"disc": {"type": "skirtor"}, "torus": {"type": "fritz"}},
    )
    assert model is not None
    assert "fritz" in str(model.spec.to_groups()["agn"])
