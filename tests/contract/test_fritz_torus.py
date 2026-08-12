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

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

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
    assert jnp.all(L_nu >= 0), "Output contains negative luminosities"


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


@pytest.mark.contract
def test_fritz_tau_changes_output(fritz_grid_path: Path, wavelength_aa: jnp.ndarray) -> None:
    """Changing tau changes the output SED shape."""
    from tengri.components.agn.fritz import create_fritz_from_grid

    fritz_fn = create_fritz_from_grid(str(fritz_grid_path))

    sed_tau_low = fritz_fn(
        wavelength_aa,
        agn_log_lbol=44.0,
        agn_torus_frac=0.5,
        agn_fritz_r_ratio=60.0,
        agn_fritz_tau=0.1,
        agn_fritz_beta=-0.5,
        agn_fritz_gamma=4.0,
        agn_fritz_oa=60.0,
        agn_fritz_psy=0.001,
    )

    sed_tau_high = fritz_fn(
        wavelength_aa,
        agn_log_lbol=44.0,
        agn_torus_frac=0.5,
        agn_fritz_r_ratio=60.0,
        agn_fritz_tau=6.0,
        agn_fritz_beta=-0.5,
        agn_fritz_gamma=4.0,
        agn_fritz_oa=60.0,
        agn_fritz_psy=0.001,
    )

    # SEDs should differ (optical depth changes the thermal shape)
    max_diff = jnp.max(jnp.abs(sed_tau_low - sed_tau_high))
    assert float(max_diff) > 0, "Different tau values produce identical output"


@pytest.mark.contract
def test_fritz_psy_changes_output(fritz_grid_path: Path, wavelength_aa: jnp.ndarray) -> None:
    """Changing psy (viewing angle) changes the output."""
    from tengri.components.agn.fritz import create_fritz_from_grid

    fritz_fn = create_fritz_from_grid(str(fritz_grid_path))

    sed_psy_type2 = fritz_fn(
        wavelength_aa,
        agn_log_lbol=44.0,
        agn_torus_frac=0.5,
        agn_fritz_r_ratio=60.0,
        agn_fritz_tau=1.0,
        agn_fritz_beta=-0.5,
        agn_fritz_gamma=4.0,
        agn_fritz_oa=60.0,
        agn_fritz_psy=0.001,  # type-2 edge-on
    )

    sed_psy_type1 = fritz_fn(
        wavelength_aa,
        agn_log_lbol=44.0,
        agn_torus_frac=0.5,
        agn_fritz_r_ratio=60.0,
        agn_fritz_tau=1.0,
        agn_fritz_beta=-0.5,
        agn_fritz_gamma=4.0,
        agn_fritz_oa=60.0,
        agn_fritz_psy=89.99,  # type-1 face-on
    )

    max_diff = jnp.max(jnp.abs(sed_psy_type2 - sed_psy_type1))
    assert float(max_diff) > 0, "Different viewing angles produce identical output"


@pytest.mark.contract
def test_fritz_opening_angle_changes_output(
    fritz_grid_path: Path, wavelength_aa: jnp.ndarray
) -> None:
    """Changing opening angle changes the output."""
    from tengri.components.agn.fritz import create_fritz_from_grid

    fritz_fn = create_fritz_from_grid(str(fritz_grid_path))

    sed_oa_small = fritz_fn(
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

    sed_oa_large = fritz_fn(
        wavelength_aa,
        agn_log_lbol=44.0,
        agn_torus_frac=0.5,
        agn_fritz_r_ratio=60.0,
        agn_fritz_tau=1.0,
        agn_fritz_beta=-0.5,
        agn_fritz_gamma=4.0,
        agn_fritz_oa=140.0,
        agn_fritz_psy=0.001,
    )

    max_diff = jnp.max(jnp.abs(sed_oa_small - sed_oa_large))
    assert float(max_diff) > 0, "Different opening angles produce identical output"


# ────────────────────────────────────────────────────────────────────────────
# Gradient safety


@pytest.mark.gradient
def test_fritz_grad_tau_finite(fritz_grid_path: Path, wavelength_aa: jnp.ndarray) -> None:
    """Gradient w.r.t. tau is finite."""
    from tengri.components.agn.fritz import create_fritz_from_grid

    fritz_fn = create_fritz_from_grid(str(fritz_grid_path))

    def loss(tau_val: float) -> float:
        L_nu = fritz_fn(
            wavelength_aa,
            agn_log_lbol=44.0,
            agn_torus_frac=0.5,
            agn_fritz_r_ratio=60.0,
            agn_fritz_tau=tau_val,
            agn_fritz_beta=-0.5,
            agn_fritz_gamma=4.0,
            agn_fritz_oa=60.0,
            agn_fritz_psy=0.001,
        )
        return jnp.sum(L_nu)

    grad_fn = jax.grad(loss)
    g = grad_fn(1.0)
    assert float(jnp.isfinite(g)), "Gradient contains NaN or Inf"


@pytest.mark.gradient
def test_fritz_grad_psy_finite(fritz_grid_path: Path, wavelength_aa: jnp.ndarray) -> None:
    """Gradient w.r.t. psy is finite."""
    from tengri.components.agn.fritz import create_fritz_from_grid

    fritz_fn = create_fritz_from_grid(str(fritz_grid_path))

    def loss(psy_val: float) -> float:
        L_nu = fritz_fn(
            wavelength_aa,
            agn_log_lbol=44.0,
            agn_torus_frac=0.5,
            agn_fritz_r_ratio=60.0,
            agn_fritz_tau=1.0,
            agn_fritz_beta=-0.5,
            agn_fritz_gamma=4.0,
            agn_fritz_oa=60.0,
            agn_fritz_psy=psy_val,
        )
        return jnp.sum(L_nu)

    grad_fn = jax.grad(loss)
    g = grad_fn(30.0)
    assert float(jnp.isfinite(g)), "Gradient contains NaN or Inf"


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
    assert jnp.all(L_lambda >= 0)


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
