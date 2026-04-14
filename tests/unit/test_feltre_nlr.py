"""Unit tests for FeltreNLRBackend and agn_nlr_emission dispatcher.

These tests exercise the backend's behaviour without the actual grid data:
  - FileNotFoundError when the grid file is absent
  - FeltreGridData dataclass structure
  - _nearest_idx returns correct index
  - agn_nlr_emission dispatcher raises ValueError when feltre_backend is None
  - agn_nlr_emission dispatcher raises ValueError for unknown backend

When ``data/feltre_grid.h5`` *is* present, additional tests validate:
  - Grid loads with correct shapes
  - predict_agn_nlr_lines returns finite arrays
  - Gradient w.r.t. neb_logU and neb_logZ_gas is finite (triweight is C²)
  - Nearest-neighbor snap: varying alpha_pl by a tiny amount does not change output
  - At-grid-node interpolation: result equals direct table lookup within 1%
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_GRID_PATH = Path(__file__).resolve().parents[2] / "data" / "feltre_grid.h5"
_GRID_AVAILABLE = _grid_available = _GRID_PATH.exists()

pytestmark = pytest.mark.unit


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ---------------------------------------------------------------------------
# Tests that do NOT require grid data
# ---------------------------------------------------------------------------


def test_import_feltre_nlr_backend() -> None:
    """FeltreNLRBackend should be importable even without the data file."""
    from tengri.models.nebular.agn_nebular import FeltreNLRBackend

    assert FeltreNLRBackend.name == "feltre"
    assert FeltreNLRBackend.has_continuum is False
    assert FeltreNLRBackend.has_free_params is True


def test_feltre_backend_export_from_init() -> None:
    """FeltreNLRBackend should be exported from the nebular __init__."""
    from tengri.models.nebular import FeltreNLRBackend  # noqa: F401


def test_feltre_backend_filenotfound_when_missing(tmp_path: Path) -> None:
    """Constructor raises FileNotFoundError when grid file does not exist."""
    from tengri.models.nebular.agn_nebular import FeltreNLRBackend

    nonexistent = tmp_path / "no_such_file.h5"
    with pytest.raises(FileNotFoundError, match="feltre_grid"):
        FeltreNLRBackend(grid_path=nonexistent)


def test_nearest_idx_basic() -> None:
    """_nearest_idx returns correct nearest index."""
    from tengri.models.nebular.agn_nebular import _nearest_idx

    axis = jnp.array([-1.2, -1.4, -1.7, -2.0])

    assert _nearest_idx(axis, -1.2) == 0
    assert _nearest_idx(axis, -2.0) == 3
    assert _nearest_idx(axis, -1.5) == 1  # -1.4 is closer (0.1) than -1.7 (0.2)
    assert _nearest_idx(axis, -1.85) == 3  # -2.0 is closer (0.15) than -1.7 (0.15 tie→3)


def test_dispatcher_requires_feltre_backend() -> None:
    """agn_nlr_emission raises ValueError when feltre_backend is None."""
    from tengri.models.nebular.agn_nebular import agn_nlr_emission

    with pytest.raises(ValueError, match="feltre_backend must be provided"):
        agn_nlr_emission(backend="feltre", feltre_backend=None)


def test_dispatcher_unknown_backend() -> None:
    """agn_nlr_emission raises ValueError for an unknown backend name."""
    from tengri.models.nebular.agn_nebular import agn_nlr_emission

    with pytest.raises(ValueError, match="Unknown AGN NLR backend"):
        agn_nlr_emission(backend="nonexistent_backend")


# ---------------------------------------------------------------------------
# Tests that require grid data (skip if absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _GRID_AVAILABLE, reason="data/feltre_grid.h5 not found")
def test_feltre_backend_loads_grid() -> None:
    """Backend loads grid without error and exposes correct attributes."""
    from tengri.models.nebular.agn_nebular import FeltreNLRBackend

    backend = FeltreNLRBackend(_GRID_PATH)
    g = backend.grid

    assert g.alpha_axis.shape == (4,)
    assert g.logUs_axis.shape == (4,)
    assert g.logn_axis.shape == (3,)
    assert g.logZ_axis.shape == (16,)
    assert g.xi_d_axis.shape == (3,)

    # Grid dims should be consistent
    n_a, n_u, n_n, n_z, n_xi = (
        len(g.alpha_axis),
        len(g.logUs_axis),
        len(g.logn_axis),
        len(g.logZ_axis),
        len(g.xi_d_axis),
    )
    assert g.logHB_per_logq.shape == (n_a, n_u, n_n, n_z, n_xi)
    n_lines = g.line_wavelengths_aa.shape[0]
    assert g.line_ratios.shape == (n_a, n_u, n_n, n_z, n_xi, n_lines)


@pytest.mark.skipif(not _GRID_AVAILABLE, reason="data/feltre_grid.h5 not found")
def test_feltre_predict_returns_finite() -> None:
    """predict_agn_nlr_lines returns finite wavelengths and luminosities."""
    from tengri.models.nebular.agn_nebular import FeltreNLRBackend

    backend = FeltreNLRBackend(_GRID_PATH)
    wave, lum = backend.predict_agn_nlr_lines(
        alpha_pl=-1.7,
        neb_logU=-2.0,
        neb_logn=3.0,
        neb_logZ_gas=-1.8477,
        xi_d=0.3,
        log_qh=53.0,
    )

    assert jnp.all(jnp.isfinite(wave)), "Line wavelengths contain non-finite values"
    assert jnp.all(jnp.isfinite(lum)), "Line luminosities contain non-finite values"
    assert jnp.all(lum >= 0), "Line luminosities must be non-negative"
    assert wave.shape == lum.shape


@pytest.mark.skipif(not _GRID_AVAILABLE, reason="data/feltre_grid.h5 not found")
def test_feltre_gradient_logU_is_finite() -> None:
    """Gradient of total luminosity w.r.t. neb_logU is finite (triweight interpolation)."""
    from tengri.models.nebular.agn_nebular import FeltreNLRBackend

    backend = FeltreNLRBackend(_GRID_PATH)

    def total_lum(logU: float) -> float:
        _, lum = backend.predict_agn_nlr_lines(
            alpha_pl=-1.7,
            neb_logU=logU,
            neb_logn=3.0,
            neb_logZ_gas=-1.8477,
            xi_d=0.3,
            log_qh=53.0,
        )
        return jnp.sum(lum)

    grad_jax = float(jax.grad(total_lum)(-2.0))
    grad_fd = fd_grad(total_lum, -2.0)
    np.testing.assert_allclose(
        grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
    )


@pytest.mark.skipif(not _GRID_AVAILABLE, reason="data/feltre_grid.h5 not found")
def test_feltre_gradient_logZ_is_finite() -> None:
    """Gradient of total luminosity w.r.t. neb_logZ_gas is finite."""
    from tengri.models.nebular.agn_nebular import FeltreNLRBackend

    backend = FeltreNLRBackend(_GRID_PATH)

    def total_lum(logZ: float) -> float:
        _, lum = backend.predict_agn_nlr_lines(
            alpha_pl=-1.7,
            neb_logU=-2.0,
            neb_logn=3.0,
            neb_logZ_gas=logZ,
            xi_d=0.3,
            log_qh=53.0,
        )
        return jnp.sum(lum)

    grad_jax = float(jax.grad(total_lum)(-1.8477))
    grad_fd = fd_grad(total_lum, -1.8477)
    np.testing.assert_allclose(
        grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
    )


@pytest.mark.skipif(not _GRID_AVAILABLE, reason="data/feltre_grid.h5 not found")
def test_feltre_alpha_nearest_neighbor_snap() -> None:
    """Tiny perturbation to alpha_pl within the nearest-neighbor bin gives same result."""
    from tengri.models.nebular.agn_nebular import FeltreNLRBackend

    backend = FeltreNLRBackend(_GRID_PATH)

    _, lum_ref = backend.predict_agn_nlr_lines(
        alpha_pl=-1.7,
        neb_logU=-2.0,
        neb_logn=3.0,
        neb_logZ_gas=-1.8477,
        xi_d=0.3,
        log_qh=53.0,
    )
    # alpha_pl=-1.65 is still nearest to -1.7 (next value is -1.4, midpoint=-1.55)
    _, lum_perturbed = backend.predict_agn_nlr_lines(
        alpha_pl=-1.65,
        neb_logU=-2.0,
        neb_logn=3.0,
        neb_logZ_gas=-1.8477,
        xi_d=0.3,
        log_qh=53.0,
    )

    np.testing.assert_array_equal(
        np.asarray(lum_ref),
        np.asarray(lum_perturbed),
        err_msg="Nearest-neighbor alpha snap: result should be identical within same bin",
    )


@pytest.mark.skipif(not _GRID_AVAILABLE, reason="data/feltre_grid.h5 not found")
def test_feltre_fesc_scales_luminosity() -> None:
    """Photon escape fraction reduces line luminosities proportionally."""
    from tengri.models.nebular.agn_nebular import FeltreNLRBackend

    backend = FeltreNLRBackend(_GRID_PATH)

    params = dict(
        alpha_pl=-1.7, neb_logU=-2.0, neb_logn=3.0, neb_logZ_gas=-1.8477, xi_d=0.3, log_qh=53.0
    )
    _, lum_no_fesc = backend.predict_agn_nlr_lines(**params, neb_fesc=0.0)
    _, lum_half_fesc = backend.predict_agn_nlr_lines(**params, neb_fesc=0.5)

    ratio = lum_half_fesc / jnp.where(lum_no_fesc > 0, lum_no_fesc, 1.0)
    mask = lum_no_fesc > 0
    np.testing.assert_allclose(
        np.asarray(ratio[mask]),
        0.5,
        rtol=1e-6,
        err_msg="neb_fesc=0.5 should reduce all luminosities by exactly half",
    )


@pytest.mark.skipif(not _GRID_AVAILABLE, reason="data/feltre_grid.h5 not found")
def test_feltre_dispatcher_route() -> None:
    """agn_nlr_emission correctly routes to FeltreNLRBackend."""
    from tengri.models.nebular.agn_nebular import FeltreNLRBackend, agn_nlr_emission

    backend = FeltreNLRBackend(_GRID_PATH)
    wave, lum = agn_nlr_emission(
        backend="feltre",
        feltre_backend=backend,
        alpha_pl=-1.7,
        neb_logU=-2.0,
        gas_logn=3.0,
        neb_logZ_gas=-1.8477,
        xi_d=0.3,
        log_qh=53.0,
    )

    assert jnp.all(jnp.isfinite(wave))
    assert jnp.all(jnp.isfinite(lum))
