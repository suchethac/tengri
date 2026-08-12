# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for FeltreNLRBackend and agn_nlr_emission dispatcher.

These tests exercise the backend's behavior without the actual grid data:
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

import chex
import pytest

pytestmark = pytest.mark.regression_paper

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tengri._data_setup import find_data
from tests._jit_parity import assert_jit_matches_eager

# parents[2] is tests/ from tests/regression/paper/, so this pointed at
# tests/data/ — which never exists, and the tests below never ran (#1431).
_GRID_PATH = find_data("feltre_grid.h5")
_GRID_AVAILABLE = _grid_available = _GRID_PATH is not None


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Tests that do NOT require grid data ───────────────────────────
def test_import_feltre_nlr_backend() -> None:
    """FeltreNLRBackend should be importable even without the data file."""
    from tengri.components.nebular.agn_nebular import FeltreNLRBackend

    assert FeltreNLRBackend.name == "feltre"
    assert FeltreNLRBackend.has_continuum is False
    assert FeltreNLRBackend.has_free_params is True


def test_feltre_backend_export_from_init() -> None:
    """FeltreNLRBackend should be exported from the nebular __init__."""
    from tengri.components.nebular import FeltreNLRBackend as exported_backend
    from tengri.components.nebular.agn_nebular import FeltreNLRBackend as canonical_backend

    # Assert the exported symbol is the same class as the canonical one
    assert exported_backend is canonical_backend


def test_feltre_backend_filenotfound_when_missing(tmp_path: Path) -> None:
    """Constructor raises FileNotFoundError when grid file does not exist."""
    from tengri.components.nebular.agn_nebular import FeltreNLRBackend

    nonexistent = tmp_path / "no_such_file.h5"
    with pytest.raises(FileNotFoundError, match="feltre_grid"):
        FeltreNLRBackend(grid_path=nonexistent)


def test_nearest_idx_basic() -> None:
    """_nearest_idx returns correct nearest index."""
    from tengri.components.nebular.agn_nebular import _nearest_idx

    axis = jnp.array([-1.2, -1.4, -1.7, -2.0])
    assert _nearest_idx(axis, -1.2) == 0
    assert _nearest_idx(axis, -2.0) == 3
    assert _nearest_idx(axis, -1.5) == 1  # -1.4 is closer (0.1) than -1.7 (0.2)
    assert _nearest_idx(axis, -1.85) == 3  # -2.0 is closer (0.15) than -1.7 (0.15 tie→3)


def test_dispatcher_requires_feltre_backend() -> None:
    """agn_nlr_emission raises ValueError when feltre_backend is None."""
    from tengri.components.nebular.agn_nebular import agn_nlr_emission

    with pytest.raises(ValueError, match="feltre_backend must be provided"):
        agn_nlr_emission(backend="feltre", feltre_backend=None)


def test_dispatcher_unknown_backend() -> None:
    """agn_nlr_emission raises ValueError for an unknown backend name."""
    from tengri.components.nebular.agn_nebular import agn_nlr_emission

    with pytest.raises(ValueError, match="Unknown AGN NLR backend"):
        agn_nlr_emission(backend="nonexistent_backend")


# ── Tests that require grid data (skip if absent) ─────────────────
@pytest.mark.skipif(not _GRID_AVAILABLE, reason="data/feltre_grid.h5 not found")
def test_feltre_backend_loads_grid() -> None:
    """Backend loads grid without error and exposes correct attributes."""
    from tengri.components.nebular.agn_nebular import FeltreNLRBackend

    backend = FeltreNLRBackend(_GRID_PATH)
    g = backend.grid
    chex.assert_shape(g.alpha_axis, (4,))
    chex.assert_shape(g.logUs_axis, (9,))
    chex.assert_shape(g.logn_axis, (3,))
    chex.assert_shape(g.logZ_axis, (16,))
    chex.assert_shape(g.xi_d_axis, (3,))
    # Grid dims should be consistent
    n_a, n_u, n_n, n_z, n_xi = (
        len(g.alpha_axis),
        len(g.logUs_axis),
        len(g.logn_axis),
        len(g.logZ_axis),
        len(g.xi_d_axis),
    )
    chex.assert_shape(g.logHB_per_logq, (n_a, n_u, n_n, n_z, n_xi))
    n_lines = g.line_wavelengths_aa.shape[0]
    chex.assert_shape(g.line_ratios, (n_a, n_u, n_n, n_z, n_xi, n_lines))


@pytest.mark.skipif(not _GRID_AVAILABLE, reason="data/feltre_grid.h5 not found")
def test_feltre_predict_returns_finite() -> None:
    """predict_agn_nlr_lines returns finite wavelengths and luminosities."""
    from tengri.components.nebular.agn_nebular import FeltreNLRBackend

    backend = FeltreNLRBackend(_GRID_PATH)
    wave, lum = backend.predict_agn_nlr_lines(
        alpha_pl=-1.7,
        neb_logU=-2.0,
        neb_logn=3.0,
        neb_logZ_gas=-1.8477,
        xi_d=0.3,
        log_qh=53.0,
    )
    chex.assert_tree_all_finite(wave)
    chex.assert_tree_all_finite(lum)
    assert jnp.all(lum >= 0), "Line luminosities must be non-negative"
    chex.assert_equal_shape([wave, lum])


@pytest.mark.skipif(not _GRID_AVAILABLE, reason="data/feltre_grid.h5 not found")
def test_feltre_gradient_logU_is_finite() -> None:
    """Gradient of total luminosity w.r.t. neb_logU is finite (triweight interpolation)."""
    from tengri.components.nebular.agn_nebular import FeltreNLRBackend

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
    from tengri.components.nebular.agn_nebular import FeltreNLRBackend

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
    from tengri.components.nebular.agn_nebular import FeltreNLRBackend

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
    from tengri.components.nebular.agn_nebular import FeltreNLRBackend

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


# ── Tests for agn_ionspec_from_alpha_pl (no data required) ────────
def test_agn_ionspec_returns_all_keys() -> None:
    """agn_ionspec_from_alpha_pl returns a dict with all 7 Cue keys."""
    from tengri.components.nebular.agn_nebular import agn_ionspec_from_alpha_pl

    result = agn_ionspec_from_alpha_pl(-1.7)
    expected_keys = {
        "ionspec_index1",
        "ionspec_index2",
        "ionspec_index3",
        "ionspec_index4",
        "ionspec_logLratio1",
        "ionspec_logLratio2",
        "ionspec_logLratio3",
    }
    assert set(result.keys()) == expected_keys


def test_agn_ionspec_log_ratios_match_reference() -> None:
    """Log luminosity ratios match a direct numpy reference computation.
    The reference mirrors the JAX logic using numpy:
      integral(lambda^{s-2}, lo, hi) = (hi^{s-1} - lo^{s-1}) / (s-1)
      logLratio_k = log10(integral_{k+1}) - log10(integral_k)
    This catches bugs in: wrong alpha sign, wrong exponent, wrong log base,
    wrong segment edge indexing, or wrong diff direction.
    """
    from tengri.components.nebular.agn_nebular import agn_ionspec_from_alpha_pl
    from tengri.components.nebular.ionizing_spectrum import _CLIP_RANGES, SEGMENT_EDGES

    alpha_pl = -1.7
    s = -alpha_pl  # wavelength_slope = 1.7
    sp1 = s - 1.0  # = 0.7
    edges = np.asarray(SEGMENT_EDGES, dtype=np.float64)

    def _seg_integral(lo: float, hi: float) -> float:
        safe_sp1 = sp1 if abs(sp1) > 1e-8 else 1e-8
        return (hi**safe_sp1 - lo**safe_sp1) / safe_sp1

    log_integrals = [np.log10(abs(_seg_integral(edges[i], edges[i + 1]))) for i in range(4)]
    ref_ratios = np.diff(log_integrals)  # shape (3,)
    ref_lr = [
        float(np.clip(ref_ratios[k], *_CLIP_RANGES[f"ionspec_logLratio{k + 1}"])) for k in range(3)
    ]
    result = agn_ionspec_from_alpha_pl(alpha_pl)
    np.testing.assert_allclose(float(result["ionspec_logLratio1"]), ref_lr[0], rtol=1e-5)
    np.testing.assert_allclose(float(result["ionspec_logLratio2"]), ref_lr[1], rtol=1e-5)
    np.testing.assert_allclose(float(result["ionspec_logLratio3"]), ref_lr[2], rtol=1e-5)


def test_agn_ionspec_indices_equal_wavelength_slope() -> None:
    """All 4 segment indices equal -alpha_pl for a pure power law."""
    from tengri.components.nebular.agn_nebular import agn_ionspec_from_alpha_pl

    alpha_pl = -1.7
    result = agn_ionspec_from_alpha_pl(alpha_pl)
    expected_slope = -alpha_pl  # = 1.7
    # 1.7 is within clip range for all indices, so no clipping occurs
    for k in ("ionspec_index1", "ionspec_index2", "ionspec_index3", "ionspec_index4"):
        np.testing.assert_allclose(float(result[k]), expected_slope, rtol=1e-5, err_msg=k)


def test_agn_ionspec_indices_clipped_steep_slope() -> None:
    """Extremely steep slope triggers clip bounds on all indices."""
    from tengri.components.nebular.agn_nebular import agn_ionspec_from_alpha_pl
    from tengri.components.nebular.ionizing_spectrum import _CLIP_RANGES

    # alpha_pl = -50 → wavelength_slope = 50, which exceeds all upper bounds
    result = agn_ionspec_from_alpha_pl(-50.0)
    assert float(result["ionspec_index1"]) == pytest.approx(_CLIP_RANGES["ionspec_index1"][1])
    assert float(result["ionspec_index2"]) == pytest.approx(_CLIP_RANGES["ionspec_index2"][1])
    assert float(result["ionspec_index3"]) == pytest.approx(_CLIP_RANGES["ionspec_index3"][1])
    assert float(result["ionspec_index4"]) == pytest.approx(_CLIP_RANGES["ionspec_index4"][1])


def test_agn_ionspec_indices_clipped_shallow_slope() -> None:
    """Very positive alpha_pl (shallow slope) triggers lower clip bounds."""
    from tengri.components.nebular.agn_nebular import agn_ionspec_from_alpha_pl
    from tengri.components.nebular.ionizing_spectrum import _CLIP_RANGES

    # alpha_pl = +10 → wavelength_slope = -10, below all lower bounds
    result = agn_ionspec_from_alpha_pl(10.0)
    assert float(result["ionspec_index1"]) == pytest.approx(_CLIP_RANGES["ionspec_index1"][0])
    assert float(result["ionspec_index2"]) == pytest.approx(_CLIP_RANGES["ionspec_index2"][0])
    assert float(result["ionspec_index3"]) == pytest.approx(_CLIP_RANGES["ionspec_index3"][0])
    assert float(result["ionspec_index4"]) == pytest.approx(_CLIP_RANGES["ionspec_index4"][0])


def test_agn_ionspec_log_ratios_vary_with_slope() -> None:
    """Log luminosity ratios change when alpha_pl changes (not constant)."""
    from tengri.components.nebular.agn_nebular import agn_ionspec_from_alpha_pl

    r1 = agn_ionspec_from_alpha_pl(-1.0)
    r2 = agn_ionspec_from_alpha_pl(-2.0)
    # Different slopes must give different ratios — a bug that returns a
    # slope-independent constant would pass the finite/range tests but fail here
    assert float(r1["ionspec_logLratio1"]) != pytest.approx(
        float(r2["ionspec_logLratio1"]), abs=0.01
    ), "logLratio1 must depend on alpha_pl"
    assert float(r1["ionspec_logLratio3"]) != pytest.approx(
        float(r2["ionspec_logLratio3"]), abs=0.01
    ), "logLratio3 must depend on alpha_pl"


def test_agn_ionspec_slope_one_no_nan() -> None:
    """alpha_pl = -1.0 gives wavelength_slope = 1.0, hits safe denominator path."""
    from tengri.components.nebular.agn_nebular import agn_ionspec_from_alpha_pl

    result = agn_ionspec_from_alpha_pl(-1.0)  # sp1 = s-1 = 0 → hits safe_sp1 branch
    for key, val in result.items():
        assert jnp.isfinite(val), f"slope=1 case: {key} = {val} is NaN/Inf"


def test_agn_ionspec_jit_compatible() -> None:
    """agn_ionspec_from_alpha_pl runs under jax.jit without error."""

    from tengri.components.nebular.agn_nebular import agn_ionspec_from_alpha_pl

    result = assert_jit_matches_eager(
        lambda a: agn_ionspec_from_alpha_pl(a)["ionspec_index1"], jnp.array(-1.7)
    )
    assert jnp.isfinite(result)


# ── Tests for _log_qh_from_lacc (no data required) ────────────────
def test_log_qh_matches_reference() -> None:
    """_log_qh_from_lacc matches a direct numpy reference computation.
    The reference mirrors the JAX logic using numpy with the same physical
    constants, catching bugs in: wrong sign on ap1, wrong ionizing fraction
    formula, wrong mean photon energy calculation, or wrong log10 application.
    """
    from tengri.components.nebular._constants import _C_CGS, _H_PLANCK
    from tengri.components.nebular.agn_nebular import _NU_LYMAN, _RYDBERG_ERG, _log_qh_from_lacc

    l_acc = 1e45
    a = -1.7
    ap1 = a + 1.0  # = -0.7, not near zero so safe branch not triggered
    nu_lyman = float(_NU_LYMAN)
    nu_max = _C_CGS / 1e-8
    nu_min = _C_CGS / (10.0e-4)
    # Ionizing fraction: integral(nu^a, nu_Ly, nu_max) / integral(nu^a, nu_min, nu_max)
    int_total = (nu_max**ap1 - nu_min**ap1) / ap1
    int_ion = (nu_max**ap1 - nu_lyman**ap1) / ap1
    f_ion = float(np.clip(abs(int_ion / int_total), 0.01, 1.0))
    # Mean ionizing photon energy: h * integral(nu^a) / integral(nu^{a-1})
    int_den = (nu_max**a - nu_lyman**a) / a
    mean_hnu = max(float(_H_PLANCK) * abs(int_ion / int_den), float(_RYDBERG_ERG))
    expected = np.log10(max(f_ion * l_acc / mean_hnu, 1.0))
    result = float(_log_qh_from_lacc(l_acc, a))
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_log_qh_clamps_at_zero_for_tiny_luminosity() -> None:
    """Vanishingly small L_acc clamps q_h to 1, so log10(q_h) = 0."""
    from tengri.components.nebular.agn_nebular import _log_qh_from_lacc

    log_qh = float(_log_qh_from_lacc(1e-20, -1.7))
    assert log_qh == pytest.approx(0.0, abs=1e-6)


def test_log_qh_linear_in_log_luminosity() -> None:
    """Doubling L_acc increases log_qh by exactly log10(2)."""
    from tengri.components.nebular.agn_nebular import _log_qh_from_lacc

    # At high luminosities the clamp at q_h=1 is inactive, so the
    # relationship is purely multiplicative: q_h ∝ L_acc.
    lq1 = float(_log_qh_from_lacc(1e44, -1.7))
    lq2 = float(_log_qh_from_lacc(2e44, -1.7))
    np.testing.assert_allclose(lq2 - lq1, np.log10(2.0), rtol=1e-5)


def test_log_qh_shallower_slope_gives_higher_qh() -> None:
    """Shallower EUV slope carries more bolometric power at ionizing frequencies.
    For f_nu ~ nu^alpha_pl, a less-negative alpha_pl means relatively more
    EUV flux above the Lyman limit.  The ionizing fraction f_ion dominates,
    so log_qh(alpha_pl=-0.5) > log_qh(alpha_pl=-2.0) at fixed L_acc.
    """
    from tengri.components.nebular.agn_nebular import _log_qh_from_lacc

    l_acc = 1e45
    log_qh_shallow = float(_log_qh_from_lacc(l_acc, -0.5))  # more EUV
    log_qh_steep = float(_log_qh_from_lacc(l_acc, -2.0))  # less EUV
    assert log_qh_shallow > log_qh_steep, (
        f"Expected shallow slope ({log_qh_shallow:.2f}) > steep slope ({log_qh_steep:.2f})"
    )


def test_log_qh_safe_denominator_path() -> None:
    """alpha_pl = -1.0 hits the safe_ap1 branch (ap1 = 0) without NaN."""
    from tengri.components.nebular.agn_nebular import _log_qh_from_lacc

    log_qh = _log_qh_from_lacc(1e45, -1.0)
    assert jnp.isfinite(log_qh), f"alpha_pl=-1 case: log_qh = {log_qh}"


def test_log_qh_jit_compatible() -> None:
    """_log_qh_from_lacc runs under jax.jit without error."""

    from tengri.components.nebular.agn_nebular import _log_qh_from_lacc

    result = assert_jit_matches_eager(_log_qh_from_lacc, jnp.array(1e45), jnp.array(-1.7))
    assert jnp.isfinite(result)


@pytest.mark.skipif(not _GRID_AVAILABLE, reason="data/feltre_grid.h5 not found")
def test_feltre_dispatcher_route() -> None:
    """agn_nlr_emission correctly routes to FeltreNLRBackend."""
    from tengri.components.nebular.agn_nebular import FeltreNLRBackend, agn_nlr_emission

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
    chex.assert_tree_all_finite(wave)
    chex.assert_tree_all_finite(lum)
