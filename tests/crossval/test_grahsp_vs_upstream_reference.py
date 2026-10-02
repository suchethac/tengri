# SPDX-License-Identifier: BSD-3-Clause
"""Crossval: tengri's GRAHSP against upstream GRAHSP's own committed spectra.

``data/grahsp_upstream_reference.h5`` holds upstream GRAHSP (commit 45054ddf) spectra for
38 parameter sets (fiducial + one-at-a-time sweeps), built by
``scripts/build_grahsp_reference.py``. Each set is evaluated here with tengri's public
component API (``evaluate_grahsp_agn``) and compared component by component.

Units and normalisation (read off the builder, not assumed)
-----------------------------------------------------------
* Wavelength: nm on both sides (``wavelength_unit`` root attribute; ``evaluate_grahsp_agn``
  takes ``wave_nm``).
* Upstream arrays are ``L_lambda`` [per nm] (per wavelength, not per frequency) with the
  builder's ``activate(fracAGN=-1)`` setting ``lum5100A = 1``, i.e. ``lambda L_lambda(510 nm)
  = 1``. tengri's ``l5100`` is ``lambda L_lambda(5100 A)`` [erg/s] and its spectra are
  ``L_lambda`` [erg/s/nm]; at ``l5100 = 1`` the two are the same numbers, so no unit
  conversion is applied.
* ``attenuation.*`` are upstream's signed differences ``attenuated - intrinsic`` (root
  attribute ``attenuation_convention``); the attenuation factor is rebuilt as
  ``(intrinsic + attenuation.*) / intrinsic``. Where the factor is below
  ``ATT_FACTOR_FLOOR`` that difference is ``-intrinsic`` to within float64 rounding and
  carries no information on the factor, so those points are excluded.

Known upstream errors, applied here as explicit conversions (never in ``src/``)
-------------------------------------------------------------------------------
* Emission lines: upstream integrates every line to sqrt(2) times the unit area tengri
  gives it (``LINE_FLUX_SCALE``).
* Torus log-Gaussian width: upstream uses exp(-x^2 / W^2), tengri exp(-x^2 / (2 W^2)) as in
  the paper's Eq. 2, so ``W_tengri = W_upstream / sqrt(2)`` (``UPSTREAM_WIDTH_TO_TENGRI``).
* Torus, FeII and Balmer continuum are tabulated upstream and linearly interpolated onto the
  reference grid. The torus is therefore evaluated at upstream's 506 template nodes and
  interpolated the same way (the 12 micron normalisation lies exactly on a node); evaluated
  directly on the 5000-point grid it differs by 1.05e-2 dex from interpolation error alone.
* Balmer continuum: tengri smooths with sigma = FWHM / 2.355 and keeps the redward tail; the
  two cannot be matched pointwise, so the 210-355 nm integral is compared.
* Torus support: upstream is hard-zero outside 0.36-100 micron; compared inside 0.4-99 micron.
* Lines: upstream evaluates the summed lines on 9 nodes per line (line centre +-3 FWHM, in
  nm) and interpolates linearly; the table of line wavelengths it reads is float32. Tengri's
  lines are evaluated at the same nodes (built with the same float32 arithmetic from the same
  float32 wavelengths) and interpolated the same way, then compared point by point and per
  line window after the sqrt(2). Evaluated directly on the 5000-point grid the line widths
  differ by 20% from interpolation alone.
* tengri zeroes the disc below ``XRAY_FLOOR_NM`` (12.4 nm: GRAHSP has no X-ray physics, so the
  disc must not double-count with the alpha_ox corona); upstream does not. BBB and the BBB
  attenuation are compared at and above that wavelength, and the zeroing itself is asserted.
* The Netzer disc is not in the reference (the builder used the power-law disc only).

Every comparison asserts a minimum number of compared points and uses thresholds relative to
the component's own peak, so a component that is all zero or fully masked cannot pass.
Measured residuals sit in the comment beside each tolerance.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants as cst

from tengri.components.agn.grahsp.bbb import XRAY_FLOOR_NM
from tengri.components.agn.grahsp.model import GRAHSPSED, GRAHSPParams, evaluate_grahsp_agn
from tengri.components.agn.grahsp.templates import load_grahsp_templates
from tests._data_skip import GRAHSP_TEMPLATE, requires_grahsp

REFERENCE_PATH = Path(__file__).resolve().parents[2] / "data" / "grahsp_upstream_reference.h5"

pytestmark = [
    pytest.mark.crossval,
    requires_grahsp,
    pytest.mark.skipif(not REFERENCE_PATH.is_file(), reason=f"missing {REFERENCE_PATH}"),
]

# ---- explicit conversions for known upstream errors ------------------------------------
LINE_FLUX_SCALE = math.sqrt(2.0)  # tengri unit-area lines -> upstream's sqrt(2)-area lines
UPSTREAM_WIDTH_TO_TENGRI = 1.0 / math.sqrt(2.0)  # torus log-Gaussian width, upstream -> tengri

_FEII_NAMES = {"BruhweilerVerner08": "bruhweiler2008", "Veron-Cetty04": "veroncetty2004"}

# upstream parameter -> (tengri GRAHSPParams field, conversion)
PARAM_MAP = {
    "plslope": ("plslope", float),
    "uvslope": ("uvslope", float),
    "plbendloc": ("plbendloc_nm", float),
    "plbendwidth": ("plbendwidth", float),
    "cutoff": ("cutoff_nm", float),
    "afeii": ("a_feii", float),
    "alines": ("a_lines", float),
    "linewidth": ("linewidth_kms", float),
    "type": ("agn_type", int),
    "fcov": ("fcov", float),
    "si": ("si", float),
    "coollam": ("cool_lam_um", float),
    "coolwidth": ("cool_width", lambda x: float(x) * UPSTREAM_WIDTH_TO_TENGRI),
    "hotlam": ("hot_lam_um", float),
    "hotwidth": ("hot_width", lambda x: float(x) * UPSTREAM_WIDTH_TO_TENGRI),
    "hotfcov": ("hot_fcov", float),
    "ebv": ("ebv", float),
    "ebv_agn": ("ebv_agn", float),
    "abc": ("a_bc", float),
    "feii": ("feii_template", lambda x: _FEII_NAMES[str(x)]),
}
L5100_REFERENCE = 1.0  # builder: activate(fracAGN=-1) -> lum5100A = 1

# The single list of parameter sets (one per HDF5 group); a completeness test pins it.
PARAM_SETS = (
    "fiducial", "plslope_-2_p0", "plslope_-1_p5", "plslope_-1_p0",
    "uvslope_-0_p5", "uvslope_0_p5", "plbendloc_80_p0", "plbendloc_120_p0",
    "plbendwidth_0_p5", "plbendwidth_2_p0", "afeii_0", "afeii_2",
    "alines_0_p5", "alines_2", "linewidth_1000_p0", "linewidth_10000_p0",
    "abc_0_p3", "abc_1_p0", "fcov_0_p2", "fcov_0_p8",
    "si_-1_p0", "si_0_p0", "coollam_15_p0", "coollam_19_p0",
    "coolwidth_0_p35", "coolwidth_0_p55", "hotlam_1_p5", "hotlam_2_p5",
    "hotwidth_0_p4", "hotwidth_0_p6", "hotfcov_0_p5", "hotfcov_2_p0",
    "ebv_0_p05", "ebv_0_p1", "ebv_agn_0_p1", "ebv_agn_0_p2",
    "feii_Veron-Cetty04", "type_2",
)  # fmt: skip

# ---- thresholds: measured residual (over all sets) in the comment, tolerance beside it --
REL_FLOOR = 1e-6  # compare where |upstream| > REL_FLOOR * its own peak
TOL_POINTWISE_DEX = 1e-9  # measured max: BBB 5.3e-16, FeII 1.1e-12, Si 6.1e-15, torus 2.5e-15 dex
N_MIN_BBB = 3500  # measured 3558 (smallest set) .. 4905
N_MIN_FEII = 300  # measured 328 (Veron-Cetty), 543 (Bruhweiler-Verner)
N_MIN_SI = 600  # measured 654
N_MIN_TORUS = 2000  # measured 2393 inside 0.4-99 micron
LINE_WINDOW_NSIG = 6.0  # window half-width in line sigmas
LINE_WINDOW_MIN_POINTS = 5
LINE_WINDOW_FLUX_FLOOR = 1e-3  # windows with less than this fraction of the max are skipped
N_MIN_LINE_POINTS = 140  # measured 143 (1000 km/s broad) .. 890
TOL_LINE_POINT_DEX = 2e-6  # measured 6.7e-7 dex (float32 line-strength table upstream)
N_MIN_LINE_WINDOWS = 4  # measured 5 (10000 km/s) .. 29 (1000 km/s)
TOL_LINE_WINDOW_FLUX = 1e-6  # measured 1.2e-7 relative
TOL_LINE_TOTAL_FLUX = 1e-6  # measured 3.8e-8 relative
TOL_LINE_WIDTH = 1e-6  # measured 7.1e-8 relative (rms width / centroid)
TOL_NORM_12UM_TENGRI = 1e-12  # measured 2.2e-16
TOL_NORM_12UM_UPSTREAM = 5e-5  # measured 1.2e-5: base-grid interpolation across the 12 um node
BC_WINDOW_NM = (210.0, 355.0)
BC_BLUE_WINDOW_NM = (210.0, 250.0)  # below 250 nm neither code smooths
N_MIN_BC_POINTS = 200  # measured 228
TOL_BC_INTEGRAL = 2e-3  # measured 9.5e-4: upstream 1.0x FWHM smoothing vs tengri FWHM/2.355
TOL_BC_BLUE_INTEGRAL = 1e-5  # measured 1.1e-6
ATT_FACTOR_FLOOR = 1e-6  # cancellation noise in (u + d) is ~1e-16 / factor
TOL_ATT_DEX = 1e-9  # measured 4.2e-11 (bbb), 1.9e-16 (torus) once ATT_FACTOR_FLOOR applies
N_MIN_ATT = 1000
ATT_MIN_DEVIATION = 0.01  # a set with E(B-V) > 0 must actually attenuate by >= 1 %

_TORUS_WINDOW_NM = (400.0, 99000.0)
_C_KMS = 299792.458
_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


@dataclass(frozen=True)
class Case:
    """One parameter set: upstream arrays, params, and tengri's evaluation."""

    name: str
    up: dict  # upstream parameter name -> value
    wave: np.ndarray  # reference grid [nm]
    ref: dict  # contribution name -> upstream array [per nm]
    sed: GRAHSPSED  # tengri on the reference grid
    torus_nodes: np.ndarray  # tengri torus on upstream's template nodes
    torus_node_wave: np.ndarray
    line_nodes: np.ndarray  # upstream's line sampling nodes [nm]
    broad_on_nodes: np.ndarray  # tengri lines at those nodes (unit-area, before sqrt(2))
    narrow_on_nodes: np.ndarray


def _tengri_params(up: dict) -> GRAHSPParams:
    kwargs = {tengri: conv(up[key]) for key, (tengri, conv) in PARAM_MAP.items() if key in up}
    return GRAHSPParams(l5100=L5100_REFERENCE, **kwargs)


@functools.lru_cache(maxsize=1)
def _templates():
    return load_grahsp_templates()


def _upstream_line_waves(templates) -> np.ndarray:
    """Line centres as upstream reads them: float32 Angstrom table, times 0.1 in float32."""
    angstrom = (np.asarray(templates.line_wave_nm) * 10.0).astype(np.float32)
    return angstrom * 0.1


def _upstream_line_nodes(line_wave32: np.ndarray, linewidth_kms: float) -> np.ndarray:
    """activatelines._init_code: 9 nodes per line over +-3 FWHM, same float32 arithmetic."""
    chunks = []
    for lam in line_wave32:
        width = lam * (linewidth_kms * 1000) / cst.c  # FWHM [nm], float32
        chunks.append(np.linspace(lam - 3.0 * width, lam + 3.0 * width, 9))
    return np.unique(np.concatenate(chunks).astype(np.float64))


@functools.lru_cache(maxsize=1)
def _torus_node_wave() -> np.ndarray:
    with h5py.File(GRAHSP_TEMPLATE, "r") as f:
        return np.asarray(f["torus"]["wave_nm"][:])


@functools.cache
def load_case(name: str) -> Case:
    with h5py.File(REFERENCE_PATH, "r") as f:
        grp = f[name]
        up = {
            k.removeprefix("param_"): (v if isinstance(v, str) else v.item())
            for k, v in grp.attrs.items()
        }
        ref = {k: np.asarray(grp[k][:]) for k in grp}
        wave = np.asarray(f["wavelength_nm"][:])
    params = _tengri_params(up)
    sed = evaluate_grahsp_agn(jnp.asarray(wave), params, _templates())
    nodes = _torus_node_wave()
    node_sed = evaluate_grahsp_agn(jnp.asarray(nodes), params, _templates())
    line_wave32 = _upstream_line_waves(_templates())
    line_nodes = _upstream_line_nodes(line_wave32, params.linewidth_kms)
    templates32 = dataclasses.replace(_templates(), line_wave_nm=line_wave32.astype(np.float64))
    line_sed = evaluate_grahsp_agn(jnp.asarray(line_nodes), params, templates32)
    return Case(
        name, up, wave, ref, sed, np.asarray(node_sed.torus), nodes,
        line_nodes, np.asarray(line_sed.broad_lines), np.asarray(line_sed.narrow_lines),
    )  # fmt: skip


def _get(case: Case, key: str) -> np.ndarray:
    return case.ref.get(key, np.zeros_like(case.wave))


def _report(kind: str, case: Case, n: int, value: float) -> None:
    print(f"CROSSVAL {kind:<22s} {case.name:<22s} n={n:<5d} max={value:.3e}")


def _pointwise_dex(t: np.ndarray, u: np.ndarray, n_min: int, window=None) -> tuple[int, float]:
    """Max |log10(t/u)| where |u| > REL_FLOOR * peak(|u|); signs must agree."""
    t, u = np.asarray(t, dtype=float), np.asarray(u, dtype=float)
    peak = np.max(np.abs(u))
    assert peak > 0.0, "upstream component is identically zero; caller must treat it as inactive"
    mask = np.abs(u) > REL_FLOOR * peak
    if window is not None:
        mask &= window
    n = int(mask.sum())
    assert n >= n_min, f"only {n} points compared (< {n_min}); comparison would be vacuous"
    assert np.all(np.sign(t[mask]) == np.sign(u[mask])), "sign mismatch against upstream"
    with np.errstate(divide="ignore"):
        dex = np.abs(np.log10(np.abs(t[mask]) / np.abs(u[mask])))
    return n, float(np.max(dex))


def _assert_inactive(case: Case, tengri: np.ndarray, upstream: np.ndarray, label: str) -> None:
    assert np.all(upstream == 0.0), f"[{case.name}] upstream {label} expected zero"
    assert np.all(np.asarray(tengri) == 0.0), (
        f"[{case.name}] {label}: upstream is zero but tengri peaks at "
        f"{np.max(np.abs(np.asarray(tengri))):.3e}"
    )


# ------------------------------------------------------------------------------------------
def test_reference_groups_match_param_sets():
    with h5py.File(REFERENCE_PATH, "r") as f:
        groups = {k for k in f if k != "wavelength_nm"}
    assert groups == set(PARAM_SETS)
    assert len(PARAM_SETS) == 38 == len(set(PARAM_SETS))


def test_param_map_covers_every_reference_parameter():
    case = load_case("fiducial")
    assert set(case.up) == set(PARAM_MAP)


@pytest.mark.parametrize("name", PARAM_SETS)
def test_l5100_normalisation(name):
    """Upstream lambda L_lambda(510 nm) = 1 = tengri l5100 (the unit bridge, measured)."""
    case = load_case(name)
    upstream = 510.0 * np.interp(510.0, case.wave, _get(case, "agn.activate_Disk"))
    sed = evaluate_grahsp_agn(jnp.asarray([510.0]), _tengri_params(case.up), _templates())
    tengri = 510.0 * float(sed.bbb[0])
    assert abs(upstream - L5100_REFERENCE) < 1e-5  # measured 2.0e-6 (grid interpolation)
    assert abs(tengri - L5100_REFERENCE) < 1e-5  # measured 3e-9 (bend/cutoff tails)


@pytest.mark.parametrize("name", PARAM_SETS)
def test_bbb(name):
    case = load_case(name)
    upstream = _get(case, "agn.activate_Disk")
    below = case.wave < XRAY_FLOOR_NM
    assert below.sum() >= 1 and np.all(upstream[below] > 0.0)  # the exclusion is real ...
    assert np.all(np.asarray(case.sed.bbb)[below] == 0.0)  # ... and tengri's choice, not a gap
    n, dex = _pointwise_dex(case.sed.bbb, upstream, N_MIN_BBB, ~below)
    _report("bbb", case, n, dex)
    assert dex <= TOL_POINTWISE_DEX, f"[{name}] BBB {dex:.3e} dex over {n} points"


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            n,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "FINDING: evaluate_grahsp_agn emits the FeII forest for agn_type=2 "
                    "(tengri peak 1.68e-3 vs upstream exactly 0). FeII is broad-line-region "
                    "emission and upstream gates it on type 1; model.py gates only the "
                    "broad lines and the Balmer continuum."
                ),
            ),
        )
        if n == "type_2"
        else n
        for n in PARAM_SETS
    ],
)
def test_feii(name):
    case = load_case(name)
    upstream = _get(case, "agn.activate_FeLines")
    if case.up["type"] != 1 or case.up["afeii"] == 0:
        _assert_inactive(case, case.sed.feii, upstream, "FeII")
        return
    n, dex = _pointwise_dex(case.sed.feii, upstream, N_MIN_FEII)
    _report("feii", case, n, dex)
    assert dex <= TOL_POINTWISE_DEX, f"[{name}] FeII {dex:.3e} dex over {n} points"


@pytest.mark.parametrize("name", PARAM_SETS)
def test_si(name):
    case = load_case(name)
    upstream = _get(case, "agn.activate_Torus_Si")
    if case.up["si"] == 0:
        _assert_inactive(case, case.sed.si, upstream, "Si")
        return
    n, dex = _pointwise_dex(case.sed.si, upstream, N_MIN_SI)  # si=-1 is negative: signs checked
    _report("si", case, n, dex)
    assert dex <= TOL_POINTWISE_DEX, f"[{name}] Si {dex:.3e} dex over {n} points"


@pytest.mark.parametrize("name", PARAM_SETS)
def test_torus_shape(name):
    """Torus after W_tengri = W_up/sqrt(2), on upstream's template nodes, inside 0.4-99 um."""
    case = load_case(name)
    interpolated = np.interp(
        case.wave, case.torus_node_wave, case.torus_nodes, left=0.0, right=0.0
    )
    lo, hi = _TORUS_WINDOW_NM
    window = (case.wave >= lo) & (case.wave <= hi)
    n, dex = _pointwise_dex(interpolated, _get(case, "agn.activate_Torus"), N_MIN_TORUS, window)
    _report("torus", case, n, dex)
    assert dex <= TOL_POINTWISE_DEX, f"[{name}] torus {dex:.3e} dex over {n} points"


@pytest.mark.parametrize("name", PARAM_SETS)
def test_torus_12um_normalisation(name):
    """lambda L_lambda(12 um) = 2.5 * l5100 * fcov (Eq. fcov), asserted on both sides."""
    case = load_case(name)
    target = 2.5 * L5100_REFERENCE * case.up["fcov"]
    sed = evaluate_grahsp_agn(jnp.asarray([12000.0]), _tengri_params(case.up), _templates())
    tengri = 12000.0 * float(sed.torus[0])
    upstream = 12000.0 * np.interp(12000.0, case.wave, _get(case, "agn.activate_Torus"))
    _report("torus_12um(tengri)", case, 1, abs(tengri / target - 1.0))
    assert abs(tengri / target - 1.0) <= TOL_NORM_12UM_TENGRI
    assert abs(upstream / target - 1.0) <= TOL_NORM_12UM_UPSTREAM


def _line_windows(line_wave: np.ndarray, linewidth_kms: float) -> list[tuple[float, float]]:
    sigma = line_wave * (linewidth_kms / _C_KMS) * _FWHM_TO_SIGMA
    edges = sorted(zip(line_wave - LINE_WINDOW_NSIG * sigma, line_wave + LINE_WINDOW_NSIG * sigma))
    merged: list[list[float]] = []
    for lo, hi in edges:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def _window_flux_and_width(wave, y, windows):
    flux, width = [], []
    for lo, hi in windows:
        m = (wave >= lo) & (wave <= hi)
        assert m.sum() >= LINE_WINDOW_MIN_POINTS, f"line window {lo:.1f}-{hi:.1f} nm undersampled"
        i0 = np.trapezoid(y[m], wave[m])
        if i0 <= 0.0:
            flux.append(0.0)
            width.append(np.nan)
            continue
        mean = np.trapezoid(y[m] * wave[m], wave[m]) / i0
        var = np.trapezoid(y[m] * (wave[m] - mean) ** 2, wave[m]) / i0
        flux.append(i0)
        width.append(math.sqrt(var) / mean)
    return np.array(flux), np.array(width)


@pytest.mark.parametrize("component", ["broad", "narrow"])
@pytest.mark.parametrize("name", PARAM_SETS)
def test_lines(name, component):
    """Tengri x sqrt(2), on upstream's line nodes, vs upstream: points, window flux and width."""
    case = load_case(name)
    key = {"broad": "agn.activate_EmLines_BL", "narrow": "agn.activate_EmLines_NL"}[component]
    on_nodes = case.broad_on_nodes if component == "broad" else case.narrow_on_nodes
    tengri = np.interp(case.wave, case.line_nodes, on_nodes * LINE_FLUX_SCALE, left=0.0, right=0.0)
    upstream = _get(case, key)
    if component == "broad" and case.up["type"] != 1:
        _assert_inactive(case, tengri, upstream, "broad lines")
        return
    line_wave = np.asarray(_templates().line_wave_nm)
    windows = _line_windows(line_wave, case.up["linewidth"])
    t_flux, t_width = _window_flux_and_width(case.wave, tengri, windows)
    u_flux, u_width = _window_flux_and_width(case.wave, upstream, windows)
    n_pts, dex = _pointwise_dex(tengri, upstream, N_MIN_LINE_POINTS)
    _report(f"lines_{component}(point)", case, n_pts, dex)
    assert dex <= TOL_LINE_POINT_DEX, f"[{name}] line points {dex:.3e} dex over {n_pts} points"
    keep = u_flux > LINE_WINDOW_FLUX_FLOOR * u_flux.max()
    n_win = int(keep.sum())
    assert n_win >= N_MIN_LINE_WINDOWS, f"only {n_win} line windows compared"
    flux_err = float(np.max(np.abs(t_flux[keep] / u_flux[keep] - 1.0)))
    total_err = abs(t_flux.sum() / u_flux.sum() - 1.0)
    width_err = float(np.max(np.abs(t_width[keep] / u_width[keep] - 1.0)))
    _report(f"lines_{component}(flux)", case, n_win, flux_err)
    _report(f"lines_{component}(total)", case, n_win, total_err)
    _report(f"lines_{component}(width)", case, n_win, width_err)
    assert flux_err <= TOL_LINE_WINDOW_FLUX, f"[{name}] per-window line flux {flux_err:.3e}"
    assert total_err <= TOL_LINE_TOTAL_FLUX, f"[{name}] total line flux {total_err:.3e}"
    assert width_err <= TOL_LINE_WIDTH, f"[{name}] line width {width_err:.3e}"


@pytest.mark.parametrize("name", PARAM_SETS)
def test_balmer_continuum(name):
    case = load_case(name)
    upstream = _get(case, "agn.activate_BC")
    if case.up["abc"] == 0 or case.up["type"] != 1:
        _assert_inactive(case, case.sed.balmer, upstream, "Balmer continuum")
        return
    t, u, w = np.asarray(case.sed.balmer), upstream, case.wave
    window = (w >= BC_WINDOW_NM[0]) & (w <= BC_WINDOW_NM[1])
    assert int(window.sum()) >= N_MIN_BC_POINTS
    assert np.max(u[window]) > 0.0
    rel = np.trapezoid(t[window], w[window]) / np.trapezoid(u[window], w[window]) - 1.0
    blue = (w >= BC_BLUE_WINDOW_NM[0]) & (w <= BC_BLUE_WINDOW_NM[1])
    rel_blue = np.trapezoid(t[blue], w[blue]) / np.trapezoid(u[blue], w[blue]) - 1.0
    _report("bc_integral", case, int(window.sum()), abs(rel))
    _report("bc_integral_blue", case, int(blue.sum()), abs(rel_blue))
    assert abs(rel) <= TOL_BC_INTEGRAL, f"[{name}] BC integral {rel:.3e}"
    assert abs(rel_blue) <= TOL_BC_BLUE_INTEGRAL, f"[{name}] BC blue integral {rel_blue:.3e}"


def _factor(attenuated, intrinsic, floor_peak):
    """Attenuation ratio where the intrinsic spectrum is above its relative floor."""
    mask = intrinsic > REL_FLOOR * floor_peak
    ratio = np.ones_like(intrinsic)
    ratio[mask] = attenuated[mask] / intrinsic[mask]
    return mask, ratio


@pytest.mark.parametrize("group", ["bbb", "torus"])
@pytest.mark.parametrize("name", PARAM_SETS)
def test_attenuation_factor(name, group):
    """tengri attenuated/intrinsic vs (intrinsic + attenuation.*)/intrinsic of upstream."""
    case = load_case(name)
    s = case.sed
    if group == "bbb":
        keys = ["agn.activate_Disk", "agn.activate_EmLines_BL", "agn.activate_EmLines_NL",
                "agn.activate_FeLines", "agn.activate_BC"]  # fmt: skip
        t_int = s.bbb + s.broad_lines + s.narrow_lines + s.feii + s.balmer
        t_att = s.bbb_attenuated
    else:
        keys = ["agn.activate_Torus", "agn.activate_Torus_Si"]
        t_int = s.torus + s.si
        t_att = s.torus_attenuated
    u_int = sum(_get(case, k) for k in keys)
    u_att = u_int + sum(_get(case, "attenuation." + k) for k in keys)
    t_int, t_att = np.asarray(t_int), np.asarray(t_att)
    if case.up["ebv"] + case.up["ebv_agn"] == 0.0:
        assert np.all(u_att == u_int), f"[{name}] upstream attenuates with E(B-V)=0"
        assert np.all(t_att == t_int), f"[{name}] tengri attenuates with E(B-V)=0"
        return
    mask, u_ratio = _factor(u_att, u_int, u_int.max())
    if group == "bbb":
        mask &= case.wave >= XRAY_FLOOR_NM
    mask &= u_ratio > ATT_FACTOR_FLOOR
    n = int(mask.sum())
    assert n >= N_MIN_ATT, f"only {n} points compared"
    _, t_ratio = _factor(t_att, t_int, u_int.max())
    assert np.max(np.abs(np.log10(u_ratio[mask]))) > math.log10(1 + ATT_MIN_DEVIATION), (
        "set does not attenuate: comparison would be vacuous"
    )
    dex = float(np.max(np.abs(np.log10(t_ratio[mask] / u_ratio[mask]))))
    _report(f"attenuation_{group}", case, n, dex)
    assert dex <= TOL_ATT_DEX, f"[{name}] {group} attenuation factor {dex:.3e} dex over {n} points"
