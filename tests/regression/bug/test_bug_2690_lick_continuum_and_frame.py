# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2690: Lick equivalent widths on F_λ with the sideband line.

Trager et al. (1998, ApJS 116, 1, Sect. 2.2, Eqs. 1-3): the
pseudo-continuum F_Cλ is the straight line through the mean F_λ of the two
sidebands placed at the sideband mid-wavelengths, EW = ∫(1 - F_Iλ/F_Cλ)dλ over
the feature passband, and a magnitude index is -2.5 log10[(1/Δλ)∫F_Iλ/F_Cλ dλ].
The index operator takes a flux density per unit frequency (L_ν or F_ν) and
converts it to F_λ ∝ L_ν/λ² itself. ``pseudo_continuum="mean"`` is BAGPIPES'
arithmetic (``single_index``): a constant mean-of-sidebands continuum on the
array as given.

Expected values are numpy implementations of the definition written here, and a
closed form (a Gaussian line on a straight-line continuum). Window edges are hard
by default (#2637, Worthey et al. 1994), so the definition evaluated with hard
windows is the reference of the default cells. The opt-in 1 Å sigmoid edge
(``edge_width=1.0``) is checked against the definition evaluated with the *same*
soft weights, and against the hard definition within its measured bias.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    load_ssp_data,
    measure,
)
from tengri.observation.spectral_indices import (
    _LICK_AIR_WINDOWS,
    STANDARD_INDICES,
    SpectralIndexDef,
    measure_index_jax,
    measure_indices_from_point_terms,
    measure_indices_from_windows,
    precompute_index_windows,
)

pytestmark = pytest.mark.regression_bug

C_AA = 2.99792458e18  # speed of light [Å/s]
SOLAR_LGMET = -1.848  # log10(Z) of the solar-metallicity SSP column

#: Every Lick-type index of the registry (enumerated, never hand-listed).
EW_NAMES = tuple(n for n, d in STANDARD_INDICES.items() if d.index_type == "EW")
BREAK_NAMES = tuple(n for n, d in STANDARD_INDICES.items() if d.index_type == "break")
SLOPE_NAMES = tuple(n for n, d in STANDARD_INDICES.items() if d.index_type == "slope")

AGES_GYR = (1.0, 3.0, 10.0)

#: Largest |soft-window reference - hard-window reference| over the three ages,
#: per index [Å], times 1.25 and rounded up to 0.005. The reference differs only
#: in the 1 Å sigmoid edges, the opt-in ``edge_width=1.0`` of #2637 (the default
#: is hard); the continuum and frame error #2690 removed is 0.3-1.0 Å for HγA, HγF, Fe4383.
#: Re-measured when the Lick windows moved from their published air numbers to
#: vacuum (every edge +1.1 to +1.8 Å): the soft edges now sit on different SSP
#: pixels, so the (sampling-dependent) soft-vs-hard difference changed; the rule
#: above is unchanged.
EDGE_TOL_AA = {
    "HdA": 0.100,
    "HdF": 0.110,
    "HgA": 0.145,
    "HgF": 0.070,
    "Mgb": 0.015,
    "Fe5270": 0.165,
    "Fe5335": 0.225,
    "Hbeta": 0.055,
    "Fe4383": 0.030,
    "Ca4227": 0.100,
}

#: Largest |tengri (hard windows) - numpy hard-window definition| over the 30 registry
#: cells (10 indices x 3 ages) was 6.3e-6 Å and 6.3e-7 mag, rounded up to one digit. The two
#: differ only in where the Lick line is sampled on the pixel straddling a window edge.
HARD_TOL_AA = 1e-5
HARD_TOL_MAG = 1e-6

#: Same rule for the magnitude form of each index [mag].
EDGE_TOL_MAG = {
    "HdA": 0.0030,
    "HdF": 0.0075,
    "HgA": 0.0035,
    "HgF": 0.0050,
    "Mgb": 0.0005,
    "Fe5270": 0.0050,
    "Fe5335": 0.0070,
    "Hbeta": 0.0025,
    "Fe4383": 0.0010,
    "Ca4227": 0.0100,
}

#: Bound on |window-LUT - exact path| for an EW index [Å]. The LUT evaluates the Lick
#: definition at the window grid points, with the continuum of the exact path
#: (``_lick_continuum``); the two sum the same terms, so they agree to round-off in
#: float64 (the 15 x 93 SSP spectra of the shipped grid, native and non-uniform grids:
#: below 1e-12 Å). In float32 the F_λ sideband means and the feature sum carry the
#: working-precision rounding of the two summation orders: 3.1e-5 Å at most, bound 5 x that.
LUT_BOUND_AA = {jnp.float64: 1e-9, jnp.float32: 1.5e-4}

#: Same comparison for a deep emission feature, relative to |EW| (synthetic Hα cells
#: below, EW -135 to -678 Å).
LUT_BOUND_EMISSION_REL = 1e-9

#: Added to the bounds in float32 [Å]: |float32 - float64| of an index was <= 2.3e-3 Å
#: on the cells below.
FLOAT32_SLACK_AA = 5e-3


# ── numpy references ───────────────────────────────────────────────


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _soft_weights(wave, lo, hi, edge=1.0):
    return _sigmoid((wave - lo) / edge) * _sigmoid((hi - wave) / edge)


def _soft_mean(wave, y, lo, hi):
    w = _soft_weights(wave, lo, hi)
    return np.trapezoid(y * w, wave) / np.trapezoid(w, wave)


def lick_soft_reference(wave, flam, idx, magnitude=False):
    """Trager et al. 1998 Eqs. 1-3 with the same 1 Å soft window weights as tengri."""
    (b0, b1), (r0, r1) = idx.continuum
    f0, f1 = idx.feature
    x_b, x_r = 0.5 * (b0 + b1), 0.5 * (r0 + r1)
    f_b, f_r = _soft_mean(wave, flam, b0, b1), _soft_mean(wave, flam, r0, r1)
    f_c = f_b + (f_r - f_b) * (wave - x_b) / (x_r - x_b)
    w = _soft_weights(wave, f0, f1)
    ratio = np.trapezoid(w * flam / f_c, wave) / np.trapezoid(w, wave)
    return -2.5 * np.log10(ratio) if magnitude else (f1 - f0) * (1.0 - ratio)


def _segment(wave, y, lo, hi):
    x = np.concatenate(([lo], wave[(wave > lo) & (wave < hi)], [hi]))
    return x, np.interp(x, wave, y)


def _hard_mean(wave, y, lo, hi):
    x, v = _segment(wave, y, lo, hi)
    return np.trapezoid(v, x) / (hi - lo)


def lick_hard_reference(wave, flam, idx, magnitude=False):
    """Trager et al. 1998 Eqs. 1-3, hard windows, flux interpolated at the bounds."""
    (b0, b1), (r0, r1) = idx.continuum
    f0, f1 = idx.feature
    x_b, x_r = 0.5 * (b0 + b1), 0.5 * (r0 + r1)
    f_b, f_r = _hard_mean(wave, flam, b0, b1), _hard_mean(wave, flam, r0, r1)
    x, v = _segment(wave, flam, f0, f1)
    f_c = f_b + (f_r - f_b) * (x - x_b) / (x_r - x_b)
    if magnitude:
        return -2.5 * np.log10(np.trapezoid(v / f_c, x) / (f1 - f0))
    return np.trapezoid(1.0 - v / f_c, x)


def mean_hard_reference(wave, flux, idx):
    """BAGPIPES ``single_index`` arithmetic (hard ``>``/``<`` masks, per-pixel means)."""
    conts = [np.mean(flux[(wave > lo) & (wave < hi)]) for lo, hi in idx.continuum]
    f0, f1 = idx.feature
    feat = np.mean(flux[(wave > f0) & (wave < f1)])
    c = np.mean(conts)
    return (f1 - f0) * (c - feat) / c


# ── shared data ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ssp():
    return load_ssp_data("data/fsps_prsc_miles_chabrier.h5")


@pytest.fixture(scope="module")
def grid(ssp):
    """Optical slice of the shipped SSP grid: (wave, flux[met, age, wave])."""
    wave = np.asarray(ssp.ssp_wave, dtype=float)
    sel = (wave > 3600.0) & (wave < 5600.0)
    return wave[sel], np.asarray(ssp.ssp_flux, dtype=float)[:, :, sel]


def _solar_spectrum(ssp, grid, age_gyr):
    wave, flux = grid
    i_met = int(np.argmin(np.abs(np.asarray(ssp.ssp_lgmet) - SOLAR_LGMET)))
    i_age = int(np.argmin(np.abs(np.asarray(ssp.ssp_lg_age_gyr) - np.log10(age_gyr))))
    return wave, flux[i_met, i_age]


def _flam(wave, lnu):
    return lnu * C_AA / wave**2


# ── (a) closed form ────────────────────────────────────────────────

#: |tengri - closed form| over the 16 cells below was 1.6e-4 Å (0.25 Å grid) and
#: 7.1e-3 Å (1.5 Å grid with a random cluster): trapezoid error of a 3 Å Gaussian.
CLOSED_FORM_TOL_AA = {"uniform": 5e-4, "nonuniform": 1.5e-2}
CLOSED_FORM_TOL_MAG = {"uniform": 2e-5, "nonuniform": 6e-4}

_SIGMA = 3.0  # Gaussian line width [Å]
_LINE_CENTER = 5000.0
_GEOMETRIES = {
    # sideband mid-wavelengths 4950, 5050: symmetric about the feature center
    "symmetric": SpectralIndexDef(
        name="sym",
        index_type="EW",
        continuum=((4930.0, 4970.0), (5030.0, 5070.0)),
        feature=(4985.0, 5015.0),
    ),
    # sideband mid-wavelengths 4967.5, 5125: 32.5 Å and 125 Å from the center
    "asymmetric": SpectralIndexDef(
        name="asym",
        index_type="EW",
        continuum=((4960.0, 4975.0), (5050.0, 5200.0)),
        feature=(4985.0, 5015.0),
    ),
}


def _grid_uniform():
    return np.arange(4800.0, 5300.0 + 1e-9, 0.25)


def _grid_nonuniform():
    rng = np.random.default_rng(2690)
    coarse = np.arange(4800.0, 5300.0, 1.5)
    cluster = rng.uniform(4975.0, 5025.0, 120)
    return np.unique(np.concatenate([coarse, cluster, [5300.0]]))


_GRIDS = {"uniform": _grid_uniform, "nonuniform": _grid_nonuniform}


def _line_on_continuum(wave, kind, amp):
    """F_λ of ``C(λ) (1 - amp G(λ))`` with G a unit-peak Gaussian of width 3 Å."""
    gauss = np.exp(-0.5 * ((wave - _LINE_CENTER) / _SIGMA) ** 2)
    if kind == "linear+":
        cont = 1.0 + 1.5 * (wave - _LINE_CENTER) / _LINE_CENTER
    elif kind == "linear-":
        cont = 1.0 - 1.5 * (wave - _LINE_CENTER) / _LINE_CENTER
    else:
        cont = (wave / _LINE_CENTER) ** float(kind)
    return cont * (1.0 - amp * gauss)


def _closed_form_ew(amp, half_width):
    """∫ amp G dλ over the feature: amp σ sqrt(2π) erf(h / (σ sqrt 2))."""
    from math import erf, pi, sqrt

    return amp * _SIGMA * sqrt(2.0 * pi) * erf(half_width / (_SIGMA * sqrt(2.0)))


@pytest.mark.parametrize("amp", [0.4, -0.4], ids=["absorption", "emission"])
@pytest.mark.parametrize("kind", ["linear+", "linear-"])
@pytest.mark.parametrize("grid_name", list(_GRIDS))
@pytest.mark.parametrize("geometry", list(_GEOMETRIES))
def test_gaussian_on_a_straight_line_continuum_matches_the_closed_form(
    geometry, grid_name, kind, amp
):
    """The sideband line IS the continuum, so EW = ∫ amp G dλ (and the mag form follows)."""
    idx = _GEOMETRIES[geometry]
    wave = _GRIDS[grid_name]()
    flam = _line_on_continuum(wave, kind, amp)
    lnu = flam * wave**2 / C_AA  # the operator is given a per-frequency flux
    half = 0.5 * (idx.feature[1] - idx.feature[0])
    truth = _closed_form_ew(amp, half)
    got = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(lnu), idx))
    assert got == pytest.approx(truth, abs=CLOSED_FORM_TOL_AA[grid_name]), (geometry, kind, amp)
    width = idx.feature[1] - idx.feature[0]
    mag_idx = dataclasses.replace(idx, units="mag")
    mag = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(lnu), mag_idx))
    mag_truth = -2.5 * np.log10(1.0 - truth / width)
    assert mag == pytest.approx(mag_truth, abs=CLOSED_FORM_TOL_MAG[grid_name])


@pytest.mark.parametrize("amp", [0.4, -0.4], ids=["absorption", "emission"])
@pytest.mark.parametrize("kind", ["linear+", "-4", "3"])
@pytest.mark.parametrize("grid_name", list(_GRIDS))
@pytest.mark.parametrize("geometry", list(_GEOMETRIES))
def test_soft_edge_option_equals_the_definition_with_the_same_window_weights(
    geometry, grid_name, kind, amp
):
    """Opt-in ``edge_width=1.0``: tengri == numpy Eqs. 1-2 with identical soft weights."""
    idx = _GEOMETRIES[geometry]
    wave = _GRIDS[grid_name]()
    flam = _line_on_continuum(wave, kind, amp)
    lnu = flam * wave**2 / C_AA
    w, f = jnp.asarray(wave), jnp.asarray(lnu)
    got = float(measure_index_jax(w, f, idx, edge_width=1.0))
    assert got == pytest.approx(lick_soft_reference(wave, flam, idx), abs=1e-9, rel=1e-9)
    mag_idx = dataclasses.replace(idx, units="mag")
    mag = float(measure_index_jax(w, f, mag_idx, edge_width=1.0))
    assert mag == pytest.approx(lick_soft_reference(wave, flam, idx, magnitude=True), abs=1e-10)


@pytest.mark.parametrize("amp", [0.4, -0.4], ids=["absorption", "emission"])
def test_the_constant_continuum_misses_the_closed_form_on_asymmetric_sidebands(amp):
    """Sensitivity: BAGPIPES' constant continuum is off by >= 0.1 Å here; the default is not."""
    idx = _GEOMETRIES["asymmetric"]
    wave = _grid_uniform()
    flam = _line_on_continuum(wave, "linear+", amp)
    half = 0.5 * (idx.feature[1] - idx.feature[0])
    truth = _closed_form_ew(amp, half)
    mean_idx = dataclasses.replace(idx, pseudo_continuum="mean")
    old = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(flam), mean_idx))
    assert abs(old - truth) > 0.1


# ── (b) frame invariance and the flux-argument contract ────────────


@pytest.mark.parametrize("name", EW_NAMES)
def test_one_spectrum_in_three_declared_frames_gives_one_answer(ssp, grid, name):
    """L_ν, F_ν (any scale) and F_λ·λ² (the documented way to pass F_λ) agree."""
    idx = STANDARD_INDICES[name]
    wave, lnu = _solar_spectrum(ssp, grid, 10.0)
    flam = _flam(wave, lnu)
    w = jnp.asarray(wave)
    from_lnu = float(measure_index_jax(w, jnp.asarray(lnu), idx))
    from_fnu = float(measure_index_jax(w, jnp.asarray(lnu * 1e-6), idx))
    from_flam = float(measure_index_jax(w, jnp.asarray(flam * wave**2), idx))
    public = float(measure.spectral_index(wave, lnu, name))
    assert from_fnu == pytest.approx(from_lnu, rel=1e-9, abs=1e-9)
    assert from_flam == pytest.approx(from_lnu, rel=1e-9, abs=1e-9)
    assert public == from_lnu


def test_an_f_lambda_array_passed_as_given_is_a_different_quantity(ssp, grid):
    """The contract is per-frequency flux: F_λ passed unconverted moves HγA by 0.10 Å."""
    idx = STANDARD_INDICES["HgA"]
    wave, lnu = _solar_spectrum(ssp, grid, 10.0)
    w = jnp.asarray(wave)
    right = float(measure_index_jax(w, jnp.asarray(lnu), idx))
    wrong = float(measure_index_jax(w, jnp.asarray(_flam(wave, lnu)), idx))
    assert abs(wrong - right) > 0.05


def test_the_mean_option_measures_the_array_as_given(ssp, grid):
    """``pseudo_continuum="mean"`` has no frame conversion: L_ν and F_λ inputs differ."""
    idx = dataclasses.replace(STANDARD_INDICES["HgA"], pseudo_continuum="mean")
    wave, lnu = _solar_spectrum(ssp, grid, 10.0)
    w = jnp.asarray(wave)
    on_lnu = float(measure_index_jax(w, jnp.asarray(lnu), idx))
    on_flam = float(measure_index_jax(w, jnp.asarray(_flam(wave, lnu)), idx))
    assert abs(on_lnu - on_flam) > 0.1


# ── (c) registry sweep against the hard-window definition ──────────


def test_the_edge_tolerance_tables_cover_exactly_the_registry_ew_indices():
    assert set(EDGE_TOL_AA) == set(EW_NAMES)
    assert set(EDGE_TOL_MAG) == set(EW_NAMES)


@pytest.mark.parametrize("age", AGES_GYR)
@pytest.mark.parametrize("name", EW_NAMES)
def test_registry_ew_matches_the_hard_window_lick_definition(ssp, grid, name, age):
    idx = STANDARD_INDICES[name]
    wave, lnu = _solar_spectrum(ssp, grid, age)
    got = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(lnu), idx))
    ref = lick_hard_reference(wave, _flam(wave, lnu), idx)
    assert abs(got - ref) < HARD_TOL_AA, f"{name} {age} Gyr: {got:.4f} vs Lick {ref:.4f}"


@pytest.mark.parametrize("age", AGES_GYR)
@pytest.mark.parametrize("name", EW_NAMES)
def test_soft_edge_option_stays_within_its_documented_bias_of_the_hard_window(
    ssp, grid, name, age
):
    idx = STANDARD_INDICES[name]
    wave, lnu = _solar_spectrum(ssp, grid, age)
    soft = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(lnu), idx, edge_width=1.0))
    ref = lick_hard_reference(wave, _flam(wave, lnu), idx)
    assert abs(soft - ref) < EDGE_TOL_AA[name], f"{name} {age} Gyr: {soft:.4f} vs Lick {ref:.4f}"


@pytest.mark.parametrize("age", AGES_GYR)
@pytest.mark.parametrize("name", EW_NAMES)
def test_registry_mag_form_matches_the_hard_window_lick_definition(ssp, grid, name, age):
    idx = dataclasses.replace(STANDARD_INDICES[name], units="mag")
    wave, lnu = _solar_spectrum(ssp, grid, age)
    got = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(lnu), idx))
    ref = lick_hard_reference(wave, _flam(wave, lnu), idx, magnitude=True)
    assert abs(got - ref) < HARD_TOL_MAG, f"{name} {age} Gyr: {got:.5f} vs Lick {ref:.5f}"
    soft = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(lnu), idx, edge_width=1.0))
    assert abs(soft - ref) < EDGE_TOL_MAG[name]


@pytest.mark.parametrize("age", AGES_GYR)
@pytest.mark.parametrize("name", EW_NAMES)
def test_registry_ew_soft_option_equals_the_soft_window_reference_exactly(ssp, grid, name, age):
    idx = STANDARD_INDICES[name]
    wave, lnu = _solar_spectrum(ssp, grid, age)
    got = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(lnu), idx, edge_width=1.0))
    ref = lick_soft_reference(wave, _flam(wave, lnu), idx)
    assert got == pytest.approx(ref, abs=1e-9, rel=1e-10)


def mean_hard_exact_reference(wave, flux, idx):
    """Constant mean-of-sidebands continuum, hard windows (exact top-hat means)."""
    (b0, b1), (r0, r1) = idx.continuum
    f0, f1 = idx.feature
    cont = 0.5 * (_hard_mean(wave, flux, b0, b1) + _hard_mean(wave, flux, r0, r1))
    return (f1 - f0) * (cont - _hard_mean(wave, flux, f0, f1)) / cont


@pytest.mark.parametrize(
    "name,age,gap",
    [
        # Signed gap (constant continuum on L_nu, "mean" option) - (Lick straight-line
        # pseudo-continuum on F_lambda), both on hard windows, from the numpy references below.
        ("HgA", 1.0, 0.307473),
        ("Fe4383", 1.0, 0.404708),
        ("HgA", 10.0, 0.991276),
        ("Fe4383", 10.0, 0.430232),
    ],
)
def test_the_mean_continuum_gap_to_the_lick_index_is_pinned_on_hard_windows(
    ssp, grid, name, age, gap
):
    """The mean option and the Lick index differ by a fixed signed amount on hard windows.

    The mean option is pinned to its numpy definition (1e-9), the numpy-vs-numpy gap to the
    values above (1e-6), and the code-vs-code gap to the same values (1e-4).
    """
    mean_idx = dataclasses.replace(STANDARD_INDICES[name], pseudo_continuum="mean")
    wave, lnu = _solar_spectrum(ssp, grid, age)
    mean_code = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(lnu), mean_idx))
    lick_code = float(
        measure_index_jax(jnp.asarray(wave), jnp.asarray(lnu), STANDARD_INDICES[name])
    )
    mean_ref = mean_hard_exact_reference(wave, lnu, mean_idx)
    lick_ref = lick_hard_reference(wave, _flam(wave, lnu), STANDARD_INDICES[name])
    assert mean_code == pytest.approx(mean_ref, abs=1e-9)
    assert mean_ref - lick_ref == pytest.approx(gap, abs=1e-6)
    assert mean_code - lick_code == pytest.approx(gap, abs=1e-4)


# ── (d) window-LUT path equals the exact path ──────────────────────


def _nonuniform_copy(wave, flux):
    rng = np.random.default_rng(0)
    wave_nu = np.unique(np.concatenate([wave[::2], rng.uniform(wave[0], wave[-1], 700)]))
    flux_nu = np.stack([np.stack([np.interp(wave_nu, wave, f) for f in row]) for row in flux])
    return wave_nu, flux_nu


def _lut_and_exact(wave, flux, idx, dtype):
    """(LUT, exact) values for every (metallicity, age) spectrum of ``flux``."""
    w = jnp.asarray(wave, dtype=dtype)
    f = jnp.asarray(flux, dtype=dtype)
    pc = precompute_index_windows(w, f, [idx])
    # The point terms of a single (metallicity, age) spectrum are its own integrand.
    terms = pc.points.integrands.reshape(-1, pc.points.integrands.shape[-1])
    lut = jax.vmap(lambda t: measure_indices_from_point_terms(t, pc)[0])(terms)
    exact = jax.vmap(lambda s: measure_index_jax(w, s, idx))(f.reshape(-1, f.shape[-1]))
    return np.asarray(lut, dtype=float), np.asarray(exact, dtype=float)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("grid_name", ["native", "nonuniform"])
@pytest.mark.parametrize("name", EW_NAMES)
def test_window_lut_equals_the_exact_path_point_by_point(ssp, grid, name, grid_name, dtype):
    wave, flux = grid
    if grid_name == "nonuniform":
        wave, flux = _nonuniform_copy(wave, flux)
    # float32 runs under ``enable_x64(False)``, the repository's float32 convention: the
    # derivative-safe floors are sized from the working dtype.
    with jax.enable_x64(dtype == jnp.float64):
        lut, exact = _lut_and_exact(wave, flux, STANDARD_INDICES[name], dtype)
    assert np.all(np.isfinite(lut)) and np.all(np.isfinite(exact))
    assert np.max(np.abs(lut - exact)) < LUT_BOUND_AA[dtype]


def test_point_terms_sum_to_the_window_integrals_and_means_cannot_form_a_lick_ew(grid):
    """The per-point terms are the window integrals resolved by grid point.

    A Lick EW is not a function of window means (its feature term is a sum of
    ``F_λ/F_C`` over points), so the means-only entry point refuses it.
    """
    wave, flux = grid
    idx = STANDARD_INDICES["HgF"]
    pc = precompute_index_windows(wave, flux, [idx])
    summed = jax.vmap(
        lambda t: jax.ops.segment_sum(t, pc.points.window, num_segments=pc.window_norms.shape[0]),
        in_axes=(0,),
    )(pc.points.integrands.reshape(-1, pc.points.integrands.shape[-1]))
    np.testing.assert_allclose(
        np.asarray(summed),
        np.asarray(pc.window_integrals).reshape(summed.shape),
        rtol=1e-12,
    )
    means = (pc.window_integrals / pc.window_norms)[0, 0]
    with pytest.raises(ValueError, match="measure_indices_from_point_terms"):
        measure_indices_from_windows(means, pc)


@pytest.mark.parametrize("amplitude", [20.0, 100.0])
def test_window_lut_follows_a_deep_emission_feature(ssp, amplitude):
    """Hα-like emission, EW from about -150 to -750 Å, on a sloped SSP continuum."""
    wave_all = np.asarray(ssp.ssp_wave, dtype=float)
    sel = (wave_all > 6300.0) & (wave_all < 6800.0)
    wave = wave_all[sel]
    cont = np.asarray(ssp.ssp_flux, dtype=float)[::4, ::8][:, :, sel]
    line = amplitude * np.exp(-0.5 * ((wave - 6563.0) / 3.0) ** 2)
    flux = cont * (1.0 + line)
    idx = SpectralIndexDef(
        name="Halpha_EW",
        index_type="EW",
        continuum=((6520.0, 6540.0), (6590.0, 6610.0)),
        feature=(6558.0, 6572.0),
    )
    lut, exact = _lut_and_exact(wave, flux, idx, jnp.float64)
    assert np.min(exact) < -100.0
    assert np.max(np.abs(lut - exact) / np.abs(exact)) < LUT_BOUND_EMISSION_REL


@pytest.mark.parametrize("name", EW_NAMES)
def test_window_lut_with_the_mean_option_is_the_exact_arithmetic(ssp, grid, name):
    """The BAGPIPES arithmetic is a ratio of window means in both paths: parity to rounding."""
    idx = dataclasses.replace(STANDARD_INDICES[name], pseudo_continuum="mean")
    wave, flux = grid
    lut, exact = _lut_and_exact(wave, flux, idx, jnp.float64)
    np.testing.assert_allclose(lut, exact, rtol=1e-9, atol=1e-9)


def test_windows_are_shared_between_the_continuum_options(grid):
    """One window per band serves the Lick, mean and break indices; the geometry is in meta."""
    wave, flux = grid
    lin = STANDARD_INDICES["HgA"]
    mean = dataclasses.replace(lin, pseudo_continuum="mean")
    brk = STANDARD_INDICES["Dn4000"]
    pc = precompute_index_windows(wave, flux, [lin, mean, brk])
    assert pc.window_integrals.shape[-1] == 3 + 2  # HgA blue, red, feature + Dn4000 bands
    assert pc.index_slots[1][2][2] is None
    assert pc.index_slots[0][2][2] == (
        0.5 * sum(lin.continuum[0]),
        0.5 * sum(lin.continuum[1]),
        *lin.feature,
    )


# ── (d') the model-level LUT under dust and a free redshift ────────

#: Model-level LUT vs exact path [Å]. Both paths apply the age-dependent two-component
#: screen at the window grid points and sum the same terms: float64 round-off.
MODEL_LUT_BOUND_AA = 1e-9


@pytest.fixture(scope="module")
def dusty_model(ssp):
    import warnings

    obs = Observation(photometry=Photometry.from_names(["des_g", "des_r"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_diff": Uniform(0.0, 2.0),
                "tau_bc": Uniform(0.0, 2.0),
            },
            neb={"type": "none"},
            redshift=Uniform(0.01, 1.0),
        )


@pytest.mark.parametrize("pseudo_continuum", ["linear", "mean"])
@pytest.mark.parametrize("redshift", [0.1, 0.8])
@pytest.mark.parametrize("tau", [0.0, 0.8, 2.0])
def test_model_window_lut_follows_the_exact_path_under_dust_and_a_free_redshift(
    dusty_model, tau, redshift, pseudo_continuum
):
    defs = tuple(
        dataclasses.replace(STANDARD_INDICES[n], pseudo_continuum=pseudo_continuum)
        for n in EW_NAMES
    )
    params = dict(dusty_model.spec.sample(jax.random.PRNGKey(3)))
    params["redshift"] = jnp.asarray(redshift)
    params["dust_tau_diff"] = jnp.asarray(tau)
    params["dust_tau_bc"] = jnp.asarray(1.5 * tau)
    exact = np.asarray(dusty_model.predict_spectral_indices(params, defs, approx=False))
    fast = np.asarray(dusty_model.predict_spectral_indices(params, defs, approx=True))
    assert np.all(np.isfinite(exact)) and np.all(np.isfinite(fast))
    bound = MODEL_LUT_BOUND_AA
    assert np.max(np.abs(fast - exact)) < bound, dict(zip(EW_NAMES, np.abs(fast - exact)))


def test_model_ew_is_the_lick_definition_on_the_model_spectrum(dusty_model):
    """predict_spectral_indices measures the Lick EW of the model's own L_ν spectrum."""
    params = dict(dusty_model.spec.sample(jax.random.PRNGKey(5)))
    defs = tuple(STANDARD_INDICES[n] for n in EW_NAMES)
    rest = dusty_model.predict_rest_sed(params)
    wave, lnu = np.asarray(rest.wavelength, float), np.asarray(rest.sed, float)
    got = np.asarray(dusty_model.predict_spectral_indices(params, defs))
    ref = np.array([lick_hard_reference(wave, _flam(wave, lnu), d) for d in defs])
    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=HARD_TOL_AA)


# ── (e) the mean option reproduces BAGPIPES ────────────────────────

#: |tengri "mean" - BAGPIPES| over the six cells below was at most 0.157 Å (HβA 3 Gyr):
#: tengri's 1 Å sigmoid window edges and trapezoid means against BAGPIPES' hard masks
#: and per-pixel means. Tolerance 1.25 x that, rounded up.
BAGPIPES_MEAN_TOL_AA = 0.2

#: ``bagpipes.input.spectral_indices.single_index`` (bagpipes 1.3.5) on the
#: solar SSP F_λ spectrum, native grid 3600-5600 Å, hard masks.
BAGPIPES_LITERALS = {
    ("HgA", 1.0): 5.709245195280442,
    ("HgA", 10.0): -5.561686797639739,
    ("Fe4383", 1.0): 0.8600633274933599,
    ("Fe4383", 10.0): 5.91495987399853,
    ("Fe5270", 10.0): 3.4764181995347703,
    ("Hbeta", 3.0): 2.758479482114802,
}


def _bagpipes_index(name):
    """The index as BAGPIPES defines it: the published (air) window numbers, verbatim.

    BAGPIPES applies them to its vacuum wavelengths unconverted. These cells test the
    *arithmetic* against BAGPIPES, so they use its windows, not tengri's vacuum ones.
    """
    continuum, feature = _LICK_AIR_WINDOWS[name]
    return dataclasses.replace(
        STANDARD_INDICES[name], pseudo_continuum="mean", continuum=continuum, feature=feature
    )


def _bagpipes_dict(idx):
    return {"type": "EW", "continuum": list(idx.continuum), "feature": list(idx.feature)}


@pytest.mark.parametrize("name,age", list(BAGPIPES_LITERALS))
def test_mean_option_equals_bagpipes_single_index_pinned(ssp, grid, name, age):
    """Literal twin of the BAGPIPES cell (no BAGPIPES needed) on a sloped SSP F_λ spectrum."""
    idx = _bagpipes_index(name)
    wave, lnu = _solar_spectrum(ssp, grid, age)
    flam = _flam(wave, lnu)
    got = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(flam), idx))
    assert got == pytest.approx(BAGPIPES_LITERALS[(name, age)], abs=BAGPIPES_MEAN_TOL_AA)
    assert mean_hard_reference(wave, flam, idx) == pytest.approx(
        BAGPIPES_LITERALS[(name, age)], abs=1e-9
    )


@pytest.mark.parametrize("name,age", list(BAGPIPES_LITERALS))
def test_mean_option_equals_bagpipes_single_index(ssp, grid, name, age):
    bp = pytest.importorskip("bagpipes.input.spectral_indices")
    idx = _bagpipes_index(name)
    wave, lnu = _solar_spectrum(ssp, grid, age)
    flam = _flam(wave, lnu)
    ref = float(bp.single_index(_bagpipes_dict(idx), np.column_stack([wave, flam]), 0.0))
    got = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(flam), idx))
    assert got == pytest.approx(ref, abs=BAGPIPES_MEAN_TOL_AA)
    assert ref == pytest.approx(BAGPIPES_LITERALS[(name, age)], abs=1e-9)


# ── (f) break indices and the slope are unchanged ──────────────────


@pytest.mark.parametrize("name", BREAK_NAMES)
@pytest.mark.parametrize("age", AGES_GYR)
def test_break_indices_are_the_f_nu_window_mean_ratio(ssp, grid, name, age):
    idx = STANDARD_INDICES[name]
    wave, lnu = _solar_spectrum(ssp, grid, age)
    (b0, b1), (r0, r1) = idx.continuum
    ref = _hard_mean(wave, lnu, r0, r1) / _hard_mean(wave, lnu, b0, b1)
    got = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(lnu), idx))
    assert got == pytest.approx(ref, rel=1e-9)
    soft_ref = _soft_mean(wave, lnu, r0, r1) / _soft_mean(wave, lnu, b0, b1)
    soft = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(lnu), idx, edge_width=1.0))
    assert soft == pytest.approx(soft_ref, rel=1e-12)
    other = dataclasses.replace(idx, pseudo_continuum="mean")
    assert float(measure_index_jax(jnp.asarray(wave), jnp.asarray(lnu), other)) == got


@pytest.mark.parametrize("name", SLOPE_NAMES)
@pytest.mark.parametrize("age", AGES_GYR)
@pytest.mark.parametrize("edge_width", [0.0, 1.0], ids=["hard", "soft_option"])
def test_uv_slope_is_the_weighted_log_log_fit_of_f_nu(ssp, name, age, edge_width):
    idx = STANDARD_INDICES[name]
    wave_all = np.asarray(ssp.ssp_wave, dtype=float)
    sel = (wave_all > 1000.0) & (wave_all < 3000.0)
    wave = wave_all[sel]
    i_met = int(np.argmin(np.abs(np.asarray(ssp.ssp_lgmet) - SOLAR_LGMET)))
    i_age = int(np.argmin(np.abs(np.asarray(ssp.ssp_lg_age_gyr) - np.log10(age))))
    lnu = np.asarray(ssp.ssp_flux, dtype=float)[i_met, i_age][sel]
    lo, hi = idx.feature
    w = (wave >= lo) * (wave <= hi) * 1.0 if edge_width == 0.0 else _soft_weights(wave, lo, hi)
    x, y = np.log(wave), np.log(np.maximum(lnu, 1e-50))
    sw, sx, sy = w.sum(), (w * x).sum(), (w * y).sum()
    slope = ((w * x * y).sum() - sx * sy / sw) / ((w * x * x).sum() - sx**2 / sw)
    got = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(lnu), idx, edge_width=edge_width))
    assert got == pytest.approx(slope - 2.0, rel=1e-9, abs=1e-9)


# ── (g) gradient and float32 ───────────────────────────────────────


def _tilted(wave, base, slope):
    return base * (1.0 + slope * (wave - 4400.0) / 1000.0)


@pytest.mark.parametrize("name", EW_NAMES)
def test_ew_gradient_wrt_a_continuum_slope_matches_finite_differences(ssp, grid, name):
    idx = STANDARD_INDICES[name]
    wave, base = _solar_spectrum(ssp, grid, 3.0)
    w, b = jnp.asarray(wave), jnp.asarray(base)

    def ew(s):
        return measure_index_jax(w, _tilted(w, b, s), idx)

    s0, h = 0.3, 1e-5
    grad = float(jax.grad(ew)(s0))
    fd = float((ew(s0 + h) - ew(s0 - h)) / (2.0 * h))
    assert np.isfinite(grad) and grad != 0.0
    assert grad == pytest.approx(fd, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("name", EW_NAMES)
def test_float32_values_and_gradient_are_finite_and_close_to_float64(ssp, grid, name):
    idx = STANDARD_INDICES[name]
    wave, base = _solar_spectrum(ssp, grid, 3.0)
    value64 = measure_index_jax(
        jnp.asarray(wave), _tilted(jnp.asarray(wave), jnp.asarray(base), 0.3), idx
    )
    with jax.enable_x64(False):
        w32, b32 = jnp.asarray(wave, dtype=jnp.float32), jnp.asarray(base, dtype=jnp.float32)
        value32 = measure_index_jax(w32, _tilted(w32, b32, jnp.float32(0.3)), idx)
        grad32 = jax.grad(lambda s: measure_index_jax(w32, _tilted(w32, b32, s), idx))(
            jnp.float32(0.3)
        )
        assert value32.dtype == jnp.float32
        assert np.isfinite(float(value32)) and np.isfinite(float(grad32)) and float(grad32) != 0.0
        assert float(value32) == pytest.approx(float(value64), abs=FLOAT32_SLACK_AA)


@pytest.mark.parametrize("scale", [1.0, 1e30], ids=["per_msun", "physical_scale"])
@pytest.mark.parametrize("name", EW_NAMES)
def test_float32_gradient_is_finite_on_the_full_ssp_grid(ssp, name, scale):
    """Whole 91 Å-10 µm grid, L_ν at SSP and at galaxy scale: the straight line is not
    extrapolated to pixels the window cannot see (a zero crossing there would floor F_C
    and overflow the VJP's flux / floor**2 in float32)."""
    idx = STANDARD_INDICES[name]
    wave = np.asarray(ssp.ssp_wave, dtype=float)
    i_met = int(np.argmin(np.abs(np.asarray(ssp.ssp_lgmet) - SOLAR_LGMET)))
    i_age = int(np.argmin(np.abs(np.asarray(ssp.ssp_lg_age_gyr) - np.log10(3.0))))
    lnu = np.asarray(ssp.ssp_flux, dtype=float)[i_met, i_age] * scale
    with jax.enable_x64(False):
        w32, f32 = jnp.asarray(wave, dtype=jnp.float32), jnp.asarray(lnu, dtype=jnp.float32)

        def ew(s):
            return measure_index_jax(w32, _tilted(w32, f32, s), idx)

        assert np.isfinite(float(ew(jnp.float32(0.3))))
        assert np.isfinite(float(jax.grad(ew)(jnp.float32(0.3))))
        grad_flux = jax.grad(lambda f: measure_index_jax(w32, f, idx))(f32)
        assert bool(jnp.all(jnp.isfinite(grad_flux))) and bool(jnp.any(grad_flux != 0.0))
