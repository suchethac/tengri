# SPDX-License-Identifier: BSD-3-Clause
"""The Cue ionizing-parameter table covers old ages, exactly in Q_H, and caches per age cutoff.

``precompute_ionizing_params_table`` fitted only bins younger than 100 Myr and wrote -99
for the rest. With every age counted, old bins need a Q_H and a shape too. A fine age grid
(BC03 from CIGALE: 13700 ages, 13600 of them old) cannot afford a scipy fit per bin
(~82k fits, 7-14 min), so old bins are fitted on a thinned, log-spaced subset and take the
nearest fitted shape, while their Q_H is the exact quadrature of their own spectrum.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.lyc import edge_trapezoid
from tengri.components.nebular import ionizing_spectrum as ion
from tengri.utils.physics_constants import C_AA, H_PLANCK, L_SUN

pytestmark = pytest.mark.regression_bug

_N_AGE = 600
_LG_AGE = np.linspace(5.0, 10.1, _N_AGE)  # log10(age/yr)
_LG_MET = np.array([-3.0, -2.0, -1.5])


def _grid(n_wave: int = 700):
    """Blackbody-like SSP grid: young bins hot, old bins cool, a metallicity tilt."""
    wave = np.logspace(2.0, 4.0, n_wave)
    nu = C_AA / wave
    temperature = 1.0e5 - 7.0e4 * (_LG_AGE - _LG_AGE[0]) / (_LG_AGE[-1] - _LG_AGE[0])
    x = (H_PLANCK / 1.380649e-16) * nu[None, :] / temperature[:, None]
    shape = nu[None, :] ** 3 / np.expm1(np.clip(x, 1e-6, 600.0))
    scale = 1e-26 / shape.max()
    flux = np.stack([scale * shape * (1.0 + 0.1 * z) for z in _LG_MET])
    return wave, flux, _LG_MET.copy()


def _exact_log_q(wave, flux):
    """``log10 Q_H`` per (met, age) from the same edge-aware quadrature the fit uses."""
    integrand = (flux * L_SUN) / (H_PLANCK * (C_AA / wave))
    q = np.asarray(
        edge_trapezoid(integrand, wave, variable="nu", side="ionizing", edge_aa=ion.HI_LIMIT)
    )
    return np.log10(np.maximum(q, 1e-99))


@pytest.fixture(autouse=True)
def _isolated_caches(tmp_path, monkeypatch):
    """Private disk cache and an empty in-process memo, so each test fits for itself."""
    monkeypatch.setenv("TENGRI_JAX_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(ion, "_IONSPEC_TABLE_CACHE", {})


def test_old_bins_get_the_exact_q_h_and_a_finite_shape():
    wave, flux, lgmet = _grid()
    table = ion.precompute_ionizing_params_table(wave, flux, lgmet, ssp_log_age_yr=_LG_AGE)
    exact = _exact_log_q(wave, flux)
    old = _LG_AGE > ion.MAX_NEB_LOG_AGE
    assert old.sum() > 200
    np.testing.assert_allclose(table["logqion_table"][:, old], exact[:, old], atol=1e-9)
    assert np.all(np.isfinite(table["ionspec_table"][:, old]))
    # the shape tracks the fitted neighbour: every old bin has a finite segment luminosity
    assert np.all(table["seglum_table"][:, old] > -99.0)


def test_old_bin_segment_luminosities_follow_the_exact_q_h():
    """A copied shape keeps the donor's segment ratios and takes the bin's own Q_H level."""
    wave, flux, lgmet = _grid()
    table = ion.precompute_ionizing_params_table(wave, flux, lgmet, ssp_log_age_yr=_LG_AGE)
    seg = table["seglum_table"]
    q = table["logqion_table"]
    old = np.where(_LG_AGE > ion.MAX_NEB_LOG_AGE)[0]
    assert np.all(seg[:, old, :] > -99.0)
    offsets = seg[:, old, :] - q[:, old, None]  # per-segment luminosity relative to Q_H
    # adjacent old bins share a donor and so a shape: offsets change slowly with age
    assert np.max(np.abs(np.diff(offsets, axis=1))) < 0.5


def test_old_bins_are_fitted_on_a_thinned_subset(monkeypatch):
    wave, flux, lgmet = _grid()
    calls = []
    real = ion.fit_ionizing_spectrum

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ion, "fit_ionizing_spectrum", counting)
    ion.precompute_ionizing_params_table(wave, flux, lgmet, ssp_log_age_yr=_LG_AGE)
    n_young = int((_LG_AGE <= ion.MAX_NEB_LOG_AGE).sum())
    n_old = _N_AGE - n_young
    assert n_old > ion.MAX_OLD_FITS_PER_METALLICITY
    expected = (n_young + ion.MAX_OLD_FITS_PER_METALLICITY) * len(lgmet)
    assert len(calls) == expected, "every young bin and a thinned subset of old bins are fitted"


def test_a_coarse_old_grid_is_fitted_bin_by_bin(monkeypatch):
    """Below the thinning threshold every old bin keeps its own fit (FSPS-sized grids)."""
    wave, flux, lgmet = _grid()
    keep = np.r_[np.where(_LG_AGE <= 8.0)[0][::10], np.where(_LG_AGE > 8.0)[0][::20]]
    n_old = int((_LG_AGE[keep] > ion.MAX_NEB_LOG_AGE).sum())
    assert 0 < n_old <= ion.MAX_OLD_FITS_PER_METALLICITY
    calls = []
    real = ion.fit_ionizing_spectrum
    monkeypatch.setattr(
        ion, "fit_ionizing_spectrum", lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    )
    ion.precompute_ionizing_params_table(wave, flux[:, keep], lgmet, ssp_log_age_yr=_LG_AGE[keep])
    assert len(calls) == len(keep) * len(lgmet)


def test_age_cutoff_restores_the_young_only_table():
    """``max_log_age=8`` leaves old bins at the -99 / zero 'unfitted' convention."""
    wave, flux, lgmet = _grid()
    table = ion.precompute_ionizing_params_table(
        wave, flux, lgmet, ssp_log_age_yr=_LG_AGE, max_log_age=8.0
    )
    old = _LG_AGE > 8.0
    assert np.all(table["logqion_table"][:, old] == -99.0)
    assert np.all(table["seglum_table"][:, old] == -99.0)
    assert np.all(table["ionspec_table"][:, old] == 0.0)
    assert np.all(table["logqion_table"][:, ~old] > 0.0)


def test_cutoff_and_age_axis_are_part_of_the_cache_key():
    wave, flux, lgmet = _grid()
    keys = {
        ion._ssp_fingerprint(wave, flux, lgmet, ssp_log_age_yr=_LG_AGE, max_log_age=cut)
        for cut in (None, 8.0, 9.0)
    }
    assert len(keys) == 3
    shifted = ion._ssp_fingerprint(wave, flux, lgmet, ssp_log_age_yr=_LG_AGE + 0.01)
    assert shifted != ion._ssp_fingerprint(wave, flux, lgmet, ssp_log_age_yr=_LG_AGE)
    hashes = {ion._fingerprint_hash(k) for k in keys}
    assert len(hashes) == 3


def test_cache_version_was_bumped_past_the_young_only_tables():
    """Version 3 wrote young-only tables; a version-3 file must never be served again."""
    assert ion._IONSPEC_CACHE_VERSION >= 4


def test_disk_cache_serves_each_cutoff_its_own_table(tmp_path):
    wave, flux, lgmet = _grid()
    full = ion.precompute_ionizing_params_table(wave, flux, lgmet, ssp_log_age_yr=_LG_AGE)
    cut = ion.precompute_ionizing_params_table(
        wave, flux, lgmet, ssp_log_age_yr=_LG_AGE, max_log_age=8.0
    )
    assert len(list((tmp_path / "ionspec_tables").glob("*.npz"))) == 2
    # a fresh process: empty memo, tables come back from disk, each with its own content
    ion._IONSPEC_TABLE_CACHE.clear()
    full_again = ion.precompute_ionizing_params_table(wave, flux, lgmet, ssp_log_age_yr=_LG_AGE)
    cut_again = ion.precompute_ionizing_params_table(
        wave, flux, lgmet, ssp_log_age_yr=_LG_AGE, max_log_age=8.0
    )
    np.testing.assert_array_equal(full_again["logqion_table"], full["logqion_table"])
    np.testing.assert_array_equal(cut_again["logqion_table"], cut["logqion_table"])
    assert not np.array_equal(full["logqion_table"], cut["logqion_table"])


def test_an_old_ionizing_flux_free_bin_stays_unfitted():
    """Bins with no ionizing flux keep the -99 convention (nothing to count, nothing to fit)."""
    wave, flux, lgmet = _grid()
    flux = flux.copy()
    flux[:, -5:, wave < ion.HI_LIMIT] = 0.0
    table = ion.precompute_ionizing_params_table(wave, flux, lgmet, ssp_log_age_yr=_LG_AGE)
    assert np.all(table["logqion_table"][:, -5:] == -99.0)
    assert np.all(np.isfinite(table["logqion_table"]))


def test_old_bins_below_the_photon_floor_are_left_unfitted():
    """An old bin with Q_H below ``MIN_COUNTED_OLD_LOG_QION`` is treated as having no continuum."""
    wave, flux, lgmet = _grid()
    flux = flux.copy()
    faint = (_LG_AGE > 9.5) & (_LG_AGE < 9.7)
    exact = _exact_log_q(wave, flux)
    flux[:, faint, :] *= 10.0 ** (ion.MIN_COUNTED_OLD_LOG_QION - 4.0 - exact[:, faint].max())
    table = ion.precompute_ionizing_params_table(wave, flux, lgmet, ssp_log_age_yr=_LG_AGE)
    assert np.all(table["logqion_table"][:, faint] == -99.0)
    assert np.all(table["ionspec_table"][:, faint] == 0.0)
    assert np.all(table["logqion_table"][:, _LG_AGE > 9.7] > ion.MIN_COUNTED_OLD_LOG_QION)
