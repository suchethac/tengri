# SPDX-License-Identifier: BSD-3-Clause
r"""The four wavelength conventions still unestablished after #2778, settled from primary sources.

Every wavelength inside tengri is vacuum; an air input converts once, at ingestion.

1. Synthesizer NLR ``spectra/wavelength`` -> VACUUM. synthesizer-grids writes it from Cloudy's
   ``save last continuum ... units Angstroms`` (``cloudy17.py`` / ``cloudy23.py`` l.370 / 451,
   ``read_wavelength`` = column 0 of ``.cont``); Cloudy computes that column as ``RYDLAM / E``
   (``energy.h``: "all remaining methods return vacuum wavelengths"; ``service.cpp AnuUnit``).
   Independent check on the shipped file: the mesh is geometric in energy, so ln(lambda) is
   linear in the pixel index in vacuum, whereas the air refractive index would bend it by
   -7.5e-6 (4000-5000 A) and -4.3e-6 (5000-6500 A); the file shows -0.2e-6 and -0.3e-6.
2. qsogen emission-line template above 5100 A -> VACUUM. Glikman+2006 reduced its IRTF/SpeX
   spectra with Spextool, whose wavelengths "are given in vacuum" (Cushing+2004, PASP 116, 362,
   Sec. 4.2.6); qsogen (MJTemple/qsogen) applies no air/vacuum conversion.
3. SVO filter curves -> space missions VACUUM (registered, per facility); ground facilities stay
   unestablished (no source states the medium).
4. Cloudy line labels beyond 100 micron -> AIR. ``t_wavl::sprt_wl`` prints
   ``wl / p_RefIndex(1e8/wl)`` for every positive wavelength and ``p_RefIndex`` applies the
   Peck & Reeder (1972) index wherever the wavenumber is below 5e4 cm^-1, with no upper limit.
   There is no 100 micron cap.

At the parent commit the registry holds the first two as "unestablished", has no entry for the
space filters, and the Cloudy label window stops at 100 micron.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from tengri.components.nebular import _line_ingest
from tengri.components.nebular._line_ingest import catalog_air_to_vacuum
from tengri.utils import wavelength_conventions as wc
from tengri.utils.air_vacuum import air_to_vac

pytestmark = pytest.mark.regression_bug

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
_BY_NAME = {s.name: s for s in wc.SOURCES}


def _nlr_grid_file() -> Path | None:
    candidates = []
    env = os.environ.get("TENGRI_SYNTHESIZER_AGN_GRID_DIR")
    if env:
        candidates.append(Path(env))
    candidates += [
        DATA / "synthesizer_grids",
        Path.home() / "Library/Application Support/Synthesizer/grids",
    ]
    for base in candidates:
        path = base / "test_grid_agn-nlr.hdf5"
        if path.is_file():
            return path
    return None


# ── 1. Synthesizer NLR spectra axis ─────────────────────────────────────────────────────────


def test_registry_nlr_spectra_axis_is_vacuum_with_the_cloudy_evidence():
    s = _BY_NAME["synthesizer_nlr_spectra"]
    assert s.convention == "vacuum"
    assert s.reason == ""
    for needle in ("units Angstroms", "RYDLAM", "energy.h"):
        assert needle in s.evidence, needle


def _curvature_residual(w: np.ndarray, lo: float, hi: float) -> float:
    """Mean departure of ln(lambda) from the geometric mesh fitted on 9000-30000 A."""
    i = np.arange(w.size)
    lw = np.log(w)
    ref = (w > 9000.0) & (w < 30000.0)
    slope, icpt = np.polyfit(i[ref], lw[ref], 1)
    sel = (w > lo) & (w < hi)
    return float((lw - (icpt + slope * i))[sel].mean())


def test_nlr_spectra_mesh_is_geometric_in_vacuum_and_rejects_the_air_curvature():
    path = _nlr_grid_file()
    if path is None:
        pytest.skip("Synthesizer test_grid_agn-nlr.hdf5 not present")
    with h5py.File(path) as f:
        w = f["spectra/wavelength"][:].astype(np.float64)
    ln_n = lambda lam: np.log(lam / np.asarray(wc_vac_to_air(lam)))  # noqa: E731
    ref = (w > 9000.0) & (w < 30000.0)
    n_ref = np.median(ln_n(w[ref]))
    for lo, hi, observed_max in ((4000.0, 5000.0, 2.0e-6), (5000.0, 6500.0, 1.5e-6)):
        sel = (w > lo) & (w < hi)
        air_prediction = float((-(ln_n(w[sel]) - n_ref)).mean())
        observed = _curvature_residual(w, lo, hi)
        assert abs(observed) < observed_max, (lo, hi, observed)  # vacuum: no curvature
        assert abs(air_prediction) > 2.0 * observed_max, (lo, hi, air_prediction)  # air bends it


def wc_vac_to_air(lam):
    from tengri.utils.air_vacuum import vac_to_air

    return vac_to_air(lam)


def test_nlr_spectra_wavelengths_reach_the_grid_unconverted():
    path = _nlr_grid_file()
    if path is None:
        pytest.skip("Synthesizer test_grid_agn-nlr.hdf5 not present")
    from tengri.components.nebular.agn_nebular import _load_synthesizer_nlr_grid

    with h5py.File(path) as f:
        raw = f["spectra/wavelength"][:]
    got = np.asarray(_load_synthesizer_nlr_grid(path).spectra_wavelengths_aa)
    np.testing.assert_allclose(got, raw, rtol=2e-7, atol=0.0)  # float32 on device


# ── 2. qsogen above 5100 A ──────────────────────────────────────────────────────────────────


def test_registry_qsogen_nir_is_vacuum_with_the_spextool_evidence():
    s = _BY_NAME["qsogen_emlines_nir"]
    assert s.convention == "vacuum"
    assert s.reason == ""
    for needle in ("Spextool", "Cushing", "4.2.6", "Glikman"):
        assert needle in s.evidence, needle


def test_qsogen_wavelengths_stay_unconverted_in_both_registered_parts():
    for name in ("qsogen_emlines", "qsogen_emlines_nir"):
        assert _BY_NAME[name].convention == "vacuum"
        assert _BY_NAME[name].path == "data/qsogen_emline_template.dat"


# ── 3. SVO filters: space missions vacuum, ground families still open ─────────────────────


def test_space_filter_facilities_are_registered_vacuum():
    s = _BY_NAME["svo_filters_space"]
    assert s.convention == "vacuum"
    assert s.path == "data/filters"
    assert "space" in s.evidence.lower()
    for fac in ("JWST", "HST", "Euclid", "Spitzer", "WISE", "GALEX"):
        assert wc.filter_wavelength_convention(f"{fac}_X_Y") == "vacuum", fac


def test_ground_filter_facilities_stay_unestablished_with_what_was_checked():
    s = _BY_NAME["svo_filters"]
    assert s.convention == "unestablished"
    for needle in ("Doi", "Hg+Ar", "LSST", "Cohen", "HSC"):
        assert needle in s.reason + s.evidence, needle
    for fac in ("SLOAN", "CTIO", "Subaru", "LSST", "2MASS", "CFHT", "Paranal", "OAJ"):
        assert wc.filter_wavelength_convention(f"{fac}_X_Y") == "unestablished", fac


def test_every_shipped_svo_curve_has_a_declared_facility_class():
    filters = DATA / "filters"
    if not filters.is_dir():
        pytest.skip("data/filters not present")
    files = sorted(filters.glob("*.dat"))
    assert files
    declared = set(wc.SPACE_FILTER_FACILITIES) | set(wc.GROUND_FILTER_FACILITIES)
    assert not set(wc.SPACE_FILTER_FACILITIES) & set(wc.GROUND_FILTER_FACILITIES)
    for p in files:
        assert p.name.split("_")[0] in declared, p.name


# ── 4. Cloudy labels beyond 100 micron ──────────────────────────────────────────────────────

#: (Cloudy air label, vacuum value) [Angstrom]; vacuum = air * (1 + 2.726e-4) at these wavelengths.
_FAR_IR_AIR = (1217690.0, 1454950.0, 1576360.0, 2052830.0, 3702690.0, 6095900.0)


def test_cloudy_label_window_has_no_upper_limit():
    assert _line_ingest.CLOUDY_LABEL_AIR_MAX_AA > 1.0e9


def test_far_infrared_cloudy_labels_convert_with_the_constant_long_wave_index():
    air = np.array(_FAR_IR_AIR)
    got = catalog_air_to_vacuum(air, max_air_aa=_line_ingest.CLOUDY_LABEL_AIR_MAX_AA)
    np.testing.assert_allclose(got, air_to_vac(air), rtol=0, atol=1e-6)
    # Peck & Reeder (1972) as coded in Cloudy: n-1 -> 2.7260e-4 for lambda >> 10 micron
    assert np.all(np.abs(got / air - 1.0 - 2.726e-4) < 3e-7)


def test_cue_far_infrared_labels_are_ingested_as_vacuum():
    path = DATA / "cue_weights.npz"
    if not path.is_file():
        pytest.skip("cue_weights.npz not present")
    from tengri.components.nebular.cue import load_cue_weights

    raw = np.load(path)["nn_line_wavelength"]
    waves = np.asarray(load_cue_weights(str(path)).nn_line_wav)
    far = raw > 1.0e6
    assert far.sum() == len(_FAR_IR_AIR)
    np.testing.assert_allclose(waves[far], air_to_vac(raw[far]), rtol=0, atol=1e-6)
    assert np.all(waves[far] / raw[far] - 1.0 > 2.7e-4)
