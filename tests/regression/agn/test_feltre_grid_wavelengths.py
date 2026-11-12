# SPDX-License-Identifier: BSD-3-Clause
r"""The Feltre+2016 grid stores vacuum line wavelengths, converted once at build time (#3180).

The NEOGAL tables follow Cloudy's convention: vacuum below 2000 Angstrom, air at
and above (Ferland et al. 2017, sec. 2.3.3). ``scripts/build_feltre_grid.py``
converts the optical labels with :func:`tengri.utils.air_vacuum.air_to_vac` and
records the convention in the file. The loader does not convert again.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

ROOT = Path(__file__).resolve().parents[3]
H5_PATH = ROOT / "data" / "feltre_grid.h5"
BUILDER_PATH = ROOT / "scripts" / "build_feltre_grid.py"
INPUT_DIR = ROOT / "data" / "neogal" / "AGN_NLR_nebular_feltre16"

#: Vacuum wavelengths [Angstrom] of the optical lines (NIST air value through air_to_vac).
_OPTICAL_VAC_AA = {
    "Hbeta": (4862.68, 0.02),
    "Halpha": (6564.61, 0.02),
    "[OIII]4959": (4960.30, 0.02),
    "[OIII]5007": (5008.24, 0.02),
    "[OI]6300": (6302.05, 0.02),
    "[NII]6548": (6549.86, 0.02),
    "[NII]6584": (6585.27, 0.02),
    "[SII]6717": (6718.29, 0.02),
    "[SII]6731": (6732.67, 0.02),
    # [O II] 3726.032 air (the 3726 doublet component) -> vacuum 3727.09.
    "[OII]3727": (3727.09, 0.05),
}

#: NIST vacuum wavelengths of the UV labels, which the table already quotes in vacuum.
_UV_VAC_AA = {
    "NV1240": (1240.81, 0.05),  # doublet centroid of 1238.82 and 1242.80
    "CIV1548": (1548.19, 0.05),
    "CIV1551": (1550.77, 0.05),
    "HeII1640": (1640.42, 0.05),
    "OIII]1661": (1660.81, 0.05),
    "OIII]1666": (1666.15, 0.05),
    "[SiIII]1883": (1882.71, 0.05),
    "SiIII]1888": (1892.03, 0.05),
    "[CIII]1907": (1906.68, 0.05),
    "CIII]1910": (1908.73, 0.05),
}


def _stored_lines() -> dict[str, float]:
    with h5py.File(H5_PATH, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["feltre/line_names"][:]]
        waves = f["feltre/line_wavelengths_aa"][:]
    return dict(zip(names, (float(w) for w in waves), strict=True))


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_feltre_grid", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_optical_labels_are_vacuum():
    stored = _stored_lines()
    for name, (expected, tol) in _OPTICAL_VAC_AA.items():
        assert abs(stored[name] - expected) < tol, (name, stored[name], expected)


def test_uv_labels_unchanged_vacuum():
    stored = _stored_lines()
    for name, (expected, tol) in _UV_VAC_AA.items():
        assert abs(stored[name] - expected) < tol, (name, stored[name], expected)


def test_h5_records_conversion_attr():
    with h5py.File(H5_PATH, "r") as f:
        convention = str(f.attrs["wavelength_convention"])
    assert "air_to_vac" in convention
    assert "vacuum" in convention


def test_rebuild_matches_tracked(tmp_path):
    builder = _load_builder()
    out = tmp_path / "feltre_grid.h5"
    builder.build(INPUT_DIR, out)
    with h5py.File(H5_PATH, "r") as tracked, h5py.File(out, "r") as rebuilt:
        keys = sorted(tracked["feltre"].keys())
        assert keys == sorted(rebuilt["feltre"].keys())
        for key in keys:
            a = tracked["feltre"][key][:]
            b = rebuilt["feltre"][key][:]
            if a.dtype.kind in "OSU":
                a = np.array([x.decode() if isinstance(x, bytes) else x for x in a])
                b = np.array([x.decode() if isinstance(x, bytes) else x for x in b])
            if a.dtype.kind == "f":
                # The builder's Q_H uses Python float ``**`` (the platform libm ``pow``), which
                # differs between macOS and glibc in the last bit; a few ulp is the same table.
                np.testing.assert_allclose(b, a, rtol=1e-13, atol=0.0, equal_nan=True, err_msg=key)
            else:
                assert np.array_equal(a, b), key
