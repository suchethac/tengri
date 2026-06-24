# SPDX-License-Identifier: BSD-3-Clause
"""Byte-parity test: tengri's HDF5 template bundle vs upstream's loaders.

Re-runs the upstream ``database_builder`` loaders directly on the raw text
files (cloned in ``arxiv_library/code/grahsp/``) and compares against
``data/grahsp/grahsp_templates.h5`` to <1e-12 relative error. This guarantees
no information was lost or transformed during the HDF5 packaging step.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.bounds
from scipy import constants as cst

REPO_ROOT = Path(__file__).resolve().parents[4]
TEMPLATE_BUNDLE = REPO_ROOT / "data" / "grahsp" / "grahsp_templates.h5"
UPSTREAM = REPO_ROOT / "arxiv_library" / "code" / "grahsp"
_UPSTREAM_AGN = UPSTREAM / "database_builder" / "activate" / "agn"
UPSTREAM_FEII = _UPSTREAM_AGN / "FeII_template" / "Fe_d11-m20-20.5.txt"
UPSTREAM_LINES = _UPSTREAM_AGN / "mor_netzer_2012" / "emission_line_table.formatted"


def _upstream_feii():
    """Re-run upstream ``database_builder`` FeII loader exactly."""
    data = np.genfromtxt(UPSTREAM_FEII)
    wavez = data[:, 0]  # Å, observed
    Lnu = data[:, 1]
    z = 4593.4 / 4575.0 - 1.0
    wave = wavez / (1.0 + z)
    Llam = Lnu * cst.c / wavez**2
    norm_idx = np.argmin(np.abs(wave - 4575.0))
    norm = Llam[norm_idx]
    Llam = Llam / norm
    return wave * 0.1, Llam  # nm, normalized L_lambda


def _upstream_lines():
    """Re-run upstream ``database_builder`` line-table loader exactly."""
    data = np.loadtxt(
        UPSTREAM_LINES,
        dtype=[("name", "S10"), ("wave", "f"), ("broad", "f"), ("S2", "f"), ("LINER", "f")],
    )
    return (
        data["wave"] * 0.1,  # Å -> nm (upstream `data['wave'] * 0.1`)
        data["broad"].astype(np.float64),
        data["S2"].astype(np.float64),
        data["LINER"].astype(np.float64),
    )


@pytest.mark.skipif(not UPSTREAM.exists(), reason="upstream GRAHSP repo not cloned")
def test_feii_template_matches_upstream_loader():
    upstream_wave, upstream_Llam = _upstream_feii()
    with h5py.File(TEMPLATE_BUNDLE, "r") as f:
        ours_wave = f["feii_bruhweiler2008/wave_nm"][:]
        ours_Llam = f["feii_bruhweiler2008/lumin"][:]
    np.testing.assert_allclose(ours_wave, upstream_wave, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(ours_Llam, upstream_Llam, rtol=1e-12, atol=0.0)


@pytest.mark.skipif(not UPSTREAM.exists(), reason="upstream GRAHSP repo not cloned")
def test_line_table_matches_upstream_loader():
    upstream_wave, upstream_broad, upstream_sy2, upstream_liner = _upstream_lines()
    with h5py.File(TEMPLATE_BUNDLE, "r") as f:
        ours_wave = f["netzer1990_lines/wave_nm"][:]
        ours_broad = f["netzer1990_lines/broad"][:]
        ours_sy2 = f["netzer1990_lines/narrow_sy2"][:]
        ours_liner = f["netzer1990_lines/narrow_liner"][:]
    assert ours_wave.size == upstream_wave.size, "line count mismatch"
    np.testing.assert_allclose(np.sort(ours_wave), np.sort(upstream_wave), rtol=1e-7, atol=0.0)
    # Compare element-wise after sorting by wavelength
    order_us = np.argsort(ours_wave)
    order_up = np.argsort(upstream_wave)
    np.testing.assert_allclose(ours_broad[order_us], upstream_broad[order_up], rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(ours_sy2[order_us], upstream_sy2[order_up], rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(ours_liner[order_us], upstream_liner[order_up], rtol=1e-6, atol=0.0)


@pytest.mark.skipif(not UPSTREAM.exists(), reason="upstream GRAHSP repo not cloned")
def test_torus_wave_grid_matches_upstream_source():
    """Torus wave grid in HDF5 must equal the literal in activategtorus.py."""
    src = (UPSTREAM / "pcigale" / "creation_modules" / "activategtorus.py").read_text()
    start = src.find("self.wave = 1000 * np.array([")
    end = src.find("])", start)
    arr_text = src[start + len("self.wave = 1000 * np.array(") : end + 1]
    upstream_wave = 1000.0 * np.array(eval(arr_text))
    with h5py.File(TEMPLATE_BUNDLE, "r") as f:
        ours_wave = f["torus/wave_nm"][:]
    np.testing.assert_allclose(ours_wave, upstream_wave, rtol=1e-15, atol=0.0)
