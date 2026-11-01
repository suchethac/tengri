# SPDX-License-Identifier: BSD-3-Clause
r"""The Gutkin+2016 NEOGAL star-forming grid is repackaged faithfully (#983).

``data/gutkin16_grid.h5`` is built from the public NEOGAL text tables
``data/neogal/nebular_emission_Z*.txt`` by ``scripts/build_gutkin16_grid.py``.
Each stored line luminosity is the tabulated value in L_sun per (M_sun/yr),
converted to erg/s and divided by the table's own ionizing photon rate per unit
SFR, so the grid is a line-to-ionizing-photon ratio:

.. math::

    \log_{10}\!\left(\frac{L_{\rm line}}{Q_H}\right)
    = \log_{10}\!\left(L^{\rm table}_{\rm line}\,L_\odot\right)
    - \log_{10} Q_H^{\rm table}(Z, m_{\rm up})

Gutkin, Charlot & Bruzual 2016, MNRAS 462, 1757, Section 2 (the constant-SFR
nebular tables, 100 Myr continuous star formation). The Case B anchor follows
Osterbrock & Ferland 2006, Table 4.4 (T = 1e4 K), giving
:math:`\log_{10} Q_H` from :math:`L_{\rm H\beta}` at :math:`\xi_d \to 0`.
The pinned anchor values below are the table's own values at
:math:`\log U = -4`, :math:`n_H = 100`, :math:`({\rm C/O})/({\rm C/O})_\odot = 1`,
:math:`m_{\rm up} = 100`, with an absolute tolerance of 0.03 dex.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

from tengri.observation.eline_catalog import EMISSION_LINES

pytestmark = pytest.mark.regression_paper

_REPO = Path(__file__).resolve().parents[3]
_NEOGAL_DIR = _REPO / "data" / "neogal"
_TRACKED_H5 = _REPO / "data" / "gutkin16_grid.h5"
_SSP_FILE = _REPO / "data" / "bc03_pdva_stelib_chabrier.h5"
_BUILDER_PATH = _REPO / "scripts" / "build_gutkin16_grid.py"

_LSUN_ERG = 3.826e33
_LOGZ_EXPECTED = np.log10(
    [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 0.010, 0.014, 0.017, 0.020, 0.030, 0.040]
)
_LOGU_EXPECTED = np.arange(-4.0, -0.9, 0.5)
_XI_EXPECTED = np.array([0.1, 0.3, 0.5])
_CO_EXPECTED = np.array([0.10, 0.14, 0.20, 0.27, 0.38, 0.52, 0.72, 1.00, 1.40])
_MUP_EXPECTED = np.array([100.0, 300.0])
_NH_EXPECTED = np.array([2.0, 3.0])

#: Index of Z = 1e-4, 0.004 and 0.02 on the 14-point metallicity axis.
_IZ_PINNED = (0, 5, 11)
_QH_PINNED_M100 = (53.55, 53.41, 53.32)

_UV_LABEL_VACUUM = {
    "NV1240": 1240.81,
    "CIV1548": 1548.19,
    "CIV1551": 1550.77,
    "HeII1640": 1640.42,
    "OIII]1661": 1660.81,
    "OIII]1666": 1666.15,
    "[SiIII]1883": 1882.71,
    "SiIII]1888": 1892.03,
    "CIII]1908": 1908.73,
}


#: Converted optical labels and their catalog rows (vacuum, tengri.observation.eline_catalog).
_OPTICAL_LABEL_CATALOG = {
    "[OII]3727": "OII3726",
    "Hbeta": "Hbeta",
    "[OIII]4959": "OIII4959",
    "[OIII]5007": "OIII5007",
    "[NII]6548": "NII6548",
    "Halpha": "Halpha",
    "[NII]6584": "NII6583",
    "[SII]6717": "SII6716",
    "[SII]6731": "SII6731",
}


def _load_builder():
    """Import ``scripts/build_gutkin16_grid.py`` by path (``scripts`` is not a package)."""
    spec = importlib.util.spec_from_file_location("build_gutkin16_grid", _BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_text_tables() -> dict[float, np.ndarray]:
    """Parse every NEOGAL text table into ``{Z: (N, 23) array}``, skipping ``#`` lines."""
    tables = {}
    for path in sorted(_NEOGAL_DIR.glob("nebular_emission_Z*.txt")):
        digits = path.stem.split("_Z")[1]
        z_value = float("0." + digits)
        rows = []
        with open(path) as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                rows.append([float(tok) for tok in stripped.split()])
        tables[z_value] = np.array(rows)
    return tables


@pytest.fixture(scope="module")
def tracked():
    """The tracked grid, read once for the whole module."""
    assert _TRACKED_H5.exists(), f"tracked grid missing: {_TRACKED_H5}"
    with h5py.File(_TRACKED_H5, "r") as fh:
        grp = fh["gutkin16"]
        out = {name: grp[name][()] for name in grp}
        out["_attrs"] = dict(grp.attrs)
        out["_root_attrs"] = dict(fh.attrs)
    return out


def test_axes_match_neogal_table(tracked):
    """Axes are the NEOGAL table's own values, with n_H restricted to 100 and 1000.

    Gutkin+2016 Section 2: log U in -4..-1 (step 0.5), xi_d in {0.1, 0.3, 0.5},
    n_H in {10..1e4} of which only 1e2 and 1e3 are fully tabulated, C/O in nine
    values, m_up in {100, 300} Msun.
    """
    np.testing.assert_allclose(tracked["logZ_axis"], _LOGZ_EXPECTED, rtol=0, atol=1e-12)
    np.testing.assert_allclose(tracked["logU_axis"], _LOGU_EXPECTED, rtol=0, atol=1e-12)
    np.testing.assert_allclose(tracked["xi_d_axis"], _XI_EXPECTED, rtol=0, atol=0)
    np.testing.assert_allclose(tracked["log_nH_axis"], _NH_EXPECTED, rtol=0, atol=0)
    np.testing.assert_allclose(tracked["co_ratio_axis"], _CO_EXPECTED, rtol=0, atol=1e-12)
    np.testing.assert_allclose(tracked["mup_axis"], _MUP_EXPECTED, rtol=0, atol=0)

    tables = _parse_text_tables()
    assert sorted(tables) == pytest.approx(sorted(10.0**_LOGZ_EXPECTED), rel=1e-12)
    for z_value, table in tables.items():
        np.testing.assert_array_equal(np.unique(table[:, 0]), _LOGU_EXPECTED)
        np.testing.assert_array_equal(np.unique(table[:, 1]), _XI_EXPECTED)
        np.testing.assert_array_equal(np.unique(table[:, 3]), _CO_EXPECTED)
        np.testing.assert_array_equal(np.unique(table[:, 4]), _MUP_EXPECTED)
        assert set(np.unique(table[:, 2])) >= {100.0, 1000.0}, z_value


def test_tabulated_nodes_bit_exact_vs_txt(tracked):
    """Every tabulated node equals the text table through the stated float operations.

    The reference is recomputed here from the raw ``.txt`` rows, with no code
    shared with the builder, so a silent edit to either side breaks the match.
    """
    log_line = tracked["log_line_per_qh"]
    log_qh = tracked["log_qh_per_sfr_g16"]
    filled = tracked["filled_mask"]
    names = tracked["line_names"]
    assert log_line.shape == (14, 7, 3, 2, 9, 2, 18)
    assert len(names) == 18

    tables = _parse_text_tables()
    checked = 0
    for iz, z_value in enumerate(sorted(tables)):
        table = tables[z_value]
        for row in table:
            log_u, xi_d, n_h, co, mup = row[0], row[1], row[2], row[3], row[4]
            if n_h not in (100.0, 1000.0):
                continue
            iu = int(np.argmin(np.abs(_LOGU_EXPECTED - log_u)))
            ix = int(np.argmin(np.abs(_XI_EXPECTED - xi_d)))
            inh = 0 if n_h == 100.0 else 1
            ic = int(np.argmin(np.abs(_CO_EXPECTED - co)))
            im = int(np.argmin(np.abs(_MUP_EXPECTED - mup)))
            idx = (iz, iu, ix, inh, ic, im)
            if filled[idx]:
                continue
            line_lum = row[5:]
            expected = np.log10(np.maximum(line_lum, 1e-30) * _LSUN_ERG) - log_qh[iz, im]
            assert np.array_equal(log_line[idx], expected), idx
            checked += 1
    assert checked == 14 * 7 * 3 * 2 * 9 * 2 - 19


def test_filled_mask_count_is_19(tracked):
    """Exactly the 19 nodes absent from the text tables are marked as filled."""
    filled = tracked["filled_mask"]
    assert filled.dtype == bool
    assert int(filled.sum()) == 19
    assert int(tracked["_attrs"]["n_filled"]) == 19


def test_z040_mup_duplication_recorded(tracked):
    """The Z = 0.040 m_up = 300 block is a byte copy of m_up = 100, and the file says so."""
    recorded = np.atleast_1d(tracked["_attrs"]["mup300_equals_mup100_at_logZ"])
    assert recorded.tolist() == pytest.approx([np.log10(0.040)])

    tables = _parse_text_tables()
    table = tables[0.040]
    m100 = table[table[:, 4] == 100.0]
    m300 = table[table[:, 4] == 300.0]
    lookup = {tuple(row[:4]): row[5:] for row in m100}
    for row in m300:
        assert np.array_equal(lookup[tuple(row[:4])], row[5:])


def test_hbeta_vacuum_is_4862p68_and_uv_labels_unchanged(tracked):
    """H-beta is the vacuum image of NIST air 4861.33 A; UV labels stay vacuum values.

    Labels at or above 2000 A are converted once from air with
    ``tengri.utils.air_vacuum.air_to_vac``. Labels below 2000 A are vacuum values
    already and are stored as they are.
    """
    wave = tracked["line_wavelengths_aa"]
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in tracked["line_names"]]
    assert names[1] == "Hbeta"
    assert wave[1] == pytest.approx(4862.68, abs=0.01)
    assert wave[names.index("[OII]3727")] == pytest.approx(3727.09, abs=0.05)
    for name, catalog_name in _OPTICAL_LABEL_CATALOG.items():
        idx = names.index(name)
        assert wave[idx] == pytest.approx(EMISSION_LINES[catalog_name][0], abs=0.05), name
    for name, expected in _UV_LABEL_VACUUM.items():
        idx = names.index(name)
        assert wave[idx] == pytest.approx(expected, abs=1e-9), name
    assert tracked["_root_attrs"]["wavelength_convention"] == (
        "vacuum; labels >= 2000 A converted from air with tengri.utils.air_vacuum "
        "(Cloudy convention)"
    )


def test_qh_anchor_within_0p03_dex_of_pinned_values(tracked):
    """The Case B anchor reproduces the pinned table values to 0.03 dex.

    Osterbrock & Ferland 2006, Table 4.4: Hbeta per ionizing photon at
    T = 1e4 K, Case B, with the xi_d -> 0 extrapolation from the table's own
    Hbeta at log U = -4, n_H = 100, C/O = 1.
    """
    qh = tracked["log_qh_per_sfr_g16"]
    assert qh.shape == (14, 2)
    assert float(tracked["_attrs"]["hb_per_qh_erg"]) == pytest.approx(4.78e-13)
    for iz, pinned in zip(_IZ_PINNED, _QH_PINNED_M100, strict=True):
        assert qh[iz, 0] == pytest.approx(pinned, abs=0.03)


def test_tracked_h5_matches_rebuild(tracked, tmp_path):
    """Rebuilding from the tracked inputs reproduces every dataset exactly."""
    if not _SSP_FILE.exists():
        pytest.skip(f"BC03 diagnostic SSP absent: {_SSP_FILE}")
    builder = _load_builder()
    out = tmp_path / "gutkin16_grid.h5"
    builder.build(_NEOGAL_DIR, out, ssp_path=_SSP_FILE)

    with h5py.File(out, "r") as rebuilt, h5py.File(_TRACKED_H5, "r") as tracked_fh:
        rebuilt_names = set(rebuilt["gutkin16"].keys())
        tracked_names = set(tracked_fh["gutkin16"].keys())
        assert rebuilt_names == tracked_names
        for name in sorted(tracked_names):
            assert np.array_equal(
                rebuilt["gutkin16"][name][()], tracked_fh["gutkin16"][name][()]
            ), name
