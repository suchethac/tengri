# SPDX-License-Identifier: BSD-3-Clause
r"""The six conventions #2778 left open, established from evidence (owner, 2026-10-07).

Every wavelength inside tengri is vacuum; an air input converts once, at ingestion.
#2778 flagged six sources whose convention it had not established:

1. the Synthesizer NLR grid ``lines/wavelength``  -> AIR (Cloudy c23.01 labels), converted;
2. Cloudy/PyNeb labels above 10000 A              -> Cloudy's are AIR to 100 micron (converted),
   Flury+2024's infrared labels are the NIST VACUUM values (left alone);
3. SVO filter curves                              -> no source states the medium: UNESTABLISHED;
4. ``qsogen_emline_template.dat``                 -> VACUUM below 5100 A (Temple+2021), the
   near-infrared part UNESTABLISHED;
5. the Dn4000 / D4000 windows                     -> AIR (Balogh+1999, Westfall+2019 Table 4),
   stored as published and converted at import;
6. the Flury/MAPPINGS Hgamma and MgII labels      -> AIR labels, residuals are label rounding.

At the parent commit the NLR labels reach the SED unconverted (H-alpha 6562.80), the Cue and
Synthesizer infrared labels are read as vacuum (Pa-alpha 18751.0 against 18756.1), the
Dn4000 windows are the air numbers, and the registry has none of these entries.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular._line_ingest import (
    CLOUDY_LABEL_AIR_MAX_AA,
    LABEL_AIR_MAX_AA,
    catalog_air_to_vacuum,
)
from tengri.observation.eline_catalog import EMISSION_LINES
from tengri.utils.air_vacuum import air_to_vac
from tengri.utils.wavelength_conventions import SOURCES

pytestmark = pytest.mark.regression_bug

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"

# Cloudy c23.01 air labels [Angstrom] and the vacuum value of the same line.
_NLR_AIR_TO_VAC = {
    6562.80: 6564.61,  # H-alpha
    5006.84: 5008.24,  # [O III]
    4861.32: 4862.68,  # H-beta
    9068.62: 9071.1,  # [S III]
    18751.0: 18756.1,  # Pa-alpha
}


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


def _write_minimal_nlr_grid(path: Path, labels: np.ndarray) -> None:
    """A 2x2x2x2x2x2 Synthesizer-layout NLR grid carrying ``labels`` as lines/wavelength."""
    shape = (2,) * 6
    with h5py.File(path, "w") as f:
        axes = f.create_group("axes")
        axes["mass"] = np.array([1.988e38, 1.988e39])
        axes["accretion_rate_eddington"] = np.array([0.1, 1.0])
        axes["cosine_inclination"] = np.array([0.2, 0.8])
        axes["metallicities"] = np.array([0.01, 0.02])
        axes["ionisation_parameter"] = np.array([0.001, 0.01])
        axes["hydrogen_density"] = np.array([100.0, 1000.0])
        lines = f.create_group("lines")
        lines["wavelength"] = labels
        lines["luminosity"] = np.full(shape + (labels.size,), 1.0e30)
        lines["id"] = np.array([f"line {i}" for i in range(labels.size)], dtype="S16")
        f.create_group("log10_specific_ionising_luminosity")["HI"] = np.full(shape, 43.0)


# ── 1. Synthesizer NLR grid: air labels, converted at ingestion ─────────────────────────────


def test_nlr_grid_labels_are_air_and_land_on_the_vacuum_lines(tmp_path):
    from tengri.components.nebular.agn_nebular import _load_synthesizer_nlr_grid

    labels = np.array([1215.67, 1640.4, *_NLR_AIR_TO_VAC.keys()])
    _write_minimal_nlr_grid(tmp_path / "nlr.hdf5", labels)
    got = np.asarray(_load_synthesizer_nlr_grid(tmp_path / "nlr.hdf5").line_wavelengths_aa)

    np.testing.assert_array_equal(got[:2], labels[:2])  # < 2000 A: vacuum, untouched
    for air, vac in _NLR_AIR_TO_VAC.items():
        value = got[int(np.argmin(np.abs(labels - air)))]
        assert abs(value - vac) < 0.06, (air, value, vac)
    # the catalog entries Halpha / [O III] are the same vacuum lines
    assert abs(got[2] - EMISSION_LINES["Halpha"][0]) < 0.02
    assert abs(got[3] - EMISSION_LINES["OIII5007"][0]) < 0.02


def test_nlr_grid_file_labels_are_air_in_the_real_synthesizer_file():
    path = _nlr_grid_file()
    if path is None:
        pytest.skip("Synthesizer test_grid_agn-nlr.hdf5 not present")
    from tengri.components.nebular.agn_nebular import _load_synthesizer_nlr_grid

    with h5py.File(path) as f:
        raw = f["lines/wavelength"][:]
    # The file holds the Cloudy air labels: H-alpha 6562.80, [O III] 5006.84, Pa-alpha 18751.0 ...
    for air in _NLR_AIR_TO_VAC:
        assert np.min(np.abs(raw - air)) < 0.011, air
    for vac in _NLR_AIR_TO_VAC.values():
        assert np.min(np.abs(raw - vac)) > 0.5, vac  # ... and none of the vacuum values
    loaded = np.asarray(_load_synthesizer_nlr_grid(path).line_wavelengths_aa)
    for air, vac in _NLR_AIR_TO_VAC.items():
        assert np.min(np.abs(loaded - vac)) < 0.06, vac
        assert np.min(np.abs(loaded - air)) > 0.5, air


# ── 2. Labels above 10000 A: Cloudy air to 100 micron, Flury infrared vacuum ───────────────

#: (Cloudy air label, NIST vacuum value) [Angstrom]: the infrared lines both catalogs carry.
_CLOUDY_IR = (
    (105076.0, 105105.0),  # [S IV] 10.51 micron
    (128101.0, 128136.0),  # [Ne II] 12.8136 micron (vacuum by IR convention)
    (155509.0, 155551.0),  # [Ne III]
    (187078.0, 187130.0),  # [S III] 18.7
    (518004.0, 518145.0),  # [O III] 51.8
    (883323.0, 883564.0),  # [O III] 88.4
    (18751.0, 18756.13),  # Pa-alpha
    (12818.1, 12821.67),  # Pa-beta
    (21655.3, 21661.2),  # Br-gamma
)


def test_cloudy_infrared_labels_are_air_and_convert_to_the_vacuum_value():
    air = np.array([a for a, _ in _CLOUDY_IR])
    vac = np.array([v for _, v in _CLOUDY_IR])
    got = catalog_air_to_vacuum(air, max_air_aa=CLOUDY_LABEL_AIR_MAX_AA)
    assert np.max(np.abs(got / vac - 1.0)) < 1e-5
    assert np.min(np.abs(air / vac - 1.0)) > 2.5e-4  # unconverted they are 2.7e-4 short
    # the default (PyNeb/MAPPINGS) window stops at 1e4 A and leaves them alone
    np.testing.assert_array_equal(catalog_air_to_vacuum(air[air > LABEL_AIR_MAX_AA]), air[air > 1e4])
    # beyond the Cloudy cap nothing is converted (far-infrared labels not established)
    far = np.array([1.2e6, 1.576e6])
    np.testing.assert_array_equal(catalog_air_to_vacuum(far, max_air_aa=CLOUDY_LABEL_AIR_MAX_AA), far)


def test_cue_infrared_labels_ingest_to_the_vacuum_value():
    path = DATA / "cue_weights.npz"
    if not path.is_file():
        pytest.skip("cue_weights.npz not present")
    from tengri.components.nebular.cue import load_cue_weights

    raw = np.load(path)["lineList_wav"]
    waves = load_cue_weights(str(path)).nn_line_wav
    for air, vac in _CLOUDY_IR:
        assert np.min(np.abs(raw - air)) < 0.5 * air * 1e-5 + 0.011, air  # the file holds the label
        got = float(waves[int(np.argmin(np.abs(waves - vac)))])
        assert abs(got / vac - 1.0) < 1e-4, (air, got, vac)  # 2.7e-4 unconverted
    assert not np.any(np.abs(waves - 18751.0) < 0.5)  # the air Pa-alpha label is gone


def test_flury_infrared_labels_are_the_vacuum_values_and_stay_put():
    path = DATA / "flury2024_grids.h5"
    if not path.is_file():
        pytest.skip("flury2024_grids.h5 not present")
    from tengri.components.nebular.mappings_photo import _load_agn_grid

    grid = _load_agn_grid(path, "cdn")
    with h5py.File(path) as f:
        grp = f["agn_oxaf/cdn"]
        names = [n.decode() for n in grp["line_names"][:]]
        raw = grp["line_wavelengths_aa"][:].astype(np.float64)
    got = np.asarray(grid.line_wavelengths, dtype=np.float64)
    # [O III] 88.36 and [C II] 157.74 micron are the NIST VACUUM values (air: 88.33, 157.70)
    for name, vacuum_aa in (("O3_8836um", 883564.0), ("C2_15774um", 1577409.0), ("O4_2589um", 258903.0)):
        i = names.index(name)
        assert abs(raw[i] / vacuum_aa - 1.0) < 3e-4, name
        assert got[i] == pytest.approx(raw[i], rel=1e-6), name  # not shifted by 2.7e-4
    ir = raw > LABEL_AIR_MAX_AA
    assert ir.sum() == 35
    np.testing.assert_allclose(got[ir], raw[ir], rtol=1e-6)


# ── 5. Dn4000 / D4000: published in air, applied in vacuum ────────────────────────────────


def test_break_windows_are_the_converted_published_air_values():
    from tengri.observation.spectral_indices import STANDARD_INDICES
    from tengri.utils.break_windows import BREAK_AIR_WINDOWS

    assert BREAK_AIR_WINDOWS["Dn4000"] == ((3850.0, 3950.0), (4000.0, 4100.0))  # Balogh+1999
    assert BREAK_AIR_WINDOWS["D4000"] == ((3750.0, 3950.0), (4050.0, 4250.0))  # Bruzual 1983
    for name, windows in BREAK_AIR_WINDOWS.items():
        got = np.array([e for w in STANDARD_INDICES[name].continuum for e in w])
        air = np.array([e for w in windows for e in w])
        np.testing.assert_allclose(got, air_to_vac(air), rtol=0, atol=1e-9)
        assert np.all(got - air > 1.0), name  # vacuum, not the air numbers


def _strip_spectrum(lo: float, hi: float):
    """f_nu = 1 everywhere except 100 on [lo, hi]: only a window reaching in sees it."""
    wave = np.arange(3700.0, 4400.0, 0.05)
    flux = np.where((wave >= lo) & (wave <= hi), 100.0, 1.0)
    return jnp.asarray(wave), jnp.asarray(flux)


@pytest.mark.parametrize(
    ("lo", "hi"),
    [(3849.9, 3851.0), (4100.0, 4101.0)],
    ids=["air_blue_edge_strip", "air_red_edge_strip"],
)
def test_dn4000_integrates_the_vacuum_windows(lo, hi):
    """Strips that lie inside the air window edge but outside the vacuum one do not count."""
    from tengri.analysis.diagnostics.spectral import dn4000
    from tengri.utils.sed_quantities import compute_dn4000

    wave, flux = _strip_spectrum(lo, hi)
    assert float(dn4000(wave, flux)) == pytest.approx(_expected_dn4000(lo, hi), rel=2e-3)
    assert float(compute_dn4000(flux, wave)) == pytest.approx(_expected_dn4000(lo, hi), rel=2e-3)


def _expected_dn4000(lo: float, hi: float) -> float:
    from tengri.utils.break_windows import DN4000_BLUE_AA, DN4000_RED_AA

    def mean(window):
        a, b = window
        overlap = max(0.0, min(b, hi) - max(a, lo))
        return 1.0 + 99.0 * overlap / (b - a)

    return mean(DN4000_RED_AA) / mean(DN4000_BLUE_AA)


def test_dn4000_equals_the_catalog_break_index():
    """The two code paths and the catalog share one set of windows."""
    from tengri.observation.spectral_indices import STANDARD_INDICES
    from tengri.utils.break_windows import DN4000_BLUE_AA, DN4000_RED_AA

    assert STANDARD_INDICES["Dn4000"].continuum == (DN4000_BLUE_AA, DN4000_RED_AA)


# ── 4. qsogen emission-line template ───────────────────────────────────────────────────────


def _qsogen_template() -> np.ndarray:
    path = DATA / "qsogen_emline_template.dat"
    if not path.is_file():
        pytest.skip("qsogen_emline_template.dat not present")
    return np.loadtxt(path)


def test_qsogen_narrow_template_sits_on_the_vacuum_lines():
    """Hbeta is 0.3 A from vacuum and 1.1 A from air; [O III] sits below the vacuum value.

    The narrow-line template (column 5) has a 1.15 A pixel at 5000 A. Air would put
    [O III] 5007/4959 a fraction of a pixel above 5006.84/4958.91 and Hbeta near
    4861.3. The centroids are Hbeta 4862.4 (vacuum 4862.68), [O III] 5007.4 (vacuum
    5008.24, air 5006.84) and 4959.5 (4960.29, 4958.91): a 0.8 A blueshift of the
    narrow [O III], the usual quasar offset, against a +0.6 A redshift in air.
    """
    a = _qsogen_template()
    wave, narrow = a[:, 0], a[:, 5]

    def centroid(center: float, half: float = 12.0) -> float:
        m = np.abs(wave - center) < half
        weight = np.clip(narrow[m] - np.median(narrow[np.abs(wave - center) < 40]), 0.0, None)
        return float((wave[m] * weight).sum() / weight.sum())

    vac_hb = float(air_to_vac(4861.32))
    air_hb = 4861.32
    assert abs(centroid(vac_hb) - vac_hb) < abs(centroid(vac_hb) - air_hb) - 0.5
    for air_line in (5006.84, 4958.91):
        vac_line = float(air_to_vac(air_line))
        c = centroid(vac_line)
        assert abs(c - vac_line) < 1.0, air_line
        assert c < vac_line  # blueshifted narrow [O III], not an air-frame redshift


def test_qsogen_template_wavelengths_are_used_unconverted():
    from tengri.components.agn.qsogen import _EMLINE_WAV

    if _EMLINE_WAV is None:
        pytest.skip("qsogen template not present")
    np.testing.assert_array_equal(np.asarray(_EMLINE_WAV), _qsogen_template()[:, 0])


# ── 6. Flury / MAPPINGS Hgamma and MgII labels ─────────────────────────────────────────────


def test_mappings_hgamma_and_mgii_labels_read_as_air_labels():
    """3MdB integer labels round the AIR value; the converted line is within one label unit."""
    from tengri.components.nebular.shock import _load_mappings_grids

    if not (DATA / "mappings_templates.h5").is_file():
        pytest.skip("mappings_templates.h5 not present")
    g = _load_mappings_grids()["mappings5"]
    names = list(g["line_names"])
    vac = {
        "Hg_4341A": EMISSION_LINES["Hgamma"][0],
        "MgII_2796A": 2796.352,
        "MgII_2803A": 2803.531,
    }
    for name, vacuum_aa in vac.items():
        got = float(g["line_wavelengths_aa"][names.index(name)])
        # nearest-integer labels: residual is the rounding, bounded by one label unit
        assert abs(got - vacuum_aa) < 0.55, (name, got)
    # the air reading of the DB columns: HI_4340 rounds 4340.47, vacuum 4341.69 would be HI_4342
    assert round(4340.47) == 4340
    assert round(float(air_to_vac(4340.47))) == 4342


def test_flury_mgii_and_hgamma_labels_read_as_air_labels():
    from tengri.components.nebular.mappings_photo import _load_agn_grid

    path = DATA / "flury2024_grids.h5"
    if not path.is_file():
        pytest.skip("flury2024_grids.h5 not present")
    grid = _load_agn_grid(path, "cdn")
    with h5py.File(path) as f:
        names = [n.decode() for n in f["agn_oxaf/cdn/line_names"][:]]
    got = {n: float(grid.line_wavelengths[names.index(n)]) for n in ("H1r_4340A", "MgII_2798A", "MgII_2802A")}
    assert abs(got["H1r_4340A"] - EMISSION_LINES["Hgamma"][0]) < 0.55
    assert abs(got["MgII_2798A"] - 2798.74) < 0.55  # 2:1 flux-weighted vacuum doublet centroid
    assert abs(got["MgII_2802A"] - 2803.531) < 0.8  # truncated air label: 0.70 A, a label unit


# ── Registry: every one of the six is declared, evidence or an explicit reason ─────────────

_REQUIRED = {
    "synthesizer_nlr_lines": "air",
    "synthesizer_nlr_spectra": "unestablished",
    "svo_filters": "unestablished",
    "qsogen_emlines": "vacuum",
    "qsogen_emlines_nir": "unestablished",
    "break_indices": "air",
    "mappings5_3mdb": "air",
    "flury24_mappings5": "air",
    "cue_network": "air",
}


def _resolve(dotted: str):
    try:
        return importlib.import_module(dotted)
    except ModuleNotFoundError:
        module, _, attr = dotted.rpartition(".")
        return getattr(importlib.import_module(module), attr)


def test_registry_declares_every_formerly_open_source():
    by_name = {s.name: s for s in SOURCES}
    for name, convention in _REQUIRED.items():
        assert name in by_name, name
        assert by_name[name].convention == convention, name
    assert len(by_name) == len(SOURCES)  # no duplicate names


def test_registry_has_no_unknown_convention_and_every_unestablished_has_a_reason():
    for s in SOURCES:
        assert s.convention in {"vacuum", "air", "mixed", "unestablished"}, s.name
        assert len(s.evidence) > 40, s.name
        if s.convention == "unestablished":
            assert len(s.reason) > 80, f"{s.name}: an open source must say why"
        else:
            assert s.reason == "", s.name


def test_registry_ingest_targets_exist():
    for s in SOURCES:
        assert _resolve(s.ingest) is not None, (s.name, s.ingest)


def test_registry_covers_the_template_and_filter_files():
    paths = {s.path for s in SOURCES}
    assert "data/qsogen_emline_template.dat" in paths
    assert "data/filters" in paths
    assert "data/synthesizer_grids/test_grid_agn-nlr.hdf5" in paths
    filters = DATA / "filters"
    if filters.is_dir():
        assert any(filters.glob("*.dat"))  # the registry entry covers a real directory
