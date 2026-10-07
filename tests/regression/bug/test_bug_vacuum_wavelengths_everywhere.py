# SPDX-License-Identifier: BSD-3-Clause
r"""Every wavelength inside tengri is vacuum; air inputs convert once, at ingestion.

Owner ruling 2026-10-07. The Lick windows, the Feltre+2016 table and the
MAPPINGS V / Flury+2024 line labels were published in air and used as if they
were vacuum (1-2 Angstrom off in the optical); two refractive-index formulas
(Morton 1991 forward, Edlen 1953 inverse) disagreed with each other.

At HEAD (before this fix) the Lick, Feltre and MAPPINGS tests fail and there is
no ``tengri.utils.air_vacuum`` module.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import jax
import numpy as np
import pytest

from tengri.utils.wavelength_conventions import SOURCES

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
VAC_HALPHA, VAC_HBETA = 6564.61, 4862.68


def _need(name: str) -> Path:
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return path


def test_lick_windows_are_the_converted_air_values():
    from tengri.observation.spectral_indices import _LICK_AIR_WINDOWS, STANDARD_INDICES
    from tengri.utils.air_vacuum import air_to_vac

    for name, (continuum_air, feature_air) in _LICK_AIR_WINDOWS.items():
        got = STANDARD_INDICES[name]
        edges_air = np.array([e for w in (*continuum_air, feature_air) for e in w])
        edges_got = np.array([e for w in (*got.continuum, got.feature) for e in w])
        np.testing.assert_allclose(edges_got, air_to_vac(edges_air), rtol=0, atol=1e-9)
        assert np.all(edges_got - edges_air > 1.0), name  # vacuum, not the air numbers


def test_lick_hbeta_feature_brackets_vacuum_hbeta():
    from tengri.observation.spectral_indices import STANDARD_INDICES

    lo, hi = STANDARD_INDICES["Hbeta"].feature
    assert lo < VAC_HBETA < hi
    assert VAC_HBETA - lo > 10.0  # the air window put the line 1.35 A nearer the edge


def test_feltre_catalog_lands_on_the_vacuum_lines():
    from tengri.components.nebular.agn_nebular import _load_feltre_grid
    from tengri.observation.eline_catalog import EMISSION_LINES

    grid = _load_feltre_grid(_need("feltre_grid.h5"))
    names = [n.decode() for n in h5py.File(DATA / "feltre_grid.h5")["feltre/line_names"][:]]
    vac = {
        "[OII]3727": "OII3726", "Hbeta": "Hbeta", "[OIII]4959": "OIII4959",
        "[OIII]5007": "OIII5007", "[OI]6300": "OI6300", "[NII]6548": "NII6548",
        "Halpha": "Halpha", "[NII]6584": "NII6583", "[SII]6717": "SII6716",
        "[SII]6731": "SII6731",
    }  # fmt: skip
    for name, key in vac.items():
        got = float(grid.line_wavelengths_aa[names.index(name)])
        assert abs(got - EMISSION_LINES[key][0]) < 0.05, (name, got)


def test_mappings_labels_land_on_the_vacuum_lines_within_label_rounding():
    from tengri.components.nebular.shock import _load_mappings_grids
    from tengri.observation.eline_catalog import EMISSION_LINES

    _need("mappings_templates.h5")
    g = _load_mappings_grids()["mappings5"]
    names = list(g["line_names"])
    want = {"O3_5007A": "OIII5007", "O3_4959A": "OIII4959", "HA_6563A": "Halpha",
            "Hb_4861A": "Hbeta", "NII_6583A": "NII6583", "SII_6716A": "SII6716"}  # fmt: skip
    for name, key in want.items():
        got = float(g["line_wavelengths_aa"][names.index(name)])
        # integer labels round to 0.5 A; unconverted they are 1.3-1.8 A short
        assert abs(got - EMISSION_LINES[key][0]) < 0.5, (name, got)


def test_flury_labels_are_vacuum_after_load():
    from tengri.components.nebular.mappings_photo import _load_agn_grid

    path = _need("flury2024_grids.h5")
    grid = _load_agn_grid(path, "cdn")
    names = [n.decode() for n in h5py.File(path)["agn_oxaf/cdn/line_names"][:]]
    got = float(grid.line_wavelengths[names.index("H1r_6563A")])
    assert abs(got - VAC_HALPHA) < 0.5


def test_cue_air_catalog_ingests_to_vacuum():
    from tengri.components.nebular._shared import nebular_line_waves_to_vacuum

    waves = np.load(_need("cue_weights.npz"))["lineList_wav"]
    out = np.asarray(nebular_line_waves_to_vacuum(waves, xp=np))
    j = int(np.argmin(np.abs(waves - 6562.80)))
    assert abs(out[j] - VAC_HALPHA) < 0.05


def test_fsps_ssp_is_vacuum():
    h = h5py.File(_need("fsps_prsc_miles_chabrier.h5"))
    w, age = h["ssp_wave"][:], h["ssp_lg_age_gyr"][:]
    spec = h["ssp_flux"][-3, int(np.argmin(np.abs(age - 1.0)))]
    m = (w > 3925.0) & (w < 3945.0)
    k_min = w[m][np.argmin(spec[m])]  # Ca II K: 3933.66 air, 3934.77 vacuum
    assert abs(k_min - 3934.77) < abs(k_min - 3933.66)


def test_sweep_every_source_declares_a_convention_and_is_registered():
    from tengri.observation.spectral_indices import STANDARD_INDICES

    assert all(s.convention in {"vacuum", "air", "mixed"} and s.evidence for s in SOURCES)
    registered = {s.path for s in SOURCES}
    for path in sorted(DATA.glob("*.h5")) + [DATA / "cue_weights.npz"]:
        if not path.exists() or path.suffix != ".h5":
            continue
        with h5py.File(path) as f:
            has_lines = []
            f.visit(lambda n: has_lines.append(n) if "line_wavelength" in n else None)
        if has_lines:
            assert f"data/{path.name}" in registered, f"{path.name}: undeclared line wavelengths"
    for s in SOURCES:
        assert (ROOT / s.path).exists() or s.path.startswith("data/"), s.path
        if s.convention != "vacuum":
            assert "." in s.ingest
    assert {"lick_indices"} <= {s.name for s in SOURCES if s.convention == "air"}
    assert set(STANDARD_INDICES)  # the Lick subset is covered by test_lick_*


_REFRACTIVE = re.compile(r"131\.4182|2\.76249e8|6\.4328e-5|2\.94981e-2|8\.34254e-5|2\.406147e-2|130\.1065924522|8\.336624212083e-5|2\.408926869968e-2", re.I)


def test_no_second_refractive_index_formula_anywhere():
    offenders = []
    for base in ("src", "scripts", "reproduction", "tools", "analysis"):
        for path in (ROOT / base).rglob("*.py"):
            if path.name == "air_vacuum.py":
                continue
            if _REFRACTIVE.search(path.read_text(errors="ignore")):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders


def test_converter_pair_properties():
    from tengri.utils.air_vacuum import air_to_vac, vac_to_air

    x = np.linspace(2000.0, 5.0e4, 20001)
    assert np.max(np.abs(air_to_vac(vac_to_air(x)) - x)) < 1e-6
    for vac, air in [(6564.61, 6562.80), (5008.24, 5006.84), (6585.27, 6583.45)]:
        assert abs(float(vac_to_air(vac)) - air) < 0.01
        assert abs(float(air_to_vac(air)) - vac) < 0.01
    short = np.array([912.0, 1215.67, 1500.0, 1999.0])
    np.testing.assert_array_equal(air_to_vac(short), short)
    np.testing.assert_array_equal(vac_to_air(short), short)
    for fn, x0 in [(air_to_vac, 6562.8), (vac_to_air, 6564.61)]:
        g = float(jax.grad(fn)(x0))
        assert np.isfinite(g) and g != 0.0
