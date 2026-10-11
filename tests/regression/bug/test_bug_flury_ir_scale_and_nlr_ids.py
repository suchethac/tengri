# SPDX-License-Identifier: BSD-3-Clause
r"""Flury+2024 infrared line wavelengths and the Synthesizer NLR line names.

Two ingestion bugs found by the vacuum-wavelength audit.

1. ``scripts/build_flury2024_grids.py`` read the digits of an infrared label as
   microns. Flury's labels encode the wavelength in hundredths of a micron
   (``O3_8836um`` is 88.36 um = 8.836e5 Angstrom, ``C2_15774um`` is 157.74 um),
   so every IR line was stored 100 times too long (``O3_8836um`` at 8.836e7 A).

2. The Synthesizer AGN-NLR grid stores ``lines/id`` alphabetically but
   ``lines/wavelength`` sorted, and ``lines/luminosity`` follows the wavelength
   order. tengri paired the names with the wavelengths by index, so
   ``line_ids[i]`` named a different line from the one at ``wavelength[i]``
   (H-alpha / H-beta = 2.98 by wavelength, 0.22 by id).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

ROOT = Path(__file__).resolve().parents[3]
FLURY_H5 = ROOT / "data" / "flury2024_grids.h5"
NLR_H5 = ROOT / "data" / "synthesizer_grids" / "test_grid_agn-nlr.hdf5"
VAC_OIII_88UM_AA = 883564.0  # [O III] 88.356 um, vacuum (label 8836 rounds it)


def _builder():
    path = ROOT / "scripts" / "build_flury2024_grids.py"
    spec = importlib.util.spec_from_file_location("build_flury2024_grids", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_agn_grid(builder, labels):
    """A minimal AGN-OXAF group dict, with wavelengths from the builder's parser."""
    waves = np.array([builder._parse_wavelength_aa(c) for c in labels], dtype=np.float32)
    shape = (1, 1, 1, 1, 1)
    return {
        "line_names": labels,
        "line_waves_aa": waves,
        "logHB_per_lum": np.zeros(shape, dtype=np.float32),
        "line_ratios": np.ones((*shape, len(labels)), dtype=np.float32),
        "z_axis": np.array([1.0]),
        "logU_axis": np.array([-2.0]),
        "logn_axis": np.array([2.0]),
        "logmbh_axis": np.array([8.0]),
        "logedd_axis": np.array([-1.0]),
    }


@pytest.fixture(scope="module")
def tiny_flury_file(tmp_path_factory):
    builder = _builder()
    labels = ["O3_5007A", "O3_8836um", "C2_15774um", "Ar2_0698um"]
    path = tmp_path_factory.mktemp("flury") / "tiny.h5"
    builder._write_hdf5(path, {"agn-oxaf-cpr": _tiny_agn_grid(builder, labels)})
    return path


# -- Flury: the parser -------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "wave_aa"),
    [
        ("O3_8836um", 88.36e4),  # 88.36 um
        ("C2_15774um", 157.74e4),  # 157.74 um
        ("Ar2_0698um", 6.98e4),  # 6.98 um
        ("H1r_0109um", 1.09e4),  # 1.09 um (Pa-delta region)
        ("O3_5007A", 5007.0),
        ("H1r_6563A", 6563.0),
    ],
)
def test_parser_reads_ir_labels_as_hundredths_of_a_micron(label, wave_aa):
    got = _builder()._parse_wavelength_aa(label)
    assert got == pytest.approx(wave_aa, rel=1e-12)


def test_parser_rejects_unknown_unit_suffix():
    assert _builder()._parse_wavelength_aa("O3_5007X") is None
    assert _builder()._parse_wavelength_aa("logq") is None


# -- Flury: the loader -------------------------------------------------------


def test_loaded_oiii_88um_sits_at_its_vacuum_wavelength(tiny_flury_file):
    from tengri.components.nebular.mappings_photo import _load_agn_grid

    grid = _load_agn_grid(tiny_flury_file, "cpr")
    waves = np.asarray(grid.line_wavelengths)
    # the label rounds 88.3564 um to 88.36 um: within label rounding
    assert waves[1] == pytest.approx(VAC_OIII_88UM_AA, abs=0.005 * 1e4)
    assert waves[2] == pytest.approx(157.74e4, rel=1e-6)


def test_loader_refuses_a_file_built_before_the_ir_fix(tiny_flury_file, tmp_path):
    from tengri.components.nebular.mappings_photo import _load_agn_grid

    stale = tmp_path / "stale.h5"
    stale.write_bytes(tiny_flury_file.read_bytes())
    with h5py.File(stale, "r+") as f:
        del f.attrs["schema_version"]  # the pre-fix builder wrote none
    with pytest.raises(RuntimeError, match=r"build_flury2024_grids\.py"):
        _load_agn_grid(stale, "cpr")


def test_builder_and_loader_agree_on_the_schema_version(tiny_flury_file):
    from tengri.components.nebular.mappings_photo import FLURY_SCHEMA_VERSION

    with h5py.File(tiny_flury_file, "r") as f:
        assert int(f.attrs["schema_version"]) == FLURY_SCHEMA_VERSION


@pytest.mark.skipif(not FLURY_H5.exists(), reason="data/flury2024_grids.h5 not present")
@pytest.mark.parametrize("model", ["sb99", "bpass", "agn_oxaf"])
def test_built_flury_file_has_no_wavelength_above_a_quarter_millimeter(model):
    """Every Flury line is at most 157.74 um (C II 158); 100x too long would be 1.6e8 A."""
    with h5py.File(FLURY_H5, "r") as f:
        waves = f[f"{model}/cpr/line_wavelengths_aa"][:]
        names = [n.decode() for n in f[f"{model}/cpr/line_names"][:]]
    assert waves.max() < 2.0e6
    i = names.index("O3_8836um")
    assert waves[i] == pytest.approx(883600.0, rel=1e-6)


# -- Synthesizer NLR line names ---------------------------------------------


def _id_wavelength_aa(line_id: str) -> float:
    """Wavelength quoted in a Cloudy id such as ``H 1 6562.80A`` / ``Fe 2 1.25668m``."""
    token = line_id.split()[-1]
    value, unit = float(token[:-1]), token[-1]
    return value * {"A": 1.0, "m": 1.0e4, "c": 1.0e8}[unit]


@pytest.fixture(scope="module")
def nlr_grid():
    if not NLR_H5.exists():
        pytest.skip(f"{NLR_H5} not present")
    from tengri.components.nebular.agn_nebular import _load_synthesizer_nlr_grid

    return _load_synthesizer_nlr_grid(NLR_H5)


def test_file_ids_are_not_in_wavelength_order():
    """The premise: the file stores ids alphabetically, wavelengths sorted."""
    if not NLR_H5.exists():
        pytest.skip(f"{NLR_H5} not present")
    with h5py.File(NLR_H5, "r") as f:
        ids = [i.decode() for i in f["lines/id"][:]]
        waves = f["lines/wavelength"][:]
    assert np.all(np.diff(waves) >= 0)
    assert ids == sorted(ids)
    assert not np.allclose([_id_wavelength_aa(i) for i in ids], waves, rtol=1e-6)


def test_each_nlr_line_name_carries_its_loaded_wavelength(nlr_grid):
    waves = np.asarray(nlr_grid.line_wavelengths_aa)
    named = np.array([_id_wavelength_aa(i) for i in nlr_grid.line_ids])
    # name wavelengths are the catalog's own (air above 2000 A); the loaded ones
    # may be converted to vacuum, which moves a line by < 1e-3 relative
    np.testing.assert_allclose(named, waves, rtol=1e-3, atol=0)
    assert len(set(nlr_grid.line_ids)) == len(nlr_grid.line_ids)


def test_halpha_over_hbeta_is_read_by_name(nlr_grid):
    ids = list(nlr_grid.line_ids)
    lum = np.asarray(10.0 ** np.asarray(nlr_grid.log_line_per_lbol))
    ratio = lum[..., ids.index("H 1 6562.80A")] / lum[..., ids.index("H 1 4861.32A")]
    # every node is at or above the case-B decrement (collisional excitation raises it)
    assert np.all(ratio > 2.7)
    assert ratio[(0,) * 6] == pytest.approx(2.98, abs=0.01)  # 0.22 when read by file id


def test_unpairable_ids_and_wavelengths_raise():
    from tengri.components.nebular.agn_nebular import _synthesizer_line_ids_by_wavelength

    ids = [b"H 1 6562.80A", b"H 1 4861.32A", b"Fe 2 1.25668m"]
    ok = _synthesizer_line_ids_by_wavelength(ids, [4861.32, 6562.80, 12566.8])
    assert ok == ("H 1 4861.32A", "H 1 6562.80A", "Fe 2 1.25668m")
    with pytest.raises(ValueError, match="do not match"):
        _synthesizer_line_ids_by_wavelength(ids, [4861.32, 6000.0, 12566.8])
