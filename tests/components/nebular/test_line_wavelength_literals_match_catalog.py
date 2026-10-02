# SPDX-License-Identifier: BSD-3-Clause
"""Line-wavelength tables agree with the vacuum emission-line catalog (#2617).

``observation/eline_catalog.py`` is the single source of rest-frame vacuum line
wavelengths. Three checks:

1. every catalog line above 2000 A agrees with the FSPS vacuum line list to
   0.3 A where that list has the line (a sanity reference, not the source);
2. the catalog's [O I] 6300 entry equals the air wavelength 6300.304 A converted
   to vacuum with the IAU standard relation;
3. every number in a line-wavelength table of ``src/tengri`` that lies within
   3 A of a catalog line equals that line to 0.02 A.

The air values declared in ``components/nebular/_shared.py`` are deliberate
(air, vacuum) pairs, sit in no line-wavelength table, and are not swept.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tengri.observation.eline_catalog import EMISSION_LINES

pytestmark = pytest.mark.bounds

CATALOG_AA = {name: row[0] for name, row in EMISSION_LINES.items()}
CATALOG_SWEEP_AA = np.array(sorted(CATALOG_AA.values()))

#: Tolerance for the table sweep [A].
TABLE_TOL_AA = 0.02
#: A table number this close to a catalog line is taken to be that line [A].
MATCH_WINDOW_AA = 3.0
#: FSPS list values sit up to ~0.24 A from the NIST values for some lines [A].
FSPS_TOL_AA = 0.3


def _flatten(obj):
    """Yield every float inside nested tuples, lists, arrays and mapping values."""
    if isinstance(obj, dict):
        for item in obj.values():
            yield from _flatten(item)
    elif hasattr(obj, "wavelength"):
        yield float(obj.wavelength)
    elif isinstance(obj, (list, tuple, np.ndarray)):
        for item in obj:
            yield from _flatten(item)
    else:
        yield float(obj)


#: (module, attribute) of every table that holds line wavelengths.
LINE_TABLES = (
    ("tengri.components.agn.nlr", "_RICHARDSON_WAVES"),
    ("tengri.components.nebular.shock", "_FALLBACK_LINE_WAVES"),
    ("tengri.utils.sed_quantities", "KEY_LINES"),
    ("tengri.observation.line_measurement", "DESI_LINES"),
    ("tengri.analysis.plotting.styles", "SPECTRAL_FEATURES"),
)


def _fsps_line_waves():
    spec = importlib.util.find_spec("dsps")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("dsps is not installed: FSPS emlines_info.dat unavailable")
    path = Path(next(iter(spec.submodule_search_locations))) / "data" / "emlines_info.dat"
    if not path.is_file():
        pytest.skip(f"FSPS line list not found at {path}")
    rows = [line.split(",", 1)[0] for line in path.read_text().splitlines() if line.strip()]
    return np.array([float(r) for r in rows])


def _air_to_vacuum_morton(wave_air_aa):
    """Vacuum wavelength [A] from air via Morton (2000, ApJS 130, 403), eq. 8.

    n - 1 = 8.34254e-5 + 2.406147e-2 / (130 - s^2) + 1.5998e-4 / (38.9 - s^2),
    s = 1e4 / lambda_vac [A]; lambda_vac = lambda_air * n(lambda_vac), iterated.
    """
    wave_vac = wave_air_aa
    for _ in range(50):
        s2 = (1.0e4 / wave_vac) ** 2
        n_index = 1.0 + 8.34254e-5 + 2.406147e-2 / (130.0 - s2) + 1.5998e-4 / (38.9 - s2)
        wave_vac = wave_air_aa * n_index
    return wave_vac


def test_morton_conversion_of_oi_air_wavelength():
    """[O I] 6300: catalog holds the vacuum value of the 6300.304 A air line."""
    vac = _air_to_vacuum_morton(6300.304)
    assert vac - 6300.304 == pytest.approx(1.74, abs=0.01)
    assert CATALOG_AA["OI6300"] == pytest.approx(vac, abs=0.01)


def test_catalog_agrees_with_fsps_vacuum_list():
    """Each optical catalog line within 3 A of an FSPS line matches it to 0.3 A."""
    fsps = _fsps_line_waves()
    bad = []
    for name, wave in CATALOG_AA.items():
        if wave <= 2000.0:
            continue
        nearest = fsps[np.argmin(np.abs(fsps - wave))]
        diff = wave - nearest
        if FSPS_TOL_AA < abs(diff) <= MATCH_WINDOW_AA:
            bad.append(f"{name}: catalog {wave:.2f}, FSPS {nearest:.3f}, diff {diff:+.2f}")
    assert not bad, "catalog lines off the FSPS vacuum list:\n" + "\n".join(bad)


@pytest.mark.parametrize(("module_name", "attr"), LINE_TABLES, ids=[t[1] for t in LINE_TABLES])
def test_line_table_literals_match_catalog(module_name, attr):
    """Every table number within 3 A of a catalog line equals it to 0.02 A."""
    table = getattr(importlib.import_module(module_name), attr)
    values = list(_flatten(table))
    assert values, f"{module_name}.{attr} is empty"
    bad = []
    n_matched = 0
    for value in values:
        nearest = CATALOG_SWEEP_AA[np.argmin(np.abs(CATALOG_SWEEP_AA - value))]
        if abs(nearest - value) > MATCH_WINDOW_AA:
            continue
        n_matched += 1
        if abs(nearest - value) > TABLE_TOL_AA:
            bad.append(f"{value} vs catalog {nearest} (diff {value - nearest:+.3f})")
    assert n_matched > 0, f"{module_name}.{attr} holds no catalog line"
    assert not bad, f"{module_name}.{attr} off the catalog vacuum values:\n" + "\n".join(bad)
