# SPDX-License-Identifier: BSD-3-Clause
"""Declared wavelength convention of every tabulated wavelength source.

Every wavelength inside tengri is vacuum. A source whose published convention
is air declares ``"air"`` here and names the loader that converts it, once, to
vacuum with :func:`tengri.utils.air_vacuum.air_to_vac`. The sweep test
``tests/regression/bug/test_bug_vacuum_wavelengths_everywhere.py`` enumerates
this registry against the data files and the index catalog, so a new source
cannot ship without declaring its convention.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SOURCES", "WavelengthSource"]


@dataclass(frozen=True)
class WavelengthSource:
    """One tabulated wavelength source.

    Parameters
    ----------
    name : str
        Short identifier.
    path : str
        Data file or module, relative to the repository root.
    convention : str
        Convention of the file *as published*: ``"vacuum"``, ``"air"``, or
        ``"mixed"`` (per line; the loader carries the mask).
    evidence : str
        Where the convention was established.
    ingest : str
        Dotted name of the function that returns the vacuum wavelengths.
    """

    name: str
    path: str
    convention: str
    evidence: str
    ingest: str


SOURCES: tuple[WavelengthSource, ...] = (
    WavelengthSource(
        "fsps_ssp",
        "data/fsps_prsc_miles_chabrier.h5",
        "vacuum",
        "FSPS manual: 'Wavelengths are in angstroms in vacuum'; Ca K/H, Na D troughs "
        "of the shipped 10 Gyr SSP sit at the vacuum positions",
        "tengri.components.stellar.sps",
    ),
    WavelengthSource(
        "lick_indices",
        "src/tengri/observation/spectral_indices.py",
        "air",
        "Trager+1998 Table 1, Worthey & Ottaviani 1997 Table 1; FSPS sps_setup.f90 "
        "converts allindices.dat with airtovac",
        "tengri.observation.spectral_indices._lick_index_def",
    ),
    WavelengthSource(
        "feltre16",
        "data/feltre_grid.h5",
        "mixed",
        "NIST air values for [OIII], [OI], [NII], [SII], Hbeta; Halpha and [OII]3727 "
        "are vacuum (offsets from eline_catalog); UV lines vacuum",
        "tengri.components.nebular.agn_nebular._load_feltre_grid",
    ),
    WavelengthSource(
        "mappings5_3mdb",
        "data/mappings_templates.h5",
        "air",
        "integer PyNeb-style labels (6563, 5007, 4959, 3726, 9069) round the air, "
        "not the vacuum, wavelengths",
        "tengri.components.nebular.shock._load_mappings_grids",
    ),
    WavelengthSource(
        "flury24_mappings5",
        "data/flury2024_grids.h5",
        "air",
        "integer PyNeb-style labels (H1r_6563A, O3_5007A, ...) round the air wavelengths",
        "tengri.components.nebular.mappings_photo._load_agn_grid",
    ),
    WavelengthSource(
        "cb19_templates",
        "data/cb19_templates.h5",
        "vacuum",
        "scripts/download_cb19_templates.py _LINE_MAP, now all vacuum (five air entries "
        "converted). The hosted file predates this and was NOT inspectable offline",
        "tengri.components.nebular.cloudy_cb19",
    ),
    WavelengthSource(
        "cue_network",
        "data/cue_weights.npz",
        "air",
        "lineList_wav carries 6562.80, 5006.84, 4861.32 (Cloudy air labels); "
        "scripts/convert_cue_weights.py",
        "tengri.components.nebular._shared.nebular_line_waves_to_vacuum",
    ),
    WavelengthSource(
        "cloudy_fsps_grid",
        "data/cloudy_raw/emlines_info.dat",
        "vacuum",
        "FSPS emlines_info.dat carries 6564.72, 4862.76, 5008.31 (vacuum); FSPS manual",
        "tengri.components.nebular.cloudy_grid",
    ),
)
