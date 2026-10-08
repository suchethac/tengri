# SPDX-License-Identifier: BSD-3-Clause
"""Declared wavelength convention of every tabulated wavelength source.

Every wavelength inside tengri is vacuum. A source whose published convention
is air declares ``"air"`` here and names the loader that converts it, once, to
vacuum with :func:`tengri.utils.air_vacuum.air_to_vac`. A source whose
convention could not be established from its own metadata, its generating code
or its paper declares ``"unestablished"`` and states, in ``reason``, what was
tried and why it did not settle; tengri then uses its wavelengths unchanged and
the source stays on the open list. The sweep tests
``tests/regression/bug/test_bug_vacuum_wavelengths_everywhere.py`` and
``test_bug_vacuum_conventions_remaining.py`` enumerate this registry against
the data files and the index catalog, so a new source cannot ship without
declaring its convention.
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
        ``"mixed"`` (per line; the loader carries the mask), or
        ``"unestablished"`` (see ``reason``).
    evidence : str
        Where the convention was established (or, for ``"unestablished"``,
        what was examined).
    ingest : str
        Dotted name of the function that returns the vacuum wavelengths.
    reason : str
        Required for ``"unestablished"``: why the evidence does not settle the
        convention. Empty otherwise.
    """

    name: str
    path: str
    convention: str
    evidence: str
    ingest: str
    reason: str = ""


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
        "break_indices",
        "src/tengri/utils/break_windows.py",
        "air",
        "Balogh+1999 (ApJ 527, 54) Table 1 centres its [O II] window 3713-3741 on the air "
        "name lambda3727 (vacuum doublet centroid 3728.5); Westfall+2019 (AJ 158, 231) "
        "Table 4 rows 44-45 list the D4000 (Bruzual 1983) and Dn4000 passbands as defined "
        "for air. FSPS allindices.dat holds Dn4000 but converts only its first 25 "
        "(Lick) rows with airtovac, which is a choice, not evidence for vacuum",
        "tengri.utils.break_windows.BREAK_VACUUM_WINDOWS",
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
        "not the vacuum, wavelengths. Hg_4341A: scripts/download_mappings_templates.py "
        "reads the 3MdB column HI_4340, whose label rounds the air 4340.47 (vacuum "
        "4341.69 would be 4342); the stored 4341.0 is tengri's own hand-set value, so the "
        "converted line sits 0.53 A above the vacuum Hgamma (label rounding, not a "
        "convention error). MgII_2796A/MgII_2803A: 3MdB columns MgII_2796/MgII_2803 round "
        "the air 2795.53/2802.71 (vacuum 2796.35/2803.53 would round to 2796/2804), "
        "residuals after conversion +0.47/+0.30 A",
        "tengri.components.nebular.shock._load_mappings_grids",
    ),
    WavelengthSource(
        "flury24_mappings5",
        "data/flury2024_grids.h5",
        "air",
        "optical integer labels (H1r_6563A, O3_5007A, ...) round the air wavelengths; "
        "H1r_4340A rounds the air 4340.47 (vacuum would be 4342); MgII_2798A is the "
        "blended-doublet label whose flux-weighted centroid is 2797.9 in air (2798.7 in "
        "vacuum, which would round to 2799) and MgII_2802A truncates the air 2802.71 "
        "(vacuum 2803.53 is 1.5 A away), so both read as air with residuals of "
        "-0.47/+0.08/-0.70 A (Hgamma/MgII 2798/MgII 2802) after conversion, which is the label rounding; infrared labels "
        "(Ne2_1281um ... C2_15774um, > 1e4 A) are the NIST VACUUM values (O3_8836um 88.36 "
        "and C2_15774um 157.74, air would be 88.33 and 157.70) and are left alone",
        "tengri.components.nebular.mappings_photo._load_agn_grid",
    ),
    WavelengthSource(
        "cb19_templates",
        "data/cb19_templates.h5",
        "mixed",
        "measured 2026-10-07 on the hosted file: Hbeta 4862.68, Halpha 6564.61, [OIII]5008.24 "
        "vacuum; Hgamma 4340.47, [OI]6300.30, [NII]6548.05, [NII]6583.45 Cloudy air labels. "
        "scripts/download_cb19_templates.py _LINE_MAP is all vacuum",
        "tengri.components.nebular.cloudy_cb19.load_cb19_grid",
    ),
    WavelengthSource(
        "cue_network",
        "data/cue_weights.npz",
        "air",
        "lineList_wav carries 6562.80, 5006.84, 4861.32 (Cloudy air labels), and the "
        "infrared labels are air too: Paalpha 18751.0, [NeII] 12.8101 micron (vacuum "
        "18756.1, 12.8136) all agree with the vacuum values to 7e-5 after air_to_vac up to "
        "100 micron; Cloudy converts every label above 2000 A to air (Cloudy workshop "
        "slides: vacuum below 2000 A, air above; Byler+2017 arXiv:1611.08305 convert "
        "Cloudy's air labels with Morton 1991); "
        "scripts/convert_cue_weights.py",
        "tengri.components.nebular.cue._load_cue_weights_eager",
    ),
    WavelengthSource(
        "cloudy_fsps_grid",
        "data/cloudy_raw/emlines_info.dat",
        "vacuum",
        "FSPS emlines_info.dat carries 6564.72, 4862.76, 5008.31 (vacuum); FSPS manual",
        "tengri.components.nebular.cloudy_grid",
    ),
    WavelengthSource(
        "synthesizer_nlr_lines",
        "data/synthesizer_grids/test_grid_agn-nlr.hdf5",
        "air",
        "lines/wavelength (Cloudy c23.01 line labels, CloudyParams cloudy_version) holds "
        "6562.80, 5006.84, 4958.91, 4861.32, 6583.45, 9068.62 and Paalpha 18751.0, "
        "Pabeta 12818.1, Brgamma 21655.3: the air values (vacuum 6564.61, 5008.24, "
        "18756.1, ...). Cloudy air-labels everything above 2000 A. Note the dataset is "
        "sorted and pairs with lines/luminosity by index, not with lines/id (the ids are "
        "alphabetical)",
        "tengri.components.nebular.agn_nebular._load_synthesizer_nlr_grid",
    ),
    WavelengthSource(
        "synthesizer_nlr_spectra",
        "data/synthesizer_grids/test_grid_agn-nlr.hdf5",
        "unestablished",
        "spectra/wavelength is Cloudy's continuum mesh (9244 pixels, 15-20 A wide in "
        "the optical); the linecont array puts each line in one pixel",
        "tengri.components.nebular.agn_nebular._load_synthesizer_nlr_grid",
        "the mesh is an energy grid (vacuum by construction) but the pixel is 15-20 A "
        "wide at H-alpha/[OIII] against a 1.4-1.8 A air-vacuum offset, so no line "
        "position in the file can confirm it, and Synthesizer documents no convention; "
        "used unchanged",
    ),
    WavelengthSource(
        "svo_filters",
        "data/filters",
        "unestablished",
        "SVO FPS VOTables (SDSS, DECam, HSC, LSST, VISTA, 2MASS, MegaCam examined) carry "
        "unit Angstrom and effective wavelengths but no air/vacuum field or comment; the "
        "LSST throughputs README gives only 'nanometers'; speclite/Doi+2010 and the "
        "SVO FPS documentation state no medium; the 1 A sampling of the O2 A band in "
        "LSST_i and Suprime IB767 is too smooth to place the head to 2 A",
        "tengri.observation.filters",
        "no source states the medium of any filter curve, and a curve combines a "
        "lab-measured filter (air) with a model atmosphere and detector; the shift is "
        "2.7e-4 in wavelength (1.7 A at r), insensitive for broadband photometry, so the "
        "curves are used unchanged and every filter stays on the open list; the 7DT "
        "delivery (nm) names no medium either",
    ),
    WavelengthSource(
        "qsogen_emlines",
        "data/qsogen_emline_template.dat",
        "vacuum",
        "Temple+2021 (MNRAS 508, 737, arXiv:2109.04472) Sec. 1: 'All emission lines are "
        "identified with their wavelengths in vacuum'; the 970-5100 A part is built from "
        "SDSS DR7 composites (SDSS wavelengths are vacuum) and an MFICA reconstruction of "
        "the same spectra (App. B1-B2). Narrow-line template: Hbeta centroid 4862.4 vs "
        "vacuum 4862.68 / air 4861.32; [OIII] 5007/4959 sit 0.8/0.65 A below the vacuum "
        "values (the usual quasar [OIII] blueshift, Coatman+2019) rather than 0.6/0.7 A "
        "above the air ones; the 69 km/s pixel is 1.15 A",
        "tengri.components.agn.qsogen._load_emline_template_arrays",
    ),
    WavelengthSource(
        "qsogen_emlines_nir",
        "data/qsogen_emline_template.dat",
        "unestablished",
        "Temple+2021 App. B3: wavelengths longward of 5100 A (H-alpha, near-infrared "
        "lines) come from the Glikman+2006 composite, smoothed to a 490 km/s pixel",
        "tengri.components.agn.qsogen._load_emline_template_arrays",
        "the Glikman+2006 composite and the paper state no medium for it, and the pixel "
        "(about 11 A at H-alpha, 1.8-2.8 A air-vacuum offset) cannot place a line to the "
        "offset; used unchanged",
    ),
)
