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

__all__ = [
    "GROUND_FILTER_FACILITIES",
    "SOURCES",
    "SPACE_FILTER_FACILITIES",
    "WavelengthSource",
    "filter_wavelength_convention",
]


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
        "Balogh+1999 (ApJ 527, 54) Table 1 centers its [O II] window 3713-3741 on the air "
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
        "residuals after conversion +0.47/+0.30 A. Since #3113 each label that rounds a "
        "line of observation/line_list.py (within 1.5 A of its air wavelength) takes that "
        "line's vacuum wavelength, and the 7320/7330 labels the blend centers, so those "
        "residuals are gone",
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
        "-0.47/+0.08/-0.70 A (Hgamma/MgII 2798/MgII 2802) after conversion, which is the "
        "label rounding; infrared labels (Ne2_1281um ... C2_15774um, > 1e4 A) are the "
        "NIST VACUUM values (O3_8836um 88.36 and C2_15774um 157.74, air would be 88.33 "
        "and 157.70) and are left alone (the stored values are 100x the label, a "
        "separate scale defect in scripts/build_flury2024_grids.py _parse_wavelength_aa). "
        "Since #3113 each optical label within 0.5 A of the air wavelength of a line of "
        "observation/line_list.py takes that line's vacuum wavelength (the O2_7319/7320/"
        "7330/7331 labels name single components and keep the plain conversion)",
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
        "18756.1, 12.8136) all agree with the vacuum values to 7e-5 after air_to_vac. The "
        "rule has no upper limit: cloudy-master source/cddefines.cpp t_wavl::sprt_wl "
        "prints wl / p_RefIndex(1e8/wl) for every positive wavelength, with "
        "prt.lgPrintLineAirWavelengths true by default (init_defaults_preparse.cpp) and the "
        "Peck & Reeder 1972 index applied wherever the wavenumber is below 5e4 cm^-1 "
        "(longward of 2000 A), so the six labels beyond 100 micron (121.769, 145.495, "
        "157.636, 205.283, 370, 609 micron) are air labels too; Byler+2017 "
        "(arXiv:1611.08305) convert Cloudy's air labels with Morton 1991; "
        "scripts/convert_cue_weights.py. The blend labels 7323, 7332 and 7332.15 name the "
        "[O II] 7320 and 7330 blends and are not air wavelengths (the air blends sit near "
        "7320.5 and 7330.2): they take the vacuum blend centers of observation/line_list.py "
        "(7321.47, 7332.21) instead of the converted label (#3113)",
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
        "18756.1, ...). Cloudy air-labels every wavelength above 2000 A, with no upper limit "
        "(source/cddefines.cpp t_wavl::sprt_wl); synthesizer-grids cloudy17.py/cloudy23.py "
        "leave 'print line vacuum' commented out. Note the dataset is "
        "sorted and pairs with lines/luminosity by index, not with lines/id (the ids are "
        "alphabetical)",
        "tengri.components.nebular.agn_nebular._load_synthesizer_nlr_grid",
    ),
    WavelengthSource(
        "synthesizer_nlr_spectra",
        "data/synthesizer_grids/test_grid_agn-nlr.hdf5",
        "vacuum",
        "spectra/wavelength is column 0 of Cloudy's save-continuum file. "
        "synthesizer-grids (synthesizer-project/grid-generation, src/synthesizer_grids/"
        "cloudy/cloudy17.py l.370 and cloudy23.py l.451) writes 'save last continuum "
        '"{model}.cont" units Angstroms\' and read_wavelength() takes that column '
        "unchanged. Cloudy fills it with Energy::angstromVac() = RYDLAM / E (source/"
        "energy.h: 'all remaining methods return vacuum wavelengths'; service.cpp "
        "AnuUnit; save_do.cpp l.907): a conversion from the Rydberg energy mesh that "
        "never touches the air-index code used for line labels (cddefines.cpp "
        "t_wavl::sprt_wl), so the 'units Angstroms' option is vacuum. Independent "
        "check on the shipped file: the mesh is geometric in energy (constant "
        "dln(lambda) = 3.33298e-3 to 4e-9 over 4000-30000 A), and the air index would "
        "bend ln(lambda) vs pixel by -7.5e-6 (4000-5000 A) and -4.3e-6 (5000-6500 A); "
        "the file shows -0.2e-6 and -0.3e-6 (noise 0.2e-6), which rejects air. The "
        "12-27 A pixels cannot place the Balmer (3646/3647 A) or Paschen (8204/8206 A) "
        "jump to the 1.2-2 A offset, so the edges were not used",
        "tengri.components.nebular.agn_nebular._load_synthesizer_nlr_grid",
    ),
    WavelengthSource(
        "svo_filters_space",
        "data/filters",
        "vacuum",
        "space missions have no air: AKARI, Euclid, Gaia, GALEX, HST, Herschel, JWST, "
        "Planck, Roman, Spitzer, Swift, WISE and XMM-Newton throughputs are measured or "
        "modeled for an instrument that observes in vacuum, and a lab calibration in air "
        "would be a documented conversion that no SVO entry mentions; classified by the "
        "SVO facility prefix (SPACE_FILTER_FACILITIES)",
        "tengri.observation.filters",
    ),
    WavelengthSource(
        "svo_filters",
        "data/filters",
        "unestablished",
        "ground-based SVO FPS VOTables (SDSS, DECam, HSC, LSST, VISTA, 2MASS, MegaCam "
        "examined) carry unit Angstrom and effective wavelengths but no air/vacuum field "
        "or comment. Checked 2026-10-11: Doi+2010 (AJ 139, 1628, arXiv:1002.3701) "
        "calibrated the SDSS monochromator against Hg+Ar lamp lines (Asahi Spectra "
        "HSU-100S, 250-1000 nm) to +-3 A and state no medium; lsst/throughputs README "
        "says only 'All curves are in nanometers' (as-measured filter curves with a "
        "modeled X=1.2 atmosphere); Cohen+2003 (AJ 126, 1090, arXiv:astro-ph/0304350) "
        "and the DES DR1 paper (arXiv:1801.03181) say nothing about air or vacuum for the "
        "bandpasses; the NAOJ HSC observer page (forobservers.html, Kawanomoto+2018) "
        "links the HSC curves with no medium; the NOIRLab DECam filter page returned "
        "nothing usable; the 1 A sampling of the O2 A band in LSST_i and Suprime IB767 "
        "is too smooth to place the head to 2 A",
        "tengri.observation.filters",
        "no source states the medium of any ground-based filter curve, and the best "
        "documented calibration (Doi+2010, +-3 A) is coarser than the 1.7 A (2.7e-4) air "
        "shift at r; a curve also combines a lab-measured filter (air) with a model "
        "atmosphere and detector, so a global conversion would be a guess; the shift is "
        "insensitive for broadband photometry (~1e-4), so the curves are used unchanged "
        "and every ground family (SDSS, DECam, HSC, Suprime, LSST, 2MASS, MegaCam, WIRCam, "
        "VISTA, HAWK-I, UKIDSS, Pan-STARRS, SkyMapper, ALHAMBRA, J-PAS/J-PLUS, GTC, "
        "NEWFIRM, FourStar, Johnson/Cousins, the (sub)mm receivers) stays on the open "
        "list; the 7DT delivery (nm) names no medium either",
    ),
    WavelengthSource(
        "qsogen_emlines",
        "data/qsogen_emline_template.dat",
        "vacuum",
        "Temple+2021 (MNRAS 508, 737, arXiv:2109.04472) Sec. 1: 'All emission lines are "
        "identified with their wavelengths in vacuum'; the 970-5100 A part is built from "
        "SDSS DR7 composites (SDSS wavelengths are vacuum) and an MFICA reconstruction of "
        "the same spectra (App. B1-B2). The narrow-line template (1.15 A pixel at 5000 A) puts "
        "Hbeta, [OIII] 5007 and [OIII] 4959 at 4861.61, 5007.36, 4959.48: 0.66-0.88 A "
        "(50-66 km/s) below the vacuum lines, the usual blueshift of quasar narrow lines "
        "against the systemic redshift, where an air frame would need a +18 to +32 km/s "
        "redshift; that supports but does not by itself decide the frame",
        "tengri.components.agn.qsogen._load_emline_template_arrays",
    ),
    WavelengthSource(
        "qsogen_emlines_nir",
        "data/qsogen_emline_template.dat",
        "vacuum",
        "Temple+2021 App. B3: wavelengths longward of 5100 A (H-alpha, near-infrared "
        "lines) come from the Glikman+2006 composite (ApJ 640, 579, arXiv:astro-ph/"
        "0511640), smoothed to a 490 km/s pixel. Glikman+2006 reduced its IRTF "
        "SpeX spectra with Spextool (Cushing+2004) and Telluric (Vacca+2003); Cushing+2004 "
        "(PASP 116, 362, Sec. 4.2.6, 'Wavelength Calibration': argon lines from Rao+1966 "
        "and ATRAN sky lines) states 'All wavelengths are given in vacuum'. The composite "
        "is shifted to the rest frame, which does not change the medium, and qsogen "
        "(github.com/MJTemple/qsogen, qsosed.py) applies no air/vacuum conversion to the "
        "template. The 11 A pixel at H-alpha cannot test it, so the chain of documentation "
        "is the evidence",
        "tengri.components.agn.qsogen._load_emline_template_arrays",
    ),
)

#: SVO facility prefixes (the text before the first underscore of the file name) of space
#: missions: vacuum by definition.
SPACE_FILTER_FACILITIES: tuple[str, ...] = (
    "AKARI",
    "Euclid",
    "GAIA",
    "GALEX",
    "HST",
    "Herschel",
    "JWST",
    "PLANCK",
    "Roman",
    "Spitzer",
    "Swift",
    "WISE",
    "XMM",
)

#: SVO facility prefixes of ground-based or ground-calibrated curves: convention unestablished.
GROUND_FILTER_FACILITIES: tuple[str, ...] = (
    "2MASS",
    "APEX",
    "CAHA",
    "CFHT",
    "CTIO",
    "GTC",
    "Generic",
    "JCMT",
    "LCO",
    "LSST",
    "NOAO",
    "OAJ",
    "PAN-STARRS",
    "Paranal",
    "SLOAN",
    "SPT",
    "SkyMapper",
    "Subaru",
    "UKIRT",
)


def filter_wavelength_convention(svo_name: str) -> str:
    """Declared convention of an SVO-style filter name (``Facility_Instrument_Band``).

    Parameters
    ----------
    svo_name : str
        SVO filter name or cache file stem, e.g. ``"JWST_NIRCam_F200W"``.

    Returns
    -------
    str
        ``"vacuum"`` for a space mission, ``"unestablished"`` for a ground facility.

    Raises
    ------
    KeyError
        If the facility is in neither list (a new family must declare its convention).
    """
    facility = svo_name.split("_", 1)[0]
    if facility in SPACE_FILTER_FACILITIES:
        return "vacuum"
    if facility in GROUND_FILTER_FACILITIES:
        return "unestablished"
    raise KeyError(f"filter facility {facility!r} has no declared wavelength convention")
