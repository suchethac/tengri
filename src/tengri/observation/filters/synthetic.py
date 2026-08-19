# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic (top-hat) bandpass filters for facilities without SVO entries.

These filters are rectangular bandpass approximations defined by frequency or
energy ranges (GHz or keV). They are suitable for continuum photometry where the
source spectrum is smooth across the band, but NOT for cases where the bandpass
shape or response function matters.

Contains:
- ALMA receiver bands (1–10): GHz → Å conversion
- X-ray facilities (Chandra ACIS, NuSTAR, XMM-Newton EPIC): keV → Å
- (Sub)millimeter continuum (SPT, ACT, NIKA2, etc.): GHz → Å
- Literature X-ray bands (xray_soft, xray_hard, etc.)

**Approximation warning**: Every band in this module uses a rectangular
(top-hat) transmission profile (T=1 inside band, T=0 outside). Real detector
responses are more complex. The top-hat approximation is appropriate for
continuum photometry with smooth source spectra; for finer spectral features
or when detector response shape matters, use real bandpass curves from SVO.

**X-ray convention**: tengri's default photometric convention is photon-counting
Bessell (w = 1/λ; ADR-0017). For X-ray work, the energy convention (w = 1/E)
may be more appropriate — see Photometry.from_names(..., convention="energy").

References
----------
ALMA bands follow the Cycle 11 receiver specifications:
- https://www.almaobservatory.org/en/home/

Chandra ACIS bands are from the Chandra Source Catalog (CSC) 2.0:
- https://cxc.cfa.harvard.edu/csc/

NuSTAR bands are from the NuSTAR instrument documentation:
- https://heasarc.gsfc.nasa.gov/docs/nustar/

X-ray literature conventions (soft/hard bands) follow common SED-fitting
practice in the AGN literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tengri.utils.physics_constants import C_AA, HC_KEV_ANGSTROM

if TYPE_CHECKING:
    from tengri.observation.photometry import FilterCurve


# ── ALMA receiver band definitions (GHz) ──────────────────────────
# ALMA Cycle 11 specifications: each entry maps band number → (lo_ghz, hi_ghz).
# Full bandwidth used for top-hat width.
_ALMA_BANDS_GHZ: dict[int, tuple[float, float]] = {
    1: (35.0, 50.0),
    2: (67.0, 90.0),
    3: (84.0, 116.0),
    4: (125.0, 163.0),
    5: (163.0, 211.0),
    6: (211.0, 275.0),
    7: (275.0, 373.0),
    8: (385.0, 500.0),
    9: (602.0, 720.0),
    10: (787.0, 950.0),
}


@dataclass(frozen=True)
class SyntheticBand:
    """Metadata for a synthetic rectangular bandpass.

    Parameters
    ----------
    name : str
        Alias for the band (e.g., "alma_band6", "xray_soft").
    facility : str
        Human-readable facility name (e.g., "ALMA", "Chandra/ACIS").
    lo : float
        Low edge of the band in native units.
    hi : float
        High edge of the band in native units.
    unit : str
        Native unit: "GHz" (radio/submillimeter) or "keV" (X-ray).
    description : str
        One-line description: what this band is and its scientific context.

    """

    name: str
    facility: str
    lo: float
    hi: float
    unit: str
    description: str


# ── Synthetic band registry ────────────────────────────────────────

SYNTHETIC_BAND_REGISTRY: dict[str, SyntheticBand] = {
    # ── ALMA receiver bands (Cycle 11) ────────────────────────────
    "alma_band1": SyntheticBand(
        name="alma_band1",
        facility="ALMA",
        lo=35.0,
        hi=50.0,
        unit="GHz",
        description="ALMA Band 1 receiver (35–50 GHz); 6–9 mm continuum",
    ),
    "alma_band2": SyntheticBand(
        name="alma_band2",
        facility="ALMA",
        lo=67.0,
        hi=90.0,
        unit="GHz",
        description="ALMA Band 2 receiver (67–90 GHz); 3–4.5 mm continuum",
    ),
    "alma_band3": SyntheticBand(
        name="alma_band3",
        facility="ALMA",
        lo=84.0,
        hi=116.0,
        unit="GHz",
        description="ALMA Band 3 receiver (84–116 GHz); 2.6–3.6 mm continuum",
    ),
    "alma_band4": SyntheticBand(
        name="alma_band4",
        facility="ALMA",
        lo=125.0,
        hi=163.0,
        unit="GHz",
        description="ALMA Band 4 receiver (125–163 GHz); 1.8–2.4 mm continuum",
    ),
    "alma_band5": SyntheticBand(
        name="alma_band5",
        facility="ALMA",
        lo=163.0,
        hi=211.0,
        unit="GHz",
        description="ALMA Band 5 receiver (163–211 GHz); 1.4–1.8 mm continuum",
    ),
    "alma_band6": SyntheticBand(
        name="alma_band6",
        facility="ALMA",
        lo=211.0,
        hi=275.0,
        unit="GHz",
        description="ALMA Band 6 receiver (211–275 GHz); 1.1–1.4 mm continuum",
    ),
    "alma_band7": SyntheticBand(
        name="alma_band7",
        facility="ALMA",
        lo=275.0,
        hi=373.0,
        unit="GHz",
        description="ALMA Band 7 receiver (275–373 GHz); 0.8–1.1 mm continuum",
    ),
    "alma_band8": SyntheticBand(
        name="alma_band8",
        facility="ALMA",
        lo=385.0,
        hi=500.0,
        unit="GHz",
        description="ALMA Band 8 receiver (385–500 GHz); 0.6–0.8 mm continuum",
    ),
    "alma_band9": SyntheticBand(
        name="alma_band9",
        facility="ALMA",
        lo=602.0,
        hi=720.0,
        unit="GHz",
        description="ALMA Band 9 receiver (602–720 GHz); 0.4–0.5 mm continuum",
    ),
    "alma_band10": SyntheticBand(
        name="alma_band10",
        facility="ALMA",
        lo=787.0,
        hi=950.0,
        unit="GHz",
        description="ALMA Band 10 receiver (787–950 GHz); 0.3–0.4 mm continuum",
    ),
    # ── Chandra ACIS science bands (CSC 2.0) ──────────────────────
    "chandra_ultrasoft": SyntheticBand(
        name="chandra_ultrasoft",
        facility="Chandra/ACIS",
        lo=0.2,
        hi=0.5,
        unit="keV",
        description="Chandra ACIS ultrasoft band (0.2–0.5 keV); hot gas, stars",
    ),
    "chandra_soft": SyntheticBand(
        name="chandra_soft",
        facility="Chandra/ACIS",
        lo=0.5,
        hi=1.2,
        unit="keV",
        description="Chandra ACIS soft band (0.5–1.2 keV); thermal emission",
    ),
    "chandra_medium": SyntheticBand(
        name="chandra_medium",
        facility="Chandra/ACIS",
        lo=1.2,
        hi=2.0,
        unit="keV",
        description="Chandra ACIS medium band (1.2–2.0 keV); thermal + AGN",
    ),
    "chandra_hard": SyntheticBand(
        name="chandra_hard",
        facility="Chandra/ACIS",
        lo=2.0,
        hi=7.0,
        unit="keV",
        description="Chandra ACIS hard band (2–7 keV); AGN-dominated",
    ),
    "chandra_broad": SyntheticBand(
        name="chandra_broad",
        facility="Chandra/ACIS",
        lo=0.5,
        hi=7.0,
        unit="keV",
        description="Chandra ACIS broad band (0.5–7 keV); full ACIS coverage",
    ),
    "chandra_hrc": SyntheticBand(
        name="chandra_hrc",
        facility="Chandra/HRC",
        lo=0.1,
        hi=10.0,
        unit="keV",
        description="Chandra HRC wide band (0.1–10 keV); high-energy window",
    ),
    # ── NuSTAR bands ──────────────────────────────────────────────
    # ``nustar_full`` is the instrument bandpass, 3-79 keV. It carried 3-24 keV
    # when this registry was first written, which is a real subband but not the
    # full one -- a name that promises the whole detector and delivers a third
    # of it is the kind of quiet wrongness that never raises. The subbands below
    # are spelled with their edges for exactly that reason.
    "nustar_full": SyntheticBand(
        name="nustar_full",
        facility="NuSTAR",
        lo=3.0,
        hi=79.0,
        unit="keV",
        description="NuSTAR full bandpass (3-79 keV); the instrument's whole range",
    ),
    "nustar_soft": SyntheticBand(
        name="nustar_soft",
        facility="NuSTAR",
        lo=3.0,
        hi=8.0,
        unit="keV",
        description="NuSTAR soft subband (3-8 keV); overlaps Chandra/XMM hard band",
    ),
    "nustar_hard": SyntheticBand(
        name="nustar_hard",
        facility="NuSTAR",
        lo=8.0,
        hi=24.0,
        unit="keV",
        description="NuSTAR hard subband (8-24 keV); above Chandra/XMM coverage",
    ),
    "nustar_3_24": SyntheticBand(
        name="nustar_3_24",
        facility="NuSTAR",
        lo=3.0,
        hi=24.0,
        unit="keV",
        description="NuSTAR 3-24 keV; the most-quoted combined survey window",
    ),
    "nustar_10_40": SyntheticBand(
        name="nustar_10_40",
        facility="NuSTAR",
        lo=10.0,
        hi=40.0,
        unit="keV",
        description="NuSTAR 10-40 keV; high-energy AGN / Compton-thick probe",
    ),
    # ── XMM-Newton EPIC bands ─────────────────────────────────────
    "xmm_epic_soft": SyntheticBand(
        name="xmm_epic_soft",
        facility="XMM-Newton/EPIC",
        lo=0.5,
        hi=2.0,
        unit="keV",
        description="XMM-Newton EPIC soft band (0.5–2 keV); thermal gas",
    ),
    "xmm_epic_hard": SyntheticBand(
        name="xmm_epic_hard",
        facility="XMM-Newton/EPIC",
        lo=2.0,
        hi=12.0,
        unit="keV",
        description="XMM-Newton EPIC hard band (2–12 keV); AGN-dominated",
    ),
    "xmm_epic_broad": SyntheticBand(
        name="xmm_epic_broad",
        facility="XMM-Newton/EPIC",
        lo=0.5,
        hi=12.0,
        unit="keV",
        description="XMM-Newton EPIC broad band (0.5–12 keV); full EPIC coverage",
    ),
    # ── Swift/XRT band ────────────────────────────────────────────
    "swift_xrt": SyntheticBand(
        name="swift_xrt",
        facility="Swift/XRT",
        lo=0.3,
        hi=10.0,
        unit="keV",
        description="Swift X-Ray Telescope band (0.3–10 keV); wide-field X-ray monitor",
    ),
    # ── Literature X-ray bands (AGN SED convention) ────────────────
    "xray_soft": SyntheticBand(
        name="xray_soft",
        facility="Generic/X-ray",
        lo=0.5,
        hi=2.0,
        unit="keV",
        description="Standard soft X-ray band (0.5–2 keV); literature convention",
    ),
    "xray_hard": SyntheticBand(
        name="xray_hard",
        facility="Generic/X-ray",
        lo=2.0,
        hi=10.0,
        unit="keV",
        description="Standard hard X-ray band (2–10 keV); literature convention",
    ),
    "xray_full": SyntheticBand(
        name="xray_full",
        facility="Generic/X-ray",
        lo=0.5,
        hi=10.0,
        unit="keV",
        description="Standard full X-ray band (0.5–10 keV); literature convention",
    ),
    # ── Submillimeter/millimeter continuum ─────────────────────────
    # SPT (South Pole Telescope). Only the 95 GHz band is synthetic: SVO serves
    # measured SPT-SZ bandpasses at 150 and 220 GHz, and those are registered as
    # ordinary filters in FILTER_REGISTRY. A measured response always wins over a
    # rectangle, so ``spt_150ghz`` and ``spt_220ghz`` deliberately do not appear
    # here -- duplicating them would shadow real data with an approximation.
    "spt_95ghz": SyntheticBand(
        name="spt_95ghz",
        facility="SPT",
        lo=85.0,
        hi=105.0,
        unit="GHz",
        description="SPT-SZ 95 GHz band (85-105 GHz); 3.2 mm continuum. Synthetic: "
        "SVO serves no measured curve for this band (150/220 GHz are measured)",
    ),
    # ACT (Atacama Cosmology Telescope)
    "act_98ghz": SyntheticBand(
        name="act_98ghz",
        facility="ACT",
        lo=90.0,
        hi=106.0,
        unit="GHz",
        description="ACT 98 GHz band (90–106 GHz); 3.0 mm continuum",
    ),
    "act_150ghz": SyntheticBand(
        name="act_150ghz",
        facility="ACT",
        lo=140.0,
        hi=160.0,
        unit="GHz",
        description="ACT 150 GHz band (140–160 GHz); 2.0 mm continuum",
    ),
    "act_224ghz": SyntheticBand(
        name="act_224ghz",
        facility="ACT",
        lo=215.0,
        hi=233.0,
        unit="GHz",
        description="ACT 224 GHz band (215–233 GHz); 1.3 mm continuum",
    ),
    # NIKA2 (IRAM 30m)
    "nika2_1p15mm": SyntheticBand(
        name="nika2_1p15mm",
        facility="NIKA2",
        lo=250.0,
        hi=270.0,
        unit="GHz",
        description="NIKA2 1.15 mm band (250–270 GHz); 1.15 mm continuum",
    ),
    "nika2_2mm": SyntheticBand(
        name="nika2_2mm",
        facility="NIKA2",
        lo=145.0,
        hi=165.0,
        unit="GHz",
        description="NIKA2 2 mm band (145–165 GHz); 2 mm continuum",
    ),
    # AzTEC
    "aztec_1p1mm": SyntheticBand(
        name="aztec_1p1mm",
        facility="AzTEC",
        lo=270.0,
        hi=290.0,
        unit="GHz",
        description="AzTEC 1.1 mm band (270–290 GHz); 1.1 mm continuum",
    ),
    # MAMBO
    "mambo_1p2mm": SyntheticBand(
        name="mambo_1p2mm",
        facility="MAMBO",
        lo=240.0,
        hi=250.0,
        unit="GHz",
        description="MAMBO 1.2 mm band (240–250 GHz); 1.2 mm continuum",
    ),
    # Bolocam
    "bolocam_1p1mm": SyntheticBand(
        name="bolocam_1p1mm",
        facility="Bolocam",
        lo=270.0,
        hi=280.0,
        unit="GHz",
        description="Bolocam 1.1 mm band (270–280 GHz); 1.1 mm continuum",
    ),
    # TolTEC (LMT)
    "toltec_1p1mm": SyntheticBand(
        name="toltec_1p1mm",
        facility="TolTEC",
        lo=270.0,
        hi=300.0,
        unit="GHz",
        description="TolTEC 1.1 mm band (270–300 GHz); 1.1 mm continuum",
    ),
    "toltec_1p4mm": SyntheticBand(
        name="toltec_1p4mm",
        facility="TolTEC",
        lo=210.0,
        hi=240.0,
        unit="GHz",
        description="TolTEC 1.4 mm band (210–240 GHz); 1.4 mm continuum",
    ),
    "toltec_2p0mm": SyntheticBand(
        name="toltec_2p0mm",
        facility="TolTEC",
        lo=140.0,
        hi=170.0,
        unit="GHz",
        description="TolTEC 2.0 mm band (140–170 GHz); 2.0 mm continuum",
    ),
    # SMA (Submillimeter Array)
    "sma_870um": SyntheticBand(
        name="sma_870um",
        facility="SMA",
        lo=330.0,
        hi=360.0,
        unit="GHz",
        description="SMA 870 µm band (330–360 GHz); 0.87 mm continuum",
    ),
    "sma_1p3mm": SyntheticBand(
        name="sma_1p3mm",
        facility="SMA",
        lo=210.0,
        hi=240.0,
        unit="GHz",
        description="SMA 1.3 mm band (210–240 GHz); 1.3 mm continuum",
    ),
}


def load_synthetic_band(name: str) -> FilterCurve:
    """Load a synthetic top-hat bandpass by name.

    Parameters
    ----------
    name : str
        Synthetic band name from :data:`SYNTHETIC_BAND_REGISTRY`
        (e.g., "alma_band6", "chandra_soft", "spt_150ghz").

    Returns
    -------
    FilterCurve
        Top-hat rectangular bandpass converted to wavelength space (Angstrom),
        with uniform transmission = 1.0.

    Raises
    ------
    KeyError
        If *name* is not in :data:`SYNTHETIC_BAND_REGISTRY`.

    Notes
    -----
    **JIT-compatible**: no — file I/O is required at load time.
    Once loaded, the returned FilterCurve can be used in JIT functions.

    **Top-hat approximation**: The bandpass is a perfect rectangular function
    in wavelength space. This is appropriate for continuum photometry where the
    source spectrum is smooth across the band (as for most galaxy SEDs), but
    NOT appropriate for narrow spectral features or where detector response
    shape matters.

    **Unit conversion**: GHz → Å uses λ = c / ν with c in Å/s.
    keV → Å uses λ = hc / E with hc in keV·Å.

    **X-ray photometric convention**: The default tengri photometry convention
    is photon-counting Bessell (w = 1/λ; ADR-0017). For X-ray work, the energy
    convention (w = 1/E) may be more appropriate. Pass
    ``Photometry.from_names(..., convention="energy")`` to use it.

    References
    ----------
    See module docstring for facility documentation and calibration sources.

    Examples
    --------
    >>> from tengri.observation.filters import load_synthetic_band
    >>> alma6 = load_synthetic_band("alma_band6")
    >>> alma6.name
    'alma_band6'
    >>> float(alma6.wave[0])  # shortest wavelength in band [Angstrom]
    1.0909090...
    """
    from tengri.observation.filters import load_tophat_filter
    from tengri.observation.photometry import FilterCurve as _FilterCurve

    if name not in SYNTHETIC_BAND_REGISTRY:
        raise KeyError(
            f"Unknown synthetic band '{name}'. Available: {sorted(SYNTHETIC_BAND_REGISTRY.keys())}"
        )

    band = SYNTHETIC_BAND_REGISTRY[name]

    # Convert band edges to wavelength (Angstrom)
    if band.unit == "GHz":
        # λ [Å] = c [Å/s] / ν [Hz] = c [Å/s] / (ν [GHz] * 1e9)
        lo_aa = C_AA / (band.hi * 1e9)  # high freq = short wavelength
        hi_aa = C_AA / (band.lo * 1e9)  # low freq = long wavelength
    elif band.unit == "keV":
        # λ [Å] = hc [keV·Å] / E [keV]
        lo_aa = HC_KEV_ANGSTROM / band.hi  # high energy = short wavelength
        hi_aa = HC_KEV_ANGSTROM / band.lo  # low energy = long wavelength
    else:
        raise ValueError(f"Unknown unit {band.unit} for {name}")

    center_aa = (lo_aa + hi_aa) / 2.0
    width_aa = hi_aa - lo_aa

    # Use the existing load_tophat_filter function to build the curve
    result: _FilterCurve = load_tophat_filter(center_aa, width_aa, name=name)
    return result


def load_alma_band(band: int, name: str | None = None) -> FilterCurve:
    """Create a synthetic top-hat filter for an ALMA continuum band.

    This function is kept in filters/__init__.py for backward compatibility,
    and handles both the standard registry names and custom names.

    Parameters
    ----------
    band : int
        ALMA band number (1–10).
    name : str, optional
        Label for the filter. Defaults to ``"alma_band{N}"``.

    Returns
    -------
    FilterCurve
        Top-hat bandpass in observed-frame wavelengths (Angstrom).

    Raises
    ------
    ValueError
        If band is not in 1–10.

    Notes
    -----
    Band definitions follow the ALMA Cycle 11 receiver specifications.
    Wavelengths are in the *observed* frame — the filter should be applied
    at the observed frequency. For a source at redshift *z*, Band N probes
    rest-frame wavelength λ_rest = λ_obs / (1 + z).

    Examples
    --------
    >>> fc = load_alma_band(6)  # 1.23 mm continuum (211–275 GHz)
    >>> fc = load_alma_band(7)  # 870 μm continuum (275–373 GHz)
    """
    if band not in _ALMA_BANDS_GHZ:
        valid = sorted(_ALMA_BANDS_GHZ)
        raise ValueError(f"ALMA band must be one of {valid}, got {band}.")

    lo_ghz, hi_ghz = _ALMA_BANDS_GHZ[band]
    # High frequency = short wavelength and vice versa.
    lo_aa = C_AA / (hi_ghz * 1e9)
    hi_aa = C_AA / (lo_ghz * 1e9)
    center_aa = (lo_aa + hi_aa) / 2.0
    width_aa = hi_aa - lo_aa

    label = name if name is not None else f"alma_band{band}"
    # Use the existing load_tophat_filter function to build the curve
    from tengri.observation.filters import load_tophat_filter

    return load_tophat_filter(center_aa, width_aa, name=label)


def list_synthetic_bands(facility: str | None = None):
    """List every synthetic top-hat band, as a table.

    Parameters
    ----------
    facility : str or None, optional
        Keep only bands whose facility contains this substring
        (case-insensitive), e.g. ``"ALMA"``, ``"NuSTAR"``, ``"SPT"``.
        Default: ``None`` (all).

    Returns
    -------
    _RegistryTable
        One row per band, with columns ``name`` (the alias
        :func:`tengri.load_filter` accepts), ``facility``, ``band`` (the edges
        in their native unit), ``lambda_eff`` and ``description``.

    Notes
    -----
    These bands are **rectangular approximations**, not measured detector
    responses — see this module's docstring. They exist because ALMA, the X-ray
    observatories and most (sub)mm cameras publish spectral windows in GHz or
    keV rather than a transmission curve, so there is nothing on the SVO Filter
    Profile Service to cache.

    This menu is deliberately separate from :func:`tengri.list_filters`, which
    lists measured SVO curves. Where a facility has both, the measured curve is
    registered as an ordinary filter and no synthetic twin is defined, so a name
    never resolves to an approximation while real data exists (SPT-SZ 150 and
    220 GHz are the worked example).

    Not JAX-compatible; a discovery helper, not a forward-model function.

    Examples
    --------
    >>> list_synthetic_bands()
    >>> list_synthetic_bands(facility="NuSTAR").names()
    """
    from tengri.registry import _RegistryTable

    bands = sorted(SYNTHETIC_BAND_REGISTRY.values(), key=lambda b: (b.facility, b.name))
    if facility is not None:
        needle = facility.lower()
        bands = [b for b in bands if needle in b.facility.lower()]

    rows = []
    for band in bands:
        curve = load_synthetic_band(band.name)
        wave = curve.wave
        rows.append(
            {
                "name": band.name,
                "kind": "synthetic_band",
                "facility": band.facility,
                "band": f"{band.lo:g}-{band.hi:g} {band.unit}",
                "lambda_eff": _format_band_wavelength(float((wave[0] + wave[-1]) / 2.0)),
                "description": band.description,
                "use": f'tengri.load_filter("{band.name}")',
            }
        )
    return _RegistryTable(rows)


def _format_band_wavelength(wave_aa: float) -> str:
    """Format a band center in whichever unit keeps it readable."""
    if wave_aa >= 1e7:
        return f"{wave_aa / 1e7:.2f} mm"
    if wave_aa >= 1e4:
        return f"{wave_aa / 1e4:.2f} μm"
    if wave_aa >= 1.0:
        return f"{wave_aa:.1f} Å"
    return f"{wave_aa:.4f} Å"
