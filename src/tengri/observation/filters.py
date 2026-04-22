"""Photometric filter management via the SVO Filter Profile Service.

Downloads, caches, and loads photometric filter transmission curves from
the Spanish Virtual Observatory (SVO) Filter Profile Service:
https://svo2.cab.inta-csic.es/theory/fps/

Uses astroquery.svo_fps for downloads. Filters are cached as two-column
text files (wavelength in Angstrom, transmission) under a configurable
cache directory.

Note on ALMA / interferometers
------------------------------
ALMA and similar interferometric arrays do not use photometric bandpass
filters — observations are defined by spectral windows in GHz. SVO has
no ALMA entries. For SED fitting at (sub)mm continuum frequencies, use
``load_tophat_filter()`` to create a synthetic rectangular bandpass
centered on the observed frequency.

Available submm photometric instruments (real bandpasses on SVO):
  JCMT SCUBA-2 : 450 μm, 850 μm  → scuba2_450, scuba2_850
  APEX LABOCA  : 870 μm           → laboca_870
  APEX SABOCA  : 350 μm           → saboca_350
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tengri.observation.photometry import FilterCurve

# ── Registry: short name -> SVO Filter Profile Service ID ─────────

FILTER_REGISTRY: dict[str, str] = {
    # ── UV / optical — ground-based survey imagers ────────────────
    # SDSS
    "sdss_u": "SLOAN/SDSS.u",
    "sdss_g": "SLOAN/SDSS.g",
    "sdss_r": "SLOAN/SDSS.r",
    "sdss_i": "SLOAN/SDSS.i",
    "sdss_z": "SLOAN/SDSS.z",
    # LSST / Rubin Observatory
    "lsst_u": "LSST/LSST.u",
    "lsst_g": "LSST/LSST.g",
    "lsst_r": "LSST/LSST.r",
    "lsst_i": "LSST/LSST.i",
    "lsst_z": "LSST/LSST.z",
    "lsst_y": "LSST/LSST.y",
    # Pan-STARRS PS1
    "ps1_g": "PAN-STARRS/PS1.g",
    "ps1_r": "PAN-STARRS/PS1.r",
    "ps1_i": "PAN-STARRS/PS1.i",
    "ps1_z": "PAN-STARRS/PS1.z",
    "ps1_y": "PAN-STARRS/PS1.y",
    # DES / DECam (CTIO)
    "des_g": "CTIO/DECam.g_filter",
    "des_r": "CTIO/DECam.r_filter",
    "des_i": "CTIO/DECam.i_filter",
    "des_z": "CTIO/DECam.z_filter",
    "des_y": "CTIO/DECam.Y",
    # CFHT MegaCam (MP9301/MP9302 u variants; standard griz)
    "megacam_u": "CFHT/MegaCam.u",  # MP9301 (old u)
    "megacam_u2": "CFHT/MegaCam.u_1",  # MP9302 (updated u)
    "megacam_g": "CFHT/MegaCam.g",
    "megacam_g2": "CFHT/MegaCam.g_1",
    "megacam_r": "CFHT/MegaCam.r",
    "megacam_r2": "CFHT/MegaCam.r_1",
    "megacam_i": "CFHT/MegaCam.i",
    "megacam_i2": "CFHT/MegaCam.i_1",
    "megacam_i3": "CFHT/MegaCam.i_2",
    "megacam_z": "CFHT/MegaCam.z",
    "megacam_z2": "CFHT/MegaCam.z_1",
    # Subaru HSC (r2/i2 = second-gen filters used in HSC-SSP PDR2+)
    "hsc_g": "Subaru/HSC.g",
    "hsc_r": "Subaru/HSC.r2_filter",
    "hsc_i": "Subaru/HSC.i2_filter",
    "hsc_z": "Subaru/HSC.z",
    "hsc_y": "Subaru/HSC.Y",
    # Subaru SuprimeCam — broadband
    "suprime_b": "Subaru/Suprime.B",
    "suprime_g": "Subaru/Suprime.g",
    "suprime_v": "Subaru/Suprime.V",
    "suprime_r": "Subaru/Suprime.r",
    "suprime_rc": "Subaru/Suprime.Rc_filter",
    "suprime_i": "Subaru/Suprime.i",
    "suprime_ic": "Subaru/Suprime.Ic_filter",
    "suprime_z": "Subaru/Suprime.z",
    "suprime_y": "Subaru/Suprime.Y_filter",
    # Subaru SuprimeCam — intermediate-band (IB, used in COSMOS photo-z)
    "suprime_ib427": "Subaru/Suprime.IB427",
    "suprime_ib464": "Subaru/Suprime.IB464",
    "suprime_ib484": "Subaru/Suprime.IB484",
    "suprime_ib505": "Subaru/Suprime.IB505",
    "suprime_ib527": "Subaru/Suprime.IB527",
    "suprime_ib574": "Subaru/Suprime.IB574",
    "suprime_ib624": "Subaru/Suprime.IB624",
    "suprime_ib679": "Subaru/Suprime.IB679",
    "suprime_ib709": "Subaru/Suprime.IB709",
    "suprime_ib738": "Subaru/Suprime.IB738",
    "suprime_ib767": "Subaru/Suprime.IB767",
    "suprime_ib827": "Subaru/Suprime.IB827",
    # Subaru SuprimeCam — narrow-band
    "suprime_na656": "Subaru/Suprime.NA656_filter",  # Hα
    "suprime_nb711": "Subaru/Suprime.NB711",
    "suprime_nb816": "Subaru/Suprime.NB816",
    "suprime_nb921": "Subaru/Suprime.NB921_filter",
    # ── UV / optical — space telescopes ───────────────────────────
    # GALEX
    "galex_fuv": "GALEX/GALEX.FUV",
    "galex_nuv": "GALEX/GALEX.NUV",
    # XMM-Newton OM (UV/optical, AGN monitoring)
    "xmm_uvw2": "XMM/OM.UVW2",
    "xmm_uvm2": "XMM/OM.UVM2",
    "xmm_uvw1": "XMM/OM.UVW1",
    "xmm_u": "XMM/OM.U",
    "xmm_b": "XMM/OM.B",
    "xmm_v": "XMM/OM.V",
    # Swift UVOT (UV/optical, transient/AGN)
    "uvot_uvw2": "Swift/UVOT.UVW2",
    "uvot_uvm2": "Swift/UVOT.UVM2",
    "uvot_uvw1": "Swift/UVOT.UVW1",
    "uvot_u": "Swift/UVOT.U",
    "uvot_b": "Swift/UVOT.B",
    "uvot_v": "Swift/UVOT.V",
    "uvot_white": "Swift/UVOT.white",
    # ── NIR — ground-based ────────────────────────────────────────
    # 2MASS
    "2mass_j": "2MASS/2MASS.J",
    "2mass_h": "2MASS/2MASS.H",
    "2mass_ks": "2MASS/2MASS.Ks",
    # VISTA / VIRCAM (ESO Paranal; UltraVISTA, VIDEO, VHS)
    "vista_z": "Paranal/VISTA.Z",
    "vista_y": "Paranal/VISTA.Y",
    "vista_j": "Paranal/VISTA.J",
    "vista_h": "Paranal/VISTA.H",
    "vista_ks": "Paranal/VISTA.Ks",
    # UKIRT WFCAM / UKIDSS
    "ukidss_z": "UKIRT/UKIDSS.Z",
    "ukidss_y": "UKIRT/UKIDSS.Y",
    "ukidss_j": "UKIRT/UKIDSS.J",
    "ukidss_h": "UKIRT/UKIDSS.H",
    "ukidss_k": "UKIRT/UKIDSS.K",
    # ── HST (UV → NIR) ────────────────────────────────────────────
    # ACS/WFC (optical / red)
    "hst_f435w": "HST/ACS_WFC.F435W",
    "hst_f606w": "HST/ACS_WFC.F606W",
    "hst_f775w": "HST/ACS_WFC.F775W",
    "hst_f814w": "HST/ACS_WFC.F814W",
    "hst_f850lp": "HST/ACS_WFC.F850LP",
    # WFC3/IR (NIR)
    "hst_f105w": "HST/WFC3_IR.F105W",
    "hst_f125w": "HST/WFC3_IR.F125W",
    "hst_f140w": "HST/WFC3_IR.F140W",
    "hst_f160w": "HST/WFC3_IR.F160W",
    # ── JWST NIRCam — original throughput curves ──────────────────
    "jwst_f070w": "JWST/NIRCam.F070W",
    "jwst_f090w": "JWST/NIRCam.F090W",
    "jwst_f115w": "JWST/NIRCam.F115W",
    "jwst_f140m": "JWST/NIRCam.F140M",
    "jwst_f150w": "JWST/NIRCam.F150W",
    "jwst_f150w2": "JWST/NIRCam.F150W2",
    "jwst_f162m": "JWST/NIRCam.F162M",
    "jwst_f164n": "JWST/NIRCam.F164N",
    "jwst_f182m": "JWST/NIRCam.F182M",
    "jwst_f187n": "JWST/NIRCam.F187N",
    "jwst_f200w": "JWST/NIRCam.F200W",
    "jwst_f210m": "JWST/NIRCam.F210M",
    "jwst_f212n": "JWST/NIRCam.F212N",
    "jwst_f250m": "JWST/NIRCam.F250M",
    "jwst_f277w": "JWST/NIRCam.F277W",
    "jwst_f300m": "JWST/NIRCam.F300M",
    "jwst_f322w2": "JWST/NIRCam.F322W2",
    "jwst_f323n": "JWST/NIRCam.F323N",
    "jwst_f335m": "JWST/NIRCam.F335M",
    "jwst_f356w": "JWST/NIRCam.F356W",
    "jwst_f360m": "JWST/NIRCam.F360M",
    "jwst_f405n": "JWST/NIRCam.F405N",
    "jwst_f410m": "JWST/NIRCam.F410M",
    "jwst_f430m": "JWST/NIRCam.F430M",
    "jwst_f444w": "JWST/NIRCam.F444W",
    "jwst_f460m": "JWST/NIRCam.F460M",
    "jwst_f466n": "JWST/NIRCam.F466N",
    "jwst_f470n": "JWST/NIRCam.F470N",
    "jwst_f480m": "JWST/NIRCam.F480M",
    # ── JWST NIRCam2025 — recalibrated throughputs (use for data after 2025)
    "nircam25_f070w": "JWST/NIRCam2025.F070W",
    "nircam25_f090w": "JWST/NIRCam2025.F090W",
    "nircam25_f115w": "JWST/NIRCam2025.F115W",
    "nircam25_f140m": "JWST/NIRCam2025.F140M",
    "nircam25_f150w": "JWST/NIRCam2025.F150W",
    "nircam25_f150w2": "JWST/NIRCam2025.F150W2",
    "nircam25_f162m": "JWST/NIRCam2025.F162M",
    "nircam25_f164n": "JWST/NIRCam2025.F164N",
    "nircam25_f182m": "JWST/NIRCam2025.F182M",
    "nircam25_f187n": "JWST/NIRCam2025.F187N",
    "nircam25_f200w": "JWST/NIRCam2025.F200W",
    "nircam25_f210m": "JWST/NIRCam2025.F210M",
    "nircam25_f212n": "JWST/NIRCam2025.F212N",
    "nircam25_f250m": "JWST/NIRCam2025.F250M",
    "nircam25_f277w": "JWST/NIRCam2025.F277W",
    "nircam25_f300m": "JWST/NIRCam2025.F300M",
    "nircam25_f322w2": "JWST/NIRCam2025.F322W2",
    "nircam25_f323n": "JWST/NIRCam2025.F323N",
    "nircam25_f335m": "JWST/NIRCam2025.F335M",
    "nircam25_f356w": "JWST/NIRCam2025.F356W",
    "nircam25_f360m": "JWST/NIRCam2025.F360M",
    "nircam25_f405n": "JWST/NIRCam2025.F405N",
    "nircam25_f410m": "JWST/NIRCam2025.F410M",
    "nircam25_f430m": "JWST/NIRCam2025.F430M",
    "nircam25_f444w": "JWST/NIRCam2025.F444W",
    "nircam25_f460m": "JWST/NIRCam2025.F460M",
    "nircam25_f466n": "JWST/NIRCam2025.F466N",
    "nircam25_f470n": "JWST/NIRCam2025.F470N",
    "nircam25_f480m": "JWST/NIRCam2025.F480M",
    # ── JWST NIRISS ───────────────────────────────────────────────
    "niriss_f090w": "JWST/NIRISS.F090W",
    "niriss_f115w": "JWST/NIRISS.F115W",
    "niriss_f140m": "JWST/NIRISS.F140M",
    "niriss_f150w": "JWST/NIRISS.F150W",
    "niriss_f158m": "JWST/NIRISS.F158M",
    "niriss_f200w": "JWST/NIRISS.F200W",
    "niriss_f277w": "JWST/NIRISS.F277W",
    "niriss_f356w": "JWST/NIRISS.F356W",
    "niriss_f380m": "JWST/NIRISS.F380M",
    "niriss_f430m": "JWST/NIRISS.F430M",
    "niriss_f444w": "JWST/NIRISS.F444W",
    "niriss_f480m": "JWST/NIRISS.F480M",
    # ── JWST NIRSpec (disperser+filter combos used as pseudo-photometry)
    "nirspec_prism": "JWST/NIRSpec.Prism",
    "nirspec_g235m": "JWST/NIRSpec.G235M_F170LP",
    "nirspec_g235h": "JWST/NIRSpec.G235H_F170LP",
    "nirspec_g395m": "JWST/NIRSpec.G395M_F290LP",
    "nirspec_g395h": "JWST/NIRSpec.G395H_F290LP",
    # ── JWST MIRI (5.6–25.5 μm; W=wide, C=coronagraphic) ──────────
    "miri_f560w": "JWST/MIRI.F560W",
    "miri_f770w": "JWST/MIRI.F770W",
    "miri_f1000w": "JWST/MIRI.F1000W",
    "miri_f1065c": "JWST/MIRI.F1065C",
    "miri_f1130w": "JWST/MIRI.F1130W",
    "miri_f1140c": "JWST/MIRI.F1140C",
    "miri_f1280w": "JWST/MIRI.F1280W",
    "miri_f1500w": "JWST/MIRI.F1500W",
    "miri_f1550c": "JWST/MIRI.F1550C",
    "miri_f1800w": "JWST/MIRI.F1800W",
    "miri_f2100w": "JWST/MIRI.F2100W",
    "miri_f2300c": "JWST/MIRI.F2300C",
    "miri_f2550w": "JWST/MIRI.F2550W",
    # ── Roman Space Telescope / WFI ───────────────────────────────
    "roman_f062": "Roman/WFI.F062",
    "roman_f087": "Roman/WFI.F087",
    "roman_f106": "Roman/WFI.F106",
    "roman_f129": "Roman/WFI.F129",
    "roman_f158": "Roman/WFI.F158",
    "roman_f184": "Roman/WFI.F184",
    "roman_f213": "Roman/WFI.F213",
    # ── Euclid ────────────────────────────────────────────────────
    "euclid_vis": "Euclid/VIS.vis",
    "euclid_y": "Euclid/NISP.Y",
    "euclid_j": "Euclid/NISP.J",
    "euclid_h": "Euclid/NISP.H",
    # ── Spitzer (mid-IR) ──────────────────────────────────────────
    # IRAC (3.6–8 μm)
    "irac_36": "Spitzer/IRAC.I1",
    "irac_45": "Spitzer/IRAC.I2",
    "irac_58": "Spitzer/IRAC.I3",
    "irac_80": "Spitzer/IRAC.I4",
    # MIPS (24–160 μm)
    "mips_24": "Spitzer/MIPS.24mu",
    "mips_70": "Spitzer/MIPS.70mu",
    "mips_160": "Spitzer/MIPS.160mu",
    # ── WISE (mid-IR all-sky) ─────────────────────────────────────
    "wise_w1": "WISE/WISE.W1",
    "wise_w2": "WISE/WISE.W2",
    "wise_w3": "WISE/WISE.W3",
    "wise_w4": "WISE/WISE.W4",
    # ── AKARI (near-IR + far-IR) ──────────────────────────────────
    # IRC (2–24 μm)
    "akari_n2": "AKARI/IRC.N2",
    "akari_n3": "AKARI/IRC.N3",
    "akari_n4": "AKARI/IRC.N4",
    "akari_s7": "AKARI/IRC.S7",
    "akari_s9w": "AKARI/IRC.S9W",
    "akari_s11": "AKARI/IRC.S11",
    "akari_l15": "AKARI/IRC.L15",
    "akari_l18w": "AKARI/IRC.L18W",
    "akari_l24": "AKARI/IRC.L24",
    # FIS (60–160 μm)
    "akari_n60": "AKARI/FIS.N60",
    "akari_wides": "AKARI/FIS.WIDE-S",
    "akari_widel": "AKARI/FIS.WIDE-L",
    "akari_n160": "AKARI/FIS.N160",
    # ── Herschel (far-IR / submm) ─────────────────────────────────
    # PACS (70–160 μm)
    "herschel_70": "Herschel/Pacs.blue",
    "herschel_100": "Herschel/Pacs.green",
    "herschel_160": "Herschel/Pacs.red",
    # SPIRE (250–500 μm) — point-source and extended-source variants
    "herschel_250": "Herschel/SPIRE.PSW",
    "herschel_350": "Herschel/SPIRE.PMW",
    "herschel_500": "Herschel/SPIRE.PLW",
    "herschel_250_ext": "Herschel/SPIRE.PSW_ext",
    "herschel_350_ext": "Herschel/SPIRE.PMW_ext",
    "herschel_500_ext": "Herschel/SPIRE.PLW_ext",
    # ── (Sub)millimeter photometric instruments ───────────────────
    # JCMT SCUBA-2
    "scuba2_450": "JCMT/SCUBA2.450GHz",
    "scuba2_850": "JCMT/SCUBA2.850GHz",
    # APEX bolometer cameras
    "laboca_870": "APEX/LABOCA.345GHz",
    "saboca_350": "APEX/SABOCA.852GHz",
    # ── Generic / standard photometric systems ────────────────────
    # Johnson UBVRI
    "johnson_u": "Generic/Johnson.U",
    "johnson_b": "Generic/Johnson.B",
    "johnson_v": "Generic/Johnson.V",
    "johnson_r": "Generic/Johnson.R",
    "johnson_i": "Generic/Johnson.I",
    "johnson_j": "2MASS/2MASS.J",  # 2MASS J as proxy for Johnson J
    # Cousins RI
    "cousins_r": "Generic/Cousins.R",
    "cousins_i": "Generic/Cousins.I",
}

_DEFAULT_CACHE_DIR = "data/filters"

# Speed of light in Å/s — used for GHz ↔ Å conversion.
_C_AA_S = 2.99792458e18

# ALMA receiver band definitions (ALMA Cycle 11 specifications).
# Each entry maps band number → (lo_ghz, hi_ghz) at the edges of the
# receiver bandwidth.  The full band width is used as the top-hat width
# so that continuum photometry integrates over the realistic frequency
# coverage rather than an arbitrarily narrow window.
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


# ── Filter metadata: facility and description for rich listing ────

_FACILITY_FROM_PREFIX: dict[str, str] = {
    "sdss": "SDSS",
    "lsst": "LSST/Rubin",
    "ps1": "Pan-STARRS",
    "des": "DES/DECam",
    "megacam": "CFHT/MegaCam",
    "hsc": "Subaru/HSC",
    "suprime": "Subaru/SuprimeCam",
    "galex": "GALEX",
    "xmm": "XMM-Newton/OM",
    "uvot": "Swift/UVOT",
    "2mass": "2MASS",
    "vista": "VISTA/VIRCAM",
    "ukidss": "UKIRT/WFCAM",
    "hst": "HST",
    "jwst": "JWST/NIRCam",
    "nircam25": "JWST/NIRCam2025",
    "niriss": "JWST/NIRISS",
    "nirspec": "JWST/NIRSpec",
    "miri": "JWST/MIRI",
    "roman": "Roman/WFI",
    "euclid": "Euclid",
    "irac": "Spitzer/IRAC",
    "mips": "Spitzer/MIPS",
    "wise": "WISE",
    "akari": "AKARI",
    "herschel": "Herschel",
    "scuba2": "JCMT/SCUBA-2",
    "laboca": "APEX/LABOCA",
    "saboca": "APEX/SABOCA",
    "johnson": "Generic/Johnson",
    "cousins": "Generic/Cousins",
}


def _infer_facility(name: str) -> str:
    """Infer facility from filter short name prefix."""
    for prefix, facility in _FACILITY_FROM_PREFIX.items():
        if name.startswith(prefix):
            return facility
    return "Other"


# ── Filter property computation (pure numpy, no JAX) ──────────────


def compute_effective_wavelength(wave: np.ndarray, trans: np.ndarray) -> float:
    """Photon-counting effective wavelength: λ_eff = ∫T·λ·dλ / ∫T·dλ.

    Parameters
    ----------
    wave : array, shape (n_wave,)
        Wavelength [Angstrom].
    trans : array, shape (n_wave,)
        Transmission (dimensionless [0, 1]).

    Returns
    -------
    float
        Effective wavelength [Angstrom].

    Notes
    -----
    Not JAX-compatible (uses NumPy). Intended for filter metadata
    computation, not forward model evaluation.

    """
    num = np.trapz(trans * wave, wave)
    den = np.trapz(trans, wave)
    if den == 0:
        return 0.0
    return float(num / den)


def compute_fwhm(wave: np.ndarray, trans: np.ndarray) -> float:
    """Full width at half maximum of the transmission curve.

    Parameters
    ----------
    wave : array, shape (n_wave,)
        Wavelength [Angstrom].
    trans : array, shape (n_wave,)
        Transmission (dimensionless [0, 1]).

    Returns
    -------
    float
        FWHM [Angstrom]. Returns 0 if the curve never exceeds half-max.

    Notes
    -----
    Not JAX-compatible (uses NumPy). Intended for filter metadata
    computation, not forward model evaluation.

    """
    peak = np.max(trans)
    if peak == 0:
        return 0.0
    half_max = peak / 2.0
    above = wave[trans >= half_max]
    if len(above) < 2:
        return 0.0
    return float(above[-1] - above[0])


def _format_wavelength(wave_aa: float) -> str:
    """Format wavelength with appropriate units."""
    if wave_aa >= 1e7:
        return f"{wave_aa / 1e8:.2f} cm"
    elif wave_aa >= 1e4:
        return f"{wave_aa / 1e4:.2f} \u03bcm"
    else:
        return f"{wave_aa:.0f} \u00c5"


def filter_info(name: str, *, cache_dir: str | None = None) -> dict:
    """Return metadata for a single filter.

    Loads the transmission curve from the local cache (downloading from
    SVO if needed) and computes derived properties.

    Parameters
    ----------
    name : str
        Short filter name from ``FILTER_REGISTRY``.
    cache_dir : str, optional
        Override cache directory.

    Returns
    -------
    dict
        Keys: ``name``, ``svo_id``, ``facility``, ``lambda_eff_aa``,
        ``fwhm_aa``, ``lambda_eff_str``, ``fwhm_str``.

    Raises
    ------
    KeyError
        If *name* is not in the registry.

    """
    if name not in FILTER_REGISTRY:
        raise KeyError(f"Unknown filter '{name}'. Use list_available_filters() to see options.")
    kwargs = {"cache_dir": cache_dir} if cache_dir is not None else {}
    fc = load_filter(name, **kwargs)
    wave_np = np.asarray(fc.wave)
    trans_np = np.asarray(fc.trans)
    lam_eff = compute_effective_wavelength(wave_np, trans_np)
    fwhm = compute_fwhm(wave_np, trans_np)
    return {
        "name": name,
        "svo_id": FILTER_REGISTRY[name],
        "facility": _infer_facility(name),
        "lambda_eff_aa": lam_eff,
        "fwhm_aa": fwhm,
        "lambda_eff_str": _format_wavelength(lam_eff),
        "fwhm_str": _format_wavelength(fwhm),
    }


# ── Internal helpers ──────────────────────────────────────────────


def _svo_id_to_filename(svo_id: str) -> str:
    """Convert SVO filter ID to a safe filename."""
    return svo_id.replace("/", "_").replace(".", "_") + ".dat"


def _save_filter(filepath: Path, wave: np.ndarray, trans: np.ndarray) -> None:
    """Write wavelength and transmission columns to a two-column text file."""
    header = "# Wavelength(Angstrom)  Transmission"
    np.savetxt(str(filepath), np.column_stack([wave, trans]), header=header, fmt="%.6e")


def _load_filter_file(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read and return wavelength and transmission columns from a text file."""
    data = np.loadtxt(str(filepath))
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Filter file {filepath} must have at least 2 columns "
            f"(wavelength, transmission). Got shape {data.shape}."
        )
    return data[:, 0], data[:, 1]


def _fetch_from_svo(svo_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Fetch filter curve from SVO using astroquery.svo_fps."""
    try:
        from astroquery.svo_fps import SvoFps
    except ImportError as exc:
        raise ImportError(
            "astroquery is required to download filters from SVO. "
            "Install it with:  pip install astroquery"
        ) from exc

    table = SvoFps.get_transmission_data(svo_id)
    wave = np.asarray(table["Wavelength"], dtype=float)
    trans = np.asarray(table["Transmission"], dtype=float)
    if len(wave) == 0:
        raise ValueError(
            f"SVO returned zero rows for filter '{svo_id}'. Check that the filter ID is correct."
        )
    return wave, trans


# ── Public API ────────────────────────────────────────────────────


def download_filter(
    svo_id: str,
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Download a single filter from the SVO Filter Profile Service.

    If the filter is already cached on disk, it is loaded from the cache
    instead of re-downloading.

    Parameters
    ----------
    svo_id : str
        SVO filter identifier (e.g. ``"JWST/NIRCam.F200W"``).
    cache_dir : str
        Directory for cached filter files.

    Returns
    -------
    wave : ndarray
        Wavelength in Angstrom.
    trans : ndarray
        Transmission (dimensionless).

    Raises
    ------
    ImportError
        If ``astroquery`` is not installed.
    ValueError
        If SVO returns no data for the given filter ID.

    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    filepath = cache_path / _svo_id_to_filename(svo_id)

    if filepath.exists():
        return _load_filter_file(filepath)

    wave, trans = _fetch_from_svo(svo_id)

    order = np.argsort(wave)
    wave = wave[order]
    trans = trans[order]

    _save_filter(filepath, wave, trans)
    return wave, trans


def load_filter(
    name: str,
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> FilterCurve:
    """Load a filter by its short registry name.

    Downloads from SVO if not already cached.

    Parameters
    ----------
    name : str
        Short name from ``FILTER_REGISTRY`` (e.g. ``"jwst_f200w"``).
    cache_dir : str
        Directory for cached filter files.

    Returns
    -------
    FilterCurve
        Filter with wavelength (Angstrom), raw transmission as returned
        by SVO, and name.  Transmission values are not normalized — the
        absolute scale cancels in the photometry integral
        ``∫fλTλdλ / ∫Tλdλ``.

    Raises
    ------
    KeyError
        If *name* is not in ``FILTER_REGISTRY``.

    """
    if name not in FILTER_REGISTRY:
        raise KeyError(
            f"Unknown filter '{name}'. Use list_available_filters() to see "
            f"valid names, or use load_custom_filter() for arbitrary files."
        )

    svo_id = FILTER_REGISTRY[name]
    wave, trans = download_filter(svo_id, cache_dir=cache_dir)
    return FilterCurve(wave=jnp.array(wave), trans=jnp.array(trans), name=name)


def load_filter_set(
    names: list[str],
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> tuple[list[jnp.ndarray], list[jnp.ndarray], list[FilterCurve]]:
    """Load multiple filters by short name.

    Parameters
    ----------
    names : list of str
        Short names from ``FILTER_REGISTRY``.
    cache_dir : str
        Directory for cached filter files. Default: ``"data/filters"``.

    Returns
    -------
    filter_waves : list of jnp.ndarray
        Wavelength arrays per filter, each shape ``(n_wave,)`` [Angstrom].
    filter_trans : list of jnp.ndarray
        Transmission arrays per filter, each shape ``(n_wave,)``
        (dimensionless [0, 1]).
    filter_curves : list of FilterCurve
        Full FilterCurve objects with wavelength, transmission, and name.

    Raises
    ------
    KeyError
        If any name is not in ``FILTER_REGISTRY``.

    Notes
    -----
    Filters are downloaded from SVO on first use and cached locally.
    See ``load_filter()`` for single-filter loading.

    Examples
    --------
    >>> from tengri import load_filter_set
    >>> waves, trans, curves = load_filter_set(["sdss_r", "sdss_i", "sdss_z"])
    >>> len(curves)
    3
    >>> curves[0].name
    'sdss_r'
    """
    filter_waves: list[jnp.ndarray] = []
    filter_trans: list[jnp.ndarray] = []
    filter_curves: list[FilterCurve] = []
    for name in names:
        fc = load_filter(name, cache_dir=cache_dir)
        filter_waves.append(fc.wave)
        filter_trans.append(fc.trans)
        filter_curves.append(fc)
    return filter_waves, filter_trans, filter_curves


def load_custom_filter(filepath: str) -> FilterCurve:
    """Load a custom filter from a two-column text file.

    Parameters
    ----------
    filepath : str
        Path to a text file with columns: wavelength (Angstrom),
        transmission.

    Returns
    -------
    FilterCurve
        Filter with raw transmission values (not normalized).  The
        absolute scale cancels in the photometry integral
        ``∫fλTλdλ / ∫Tλdλ``.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file format is invalid.

    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Filter file not found: {filepath}")

    wave, trans = _load_filter_file(path)

    order = np.argsort(wave)
    wave, trans = wave[order], trans[order]

    return FilterCurve(wave=jnp.array(wave), trans=jnp.array(trans), name=path.stem)


def load_tophat_filter(
    wave_center_aa: float,
    width_aa: float,
    name: str = "",
    n_points: int = 50,
) -> FilterCurve:
    """Create a synthetic top-hat filter (e.g. for ALMA continuum bands).

    Use this when the photometric measurement does not correspond to a
    standard bandpass on SVO — for example, an ALMA continuum flux at a
    given observed frequency.

    Parameters
    ----------
    wave_center_aa : float
        Central wavelength [Angstrom].
    width_aa : float
        Full width of the top-hat [Angstrom].
    name : str
        Label for this filter (e.g. ``"alma_band6"``). Default: empty string.
    n_points : int
        Number of wavelength samples. Default: 50.

    Returns
    -------
    FilterCurve
        Rectangular transmission curve with uniform transmission = 1.0.

    Notes
    -----
    Useful for continuum photometry measurements (e.g., ALMA, SCUBA-2)
    that are defined in frequency space rather than bandpass shape.

    """
    wave = jnp.linspace(wave_center_aa - width_aa / 2, wave_center_aa + width_aa / 2, n_points)
    trans = jnp.ones(n_points)
    return FilterCurve(wave=wave, trans=trans, name=name)


def load_alma_band(band: int, name: str | None = None) -> FilterCurve:
    """Create a synthetic top-hat filter for an ALMA continuum band.

    ALMA is an interferometric array with no entries on the SVO Filter
    Profile Service.  This function constructs a rectangular bandpass
    spanning the full receiver bandwidth of the requested band, which is
    appropriate for fitting SED continuum photometry.

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

    Examples
    --------
    >>> fc = load_alma_band(6)  # 1.23 mm continuum (211–275 GHz)
    >>> fc = load_alma_band(7)  # 870 μm continuum (275–373 GHz)

    Notes
    -----
    Band definitions follow the ALMA Cycle 11 receiver specifications.
    Wavelengths are in the *observed* frame — the filter should be applied
    at the observed frequency.  For a source at redshift *z*, Band N probes
    rest-frame wavelength λ_rest = λ_obs / (1 + z).

    """
    if band not in _ALMA_BANDS_GHZ:
        valid = sorted(_ALMA_BANDS_GHZ)
        raise ValueError(f"ALMA band must be one of {valid}, got {band}.")

    lo_ghz, hi_ghz = _ALMA_BANDS_GHZ[band]
    # High frequency = short wavelength and vice versa.
    lo_aa = _C_AA_S / (hi_ghz * 1e9)
    hi_aa = _C_AA_S / (lo_ghz * 1e9)
    center_aa = (lo_aa + hi_aa) / 2.0
    width_aa = hi_aa - lo_aa

    label = name if name is not None else f"alma_band{band}"
    return load_tophat_filter(center_aa, width_aa, name=label)


def list_available_filters(
    *,
    group_by: str = "facility",
    compute_properties: bool = False,
    cache_dir: str | None = None,
) -> dict[str, str]:
    """Print and return the filter registry, optionally grouped by facility.

    Parameters
    ----------
    group_by : str
        Grouping key. ``"facility"`` (default) groups by telescope/instrument.
        ``"none"`` lists filters alphabetically without grouping.
        Default: ``"facility"``.
    compute_properties : bool
        If ``True``, load each filter's transmission curve and display
        effective wavelength and FWHM columns. This triggers SVO
        downloads for any filters not yet cached. Default: ``False``.
    cache_dir : str, optional
        Override cache directory for filter downloads. Default: ``None``.

    Returns
    -------
    dict
        Copy of ``FILTER_REGISTRY`` (short name -> SVO ID). Also prints
        the registry to stdout in human-readable format.

    Notes
    -----
    This function is primarily for interactive exploration of available
    filters. For programmatic use, access ``FILTER_REGISTRY`` directly.

    """
    if group_by == "none":
        _print_flat_listing(compute_properties, cache_dir)
    else:
        _print_grouped_listing(compute_properties, cache_dir)
    return dict(FILTER_REGISTRY)


def _print_flat_listing(compute_properties: bool, cache_dir: str | None) -> None:
    """Print filter registry as a flat (non-grouped) alphabetical list."""
    if compute_properties:
        hdr = f"{'Name':<22s} {'SVO ID':<35s} {'lambda_eff':>12s} {'FWHM':>10s}"
        print(hdr)
        print("-" * len(hdr))
        for name in sorted(FILTER_REGISTRY):
            kwargs = {"cache_dir": cache_dir} if cache_dir is not None else {}
            info = filter_info(name, **kwargs)
            print(
                f"{name:<22s} {info['svo_id']:<35s} "
                f"{info['lambda_eff_str']:>12s} {info['fwhm_str']:>10s}"
            )
    else:
        hdr = f"{'Name':<22s} {'SVO ID':<35s}"
        print(hdr)
        print("-" * len(hdr))
        for name, svo_id in sorted(FILTER_REGISTRY.items()):
            print(f"{name:<22s} {svo_id:<35s}")
    print(f"\nTotal: {len(FILTER_REGISTRY)} filters")


def _print_grouped_listing(compute_properties: bool, cache_dir: str | None) -> None:
    """Print filter registry grouped by facility/telescope."""
    groups: dict[str, list[str]] = {}
    for name in sorted(FILTER_REGISTRY):
        fac = _infer_facility(name)
        groups.setdefault(fac, []).append(name)

    for fac in sorted(groups):
        names = groups[fac]
        print(f"\n{'=' * 60}")
        print(f"  {fac}  ({len(names)} filters)")
        print(f"{'=' * 60}")
        if compute_properties:
            hdr = f"  {'Name':<22s} {'lambda_eff':>12s} {'FWHM':>10s}"
            print(hdr)
            print(f"  {'-' * (len(hdr) - 2)}")
            kwargs = {"cache_dir": cache_dir} if cache_dir is not None else {}
            for name in names:
                info = filter_info(name, **kwargs)
                print(f"  {name:<22s} {info['lambda_eff_str']:>12s} {info['fwhm_str']:>10s}")
        else:
            for name in names:
                print(f"  {name:<22s} {FILTER_REGISTRY[name]}")
    print(f"\nTotal: {len(FILTER_REGISTRY)} filters across {len(groups)} facilities")
