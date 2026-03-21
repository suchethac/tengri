"""Photometric filter management via the SVO Filter Profile Service.

Downloads, caches, and loads photometric filter transmission curves from
the Spanish Virtual Observatory (SVO) Filter Profile Service:
https://svo2.cab.inta-csic.es/theory/fps/

Filters are cached as simple two-column text files (wavelength in Angstrom,
transmission) under a configurable cache directory.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import jax.numpy as jnp
import numpy as np

from tengri.models.observation.photometry import FilterCurve

# ---------------------------------------------------------------------------
# Registry: short name -> SVO Filter Profile Service ID
# ---------------------------------------------------------------------------

FILTER_REGISTRY: dict[str, str] = {
    # SDSS
    "sdss_u": "SLOAN/SDSS.u",
    "sdss_g": "SLOAN/SDSS.g",
    "sdss_r": "SLOAN/SDSS.r",
    "sdss_i": "SLOAN/SDSS.i",
    "sdss_z": "SLOAN/SDSS.z",
    # LSST / Rubin
    "lsst_u": "LSST/LSST.u",
    "lsst_g": "LSST/LSST.g",
    "lsst_r": "LSST/LSST.r",
    "lsst_i": "LSST/LSST.i",
    "lsst_z": "LSST/LSST.z",
    "lsst_y": "LSST/LSST.y",
    # HST ACS/WFC
    "hst_f435w": "HST/ACS_WFC.F435W",
    "hst_f606w": "HST/ACS_WFC.F606W",
    "hst_f775w": "HST/ACS_WFC.F775W",
    "hst_f814w": "HST/ACS_WFC.F814W",
    "hst_f850lp": "HST/ACS_WFC.F850LP",
    # HST WFC3/IR
    "hst_f105w": "HST/WFC3_IR.F105W",
    "hst_f125w": "HST/WFC3_IR.F125W",
    "hst_f140w": "HST/WFC3_IR.F140W",
    "hst_f160w": "HST/WFC3_IR.F160W",
    # JWST NIRCam
    "jwst_f090w": "JWST/NIRCam.F090W",
    "jwst_f115w": "JWST/NIRCam.F115W",
    "jwst_f150w": "JWST/NIRCam.F150W",
    "jwst_f200w": "JWST/NIRCam.F200W",
    "jwst_f277w": "JWST/NIRCam.F277W",
    "jwst_f356w": "JWST/NIRCam.F356W",
    "jwst_f410m": "JWST/NIRCam.F410M",
    "jwst_f444w": "JWST/NIRCam.F444W",
    # Roman / WFI
    "roman_f062": "Roman/WFI.F062",
    "roman_f087": "Roman/WFI.F087",
    "roman_f106": "Roman/WFI.F106",
    "roman_f129": "Roman/WFI.F129",
    "roman_f158": "Roman/WFI.F158",
    "roman_f184": "Roman/WFI.F184",
    "roman_f213": "Roman/WFI.F213",
    # Euclid
    "euclid_vis": "Euclid/VIS.vis",
    "euclid_y": "Euclid/NISP.Y",
    "euclid_j": "Euclid/NISP.J",
    "euclid_h": "Euclid/NISP.H",
    # Subaru HSC
    "hsc_g": "Subaru/HSC.g2",
    "hsc_r": "Subaru/HSC.r2",
    "hsc_i": "Subaru/HSC.i2",
    "hsc_z": "Subaru/HSC.z",
    "hsc_y": "Subaru/HSC.Y",
    # DES / DECam
    "des_g": "CTIO/DECam.g_filter",
    "des_r": "CTIO/DECam.r_filter",
    "des_i": "CTIO/DECam.i_filter",
    "des_z": "CTIO/DECam.z_filter",
    "des_Y": "CTIO/DECam.Y",
    # GALEX
    "galex_fuv": "GALEX/GALEX.FUV",
    "galex_nuv": "GALEX/GALEX.NUV",
    # WISE
    "wise_w1": "WISE/WISE.W1",
    "wise_w2": "WISE/WISE.W2",
    "wise_w3": "WISE/WISE.W3",
    "wise_w4": "WISE/WISE.W4",
    # 2MASS
    "2mass_j": "2MASS/2MASS.J",
    "2mass_h": "2MASS/2MASS.H",
    "2mass_ks": "2MASS/2MASS.Ks",
    # Herschel PACS
    "herschel_70": "Herschel/Pacs.blue",
    "herschel_100": "Herschel/Pacs.green",
    "herschel_160": "Herschel/Pacs.red",
    # UVJ rest-frame (Johnson)
    "johnson_u": "Generic/Johnson.U",
    "johnson_v": "Generic/Johnson.V",
    "johnson_j": "2MASS/2MASS.J",  # Use 2MASS J as proxy
}

_SVO_BASE_URL = "https://svo2.cab.inta-csic.es/theory/fps/fps.php"
_DEFAULT_CACHE_DIR = "data/filters"
_VOT_NS = "{http://www.ivoa.net/xml/VOTable/v1.2}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _svo_id_to_filename(svo_id: str) -> str:
    """Convert SVO filter ID to a safe filename."""
    return svo_id.replace("/", "_").replace(".", "_") + ".dat"


def _parse_votable(xml_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse SVO VOTable XML and extract wavelength/transmission arrays.

    Parameters
    ----------
    xml_bytes : bytes
        Raw XML response from the SVO FPS.

    Returns
    -------
    wave : ndarray
        Wavelength in Angstrom.
    trans : ndarray
        Transmission (dimensionless).

    Raises
    ------
    ValueError
        If the XML cannot be parsed or contains no data rows.
    """
    root = ET.fromstring(xml_bytes)

    # Try multiple namespace variants (SVO sometimes uses v1.2, sometimes v1.1)
    rows = None
    for ns in [_VOT_NS, "{http://www.ivoa.net/xml/VOTable/v1.1}", ""]:
        tabledata = root.find(f".//{ns}TABLEDATA")
        if tabledata is not None:
            rows = tabledata.findall(f"{ns}TR")
            if rows:
                break

    if not rows:
        raise ValueError(
            "No TABLEDATA rows found in SVO VOTable response. "
            "The filter ID may be invalid or the service may be unavailable."
        )

    wave_list: list[float] = []
    trans_list: list[float] = []
    for row in rows:
        # Try with namespace, then without
        cells = row.findall(f"{_VOT_NS}TD")
        if not cells:
            cells = row.findall("{http://www.ivoa.net/xml/VOTable/v1.1}TD")
        if not cells:
            cells = row.findall("TD")
        if len(cells) >= 2:
            wave_list.append(float(cells[0].text))
            trans_list.append(float(cells[1].text))

    if len(wave_list) == 0:
        raise ValueError("Parsed zero data points from SVO VOTable response.")

    return np.array(wave_list), np.array(trans_list)


def _save_filter(filepath: Path, wave: np.ndarray, trans: np.ndarray) -> None:
    """Save filter to a two-column text file."""
    header = "# Wavelength(Angstrom)  Transmission"
    np.savetxt(str(filepath), np.column_stack([wave, trans]), header=header, fmt="%.6e")


def _load_filter_file(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column filter text file."""
    data = np.loadtxt(str(filepath))
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Filter file {filepath} must have at least 2 columns "
            f"(wavelength, transmission). Got shape {data.shape}."
        )
    return data[:, 0], data[:, 1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    RuntimeError
        If the download fails after retries.
    ValueError
        If the VOTable response cannot be parsed.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    filepath = cache_path / _svo_id_to_filename(svo_id)

    # Return cached version if available
    if filepath.exists():
        return _load_filter_file(filepath)

    url = f"{_SVO_BASE_URL}?ID={svo_id}"
    request = Request(url, headers={"User-Agent": "tengri/1.0"})

    try:
        with urlopen(request, timeout=30) as response:
            xml_bytes = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Failed to download filter '{svo_id}' from SVO FPS.\n"
            f"URL: {url}\n"
            f"Error: {exc}\n"
            f"Check your network connection or try again later."
        ) from exc

    wave, trans = _parse_votable(xml_bytes)

    # Sort by wavelength (should already be, but ensure)
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
        Filter with wavelength (Angstrom), transmission normalized to
        max=1, and name.

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

    # Normalize transmission to max = 1
    t_max = np.max(trans)
    if t_max > 0:
        trans = trans / t_max

    return FilterCurve(
        wave=jnp.array(wave),
        trans=jnp.array(trans),
        name=name,
    )


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
        Directory for cached filter files.

    Returns
    -------
    filter_waves : list of jnp.ndarray
        Wavelength arrays per filter.
    filter_trans : list of jnp.ndarray
        Transmission arrays per filter.
    filter_curves : list of FilterCurve
        Full FilterCurve objects.
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
        Filter with normalized transmission (max=1).

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

    # Sort by wavelength
    order = np.argsort(wave)
    wave = wave[order]
    trans = trans[order]

    # Normalize
    t_max = np.max(trans)
    if t_max > 0:
        trans = trans / t_max

    name = path.stem
    return FilterCurve(
        wave=jnp.array(wave),
        trans=jnp.array(trans),
        name=name,
    )


def list_available_filters() -> dict[str, str]:
    """Print and return the filter registry.

    For each filter, prints the short name and SVO identifier.
    If the filter is cached, also shows the approximate effective
    wavelength.

    Returns
    -------
    dict
        Copy of ``FILTER_REGISTRY``.
    """
    print(f"{'Short Name':<20s} {'SVO ID':<30s}")
    print("-" * 52)
    for name, svo_id in sorted(FILTER_REGISTRY.items()):
        print(f"{name:<20s} {svo_id:<30s}")
    print(f"\nTotal: {len(FILTER_REGISTRY)} filters")
    return dict(FILTER_REGISTRY)
