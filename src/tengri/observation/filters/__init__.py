# SPDX-License-Identifier: BSD-3-Clause
"""Photometric filter management via the SVO Filter Profile Service.

Downloads, caches, and loads photometric filter transmission curves from
the Spanish Virtual Observatory (SVO) Filter Profile Service:
https://svo2.cab.inta-csic.es/theory/fps/

Uses astroquery.svo_fps for downloads. Filters are cached as two-column
text files (wavelength in Angstrom, transmission) under a configurable
cache directory.

Alias conventions
-----------------
Aliases are ``<facility>_<band>``, lowercase. One wrinkle is worth stating
because it looks like an inconsistency and is not: HST wide and medium bands
are spelled bare (``hst_f105w``, ``hst_f127m``), while HST *narrow* bands carry
their instrument (``hst_uvis_f656n``, ``hst_wfc3ir_f132n``, ``hst_acs_f502n``).

F502N and F658N exist on both WFC3/UVIS and ACS/WFC with genuinely different
curves, so a bare ``hst_f502n`` would have to pick one instrument silently,
an arbitrary choice that returns a wrong answer without raising. Qualifying the
whole narrow set keeps one rule rather than a per-band exception list. The bare
spelling is kept for wide and medium bands, where no such collision exists, so
the aliases predating this convention keep working.

Synthetic bands versus measured curves
--------------------------------------
Facilities whose "bandpass" is really a receiver spectral window (ALMA, in GHz)
or a detector energy range (Chandra, NuSTAR, in keV) have nothing to fetch from
SVO. Those are served as rectangular top-hats from
:mod:`tengri.observation.filters.synthetic`, and :func:`load_filter` resolves
them by name like any other filter, so they work in
``Photometry.from_names([...])`` and therefore in a fit.

Where a measured curve exists, it wins and no synthetic twin is defined. SPT-SZ
is the worked example: SVO serves measured 150 and 220 GHz bandpasses, so those
are ordinary registry entries and only the 95 GHz band is synthetic. A name must
never resolve to an approximation while real data for it exists.

Real (sub)mm bandpasses already on SVO:
  Planck HFI   : 100-857 GHz        → planck_hfi_100 … planck_hfi_857
  Planck LFI   : 30-70 GHz          → planck_lfi_030 … planck_lfi_070
  SPT-SZ       : 150, 220 GHz       → spt_150ghz, spt_220ghz
  JCMT SCUBA-2 : 450 μm, 850 μm     → scuba2_450, scuba2_850
  APEX LABOCA  : 870 μm             → laboca_870
  APEX SABOCA  : 350 μm             → saboca_350

User-supplied curves
--------------------
Three routes, all resolving through :func:`load_filter` so a custom band is
usable anywhere a built-in one is; see
:mod:`tengri.observation.filters.custom`:

* :func:`register_filter` / :func:`register_filter_from_file`: in-memory,
  by name, for the current process.
* ``$TENGRI_FILTER_DIR``, a ``:``-separated directory list; any curve file
  dropped in loads by its file stem, with no code call, and resolves the same
  way on another machine that has the directory.
* :func:`load_filter_from_dsps_transmission_curve` / DSPS curve files.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# numpy 2.0 removed np.trapz; numpy >= 1.26 provides np.trapezoid.
try:
    from numpy import trapezoid as _np_trapezoid
except ImportError:  # numpy < 1.26
    from numpy import trapz as _np_trapezoid  # type: ignore[no-redef]

from tengri._deprecated import deprecated_alias
from tengri.observation.filters.synthetic import (
    _ALMA_BANDS_GHZ,
    SYNTHETIC_BAND_REGISTRY,
    list_synthetic_bands,
    load_alma_band as _load_alma_band_impl,
    load_synthetic_band,
)
from tengri.observation.photometry import FilterCurve
from tengri.registry import _RegistryTable
from tengri.utils.physics_constants import C_AA

# Speed of light (legacy alias for backward compatibility)
_C_AA_S = C_AA

# ── Registry: short name -> SVO Filter Profile Service ID ─────────

_REGISTRY_PATH = Path(__file__).parent.parent / "data" / "filters_registry.json"
FILTER_REGISTRY: dict[str, str] = json.loads(_REGISTRY_PATH.read_text())


def default_filter_cache_dir() -> Path:
    """Where SVO filter curves are cached, independent of the working directory.

    Returns
    -------
    pathlib.Path
        The first ``<data-dir>/filters`` that already exists, searching
        ``tengri._data_setup.data_dirs``; otherwise
        ``download_dir()/"filters"``, which is always a directory the loaders
        search, so a curve fetched now is found later.

    Notes
    -----
    Resolved per call rather than frozen in a module constant. A constant would
    capture whatever the working directory was when ``tengri`` was first
    imported, and would not see a later ``$TENGRI_DATA_DIR``.

    This replaces the literal ``"data/filters"``, which
    :func:`download_filter` passed straight to ``Path(...).mkdir(parents=True)``.
    Run from anywhere but the repository root, that created a stray
    ``data/filters/`` beside the caller and re-fetched every curve from the SVO
    service, because the populated cache was never consulted. Neither symptom
    raised (#1486).
    """
    from tengri._data_setup import data_dirs, download_dir

    for directory in data_dirs():
        candidate = directory / "filters"
        if candidate.is_dir():
            return candidate
    return download_dir() / "filters"


def find_cached_filter(filename: str) -> Path | None:
    """Locate one already-cached curve across every filter cache, or ``None``.

    Parameters
    ----------
    filename : str
        Cache file name from :func:`_svo_id_to_filename`, e.g.
        ``"GALEX_GALEX_FUV.dat"``.

    Returns
    -------
    pathlib.Path or None
        The first ``<data-dir>/filters/<filename>`` that is a file, searching
        ``tengri._data_setup.data_dirs`` in order; ``None`` when no cache
        holds it and it genuinely has to be fetched.

    Notes
    -----
    This asks *where is this curve*, where :func:`default_filter_cache_dir`
    asks *which directory shall I use*. The difference is not cosmetic. The
    directory-level question commits to the first ancestor that merely owns a
    ``filters/`` folder, before knowing whether that folder holds the curve
    being requested, so a partial cache anywhere below the canonical one makes
    the canonical one unreachable.

    That is not hypothetical. ``examples/advanced/data/filters/`` held ten
    committed curves and ``examples/inference/data/filters/`` five, beside the
    249 in ``data/filters/``. Because the gallery runner ``chdir``s into each
    script's directory, every example in those two directories resolved to the
    partial copy, and any band outside it, GALEX, VISTA, 2MASS, was fetched
    from SVO on every CI run. A miss is indistinguishable from a cold cache, so
    it failed *open*: the network call succeeded and nothing reported that the
    committed curves had been bypassed. It surfaced only when ``astroquery``
    became optional and the silent fetch became an ``ImportError``.

    Searching for the file rather than the directory removes the whole class:
    a partial cache can now only ever add curves, never hide them.
    """
    from tengri._data_setup import data_dirs

    for directory in data_dirs():
        candidate = directory / "filters" / filename
        if candidate.is_file():
            return candidate
    return None


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
    "gaia": "Gaia",
    # Medium-band surveys.
    "alhambra": "CAHA/ALHAMBRA",
    "shards": "GTC/OSIRIS",
    "fourstar": "Magellan/FourStar",
    "newfirm": "NOAO/NEWFIRM",
    "wircam": "CFHT/WIRCam",
    # Narrow-band surveys.
    "jplus": "OAJ/J-PLUS",
    "jpas": "OAJ/J-PAS",
    "hawki": "VLT/HAWK-I",
    # Millimeter.
    "planck": "Planck",
    "spt": "SPT",
}


def _unknown_filter_msg(name: str) -> str:
    """Did-you-mean error message for an unrecognized filter name."""
    import difflib

    from tengri.observation.filters.custom import _USER_FILTER_REGISTRY

    # Pool includes SVO filters, synthetic bands, and user-registered filters
    pool = sorted(
        set(FILTER_REGISTRY)
        | set(_svo_name_to_key())
        | set(SYNTHETIC_BAND_REGISTRY.keys())
        | set(_USER_FILTER_REGISTRY.keys())
    )
    close = difflib.get_close_matches(name, pool, n=3, cutoff=0.6)
    hint = f" Did you mean {close}?" if close else ""
    return (
        f"Unknown filter '{name}'.{hint} tengri.list_filters() lists every "
        "SVO filter: both the SVO-style names (e.g. 'SLOAN_SDSS_g') and "
        "their short aliases (e.g. 'sdss_g'). Synthetic bands (ALMA, X-ray, "
        "submillimeter) are in SYNTHETIC_BAND_REGISTRY. "
        "User-registered filters via register_filter() or TENGRI_FILTER_DIR "
        "also load here. load_custom_filter() reads arbitrary curve files."
    )


def _infer_facility(name: str) -> str:
    """Infer facility from filter short name.

    Consults synthetic band registry first by exact name (so xmm_epic_soft
    maps to XMM-Newton/EPIC, not XMM-Newton/OM), then falls back to
    prefix-based inference from the SVO registry.
    """
    # Check synthetic bands first (exact name match)
    if name in SYNTHETIC_BAND_REGISTRY:
        return SYNTHETIC_BAND_REGISTRY[name].facility
    # Fall back to prefix-based inference for SVO filters
    for prefix, facility in _FACILITY_FROM_PREFIX.items():
        if name.startswith(prefix):
            return facility
    return "Other"


# ── Filter property computation (pure numpy, no JAX) ──────────────


def compute_effective_wavelength(wave: np.ndarray, trans: np.ndarray) -> float:
    """Photon-counting effective wavelength (pivot wavelength): λ_eff = ∫T·λ·dλ / ∫T·dλ.

    Parameters
    ----------
    wave : array, shape (n_wave,)
        Wavelength [Angstrom].
    trans : array, shape (n_wave,)
        Transmission (dimensionless [0, 1]).

    Returns
    -------
    float
        Photon-counting effective wavelength (pivot wavelength) [Angstrom].

    Notes
    -----
    This function computes the **standard astronomical photon-counting**
    effective wavelength, also known as the **pivot wavelength**. It is
    used for filter metadata, observational work, and validation against
    external codes (e.g., FSPS).

    NOT the same as the power-weighted (transmission x flux) effective
    wavelength sometimes used in approximate photometry schemes.

    Not JAX-compatible (uses NumPy). Intended for filter metadata
    computation, not forward model evaluation.

    """
    num = _np_trapezoid(trans * wave, wave)
    den = _np_trapezoid(trans, wave)
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
    SVO if needed) or constructs it for synthetic bands, and computes
    derived properties.

    Parameters
    ----------
    name : str
        Short filter name from ``FILTER_REGISTRY`` or
        ``SYNTHETIC_BAND_REGISTRY``.
    cache_dir : str, optional
        Override cache directory for SVO filters.

    Returns
    -------
    dict
        Keys: ``name``, ``svo_id`` (or ``"synthetic"`` for synthetic bands),
        ``facility``, ``lambda_eff_aa``, ``fwhm_aa``,
        ``lambda_eff_str``, ``fwhm_str``.

    Raises
    ------
    KeyError
        If *name* is not in either registry.

    Notes
    -----
    Not JAX-compatible (uses NumPy and file I/O). Intended for filter
    metadata computation and interactive exploration of the registry,
    not for forward model evaluation.

    """
    if name not in FILTER_REGISTRY and name not in SYNTHETIC_BAND_REGISTRY:
        raise KeyError(_unknown_filter_msg(name))
    kwargs = {"cache_dir": cache_dir} if cache_dir is not None else {}
    fc = load_filter(name, **kwargs)
    wave_np = np.asarray(fc.wave)
    trans_np = np.asarray(fc.trans)
    lam_eff = compute_effective_wavelength(wave_np, trans_np)
    fwhm = compute_fwhm(wave_np, trans_np)
    # Synthetic bands have no SVO ID; mark them as "synthetic"
    svo_id = FILTER_REGISTRY.get(name, "synthetic")
    return {
        "name": name,
        "svo_id": svo_id,
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


_SVO_NAME_TO_KEY_CACHE: dict[str, str] | None = None


def _svo_name_to_key() -> dict[str, str]:
    """Map SVO-style display names to their canonical short registry key.

    ``tengri.list_filters()`` shows filters by their curve-file stem (the
    SVO convention ``Telescope_Instrument_Band``, e.g. ``2MASS_2MASS_H``),
    but :data:`FILTER_REGISTRY`, and therefore :func:`load_filter` and
    :meth:`Photometry.from_names`, is keyed by short aliases (``2mass_h``).
    This reverse map lets the loader accept *either* form, so every name the
    discovery menu advertises round-trips.

    Returns
    -------
    dict[str, str]
        ``{svo_stem: short_key}``. On the rare stem collision (two aliases
        resolve to the same curve file), the first alias in registry order
        wins, both load the identical curve, so the choice is cosmetic.
    """
    global _SVO_NAME_TO_KEY_CACHE
    if _SVO_NAME_TO_KEY_CACHE is None:
        mapping: dict[str, str] = {}
        for key, svo_id in FILTER_REGISTRY.items():
            stem = _svo_id_to_filename(svo_id)[:-4]  # drop the ".dat" suffix
            mapping.setdefault(stem, key)
        _SVO_NAME_TO_KEY_CACHE = mapping
    return _SVO_NAME_TO_KEY_CACHE


def _save_filter(filepath: Path, wave: np.ndarray, trans: np.ndarray) -> None:
    """Write wavelength and transmission columns to a two-column text file."""
    header = "# Wavelength(Angstrom)  Transmission"
    np.savetxt(str(filepath), np.column_stack([wave, trans]), header=header, fmt="%.6e")


def _sanitize_filter_curve(wave: np.ndarray, trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort, merge duplicate wavelengths, and clip negative transmission.

    Published curves are imperfect: of the shipped SVO set,
    HST_ACS_WFC_F814W carries 177 duplicated wavelength rows and the
    Herschel SPIRE / Spitzer MIPS curves dip slightly negative
    (instrumental ringing in the tabulation). Duplicates break the
    ascending-grid assumption of every ``np.interp``/``searchsorted``
    consumer downstream; negative transmission is unphysical in a
    throughput integral. Duplicates are merged by averaging their
    transmission values.
    """
    wave = np.asarray(wave, dtype=np.float64)
    trans = np.asarray(trans, dtype=np.float64)
    order = np.argsort(wave, kind="stable")
    wave, trans = wave[order], trans[order]
    if wave.size and (np.diff(wave) <= 0).any():
        unique_wave, inverse = np.unique(wave, return_inverse=True)
        summed = np.zeros_like(unique_wave)
        counts = np.zeros_like(unique_wave)
        np.add.at(summed, inverse, trans)
        np.add.at(counts, inverse, 1.0)
        wave, trans = unique_wave, summed / counts
    return wave, np.maximum(trans, 0.0)


def _load_filter_file(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read and return wavelength and transmission columns from a text file."""
    data = np.loadtxt(str(filepath))
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Filter file {filepath} must have at least 2 columns "
            f"(wavelength, transmission). Got shape {data.shape}."
        )
    return _sanitize_filter_curve(data[:, 0], data[:, 1])


def _fetch_from_svo(svo_id: str, short_name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fetch filter curve from SVO using astroquery.svo_fps.

    Parameters
    ----------
    svo_id : str
        SVO filter identifier (e.g. ``"JWST/NIRCam.F200W"``).
    short_name : str, optional
        User-facing short name (e.g. ``"jwst_f200w"``) for error messages.

    Raises
    ------
    ImportError
        If ``astroquery`` is not installed or network is unavailable.
    ValueError
        If SVO returns no data for the given filter ID.
    """
    try:
        from astroquery.svo_fps import SvoFps
    except ImportError as exc:
        raise ImportError(
            "astroquery is required to download filters from SVO, and is an "
            "optional dependency.\n"
            "    pip install 'astro-tengri[filters]'\n"
            "Every filter tengri.list_filters() names ships as a cached curve "
            "under data/filters/ and loads without it, this is only needed "
            "to fetch a curve SVO has and tengri does not."
        ) from exc

    try:
        table = SvoFps.get_transmission_data(svo_id)
    except (ConnectionError, TimeoutError, OSError) as exc:
        # Network is unavailable or SVO is unreachable
        name_str = f"'{short_name}' " if short_name else f"'{svo_id}' "
        filename = Path(svo_id).name.replace("/", "_") + ".dat"
        raise ImportError(
            f"Could not fetch filter curve {name_str}from the SVO Filter Profile "
            f"Service. The network may be unavailable or SVO may be unreachable.\n\n"
            f"To use this filter offline:\n"
            f"    python tools/download_filters.py {short_name or svo_id}\n"
            f"to download it in advance, or set $TENGRI_DATA_DIR to a directory "
            f"containing the curve in data/filters/.\n\n"
            f"If you are in a CI/build environment, check that the filter is "
            f"tracked in data/filters/ (e.g. data/filters/{filename})."
        ) from exc

    wave = np.asarray(table["Wavelength"], dtype=float)
    trans = np.asarray(table["Transmission"], dtype=float)
    if len(wave) == 0:
        raise ValueError(
            f"SVO returned zero rows for filter '{svo_id}'. Check that the filter ID is correct."
        )
    _warn_if_energy_detector(svo_id)
    return wave, trans


def _warn_if_energy_detector(svo_id: str) -> None:
    """Warn when SVO marks the filter as an energy-counting detector.

    SVO records ``DetectorType`` per filter: ``"0"`` = energy counter,
    ``"1"`` = photon counter. tengri's default photometric convention is
    photon-counting Bessell (``w = 1/λ``; ADR-0017), which assumes a
    photon-counting transmission curve. For an energy-type curve the user
    likely wants the energy convention
    (``Photometry.from_names(..., convention="energy")``); emit a heads-up.
    Best-effort: silently skips if the metadata is unavailable.
    """
    import warnings

    try:
        from astroquery.svo_fps import SvoFps

        meta = SvoFps.get_filter_list(
            facility=svo_id.split("/")[0], filter_name=svo_id.split("/")[-1].split(".")[-1]
        )
        # Match the exact filterID row and read DetectorType.
        for row in meta:
            if str(row.get("filterID", "")) == svo_id:
                det = str(row.get("DetectorType", "")).strip()
                if det == "0":
                    warnings.warn(
                        f"SVO filter '{svo_id}' is an ENERGY-counting detector "
                        f"(DetectorType=0). tengri defaults to the photon-counting "
                        f"'bessell' convention (w=1/λ); for this curve you likely want "
                        f"Photometry.from_names(..., convention='energy'). "
                        f"See docs/units.md (Photometric filter-convolution convention).",
                        stacklevel=3,
                    )
                break
    except Exception:
        # Metadata lookup is best-effort; never block a filter download on it.
        pass


# ── Public API ────────────────────────────────────────────────────


def download_filter(
    svo_id: str,
    cache_dir: str | Path | None = None,
    short_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Download a single filter from the SVO Filter Profile Service.

    If the filter is already cached on disk, it is loaded from the cache
    instead of re-downloading.

    Parameters
    ----------
    svo_id : str
        SVO filter identifier (e.g. ``"JWST/NIRCam.F200W"``).
    cache_dir : str or pathlib.Path or None, optional
        Directory for cached filter files. ``None`` (default) resolves through
        :func:`default_filter_cache_dir`, which is independent of the working
        directory.
    short_name : str, optional
        Short registry name (e.g. ``"jwst_f200w"``) for error messages.

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

    Notes
    -----
    Not JAX-compatible (uses file I/O and astroquery). Caching avoids
    redundant SVO downloads. The returned transmission values are not
    normalized, the absolute scale cancels in the photometry integral
    ``∫fλTλdλ / ∫Tλdλ``.

    This is the only entry point that touches the filesystem; ``load_filter``
    and ``load_filter_set`` pass ``cache_dir`` down unchanged, so ``None``
    is resolved here once rather than in each of them.

    An explicit *cache_dir* is honored exactly as given, it is a caller
    saying "use this one". Only the ``None`` default searches every cache via
    :func:`find_cached_filter`, so a partial cache nearer the working directory
    can no longer hide a complete one further up (see that function's Notes).

    The cache directory is created on the download path only. Loading a curve
    is a read, and a read that mkdirs leaves a stray ``data/filters/`` beside
    whatever directory the caller happened to start in.

    """
    filename = _svo_id_to_filename(svo_id)

    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cached = cache_path / filename
        cached = cached if cached.is_file() else None
    else:
        cache_path = default_filter_cache_dir()
        cached = find_cached_filter(filename)

    if cached is not None:
        return _load_filter_file(cached)

    wave, trans = _sanitize_filter_curve(*_fetch_from_svo(svo_id, short_name=short_name))

    order = np.argsort(wave)
    wave = wave[order]
    trans = trans[order]

    cache_path.mkdir(parents=True, exist_ok=True)
    _save_filter(cache_path / filename, wave, trans)
    return wave, trans


def load_filter(
    name: str,
    cache_dir: str | Path | None = None,
) -> FilterCurve:
    """Load a filter by its short registry name.

    Downloads from SVO if not already cached. Also supports synthetic
    top-hat bandpasses for ALMA, X-ray, and (sub)millimeter facilities,
    user-registered filters, and curves from the user filter directory.

    Parameters
    ----------
    name : str
        Short alias (e.g. ``"jwst_f200w"``, ``"alma_band6"``, ``"chandra_soft"``),
        SVO-style display name (e.g. ``"JWST_NIRCam_F200W"``), user-registered
        name, or synthetic band name. Both SVO spellings round-trip; synthetic
        bands and user filters are accessed by their exact alias.
    cache_dir : str or pathlib.Path or None, optional
        Directory for cached filter files. ``None`` (default) resolves through
        :func:`default_filter_cache_dir`. Ignored for synthetic bands and
        user-registered filters.

    Returns
    -------
    FilterCurve
        Filter with wavelength (Angstrom), transmission, and name.
        For SVO filters, transmission values are as returned by SVO
        (not normalized). For synthetic bands and user filters, transmission
        values are as provided.

    Raises
    ------
    KeyError
        If *name* matches none of: user-registered, user filter directory,
        ``FILTER_REGISTRY`` alias, SVO display name, or
        :data:`SYNTHETIC_BAND_REGISTRY` entry.

    Notes
    -----
    Not JAX-compatible (uses file I/O and astroquery for SVO downloads).
    Preferred interface over :func:`download_filter` for user code;
    provides a JAX array return type (FilterCurve) rather than raw NumPy.

    **Resolution order** (first match wins):

    1. User-registered filters via :func:`register_filter`
    2. User filter directory (``TENGRI_FILTER_DIR`` environment variable)
    3. Built-in ``FILTER_REGISTRY`` alias (exact name match)
    4. SVO display stem (e.g., ``"SLOAN_SDSS_g"`` → ``"sdss_g"``)
    5. Synthetic band registry (ALMA, X-ray, etc.)
    6. Raise :exc:`KeyError`

    This precedence ensures user customization always wins over built-ins,
    and that a typo in a built-in name still gets found by its display stem.

    Examples
    --------
    Load a built-in filter::

        >>> from tengri.observation.filters import load_filter
        >>> fc = load_filter("sdss_r")

    Register a custom filter and load it::

        >>> import numpy as np
        >>> from tengri.observation.filters import register_filter, load_filter
        >>> wave = np.linspace(5000, 7000, 100)
        >>> trans = np.exp(-0.5 * ((wave - 6000) / 300) ** 2)
        >>> register_filter("my_band", wave, trans)
        >>> fc = load_filter("my_band")

    Load a filter from a file::

        >>> fc = load_filter("band_from_file")  # if in TENGRI_FILTER_DIR

    """
    from tengri.observation.filters.custom import (
        _USER_FILTER_REGISTRY,
        _load_filter_from_directory,
    )

    # 1. Check user-registered filters
    if name in _USER_FILTER_REGISTRY:
        return _USER_FILTER_REGISTRY[name]

    # 2. Check user filter directory (TENGRI_FILTER_DIR)
    fc_from_dir = _load_filter_from_directory(name)
    if fc_from_dir is not None:
        return fc_from_dir

    # 3. Try exact FILTER_REGISTRY first
    if name in FILTER_REGISTRY:
        svo_id = FILTER_REGISTRY[name]
        wave, trans = download_filter(svo_id, cache_dir=cache_dir, short_name=name)
        return FilterCurve(wave=jnp.array(wave), trans=jnp.array(trans), name=name)

    # 4. Try SVO display-stem resolution (e.g., "SLOAN_SDSS_g" → "sdss_g")
    alias = _svo_name_to_key().get(name)
    if alias is not None:
        svo_id = FILTER_REGISTRY[alias]
        wave, trans = download_filter(svo_id, cache_dir=cache_dir, short_name=alias)
        return FilterCurve(wave=jnp.array(wave), trans=jnp.array(trans), name=alias)

    # 5. Try synthetic band registry
    if name in SYNTHETIC_BAND_REGISTRY:
        return load_synthetic_band(name)

    # 6. Not found anywhere
    raise KeyError(_unknown_filter_msg(name))


def load_filter_set(
    names: list[str],
    cache_dir: str | Path | None = None,
) -> tuple[list[jnp.ndarray], list[jnp.ndarray], list[FilterCurve]]:
    """Load multiple filters by short name.

    Parameters
    ----------
    names : list of str
        Short names from ``FILTER_REGISTRY``.
    cache_dir : str or pathlib.Path or None, optional
        Directory for cached filter files. ``None`` (default) resolves through
        :func:`default_filter_cache_dir`.

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

    Notes
    -----
    Not JAX-compatible (uses file I/O). Useful when filter profiles
    are not available in SVO or when using synthetic, custom-defined
    bandpasses. File format is flexible: trailing whitespace and
    comment lines (starting with #) are ignored.

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
    standard bandpass on SVO, for example, an ALMA continuum flux at a
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
    Profile Service. This function constructs a rectangular bandpass
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

    Raises
    ------
    ValueError
        If band is not in 1–10.

    Examples
    --------
    >>> fc = load_alma_band(6)  # 1.23 mm continuum (211–275 GHz)
    >>> fc = load_alma_band(7)  # 870 μm continuum (275–373 GHz)

    Notes
    -----
    Band definitions follow the ALMA Cycle 11 receiver specifications.
    Wavelengths are in the *observed* frame; the filter should be applied
    at the observed frequency. For a source at redshift *z*, Band N probes
    rest-frame wavelength λ_rest = λ_obs / (1 + z).

    **Backward compatibility**: This function delegates to the synthetic
    band registry and is kept here for API stability. New code can use
    :func:`load_filter` with ``"alma_band{N}"`` directly.

    """
    # Delegate to the implementation in synthetic.py
    return _load_alma_band_impl(band, name=name)


def list_available_filters(
    *,
    group_by: str = "facility",
    compute_properties: bool = False,
    cache_dir: str | None = None,
) -> _RegistryTable:
    """List every filter and synthetic band, as a table.

    Parameters
    ----------
    group_by : str
        Row ordering. ``"facility"`` (default) sorts by telescope /
        instrument and then by name; ``"none"`` sorts alphabetically by
        name alone. Default: ``"facility"``.
    compute_properties : bool
        If ``True``, add ``lambda_eff`` and ``fwhm`` columns. This loads
        every transmission curve and triggers SVO downloads for any
        filter not yet cached. Default: ``False``.
    cache_dir : str, optional
        Override cache directory for filter downloads. Default: ``None``.

    Returns
    -------
    _RegistryTable
        One row per alias, with columns ``name`` (the short alias
        :func:`load_filter` accepts), ``kind`` (``"filter_alias"`` for SVO,
        ``"synthetic_band"`` for ALMA/X-ray/submillimeter), ``facility``
        and ``svo_id`` (``"synthetic"`` for synthetic bands).

    Notes
    -----
    Includes both SVO filters and synthetic bandpasses. Returned a plain
    ``dict`` before #1574; now every discovery verb returns a table (#1285).
    ``.to_dict("svo_id")`` returns ``{alias: svo_id}`` for SVO filters only
    (synthetic bands have ``svo_id="synthetic"``).

    Examples
    --------
    >>> list_available_filters()
    >>> list_available_filters().filter(facility__contains="ALMA")
    >>> list_available_filters(compute_properties=True)
    """
    # Collect both SVO and synthetic names
    svo_names = sorted(FILTER_REGISTRY)
    synthetic_names = sorted(SYNTHETIC_BAND_REGISTRY)
    all_names = svo_names + synthetic_names

    if group_by != "none":
        all_names.sort(key=lambda n: (_infer_facility(n), n))

    kwargs = {"cache_dir": cache_dir} if cache_dir is not None else {}
    rows: list[dict] = []

    for name in all_names:
        is_synthetic = name in SYNTHETIC_BAND_REGISTRY
        row = {
            "name": name,
            "kind": "synthetic_band" if is_synthetic else "filter_alias",
            "facility": _infer_facility(name),
            "svo_id": "synthetic" if is_synthetic else FILTER_REGISTRY[name],
            "use": f'tengri.load_filter("{name}")',
        }
        if compute_properties:
            info = filter_info(name, **kwargs)
            row["lambda_eff"] = info["lambda_eff_str"]
            row["fwhm"] = info["fwhm_str"]
        rows.append(row)

    return _RegistryTable(rows)


# ── User-facing filter discovery helpers ──────────────────────────
# Thin convenience functions for discoverability on top of load_filter_set


def list_filter_aliases(instrument: str | None = None) -> _RegistryTable:
    """List the short filter aliases that :func:`load_filter` accepts.

    Parameters
    ----------
    instrument : str, optional
        Keep only aliases containing this substring (case-insensitive).
        E.g. ``"sdss"``, ``"jwst"``, ``"hst"``. Default: ``None`` (all).

    Returns
    -------
    _RegistryTable
        One row per alias, with columns ``name`` (the short alias, e.g.
        ``"sdss_r"``) and ``svo_id``. ``.names()`` returns the plain
        sorted list of strings this gave before #1574.

    Notes
    -----
    This answers a *different question* from :func:`tengri.list_filters`,
    which lists the SVO curve-file stems shipped in ``data/filters/``
    (e.g. ``"SLOAN_SDSS_r"``). This lists the short aliases the loaders
    accept (e.g. ``"sdss_r"``). Both spellings load the same curve.

    Both functions were once named ``list_filters``, one name, two
    parameters (``survey`` vs ``instrument``), two return types and two
    value spaces, so a reader could not tell which one they held
    (#1574). The old name survives here as a deprecated alias.

    Matching is permissive: the substring may appear anywhere in the
    alias, not only at the start.

    Examples
    --------
    >>> list_filter_aliases(instrument="sdss").names()
    >>> list_filter_aliases().to_dict("svo_id")  # {alias: SVO ID}
    """
    names = sorted(FILTER_REGISTRY)
    if instrument is not None:
        needle = instrument.lower()
        names = [name for name in names if needle in name.lower()]
    return _RegistryTable(
        [
            {
                "name": name,
                "kind": "filter_alias",
                "svo_id": FILTER_REGISTRY[name],
                "use": f'tengri.load_filter("{name}")',
            }
            for name in names
        ]
    )


#: Deprecated since #1574: this and the top-level ``tengri.list_filters``
#: answered different questions under one name. Use
#: :func:`list_filter_aliases`.
list_filters = deprecated_alias(
    list_filter_aliases,
    old_name="tengri.observation.filters.list_filters",
    new_name="tengri.observation.filters.list_filter_aliases",
)


def load(names: list[str]):
    """Load multiple filters by short name.

    Thin alias for ``load_filter_set`` for discoverability and
    consistency with the filters namespace.

    Parameters
    ----------
    names : list of str
        Short filter names from the registry (e.g., ["sdss_r", "jwst_f200w"]).

    Returns
    -------
    filter_waves : list of jnp.ndarray
        Wavelength arrays per filter, shape (n_wave,) [Angstrom].
    filter_trans : list of jnp.ndarray
        Transmission arrays per filter, shape (n_wave,) (dimensionless [0, 1]).
    filter_curves : list of FilterCurve
        Full FilterCurve objects with wavelength, transmission, and name.

    Raises
    ------
    KeyError
        If any name is not in the filter registry.

    Examples
    --------
    >>> waves, trans, curves = load(["sdss_r", "sdss_i"])
    >>> len(curves)
    2
    >>> curves[0].name
    'sdss_r'

    Notes
    -----
    Filters are downloaded from the SVO Filter Profile Service on first use
    and cached locally under data/filters/.
    """
    return load_filter_set(names)


def describe(name: str) -> str:
    """Return a one-line description of a filter.

    Computes the transmission-weighted effective wavelength and wavelength
    range of the transmission curve.

    Parameters
    ----------
    name : str
        Filter short name from the registry (``"sdss_r"``) or the SVO-style
        curve-file stem (``"SLOAN_SDSS_r"``). Both resolve to the same curve.

    Returns
    -------
    str
        Human-readable description. Format:
        "<name>: lambda_eff ~ X.XXX μm (range A–B μm)".

    Raises
    ------
    KeyError
        If no filter by that name exists.

    Notes
    -----
    Effective wavelength is computed as the transmission-weighted mean.

    Until #1611 the whole body sat under a bare ``except Exception`` that
    returned ``"<name>: (filter found; no summary available)"``, for an
    unknown name too, so the message asserted the opposite of what had
    happened and an unknown filter was indistinguishable from a curve that
    failed to load. The lookup is now outside the ``try``, so an unknown name
    raises the loader's own ``KeyError``, and only the numeric summary is
    guarded. The lookup resolution for this function delegates to the same
    path as :func:`tengri.describe` (issue #1611) through registry fallback
    so that a name cannot resolve in one surface and not the other.
    """
    # Outside the try on purpose: an unknown name must raise, and
    # load_filter_set already says so with a message that lists the menus.
    fc = load_filter_set([name])[2][0]
    try:
        wave_np = np.asarray(fc.wave)
        trans_np = np.asarray(fc.trans)

        lam_eff = compute_effective_wavelength(wave_np, trans_np)

        nonzero = trans_np > 0
        if np.any(nonzero):
            wave_min = wave_np[nonzero].min()
            wave_max = wave_np[nonzero].max()
        else:
            wave_min, wave_max = wave_np.min(), wave_np.max()

        if lam_eff >= 1e4:
            unit = "μm"
            lam_eff_fmt = f"{lam_eff / 1e4:.3f}"
        else:
            unit = "Å"
            lam_eff_fmt = f"{lam_eff:.0f}"

        if wave_min >= 1e4:
            min_fmt = f"{wave_min / 1e4:.2f}"
            max_fmt = f"{wave_max / 1e4:.2f}"
            range_unit = "μm"
        else:
            min_fmt = f"{wave_min:.0f}"
            max_fmt = f"{wave_max:.0f}"
            range_unit = "Å"

        return f"{name}: λ_eff ~ {lam_eff_fmt} {unit} (range {min_fmt}–{max_fmt} {range_unit})"

    except Exception as exc:  # curve loaded, but its numbers are unusable
        return f"{name}: (curve loaded; summary unavailable, {type(exc).__name__})"


def suggest(
    redshift: float,
    coverage: str = "visible_to_nir",
) -> list[str]:
    """Suggest filters covering a rest-frame wavelength range at a redshift.

    Parameters
    ----------
    redshift : float
        Redshift of the source (z >= 0).
    coverage : str
        Rest-frame wavelength coverage preset. Options:

        - "visible": 3500–9000 Å (optical)
        - "visible_to_nir": 3500–25000 Å (optical + near-IR) [default]
        - "uv_to_ir": 1200–50000 Å (UV + optical + IR)
        - "jwst_cover": 6000–50000 Å (rest-frame for JWST epochs)

    Returns
    -------
    list of str
        Filter names with effective wavelength falling within the
        observed-frame span corresponding to the rest-frame coverage.
        Sorted by effective wavelength.

    Raises
    ------
    ValueError
        If coverage is not recognized.

    Notes
    -----
    Observed-frame wavelength is computed as:
    λ_obs = λ_rest * (1 + z).

    Examples
    --------
    >>> suggest(redshift=3.0, coverage="visible_to_nir")  # z=3 galaxies, optical→NIR
    ['jwst_f115w', 'jwst_f150w', ...]
    """
    # Coverage presets (rest-frame, Angstrom)
    coverage_map = {
        "visible": (3500, 9000),
        "visible_to_nir": (3500, 25000),
        "uv_to_ir": (1200, 50000),
        "jwst_cover": (6000, 50000),
    }

    if coverage not in coverage_map:
        raise ValueError(
            f"Unknown coverage '{coverage}'. Must be one of {list(coverage_map.keys())}."
        )

    lam_rest_min, lam_rest_max = coverage_map[coverage]

    # Convert to observed frame
    lam_obs_min = lam_rest_min * (1 + redshift)
    lam_obs_max = lam_rest_max * (1 + redshift)

    # Load all filters and compute effective wavelengths
    all_names = list_filter_aliases().names()
    if not all_names:
        return []

    # Load all filters; skip any that fail
    wavelengths_by_name = {}
    try:
        for name in all_names:
            try:
                fc = load_filter_set([name])[2][0]
                wave_np = np.asarray(fc.wave)
                trans_np = np.asarray(fc.trans)
                lam_eff = compute_effective_wavelength(wave_np, trans_np)
                wavelengths_by_name[name] = lam_eff
            except Exception:
                # Skip filters that fail to load
                continue
    except Exception:
        return []

    # Find filters within observed-frame span
    matches = [
        name
        for name, lam_eff in wavelengths_by_name.items()
        if lam_obs_min <= lam_eff <= lam_obs_max
    ]

    # Sort by effective wavelength
    matches.sort(key=lambda name: wavelengths_by_name[name])

    return matches


# ── Re-export custom filter functions ──────────────────────────────

from tengri.observation.filters.custom import (
    list_registered_filters,
    load_filter_from_dsps_file,
    load_filter_from_dsps_transmission_curve,
    register_filter,
    register_filter_from_file,
    unregister_filter,
)
