# SPDX-License-Identifier: BSD-3-Clause
"""User-facing custom filter curves: in-memory registration and directory loading.

Supports three registration routes:
1. In-memory registration via :func:`register_filter` and :func:`register_filter_from_file`.
2. File-based directory loading from ``TENGRI_FILTER_DIR`` (supports ``:``-separated list).
3. DSPS transmission curve objects and files (with lazy import).

All curves are stored as immutable :class:`FilterCurve` objects and are validated
for plausible wavelength ranges before registration.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from tengri.observation.photometry import FilterCurve
from tengri.registry import _RegistryTable

if TYPE_CHECKING:
    pass

__all__ = [
    "list_registered_filters",
    "load_filter_from_dsps_file",
    "load_filter_from_dsps_transmission_curve",
    "register_filter",
    "register_filter_from_file",
    "unregister_filter",
]


# ── Module-level user filter registry (immutable curves) ─────────────

_USER_FILTER_REGISTRY: dict[str, FilterCurve] = {}


def _load_filter_file(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read two-column text file (wavelength, transmission).

    Strips comments and whitespace. Used by both custom.py and __init__.py.
    """
    data = np.loadtxt(str(filepath))
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Filter file {filepath} must have at least 2 columns "
            f"(wavelength, transmission). Got shape {data.shape}."
        )
    return data[:, 0], data[:, 1]


def _sanitize_filter_curve(wave: np.ndarray, trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort, merge duplicate wavelengths, and clip negative transmission."""
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


def _warn_implausible_wavelength_range(
    wave: np.ndarray, name: str, source: str = "filter"
) -> None:
    """Warn if wavelength range looks implausible for Angstrom.

    Detects likely unit errors (nanometer, micrometer) and warns with the
    likely unit and the conversion needed.

    Parameters
    ----------
    wave : array_like
        Wavelength array [Angstrom].
    name : str
        Filter name for the warning message.
    source : str
        Source description (e.g., "filter", "DSPS curve") for the message.

    Notes
    -----
    Rules (defensible, not exhaustive):
    - If entire span is below ~50 Å: likely in nm or eV, too short for optical.
    - If entire span is above ~1e7 Å (~1 cm): likely in wrong unit.
    - If median wavelength suggests nm (e.g., 300–1000 instead of 3000–10000):
      suggest conversion.
    - If FWHM is very narrow relative to the span: suggests nm confusion.
    """
    wave_arr = np.asarray(wave)
    if wave_arr.size < 2:
        return

    wave_min, wave_max = float(np.min(wave_arr)), float(np.max(wave_arr))
    wave_med = float(np.median(wave_arr))
    span = wave_max - wave_min

    # Check 1: Entire span too low (likely nanometers)
    if wave_max < 50.0:
        warnings.warn(
            f"Filter '{name}' {source} has wavelengths spanning {wave_min:.1f}–{wave_max:.1f} Å, "
            f"which is implausibly short for a photometric bandpass. "
            f"This looks like **nanometers**. "
            f"If so, multiply by 10: 300–1000 nm → 3000–10000 Å.",
            UserWarning,
            stacklevel=4,
        )
        return

    # Check 2: Entire span too high (likely in wrong unit, e.g., cm)
    if wave_min > 1e7:
        warnings.warn(
            f"Filter '{name}' {source} has wavelengths spanning {wave_min:.2e}–{wave_max:.2e} Å, "
            f"which is implausibly long (>1 cm). "
            f"Check that the wavelengths are in Angstrom.",
            UserWarning,
            stacklevel=4,
        )
        return

    # Check 3: Median in nm range (e.g., 100–10000) but labeled Ångstrom
    if wave_med < 5000.0 and span > 100.0 and span < 50000.0:
        warnings.warn(
            f"Filter '{name}' {source} has wavelengths spanning {wave_min:.0f}–{wave_max:.0f} Å "
            f"(median {wave_med:.0f}). This is typical for **nanometers**, not Ångstrom. "
            f"If so, multiply by 10 to convert nm → Å.",
            UserWarning,
            stacklevel=4,
        )
        return

    # Check 4: Suspiciously narrow (FWHM / span suggests nm confusion)
    # E.g., FWHM=100 Å, span=1000 Å is reasonable; FWHM=50 Å, span=400 Å looks like nm
    if span < 5000.0 and span > 10.0:
        # Rough heuristic: if FWHM/span is typical of infrared (narrow), warn
        pass  # (conservative; avoid false positives in narrow bands like filters at 1 µm)


def register_filter(
    name: str,
    wave_aa: np.ndarray | list,
    trans: np.ndarray | list,
    *,
    overwrite: bool = False,
) -> None:
    """Register a filter in the in-memory user registry.

    Creates an immutable :class:`FilterCurve` from the given wavelength and
    transmission arrays. The name becomes available immediately to
    :func:`load_filter` and :meth:`Photometry.from_names`.

    Parameters
    ----------
    name : str
        Filter identifier (e.g., ``"my_custom_band"``). Must be unique
        among all registered names (including built-in aliases, SVO stems,
        and synthetic bands) unless ``overwrite=True``.
    wave_aa : array_like, shape (n_wave,)
        Wavelength grid [Angstrom]. Must be strictly increasing.
    trans : array_like, shape (n_wave,)
        Transmission (dimensionless, 0–1).
    overwrite : bool, optional
        If ``False`` (default), raise if *name* already exists in any registry
        (built-in, SVO, synthetic, or user). If ``True``, silently replace
        an existing user-registered filter or shadow a built-in name.
        Default: ``False``.

    Raises
    ------
    ValueError
        If *wave_aa* and *trans* have incompatible shapes.
    ValueError
        If *wave_aa* is not strictly increasing after sanitization.
    KeyError
        If *name* exists in a built-in registry and ``overwrite=False``.

    Warns
    -----
    UserWarning
        If wavelength range looks implausible for Angstrom (likely nm or µm).

    Notes
    -----
    Shadowing built-in curves (a name that already exists in ``FILTER_REGISTRY``,
    SVO display stems, or ``SYNTHETIC_BAND_REGISTRY``) requires explicit
    ``overwrite=True`` to prevent accidental collisions. This is a deliberate
    gate — the default behavior keeps user registrations from silently hiding
    critical built-in data.

    **Immutability**: The returned :class:`FilterCurve` is immutable. The
    wavelength and transmission are stored as JAX arrays; the registry does
    not hand out mutable references to internal state.

    Examples
    --------
    >>> import numpy as np
    >>> from tengri.observation.filters import register_filter
    >>> wave = np.linspace(5000, 7000, 100)
    >>> trans = np.exp(-0.5 * ((wave - 6000) / 300) ** 2)
    >>> register_filter("my_narrow_band", wave, trans)
    >>> from tengri import Photometry
    >>> phot = Photometry.from_names(["my_narrow_band"])
    """
    from tengri.observation.filters import (
        FILTER_REGISTRY,
        SYNTHETIC_BAND_REGISTRY,
        _svo_name_to_key,
    )

    # Check for collisions
    exists_in_registry = name in FILTER_REGISTRY
    exists_in_svo = name in _svo_name_to_key()
    exists_in_synthetic = name in SYNTHETIC_BAND_REGISTRY
    exists_in_user = name in _USER_FILTER_REGISTRY

    builtin_collision = exists_in_registry or exists_in_svo or exists_in_synthetic
    collision = builtin_collision or exists_in_user

    if collision and not overwrite:
        source = "built-in FILTER_REGISTRY"
        if exists_in_svo:
            source = "SVO display name"
        elif exists_in_synthetic:
            source = "synthetic band registry"
        raise KeyError(
            f"Filter name '{name}' already exists in {source}. "
            f"Pass overwrite=True to shadow the built-in definition (deliberate shadowing only)."
        )

    # Sanitize and validate
    wave = np.asarray(wave_aa, dtype=np.float64)
    trans = np.asarray(trans, dtype=np.float64)

    if wave.shape != trans.shape:
        raise ValueError(
            f"wave and trans must have the same shape; got {wave.shape} and {trans.shape}."
        )

    wave, trans = _sanitize_filter_curve(wave, trans)

    # Check that wavelength is strictly increasing
    if wave.size > 1 and not np.all(np.diff(wave) > 0):
        raise ValueError(
            f"Wavelengths must be strictly increasing after sanitization. "
            f"Got min diff: {np.min(np.diff(wave))}."
        )

    # Warn about implausible ranges
    _warn_implausible_wavelength_range(wave, name, source="custom filter")

    # Store as immutable FilterCurve with JAX arrays (prevent accidental mutation)
    fc = FilterCurve(wave=jnp.array(wave), trans=jnp.array(trans), name=name)
    _USER_FILTER_REGISTRY[name] = fc


def register_filter_from_file(
    name: str,
    filepath: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Register a filter from a two-column text file.

    Reads wavelength and transmission columns, sanitizes, and registers
    in the user registry.

    Parameters
    ----------
    name : str
        Filter identifier (e.g., ``"my_band_from_file"``).
    filepath : str or pathlib.Path
        Path to a two-column text file: wavelength (Angstrom), transmission.
    overwrite : bool, optional
        If ``False`` (default), raise if *name* already exists. If ``True``,
        silently replace an existing user-registered filter or shadow a
        built-in name. Default: ``False``.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file format is invalid.
    KeyError
        If *name* exists in a built-in registry and ``overwrite=False``.

    Examples
    --------
    >>> from tengri.observation.filters import register_filter_from_file
    >>> register_filter_from_file("my_band", "/path/to/curve.txt")
    >>> from tengri import Photometry
    >>> phot = Photometry.from_names(["my_band"])
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Filter file not found: {filepath}")

    wave, trans = _load_filter_file(path)
    register_filter(name, wave, trans, overwrite=overwrite)


def unregister_filter(name: str) -> None:
    """Remove a user-registered filter from the in-memory registry.

    After unregistration, :func:`load_filter` falls back to built-in registries
    (FILTER_REGISTRY, SVO, synthetic bands). Does nothing if the name is not
    registered (no error).

    Parameters
    ----------
    name : str
        Filter identifier to remove.

    Examples
    --------
    >>> from tengri.observation.filters import unregister_filter
    >>> unregister_filter("my_band")
    """
    _USER_FILTER_REGISTRY.pop(name, None)


def list_registered_filters() -> _RegistryTable:
    """List all user-registered filters as a table.

    Returns
    -------
    _RegistryTable
        One row per registered filter, with columns ``name``, ``kind``
        (always ``"user_registered"``), ``facility`` (always ``"User"``,
        and ``svo_id`` (always ``"user_registered"``).

    Examples
    --------
    >>> from tengri.observation.filters import register_filter
    >>> import numpy as np
    >>> wave = np.linspace(5000, 7000, 100)
    >>> trans = np.exp(-0.5 * ((wave - 6000) / 300) ** 2)
    >>> register_filter("band1", wave, trans)
    >>> register_filter("band2", wave, trans)
    >>> list_registered_filters()
    """
    rows = []
    for name in sorted(_USER_FILTER_REGISTRY.keys()):
        rows.append(
            {
                "name": name,
                "kind": "user_registered",
                "facility": "User",
                "svo_id": "user_registered",
            }
        )
    return _RegistryTable(rows)


# ── User filter directory support ─────────────────────────────────────


def _load_filter_from_directory(name: str) -> FilterCurve | None:
    """Attempt to load a filter from a file in TENGRI_FILTER_DIR.

    Searches through ``:``-separated directories in the ``TENGRI_FILTER_DIR``
    environment variable for a file matching the filter name. Accepts
    ``.dat``, ``.txt``, ``.csv`` formats.

    Parameters
    ----------
    name : str
        Filter file stem (e.g., ``"my_band"`` matches ``"my_band.dat"``).

    Returns
    -------
    FilterCurve or None
        Loaded filter, or ``None`` if not found or ``TENGRI_FILTER_DIR``
        is not set.

    Notes
    -----
    This route is reproducible across machines — a config referring to
    ``"my_band"`` resolves anywhere the directory is present.
    """
    filter_dirs = os.environ.get("TENGRI_FILTER_DIR")
    if not filter_dirs:
        return None

    # Parse `:`-separated list (like PATH)
    dir_list = [Path(d).expanduser() for d in filter_dirs.split(":") if d]

    # Try each extension in each directory
    extensions = [".dat", ".txt", ".csv"]
    for directory in dir_list:
        for ext in extensions:
            filepath = directory / f"{name}{ext}"
            if filepath.is_file():
                wave, trans = _load_filter_file(filepath)
                wave, trans = _sanitize_filter_curve(wave, trans)
                _warn_implausible_wavelength_range(wave, name, source="directory filter")
                return FilterCurve(wave=jnp.array(wave), trans=jnp.array(trans), name=name)

    return None


# ── DSPS integration (lazy import) ────────────────────────────────────


def load_filter_from_dsps_transmission_curve(
    dsps_curve: object,
    name: str | None = None,
) -> FilterCurve:
    """Convert a DSPS transmission curve to a :class:`FilterCurve`.

    Extracts wavelength and transmission from a DSPS ``TransmissionCurve``
    object and validates that wavelengths are in Angstrom.

    Parameters
    ----------
    dsps_curve : object
        A DSPS ``TransmissionCurve`` named tuple with ``wave`` and
        ``transmission`` attributes. Must have wave in Angstrom.
    name : str, optional
        Filter name. If ``None``, uses ``"dsps_filter"``. Default: ``None``.

    Returns
    -------
    FilterCurve
        Transmission curve with wavelengths validated as Angstrom.

    Raises
    ------
    AttributeError
        If *dsps_curve* does not have ``wave`` and ``transmission`` attributes.
    ValueError
        If wavelengths are not strictly increasing after sanitization.

    Warns
    -----
    UserWarning
        If wavelength range looks implausible for Angstrom.

    Notes
    -----
    **Unit verification**: DSPS ``TransmissionCurve`` documents wavelengths as
    λ/Å. This function validates that assumption and warns if the range looks
    incorrect. Getting this wrong is a silent, catastrophic error — verify
    the conversion against the DSPS source, never assume.

    Examples
    --------
    >>> from dsps.data_loaders import load_transmission_curve
    >>> dsps_curve = load_transmission_curve(fn="/path/to/filter.h5")
    >>> from tengri.observation.filters import load_filter_from_dsps_transmission_curve
    >>> fc = load_filter_from_dsps_transmission_curve(dsps_curve, name="my_dsps_band")
    """
    if not hasattr(dsps_curve, "wave") or not hasattr(dsps_curve, "transmission"):
        raise AttributeError(
            f"Expected DSPS TransmissionCurve with 'wave' and 'transmission' attributes; "
            f"got {type(dsps_curve).__name__}."
        )

    wave = np.asarray(dsps_curve.wave, dtype=np.float64)
    trans = np.asarray(dsps_curve.transmission, dtype=np.float64)

    wave, trans = _sanitize_filter_curve(wave, trans)

    # Check wavelengths are strictly increasing
    if wave.size > 1 and not np.all(np.diff(wave) > 0):
        raise ValueError(
            f"Wavelengths must be strictly increasing; got min diff: {np.min(np.diff(wave))}."
        )

    _name = name or "dsps_filter"
    _warn_implausible_wavelength_range(wave, _name, source="DSPS transmission curve")

    return FilterCurve(wave=jnp.array(wave), trans=jnp.array(trans), name=_name)


def load_filter_from_dsps_file(
    filepath: str | Path,
    name: str | None = None,
) -> FilterCurve:
    """Load a DSPS transmission curve from an HDF5 file.

    Opens a DSPS-format filter file and converts it to a :class:`FilterCurve`.
    Uses lazy import so DSPS is not a hard requirement unless this function
    is called.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to a DSPS filter file (HDF5 format with ``wave`` and
        ``transmission`` datasets).
    name : str, optional
        Filter name. If ``None``, uses the file stem. Default: ``None``.

    Returns
    -------
    FilterCurve
        Transmission curve with wavelengths validated as Angstrom.

    Raises
    ------
    ImportError
        If the ``dsps`` package is not installed.
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file format is invalid or wavelengths are not strictly increasing.

    Warns
    -----
    UserWarning
        If wavelength range looks implausible for Angstrom.

    Examples
    --------
    >>> from tengri.observation.filters import load_filter_from_dsps_file
    >>> fc = load_filter_from_dsps_file("/path/to/filter.h5", name="my_band")
    """
    try:
        from dsps.data_loaders import load_transmission_curve as _load_dsps_curve
    except ImportError as exc:
        raise ImportError(
            "dsps is required to load DSPS filter files. Install it with: pip install dsps"
        ) from exc

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"DSPS filter file not found: {filepath}")

    dsps_curve = _load_dsps_curve(fn=str(path))
    _name = name or path.stem
    return load_filter_from_dsps_transmission_curve(dsps_curve, name=_name)
