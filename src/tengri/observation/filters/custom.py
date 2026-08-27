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
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from tengri.config.exceptions import warn_measured
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

# Red edge of the extreme-UV gap, in Angstrom: the blue edge of GALEX FUV, the
# bluest bandpass tengri ships. Below this and above ~100 Å the ISM is opaque,
# so a curve lying wholly inside the interval is not a real bandpass. Measured
# from the cached curve (GALEX_GALEX_FUV.dat spans 1340-1810 Å), not rounded,
# because the whole point of the bound is that it sits just under a real filter.
_EUV_GAP_RED_EDGE_AA = 1340.0

# Multiply an input wavelength by this to reach Angstrom.
_WAVE_UNIT_TO_AA: dict[str, float] = {"AA": 1.0, "nm": 10.0, "um": 1.0e4}


def _to_angstrom(wave: np.ndarray, wave_unit: str) -> np.ndarray:
    """Convert a wavelength array to Angstrom, validating the unit name.

    Parameters
    ----------
    wave : ndarray
        Wavelength array in *wave_unit*.
    wave_unit : str
        One of ``"AA"``, ``"nm"``, ``"um"``.

    Returns
    -------
    ndarray
        Wavelengths in Angstrom.

    Raises
    ------
    ValueError
        If *wave_unit* is not a recognized name.

    Notes
    -----
    Stating the unit is the only protection against micron confusion, which
    :func:`_warn_implausible_wavelength_range` explicitly cannot detect: an
    optical curve given in microns lands at 0.5-0.7 Å, indistinguishable from
    a real NuSTAR band. The range heuristic is a backstop for callers who do
    not use this; it is not a substitute for it.
    """
    try:
        factor = _WAVE_UNIT_TO_AA[wave_unit]
    except KeyError:
        raise ValueError(
            f"Unknown wave_unit={wave_unit!r}. Valid options: {sorted(_WAVE_UNIT_TO_AA)}."
        ) from None
    return wave if factor == 1.0 else wave * factor


def _first_data_line(filepath: Path) -> tuple[str, int]:
    """Return the first non-blank, non-comment line and its 0-based line number.

    Returns ``("", 0)`` for a file with no such line.
    """
    with open(filepath) as handle:
        for index, raw in enumerate(handle):
            line = raw.strip()
            if line and not line.startswith("#"):
                return line, index
    return "", 0


def _sniff_filter_format(filepath: Path) -> tuple[str | None, int]:
    """Infer the delimiter and header-row count of a two-column curve file.

    Returns
    -------
    delimiter : str or None
        ``","`` for a comma-separated file, ``None`` for whitespace, which is
        what :func:`numpy.loadtxt` wants for its default split.
    skiprows : int
        Raw lines for :func:`numpy.loadtxt` to drop before parsing: enough to
        clear a non-numeric header row (``lam,trans``) *and* any comment or
        blank lines above it, else ``0``.

        Counting the preamble matters because ``skiprows`` is applied to raw
        lines before ``comments`` strips anything. A bare ``1`` on a file whose
        first line is ``# note`` would discard the comment and hand the header
        straight to the parser.

    Notes
    -----
    The two axes are sniffed independently because they vary independently:
    the SVO cache writes whitespace with no header, while a curve exported
    from a spreadsheet or pandas is comma-separated *with* one. The directory
    loader has advertised ``.csv`` since it was written, so a real
    comma-separated file has to parse.

    Detection is by parse attempt, not by file extension. A ``.csv`` holding
    whitespace columns and a ``.dat`` holding commas both occur in the wild,
    and the extension is the least reliable signal about either.
    """
    line, line_number = _first_data_line(filepath)
    delimiter = "," if "," in line else None
    fields = line.split(",") if delimiter else line.split()

    skiprows = 0
    if fields:
        try:
            float(fields[0])
        except ValueError:
            skiprows = line_number + 1

    return delimiter, skiprows


def _load_filter_file(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read two-column text file (wavelength, transmission).

    Accepts whitespace- or comma-separated columns, with or without a single
    non-numeric header row, and strips ``#`` comments. Used by both custom.py
    and __init__.py.
    """
    delimiter, skiprows = _sniff_filter_format(filepath)
    data = np.loadtxt(str(filepath), delimiter=delimiter, skiprows=skiprows)
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
    The rule is deliberately a single narrow one: warn when the curve lies
    **entirely inside the extreme-UV gap**, 100 Å to
    :data:`_EUV_GAP_RED_EDGE_AA`.

    That window is empty for a physical reason, not a statistical one. The
    interstellar medium is opaque between the Lyman limit and the soft X-ray
    band, so no photometric survey observes there and tengri ships no filter in
    it -- the bluest is GALEX FUV at 1340 Å and the reddest X-ray band tops out
    near 100 Å. A curve sitting wholly in the gap is therefore almost always an
    optical bandpass whose wavelengths are in nanometers (500-700 nm read as
    500-700 Å).

    The red edge is the *bluest real bandpass*, not a round number. It was
    ``max < 1000.0``, which is where the rule failed: a 7DT curve set
    zero-padded to exactly 300-1000 nm gave ``wave_max == 1000.0``, and
    ``1000.0 < 1000.0`` is False, so the guard written to catch nanometers went
    silent on 23 files that were entirely in nanometers. Padding an optical
    curve out to a round 1000 nm is the most natural grid there is, so that
    boundary was hit by the first real user to try it rather than rarely.
    Anchoring to GALEX FUV instead both closes that case and widens the catch
    to any optical/NIR set tabulated in nanometers out to 1340 nm.

    What still gets through, and why it has to: a curve in nanometers extending
    past ~1340 nm reads as >1340 Å and is indistinguishable, by range alone,
    from a genuine UV bandpass. Catching it would mean warning on GALEX FUV
    itself. Declare the unit with ``wave_unit=`` when the range cannot settle
    it.

    Rules that were tried and rejected, because tengri spans X-ray to radio and
    each produced false positives on ordinary curves:

    * ``max < 50 Å`` -- flags hard X-ray bands, which legitimately live at
      1-25 Å (``chandra_hard`` spans 1.8-6.2 Å).
    * ``min > 1e7 Å`` -- flags every (sub)mm band; ``alma_band6`` is 1.2e7 Å and
      ``planck_lfi_030`` is 1.1e8 Å.
    * ``median < 5000 Å`` -- flags any ordinary blue filter. SDSS *u* sits at
      ~3550 Å with a ~600 Å width, and so would have warned on every use.

    Micron-scale confusion is **not** detectable and is not attempted: an
    optical curve given in microns lands at 0.5-0.7 Å, which is 18-25 keV --
    indistinguishable from a real NuSTAR band. Claiming to catch it would mean
    warning on legitimate X-ray input.
    """
    wave_arr = np.asarray(wave)
    if wave_arr.size < 2:
        return

    wave_min, wave_max = float(np.min(wave_arr)), float(np.max(wave_arr))

    if wave_min >= 100.0 and wave_max <= _EUV_GAP_RED_EDGE_AA:
        warn_measured(
            f"Filter '{name}' {source} spans {wave_min:.0f}-{wave_max:.0f} Å, which lies "
            f"entirely in the extreme-UV (100-{_EUV_GAP_RED_EDGE_AA:.0f} Å). The ISM is "
            f"opaque there, so no photometric bandpass exists in that window -- these are "
            f"most likely **nanometers**. If so, multiply by 10: "
            f"{wave_min:.0f}-{wave_max:.0f} nm "
            f"-> {wave_min * 10:.0f}-{wave_max * 10:.0f} Å. tengri wavelengths are "
            f"Angstrom throughout. Pass wave_unit='nm' to convert at the boundary "
            f"instead of relying on this check.",
            UserWarning,
            stacklevel=4,
            wave_min_aa=wave_min,
            wave_max_aa=wave_max,
        )


def register_filter(
    name: str,
    wave_aa: np.ndarray | list,
    trans: np.ndarray | list,
    *,
    overwrite: bool = False,
    wave_unit: str = "AA",
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
        Wavelength grid, in *wave_unit* (Angstrom by default). Must be
        strictly increasing.
    trans : array_like, shape (n_wave,)
        Transmission (dimensionless). Normalization is free: only the shape of
        the curve enters AB synthetic photometry, since a constant scale
        cancels in the ratio of the two bandpass integrals.
    overwrite : bool, optional
        If ``False`` (default), raise if *name* already exists in any registry
        (built-in, SVO, synthetic, or user). If ``True``, silently replace
        an existing user-registered filter or shadow a built-in name.
        Default: ``False``.
    wave_unit : str, optional
        Unit of *wave_aa*: ``"AA"`` (default), ``"nm"``, or ``"um"``.
        Anything other than ``"AA"`` is converted here, at the boundary, and
        the range heuristic is skipped because the unit is no longer in doubt.
        Default: ``"AA"``.

    Raises
    ------
    ValueError
        If *wave_aa* and *trans* have incompatible shapes.
    ValueError
        If *wave_aa* is not strictly increasing after sanitization.
    ValueError
        If *wave_unit* is not one of ``"AA"``, ``"nm"``, ``"um"``.
    KeyError
        If *name* exists in a built-in registry and ``overwrite=False``.

    Warns
    -----
    UserWarning
        If the wavelength range looks implausible for Angstrom (likely nm).
        Only raised when ``wave_unit="AA"``, since otherwise the caller has
        stated the unit and there is nothing to guess.

    Notes
    -----
    Shadowing built-in curves (a name that already exists in ``FILTER_REGISTRY``,
    SVO display stems, or ``SYNTHETIC_BAND_REGISTRY``) requires explicit
    ``overwrite=True`` to prevent accidental collisions. This is a deliberate
    gate: the default behavior keeps user registrations from silently hiding
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
        BUNDLED_FILTER_REGISTRY,
        FILTER_REGISTRY,
        SYNTHETIC_BAND_REGISTRY,
        _svo_name_to_key,
    )

    # Check for collisions
    exists_in_registry = name in FILTER_REGISTRY
    exists_in_svo = name in _svo_name_to_key()
    exists_in_synthetic = name in SYNTHETIC_BAND_REGISTRY
    exists_in_bundled = name in BUNDLED_FILTER_REGISTRY
    exists_in_user = name in _USER_FILTER_REGISTRY

    builtin_collision = (
        exists_in_registry or exists_in_svo or exists_in_synthetic or exists_in_bundled
    )
    collision = builtin_collision or exists_in_user

    if collision and not overwrite:
        source = "built-in FILTER_REGISTRY"
        if exists_in_svo:
            source = "SVO display name"
        elif exists_in_synthetic:
            source = "synthetic band registry"
        elif exists_in_bundled:
            source = "bundled measured curves"
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

    # Convert to Angstrom before anything else reads the numbers as Angstrom.
    wave = _to_angstrom(wave, wave_unit)

    wave, trans = _sanitize_filter_curve(wave, trans)

    # Check that wavelength is strictly increasing
    if wave.size > 1 and not np.all(np.diff(wave) > 0):
        raise ValueError(
            f"Wavelengths must be strictly increasing after sanitization. "
            f"Got min diff: {np.min(np.diff(wave))}."
        )

    # Warn about implausible ranges, but only where the unit was left to be
    # inferred. A stated wave_unit has already settled it.
    if wave_unit == "AA":
        _warn_implausible_wavelength_range(wave, name, source="custom filter")

    # Store as immutable FilterCurve with JAX arrays (prevent accidental mutation)
    fc = FilterCurve(wave=jnp.array(wave), trans=jnp.array(trans), name=name)
    _USER_FILTER_REGISTRY[name] = fc


def register_filter_from_file(
    name: str,
    filepath: str | Path,
    *,
    overwrite: bool = False,
    wave_unit: str = "AA",
) -> None:
    """Register a filter from a two-column text file.

    Reads wavelength and transmission columns, sanitizes, and registers
    in the user registry.

    Parameters
    ----------
    name : str
        Filter identifier (e.g., ``"my_band_from_file"``).
    filepath : str or pathlib.Path
        Path to a two-column file of wavelength and transmission. Columns may
        be whitespace- or comma-separated, with or without a single header
        row; ``#`` comments are stripped.
    overwrite : bool, optional
        If ``False`` (default), raise if *name* already exists. If ``True``,
        silently replace an existing user-registered filter or shadow a
        built-in name. Default: ``False``.
    wave_unit : str, optional
        Unit of the wavelength column: ``"AA"`` (default), ``"nm"``, or
        ``"um"``. Default: ``"AA"``.

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

    A curve tabulated in nanometers, converted at the boundary:

    >>> register_filter_from_file(  # doctest: +SKIP
    ...     "m400", "7DT_transmission_23bands/m400.csv", wave_unit="nm"
    ... )
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Filter file not found: {filepath}")

    wave, trans = _load_filter_file(path)
    register_filter(name, wave, trans, overwrite=overwrite, wave_unit=wave_unit)


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
    This route is reproducible across machines: a config referring to
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
    incorrect. Getting this wrong is a silent, catastrophic error, so verify
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
