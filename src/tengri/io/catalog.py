# SPDX-License-Identifier: BSD-3-Clause
"""Photometric catalog reader (CSV / VOTable / FITS table).

Loads a row-per-galaxy catalog and tries to auto-detect the columns
that matter for SED fitting:

- a redshift column (``z``, ``redshift``, ``z_phot``, ``z_spec``, ...)
- one ``flux_<band>`` column per photometric band, plus a matching
  ``flux_err_<band>`` (or ``ferr_<band>`` / ``<band>_err``)

The result is a tidy dict with the redshift array and a per-band
``{"flux": ..., "err": ...}`` mapping. Tengri's ``Galaxy.from_catalog_row``
(F1, deferred) consumes that shape directly; users can also work with
the dict.

Examples
--------
>>> from tengri.io import load_catalog
>>> cat = load_catalog("3dhst_photoz.csv")
>>> cat["redshift"].shape
(1234,)
>>> sorted(cat["bands"])
['acs_f435w', 'acs_f606w', 'acs_f814w', 'wfc3_f160w']
>>> cat["bands"]["wfc3_f160w"]["flux"].shape
(1234,)
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["CANDELS_TO_TENGRI", "CIGALE_TO_TENGRI", "load_catalog"]


_REDSHIFT_NAMES = (
    "redshift",
    "z",
    "z_phot",
    "z_spec",
    "z_best",
    "zphot",
    "zspec",
    "z_peak",
)

#: CIGALE catalog column -> tengri filter name, for ``filter_mapping``. A column the
#: catalog lacks is ignored, so this is a table of every name a release may use.
CIGALE_TO_TENGRI: dict[str, str] = {
    "GALEX_FUV": "galex_fuv",
    "GALEX_NUV": "galex_nuv",
    "sdss.up": "sdss_u",
    "sdss.gp": "sdss_g",
    "sdss.rp": "sdss_r",
    "sdss.ip": "sdss_i",
    "sdss.zp": "sdss_z",
    "2mass.J": "2mass_j",
    "2mass.H": "2mass_h",
    "2mass.Ks": "2mass_ks",
    "IRAC1": "irac_36",
    "IRAC2": "irac_45",
    "IRAC3": "irac_58",
    "IRAC4": "irac_80",
    "MIPS1": "mips_24",
    "WISE1": "wise_w1",
    "WISE2": "wise_w2",
    "WISE3": "wise_w3",
    "WISE4": "wise_w4",
    "herschel.pacs.70": "herschel_70",
    "herschel.pacs.100": "herschel_100",
    "herschel.pacs.160": "herschel_160",
    "herschel.spire.250": "herschel_250",
    "herschel.spire.350": "herschel_350",
    "herschel.spire.500": "herschel_500",
}

#: CANDELS catalog column -> tengri filter name, for ``filter_mapping``. A reference
#: table: a column absent from a given catalog release is simply not looked up, so
#: entries here are a superset rather than a contract. The two GOODS-S U bands are
#: included because dropping them discards the bluest measurements in the catalog,
#: which at z ~ 1 are the rest-frame ultraviolet.
CANDELS_TO_TENGRI: dict[str, str] = {
    "CTIO_U": "ctio_u",
    "VIMOS_U": "vimos_u",
    "ACS_F435W": "hst_f435w",
    "ACS_F606W": "hst_f606w",
    "ACS_F775W": "hst_f775w",
    "ACS_F814W": "hst_f814w",
    "ACS_F850LP": "hst_f850lp",
    "WFC3_F105W": "hst_f105w",
    "WFC3_F125W": "hst_f125w",
    "WFC3_F140W": "hst_f140w",
    "WFC3_F160W": "hst_f160w",
    "IRAC_CH1": "irac_36",
    "IRAC_CH2": "irac_45",
    "IRAC_CH3": "irac_58",
    "IRAC_CH4": "irac_80",
}


def _norm(s: str) -> str:
    """Normalize a column name for comparison: lowercase, no punctuation."""
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def _detect_redshift_column(columns: list[str]) -> str | None:
    norm = {_norm(c): c for c in columns}
    for cand in _REDSHIFT_NAMES:
        if cand in norm:
            return norm[cand]
    return None


def _detect_flux_columns(columns: list[str]) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Pair each ``flux_<band>`` column with its matching error column.

    Recognizes three error-column conventions: ``flux_err_<band>``,
    ``ferr_<band>``, and ``<band>_err`` (case-insensitive).

    Returns
    -------
    bands : dict
        Paired bands, ``{band: {"flux": col, "err": col}}``.
    unpaired : list of str
        Bands with a ``flux_<band>`` column and no error column, sorted.
    """
    norm_to_orig = {_norm(c): c for c in columns}
    bands: dict[str, dict[str, str]] = {}
    unpaired: list[str] = []

    for n, original in norm_to_orig.items():
        if not n.startswith("flux_") or n.startswith("flux_err_"):
            continue
        band = n.removeprefix("flux_")
        # Look for the error column under each known convention.
        candidates = (f"flux_err_{band}", f"ferr_{band}", f"{band}_err", f"flux_{band}_err")
        err_col = next((norm_to_orig[c] for c in candidates if c in norm_to_orig), None)
        if err_col is None:
            unpaired.append(band)
            continue
        bands[band] = {"flux": original, "err": err_col}
    return bands, sorted(unpaired)


def _mapped_flux_columns(
    columns: list[str], filter_mapping: dict[str, str]
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Pair each ``filter_mapping`` flux column with its ``<column>_err`` error column.

    A mapping key absent from the table is ignored, so a prebuilt map may name
    columns a given release lacks. Returns the paired bands (keyed by tengri band
    name) and the band names whose error column is absent, sorted.
    """
    present = set(columns)
    bands: dict[str, dict[str, str]] = {}
    unpaired: list[str] = []
    claimed: dict[str, str] = {}
    for col, band in filter_mapping.items():
        if col not in present:
            continue
        if band in claimed:
            raise ValueError(
                f"filter_mapping maps two columns onto band {band!r}: "
                f"{claimed[band]!r} and {col!r}. Map one column to each band."
            )
        claimed[band] = col
        err_col = f"{col}_err"
        if err_col in present:
            bands[band] = {"flux": col, "err": err_col}
        else:
            unpaired.append(band)
    return bands, sorted(unpaired)


def _read_table(path: Path, delimiter: str | None) -> tuple[list[str], list[Any]]:
    """Read CSV / TSV / VOTable / FITS into (column_names, list_of_arrays).

    Each array has the same length (number of rows). ``delimiter`` applies to
    CSV / TSV / text files; ``None`` takes the tab for ``.tsv`` and sniffs otherwise.
    """
    suffix = path.suffix.lower()

    if suffix in (".csv", ".tsv", ".txt"):
        # pandas handles dtype detection and missing values better than csv.
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover
            raise ImportError("load_catalog needs pandas for CSV/TSV: pip install pandas") from e
        if path.stat().st_size == 0:
            raise ValueError(f"No headers found in {path}")
        if delimiter is not None:
            sep = delimiter
        else:
            sep = "\t" if suffix == ".tsv" else None  # None = sniff
        df = pd.read_csv(path, sep=sep, comment="#", engine="python")
        return list(df.columns), [df[c].to_numpy() for c in df.columns]

    if suffix in (".vot", ".xml"):
        try:
            from astropy.io.votable import parse_single_table
        except ImportError as e:  # pragma: no cover
            raise ImportError("load_catalog needs astropy for VOTable: pip install astropy") from e
        table = parse_single_table(str(path)).to_table()
        return list(table.colnames), [np.asarray(table[c]) for c in table.colnames]

    if suffix in (".fits", ".fit", ".fts"):
        try:
            from astropy.table import Table
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "load_catalog needs astropy for FITS tables: pip install astropy"
            ) from e
        table = Table.read(path)
        return list(table.colnames), [np.asarray(table[c]) for c in table.colnames]

    raise ValueError(
        f"Unsupported catalog format {suffix!r}. "
        f"Expected one of: .csv, .tsv, .txt, .vot, .xml, .fits, .fit, .fts."
    )


_LIMIT_STATUSES = ("upper_limit", "lower_limit")


def _parse_sentinel_status(key, status) -> tuple[str, float | None]:
    """Validate one declared sentinel status; return ``(kind, n_sigma)``.

    ``kind`` is ``"nodata"``, ``"upper_limit"`` or ``"lower_limit"``; ``n_sigma`` is
    ``None`` for ``"nodata"``. A bare limit name without a depth is refused.
    """
    allowed = "'nodata', ('upper_limit', n_sigma), ('lower_limit', n_sigma)"
    if isinstance(status, str):
        if status == "nodata":
            return "nodata", None
        if status in _LIMIT_STATUSES:
            raise ValueError(
                f"sentinels[{key!r}] = {status!r} gives no depth. Declare it as "
                f"({status!r}, n_sigma), e.g. ({status!r}, 3.0): the limit flux is "
                f"n_sigma times the error."
            )
        raise ValueError(f"sentinels[{key!r}] = {status!r} is not a status; allowed: {allowed}.")
    if isinstance(status, tuple | list) and len(status) == 2 and status[0] in _LIMIT_STATUSES:
        n_sigma = float(status[1])
        if not (np.isfinite(n_sigma) and n_sigma > 0.0):
            raise ValueError(
                f"sentinels[{key!r}] = {tuple(status)!r}: n_sigma must be finite and > 0; "
                f"got {status[1]!r}."
            )
        return str(status[0]), n_sigma
    raise ValueError(f"sentinels[{key!r}] = {status!r} is not a status; allowed: {allowed}.")


def _apply_sentinels(band, flux, err, sentinels):
    """Replace declared sentinel flux cells by absent cells or censored limits (#3547 D4).

    Returns new ``(flux, err, censor)`` arrays; the inputs are not modified. Every
    status is validated before any cell is read, so a bad declaration fails on a
    table that does not contain its sentinel.
    """
    flux = flux.copy()
    err = err.copy()
    censor = np.zeros(flux.shape, dtype=int)
    for key, status in sentinels.items():
        value = float(key)
        kind, n_sigma = _parse_sentinel_status(key, status)
        if np.any(err == value):
            raise ValueError(
                f"band {band!r}: declared sentinel {key!r} appears in the error column. "
                f"A sentinel marks a flux cell; an error cannot be a sentinel."
            )
        hit = flux == value
        if not hit.any():
            continue
        if kind == "nodata":
            flux[hit] = np.nan
            err[hit] = np.nan
            continue
        if np.any(err[hit] <= 0.0) or np.any(~np.isfinite(err[hit])):
            rows = np.flatnonzero(hit & ~((err > 0.0) & np.isfinite(err))).tolist()
            raise ValueError(
                f"band {band!r}: {kind} sentinel {key!r} at row(s) {rows} has a "
                f"non-positive or non-finite error; a limit needs a positive error as its sigma."
            )
        flux[hit] = n_sigma * err[hit]
        censor[hit] = 1 if kind == "upper_limit" else -1
    return flux, err, censor


def load_catalog(
    path: str | Path,
    redshift_col: str | None = None,
    *,
    allow_unpaired_bands: bool = False,
    filter_mapping: dict[str, str] | None = None,
    id_col: str | None = None,
    delimiter: str | None = None,
    sentinels: Mapping[float, str | tuple[str, float]] | None = None,
) -> dict[str, Any]:
    """Load a photometric catalog with auto-detected flux/error columns.

    Parameters
    ----------
    path : str or Path
        Catalog file. Format is detected from the suffix:
        ``.csv`` / ``.tsv`` / ``.txt`` (pandas), ``.vot`` / ``.xml``
        (astropy VOTable), ``.fits`` / ``.fit`` / ``.fts`` (astropy
        Table).
    redshift_col : str, optional
        Name of the redshift column. If ``None`` (default), tries the
        common names: ``redshift``, ``z``, ``z_phot``, ``z_spec``,
        ``z_best``, ``zphot``, ``zspec``, ``z_peak``.
    allow_unpaired_bands : bool, optional
        If ``False`` (default), a ``flux_<band>`` column with no error
        column raises ``ValueError``. If ``True``, such bands are dropped
        with a ``UserWarning`` and listed under ``"dropped_bands"``.
    filter_mapping : dict, optional
        Maps a catalog flux column name to a tengri band name, for catalogs whose
        columns are not named ``flux_<band>``. The error column is the same name
        with ``_err`` appended (CIGALE and CANDELS convention). Mapped bands are
        added to the auto-detected ones. A key absent from the table is ignored, so
        the prebuilt :data:`CIGALE_TO_TENGRI` and :data:`CANDELS_TO_TENGRI` maps can
        be passed whole. Default ``None``.
    id_col : str, optional
        Name of the galaxy identifier column, returned as ``"ids"``. If ``None``
        (default), the column ``id`` is used when present and no ids are returned.
        An explicitly named column that is absent raises ``KeyError``.
    delimiter : str, optional
        Field delimiter for CSV / TSV / text files. If ``None`` (default), the tab
        is used for ``.tsv`` and the delimiter is sniffed otherwise.
    sentinels : Mapping[float, str or tuple[str, float]], optional
        Sentinel flux values declared at read time, keyed by the exact value found in a
        band's flux column (compared with ``==``, no tolerance). Each band gains a
        ``"censor"`` array, shape ``(n_rows,)``, dtype int, when this is given. Statuses:

        - ``"nodata"``: the cell is absent. Its flux and error become NaN, so
          ``ingest_catalog(..., missing="mask")`` excludes it. This is no
          information, not a limit.
        - ``("upper_limit", n_sigma)`` or ``("lower_limit", n_sigma)``: the cell is a
          censored limit. Its flux becomes ``n_sigma * err`` [flux unit], its error
          stays ``err`` [flux unit] (the sigma), and ``censor`` is +1 (upper) or
          -1 (lower). ``n_sigma`` must be finite and positive; the depth is required.

        A sentinel in an error column, or a limit row with a non-positive error, raises.
        ``None`` (default) reads flux as it is in the file.

    Returns
    -------
    dict
        Always contains:

        - ``"path"`` : ``Path`` of the source file.
        - ``"n_rows"`` : int.
        - ``"bands"`` : dict mapping each detected band (e.g.
          ``"sdss_g"``) to ``{"flux": ndarray, "err": ndarray}``,
          one ndarray per row.
        - ``"dropped_bands"`` : sorted list of band names dropped for
          lacking an error column; empty unless ``allow_unpaired_bands``.

        When a redshift column is detected:

        - ``"redshift"`` : ndarray of shape ``(n_rows,)``.
        - ``"redshift_col"`` : str. Name of the column used.

        When an identifier column is detected:

        - ``"ids"`` : ndarray of str, shape ``(n_rows,)``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file suffix is not recognized, if no ``flux_*`` /
        error-column pairs or mapped bands are found, or if two mapped columns
        name the same band.
    KeyError
        If ``redshift_col`` or ``id_col`` is given explicitly but not present.

    Notes
    -----
    Recognized flux / error column conventions (case-insensitive,
    underscores collapsed):

    - ``flux_<band>`` paired with ``flux_err_<band>``
    - ``flux_<band>`` paired with ``ferr_<band>``
    - ``flux_<band>`` paired with ``<band>_err``
    - ``flux_<band>`` paired with ``flux_<band>_err``

    ``filter_mapping`` adds bands under the ``<column>`` / ``<column>_err``
    convention. A band without a matching error column cannot enter an SED fit,
    so it raises ``ValueError`` naming every such band. Pass
    ``allow_unpaired_bands=True`` to drop them instead; the drop is then
    announced with a ``UserWarning`` and listed under ``"dropped_bands"``.
    If a catalog uses a different convention, pass ``filter_mapping`` or rename
    columns before calling.

    This reader keeps flux as it is in the file. The sentinel and negative-error
    conventions of CIGALE are read by :class:`~tengri.Catalog` through its
    ``lim_flag`` option, which refuses them unless declared.

    Examples
    --------
    >>> from tengri.io import load_catalog
    >>> cat = load_catalog("3dhst_v4.1.5_photometry.csv")
    >>> cat["redshift"].shape
    (50000,)
    >>> cat["bands"]["wfc3_f160w"]["flux"].shape
    (50000,)

    A table that marks its non-detections with -99 declares what each means, and the
    result goes to ``ingest_catalog`` with ``missing="mask"`` and the censor columns:

    >>> cat = load_catalog(
    ...     "survey.csv",
    ...     sentinels={-99.0: ("upper_limit", 3.0)},  # flux -99 -> 3 sigma upper limit
    ... )  # doctest: +SKIP
    >>> cat["bands"]["sdss_g"]["censor"]  # doctest: +SKIP
    array([0, 1, 0])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    columns, arrays = _read_table(path, delimiter)
    by_name = dict(zip(columns, arrays, strict=True))
    n_rows = arrays[0].shape[0] if arrays else 0
    if n_rows == 0:
        raise ValueError(f"Empty catalog: {path}")

    out: dict[str, Any] = {"path": path, "n_rows": int(n_rows)}

    # Redshift column.
    z_col = redshift_col if redshift_col is not None else _detect_redshift_column(columns)
    if z_col is not None:
        if z_col not in by_name:
            raise KeyError(f"redshift_col {z_col!r} not in catalog columns")
        out["redshift"] = np.asarray(by_name[z_col], dtype=float)
        out["redshift_col"] = z_col

    # Identifier column: explicit, or the conventional ``id`` when present.
    id_name = id_col if id_col is not None else ("id" if "id" in by_name else None)
    if id_name is not None:
        if id_name not in by_name:
            raise KeyError(f"id_col {id_name!r} not in catalog columns")
        out["ids"] = np.asarray(by_name[id_name]).astype(str)

    # Flux/err pairs. A flux column with no error column is a band the fit
    # would silently lose, so it is an error unless the caller opted in (#3285).
    # ``filter_mapping`` names bands the auto-detection does not find; a band
    # claimed by both is refused rather than one silently overwriting the other.
    band_cols, unpaired = _detect_flux_columns(columns)
    if filter_mapping:
        mapped, mapped_unpaired = _mapped_flux_columns(columns, filter_mapping)
        clash = sorted(set(mapped) & set(band_cols))
        if clash:
            raise ValueError(
                f"filter_mapping names band(s) {clash} that the flux_<band> columns "
                f"also provide. Map those columns away, or drop the mapping."
            )
        band_cols = {**band_cols, **mapped}
        unpaired = sorted(set(unpaired) | set(mapped_unpaired))
    if not band_cols:
        raise ValueError(
            f"No flux / error column pairs found in {path.name}. "
            f"Expected columns of the form flux_<band> + flux_err_<band> "
            f"(or ferr_<band>, <band>_err, flux_<band>_err), or pass filter_mapping "
            f"to name the columns." + (f" Unpaired flux columns: {unpaired}." if unpaired else "")
        )
    if unpaired and not allow_unpaired_bands:
        raise ValueError(
            f"{path.name}: flux columns without a matching error column, which "
            f"cannot enter an SED fit: {unpaired}. Add the error columns, or pass "
            f"allow_unpaired_bands=True to drop these bands."
        )
    if unpaired:
        warnings.warn(
            f"{path.name}: dropped bands without a matching error column: {unpaired}.",
            UserWarning,
            stacklevel=2,
        )

    bands: dict[str, dict[str, np.ndarray]] = {}
    for band, cols in band_cols.items():
        flux = np.array(by_name[cols["flux"]], dtype=float)
        err = np.array(by_name[cols["err"]], dtype=float)
        if sentinels is None:
            bands[band] = {"flux": flux, "err": err}
        else:
            flux, err, censor = _apply_sentinels(band, flux, err, sentinels)
            bands[band] = {"flux": flux, "err": err, "censor": censor}
    out["bands"] = bands
    out["dropped_bands"] = unpaired
    return out
