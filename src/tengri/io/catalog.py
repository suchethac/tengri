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
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["load_catalog"]


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


def _norm(s: str) -> str:
    """Normalize a column name for comparison: lowercase, no punctuation."""
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def _detect_redshift_column(columns: list[str]) -> str | None:
    norm = {_norm(c): c for c in columns}
    for cand in _REDSHIFT_NAMES:
        if cand in norm:
            return norm[cand]
    return None


def _detect_flux_columns(columns: list[str]) -> dict[str, dict[str, str]]:
    """Pair each ``flux_<band>`` column with its matching error column.

    Recognizes three error-column conventions: ``flux_err_<band>``,
    ``ferr_<band>``, and ``<band>_err`` (case-insensitive).
    """
    norm_to_orig = {_norm(c): c for c in columns}
    bands: dict[str, dict[str, str]] = {}

    for n, original in norm_to_orig.items():
        if not n.startswith("flux_") or n.startswith("flux_err_"):
            continue
        band = n.removeprefix("flux_")
        # Look for the error column under each known convention.
        candidates = (f"flux_err_{band}", f"ferr_{band}", f"{band}_err", f"flux_{band}_err")
        err_col = next((norm_to_orig[c] for c in candidates if c in norm_to_orig), None)
        if err_col is None:
            continue
        bands[band] = {"flux": original, "err": err_col}
    return bands


def _read_table(path: Path) -> tuple[list[str], list[Any]]:
    """Read CSV / TSV / VOTable / FITS into (column_names, list_of_arrays).

    Each array has the same length (number of rows).
    """
    suffix = path.suffix.lower()

    if suffix in (".csv", ".tsv", ".txt"):
        # pandas handles dtype detection and missing values better than csv.
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover
            raise ImportError("load_catalog needs pandas for CSV/TSV: pip install pandas") from e
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


def load_catalog(
    path: str | Path,
    redshift_col: str | None = None,
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

    Returns
    -------
    dict
        Always contains:

        - ``"path"`` : ``Path`` of the source file.
        - ``"n_rows"`` : int.
        - ``"bands"`` : dict mapping each detected band (e.g.
          ``"sdss_g"``) to ``{"flux": ndarray, "err": ndarray}``,
          one ndarray per row.

        When a redshift column is detected:

        - ``"redshift"`` : ndarray of shape ``(n_rows,)``.
        - ``"redshift_col"``: str: name of the column used.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file suffix is not recognized, or if no ``flux_*`` /
        error-column pairs are found.
    KeyError
        If ``redshift_col`` is given explicitly but not present.

    Notes
    -----
    Recognized flux / error column conventions (case-insensitive,
    underscores collapsed):

    - ``flux_<band>`` paired with ``flux_err_<band>``
    - ``flux_<band>`` paired with ``ferr_<band>``
    - ``flux_<band>`` paired with ``<band>_err``
    - ``flux_<band>`` paired with ``flux_<band>_err``

    Bands without a matching error column are silently dropped; this
    is by design, since SED fitting needs (flux, err) pairs. If a
    catalog uses a different convention, rename columns before
    calling.

    Examples
    --------
    >>> from tengri.io import load_catalog
    >>> cat = load_catalog("3dhst_v4.1.5_photometry.csv")
    >>> cat["redshift"].shape
    (50000,)
    >>> cat["bands"]["wfc3_f160w"]["flux"].shape
    (50000,)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    columns, arrays = _read_table(path)
    by_name = dict(zip(columns, arrays, strict=True))
    n_rows = arrays[0].shape[0] if arrays else 0

    out: dict[str, Any] = {"path": path, "n_rows": int(n_rows)}

    # Redshift column.
    z_col = redshift_col if redshift_col is not None else _detect_redshift_column(columns)
    if z_col is not None:
        if z_col not in by_name:
            raise KeyError(f"redshift_col {z_col!r} not in catalog columns")
        out["redshift"] = np.asarray(by_name[z_col], dtype=float)
        out["redshift_col"] = z_col

    # Flux/err pairs.
    band_cols = _detect_flux_columns(columns)
    if not band_cols:
        raise ValueError(
            f"No flux / error column pairs found in {path.name}. "
            f"Expected columns of the form flux_<band> + flux_err_<band> "
            f"(or ferr_<band>, <band>_err, flux_<band>_err)."
        )

    bands: dict[str, dict[str, np.ndarray]] = {}
    for band, cols in band_cols.items():
        bands[band] = {
            "flux": np.asarray(by_name[cols["flux"]], dtype=float),
            "err": np.asarray(by_name[cols["err"]], dtype=float),
        }
    out["bands"] = bands
    return out
