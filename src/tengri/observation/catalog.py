# SPDX-License-Identifier: BSD-3-Clause
"""Photometric catalog reader with automatic filter matching and mask generation.

Reads CSV/ASCII catalogs into arrays ready for :class:`~tengri.inference.fitter.Fitter`,
handling non-detections (upper/lower limits), unit conversion, and filter name
resolution against :data:`~tengri.observation.filters.FILTER_REGISTRY`.

Conventions for flagging censored data
---------------------------------------

- Missing data: flux = -9999 **and** error = -9999 → masked out entirely
- Upper limit (CIGALE convention): positive flux + negative error → ``UPPER_LIMIT``
- Lower limit: negative flux + positive error → ``LOWER_LIMIT``

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tengri.observation.filters import FILTER_REGISTRY
from tengri.observation.noise import DETECTED, LOWER_LIMIT, UPPER_LIMIT

MISSING_VALUE = -9999.0

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

CANDELS_TO_TENGRI: dict[str, str] = {
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


@dataclass(frozen=True)
class Catalog:
    """Photometric catalog ready for fitting.

    Attributes
    ----------
    ids: array, shape (n_galaxies,)
        Galaxy identifiers (string or int).
    redshifts: array, shape (n_galaxies,)
        Redshifts [dimensionless].
    flux: array, shape (n_galaxies, n_filters)
        Flux values [mJy or specified flux_unit].
        For upper-limit bands, holds the limit value.
    noise: array, shape (n_galaxies, n_filters)
        1-sigma uncertainties (always positive) [mJy or specified flux_unit].
    mask: array, shape (n_galaxies, n_filters)
        Per-band type: 0 = detected, 1 = upper limit, -1 = lower limit
        [dimensionless].
    filter_names: tuple of str
        Tengri filter names corresponding to flux columns.
    flux_unit: str
        Unit of flux values (e.g. ``"mJy"``, ``"uJy"``). Default ``"mJy"``.

    Notes
    -----
    Immutable: use indexing and helper methods (``galaxy``, ``select_detected``)
    to extract subset data. Missing data is flagged with very large noise
    (1e30) to effectively exclude it from likelihood calculations.
    """

    ids: np.ndarray
    redshifts: np.ndarray
    flux: np.ndarray
    noise: np.ndarray
    mask: np.ndarray
    filter_names: tuple[str, ...]
    flux_unit: str = "mJy"

    def galaxy(self, idx: int) -> dict:
        """Extract single galaxy data ready for ``Fitter``.

        Parameters
        ----------
        idx: int
            Row index (0-based).

        Returns
        -------
        dict
            Keys: ``id`` (scalar), ``redshift`` (scalar),
            ``flux`` (array, shape ``(n_filters,)``), ``noise`` (array, shape
            ``(n_filters,)``), ``mask`` (array, shape ``(n_filters,)``),
            ``filter_names`` (tuple of str). Flux and noise are in ``flux_unit``.

        Notes
        -----
        Returns an immutable copy view; modifying the returned flux, noise,
        or mask arrays does not affect the catalog.

        Examples
        --------
        .. code-block:: python

            from tengri import GalaxyCatalog

            cat = GalaxyCatalog.from_fits("my_catalog.fits", filter_names=["sdss_r", "sdss_i"])
            g = cat.galaxy(0)
            g["id"]  # galaxy identifier
            g["flux"].shape  # (n_filters,)
        """
        return {
            "id": self.ids[idx],
            "redshift": self.redshifts[idx],
            "flux": self.flux[idx],
            "noise": self.noise[idx],
            "mask": self.mask[idx],
            "filter_names": self.filter_names,
        }

    def select_detected(self, idx: int) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Return only detected bands for a galaxy.

        Filters out upper and lower limits, returning only bands with
        ``mask == 0`` (DETECTED).

        Parameters
        ----------
        idx: int
            Row index (0-based).

        Returns
        -------
        flux: ndarray, shape (n_detected,)
            Flux values for detected bands [flux_unit].
        noise: ndarray, shape (n_detected,)
            1-sigma noise for detected bands [flux_unit].
        filter_names: tuple of str
            Filter names for detected bands only.

        Notes
        -----
        Returns views (not copies) when possible to minimize memory allocation.
        """
        det = self.mask[idx] == DETECTED
        names = tuple(n for n, d in zip(self.filter_names, det) if d)
        return self.flux[idx, det], self.noise[idx, det], names

    @property
    def n_galaxies(self) -> int:
        """Number of galaxies in the catalog.

        Returns
        -------
        int
            Number of rows in the catalog.

        Notes
        -----
        Computed from the first axis of the ``flux`` array. Constant
        for the lifetime of the catalog (immutable).
        """
        return self.flux.shape[0]

    @property
    def n_filters(self) -> int:
        """Number of filters (photometric bands).

        Returns
        -------
        int
            Number of photometric bands (flux columns).

        Notes
        -----
        Computed from the second axis of the ``flux`` array. Equal to the
        length of ``filter_names``.
        """
        return self.flux.shape[1]

    def __repr__(self) -> str:
        return (
            f"Catalog({self.n_galaxies} galaxies, {self.n_filters} filters, "
            f"unit={self.flux_unit!r})"
        )


def read_catalog(
    filepath: str | Path,
    *,
    filter_mapping: dict[str, str] | None = None,
    flux_unit: str = "mJy",
    redshift_col: str = "redshift",
    id_col: str = "id",
    missing_value: float = MISSING_VALUE,
    delimiter: str = ",",
) -> Catalog:
    """Read a photometric catalog from a CSV file.

    Expects columns: ``id``, ``redshift``, then pairs of
    ``<filter>`` and ``<filter>_err`` for each photometric band.
    Column names are matched to ``FILTER_REGISTRY`` either directly
    or via ``filter_mapping``.

    Parameters
    ----------
    filepath: str or Path
        Path to CSV file.
    filter_mapping: dict, optional
        Maps catalog column names → tengri filter names.  If ``None``,
        column names are matched directly against ``FILTER_REGISTRY``.
        Default: ``None``.
    flux_unit: str, optional
        Unit label stored in the returned ``Catalog``. Default ``"mJy"``.
    redshift_col: str, optional
        Name of the redshift column. Default ``"redshift"``.
    id_col: str, optional
        Name of the ID column. Default ``"id"``.
    missing_value: float, optional
        Value indicating missing data. Default ``-9999``.
    delimiter: str, optional
        CSV delimiter. Default ``","``.

    Returns
    -------
    Catalog
        Parsed catalog with ``flux`` array, shape ``(n_galaxies, n_filters)``
        [flux_unit]; ``noise`` array [flux_unit]; ``mask`` array with values
        ``0`` (detected), ``1`` (upper limit), ``-1`` (lower limit); and
        ``filter_names`` tuple.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If no filter columns are found, or if the catalog is empty.

    Notes
    -----
    **Censoring convention**:

    - Missing (both flux and error = -9999) → masked out with large noise (1e30)
    - Upper limit (flux > 0, error < 0) → mask = 1
    - Lower limit (flux < 0, error > 0) → mask = -1
    - Detected (flux > 0, error > 0) → mask = 0

    Filter columns must have corresponding ``_err`` columns. Columns without
    an ``_err`` counterpart are silently skipped.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Catalog file not found: {filepath}")

    import csv

    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError(f"No headers found in {filepath}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty catalog: {filepath}")

    mapping = filter_mapping or {}

    filter_cols: list[tuple[str, str, str]] = []
    for col in headers:
        if col.endswith("_err") or col in (id_col, redshift_col):
            continue

        err_col = f"{col}_err"
        if err_col not in headers:
            continue

        tengri_name = mapping.get(col, col)
        if tengri_name in FILTER_REGISTRY:
            filter_cols.append((col, err_col, tengri_name))

    if not filter_cols:
        raise ValueError(
            f"No filter columns matched FILTER_REGISTRY in {filepath}. "
            f"Columns found: {headers}. "
            f"Use filter_mapping to map column names to tengri filter names."
        )

    n_gal = len(rows)
    n_filt = len(filter_cols)
    filter_names = tuple(t[2] for t in filter_cols)

    ids = np.array([r.get(id_col, str(i)) for i, r in enumerate(rows)])
    redshifts = np.array([float(r.get(redshift_col, 0.0)) for r in rows])

    flux = np.zeros((n_gal, n_filt))
    noise = np.zeros((n_gal, n_filt))
    mask = np.zeros((n_gal, n_filt), dtype=int)

    for j, (flux_col, err_col, _name) in enumerate(filter_cols):
        for i, row in enumerate(rows):
            f_val = float(row[flux_col])
            e_val = float(row[err_col])

            is_missing = abs(f_val - missing_value) < 1.0 and abs(e_val - missing_value) < 1.0

            if is_missing:
                flux[i, j] = 0.0
                noise[i, j] = 1e30
                mask[i, j] = DETECTED
            elif e_val < 0 and f_val > 0:
                flux[i, j] = f_val
                noise[i, j] = abs(e_val)
                mask[i, j] = UPPER_LIMIT
            elif f_val < 0 and e_val > 0:
                flux[i, j] = abs(f_val)
                noise[i, j] = e_val
                mask[i, j] = LOWER_LIMIT
            else:
                flux[i, j] = f_val
                noise[i, j] = abs(e_val) if e_val != 0 else 1e30
                mask[i, j] = DETECTED

    return Catalog(
        ids=ids,
        redshifts=redshifts,
        flux=flux,
        noise=noise,
        mask=mask,
        filter_names=filter_names,
        flux_unit=flux_unit,
    )
