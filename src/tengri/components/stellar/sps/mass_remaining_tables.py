"""Surviving-mass fractions from each SSP grid's own isochrones (#2751).

``ssp_mass_remaining[i_Z, i_age]`` is the fraction of the mass formed in a
single-age population that is, at that age, in living stars plus stellar
remnants (per 1 Msun formed, dimensionless). It depends on the isochrones, the
IMF and the metallicity, and not on the spectral library. This module resolves
it for a grid from, in order, a companion table shipped as package data (for a
registered grid, authoritative over any table the file carries), the grid file's
own table (for grids with no companion), or (only on an explicit opt-in) DSPS's
metallicity-independent sigmoid fit; anything else raises.

References
----------
.. [1] Conroy, C., Gunn, J. E. & White, M. 2009, ApJ, 699, 486,
       "The Propagation of Uncertainties in Stellar Population Synthesis
       Modeling. I.", arXiv:0809.4261, doi:10.1088/0004-637X/699/1/486.
.. [2] Renzini, A. & Ciotti, L. 1993, ApJ, 416, L49, "Transverse Dissections of
       the Fundamental Planes of Elliptical Galaxies and Clusters of Galaxies",
       doi:10.1086/187068 (the remnant prescription FSPS implements).
.. [3] Bruzual, G. & Charlot, S. 2003, MNRAS, 344, 1000, "Stellar population
       synthesis at the resolution of 2003", arXiv:astro-ph/0309134,
       doi:10.1046/j.1365-8711.2003.06897.x.
"""

from __future__ import annotations

import warnings
from functools import cache
from importlib.resources import as_file, files
from typing import NamedTuple

import h5py
import numpy as np

#: Registry ``source`` of a grid whose table has not been built yet.
PENDING = "PENDING"
#: Registry ``source`` of a grid that carries its own ``ssp_mass_remaining``.
EMBEDDED = "embedded"

#: Values accepted for the ``mass_remaining=`` argument of the public loaders.
MODE_TABLE = "table"
MODE_DSPS_FIT = "dsps_fit"
MASS_REMAINING_MODES = (MODE_TABLE, MODE_DSPS_FIT)

#: ``mass_remaining_source`` value of a fit-derived table.
SOURCE_DSPS_FIT = MODE_DSPS_FIT

#: Metallicity nodes of a grid and its companion table must agree to this [dex].
LOGZ_ATOL = 1e-6
#: A grid age within this of a table node [dex] is that node, not an interpolation.
AGE_NODE_ATOL = 1e-5

_PACKAGE = "tengri.data.ssp_mass_remaining"

#: Leading text of a companion file's ``isochrones`` attribute when it spells
#: the isochrone set out instead of repeating the registry code.
_ISOC_ATTR_PREFIX = {"bc03pdva94": "padova 1994"}


class MassRemainingEntry(NamedTuple):
    """Registry row: which isochrones and IMF a grid uses, and where its table is.

    Parameters
    ----------
    isoc : str
        Isochrone code of the companion file name (``mist``, ``prsc``, ``pdva``,
        ``bsti``, ``gnva``, ``bc03pdva94``, ``bpss``, ``pgny_mist``, ...).
    imf : str
        ``chabrier``, ``kroupa`` or ``salpeter``.
    source : str
        ``"embedded"`` (the grid file carries the table), a companion file name
        in ``tengri.data.ssp_mass_remaining``, or ``"PENDING"``.
    """

    isoc: str
    imf: str
    source: str


def _m(isoc: str, imf: str) -> MassRemainingEntry:
    return MassRemainingEntry(isoc, imf, f"mass_remaining_{isoc}_{imf}.h5")


def _p(isoc: str, imf: str) -> MassRemainingEntry:
    return MassRemainingEntry(isoc, imf, PENDING)


#: Canonical grid name -> table. Every grid of ``tengri._data_setup._KNOWN_SSPS``
#: plus the locally produced ones appears here; a grid absent from it is a
#: user-supplied grid. The canonical name is the file stem without the
#: ``_wNE...`` suffix and with a leading ``ssp_`` read as ``fsps_``
#: (:func:`canonical_grid_name`). PENDING rows name a table that has not been
#: built (BPASS is the only PENDING row; see its row below).
MASS_REMAINING_REGISTRY: dict[str, MassRemainingEntry] = {
    # MIST isochrones: companion tables built with python-fsps.
    "fsps_mist_miles_chabrier": _m("mist", "chabrier"),
    "fsps_mist_c3k_a_chabrier": _m("mist", "chabrier"),
    "fsps_mist_basel_chabrier": _m("mist", "chabrier"),
    "fsps_mist_miles_kroupa": _m("mist", "kroupa"),
    "fsps_mist_c3k_a_kroupa": _m("mist", "kroupa"),
    "fsps_mist_basel_kroupa": _m("mist", "kroupa"),
    "fsps_mist_miles_salpeter": _m("mist", "salpeter"),
    "fsps_mist_c3k_a_salpeter": _m("mist", "salpeter"),
    "fsps_mist_basel_salpeter": _m("mist", "salpeter"),
    # PARSEC (FSPS-built, the local build whose spectra the hosted grid reproduces).
    "fsps_prsc_miles_chabrier": _m("prsc", "chabrier"),
    "fsps_prsc_c3k_a_chabrier": _m("prsc", "chabrier"),
    "fsps_prsc_basel_chabrier": _m("prsc", "chabrier"),
    "fsps_prsc_miles_kroupa": _m("prsc", "kroupa"),
    "fsps_prsc_c3k_a_kroupa": _m("prsc", "kroupa"),
    "fsps_prsc_basel_kroupa": _m("prsc", "kroupa"),
    "fsps_prsc_miles_salpeter": _m("prsc", "salpeter"),
    "fsps_prsc_c3k_a_salpeter": _m("prsc", "salpeter"),
    "fsps_prsc_basel_salpeter": _m("prsc", "salpeter"),
    # Padova (Padova 2007 set, FSPS-built): companion tables from python-fsps.
    "fsps_pdva_miles_chabrier": _m("pdva", "chabrier"),
    "fsps_pdva_c3k_a_chabrier": _m("pdva", "chabrier"),
    "fsps_pdva_basel_chabrier": _m("pdva", "chabrier"),
    "fsps_pdva_miles_kroupa": _m("pdva", "kroupa"),
    "fsps_pdva_c3k_a_kroupa": _m("pdva", "kroupa"),
    "fsps_pdva_basel_kroupa": _m("pdva", "kroupa"),
    "fsps_pdva_miles_salpeter": _m("pdva", "salpeter"),
    "fsps_pdva_basel_salpeter": _m("pdva", "salpeter"),
    # BaSTI (FSPS-built): companion tables from python-fsps.
    "fsps_bsti_miles_chabrier": _m("bsti", "chabrier"),
    "fsps_bsti_c3k_a_chabrier": _m("bsti", "chabrier"),
    "fsps_bsti_basel_chabrier": _m("bsti", "chabrier"),
    "fsps_bsti_miles_kroupa": _m("bsti", "kroupa"),
    "fsps_bsti_c3k_a_kroupa": _m("bsti", "kroupa"),
    "fsps_bsti_basel_kroupa": _m("bsti", "kroupa"),
    "fsps_bsti_miles_salpeter": _m("bsti", "salpeter"),
    "fsps_bsti_c3k_a_salpeter": _m("bsti", "salpeter"),
    "fsps_bsti_basel_salpeter": _m("bsti", "salpeter"),
    # BC03 (Padova 1994, STELIB): companion read from the BC03 *.4color files.
    "bc03_pdva_stelib_chabrier": _m("bc03pdva94", "chabrier"),
    # ProGeny (MIST, C3K, Chabrier): ProGeny's own SMstar, repackaged unaltered.
    "pgny_mist_c3k_chabrier": _m("pgny_mist", "chabrier"),
    # BPASS: PENDING. The BPASS starmass column is living stars only (manual v2.2),
    # and its remnant column is flagged untested by the BPASS team; no single-star
    # remnant recipe applies to a binary population. Not built; see PROVENANCE.md.
    "bpss_stars_c3k_a_chabrier": _p("bpss", "chabrier"),
    # Test placeholder grid (``synthetic`` attribute): hand-made table in the file.
    "fsps_prsc_bc03_chabrier": MassRemainingEntry("synthetic", "chabrier", EMBEDDED),
}


def canonical_grid_name(stem: str) -> str:
    """Map an SSP file stem to its registry key.

    Drops the ``_wNE...`` nebular suffix (nebular emission does not change the
    mass) and reads a leading ``ssp_`` (locally post-processed FSPS grids) as
    ``fsps_``.

    Parameters
    ----------
    stem : str
        File name without ``.h5``.

    Returns
    -------
    str
        Registry key.
    """
    name = stem.split("_wNE")[0]
    if name.startswith("ssp_"):
        name = "fsps_" + name[len("ssp_") :]
    return name


class MassRemainingResolution(NamedTuple):
    """Outcome of :func:`resolve_mass_remaining`.

    ``table`` is ``None`` exactly when ``source == "dsps_fit"``: the caller
    evaluates the fit with ``fit_imf``.
    """

    table: np.ndarray | None
    source: str
    fit_imf: str | None


class _Companion(NamedTuple):
    log_age_yr: np.ndarray
    log_z_abs: np.ndarray
    table: np.ndarray


@cache
def load_companion_table(filename: str, isoc: str, imf: str) -> _Companion:
    """Read a companion table from package data and validate it.

    Parameters
    ----------
    filename : str
        File in ``tengri.data.ssp_mass_remaining``.
    isoc, imf : str
        What the registry says the file holds; the file's ``isochrones`` and
        ``imf`` attributes must agree.

    Returns
    -------
    _Companion
        Read-only ``log10_age_yr`` (n_age,) [log10 yr], ``log10_z_abs``
        (n_met,) [log10 absolute Z] and ``mass_remaining`` (n_met, n_age).

    Raises
    ------
    FileNotFoundError, OSError
        The file is missing or unreadable (never swallowed).
    ValueError
        The attributes disagree with the registry or the datasets are malformed.
    """
    resource = files(_PACKAGE) / filename
    with as_file(resource) as path, h5py.File(path, "r") as f:
        got = (_attr_str(f, "isochrones"), _attr_str(f, "imf"))
        age = np.array(f["log10_age_yr"][:], dtype=np.float64)
        logz = np.array(f["log10_z_abs"][:], dtype=np.float64)
        table = np.array(f["mass_remaining"][:], dtype=np.float64)
    iso_prefix = _ISOC_ATTR_PREFIX.get(isoc, isoc)
    if not got[0].startswith(iso_prefix) or got[1].split()[0] != imf:
        raise ValueError(
            f"{filename}: attributes (isochrones, imf) = {got} but the registry "
            f"expects ({isoc!r}, {imf!r})"
        )
    if table.shape != (logz.shape[0], age.shape[0]):
        raise ValueError(f"{filename}: mass_remaining shape {table.shape} != (n_met, n_age)")
    if not (np.all(np.diff(age) > 0.0) and np.all(np.diff(logz) > 0.0)):
        raise ValueError(f"{filename}: axes are not strictly increasing")
    if not (np.all(np.isfinite(table)) and np.all(table > 0.0)):
        raise ValueError(f"{filename}: non-finite or non-positive surviving mass")
    for arr in (age, logz, table):
        arr.flags.writeable = False
    return _Companion(age, logz, table)


def _attr_str(f: h5py.File, key: str) -> str:
    value = f.attrs[key]
    return (value.decode() if isinstance(value, bytes) else str(value)).strip().lower()


def _grid_ages_to_table(
    table_age: np.ndarray, table: np.ndarray, lg_age_yr: np.ndarray, label: str
) -> np.ndarray:
    """Evaluate a companion table at a grid's age nodes.

    Parameters
    ----------
    table_age : ndarray, shape (n_tab,)
        Table ages [log10 yr].
    table : ndarray, shape (n_met, n_tab)
        Table values [dimensionless].
    lg_age_yr : ndarray, shape (n_age,)
        Grid ages [log10 yr]; ``-inf`` marks a t = 0 node.
    label : str
        Table name for messages.

    Returns
    -------
    ndarray, shape (n_met, n_age)
        Surviving fraction at the grid's ages.

    Raises
    ------
    ValueError
        An age is NaN or ``+inf``, or lies beyond the table range by more than
        one table node.

    Notes
    -----
    * A t = 0 node (``-inf``) is 1.0 by definition: no star has evolved.
    * A grid age within ``AGE_NODE_ATOL`` of a table node takes that node's value
      with no interpolation (grids and tables round log ages differently).
    * Otherwise linear interpolation in log10 age. A grid age beyond an end of
      the table by at most one table node (the spacing of the end interval) is
      clamped to the edge value; further out raises.
    """
    age = np.asarray(lg_age_yr, dtype=np.float64)
    if np.any(np.isnan(age)) or np.any(age == np.inf):
        raise ValueError(f"{label}: grid ages contain NaN or +inf")
    zero = np.isneginf(age)
    finite = np.where(zero, table_age[0], age)

    lo_node = table_age[1] - table_age[0]
    hi_node = table_age[-1] - table_age[-2]
    below = table_age[0] - finite - lo_node
    above = finite - table_age[-1] - hi_node
    bad = (below > AGE_NODE_ATOL) | (above > AGE_NODE_ATOL)
    if np.any(bad):
        raise ValueError(
            f"{label}: grid ages {np.unique(finite[bad]).tolist()} [log10 yr] lie more than "
            f"one table node beyond the table range [{table_age[0]}, {table_age[-1]}]"
        )

    nearest = np.abs(finite[:, None] - table_age[None, :]).argmin(axis=1)
    on_node = np.abs(finite - table_age[nearest]) <= AGE_NODE_ATOL
    out = np.empty((table.shape[0], age.shape[0]), dtype=np.float64)
    for i_z in range(table.shape[0]):
        interp = np.interp(finite, table_age, table[i_z])
        out[i_z] = np.where(on_node, table[i_z, nearest], interp)
    out[:, zero] = 1.0
    return out


def _companion_on_grid(
    entry: MassRemainingEntry, lg_age_yr: np.ndarray, lgmet: np.ndarray
) -> np.ndarray:
    """Companion table of ``entry`` evaluated on a grid; raises on a Z mismatch."""
    comp = load_companion_table(entry.source, entry.isoc, entry.imf)
    z = np.asarray(lgmet, dtype=np.float64)
    if z.shape != comp.log_z_abs.shape or not np.allclose(
        z, comp.log_z_abs, rtol=0.0, atol=LOGZ_ATOL
    ):
        raise ValueError(
            f"{entry.source}: the grid's metallicity nodes do not match the table's "
            f"to {LOGZ_ATOL} dex.\n  grid  log10(Z): {z.tolist()}\n  table log10(Z): "
            f"{comp.log_z_abs.tolist()}\nThe table belongs to a different isochrone/Z set."
        )
    return _grid_ages_to_table(comp.log_age_yr, comp.table, lg_age_yr, entry.source)


def resolve_mass_remaining(
    *,
    stem: str,
    detected_imf: str,
    lg_age_gyr: np.ndarray,
    lgmet: np.ndarray,
    embedded: np.ndarray | None,
    has_alpha_axis: bool,
    synthetic: bool = False,
    mode: str,
) -> MassRemainingResolution:
    """Resolve ``ssp_mass_remaining`` for one SSP grid, or refuse.

    Resolution order, with ``mode="table"`` (the default):

    1. A registered grid with a companion table (registry ``source`` is a file,
       not PENDING or EMBEDDED, and the grid has no [alpha/Fe] axis): the
       registry's companion table, built from the grid's own isochrones. Any
       ``embedded`` table in the file is ignored and not cross-checked; the
       companion is authoritative. Metallicity nodes must match the grid's to
       ``LOGZ_ATOL`` in log Z, and ages are placed as in
       :func:`_grid_ages_to_table`.
    2. Otherwise the grid's own embedded table, if it has one. This covers an
       unregistered (user) grid, a registered grid whose table is still PENDING
       or EMBEDDED, and an [alpha/Fe] grid.
    3. No table: a registered grid (shipped, PENDING) raises ``ValueError``
       naming the missing table and the opt-in; an unregistered (user) grid
       emits a ``UserWarning`` naming the opt-in and uses the DSPS fit.

    ``mode="dsps_fit"`` skips the tables and uses the fit for any grid.

    Parameters
    ----------
    stem : str
        File stem of the grid.
    detected_imf : str
        IMF from the grid's attribute or file name (used for the fit of an
        unregistered grid).
    lg_age_gyr : array_like, shape (n_age,)
        Grid ages [log10 Gyr].
    lgmet : array_like, shape (n_met,)
        Grid metallicities [log10 absolute Z].
    embedded : ndarray, shape (n_met, n_age) or None
        The grid file's own table.
    has_alpha_axis : bool
        The grid has an [alpha/Fe] axis; the (n_met, n_age) companion does not apply.
    synthetic : bool
        The file declares ``synthetic=True`` (a test fixture, not a physical grid,
        possibly written under a catalog name): it is not looked up in the registry,
        so its own table is used as is.
    mode : {"table", "dsps_fit"}
        The loader's ``mass_remaining`` argument.

    Returns
    -------
    MassRemainingResolution
        ``table`` (n_met, n_age) and ``source`` (``"embedded"``,
        ``"companion:<file>"`` or ``"dsps_fit"``).

    Raises
    ------
    ValueError
        Unknown mode; PENDING or table-less registered grid without the opt-in;
        metallicity mismatch; ages out of range; grid attribute contradicting
        the registry.
    OSError
        A companion file cannot be read.

    Notes
    -----
    Not JIT-able (file I/O); runs once at load. See [1]_, [2]_ for the
    definition and FSPS's remnants, [3]_ for BC03's.
    """
    if mode not in MASS_REMAINING_MODES:
        raise ValueError(f"mass_remaining must be one of {MASS_REMAINING_MODES}, got {mode!r}")
    entry = None if synthetic else MASS_REMAINING_REGISTRY.get(canonical_grid_name(stem))
    imf_norm = (detected_imf.lower().split() or ["unknown"])[0]
    if entry is not None and imf_norm not in ("unknown", entry.imf):
        raise ValueError(
            f"{stem}: the grid declares IMF {detected_imf!r} but the registry has "
            f"{canonical_grid_name(stem)!r} on {entry.imf!r}"
        )
    fit_imf = entry.imf if entry is not None else detected_imf
    if mode == MODE_DSPS_FIT:
        return MassRemainingResolution(None, SOURCE_DSPS_FIT, fit_imf)

    lg_age_yr = np.asarray(lg_age_gyr, dtype=np.float64) + 9.0
    has_companion = (
        entry is not None and entry.source not in (PENDING, EMBEDDED) and not has_alpha_axis
    )
    if has_companion:
        table = _companion_on_grid(entry, lg_age_yr, lgmet)
        return MassRemainingResolution(table, f"companion:{entry.source}", fit_imf)

    if embedded is not None:
        return MassRemainingResolution(np.asarray(embedded, dtype=np.float64), EMBEDDED, fit_imf)

    optin = f"load_ssp_data(..., mass_remaining={MODE_DSPS_FIT!r})"
    if entry is None:
        warnings.warn(
            f"'{stem}' is not a registered SSP grid and carries no ssp_mass_remaining: "
            "using DSPS's metallicity-independent sigmoid fit to FSPS for the surviving "
            f"mass. Pass {optin} to state this explicitly, or add an "
            "'ssp_mass_remaining' dataset built from the grid's own isochrones.",
            UserWarning,
            stacklevel=3,
        )
        return MassRemainingResolution(None, SOURCE_DSPS_FIT, fit_imf)
    missing = (
        f"mass_remaining_{entry.isoc}_{entry.imf}.h5"
        if entry.source in (PENDING, EMBEDDED)
        else entry.source
    )
    why = (
        "has an [alpha/Fe] axis, which the (n_met, n_age) table cannot serve"
        if has_alpha_axis
        else "has no surviving-mass table yet"
    )
    raise ValueError(
        f"'{stem}' {why} (missing {missing}, isochrones {entry.isoc!r}, IMF {entry.imf!r}). "
        "Its surviving mass would otherwise come from DSPS's metallicity-independent fit, "
        f"which ignores the grid's isochrones. Pass {optin} to accept the fit."
    )


def describe_source(stem: str) -> str:
    """One-line registry status of a grid, for :func:`tengri.doctor`.

    Parameters
    ----------
    stem : str
        File stem.

    Returns
    -------
    str
        ``"companion:<file>"``, ``"embedded"``, ``"PENDING (needs mass_remaining='dsps_fit')"``
        or ``"unregistered (DSPS fit with a warning)"``.
    """
    entry = MASS_REMAINING_REGISTRY.get(canonical_grid_name(stem))
    if entry is None:
        return "unregistered (DSPS fit with a warning)"
    if entry.source == PENDING:
        return f"PENDING (needs mass_remaining={MODE_DSPS_FIT!r})"
    if entry.source == EMBEDDED:
        return EMBEDDED
    return f"companion:{entry.source}"
