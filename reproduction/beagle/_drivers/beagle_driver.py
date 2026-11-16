# SPDX-License-Identifier: BSD-3-Clause
"""BEAGLE public-ingredient layer: repackaged public tables and analytic references.

BEAGLE (Chevallard & Charlot 2016 [5]_) is a binary fitter that ships its own nebular grid and its
own templates. This module does not drive that binary and does not read BEAGLE templates. It
reads only public inputs that are committed to the repository or supplied by the user:

* the Gutkin+2016 constant-SFR nebular tables (``data/neogal/nebular_emission_Z*.txt``) and the
  Feltre+2016 narrow-line-region tables (``data/neogal/AGN_NLR_nebular_feltre16/``), both
  repackaged from the NEOGAL distribution;
* the CANDELS GOODS-S workshop photometry and filter curves used by Pacifici+2023
  (``data/beagle_workshop_catalog.h5``, built by ``scripts/build_beagle_workshop_catalog.py``);
* the BEAGLE summary catalogs of that workshop, read at run time from a user-supplied
  checkout (``$ART_SEDFITTING_DIR``) and never committed.

It also implements three analytic references that BEAGLE's own prescriptions define: the
Charlot & Fall (2000 [3]_) two-component attenuation, the Chevallard et al. (2013 [4]_)
universal slope,
and the BEAGLE current-component star-formation relation.

Units and conventions
---------------------
Wavelengths are Angstrom in vacuum. The NEOGAL labels at or above 2000 Angstrom are air
wavelengths and are converted once with :func:`tengri.utils.air_vacuum.air_to_vac`; labels
below 2000 Angstrom are vacuum values. Gutkin luminosities are L_sun per (M_sun/yr). Feltre
luminosities are used as tabulated [erg/s].

Metallicity. BEAGLE's log Z/Z_sun uses Z_sun = 0.01524, tengri's uses Z_sun = 0.0142. To express
a BEAGLE log Z/Z_sun in tengri's convention (``met_logzsol``, ``neb_logZ_gas``), add
:data:`LOG10_ZSUN_SHIFT` = log10(0.01524 / 0.0142).

References
----------
.. [1] Gutkin, J., Charlot, S., & Bruzual, G. 2016, "Modelling the nebular emission from
   primeval to present-day star-forming galaxies," MNRAS, 462, 1757,
   doi:10.1093/mnras/stw1716, arXiv:1607.06086.
.. [2] Feltre, A., Charlot, S., & Gutkin, J. 2016, "Nuclear activity versus star formation:
   emission-line diagnostics at ultraviolet and optical wavelengths," MNRAS, 456, 3354,
   doi:10.1093/mnras/stv2794, arXiv:1511.08217.
.. [3] Charlot, S., & Fall, S. M. 2000, "A Simple Model for the Absorption of Starlight by
   Dust in Galaxies," ApJ, 539, 718, doi:10.1086/309250.
.. [4] Chevallard, J., Charlot, S., Wandelt, B., & Wild, V. 2013, "Insights into the content
   and spatial distribution of dust from the integrated spectral properties of galaxies,"
   MNRAS, 432, 2061, doi:10.1093/mnras/stt523, arXiv:1303.6631.
.. [5] Chevallard, J., & Charlot, S. 2016, "Modelling and interpreting spectral energy
   distributions of galaxies with BEAGLE," MNRAS, 462, 1415, doi:10.1093/mnras/stw1756,
   arXiv:1603.03037.
.. [6] Pacifici, C., et al. 2023, "The Art of Measuring Physical Parameters in Galaxies: A
   Critical Assessment of Spectral Energy Distribution Fitting Techniques," ApJ, 944, 141,
   doi:10.3847/1538-4357/acacff, arXiv:2212.01915.
.. [7] Inoue, A. K., Shimizu, I., Iwata, I., & Tanaka, M. 2014, "An updated analytic model for
   attenuation by the intergalactic medium," MNRAS, 442, 1805, doi:10.1093/mnras/stu936.
"""

from __future__ import annotations

import functools
import json
import math
import os
import re
import warnings
from pathlib import Path

import numpy as np

from tengri.utils.air_vacuum import air_to_vac
from tengri.utils.cosmology import PLANCK18
from tengri.utils.physics_constants import AB_ZEROPOINT, Z_SUN

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
GUTKIN_DIR = DATA_DIR / "neogal"
FELTRE_DIR = DATA_DIR / "neogal" / "AGN_NLR_nebular_feltre16"
WORKSHOP_PATH = DATA_DIR / "beagle_workshop_catalog.h5"
ART_ENV = "ART_SEDFITTING_DIR"
BUILDER = "scripts/build_beagle_workshop_catalog.py"

# BEAGLE's constants. Solar luminosity and mass are BEAGLE's own values, not tengri's.
L_SUN_BEAGLE = 3.826e33  # [erg/s]
M_SUN_BEAGLE = 1.9891e33  # [g]
Z_SUN_BEAGLE = 0.01524
LOG10_ZSUN_SHIFT = math.log10(Z_SUN_BEAGLE / Z_SUN)  # add to BEAGLE log Z/Z_sun
BEAGLE_H0 = 67.74  # [km/s/Mpc], Planck 2015
BEAGLE_OMEGA_M = 0.3089
BEAGLE_OMEGA_L = 0.6911
# tengri's Planck 2018 cosmology, read from the shipped parameter set.
TENGRI_H0 = 100.0 * PLANCK18.h  # [km/s/Mpc]
TENGRI_OMEGA_M = PLANCK18.Om0

# Air wavelengths [Angstrom] of the NEOGAL labels at or above 2000 Angstrom (NIST values).
_AIR_NIST = {
    "[OII]3727": 3726.032,
    "Hb": 4861.33,
    "Hbeta": 4861.33,
    "[OIII]4959": 4958.91,
    "[OIII]5007": 5006.84,
    "[OI]6300": 6300.30,
    "[NII]6548": 6548.05,
    "Ha": 6562.80,
    "Halpha": 6562.80,
    "[NII]6584": 6583.45,
    "[SII]6717": 6716.44,
    "[SII]6731": 6730.82,
}
_PHOT_SENTINEL = 90.0  # |mag| or |err| beyond this is a sentinel (98.9x or -99)
_ZSUN_TABLE_NOTE = "the file name digits after '0.' give Z: Z0001 is 0.0001, Z040 is 0.040"
_Z_NAME = re.compile(r"_Z(\d+)\.txt$")
_LOG_TO_MAG = 0.4 * math.log(10.0)  # d(flux)/flux = ln(10) * 0.4 * d(mag)
# Wavelength window [Angstrom] over which Chevallard+13 is calibrated (0.1 to 2.5 micron).
_C13_WINDOW_AA = (1000.0, 25000.0)


def _z_from_name(path: Path) -> float:
    """Absolute metallicity from a NEOGAL file name (see ``_ZSUN_TABLE_NOTE``)."""
    match = _Z_NAME.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse Z from {path.name}; {_ZSUN_TABLE_NOTE}")
    return float("0." + match.group(1))


def _header_names(path: Path) -> tuple[str, ...]:
    """Column names of a NEOGAL table, from its ``# log(Us) ...`` header line."""
    with open(path) as handle:
        for line in handle:
            if line.startswith("#") and "log(Us)" in line:
                return tuple(line.lstrip("#").split())
    raise ValueError(f"{path} has no '# log(Us) ...' header line")


def _vacuum_wavelength(name: str) -> float:
    """Vacuum wavelength [Angstrom] of a NEOGAL line label.

    Air labels (NIST table ``_AIR_NIST``) are converted with ``air_to_vac``. Other labels carry
    their nominal wavelength in the name and are vacuum values.
    """
    if name in _AIR_NIST:
        return float(air_to_vac(np.asarray(_AIR_NIST[name])))
    digits = re.search(r"(\d{4})", name)
    if digits is None:
        raise ValueError(f"no wavelength in NEOGAL line label {name!r}")
    return float(digits.group(1))


def _line_table(names: tuple[str, ...], n_params: int) -> tuple[tuple[str, float], ...]:
    """(label, vacuum Angstrom) for the line columns that follow the parameter columns."""
    return tuple((name, _vacuum_wavelength(name)) for name in names[n_params:])


@functools.cache
def _gutkin_files() -> tuple[tuple[float, Path], ...]:
    """(Z, path) for every Gutkin constant-SFR table, ascending in Z."""
    paths = sorted(GUTKIN_DIR.glob("nebular_emission_Z*.txt"), key=_z_from_name)
    if not paths:
        raise FileNotFoundError(f"no Gutkin+2016 tables in {GUTKIN_DIR}")
    return tuple((_z_from_name(p), p) for p in paths)


@functools.cache
def _gutkin_rows(path: Path) -> np.ndarray:
    """Numeric rows of one Gutkin table, shape (n_rows, 23)."""
    rows = []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                rows.append([float(tok) for tok in stripped.split()])
    return np.asarray(rows, dtype=np.float64)


GUTKIN_PARAMS = ("logU_S", "xi_d", "n_H", "C/O", "m_up")
GUTKIN_LINES = _line_table(_header_names(_gutkin_files()[0][1]), len(GUTKIN_PARAMS))


def _gutkin_path(z_abs: float) -> Path:
    """Path of the Gutkin table at metallicity ``z_abs``."""
    for z, path in _gutkin_files():
        if math.isclose(z, z_abs, rel_tol=1e-6):
            return path
    raise KeyError(f"no Gutkin+2016 table at Z = {z_abs}")


def gutkin_axes() -> dict[str, np.ndarray]:
    """Sorted unique grid axes of the Gutkin+2016 tables.

    Returns
    -------
    dict
        ``logZ_abs`` [dex], ``logU`` (ionization parameter), ``xi_d`` (dust-to-metal ratio),
        ``n_H`` [cm^-3], ``C/O`` ((C/O)/(C/O)_sun) and ``m_up`` [M_sun]; each is a 1-D array.
    """
    rows = [_gutkin_rows(path) for _, path in _gutkin_files()]
    stacked = np.vstack(rows)
    return {
        "logZ_abs": np.log10(np.array([z for z, _ in _gutkin_files()])),
        "logU": np.unique(stacked[:, 0]),
        "xi_d": np.unique(stacked[:, 1]),
        "n_H": np.unique(stacked[:, 2]),
        "C/O": np.unique(stacked[:, 3]),
        "m_up": np.unique(stacked[:, 4]),
    }


def gutkin_table(z_abs: float) -> dict[str, np.ndarray]:
    """All rows of the Gutkin table at one metallicity, keyed by column name.

    Parameters
    ----------
    z_abs : float
        Absolute metallicity Z, one of the table values (for example 0.001).

    Returns
    -------
    dict
        Column name -> ndarray, shape (n_rows,). Parameter columns use ``GUTKIN_PARAMS``
        names; line columns use the NEOGAL labels. Line values are L_sun per (M_sun/yr).
    """
    path = _gutkin_path(z_abs)
    data = _gutkin_rows(path)
    names = GUTKIN_PARAMS + tuple(name for name, _ in GUTKIN_LINES)
    return {name: data[:, j] for j, name in enumerate(names)}


def gutkin_node(
    z_abs: float, log_u: float, xi_d: float, n_h: float, co: float, mup: float
) -> dict[str, float]:
    """Line luminosities at one tabulated Gutkin node.

    Parameters
    ----------
    z_abs : float
        Absolute metallicity Z.
    log_u : float
        log10 of the ionization parameter at the Stromgren boundary, ``logU_S``.
    xi_d : float
        Dust-to-metal ratio.
    n_h : float
        Hydrogen density [cm^-3].
    co : float
        (C/O)/(C/O)_sun.
    mup : float
        Upper mass cutoff [M_sun].

    Returns
    -------
    dict
        Line label -> luminosity [L_sun per (M_sun/yr)].

    Raises
    ------
    KeyError
        If the node is absent from the public table. Nineteen nodes are absent; see
        ``scripts/build_gutkin16_grid.py`` for how the committed grid fills them.
    """
    table = gutkin_table(z_abs)
    keep = np.ones(len(table["logU_S"]), dtype=bool)
    for key, value in zip(GUTKIN_PARAMS, (log_u, xi_d, n_h, co, mup), strict=True):
        keep &= np.isclose(table[key], value, rtol=0.0, atol=1e-6)
    idx = np.flatnonzero(keep)
    if idx.size != 1:
        raise KeyError(
            f"node Z={z_abs}, logU={log_u}, xi={xi_d}, nH={n_h}, C/O={co}, mup={mup}"
            " is absent from the public Gutkin+2016 table"
        )
    return {name: float(table[name][idx[0]]) for name, _ in GUTKIN_LINES}


def gutkin_ratios_to_hbeta(node: dict[str, float]) -> dict[str, float]:
    """Line-to-Hbeta ratios of one node returned by :func:`gutkin_node`."""
    hb = node["Hb"]
    return {name: value / hb for name, value in node.items()}


@functools.cache
def _feltre_files() -> tuple[tuple[float, Path], ...]:
    """(Z, path) for every Feltre+2016 NLR table, ascending in Z."""
    paths = sorted(FELTRE_DIR.glob("nlr_nebular_Z*.txt"), key=_z_from_name)
    if not paths:
        raise FileNotFoundError(f"no Feltre+2016 tables in {FELTRE_DIR}")
    return tuple((_z_from_name(p), p) for p in paths)


@functools.cache
def _feltre_rows(path: Path) -> np.ndarray:
    rows = []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                rows.append([float(tok) for tok in stripped.split()])
    return np.asarray(rows, dtype=np.float64)


FELTRE_PARAMS = ("logU_S", "xi_d", "n_H", "alpha")
FELTRE_LINES = _line_table(_header_names(_feltre_files()[0][1]), len(FELTRE_PARAMS))


def feltre_axes() -> dict[str, np.ndarray]:
    """Sorted unique grid axes of the Feltre+2016 NLR tables.

    Returns
    -------
    dict
        ``logZ_abs`` [dex], ``logU``, ``xi_d``, ``n_H`` [cm^-3] and ``alpha`` (the power-law
        index of the ionizing continuum); each is a 1-D array.
    """
    stacked = np.vstack([_feltre_rows(path) for _, path in _feltre_files()])
    return {
        "logZ_abs": np.log10(np.array([z for z, _ in _feltre_files()])),
        "logU": np.unique(stacked[:, 0]),
        "xi_d": np.unique(stacked[:, 1]),
        "n_H": np.unique(stacked[:, 2]),
        "alpha": np.unique(stacked[:, 3]),
    }


def feltre_node(
    z_abs: float, log_u: float, xi_d: float, n_h: float, alpha: float
) -> dict[str, float]:
    """Line luminosities [erg/s] at one tabulated Feltre+2016 node.

    Raises
    ------
    KeyError
        If the node is not in the table at metallicity ``z_abs``.
    """
    path = next((p for z, p in _feltre_files() if math.isclose(z, z_abs, rel_tol=1e-6)), None)
    if path is None:
        raise KeyError(f"no Feltre+2016 table at Z = {z_abs}")
    data = _feltre_rows(path)
    keep = np.ones(len(data), dtype=bool)
    for j, value in enumerate((log_u, xi_d, n_h, alpha)):
        keep &= np.isclose(data[:, j], value, rtol=0.0, atol=1e-6)
    idx = np.flatnonzero(keep)
    if idx.size != 1:
        raise KeyError(
            f"Feltre node Z={z_abs}, logU={log_u}, xi={xi_d}, nH={n_h}, alpha={alpha}"
            " is absent from the public table"
        )
    return {
        name: float(data[idx[0], len(FELTRE_PARAMS) + k])
        for k, (name, _) in enumerate(FELTRE_LINES)
    }


def cf00_tau(wave_aa, tau_v: float, mu: float, young: bool = True) -> np.ndarray:
    """Charlot & Fall (2000) optical depth [dimensionless] at vacuum wavelengths.

    Implements :math:`\\tau_{\\rm BC}(\\lambda)=(1-\\mu)\\tau_V(\\lambda/5500\\,{\\rm\\AA})^{-1.3}`
    for young stars (birth clouds, present only when ``young`` is True) and
    :math:`\\tau_{\\rm ISM}(\\lambda)=\\mu\\tau_V(\\lambda/5500\\,{\\rm\\AA})^{-0.7}` for all
    stars. Here :math:`\\tau_V` is the V-band optical depth [dimensionless] and :math:`\\mu` the
    fraction of :math:`\\tau_V` in the diffuse ISM [dimensionless].

    Parameters
    ----------
    wave_aa : array_like, shape (n_wave,)
        Vacuum wavelength [Angstrom].
    tau_v : float
        V-band optical depth [dimensionless].
    mu : float
        Diffuse-ISM fraction [dimensionless].
    young : bool, optional
        Include the birth-cloud term. Default True.

    Returns
    -------
    ndarray, shape (n_wave,)
        Optical depth [dimensionless].
    """
    scaled = np.asarray(wave_aa, dtype=np.float64) / 5500.0
    tau = mu * tau_v * scaled**-0.7
    if young:
        tau = tau + (1.0 - mu) * tau_v * scaled**-1.3
    return tau


def c13_tau(wave_aa, tau_v_ism: float) -> np.ndarray:
    """Chevallard et al. (2013) universal attenuation optical depth [dimensionless].

    Implements the slope :math:`n(\\lambda)=n_V+b(\\lambda/\\mu{\\rm m}-0.55)` with
    :math:`n_V=2.8/(1+3\\sqrt{\\tau_V})` and :math:`b=0.3-0.05\\tau_V`, and the optical depth
    :math:`\\tau(\\lambda)=\\tau_V(\\lambda/5500\\,{\\rm\\AA})^{-n(\\lambda)}`. Here
    :math:`\\tau_V` is the V-band optical depth [dimensionless]. The slope at V is
    :math:`-n_V`.

    Approximation of the fitted universal relation in Chevallard+13 [4]_, calibrated for
    0.1 to 2.5 micron; outside that window a ``UserWarning`` is issued.

    Parameters
    ----------
    wave_aa : array_like, shape (n_wave,)
        Vacuum wavelength [Angstrom].
    tau_v_ism : float
        V-band optical depth of the diffuse ISM [dimensionless].

    Returns
    -------
    ndarray, shape (n_wave,)
        Optical depth [dimensionless].
    """
    wave = np.asarray(wave_aa, dtype=np.float64)
    if np.any((wave < _C13_WINDOW_AA[0]) | (wave > _C13_WINDOW_AA[1])):
        warnings.warn(
            "Chevallard+13 is calibrated for 0.1 to 2.5 micron; wavelengths outside that "
            "window are extrapolated",
            UserWarning,
            stacklevel=2,
        )
    n_v = 2.8 / (1.0 + 3.0 * math.sqrt(tau_v_ism))
    b = 0.3 - 0.05 * tau_v_ism
    slope = n_v + b * (wave / 1.0e4 - 0.55)
    return tau_v_ism * (wave / 5500.0) ** (-slope)


def delayed_sfr(t_yr, tau_yr: float, mass: float) -> np.ndarray:
    """Delayed-exponential star-formation rate [M_sun/yr] normalized to a stellar mass.

    Implements :math:`\\psi(t)=M\\,t\\,e^{-t/\\tau}/\\tau^2` for :math:`t\\ge 0`, which integrates
    to :math:`M` over all time. Here :math:`t` is the age [yr], :math:`\\tau` the e-folding time
    [yr] and :math:`M` the total formed mass [M_sun].

    Parameters
    ----------
    t_yr : array_like, shape (n_t,)
        Age [yr]. Values below zero give zero SFR.
    tau_yr : float
        e-folding time [yr].
    mass : float
        Total formed mass [M_sun].

    Returns
    -------
    ndarray, shape (n_t,)
        SFR [M_sun/yr].
    """
    t = np.asarray(t_yr, dtype=np.float64)
    psi = mass * t * np.exp(-t / tau_yr) / tau_yr**2
    return np.where(t >= 0.0, psi, 0.0)


def current_component(mass_smooth: float, log_ssfr: float, dt_yr: float) -> float:
    """Current-bin SFR [M_sun/yr] from the BEAGLE specific-SFR relation.

    BEAGLE (Chevallard & Charlot 2016 [5]_) parameterizes the recent SFR by the specific SFR of the
    current bin, :math:`\\psi_S=\\psi/[M(t_{\\rm start},\\Delta t)+\\psi\\Delta t]`, so that
    :math:`\\psi=\\psi_S M/(1-\\psi_S\\Delta t)`. Here :math:`M` is the smooth-part stellar mass
    formed before the bin [M_sun], :math:`\\Delta t` the bin duration [yr] and :math:`\\psi_S` the
    specific SFR [yr^-1].

    Parameters
    ----------
    mass_smooth : float
        Smooth-part stellar mass formed before the current bin [M_sun].
    log_ssfr : float
        log10 of the current-bin specific SFR [yr^-1].
    dt_yr : float
        Duration of the current bin [yr].

    Returns
    -------
    float
        Current-bin SFR [M_sun/yr].

    Raises
    ------
    ValueError
        If :math:`1-\\psi_S\\Delta t\\le 0`, where the relation has no finite solution.
    """
    ssfr = 10.0**log_ssfr
    denom = 1.0 - ssfr * dt_yr
    if denom <= 0.0:
        raise ValueError("specific SFR times bin duration must be below one")
    return ssfr * mass_smooth / denom


def inoue14_transmission(wave_obs_aa, z: float) -> np.ndarray:
    """Mean IGM transmission [dimensionless] at observed-frame wavelengths (Inoue+2014 [7]_).

    Thin wrapper over :func:`tengri.components.igm.igm.igm_transmission`.

    Parameters
    ----------
    wave_obs_aa : array_like, shape (n_wave,)
        Observed-frame wavelength [Angstrom].
    z : float
        Source redshift [dimensionless].

    Returns
    -------
    ndarray, shape (n_wave,)
        Transmitted fraction [dimensionless].
    """
    import jax.numpy as jnp

    from tengri.components.igm.igm import igm_transmission

    wave = jnp.asarray(np.asarray(wave_obs_aa, dtype=np.float64))
    return np.asarray(igm_transmission(wave, float(z)))


def available() -> bool:
    """True if the committed workshop catalog is present."""
    return WORKSHOP_PATH.is_file()


def require_available() -> None:
    """Raise with the rebuild command if the committed workshop catalog is missing."""
    if not available():
        raise FileNotFoundError(
            f"{WORKSHOP_PATH} is missing. It is committed to the repository; restore it, or "
            f"rebuild it with {BUILDER} --art-sedfitting <art_sedfitting checkout>."
        )


def workshop_catalog(z: int):
    """CANDELS GOODS-S workshop photometry at redshift ``z`` (1 or 3) as an astropy Table.

    The catalog is repackaged from Pacifici+2023 (see :data:`WORKSHOP_PATH`). Photometry columns
    are AB magnitudes and their errors; sentinels are kept and masked by
    :func:`photometry_for_row`. The band list and the band-to-curve map are in ``table.meta``.

    Parameters
    ----------
    z : int
        Catalog redshift, 1 or 3.

    Returns
    -------
    astropy.table.Table
        One row per galaxy. Columns are the source column names (``ID``, ``zz``, band
        magnitudes and their ``e``-prefixed errors, ``flg1``, ``flg2``, and ``RA``/``DEC`` in
        the z=3 catalog).

    Raises
    ------
    ValueError
        If ``z`` is not 1 or 3.
    """
    import h5py
    from astropy.table import Table

    if z not in (1, 3):
        raise ValueError(f"workshop catalogs exist for z = 1 and z = 3, not {z}")
    require_available()
    with h5py.File(WORKSHOP_PATH, "r") as f:
        group = f[f"catalog_z{z}"]
        columns = json.loads(group.attrs["columns"])
        meta = {
            "bands": json.loads(group.attrs["bands"]),
            "curve_for_band": json.loads(group.attrs["curve_for_band"]),
        }
        data = [np.asarray(group[name][:]) for name in columns]
    return Table(data, names=columns, meta=meta)


def photometry_for_row(row, bands: list[str] | None = None):
    """AB-band fluxes and errors of one workshop galaxy, masked where unmeasured.

    Parameters
    ----------
    row : astropy.table.Row
        One row of :func:`workshop_catalog`.
    bands : list of str, optional
        Band column stems. Default: ``row.table.meta["bands"]``.

    Returns
    -------
    names : list of str
        Band stems, in catalog order.
    flux : numpy.ma.MaskedArray, shape (n_band,)
        F_nu [erg/s/cm^2/Hz]. The AB zero point is ``AB_ZEROPOINT`` (-48.6), so
        F_nu = 10**(-0.4 (m_AB - AB_ZEROPOINT)).
    sigma : numpy.ma.MaskedArray, shape (n_band,)
        1-sigma flux error [erg/s/cm^2/Hz], propagated from the magnitude error.

    Notes
    -----
    A band is masked when its magnitude or error lies beyond +/-90, which covers the 98.9x
    no-measurement sentinel and the -99 no-error sentinel of the catalog.
    """
    names = list(bands) if bands is not None else list(row.table.meta["bands"])
    mags = np.array([float(row[name]) for name in names])
    errs = np.array([float(row["e" + name]) for name in names])
    masked = (np.abs(mags) > _PHOT_SENTINEL) | (np.abs(errs) > _PHOT_SENTINEL)
    flux = 10.0 ** (-0.4 * (mags - AB_ZEROPOINT))
    sigma = flux * _LOG_TO_MAG * errs
    return (
        names,
        np.ma.masked_array(flux, mask=masked),
        np.ma.masked_array(sigma, mask=masked),
    )


def paper1_ids() -> tuple[int, ...]:
    """The 20 GOODS-S galaxy IDs analyzed in Paper I of the BEAGLE reproduction."""
    return (
        79, 267, 4171, 15336, 25206, 25489, 26398, 7837, 9884, 13097,
        14099, 16455, 17376, 22173, 15775, 16049, 18160, 1826, 4056, 24786,
    )  # fmt: skip


def beagle_workshop_summary(z: int):
    """BEAGLE summary catalog of the workshop at redshift ``z``, from ``$ART_SEDFITTING_DIR``.

    These are BEAGLE outputs and are not committed. Export ``ART_SEDFITTING_DIR`` to a local
    checkout of camipacifici/art_sedfitting at commit 32d23a2.

    Parameters
    ----------
    z : int
        1 or 3.

    Returns
    -------
    astropy.table.Table
        The summary table of ``code_outputs/BEAGLE_summary_catalogue_z{z}.fits``.

    Raises
    ------
    RuntimeError
        If ``ART_SEDFITTING_DIR`` is unset.
    FileNotFoundError
        If the summary file is absent from that checkout.
    """
    import astropy.units as u
    from astropy.table import Table

    root = os.environ.get(ART_ENV)
    if not root:
        raise RuntimeError(
            f"{ART_ENV} is unset. Export it to a checkout of art_sedfitting (commit 32d23a2); "
            "the BEAGLE summary catalogs are read from its code_outputs/ directory and are "
            "not committed."
        )
    path = Path(root) / "code_outputs" / f"BEAGLE_summary_catalogue_z{z}.fits"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing from the {ART_ENV} checkout")
    with warnings.catch_warnings():
        # The summary file writes its flux unit 'M_sun', which the FITS standard does not parse.
        warnings.simplefilter("ignore", category=u.UnitsWarning)
        return Table.read(path, hdu=1)


def _selftest() -> None:
    """Print the first-row numbers of the Gutkin and Feltre tables and of the workshop catalog."""
    node = gutkin_node(0.001, -1.0, 0.1, 100.0, 1.40, 100.0)
    ratios = gutkin_ratios_to_hbeta(node)
    print(f"Gutkin Z=0.001 first row: Hb = {node['Hb']:.4e} L_sun/(M_sun/yr)")
    print(f"  [OIII]5007/Hb = {ratios['[OIII]5007']:.4f}")
    print(f"  Ha/Hb = {ratios['Ha']:.4f}")
    fnode = feltre_node(0.014, -2.5, 0.1, 100.0, -1.2)
    print(f"Feltre Z=0.014 first row: Hbeta = {fnode['Hbeta']:.4e} erg/s")
    print(f"  [OIII]5007/Hbeta = {fnode['[OIII]5007'] / fnode['Hbeta']:.4f}")
    print(f"workshop catalog present: {available()}")


def main() -> None:
    """Run the self-test (``--selftest``) or print the module summary."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--selftest", action="store_true", help="print reference node values")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
