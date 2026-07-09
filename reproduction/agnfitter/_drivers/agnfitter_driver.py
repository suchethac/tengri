"""Thin reader for AGNFITTER-RX's model template libraries.

This driver loads the disk, torus, and cold-dust template libraries that ship
inside AGNFITTER-RX (Martinez-Ramirez et al. 2024) and returns each template on
tengri's plotting convention -- ascending wavelength [Angstrom], L_nu
[erg/s/Hz]. It never runs AGNFITTER-RX's fitter; it reproduces the exact
template access in ``functions/MODEL_AGNfitter.py`` so the notebook overlays the
*same* SEDs the AGN fitter feeds its likelihood.

AGNFITTER-RX is a clone-and-load research code, not a pip package. Point the
driver at a checkout via the ``AGNFITTER_HOME`` environment variable (default
``/tmp/AGNfitter-rX``):

    git clone --depth 1 --branch AGNfitter-rX_v0.1 \\
        https://github.com/GabrielaCR/AGNfitter /tmp/AGNfitter-rX

Template files (``$AGNFITTER_HOME/models/``):

==========  ==================================================  ============================
Component   Library (driver name)                                File
==========  ==================================================  ============================
disk        ``R06``  (Richards et al. 2006)                      ``BBB/R06.pickle``
disk        ``SN12`` (Slone & Netzer 2012)                       ``BBB/SN12.pickle``
disk        ``KD18`` (Kubota & Done 2018)                        ``BBB/KD18.pickle``
disk        ``THB21`` (Temple, Hewett & Banerji 2021)            ``BBB/THB21.pickle``
torus       ``S04``  (Silva et al. 2004)                         ``TORUS/S04.pickle``
torus       ``NK08`` (Nenkova et al. 2008)                       ``TORUS/NK0_mean_1p.pickle``
torus       ``SKIRTOR`` (Stalevski et al. 2016)                  ``TORUS/SKIRTOR_mean_3p.pickle``
torus       ``CAT3D`` (Honig & Kishimoto 2017)                   ``TORUS/CAT3D_mean_3p.pickle``
cold dust   ``DH02_CE01`` (Dale & Helou 2002 + Chary & Elbaz 01) ``STARBURST/DH02_CE01.pickle``
cold dust   ``S17`` (Schreiber et al. 2018)                      ``STARBURST/s17_lowvsg_*.fits``
==========  ==================================================  ============================

Security
--------
The pickle libraries are untrusted external data. Loading uses a restricted
``pickle.Unpickler`` allow-listing only numpy/pandas/basic-builtin primitives,
plus a preflight opcode scan -- the same hardening as
``scripts/build_cat3d_wind_grid.py``.

References
----------
.. [1] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). doi:10.1051/0004-6361/202449329. arXiv:2405.12111.
"""

from __future__ import annotations

import importlib
import io
import os
import pickle
import pickletools
from functools import cache
from pathlib import Path

import numpy as np

from . import units

# ── Committed reference grids (always available; NO external clone needed) ──
# The disk/torus/cold-dust AGNfitter-RX reference templates are vendored as
# committed HDF5 under data/, so the reproduction runs in CI and on any clean
# checkout. The /tmp/AGNfitter-rX clone is needed ONLY to *regenerate* those
# grids (build-time), via scripts/build_agnfitter_bbb_reference.py — never at
# runtime. data/ lives at the repo root: <root>/reproduction/agnfitter/_drivers/.
_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_DISK_H5 = _DATA_DIR / "agnfitter_bbb_reference.h5"
_TORUS_H5 = _DATA_DIR / "agnfitter_torus_reference.h5"
_COLD_H5 = _DATA_DIR / "agnfitter_cold_dust_reference.h5"
_C_AA = 2.99792458e18  # speed of light [Å/s]

# Build-time regeneration source (optional; runtime never reads it).
AGNFITTER_HOME = Path(os.environ.get("AGNFITTER_HOME", "/tmp/AGNfitter-rX"))
_MODELS = AGNFITTER_HOME / "models"


@cache
def _ref(h5_path: Path, group: str) -> dict:
    """Load all datasets of one group from a committed reference h5 (cached)."""
    import h5py

    with h5py.File(h5_path, "r") as h:
        g = h[group]
        return {k: np.asarray(g[k]) for k in g}


def _h5_to_lnu(wavelength_aa, fnu, component: str):
    """(Å, F_nu) committed-h5 template -> (wave_aa, L_nu) in tengri units."""
    log_nu = np.log10(_C_AA / np.asarray(wavelength_aa, dtype=np.float64))
    return units.lognu_fnu_to_lnu(log_nu, _renorm(component, np.asarray(fnu, dtype=np.float64)))


def available() -> bool:
    """True if the committed reference grids are present (they ship in data/)."""
    return _DISK_H5.is_file() and _TORUS_H5.is_file() and _COLD_H5.is_file()


def require_available() -> None:
    """Raise a clear, actionable error if the committed grids are missing."""
    if not available():
        raise FileNotFoundError(
            "AGNfitter reference grids missing from data/ "
            "(agnfitter_bbb_reference.h5 / agnfitter_torus_reference.h5 / "
            "agnfitter_cold_dust_reference.h5). Regenerate with "
            "scripts/build_agnfitter_bbb_reference.py (needs an AGNfitter-rX clone)."
        )


# ── Restricted unpickler (mirrors scripts/build_cat3d_wind_grid.py) ─────────
_SAFE_CLASSES: frozenset[tuple[str, str]] = frozenset(
    {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.series", "Series"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.numeric", "Int64Index"),
        ("pandas.core.indexes.numeric", "Float64Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("pandas.core.internals.managers", "BlockManager"),
        ("pandas.core.internals.managers", "SingleBlockManager"),
        ("pandas._libs.internals", "_unpickle_block"),
        ("pandas.core.internals.blocks", "new_block"),
        ("builtins", "slice"),
        ("_codecs", "encode"),
        # functools.partial appears in newer-pandas DataFrame pickles (THB21)
        # as the block-reconstruction callable; benign under the preflight gate.
        ("functools", "partial"),
    }
)
_PY2_MODULE_ALIASES: dict[str, str] = {"__builtin__": "builtins"}


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler allow-listed to numpy / pandas / basic builtins."""

    def find_class(self, module: str, name: str):
        module = _PY2_MODULE_ALIASES.get(module, module)
        if (module, name) not in _SAFE_CLASSES:
            raise pickle.UnpicklingError(
                f"Refusing to import {module}.{name}: not in the safe allow-list. "
                "If this is a legitimate numpy/pandas primitive, add it to "
                "_SAFE_CLASSES in agnfitter_driver.py."
            )
        if (module, name) == ("pandas.core.internals.blocks", "new_block"):
            return _new_block_compat
        return getattr(importlib.import_module(module), name)


def _new_block_compat(values, placement, *args, **kwargs):
    """Version-tolerant ``pandas.new_block``.

    The THB21 disk pickle was written by an older pandas that passed a bare
    ``slice`` as the block placement; current pandas requires a
    ``BlockPlacement``. Coerce it so the legacy pickle still loads.
    """
    from pandas._libs.internals import BlockPlacement
    from pandas.core.internals.blocks import new_block as _nb

    if isinstance(placement, slice):
        placement = BlockPlacement(placement)
    return _nb(values, placement, *args, **kwargs)


def _preflight_opcode_scan(pickle_path: Path) -> None:
    """Abort if any GLOBAL reference falls outside the allow-list."""
    seen: set[tuple[str, str]] = set()
    with pickle_path.open("rb") as fh:
        out = io.StringIO()
        pickletools.dis(fh, annotate=0, out=out)
    for line in out.getvalue().splitlines():
        if "GLOBAL" not in line:
            continue
        try:
            qual = line.split("'", 1)[1].rsplit("'", 1)[0]
        except IndexError:
            continue
        parts = qual.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        mod, name = parts
        mod = _PY2_MODULE_ALIASES.get(mod, mod)
        seen.add((mod, name))
    unexpected = seen - _SAFE_CLASSES
    if unexpected:
        raise RuntimeError(
            f"Unexpected GLOBAL references in {pickle_path}: {sorted(unexpected)}. "
            "Refusing to proceed. Vet each entry, then add legitimate "
            "numpy/pandas primitives to _SAFE_CLASSES."
        )


@cache
def _safe_load(rel_path: str):
    """Load a model pickle under $AGNFITTER_HOME/models, cached by path."""
    require_available()
    path = _MODELS / rel_path
    if not path.is_file():
        raise FileNotFoundError(f"AGNFITTER-RX template not found: {path}")
    _preflight_opcode_scan(path)
    with path.open("rb") as fh:
        return _RestrictedUnpickler(fh, encoding="latin1").load()


def _renorm(component: str, fnu: np.ndarray) -> np.ndarray:
    """AGNFITTER-RX's per-component template renormalization (cosmetic).

    Reproduces ``MODEL_AGNfitter.renorm_template``. The notebook normalizes at
    physical anchors, so these constants only keep the driver byte-faithful.
    """
    factors = {"BB": 1e60, "SB": 1e20, "TO": 1e-40, "GA": 1e18}
    return np.asarray(fnu, dtype=np.float64) / factors[component]


# ── Disk (BBB) libraries ────────────────────────────────────────────────────
def list_disks() -> list[str]:
    """Names of the accretion-disk libraries this driver can load."""
    return ["R06", "SN12", "KD18", "THB21"]


def disk_axes(name: str) -> dict[str, np.ndarray]:
    """Grid axes for a parameterized disk library (empty for fixed templates).

    Parameters
    ----------
    name : {"R06", "SN12", "KD18", "THB21"}
        Disk library name.

    Returns
    -------
    dict
        ``{}`` for R06/THB21 (single template); for SN12/KD18 the
        ``log_mbh`` and ``log_edd`` grid axes.
    """
    name = name.upper()
    if name == "SN12":
        d = _ref(_DISK_H5, "sn12")
        return {
            "log_mbh": np.asarray(d["logBHmass"], dtype=np.float64).ravel(),
            "edd_index": np.arange(d["sed"].shape[1]),
        }
    if name == "KD18":
        d = _ref(_DISK_H5, "kd18")
        return {
            "log_mbh": np.sort(np.unique(d["logBHmass"])),
            "log_edd": np.sort(np.unique(d["logEddra"])),
        }
    return {}


def disk_template(
    name: str,
    *,
    log_mbh: float | None = None,
    log_edd: float | None = None,
    edd_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one accretion-disk template, unreddened, in tengri units.

    Reproduces the template access in ``MODEL_AGNfitter.BBB``. Reddening
    E(B-V)_BBB is a *free* parameter there, so the stored templates are
    unreddened; apply :func:`apply_bbb_reddening` to redden.

    Parameters
    ----------
    name : {"R06", "SN12", "KD18", "THB21"}
        Disk library name.
    log_mbh, log_edd : float, optional
        For SN12/KD18, the grid point to select (nearest-neighbor). Ignored
        for R06/THB21. Defaults to the grid midpoint.

    Returns
    -------
    wave_aa : ndarray, shape (n,)
        Wavelength [Angstrom], ascending.
    L_nu : ndarray, shape (n,)
        Disk luminosity density [erg/s/Hz], AGNFITTER-RX normalization.
    """
    name = name.upper()
    if name == "R06":
        d = _ref(_DISK_H5, "r06")
        return _h5_to_lnu(d["wavelength"], d["sed"], "BB")
    if name == "THB21":
        d = _ref(_DISK_H5, "thb21")
        return _h5_to_lnu(d["wavelength"], d["sed"][0], "BB")
    if name == "SN12":
        d = _ref(_DISK_H5, "sn12")
        sed = d["sed"]  # (n_mbh, n_edd, n_wave)
        i = _nearest(d["logBHmass"], log_mbh)
        j = sed.shape[1] // 2 if edd_index is None else int(edd_index)
        return _h5_to_lnu(d["wavelength"], sed[i, j], "BB")
    if name == "KD18":
        d = _ref(_DISK_H5, "kd18")  # flat per-row grid + per-row axis values
        mbh = np.sort(np.unique(d["logBHmass"]))
        edd = np.sort(np.unique(d["logEddra"]))
        mbh_sel = mbh[_nearest(mbh, log_mbh)]
        edd_sel = edd[_nearest(edd, log_edd)]
        k = int(np.argmax((d["logBHmass"] == mbh_sel) & (d["logEddra"] == edd_sel)))
        return _h5_to_lnu(d["wavelength"], d["sed"][k], "BB")
    raise ValueError(f"Unknown disk library {name!r}; choose from {list_disks()}")


# ── Torus libraries ─────────────────────────────────────────────────────────
def list_tori() -> list[str]:
    """Names of the torus libraries this driver can load."""
    return ["S04", "NK08", "SKIRTOR", "CAT3D"]


def torus_template(
    name: str,
    *,
    log_nh: float | None = None,
    incl: float | None = None,
    oa: float | None = None,
    tau: float | None = None,
    a: float | None = None,
    fwd: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one torus template (nearest grid point) in tengri units.

    Reproduces the template access in ``MODEL_AGNfitter.TORUS``.

    Parameters
    ----------
    name : {"S04", "NK08", "SKIRTOR", "CAT3D"}
        Torus library name.
    log_nh : float, optional
        S04 hydrogen column ``log10(N_H)``.
    incl : float, optional
        Inclination [deg] (NK08, SKIRTOR, CAT3D).
    oa : float, optional
        Half-opening angle [deg] (SKIRTOR).
    tau : float, optional
        Equatorial optical depth (SKIRTOR ``tv``).
    a : float, optional
        Radial cloud power-law index (CAT3D).
    fwd : float, optional
        Polar-wind mass fraction (CAT3D).

    Returns
    -------
    wave_aa : ndarray, shape (n,)
        Wavelength [Angstrom], ascending.
    L_nu : ndarray, shape (n,)
        Torus luminosity density [erg/s/Hz], AGNFITTER-RX normalization.
    """
    name = name.upper()
    if name == "S04":
        d = _ref(_TORUS_H5, "s04")
        i = _nearest(d["axis"], log_nh)
        return _h5_to_lnu(d["wavelength"], d["sed"][i], "TO")
    if name == "NK08":
        d = _ref(_TORUS_H5, "nk08")
        i = _nearest(d["axis"], incl)
        return _h5_to_lnu(d["wavelength"], d["sed"][i], "TO")
    if name == "SKIRTOR":
        d = _ref(_TORUS_H5, "skirtor")  # flat per-row grid + per-row axis values
        oa_ax = np.sort(np.unique(d["oa-values"]))
        incl_ax = np.sort(np.unique(d["incl-values"]))
        tv_ax = np.sort(np.unique(d["tv-values"]))
        oa_s = oa_ax[_nearest(oa_ax, oa)]
        incl_s = incl_ax[_nearest(incl_ax, incl)]
        tv_s = tv_ax[_nearest(tv_ax, tau)]
        k = int(
            np.argmax(
                (d["oa-values"] == oa_s) & (d["incl-values"] == incl_s) & (d["tv-values"] == tv_s)
            )
        )
        return _h5_to_lnu(d["wavelength"], d["sed"][k], "TO")
    if name == "CAT3D":
        d = _ref(_TORUS_H5, "cat3d")  # already the row-210+ 3-parameter view
        incl_ax = np.sort(np.unique(d["incl-values"]))
        a_ax = np.sort(np.unique(d["a-values"]))
        fwd_ax = np.sort(np.unique(d["fwd-values"]))
        incl_s = incl_ax[_nearest(incl_ax, incl)]
        a_s = a_ax[_nearest(a_ax, a)]
        fwd_s = fwd_ax[_nearest(fwd_ax, fwd)]
        k = int(
            np.argmax(
                (d["incl-values"] == incl_s) & (d["a-values"] == a_s) & (d["fwd-values"] == fwd_s)
            )
        )
        return _h5_to_lnu(d["wavelength"], d["sed"][k], "TO")
    raise ValueError(f"Unknown torus library {name!r}; choose from {list_tori()}")


def torus_axes(name: str) -> dict[str, np.ndarray]:
    """Grid axes for a torus library."""
    name = name.upper()
    if name == "S04":
        d = _ref(_TORUS_H5, "s04")
        return {"log_nh": np.asarray(d["axis"], dtype=np.float64)}
    if name == "NK08":
        d = _ref(_TORUS_H5, "nk08")
        return {"incl": np.asarray(d["axis"], dtype=np.float64)}
    if name == "SKIRTOR":
        d = _ref(_TORUS_H5, "skirtor")
        return {
            "oa": np.sort(np.unique(d["oa-values"])),
            "incl": np.sort(np.unique(d["incl-values"])),
            "tau": np.sort(np.unique(d["tv-values"])),
        }
    if name == "CAT3D":
        d = _ref(_TORUS_H5, "cat3d")
        return {
            "incl": np.sort(np.unique(d["incl-values"])),
            "a": np.sort(np.unique(d["a-values"])),
            "fwd": np.sort(np.unique(d["fwd-values"])),
        }
    raise ValueError(f"Unknown torus library {name!r}; choose from {list_tori()}")


# ── Cold-dust (starburst) libraries ─────────────────────────────────────────
def list_cold_dust() -> list[str]:
    """Names of the cold-dust libraries this driver can load."""
    return ["DH02_CE01", "S17"]


def cold_dust_template(
    name: str,
    *,
    log_irlum: float | None = None,
    tdust: float | None = None,
    fpah: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one cold-dust template in tengri units.

    Reproduces ``MODEL_AGNfitter.STARBURST``. DH02_CE01 is parameterized by IR
    luminosity; S17 (Schreiber et al. 2018) by dust temperature and PAH
    fraction.

    Parameters
    ----------
    name : {"DH02_CE01", "S17"}
        Cold-dust library name.
    log_irlum : float, optional
        DH02_CE01 ``log10(L_IR / L_sun)`` grid point.
    tdust : float, optional
        S17 dust temperature [K] grid point.
    fpah : float, optional
        S17 PAH mass fraction grid point.

    Returns
    -------
    wave_aa : ndarray, shape (n,)
        Wavelength [Angstrom], ascending.
    L_nu : ndarray, shape (n,)
        Cold-dust luminosity density [erg/s/Hz], AGNFITTER-RX normalization.
    """
    name = name.upper()
    if name == "DH02_CE01":
        d = _ref(_COLD_H5, "dh02_ce01")
        i = _nearest(d["irlum"], log_irlum)
        return _h5_to_lnu(d["wavelength"], d["sed"][i], "SB")
    if name == "S17":
        return _s17_template(tdust=tdust, fpah=fpah)
    raise ValueError(f"Unknown cold-dust library {name!r}; choose from {list_cold_dust()}")


@cache
def _s17_tables():
    """Load and cache the S17 (Schreiber+2018) dust and PAH FITS tables."""
    require_available()
    from astropy import units as u
    from astropy.table import Table

    dust = Table.read(_MODELS / "STARBURST/s17_lowvsg_dust.fits")
    pah = Table.read(_MODELS / "STARBURST/s17_lowvsg_pah.fits")
    # The FITS tables hold a single row whose LAM/SED columns are 2-D arrays
    # (n_tdust, n_wave); mirror MODEL_AGNfitter's ``column[0]`` access.
    dwl = np.asarray(dust["LAM"][0], dtype=np.float64)  # (n_tdust, n_wave) micron
    pwl = np.asarray(pah["LAM"][0], dtype=np.float64)
    d_nulnu = np.asarray(dust["SED"][0], dtype=np.float64)  # (n_tdust, n_wave) nuLnu
    p_nulnu = np.asarray(pah["SED"][0], dtype=np.float64)
    tdust = np.asarray(dust["TDUST"][0], dtype=np.float64)  # (n_tdust,) K
    dnu = (dwl * u.micron).to(u.Hz, equivalencies=u.spectral()).value  # (n_tdust, n_wave)
    pnu = (pwl * u.micron).to(u.Hz, equivalencies=u.spectral()).value
    d_lnu = d_nulnu / dnu  # (n_tdust, n_wave)
    p_lnu = p_nulnu / pnu
    fpah = np.concatenate(
        ((np.arange(0.0, 0.1, 0.01) / 100.0), (np.arange(0.1, 5.5, 0.1) / 100.0))
    )
    return dnu, d_lnu, p_lnu, tdust, fpah


def _s17_template(*, tdust=None, fpah=None):
    dnu, d_lnu, p_lnu, tdust_ax, fpah_ax = _s17_tables()
    t = _nearest(tdust_ax, tdust)
    f = _nearest(fpah_ax, fpah)
    log_nu = np.log10(dnu[t])
    fnu = (1.0 - fpah_ax[f]) * d_lnu[t] + fpah_ax[f] * p_lnu[t]
    return units.lognu_fnu_to_lnu(log_nu, _renorm("SB", fnu))


def cold_dust_axes(name: str) -> dict[str, np.ndarray]:
    """Grid axes for a cold-dust library."""
    name = name.upper()
    if name == "DH02_CE01":
        d = _safe_load("STARBURST/DH02_CE01.pickle")
        return {"log_irlum": np.asarray(d["irlum-values"], dtype=np.float64)}
    if name == "S17":
        _, _, _, tdust_ax, fpah_ax = _s17_tables()
        return {"tdust": tdust_ax, "fpah": fpah_ax}
    raise ValueError(f"Unknown cold-dust library {name!r}; choose from {list_cold_dust()}")


# ── Reddening + X-ray extension (ports of MODEL_AGNfitter helpers) ───────────
def apply_bbb_reddening(wave_aa: np.ndarray, L_nu: np.ndarray, ebv: float) -> np.ndarray:
    """Redden a disk SED with AGNFITTER-RX's Prevot SMC law (R_V = 2.72).

    Port of ``MODEL_AGNfitter.BBBred_Prevot``: ``k(lambda) = 1.39 lambda_um^-1.2
    - 0.38`` (Prevot et al. 1984 SMC), applied only blueward of the optical and
    redward of 200 eV (no reddening at X-ray energies).

    Parameters
    ----------
    wave_aa : ndarray, shape (n,)
        Wavelength [Angstrom], ascending.
    L_nu : ndarray, shape (n,)
        Unreddened disk luminosity density [erg/s/Hz].
    ebv : float
        Disk color excess E(B-V)_BBB [mag].

    Returns
    -------
    L_nu_red : ndarray, shape (n,)
        Reddened disk luminosity density [erg/s/Hz].
    """
    wave_aa = np.asarray(wave_aa, dtype=np.float64)
    L_nu = np.asarray(L_nu, dtype=np.float64)
    lam_um = wave_aa * 1e-4
    # 200 eV threshold (log nu = 16.685): redward gets reddened, blueward not.
    wl_200ev_aa = units.C_ANGSTROM_PER_S / (10.0**16.685)
    k = np.where(wave_aa > wl_200ev_aa, 1.39 * lam_um ** (-1.2) - 0.38, 0.0)
    L_nu_red = L_nu * 10.0 ** (-0.4 * k * ebv)
    return np.where(np.isfinite(L_nu_red), L_nu_red, L_nu)


# ── alpha_ox X-ray extension (port of MODEL_AGNfitter.XRAYS) ─────────────────
NU_2500: float = (3e8) / (2500 * 1e-10)  # Hz, frequency at 2500 Angstrom
NU_2KEV: float = 4.83598e17  # Hz, frequency at 2 keV
_H_KEV_PER_HZ: float = 4.135667731e-15 * 1e-3  # eV/Hz -> keV/Hz


def alpha_ox(l_2500: float, scatter: float = 0.0) -> float:
    """alpha_OX from the Just et al. 2007 L_2500 relation (with scatter).

    ``alpha_OX = -0.137 log10(L_2500) + 2.638 + scatter`` -- the relation
    AGNFITTER-RX uses to tie the 2 keV corona to the 2500 A disk continuum.

    Parameters
    ----------
    l_2500 : float
        Disk monochromatic luminosity density at 2500 A. [erg/s/Hz]
    scatter : float
        Dispersion offset Delta alpha_OX in [-0.4, 0.4]. [dimensionless]

    Returns
    -------
    float
        alpha_OX. [dimensionless]
    """
    return -0.137 * np.log10(l_2500) + 2.638 + scatter


def disk_xray_extension(
    wave_aa: np.ndarray,
    L_nu: np.ndarray,
    *,
    scatter: float = 0.0,
    gamma: float = 1.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend a disk SED into X-rays via the alpha_OX-L_2500 correlation.

    Port of ``MODEL_AGNfitter.XRAYS``: anchors a 2 keV flux from
    :func:`alpha_ox`, then lays down a Gamma=1.8 power law with a 300 keV
    exponential cutoff over log nu in [16.685, 19.7].

    Parameters
    ----------
    wave_aa : ndarray, shape (n,)
        Disk wavelength grid [Angstrom], ascending.
    L_nu : ndarray, shape (n,)
        Disk luminosity density [erg/s/Hz].
    scatter : float
        Delta alpha_OX dispersion [-0.4, 0.4].
    gamma : float
        Hard X-ray photon index. Default 1.8.

    Returns
    -------
    xray_wave_aa : ndarray, shape (1000,)
        X-ray wavelength grid [Angstrom], ascending.
    xray_L_nu : ndarray, shape (1000,)
        X-ray luminosity density [erg/s/Hz].
    """
    log_nu = np.log10(units.C_ANGSTROM_PER_S / np.asarray(wave_aa, dtype=np.float64))
    L_2500 = float(
        np.interp(np.log10(NU_2500), np.sort(log_nu), np.asarray(L_nu)[np.argsort(log_nu)])
    )
    alpha = alpha_ox(L_2500, scatter)
    fnu_2kev = L_2500 * 10.0 ** (alpha / 0.3838)
    a = fnu_2kev / ((_H_KEV_PER_HZ * NU_2KEV) ** (-gamma + 1) * np.exp(-NU_2KEV / 7.2540e19))
    xray_nu = np.logspace(16.685, 19.7, 1000)
    xray_fnu = a * (_H_KEV_PER_HZ * xray_nu) ** (-gamma + 1) * np.exp(-xray_nu / 7.2540e19)
    return units.lognu_fnu_to_lnu(np.log10(xray_nu), xray_fnu)


# ── small helpers ───────────────────────────────────────────────────────────
def _nearest(axis: np.ndarray, value: float | None) -> int:
    """Index of the grid point nearest ``value`` (midpoint if ``value`` None)."""
    axis = np.asarray(axis, dtype=np.float64)
    if value is None:
        return int(len(axis) // 2)
    return int(np.argmin(np.abs(axis - float(value))))
