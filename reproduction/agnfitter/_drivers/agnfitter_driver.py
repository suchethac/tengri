"""Thin reader for AGNFITTER-RX's model template libraries.

This driver loads the disk, torus, and cold-dust template libraries that ship
inside AGNFITTER-RX (Martinez-Ramirez et al. 2024) and returns each template on
tengri's plotting convention -- ascending wavelength [Angstrom], L_nu
[erg/s/Hz]. It never runs AGNFITTER-RX's fitter; it reproduces the exact
template access in ``functions/MODEL_AGNfitter.py`` so the notebook overlays the
*same* SEDs the AGN fitter feeds its likelihood.

AGNFITTER-RX is a clone-and-load research code, not a pip package, so every
reference template it ships is repackaged into committed HDF5 under ``data/``
and the driver reads *only* those. Nothing here touches an AGNFITTER-RX
checkout at runtime: the notebook runs on a clean clone of this repository and
in CI.

Reference grids, all committed under ``data/`` (h5 group in parentheses):

=========  ==================================================  =========================
Component  Library (driver name)                               Committed h5 (group)
=========  ==================================================  =========================
disk       ``R06``  (Richards et al. 2006)                     ``..._bbb_...`` (r06)
disk       ``SN12`` (Slone & Netzer 2012)                      ``..._bbb_...`` (sn12)
disk       ``KD18`` (Kubota & Done 2018)                       ``..._bbb_...`` (kd18)
disk       ``THB21`` (Temple, Hewett & Banerji 2021)           ``..._bbb_...`` (thb21)
torus      ``S04``  (Silva et al. 2004)                        ``..._torus_...`` (s04)
torus      ``NK08`` (Nenkova et al. 2008)                      ``..._torus_...`` (nk08)
torus      ``SKIRTOR`` (Stalevski et al. 2016)                 ``..._torus_...`` (skirtor)
torus      ``CAT3D`` (Honig & Kishimoto 2017)                  ``..._torus_...`` (cat3d)
cold dust  ``DH02_CE01`` (Dale & Helou 02 + Chary & Elbaz 01)  ``..._cold_dust_...`` (dh02_ce01)
cold dust  ``S17`` (Schreiber et al. 2018)                     ``..._cold_dust_...`` (s17)
=========  ==================================================  =========================

(``...`` stands for the ``agnfitter_`` prefix and the ``_reference.h5`` suffix.)

Regenerating a grid is a *build-time* job for the scripts under ``scripts/``
(``build_agnfitter_bbb_reference.py``, ``build_agnfitter_s17_reference.py``),
which fetch the upstream files from the pinned ``AGNfitter-rX_v0.1`` tag and
carry their own hardened pickle loaders. The contract that runtime never reads
a checkout is pinned by
``tests/contract/test_reproduction_driver_no_clone.py``.

References
----------
.. [1] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). doi:10.1051/0004-6361/202449329. arXiv:2405.12111.
"""

from __future__ import annotations

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
    """Load and cache the S17 (Schreiber+2018) dust and PAH tables.

    Reads the ``s17`` group vendored into the committed cold-dust h5 by
    ``scripts/build_agnfitter_s17_reference.py`` — runtime never touches the
    AGNfitter-rX clone (same contract as every other template here).
    """
    require_available()
    import h5py

    with h5py.File(_COLD_H5, "r") as h:
        if "s17" not in h:
            raise FileNotFoundError(
                f"{_COLD_H5} has no 's17' group — regenerate it with "
                "scripts/build_agnfitter_s17_reference.py."
            )
    d = _ref(_COLD_H5, "s17")
    dwl = np.asarray(d["dust_lam_um"], dtype=np.float64)  # (n_tdust, n_wave) micron
    pwl = np.asarray(d["pah_lam_um"], dtype=np.float64)
    d_nulnu = np.asarray(d["dust_sed_nulnu"], dtype=np.float64)  # (n_tdust, n_wave) nuLnu
    p_nulnu = np.asarray(d["pah_sed_nulnu"], dtype=np.float64)
    tdust = np.asarray(d["tdust"], dtype=np.float64)  # (n_tdust,) K
    dnu = _C_AA / (dwl * 1e4)  # micron -> Angstrom -> Hz
    pnu = _C_AA / (pwl * 1e4)
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
        d = _ref(_COLD_H5, "dh02_ce01")
        return {"log_irlum": np.asarray(d["irlum"], dtype=np.float64)}
    if name == "S17":
        _, _, _, tdust_ax, fpah_ax = _s17_tables()
        return {"tdust": tdust_ax, "fpah": fpah_ax}
    raise ValueError(f"Unknown cold-dust library {name!r}; choose from {list_cold_dust()}")


# ── Radio (AGN + starburst) formulas ────────────────────────────────────────
def agn_radio_spl(
    freq_hz: np.ndarray,
    *,
    alpha: float = -0.75,
    nu_t: float = 1.0e9,
    log_nu_cut: float = 13.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Single power-law AGN radio SED (nRADdata==1 branch).

    Computes the simple power-law radio spectrum used when insufficient radio
    data constrain the AGN. This reproduces the ``nRADdata==1`` path in
    ``MODEL_AGNfitter.AGN_RAD`` (Martinez-Ramirez et al. 2024, Eqs. 9-10).

    Parameters
    ----------
    freq_hz : ndarray, shape (n,)
        Frequencies [Hz], ascending.
    alpha : float, optional
        Spectral index (default -0.75, hardcoded in upstream).
    nu_t : float, optional
        Turnover frequency [Hz] (default 1.0e9, hardcoded in upstream).
    log_nu_cut : float, optional
        Cutoff exponent: high-frequency exponential cutoff at nu_cut =
        10**log_nu_cut [Hz] (default 13.0).

    Returns
    -------
    freq_hz : ndarray, shape (n,)
        Input frequencies, ascending.
    F_nu : ndarray, shape (n,)
        Flux density F_nu [dimensionless in upstream convention], unnormalized;
        the notebook normalizes to physical flux at 5 GHz.

    Notes
    -----
    Functional form: F_nu ∝ (freq/nu_t)^alpha * exp(-freq/10^log_nu_cut).
    The upstream code skips a cosmetic renormalization factor of 1e-30
    (applied later via ``_renorm('AGN_RAD', ...)``).

    References
    ----------
    .. [1] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
       radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
       (2024). doi:10.1051/0004-6361/202449329. arXiv:2405.12111.
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    fnu = (freq_hz / nu_t) ** alpha * np.exp(-freq_hz / (10.0**log_nu_cut))
    return freq_hz, fnu


def agn_radio_dpl(
    freq_hz: np.ndarray,
    *,
    alpha1: float = -0.75,
    alpha2: float = -0.1,
    log_nu_t: float = 10.0,
    log_nu_cut: float = 13.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Broken power-law AGN radio SED (nRADdata>3 / DPL-4 branch).

    Computes a double power-law radio spectrum used when sufficient radio data
    constrain both the low- and high-frequency slopes. This reproduces the
    ``nRADdata > 3`` / ``DPL-4`` path in ``MODEL_AGNfitter.AGN_RAD``
    (Martinez-Ramirez et al. 2024, Eqs. 9-10).

    Parameters
    ----------
    freq_hz : ndarray, shape (n,)
        Frequencies [Hz], ascending.
    alpha1 : float, optional
        Low-frequency spectral index (default -0.75).
    alpha2 : float, optional
        High-frequency spectral index (default -0.1).
    log_nu_t : float, optional
        Turnover exponent: breakpoint frequency nu_t = 10**log_nu_t [Hz]
        (default 10.0).
    log_nu_cut : float, optional
        Cutoff exponent: high-frequency exponential cutoff at nu_cut =
        10**log_nu_cut [Hz] (default 13.0).

    Returns
    -------
    freq_hz : ndarray, shape (n,)
        Input frequencies, ascending.
    F_nu : ndarray, shape (n,)
        Flux density F_nu [dimensionless in upstream convention], unnormalized;
        the notebook normalizes to physical flux at 5 GHz.

    Notes
    -----
    Functional form:
    F_nu ∝ (freq/10^log_nu_t)^alpha1 * (1 - exp(-(10^log_nu_t/freq)^(alpha1-alpha2)))
          * exp(-freq/10^log_nu_cut).
    The upstream code skips a cosmetic renormalization factor of 1e-30
    (applied later via ``_renorm('AGN_RAD', ...)``).

    References
    ----------
    .. [1] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
       radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
       (2024). doi:10.1051/0004-6361/202449329. arXiv:2405.12111.
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    nu_t = 10.0**log_nu_t
    fnu = (
        (freq_hz / nu_t) ** alpha1
        * (1.0 - np.exp(-((nu_t / freq_hz) ** (alpha1 - alpha2))))
        * np.exp(-freq_hz / (10.0**log_nu_cut))
    )
    return freq_hz, fnu


@cache
def _s17_radio_tables():
    """Load and cache the S17 radio dust table from the committed h5.

    Reads the ``s17_radio`` group vendored into the committed cold-dust h5
    by ``scripts/build_agnfitter_s17_reference.py`` — runtime never touches
    the AGNfitter-rX clone (same contract as every other template here).

    Returns
    -------
    dust_nu_hz : ndarray, shape (n_tdust, n_wave)
        Dust-table frequencies [Hz], ascending per-row.
    dust_sed_lnu : ndarray, shape (n_tdust, n_wave)
        Dust SED L_nu values [dimensionless upstream units].
    pah_nu_hz : ndarray, shape (n_pah,)
        PAH-table frequencies [Hz], from wavelengths.
    pah_sed_lnu : ndarray, shape (n_tdust, n_pah)
        PAH SED L_nu values [dimensionless upstream units].
    tdust : ndarray, shape (n_tdust,)
        Dust temperatures [K].
    lir_conv : ndarray, shape (n_tdust,)
        LIR conversion factors (Bell 2003 convention).
    fpah : ndarray, shape (n_fpah,)
        PAH mass fraction grid.
    """
    require_available()
    d = _ref(_COLD_H5, "s17_radio")
    dust_nu_hz = np.asarray(d["dust_nu_hz"], dtype=np.float64)  # (n_tdust, n_wave)
    dust_sed_lnu = np.asarray(d["dust_sed_lnu"], dtype=np.float64)
    pah_lam_um = np.asarray(d["pah_lam_um"], dtype=np.float64)  # (n_wave,) micron
    pah_sed_nulnu = np.asarray(d["pah_sed_nulnu"], dtype=np.float64)  # (n_tdust, n_wave)
    tdust = np.asarray(d["tdust"], dtype=np.float64)
    lir_conv = np.asarray(d["lir_conv"], dtype=np.float64)

    # Convert PAH wavelengths to frequencies
    pah_nu_hz = _C_AA / (pah_lam_um * 1e4)  # micron -> Angstrom -> Hz
    pah_sed_lnu = pah_sed_nulnu / pah_nu_hz  # (n_tdust, n_wave)

    fpah = np.concatenate(
        ((np.arange(0.0, 0.1, 0.01) / 100.0), (np.arange(0.1, 5.5, 0.1) / 100.0))
    )

    return dust_nu_hz, dust_sed_lnu, pah_nu_hz, pah_sed_lnu, tdust, lir_conv, fpah


def cold_dust_radio_template(
    *,
    tdust: float | None = None,
    fpah: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one S17 cold-dust radio template in tengri units.

    Extends :func:`cold_dust_template` to the S17 model with embedded radio
    extension. This reproduces the template access in
    ``MODEL_AGNfitter.STARBURST`` for the ``S17_radio`` branch.

    Parameters
    ----------
    tdust : float, optional
        S17 dust temperature [K] grid point (nearest-neighbor). Defaults to
        grid midpoint.
    fpah : float, optional
        S17 PAH mass fraction grid point (nearest-neighbor). Defaults to grid
        midpoint.

    Returns
    -------
    wave_aa : ndarray, shape (n,)
        Wavelength [Angstrom], ascending.
    L_nu : ndarray, shape (n,)
        Cold-dust (+ radio) luminosity density [erg/s/Hz], AGNFITTER-RX
        normalization.

    Notes
    -----
    This model includes radio frequencies (down to ~1.4 GHz = 0.2 mm) and was
    calibrated against the infrared-radio correlation (Bell 2003). The radio
    tail above ~1 mm is a power law with spectral index ~-0.75.

    References
    ----------
    .. [1] C. Schreiber, et al., "Dust temperature and mid-to-total infrared
       color distributions for star-forming galaxies at 0 < z < 4," A&A 609,
       A30 (2018). arXiv:1710.10276. doi:10.1051/0004-6361/201731506.
    .. [2] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
       radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
       (2024). doi:10.1051/0004-6361/202449329. arXiv:2405.12111.
    """
    dust_nu_hz, dust_sed_lnu, _pah_nu_hz, pah_sed_lnu, tdust_ax, _lir_conv, fpah_ax = (
        _s17_radio_tables()
    )
    t = _nearest(tdust_ax, tdust)
    f = _nearest(fpah_ax, fpah)

    # Mirror MODEL_AGNfitter line 404: pad PAH with 23 zeros for radio alignment
    log_nu = np.log10(dust_nu_hz[t])
    pah_padded = np.concatenate((np.zeros(23), pah_sed_lnu[t, ::-1]))
    fnu = (1.0 - fpah_ax[f]) * dust_sed_lnu[t] + fpah_ax[f] * pah_padded
    return units.lognu_fnu_to_lnu(log_nu, _renorm("SB", fnu))


def cold_dust_radio_axes() -> dict[str, np.ndarray]:
    """Grid axes for the S17_radio cold-dust library.

    Returns
    -------
    dict
        Keys ``tdust`` (dust temperature [K]) and ``fpah`` (PAH mass fraction),
        ``lir_conv`` (LIR conversion factor for Bell 2003 relation).
    """
    _, _, _, _, tdust_ax, lir_conv, fpah_ax = _s17_radio_tables()
    return {"tdust": tdust_ax, "fpah": fpah_ax, "lir_conv": lir_conv}


# ── Reddening + X-ray extension (as in MODEL_AGNfitter helpers) ──────────────
def apply_bbb_reddening(wave_aa: np.ndarray, L_nu: np.ndarray, ebv: float) -> np.ndarray:
    """Redden a disk SED with AGNFITTER-RX's Prevot SMC law (R_V = 2.72).

    Implements ``MODEL_AGNfitter.BBBred_Prevot``: ``k(lambda) = 1.39 lambda_um^-1.2
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


# ── alpha_ox X-ray extension (as in MODEL_AGNfitter.XRAYS) ───────────────────
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

    Implements ``MODEL_AGNfitter.XRAYS``: anchors a 2 keV flux from
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
