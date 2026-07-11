"""Thin ``rpy2`` wrappers around ProSpect's R forward model for the notebook.

ProSpect (Robotham et al. 2020) is an R package. This module drives it
in-process through ``rpy2``: it loads the ProSpect library and its stellar /
dust / AGN template data once, caches them at module level, and exposes one
Python function per notebook section. Each call evaluates ProSpect's own R
functions and returns the result in tengri's unit convention (erg/s/Hz,
rest-frame Å) so the notebook can overlay the two codes directly.

The heavy template libraries (the ``BC03lr`` stellar grid, the Dale 2014 dust
templates, the SKIRTOR / Fritz AGN libraries) are loaded lazily and cached, so
sections that re-run during a notebook rebuild do not pay the load cost twice.

References
----------
.. [1] Robotham, A.S.G., Bellstedt, S., Lagos, C.d.P., et al. (2020).
       ProSpect: generating rapid spectral energy distributions with complex
       star formation and metallicity histories. MNRAS, 495, 905.
       arXiv:2002.06980.
.. [2] Bellstedt, S., Robotham, A.S.G., Driver, S.P., et al. (2020). GAMA:
       a forensic SED reconstruction of the cosmic star formation history
       and metallicity evolution by galaxy type. MNRAS, 498, 5581.
       arXiv:2009.03919.
"""

from __future__ import annotations

import os

# rpy2's prebuilt API-mode shim is linked against a CRAN R framework path that
# is absent on a Homebrew R; ABI mode links at run time against whatever R is
# on PATH, which is what we want. Set this before importing rpy2.
os.environ.setdefault("RPY2_CFFI_MODE", "ABI")
# Make sure Homebrew's R resolves for rpy2's R_HOME discovery.
if "/opt/homebrew/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

from typing import Any

import numpy as np

from . import units as U

# ---------------------------------------------------------------------------
# rpy2 session + ProSpect library, lazily constructed and cached.
# ---------------------------------------------------------------------------
_R: Any = None  # rpy2.robjects module
_PROSPECT: Any = None  # importr("ProSpect")
_DATA: dict[str, Any] = {}  # cache of loaded ProSpect data objects


def _get_r():
    """Return the cached ``rpy2.robjects`` handle, importing ProSpect once."""
    global _R, _PROSPECT
    if _R is None:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr

        _PROSPECT = importr("ProSpect")
        _R = ro
    return _R


def _data(name: str):
    """Lazy-load and cache a ProSpect data object (``data(name)`` in R)."""
    if name not in _DATA:
        ro = _get_r()
        ro.r(f'data("{name}", package="ProSpectData")')
        _DATA[name] = ro.r[name]
    return _DATA[name]


def _rvec(x) -> Any:
    """Convert a 1-D NumPy array into an R numeric vector."""
    ro = _get_r()
    return ro.FloatVector(np.asarray(x, dtype=np.float64).ravel())


def _np(robj) -> np.ndarray:
    """Copy an R vector / column into an owned float64 NumPy array.

    ``np.asarray`` on an rpy2 vector can return a view that *shares memory*
    with the underlying R object. R's garbage collector may later free or
    reuse that buffer — during any subsequent R call — silently corrupting
    arrays returned by earlier calls. Forcing a copy with ``np.array`` gives
    NumPy sole ownership, so driver outputs stay valid for the whole notebook.
    """
    return np.array(robj, dtype=np.float64, copy=True)


def _listget(rlist, key: str):
    """Pull a named element out of an R list by name."""
    names = list(rlist.names)
    return rlist[names.index(key)]


def _rfn(name: str):
    """Fetch a ProSpect R function by name from the global environment."""
    return _get_r().r[name]


def _df_wave_lum(df) -> tuple[np.ndarray, np.ndarray]:
    """Pull ``(wave_aa, L_nu)`` out of a ProSpect ``data.frame(wave, lum)``.

    ProSpect carries rest-frame spectra as ``$wave`` [Å] and ``$lum``
    [L⊙/Å]; this converts the luminosity column to tengri's erg/s/Hz.
    """
    wave = _np(_listget(df, "wave"))
    lum = _np(_listget(df, "lum"))
    return U.lsun_per_aa_to_erg_per_hz(wave, lum)


# ---------------------------------------------------------------------------
# §1 — single stellar populations (direct speclib slice)
# ---------------------------------------------------------------------------
def ssp_spectrum(
    *, Z: float = 0.02, age_gyr: float = 1.0, lib: str = "BC03lr"
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return one ProSpect single-SSP spectrum :math:`L_\nu` [erg/s/Hz/M⊙].

    Indexes the ProSpect ``speclib`` directly at the grid metallicity and age
    nearest the requested values — no interpolation, so the comparison against
    tengri's own BC03 grid is library against library.

    Parameters
    ----------
    Z : float
        Absolute metallicity (metal mass fraction). 0.02 ≈ solar.
    age_gyr : float
        SSP age in Gyr.
    lib : str
        ProSpect stellar library name (``"BC03lr"``, ``"BC03hr"``, ``"EMILES"``).

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Rest-frame wavelength [Å].
    L_nu : ndarray, shape (n_wave,)
        Spectral luminosity [erg/s/Hz] per M⊙ formed.
    """
    speclib = _data(lib)
    wave = _np(_listget(speclib, "Wave"))
    age_yr = _np(_listget(speclib, "Age"))
    z_grid = _np(_listget(speclib, "Z"))
    zspec = _listget(speclib, "Zspec")  # list of [n_age, n_wave] matrices

    i_z = int(np.argmin(np.abs(z_grid - Z)))
    i_age = int(np.argmin(np.abs(age_yr - age_gyr * 1e9)))
    mat = np.asarray(zspec[i_z], dtype=np.float64)  # shape (n_age, n_wave)
    if mat.shape[0] != age_yr.shape[0] and mat.shape[1] == age_yr.shape[0]:
        mat = mat.T  # guard against (n_wave, n_age) orientation
    L_lambda = mat[i_age, :]
    return U.lsun_per_aa_to_erg_per_hz(wave, L_lambda)


def ssp_grid_info(lib: str = "BC03lr") -> dict[str, Any]:
    """Return ``{n_wave, n_age, n_z, wave_max, z_grid, age_gyr}`` for a library."""
    speclib = _data(lib)
    wave = _np(_listget(speclib, "Wave"))
    age_yr = _np(_listget(speclib, "Age"))
    z_grid = _np(_listget(speclib, "Z"))
    return {
        "n_wave": int(wave.shape[0]),
        "n_age": int(age_yr.shape[0]),
        "n_z": int(z_grid.shape[0]),
        "wave_max": float(wave.max()),
        "z_grid": z_grid,
        "age_gyr": age_yr / 1e9,
    }


# ---------------------------------------------------------------------------
# §2 — star formation history (massfunc_* evaluated on a lookback grid)
# ---------------------------------------------------------------------------
def sfh_curve(
    *, sfh: str = "snorm", ngrid: int = 500, age_universe_gyr: float = 13.7, **pars: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return :math:`\mathrm{SFR}(t_{\rm lookback})` from a ProSpect ``massfunc_*``.

    ProSpect's ``massfunc_*`` functions take age in **years** (lookback from the
    observation epoch) and SFH parameters in **Gyr** (``mpeak``, ``mtau``,
    ``mperiod``, ``magemax``), returning SFR in M⊙/yr.

    Parameters
    ----------
    sfh : str
        ProSpect SFH name without the ``massfunc_`` prefix (``"snorm"``,
        ``"dtau"``, ``"const"``, …).
    ngrid : int
        Number of lookback samples between 0 and ``age_universe_gyr``.
    age_universe_gyr : float
        Upper edge of the lookback grid [Gyr].
    **pars
        SFH parameters (e.g. ``mSFR``, ``mpeak``, ``mperiod``, ``mskew``).

    Returns
    -------
    t_lookback_yr : ndarray, shape (ngrid,)
        Lookback time [yr], 0 = observation epoch.
    sfr : ndarray, shape (ngrid,)
        Star formation rate [M⊙/yr].
    """
    fn = _rfn(f"massfunc_{sfh}")
    t_yr = np.linspace(0.0, age_universe_gyr * 1e9, ngrid)
    sfr = _np(fn(_rvec(t_yr), **pars))
    return t_yr, sfr


# ---------------------------------------------------------------------------
# §2b — metallicity history (Zfunc_* coupled to the SFH)
# ---------------------------------------------------------------------------
def metallicity_history(
    *,
    zfunc: str = "massmap_lin",
    sfh: str = "snorm",
    ngrid: int = 500,
    age_universe_gyr: float = 13.7,
    Zstart: float = 1e-4,
    Zfinal: float = 0.02,
    Zagemax: float = 13.8,
    yield_: float = 0.03,
    **sfh_pars: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return ProSpect's metallicity history Z(age) and Z vs cumulative mass.

    ProSpect's mass-mapped metallicity models (Bellstedt et al. 2020) tie the
    metallicity to the *cumulative stellar mass formed*: ``Zfunc_massmap_lin``
    maps it linearly, ``Zfunc_massmap_box`` through the Lynden-Bell closed-box
    relation with a fixed ``yield``. Both integrate the supplied ``massfunc``
    to build the cumulative-mass CDF, so the metallicity history is a direct
    consequence of the SFH.

    Parameters
    ----------
    zfunc : str
        ProSpect Zfunc name without prefix (``"massmap_lin"``,
        ``"massmap_box"``, ``"p2"``).
    sfh : str
        SFH name without the ``massfunc_`` prefix.
    ngrid, age_universe_gyr : int, float
        Lookback grid definition.
    Zstart, Zfinal, Zagemax : float
        Initial / final absolute metallicity and the age over which the
        history runs [Gyr].
    yield_ : float
        Closed-box yield ρ (``massmap_box`` only).
    **sfh_pars
        SFH parameters forwarded to the ``massfunc``.

    Returns
    -------
    age_yr : ndarray, shape (ngrid,)
        Lookback age [yr].
    Z : ndarray, shape (ngrid,)
        Absolute metallicity (metal mass fraction) at each age.
    cumulative_mass_frac : ndarray, shape (ngrid,)
        Fraction of total stellar mass formed by each age (0 at the present,
        1 at the oldest age) — ProSpect's mapping variable.
    """
    massfunc = _rfn(f"massfunc_{sfh}")
    fn_z = _rfn(f"Zfunc_{zfunc}")
    age_yr = np.linspace(1e5, age_universe_gyr * 1e9, ngrid)

    kw: dict[str, Any] = dict(
        Zstart=Zstart, Zfinal=Zfinal, Zagemax=Zagemax, massfunc=massfunc, **sfh_pars
    )
    if zfunc == "massmap_box":
        # ``yield`` is a Python reserved word; pass it through the R name.
        kw["yield"] = yield_
    Z = _np(fn_z(_rvec(age_yr), **kw))

    # Cumulative mass fraction formed by each age, from the same SFH.
    sfr = _np(massfunc(_rvec(age_yr), **sfh_pars))
    order = np.argsort(age_yr)[::-1]  # oldest -> youngest accumulates mass
    csum = np.cumsum(sfr[order])
    cmf = np.empty_like(csum)
    cmf[order] = csum / csum[-1]
    return age_yr, Z, cmf


# ---------------------------------------------------------------------------
# §4 — dust attenuation curves (Charlot & Fall, CF_screen / CF_birth)
# ---------------------------------------------------------------------------
def attenuation_curve(
    *,
    component: str = "screen",
    tau: float = 1.0,
    pow_: float = -0.7,
    Eb: float = 0.0,
    wave_aa: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return :math:`A_\lambda` [mag] for a ProSpect Charlot & Fall component.

    ProSpect's ``CF_screen`` / ``CF_birth`` return a transmission factor
    :math:`T(\lambda) \in [0, 1]`; this converts to attenuation in magnitudes,
    :math:`A_\lambda = -2.5\log_{10} T`.

    Parameters
    ----------
    component : {"screen", "birth"}
        Which Charlot & Fall term. The screen term carries the optional 2175 Å
        bump (``Eb``).
    tau : float
        Optical depth at the 5500 Å pivot.
    pow_ : float
        Power-law slope.
    Eb : float
        2175 Å bump strength (screen only).
    wave_aa : array_like, optional
        Wavelength grid [Å]. Defaults to 1000–30000 Å.

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Wavelength [Å].
    A_lambda : ndarray, shape (n_wave,)
        Attenuation [mag].
    """
    if wave_aa is None:
        wave_aa = np.logspace(np.log10(1000.0), np.log10(30000.0), 2000)
    wave_aa = np.asarray(wave_aa, dtype=np.float64)
    if component == "birth":
        T = _np(_rfn("CF_birth")(_rvec(wave_aa), tau=tau, pow=pow_))
    else:
        T = _np(_rfn("CF_screen")(_rvec(wave_aa), tau=tau, pow=pow_, Eb=Eb))
    with np.errstate(divide="ignore"):
        A = -2.5 * np.log10(np.clip(T, 1e-30, None))
    return wave_aa, A


# ---------------------------------------------------------------------------
# §3, §5, §6, §7, §9, §11 — full forward model via ProSpectSED
# ---------------------------------------------------------------------------
def prospect_sed(
    *,
    massfunc: str = "snorm",
    sfh_pars: dict[str, float] | None = None,
    Z: float = 0.02,
    z: float = 0.0,
    tau_birth: float = 0.0,
    tau_screen: float = 0.0,
    pow_birth: float = -0.7,
    pow_screen: float = -0.7,
    alpha_SF_birth: float = 1.0,
    alpha_SF_screen: float = 3.0,
    agn_model: str | None = None,
    AGNlum: float = 0.0,
    addradio_SF: bool = False,
    IGMabsorb: Any = 0,
    lib: str = "BC03lr",
    extra: dict[str, Any] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run ``ProSpectSED`` and return its components in tengri units.

    Returns a dict with keys ``FinalLum``, ``StarsAtten``, ``StarsUnAtten``,
    ``DustEmit``, and (when an AGN is on) ``AGN`` — each a ``(wave_aa, L_nu)``
    pair in erg/s/Hz. Dust emission requires the Dale 2014 templates, which are
    always passed so energy balance is enforced whenever ``tau > 0``.

    Parameters mirror ``ProSpectSED``; ``massfunc`` and ``sfh_pars`` set the
    SFH, ``Z`` the (scalar) metallicity, ``agn_model`` selects ``"SKIRTOR"`` or
    ``"Fritz"`` and is only active when ``AGNlum > 0``. ``extra`` passes any
    further ProSpectSED argument verbatim.
    """
    ro = _get_r()
    speclib = _data(lib)
    dale = _data("Dale_NormTot")
    fn = _rfn("ProSpectSED")

    kw: dict[str, Any] = dict(
        massfunc=_rfn(f"massfunc_{massfunc}"),
        speclib=speclib,
        Dale=dale,
        z=z,
        tau_birth=tau_birth,
        tau_screen=tau_screen,
        pow_birth=pow_birth,
        pow_screen=pow_screen,
        alpha_SF_birth=alpha_SF_birth,
        alpha_SF_screen=alpha_SF_screen,
        AGNlum=AGNlum,
        addradio_SF=addradio_SF,
        IGMabsorb=IGMabsorb,
        Z=Z,
        filters=ro.NULL,
        returnall=True,
    )
    if sfh_pars:
        kw.update(sfh_pars)
    if agn_model is not None and AGNlum > 0:
        kw["AGN"] = _data(agn_model)
    if extra:
        kw.update(extra)

    res = fn(**kw)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in ("FinalLum", "StarsAtten", "StarsUnAtten", "DustEmit", "AGN"):
        try:
            comp = _listget(res, key)
        except ValueError:
            continue
        names = list(comp.names) if hasattr(comp, "names") and comp.names is not ro.NULL else []
        if "wave" in names and "lum" in names:
            out[key] = _df_wave_lum(comp)
    return out


# ---------------------------------------------------------------------------
# §8 — nebular emission (SFHfunc emission isolation)
# ---------------------------------------------------------------------------
def nebular_lnu(
    *, sfr: float = 1.0, Z: float = 0.02, veldisp: float = 50.0, q: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return ProSpect's nebular-line :math:`L_\nu` [erg/s/Hz] for a given SFR.

    Calls ProSpect's ``emissionLines`` directly with the line-ratio lookup table
    ``LKL10_NormHalpha`` (Levesque et al. 2010, normalized to Hα). ProSpect ties
    the Hα luminosity to the star formation rate through a fixed coefficient
    (``L_Hα = SFR × 21612724 L⊙``) and distributes the remaining lines by the
    metallicity-dependent ratios in the table. The returned spectrum is nebular
    lines only — no stellar continuum — so no subtraction is needed.

    Parameters
    ----------
    sfr : float
        Star formation rate [M⊙/yr].
    Z : float
        Gas-phase metallicity (absolute mass fraction).
    veldisp : float
        Line velocity dispersion [km/s] used to broaden the lines.
    q : float, optional
        Ionization parameter [cm/s] passed to ``emissionLines``. If ``None``
        (default), ProSpect derives it from metallicity via ``Z2q`` (Orsi 2014),
        which at solar Z returns a low ``q ≈ 1.4e7`` cm/s appropriate for soft
        ionization — this strongly suppresses the collisionally-excited metal
        lines ([O III], [O II]), giving an anomalously weak [O III]/Hα ≈ 0.014
        (#761). To compare on a **matched ionization parameter** against a Cloudy
        grid run at ``logU``, pass ``q = U·c`` (e.g. ``logU = -2`` → ``q = 3e8``);
        recombination lines (Hα, Hβ) are q-insensitive so this only fixes the
        metal-line ratios.

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Rest-frame wavelength [Å].
    L_nu : ndarray, shape (n_wave,)
        Nebular-line spectral luminosity [erg/s/Hz].
    """
    fn = _rfn("emissionLines")
    lkl10 = _rfn("LKL10_NormHalpha")  # lazy-loaded line-ratio table
    kw = dict(SFR=sfr, Z=Z, veldisp=veldisp, LKL10=lkl10)
    if q is not None:
        kw["q"] = q
    e = fn(**kw)
    wave = _np(_listget(e, "wave"))
    lum = _np(_listget(e, "lum"))  # L⊙/Å
    return U.lsun_per_aa_to_erg_per_hz(wave, lum)


# ---------------------------------------------------------------------------
# §9 — AGN torus (raw SKIRTOR / Fritz template, no dust reprocessing)
# ---------------------------------------------------------------------------
def agn_torus_lnu(
    *, model: str = "SKIRTOR", lum_erg: float = 1e44
) -> tuple[np.ndarray, np.ndarray, float]:
    r"""Return the raw ProSpect AGN template SED :math:`L_\nu` [erg/s/Hz].

    Calls ``SKIRTOR_interp`` or ``Fritz_interp`` directly to get the AGN
    template (accretion disc continuum plus torus reprocessing) scaled to a
    bolometric luminosity ``lum_erg``. This is the template itself, *before* the
    extra Charlot & Fall screen and Dale reprocessing that ``ProSpectSED``
    applies to it — so it is the correct thing to compare against tengri's own
    SKIRTOR block.

    Parameters
    ----------
    model : {"SKIRTOR", "Fritz"}
        AGN template library.
    lum_erg : float
        AGN bolometric luminosity [erg/s].

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Rest-frame wavelength [Å].
    L_nu : ndarray, shape (n_wave,)
        AGN spectral luminosity [erg/s/Hz].
    log_lbol_lsun : float
        ``log10(L_bol / L⊙)`` of the returned template, for normalizing tengri.
    """
    fn = _rfn(f"{model}_interp")
    template = _data(model)
    res = fn(lum=lum_erg, **{model: template})
    wave = _np(_listget(res, "wave"))
    # ProSpect's AGN templates carry L_λ in erg/s/Å (cgs), not L⊙/Å like the
    # stellar libraries — so the L⊙ factor must NOT be applied here. Convert with
    # the λ²/c Jacobian alone.
    l_lambda = _np(_listget(res, "lum"))  # erg/s/Å
    L_nu = l_lambda * wave**2 / U.C_ANGSTROM_PER_S
    nu = U.C_ANGSTROM_PER_S / wave[::-1]
    l_bol_erg = float(np.trapezoid(L_nu[::-1], nu))
    return wave, L_nu, float(np.log10(l_bol_erg / U.L_SUN_ERG_PER_S))


# ---------------------------------------------------------------------------
# §12 — IGM transmission (Inoue et al. 2014)
# ---------------------------------------------------------------------------
def igm_transmission(wave_rest_aa: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    r"""Return ProSpect's Inoue et al. (2014) IGM transmission :math:`T(\lambda_{\rm rest})`.

    ProSpect's ``Inoue14_IGM`` takes observed-frame wavelengths and returns the
    transmission :math:`T \in [0, 1]`. The rest-frame grid is redshifted to the
    observed frame for the call and the transmission returned against the
    original rest-frame grid.

    Parameters
    ----------
    wave_rest_aa : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    z : float
        Source redshift.

    Returns
    -------
    wave_rest_aa : ndarray, shape (n_wave,)
        Rest-frame wavelength [Å] (unchanged).
    T : ndarray, shape (n_wave,)
        IGM transmission in [0, 1].
    """
    wave_rest = np.asarray(wave_rest_aa, dtype=np.float64)
    wave_obs = wave_rest * (1.0 + z)
    T = _np(_rfn("Inoue14_IGM")(_rvec(wave_obs), z))
    return wave_rest, np.clip(T, 0.0, 1.0)
