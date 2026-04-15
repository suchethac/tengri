"""AGN Narrow Line Region emission with physically-motivated backends.

Provides a unified interface for computing AGN NLR emission using:

- **cue**: Neural-network emulator (Li et al. 2025) driven by AGN ionizing
  spectrum parameters.  This is the disc -> Cue -> NLR pipeline (Chain 2).
  The recommended and default backend.
- **feltre**: Feltre, Charlot & Gutkin (2016) CLOUDY photoionization grids.
  Parameterized by power-law slope α, log U_S, log n_H, metallicity Z, and
  dust-to-metal ratio ξ_d.  Requires ``data/feltre_grid.h5`` built via
  ``scripts/download_feltre_grid.py``.

The key physical link is ``agn_ionspec_from_alpha_pl``, which converts an
AGN power-law slope (f_nu ~ nu^alpha_pl) into the 7 ionizing-spectrum
parameters expected by the Cue emulator.  This lets Cue predict physically
consistent NLR emission for an AGN-ionized gas cloud.

All functions are pure JAX and JIT-compatible unless noted otherwise.

References
----------
- Feltre, Charlot & Gutkin 2016, MNRAS, 456, 3354 (arXiv:1511.08217)
- Li et al. 2025, ApJ, 986, 9 (Cue emulator)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tengri.components.nebular._constants import (
    _C_CGS,
    _H_PLANCK,
    _LOG10_ZSUN,
    _LSUN_ERG,
)
from tengri.components.nebular.ionizing_spectrum import _CLIP_RANGES, SEGMENT_EDGES

# Default path for the Feltre+2016 HDF5 grid
_DEFAULT_FELTRE_GRID_PATH = Path(__file__).resolve().parents[4] / "data" / "feltre_grid.h5"

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_NU_LYMAN = _C_CGS / (911.76e-8)  # Lyman limit frequency [Hz]
_RYDBERG_ERG = 2.1799e-11  # 13.6 eV in erg


# ---------------------------------------------------------------------------
# Ionizing spectrum conversion
# ---------------------------------------------------------------------------


def agn_ionspec_from_alpha_pl(alpha_pl: float) -> dict:
    """Convert AGN EUV power-law slope to Cue ionizing spectrum parameters.

    For an AGN with f_nu ~ nu^{alpha_pl} below 912 A, compute the 7 Cue
    ionizing-spectrum parameters (4 slopes + 3 log luminosity ratios).

    Cue parameterizes f_nu as a function of wavelength in each segment:
    ``log10(F_nu) = index * log10(lambda) + const``.  Since nu = c/lambda,
    a power law f_nu ~ nu^{alpha} becomes f_nu ~ lambda^{-alpha} in
    wavelength space.

    Parameters
    ----------
    alpha_pl : float
        EUV power-law slope in the BEAGLE-AGN convention (f_nu ~ nu^alpha_pl).
        Typical AGN: alpha_pl ~ -1.7.

    Returns
    -------
    dict
        Keys: ``ionspec_index1..4``, ``ionspec_logLratio1..3``.
        All values are clipped to the valid Cue ranges.
    """
    # Cue slope convention: F_nu vs lambda, so index = -alpha_pl
    wavelength_slope = -alpha_pl

    # All 4 segments share the same slope for a pure power law
    indices = jnp.array([wavelength_slope] * 4)

    # Compute log luminosity ratios from segment-integrated fluxes.
    # For F_nu ~ A * lambda^s, the integrated luminosity in a segment is:
    #   L_seg = integral_{lam_lo}^{lam_hi} F_nu * (c / lam^2) d(lam)
    #         = A * c * integral lam^{s-2} d(lam)
    # For s != 1:  = A * c * (lam_hi^{s-1} - lam_lo^{s-1}) / (s - 1)
    # For s == 1:  = A * c * ln(lam_hi / lam_lo)
    # The ratio L_{k+1} / L_k cancels A*c, leaving:
    edges = np.asarray(SEGMENT_EDGES, dtype=np.float64)

    s = wavelength_slope  # alias
    sp1 = s - 1.0  # exponent for the integral

    def _seg_integral(lo: float, hi: float) -> jnp.ndarray:
        """Integrated luminosity (up to common factor A*c) for one segment."""
        # Use the s != 1 branch; handle s == 1 with a safe denominator
        safe_sp1 = jnp.where(jnp.abs(sp1) > 1e-8, sp1, 1e-8)
        return (hi**safe_sp1 - lo**safe_sp1) / safe_sp1

    log_integrals = jnp.array(
        [jnp.log10(jnp.abs(_seg_integral(edges[i], edges[i + 1]))) for i in range(4)]
    )

    # logLratio_k = log10(L_{k+1} / L_k) = log10(integral_{k+1}) - log10(integral_k)
    log_ratios = jnp.diff(log_integrals)

    # Clip to valid Cue ranges
    idx1 = jnp.clip(indices[0], *_CLIP_RANGES["ionspec_index1"])
    idx2 = jnp.clip(indices[1], *_CLIP_RANGES["ionspec_index2"])
    idx3 = jnp.clip(indices[2], *_CLIP_RANGES["ionspec_index3"])
    idx4 = jnp.clip(indices[3], *_CLIP_RANGES["ionspec_index4"])
    lr1 = jnp.clip(log_ratios[0], *_CLIP_RANGES["ionspec_logLratio1"])
    lr2 = jnp.clip(log_ratios[1], *_CLIP_RANGES["ionspec_logLratio2"])
    lr3 = jnp.clip(log_ratios[2], *_CLIP_RANGES["ionspec_logLratio3"])

    return {
        "ionspec_index1": idx1,
        "ionspec_index2": idx2,
        "ionspec_index3": idx3,
        "ionspec_index4": idx4,
        "ionspec_logLratio1": lr1,
        "ionspec_logLratio2": lr2,
        "ionspec_logLratio3": lr3,
    }


# ---------------------------------------------------------------------------
# Q_H computation
# ---------------------------------------------------------------------------


def _log_qh_from_lacc(l_acc_erg: float, alpha_pl: float) -> float:
    """Estimate log10(Q_H) from accretion luminosity and EUV slope.

    For f_nu ~ nu^{alpha_pl}, the ionizing photon rate is:

        Q_H = integral_{nu_Ly}^{inf} (L_nu / h*nu) d(nu)

    We normalize using the fraction of L_acc emitted below 912 A.
    For a power law extending from 100 A to 10 um, the ionizing fraction
    depends on alpha_pl.  For alpha_pl ~ -1.7, roughly 40-60% of the
    bolometric luminosity is ionizing.

    The mean ionizing photon energy for a power law is:

        <h*nu> = integral(h*nu * nu^alpha / (h*nu)) / integral(nu^alpha / (h*nu))
               = integral(nu^alpha) / integral(nu^{alpha-1})

    Parameters
    ----------
    l_acc_erg : float
        Accretion luminosity [erg s^-1].
    alpha_pl : float
        EUV power-law slope (f_nu ~ nu^alpha_pl).

    Returns
    -------
    float
        log10(Q_H) [photons s^-1].
    """
    # Frequency limits for ionizing radiation
    nu_lyman = _NU_LYMAN  # 912 A
    # Upper limit: use 1 A (hard X-ray cutoff)
    nu_max = _C_CGS / 1e-8  # 1 Angstrom

    # Ionizing fraction of L_bol:
    #   f_ion = integral_{nu_Ly}^{nu_max} nu^alpha d(nu) /
    #           integral_{nu_min}^{nu_max} nu^alpha d(nu)
    # For convergence with alpha < -1, the integral is dominated by nu_Ly.
    # Use nu_min = c / 10um for the full SED.
    nu_min = _C_CGS / (10.0e-4)  # 10 micron = 10e-4 cm

    a = alpha_pl
    ap1 = a + 1.0
    safe_ap1 = jnp.where(jnp.abs(ap1) > 1e-8, ap1, 1e-8)

    # integral(nu^a, nu_lo, nu_hi) = (nu_hi^{a+1} - nu_lo^{a+1}) / (a+1)
    int_total = (nu_max**safe_ap1 - nu_min**safe_ap1) / safe_ap1
    int_ion = (nu_max**safe_ap1 - nu_lyman**safe_ap1) / safe_ap1
    f_ion = jnp.abs(int_ion / int_total)
    f_ion = jnp.clip(f_ion, 0.01, 1.0)

    # Mean ionizing photon energy:
    #   <h*nu> = h * integral(nu^a d(nu)) / integral(nu^{a-1} d(nu))
    #          = h * [(nu^{a+1})/(a+1)] / [(nu^a)/a]  evaluated at limits
    safe_a = jnp.where(jnp.abs(a) > 1e-8, a, 1e-8)
    int_num = int_ion  # integral(nu^a d(nu)) over ionizing range
    int_den = (nu_max**safe_a - nu_lyman**safe_a) / safe_a
    mean_hnu = _H_PLANCK * jnp.abs(int_num / int_den)
    # Ensure physical: at least 1 Rydberg
    mean_hnu = jnp.maximum(mean_hnu, _RYDBERG_ERG)

    l_ion = f_ion * l_acc_erg
    q_h = l_ion / mean_hnu

    return jnp.log10(jnp.maximum(q_h, 1.0))


# ---------------------------------------------------------------------------
# Backend: Cue emulator
# ---------------------------------------------------------------------------


def agn_nlr_cue(
    cue_backend,
    l_acc_erg: float,
    covering_fraction: float = 0.1,
    neb_logU: float = -3.0,
    gas_logn: float = 3.0,
    gas_logz: float = 0.0,
    gas_logno: float = 0.0,
    gas_logco: float = 0.0,
    alpha_pl: float = -1.7,
    ionspec_params: dict | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute AGN NLR emission using the Cue neural-network emulator.

    This is the disc → Cue → NLR pipeline: the AGN power-law ionizing
    spectrum is parameterized and fed to the Cue emulator to predict
    physically consistent nebular line emission from AGN-ionized gas.

    Parameters
    ----------
    cue_backend : CueBackend
        Initialized Cue emulator backend with loaded weights.
    l_acc_erg : float
        AGN accretion luminosity [erg s^-1].
    covering_fraction : float
        NLR covering fraction (0 to 1).  Default 0.1.
    neb_logU : float
        Gas ionization parameter log10(U).  Default -3.0.
    gas_logn : float
        Gas electron density log10(n_e / cm^-3).  Default 3.0.
    gas_logz : float
        Gas metallicity log10(Z/Zsun).  Default 0.0 (solar).
    gas_logno : float
        Gas N/O abundance ratio offset.  Default 0.0.
    gas_logco : float
        Gas C/O abundance ratio offset.  Default 0.0.
    alpha_pl : float
        AGN EUV power-law slope (f_nu ~ nu^alpha_pl).  Default -1.7.
    ionspec_params : dict or None
        Explicit Cue ionizing spectrum parameters (overrides alpha_pl).
        Keys: ``ionspec_index1..4``, ``ionspec_logLratio1..3``.

    Returns
    -------
    line_wavelengths : array
        Emission line wavelengths [Angstrom].
    line_luminosities : array
        Emission line luminosities [Lsun], scaled by covering fraction.
    """
    # Step 1: ionizing spectrum parameters
    if ionspec_params is None:
        ionspec_params = agn_ionspec_from_alpha_pl(alpha_pl)

    # Step 2: compute Q_H from L_acc and alpha_pl
    log_qh = _log_qh_from_lacc(l_acc_erg, alpha_pl)

    # Step 3: call Cue low-level API
    line_wav, line_lum = cue_backend.predict_nebular_line_luminosities(
        gas_logu=neb_logU,
        gas_logn=gas_logn,
        gas_logz=gas_logz,
        gas_logno=gas_logno,
        gas_logco=gas_logco,
        gas_logqion=log_qh,
        ionspec_index1=ionspec_params["ionspec_index1"],
        ionspec_index2=ionspec_params["ionspec_index2"],
        ionspec_index3=ionspec_params["ionspec_index3"],
        ionspec_index4=ionspec_params["ionspec_index4"],
        ionspec_logLratio1=ionspec_params["ionspec_logLratio1"],
        ionspec_logLratio2=ionspec_params["ionspec_logLratio2"],
        ionspec_logLratio3=ionspec_params["ionspec_logLratio3"],
    )

    # Step 4: scale by covering fraction
    # Cue predicts total line luminosity for the given Q_H;
    # the NLR only intercepts a fraction of the ionizing photons.
    line_lum = line_lum * covering_fraction

    return line_wav, line_lum


# ---------------------------------------------------------------------------
# Feltre+2016 NLR backend
# ---------------------------------------------------------------------------


@dataclass
class FeltreGridData:
    """Container for the Feltre+2016 grid loaded from HDF5.

    Grid shape for ``logHB_per_logq`` and ``line_ratios`` leading dims:
    ``(n_alpha, n_logUs, n_logn, n_logZ, n_xi_d)``.

    Attributes
    ----------
    alpha_axis : (4,) array
        Ionizing power-law slope values (e.g. -1.2, -1.4, -1.7, -2.0).
    logUs_axis : (4,) array
        log10(U_S) values (e.g. -1, -2, -3, -4).  May be descending.
    logn_axis : (3,) array
        log10(n_H / cm^-3) values.
    logZ_axis : (16,) array
        log10(Z) absolute metallicity values.
    xi_d_axis : (3,) array
        Dust-to-metal ratio values.
    line_wavelengths_aa : (n_lines,) array
        Vacuum wavelengths [Angstrom].
    logHB_per_logq : (4, 4, 3, 16, 3) array
        log10(L_Hβ / Q_H) where Q_H is ionizing photon rate [photons/s]
        and L_Hβ is in erg/s.  Dims: (alpha, logUs, logn, logZ, xi_d).
    line_ratios : (4, 4, 3, 16, 3, n_lines) array
        L_line / L_Hβ (dimensionless).
    """

    alpha_axis: jnp.ndarray
    logUs_axis: jnp.ndarray
    logn_axis: jnp.ndarray
    logZ_axis: jnp.ndarray
    xi_d_axis: jnp.ndarray
    line_wavelengths_aa: jnp.ndarray
    logHB_per_logq: jnp.ndarray
    line_ratios: jnp.ndarray


def _load_feltre_grid(filepath: str | Path) -> FeltreGridData:
    """Load the Feltre+2016 NLR photoionization grid from HDF5.

    Parameters
    ----------
    filepath : str or Path
        Path to ``feltre_grid.h5``.

    Raises
    ------
    FileNotFoundError
        If the grid file does not exist.
    KeyError
        If required datasets are missing in the HDF5 file.
    """
    import h5py

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Feltre+2016 NLR grid not found at '{filepath}'.\n"
            "Run scripts/download_feltre_grid.py for instructions on how to\n"
            "obtain or regenerate the grid data."
        )

    with h5py.File(filepath, "r") as f:
        grp = f["feltre"]
        alpha_axis = jnp.array(grp["alpha_axis"][:])
        logUs_axis = jnp.array(grp["logUs_axis"][:])
        logn_axis = jnp.array(grp["logn_axis"][:])
        logZ_axis = jnp.array(grp["logZ_axis"][:])
        xi_d_axis = jnp.array(grp["xi_d_axis"][:])
        line_wavelengths_aa = jnp.array(grp["line_wavelengths_aa"][:])
        logHB_per_logq = jnp.array(grp["logHB_per_logq"][:])
        line_ratios = jnp.array(grp["line_ratios"][:])

    return FeltreGridData(
        alpha_axis=alpha_axis,
        logUs_axis=logUs_axis,
        logn_axis=logn_axis,
        logZ_axis=logZ_axis,
        xi_d_axis=xi_d_axis,
        line_wavelengths_aa=line_wavelengths_aa,
        logHB_per_logq=logHB_per_logq,
        line_ratios=line_ratios,
    )


def _nearest_idx(axis: jnp.ndarray, value: float) -> int:
    """Return nearest-neighbor index into a 1-D axis array."""
    return int(jnp.argmin(jnp.abs(axis - value)))


class FeltreNLRBackend:
    """Feltre, Charlot & Gutkin (2016) AGN NLR photoionization backend.

    Computes AGN narrow-line region emission by interpolating the
    CLOUDY c13.03 photoionization grids from Feltre et al. (2016, MNRAS 456,
    3354).  The grid covers AGN-ionized gas parameterized by ionizing power-law
    slope α, ionization parameter log U_S, gas density log n_H, metallicity Z,
    and dust-to-metal ratio ξ_d.

    **Grid data required**: ``data/feltre_grid.h5`` built via
    ``scripts/download_feltre_grid.py``.  If the file is absent, construction
    raises ``FileNotFoundError``.

    Interpolation strategy
    ----------------------
    - **Continuous axes** (log U_S, log Z, log n_H): C²-continuous triweight
      interpolation via ``interp_nd_triweight`` — compatible with VI/MAP.
    - **Discrete axes** (α, ξ_d): nearest-neighbor index lookup.

    This backend has ``has_continuum = False``.

    Parameters
    ----------
    grid_path : str or Path
        Path to ``feltre_grid.h5``.

    Example
    -------
    >>> backend = FeltreNLRBackend("data/feltre_grid.h5")
    >>> wave, lum = backend.predict_agn_nlr_lines(
    ...     alpha_pl=-1.7,
    ...     neb_logU=-2.0,
    ...     neb_logn=3.0,
    ...     neb_logZ_gas=-1.8477,  # log10(Z_abs) ≈ log10(Z_sun)
    ...     xi_d=0.3,
    ...     log_qh=53.0,
    ... )

    References
    ----------
    Feltre, Charlot & Gutkin 2016, MNRAS 456, 3354.
    """

    name = "feltre"
    has_free_params = True
    has_continuum = False

    def __init__(self, grid_path: str | Path = _DEFAULT_FELTRE_GRID_PATH) -> None:
        self.grid = _load_feltre_grid(grid_path)

        # Pre-compute triweight edges for continuous axes at init time.
        # Axes must be sorted ascending for interp_nd_triweight.
        from tengri.utils.interpolation import edges_for_grid

        # logUs_axis may be descending (-1, -2, -3, -4); sort ascending.
        self._logUs_sorted = jnp.sort(self.grid.logUs_axis)
        self._logUs_descending = bool(self.grid.logUs_axis[0] > self.grid.logUs_axis[-1])

        self._logn_sorted = jnp.sort(self.grid.logn_axis)
        self._logn_descending = bool(self.grid.logn_axis[0] > self.grid.logn_axis[-1])

        self._logZ_sorted = jnp.sort(self.grid.logZ_axis)
        self._logZ_descending = bool(self.grid.logZ_axis[0] > self.grid.logZ_axis[-1])

        self._edges_logUs = edges_for_grid(self._logUs_sorted)
        self._edges_logn = edges_for_grid(self._logn_sorted)
        self._edges_logZ = edges_for_grid(self._logZ_sorted)

    def predict_agn_nlr_lines(
        self,
        alpha_pl: float = -1.7,
        neb_logU: float = -3.0,
        neb_logn: float = 3.0,
        neb_logZ_gas: float = _LOG10_ZSUN,
        xi_d: float = 0.3,
        log_qh: float = 53.0,
        neb_fesc: float = 0.0,
        **_kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute AGN NLR emission line luminosities via grid interpolation.

        Parameters
        ----------
        alpha_pl : float
            AGN EUV power-law slope (f_nu ~ nu^alpha_pl).  Nearest-neighbor
            mapped to grid values [-1.2, -1.4, -1.7, -2.0].
        neb_logU : float
            Gas ionization parameter log10(U_S).  Interpolated continuously
            over [-4, -1].
        neb_logn : float
            Gas density log10(n_H / cm^-3).  Interpolated continuously
            over [2, 4].
        neb_logZ_gas : float
            Gas metallicity log10(Z) absolute.  Interpolated continuously.
            Converts to log10(Z) if absolute; use _LOG10_ZSUN = -1.8477 for solar.
        xi_d : float
            Dust-to-metal ratio.  Nearest-neighbor mapped to [0.1, 0.3, 0.5].
        log_qh : float
            log10(Q_H) ionizing photon rate [photons/s].
        neb_fesc : float
            Ionizing photon escape fraction [0, 1].

        Returns
        -------
        wavelengths : (n_lines,) array
            Vacuum wavelengths [Angstrom].
        luminosities : (n_lines,) array
            Emission line luminosities [Lsun].
        """
        from tengri.forward.precompute.grid import interp_nd_triweight

        grid = self.grid

        # --- Discrete axes: nearest-neighbor index lookup ---
        i_alpha = _nearest_idx(grid.alpha_axis, alpha_pl)
        i_xi_d = _nearest_idx(grid.xi_d_axis, xi_d)

        # --- Sort grid slice to ascending order if needed ---
        # Slice for fixed (alpha, xi_d): shape (n_logUs, n_logn, n_logZ, ...)
        logHB_slice = grid.logHB_per_logq[i_alpha, :, :, :, i_xi_d]  # (nU, nn, nZ)
        ratios_slice = grid.line_ratios[i_alpha, :, :, :, i_xi_d, :]  # (nU, nn, nZ, nl)

        if self._logUs_descending:
            logHB_slice = logHB_slice[::-1, :, :]
            ratios_slice = ratios_slice[::-1, :, :, :]
        if self._logn_descending:
            logHB_slice = logHB_slice[:, ::-1, :]
            ratios_slice = ratios_slice[:, ::-1, :, :]
        if self._logZ_descending:
            logHB_slice = logHB_slice[:, :, ::-1]
            ratios_slice = ratios_slice[:, :, ::-1, :]

        axes = (self._logUs_sorted, self._logn_sorted, self._logZ_sorted)
        edges = (self._edges_logUs, self._edges_logn, self._edges_logZ)
        point = (neb_logU, neb_logn, neb_logZ_gas)

        logHB_interp = interp_nd_triweight(logHB_slice, axes, edges, point)
        ratios_interp = interp_nd_triweight(ratios_slice, axes, edges, point)

        # L_Hβ = 10^{logHB_per_logq} × Q_H  [erg/s]
        # L_line = ratio × L_Hβ × (1 − fesc) / L_sun
        l_hb_erg = (10.0**logHB_interp) * (10.0**log_qh)
        line_lum = ratios_interp * l_hb_erg * (1.0 - neb_fesc) / _LSUN_ERG

        return grid.line_wavelengths_aa, line_lum


# ---------------------------------------------------------------------------
# Backend: analytic (existing)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def agn_nlr_emission(
    backend: str = "cue",
    cue_backend=None,
    feltre_backend: FeltreNLRBackend | None = None,
    l_acc_erg: float = 1e44,
    covering_fraction: float = 0.1,
    alpha_pl: float = -1.7,
    neb_logU: float = -3.0,
    gas_logn: float = 3.0,
    gas_logz: float = 0.0,
    gas_logno: float = 0.0,
    gas_logco: float = 0.0,
    ionspec_params: dict | None = None,
    neb_logZ_gas: float | None = None,
    xi_d: float = 0.3,
    log_qh: float = 53.0,
    neb_fesc: float = 0.0,
    **kwargs,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Unified AGN NLR emission dispatcher.

    Routes to one of the available NLR backends depending on
    ``backend``.

    Parameters
    ----------
    backend : str
        Backend name: ``"cue"`` or ``"feltre"``.
    cue_backend : CueBackend or None
        Required when ``backend="cue"``.
    feltre_backend : FeltreNLRBackend or None
        Required when ``backend="feltre"``.  Initialize with
        ``FeltreNLRBackend(grid_path)`` before calling.
    l_acc_erg : float
        AGN accretion luminosity [erg s^-1].  Default 1e44.
    covering_fraction : float
        NLR covering fraction.  Default 0.1.
    alpha_pl : float
        AGN EUV power-law slope.  Default -1.7.
    neb_logU : float
        Gas ionization parameter log10(U).
    gas_logn : float
        Gas density log10(n_e / cm^-3) (Cue backend).
    gas_logz : float
        Gas metallicity log10(Z/Zsun) (Cue backend).
    gas_logno : float
        Gas N/O offset (Cue backend).
    gas_logco : float
        Gas C/O offset (Cue backend).
    ionspec_params : dict or None
        Explicit Cue ionizing spectrum parameters.
    neb_logZ_gas : float or None
        Gas metallicity log10(Z) absolute (Feltre backend).  If None,
        defaults to log10(Z_sun) = -1.8477.
    xi_d : float
        Dust-to-metal ratio (Feltre backend).  Default 0.3.
    log_qh : float
        log10(Q_H) ionizing photon rate (Feltre backend).  Default 53.0.
    neb_fesc : float
        Ionizing photon escape fraction (Feltre backend).  Default 0.0.
    **kwargs
        Additional keyword arguments (ignored).

    Returns
    -------
    tuple (line_wavelengths, line_luminosities)
        wavelengths in Angstrom, luminosities in Lsun.

    Raises
    ------
    ValueError
        If ``backend`` is not recognized or required backend object is missing.
    """
    if backend == "cue":
        if cue_backend is None:
            raise ValueError(
                "cue_backend must be provided when backend='cue'. "
                "Initialize with CueBackend(weights_path)."
            )
        return agn_nlr_cue(
            cue_backend=cue_backend,
            l_acc_erg=l_acc_erg,
            covering_fraction=covering_fraction,
            neb_logU=neb_logU,
            gas_logn=gas_logn,
            gas_logz=gas_logz,
            gas_logno=gas_logno,
            gas_logco=gas_logco,
            alpha_pl=alpha_pl,
            ionspec_params=ionspec_params,
        )
    elif backend == "feltre":
        if feltre_backend is None:
            raise ValueError(
                "feltre_backend must be provided when backend='feltre'. "
                "Initialize with FeltreNLRBackend(grid_path='data/feltre_grid.h5')."
            )
        _logZ = neb_logZ_gas if neb_logZ_gas is not None else _LOG10_ZSUN
        return feltre_backend.predict_agn_nlr_lines(
            alpha_pl=alpha_pl,
            neb_logU=neb_logU,
            neb_logn=gas_logn,
            neb_logZ_gas=_logZ,
            xi_d=xi_d,
            log_qh=log_qh,
            neb_fesc=neb_fesc,
        )
    else:
        raise ValueError(f"Unknown AGN NLR backend '{backend}'. Choose from: 'cue', 'feltre'.")
