"""AGN Narrow Line Region emission with physically-motivated backends.

Provides a unified interface for computing AGN NLR emission using:

- **cue**: Neural-network emulator (Li et al. 2025) driven by AGN ionizing
  spectrum parameters.  This is the disc -> Cue -> NLR pipeline (Chain 2).
  The recommended and default backend.
- **feltre**: Placeholder for Feltre et al. (2016) photoionization grids
  (not yet implemented).

The key physical link is ``agn_ionspec_from_alpha_pl``, which converts an
AGN power-law slope (f_nu ~ nu^alpha_pl) into the 7 ionizing-spectrum
parameters expected by the Cue emulator.  This lets Cue predict physically
consistent NLR emission for an AGN-ionized gas cloud.

All functions are pure JAX and JIT-compatible unless noted otherwise.

References
----------
- Feltre et al. 2016, MNRAS, 456, 3354
- Li et al. 2025, ApJ, 986, 9 (Cue emulator)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tengri.models.nebular.ionizing_spectrum import _CLIP_RANGES, SEGMENT_EDGES

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_C_CGS = 2.9979e10  # Speed of light [cm s^-1]
_H_PLANCK = 6.626e-27  # Planck constant [erg s]
_LSUN_ERG = 3.828e33  # Solar luminosity [erg s^-1]
_C_AA = 2.9979e18  # Speed of light [Angstrom s^-1]
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
    wavelength: jnp.ndarray,
    cue_backend,
    l_acc_erg: float,
    covering_fraction: float = 0.1,
    gas_logu: float = -3.0,
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
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].  Not directly used by Cue
        (which returns its own line wavelengths), but kept for API symmetry.
    cue_backend : CueBackend
        Initialized Cue emulator backend with loaded weights.
    l_acc_erg : float
        AGN accretion luminosity [erg s^-1].
    covering_fraction : float
        NLR covering fraction (0 to 1).  Default 0.1.
    gas_logu : float
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
        gas_logu=gas_logu,
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
# Backend: analytic (existing)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def agn_nlr_emission(
    wavelength: jnp.ndarray,
    backend: str = "cue",
    cue_backend=None,
    l_acc_erg: float = 1e44,
    covering_fraction: float = 0.1,
    alpha_pl: float = -1.7,
    gas_logu: float = -3.0,
    gas_logn: float = 3.0,
    gas_logz: float = 0.0,
    gas_logno: float = 0.0,
    gas_logco: float = 0.0,
    ionspec_params: dict | None = None,
    **kwargs,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Unified AGN NLR emission dispatcher.

    Routes to one of the available NLR backends depending on
    ``backend``.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    backend : str
        Backend name: ``"cue"`` or ``"feltre"``.
    cue_backend : CueBackend or None
        Required when ``backend="cue"``.
    l_acc_erg : float
        AGN accretion luminosity [erg s^-1].  Default 1e44.
    covering_fraction : float
        NLR covering fraction.  Default 0.1.
    alpha_pl : float
        AGN EUV power-law slope.  Default -1.7.
    gas_logu : float
        Gas ionization parameter log10(U) (Cue backend).
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
    **kwargs
        Additional keyword arguments forwarded to the backend.

    Returns
    -------
    For ``"cue"``: tuple (line_wavelengths, line_luminosities) in [Lsun].

    Raises
    ------
    ValueError
        If ``backend`` is not recognized.
    NotImplementedError
        If ``backend="feltre"`` (not yet implemented).
    """
    if backend == "cue":
        if cue_backend is None:
            raise ValueError(
                "cue_backend must be provided when backend='cue'. "
                "Initialize with CueBackend(weights_path)."
            )
        return agn_nlr_cue(
            wavelength,
            cue_backend=cue_backend,
            l_acc_erg=l_acc_erg,
            covering_fraction=covering_fraction,
            gas_logu=gas_logu,
            gas_logn=gas_logn,
            gas_logz=gas_logz,
            gas_logno=gas_logno,
            gas_logco=gas_logco,
            alpha_pl=alpha_pl,
            ionspec_params=ionspec_params,
        )
    elif backend == "feltre":
        raise NotImplementedError("Feltre+2016 grid backend not yet implemented. Use 'cue'.")
    else:
        raise ValueError(f"Unknown AGN NLR backend '{backend}'. Choose from: 'cue', 'feltre'.")
