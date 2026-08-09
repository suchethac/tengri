# SPDX-License-Identifier: BSD-3-Clause
"""AGN Narrow Line Region emission with physically-motivated backends.

Provides a unified interface for computing AGN NLR emission using:

- **cue**: Neural-network emulator (Li et al. 2025) driven by AGN ionizing
  spectrum parameters.  This is the disc -> Cue -> NLR pipeline (Chain 2).
  The recommended and default backend.
- **feltre**: Feltre, Charlot & Gutkin (2016) CLOUDY photoionization grids.
  Parameterized by power-law slope α, log U_S, log n_H, metallicity Z, and
  dust-to-metal ratio ξ_d.  Requires ``data/feltre_grid.h5`` built via
  ``scripts/build_feltre_grid.py``.

The key physical link is ``agn_ionspec_from_alpha_pl``, which converts an
AGN power-law slope (f_nu ~ nu^alpha_pl) into the 7 ionizing-spectrum
parameters expected by the Cue emulator.  This lets Cue predict physically
consistent NLR emission for an AGN-ionized gas cloud.

All functions are pure JAX and JIT-compatible unless noted otherwise.

Comparison with BEAGLE
----------------------
BEAGLE (Chevallard & Charlot 2016) uses the Feltre+2016 CLOUDY c13.03
photoionization grids as its AGN NLR model.  tengri implements the same
grids via ``FeltreNLRBackend`` and additionally provides the Cue neural-
network emulator as a default alternative.

Feltre+2016 grid axes (BEAGLE-equivalent):

+------------------+---------------------------+-------------------+
| Axis             | Range                     | N points          |
+==================+===========================+===================+
| α (EUV slope)    | -2.0, -1.7, -1.4, -1.2   | 4 (discrete)      |
+------------------+---------------------------+-------------------+
| log U_S          | -5.0 to -1.0              | 9 (continuous)    |
+------------------+---------------------------+-------------------+
| log n_H [cm⁻³]   | 2.0, 3.0, 4.0             | 3 (continuous)    |
+------------------+---------------------------+-------------------+
| Z (metallicity)  | 0.0001 to 0.07            | 16 (continuous)   |
+------------------+---------------------------+-------------------+
| ξ_d (dust/metal) | 0.1, 0.3, 0.5             | 3 (discrete)      |
+------------------+---------------------------+-------------------+
| **Total**        | 4 × 9 × 3 × 16 × 3        | **5,184 models**  |
+------------------+---------------------------+-------------------+

20 emission lines: [OII]3727, Hβ, [OIII]4959/5007, [OI]6300, [NII]6548/6584,
Hα, [SII]6717/6731, NV1240, CIV1548/1551, HeII1640, OIII]1661/1666,
[SiIII]1883, SiIII]1888, [CIII]1907, CIII]1910.

Normalization: the NEOGAL ASCII files (``data/neogal/``) store luminosities in
erg/s per L_acc = 10^45 erg/s.  ``FeltreNLRBackend`` uses the internally
consistent normalization ``log10(L_Hβ / Q_H)``.  The conversion is performed
by ``scripts/build_feltre_grid.py`` using ``_log_qh_from_lacc`` in this module.

Ionizing spectrum: Feltre+2016 vs Cue
--------------------------------------
The parameter ``alpha`` in the Feltre+2016 grid is the **UV spectral index**
defined as F_ν ∝ ν^α (Feltre et al. 2016, MNRAS 456, 3354; NEOGAL README).
It is a single power-law slope parameterizing the EUV continuum from the Lyman
limit to ~ 2500 Å.  The rest of the AGN SED (X-ray, IR) has fixed slopes
following Charlot & Longhetti (2001).

This is very different from Cue's 7-parameter ionizing SED description:
Cue uses 4 segment slopes (``ionspec_index1..4``) in wavelength space plus
3 log-luminosity ratios between adjacent segments (``ionspec_logLratio1..3``).
Cue can represent an arbitrary broken power-law, while Feltre is restricted to
4 choices of a single UV slope.

Cue vs Feltre+2016 comparison:

+----------------------------+---------------------------+---------------------------+
| Feature                    | Feltre+2016 (BEAGLE)      | Cue (tengri default)      |
+============================+===========================+===========================+
| Ionizing SED shape         | Single power-law f_ν ~    | 4-segment broken power-   |
|                            | ν^α; 4 discrete α values  | law; 7 continuous params  |
+----------------------------+---------------------------+---------------------------+
| Free α / slope values      | 4 grid points (-2,-1.7,   | fully continuous via NN   |
|                            | -1.4,-1.2); nearest-nbr   | interpolation             |
+----------------------------+---------------------------+---------------------------+
| N emission lines           | 20                        | ~271                      |
+----------------------------+---------------------------+---------------------------+
| CLOUDY version             | c13.03 (2013)             | training grids c17+       |
+----------------------------+---------------------------+---------------------------+
| UV line coverage           | limited (UV-optical)      | UV to NIR                 |
+----------------------------+---------------------------+---------------------------+
| JAX / JIT                  | yes (triweight interp.)   | yes (neural network)      |
+----------------------------+---------------------------+---------------------------+
| Differentiable             | yes (C² triweight)        | yes (smooth NN)           |
+----------------------------+---------------------------+---------------------------+
| C/O axis                   | no                        | yes (gas_logco)           |
+----------------------------+---------------------------+---------------------------+
| N/O axis                   | no                        | yes (gas_logno)           |
+----------------------------+---------------------------+---------------------------+

Grid data: ``data/neogal/AGN_NLR_nebular_feltre16/`` (raw ASCII from NEOGAL).
Build the HDF5 grid with ``python scripts/build_feltre_grid.py``.

Comparison with Synthesizer (Lovell et al. 2025; Roper et al. 2026)
-------------------------------------------------
Synthesizer (https://github.com/synthesizer-project/synthesizer) ships
separate AGN NLR and BLR CLOUDY grids run with CLOUDY c23.01, a decade
newer than Feltre+2016 (c13.03).  Key structural differences:

+----------------------------+---------------------------+---------------------------+
| Feature                    | Feltre+2016 / tengri      | Synthesizer c23.01 grids  |
+============================+===========================+===========================+
| CLOUDY version             | c13.03 (2013)             | c23.01 (2023)             |
+----------------------------+---------------------------+---------------------------+
| NLR / BLR separation       | NLR only                  | separate NLR + BLR grids  |
+----------------------------+---------------------------+---------------------------+
| N emission lines           | 20                        | 215                       |
+----------------------------+---------------------------+---------------------------+
| Full spectra stored        | no (lines only)           | yes (9244 λ pts, UV–mm)   |
+----------------------------+---------------------------+---------------------------+
| AGN luminosity axis        | fixed L_acc = 10^45 erg/s | BH mass + Eddington ratio |
+----------------------------+---------------------------+---------------------------+
| Inclination axis           | no                        | cos(inclination) [0.09,   |
|                            |                           | 0.5] (type-1 / type-2)   |
+----------------------------+---------------------------+---------------------------+
| Ionizing SED               | single power-law α        | emergent from BH+disc     |
|                            | (4 discrete values)       | Shakura-Sunyaev spectrum  |
+----------------------------+---------------------------+---------------------------+
| Metallicity axis           | 16 Z values               | 2 pts (test); prod TBD    |
+----------------------------+---------------------------+---------------------------+
| Normalization (lines)      | erg/s per L_acc=10^45     | W per bolometric L_bol    |
+----------------------------+---------------------------+---------------------------+
| JAX / JIT in tengri        | yes (FeltreNLRBackend)    | not yet integrated        |
+----------------------------+---------------------------+---------------------------+

The Synthesizer test grids (2-point per axis, 19 MB each) are at
``data/synthesizer_grids/test_grid_agn-nlr.hdf5`` and
``data/synthesizer_grids/test_grid_agn-blr.hdf5``.  Production grids
(full axis coverage) require Synthesizer Box credentials
(see ``synthesizer-download --agn-grids``).

References
----------

- Feltre, Charlot & Gutkin 2016, MNRAS, 456, 3354 (arXiv:1511.08217)
- Chevallard & Charlot 2016, MNRAS, 462, 1415 (BEAGLE)
- Li et al. 2024, ApJ, 969, 28 (Cue v1)
- Li et al. 2025, ApJ, 986, 9 (Cue v2, AGN extension)
- Lovell et al. 2025 (doi:10.33232/001c.145766) + Roper et al. 2026 (doi:10.21105/joss.09436)

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tengri._data_setup import package_or_env_data_path
from tengri.components.nebular._constants import (
    _C_CGS,
    _H_PLANCK,
    _LOG10_ZSUN,
    _LSUN_ERG,
)
from tengri.components.nebular.ionizing_spectrum import _CLIP_RANGES, SEGMENT_EDGES

# Default path for the Feltre+2016 HDF5 grid. Honors $TENGRI_DATA_DIR (#1431).
_DEFAULT_FELTRE_GRID_PATH = package_or_env_data_path("feltre_grid.h5")

# ── Physical constants ────────────────────────────────────────────
_NU_LYMAN = _C_CGS / (911.76e-8)  # Lyman limit frequency [Hz]
_RYDBERG_ERG = 2.1799e-11  # 13.6 eV in erg


# ── Ionizing spectrum conversion ──────────────────────────────────


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
        EUV power-law slope in the BEAGLE-AGN convention (f_nu ~ nu^alpha_pl)
        [dimensionless]. Typical AGN: alpha_pl ~ -1.7.

    Returns
    -------
    dict
        Keys: ``ionspec_index1..4``, ``ionspec_logLratio1..3``.
        All values are clipped to the valid Cue ranges [dimensionless].

    References
    ----------
    .. [1] M. Li et al., "The Cue Nebular Emulator: Fast, Interpretable
       Predictions of Emission-Line Strengths from Stellar Populations,"
       ApJ, 986, 9 (2025). arXiv:2405.04598.
       https://doi.org/10.3847/1538-4357/ad7fe3
    .. [2] A. Feltre, S. Charlot, and J. Gutkin, "Updated photoionization
       models of the CLOUDY c13.03 code," MNRAS, 456, 3354 (2016).
       arXiv:1511.08217. https://doi.org/10.1093/mnras/stw2180

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    For a pure power law (single slope across all segments), the wavelength-space
    slope is ``-alpha_pl``. The log luminosity ratios between adjacent segments
    are computed from segment-integrated fluxes, which depend on the slope and
    segment boundaries.

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


# ── Q_H computation ───────────────────────────────────────────────


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
        EUV power-law slope (f_nu ~ nu^alpha_pl) [dimensionless].

    Returns
    -------
    float
        log10(Q_H) ionizing photon rate [log10(photons s^-1)].

    References
    ----------
    .. [1] A. Feltre, S. Charlot, and J. Gutkin, "Updated photoionization
       models of the CLOUDY c13.03 code," MNRAS, 456, 3354 (2016).
       arXiv:1511.08217. https://doi.org/10.1093/mnras/stw2180

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

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


# ── Backend: Cue emulator ─────────────────────────────────────────


def agn_nlr_cue(
    cue_backend,
    l_acc_erg: float,
    covering_fraction: float = 0.1,
    neb_logU: float = -3.0,
    # Differs from the declared gas_logn default (2.0) on purpose: that
    # declaration is the *galaxy* Cue HII-region density, while this is the AGN
    # narrow-line region, whose canonical density is ~1e3 (matching the separate
    # agn_nlr_logn declaration). Same parameter name, different physical region.
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
        NLR covering fraction (0 to 1). Default 0.1 [dimensionless].
    neb_logU : float
        Gas ionization parameter log10(U). Default -3.0 [log10(U)].
    gas_logn : float
        Gas electron density log10(n_e / cm^-3). Default 3.0 [log10(cm^-3)].
    gas_logz : float
        Gas metallicity log10(Z/Zsun). Default 0.0 (solar) [dimensionless].
    gas_logno : float
        Gas N/O abundance ratio offset [dimensionless]. Default 0.0.
    gas_logco : float
        Gas C/O abundance ratio offset [dimensionless]. Default 0.0.
    alpha_pl : float
        AGN EUV power-law slope (f_nu ~ nu^alpha_pl) [dimensionless].
        Default -1.7.
    ionspec_params : dict or None
        Explicit Cue ionizing spectrum parameters (overrides alpha_pl).
        Keys: ``ionspec_index1..4``, ``ionspec_logLratio1..3`` [dimensionless].

    Returns
    -------
    line_wavelengths : array, shape (n_lines,)
        Emission line vacuum wavelengths [Angstrom].
    line_luminosities : array, shape (n_lines,)
        Emission line luminosities [L_sun], scaled by covering fraction.

    References
    ----------
    .. [1] M. Li et al., "The Cue Nebular Emulator: Fast, Interpretable
       Predictions of Emission-Line Strengths from Stellar Populations,"
       ApJ, 986, 9 (2025). arXiv:2405.04598.
       https://doi.org/10.3847/1538-4357/ad7fe3

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable w.r.t. all continuous parameters.

    The pipeline consists of:
    1. Compute ionizing spectrum parameters from power-law slope (via
       ``agn_ionspec_from_alpha_pl``).
    2. Estimate the ionizing photon rate Q_H from accretion luminosity and slope.
    3. Call the Cue neural-network emulator with gas parameters and spectrum params.
    4. Scale line luminosities by the NLR covering fraction.

    """
    if ionspec_params is None:
        ionspec_params = agn_ionspec_from_alpha_pl(alpha_pl)

    log_qh = _log_qh_from_lacc(l_acc_erg, alpha_pl)

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

    # Cue predicts the line luminosity for the full Q_H; the NLR
    # intercepts only a fraction of those ionizing photons.
    line_lum_erg = line_lum * covering_fraction

    # ``predict_nebular_line_luminosities`` returns erg/s, but this function's
    # contract — shared with the Feltre and Synthesizer backends behind
    # :func:`agn_nlr_emission` — is L_sun (#1073). Every consumer multiplies by
    # L_SUN on the way out, so returning erg/s here scaled the NLR lines by an
    # extra L_SUN (~3.8e33).
    return line_wav, line_lum_erg / _LSUN_ERG


# ── Synthesizer NLR backend ───────────────────────────────────────


@dataclass
class SynthesizerGridData:
    """Container for Synthesizer CLOUDY c23.01 grid loaded from HDF5.

    Grid shape for ``log_line_per_qh`` leading dims:
    ``(n_mass, n_edd, n_inc, n_met, n_ionU, n_nH)``.

    Attributes
    ----------
    mass_axis : ndarray, shape (n_mass,)
        Black hole mass in log10 space [log10(kg)].
    eddington_axis : ndarray, shape (n_edd,)
        Accretion rate (Eddington ratio) in log10 space [log10(Eddington ratio)].
    cosine_axis : ndarray, shape (n_inc,)
        Inclination angle cosine, linear not log [dimensionless].
    metallicity_axis : ndarray, shape (n_met,)
        Metallicity in log10 space [log10(Z_sun)].
    logU_axis : ndarray, shape (n_ionU,)
        Ionization parameter in log10 space [log10(U)].
    logn_axis : ndarray, shape (n_nH,)
        Hydrogen density in log10 space [log10(n_H / cm^-3)].
    line_wavelengths_aa : ndarray, shape (n_lines,)
        Emission line vacuum wavelengths [Angstrom].
    log_line_per_qh : ndarray, shape (n_mass, n_edd, n_inc, n_met, n_ionU, n_nH, n_lines)
        log10(L_line / Q_H) where L_line is in L_sun and Q_H is in photons/s
        [log10(L_sun·s/photons)].

    Notes
    -----
    Grid data is interpolated using C²-continuous triweight interpolation on all
    6 parameter axes. All axes except ``cosine_axis`` are stored in log10 space
    internally for uniform sampling.

    """

    mass_axis: jnp.ndarray
    eddington_axis: jnp.ndarray
    cosine_axis: jnp.ndarray
    metallicity_axis: jnp.ndarray
    logU_axis: jnp.ndarray
    logn_axis: jnp.ndarray
    line_wavelengths_aa: jnp.ndarray
    log_line_per_qh: jnp.ndarray
    # log10(Q_H / L_bol) on the six-axis grid [log10(photons/s per W)], i.e. the
    # disc model's specific ionizing luminosity. Lets a caller recover the grid's
    # own Q_H normalization instead of assuming an ionizing-spectrum slope.
    log_qh_specific: jnp.ndarray | None = None
    # Discrete grid line luminosities per unit bolometric, log10(L_line / L_bol),
    # on the six-axis grid + line axis. Used by the line-ratio parity test.
    log_line_per_lbol: jnp.ndarray | None = None
    line_ids: tuple[str, ...] | None = None
    # Reprocessed nebular spectrum (continuum + lines) on the grid's native
    # wavelength axis, per unit bolometric luminosity: physical L_nu [erg/s/Hz]
    # = nebular_per_lbol * L_bol[erg/s] * covering_fraction. This is the array
    # Synthesizer's UnifiedAGN extracts (``extract="nebular"``) for its NLR/BLR
    # components, so reading it reproduces UnifiedAGN exactly (issue #694) rather
    # than re-broadening the discrete ``/lines`` table.
    spectra_wavelengths_aa: jnp.ndarray | None = None
    nebular_per_lbol: jnp.ndarray | None = None


def _load_synthesizer_nlr_grid(filepath: str | Path) -> SynthesizerGridData:
    """Load Synthesizer CLOUDY c23.01 AGN NLR grid from HDF5.

    Loads grid axes and emission line data, converts line luminosities from
    per-unit-bolometric to per-unit-ionizing-photon normalization.

    Parameters
    ----------
    filepath : str or Path
        Path to Synthesizer test grid HDF5 file
        (e.g. ``data/synthesizer_grids/test_grid_agn-nlr.hdf5``).

    Returns
    -------
    SynthesizerGridData
        Loaded grid data with log_line_per_qh in log10(L_sun·s/photons).

    Raises
    ------
    FileNotFoundError
        If the grid file does not exist.
    KeyError
        If required datasets are missing.

    """
    import h5py

    from tengri.utils.physics_constants import L_SUN

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Synthesizer NLR grid not found at '{filepath}'.\n"
            "Download or generate the grid with Synthesizer Box credentials."
        )

    with h5py.File(filepath, "r") as f:
        # Load axes: order from root attributes is preserved in the grid
        mass_kg = jnp.array(f["axes"]["mass"][:])
        eddington = jnp.array(f["axes"]["accretion_rate_eddington"][:])
        cosine_inc = jnp.array(f["axes"]["cosine_inclination"][:])
        metallicities = jnp.array(f["axes"]["metallicities"][:])
        ionU = jnp.array(f["axes"]["ionisation_parameter"][:])
        nH = jnp.array(f["axes"]["hydrogen_density"][:])

        # Convert to log10 where appropriate. The grid stores BH mass in kg; the
        # backend's API uses log10(M_sun) (and its defaults assume it), so convert
        # here — otherwise the mass coordinate sits ~30 dex off the grid and is
        # silently clamped to an edge. M_sun = 1.98841586e30 kg (the grid's own
        # convention: its mass[0] = 1.988e38 kg is exactly 1e8 M_sun).
        _M_SUN_KG = 1.98841586e30
        mass_axis = jnp.log10(mass_kg / _M_SUN_KG)
        eddington_axis = jnp.log10(eddington)
        cosine_axis = cosine_inc  # Linear
        metallicity_axis = jnp.log10(metallicities)
        logU_axis = jnp.log10(ionU)
        logn_axis = jnp.log10(nH)

        # Load emission lines
        line_wav = jnp.array(f["lines"]["wavelength"][:])
        line_lum = jnp.array(f["lines"]["luminosity"][:])  # (2,2,2,2,2,2,215)
        line_ids = tuple(
            i.decode() if isinstance(i, bytes) else str(i) for i in f["lines"]["id"][:]
        )

        # Load log10(specific ionizing luminosity for HI)
        log10_qh_specific = jnp.array(f["log10_specific_ionising_luminosity"]["HI"][:])

        # Load the reprocessed nebular spectrum (continuum + lines) on the grid's
        # native wavelength axis — the array UnifiedAGN extracts for NLR/BLR.
        # Stored per unit bolometric luminosity (see SynthesizerGridData docstring).
        spectra_wav = None
        nebular_per_lbol = None
        if "spectra" in f and "nebular" in f["spectra"]:
            spectra_wav = jnp.array(f["spectra"]["wavelength"][:])
            nebular_per_lbol = jnp.array(f["spectra"]["nebular"][:])

    # Normalize line luminosities from per-unit-bolometric to per-unit-Q_H
    # Synthesizer stores: L_line [W] per unit L_bol [W]
    # We need: L_line / Q_H [L_sun·s/photons]
    #
    # log10(L_line / Q_H) = log10(L_line) - log10(Q_H)
    #   where L_line is in W and Q_H is in photons/s
    #
    # The relationship: L_line / L_bol = line_luminosity
    #                   Q_H / L_bol = 10^(log10_qh_specific)
    #                   L_line / Q_H = line_luminosity / 10^(log10_qh_specific)
    #
    # In log10: log10(L_line / Q_H [W·s/photons]) = log10(L_line / L_bol)
    #                                                 - log10(Q_H / L_bol)
    #         = log10(line_lum) - log10_qh_specific
    #
    # Convert from W·s/photons to L_sun·s/photons:
    # L_sun = 3.828e33 erg/s = 3.828e26 W
    # L_sun·s/photon = 3.828e26 W·s/photon
    log_line_lum_per_qh_w = (
        jnp.log10(jnp.maximum(line_lum, 1e-99)) - log10_qh_specific[:, :, :, :, :, :, None]
    )
    log_L_sun = jnp.log10(L_SUN / 1e7)  # Convert erg/s to W
    log_line_per_qh = log_line_lum_per_qh_w - log_L_sun

    return SynthesizerGridData(
        mass_axis=mass_axis,
        eddington_axis=eddington_axis,
        cosine_axis=cosine_axis,
        metallicity_axis=metallicity_axis,
        logU_axis=logU_axis,
        logn_axis=logn_axis,
        line_wavelengths_aa=line_wav,
        log_line_per_qh=log_line_per_qh,
        log_qh_specific=log10_qh_specific,
        log_line_per_lbol=jnp.log10(jnp.maximum(line_lum, 1e-99)),
        line_ids=line_ids,
        spectra_wavelengths_aa=spectra_wav,
        nebular_per_lbol=nebular_per_lbol,
    )


class SynthesizerNLRBackend:
    """Synthesizer CLOUDY c23.01 AGN NLR photoionization backend.

    Computes AGN narrow-line region emission by interpolating the
    CLOUDY c23.01 photoionization grids from Synthesizer (Lovell et al. 2025; Roper et al. 2026).
    The grid covers AGN-ionized gas parameterized by BH mass, accretion rate
    (Eddington ratio), inclination angle, metallicity, ionization parameter,
    and hydrogen density.

    **Grid data required**: Synthesizer test grids at
    ``data/synthesizer_grids/test_grid_agn-nlr.hdf5`` (production grids
    require Synthesizer Box credentials).

    Interpolation strategy
    ----------------------
    All 6 axes use C²-continuous triweight interpolation via
    ``interp_nd_triweight``:

    - Internal storage in log10 (except cosine_inclination, which is linear)
    - All axes interpolated independently

    This backend has ``has_continuum = False``.

    Parameters
    ----------
    grid_path : str or Path
        Path to Synthesizer grid HDF5 file.

    Example
    -------
    >>> backend = SynthesizerNLRBackend("data/synthesizer_grids/test_grid_agn-nlr.hdf5")
    >>> wave, lum = backend.predict_agn_nlr_lines(
    ...     log_bh_mass=8.0,  # [log10(M_sun)]
    ...     log_eddington=-0.3,  # [log10(Eddington ratio)]
    ...     cosine_inclination=0.2,  # [linear, 0=edge-on]
    ...     log_metallicity=0.0,  # [log10(Z_sun)]
    ...     log_ionU=-1.5,  # [log10(U)]
    ...     log_nH=4.0,  # [log10(n_H [cm^-3])]
    ...     log_qh=53.0,  # [log10(Q_H [photons/s])]
    ... )

    References
    ----------
    Lovell et al. 2025 (doi:10.33232/001c.145766) + Roper et al. 2026 (doi:10.21105/joss.09436)

    """

    name = "synthesizer_nlr"
    has_free_params = True
    has_continuum = False

    def __init__(self, grid_path: str | Path) -> None:
        # Stored so the lazy singleton accessors can detect a grid-path change
        # on repeat calls (without it, the second call AttributeErrors).
        self.grid_path = str(grid_path)
        self.grid = _load_synthesizer_nlr_grid(grid_path)

        # Pre-compute triweight edges for all 6 axes at init time.
        # Axes must be sorted ascending for interp_nd_triweight.
        from tengri.utils.interpolation import edges_for_grid

        # Check and sort each axis
        self._mass_sorted = jnp.sort(self.grid.mass_axis)
        self._mass_descending = bool(self.grid.mass_axis[0] > self.grid.mass_axis[-1])

        self._edd_sorted = jnp.sort(self.grid.eddington_axis)
        self._edd_descending = bool(self.grid.eddington_axis[0] > self.grid.eddington_axis[-1])

        self._cos_sorted = jnp.sort(self.grid.cosine_axis)
        self._cos_descending = bool(self.grid.cosine_axis[0] > self.grid.cosine_axis[-1])

        self._met_sorted = jnp.sort(self.grid.metallicity_axis)
        self._met_descending = bool(self.grid.metallicity_axis[0] > self.grid.metallicity_axis[-1])

        self._ionU_sorted = jnp.sort(self.grid.logU_axis)
        self._ionU_descending = bool(self.grid.logU_axis[0] > self.grid.logU_axis[-1])

        self._nH_sorted = jnp.sort(self.grid.logn_axis)
        self._nH_descending = bool(self.grid.logn_axis[0] > self.grid.logn_axis[-1])

        # Precompute edges for triweight interpolation
        self._edges_mass = edges_for_grid(self._mass_sorted)
        self._edges_edd = edges_for_grid(self._edd_sorted)
        self._edges_cos = edges_for_grid(self._cos_sorted)
        self._edges_met = edges_for_grid(self._met_sorted)
        self._edges_ionU = edges_for_grid(self._ionU_sorted)
        self._edges_nH = edges_for_grid(self._nH_sorted)

    def predict_agn_nlr_lines(
        self,
        log_bh_mass: float = 8.0,
        log_eddington: float = -0.3,
        cosine_inclination: float = 0.2,
        log_metallicity: float = 0.0,
        log_ionU: float = -1.5,
        log_nH: float = 4.0,
        log_qh: float = 53.0,
        neb_fesc: float = 0.0,
        **_kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute AGN NLR emission line luminosities via grid interpolation.

        Parameters
        ----------
        log_bh_mass : float
            log10(BH mass [M_sun]).  Default 8.0.
        log_eddington : float
            log10(accretion rate / L_Eddington).  Default -0.3.
        cosine_inclination : float
            cos(inclination angle).  Linear, not log.  Default 0.2.
        log_metallicity : float
            log10(metallicity [Z_sun]).  Default 0.0 (solar).
        log_ionU : float
            log10(ionization parameter U).  Default -1.5.
        log_nH : float
            log10(hydrogen density [cm^-3]).  Default 4.0.
        log_qh : float
            log10(Q_H) ionizing photon rate [photons/s].  Default 53.0.
        neb_fesc : float
            Ionizing photon escape fraction [0, 1].  Default 0.0.
        **_kwargs
            Additional keyword arguments (ignored).

        Returns
        -------
        wavelengths : ndarray, shape (n_lines,)
            Emission line vacuum wavelengths [Angstrom].
        luminosities : ndarray, shape (n_lines,)
            Emission line luminosities [L_sun], scaled by ionizing photon
            rate and escape fraction.

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.

        Interpolation is performed on all 6 axes (mass, Eddington ratio,
        inclination, metallicity, ionU, nH) using C²-continuous triweight
        interpolation. Grid axes are stored in log10 (except cosine_inclination,
        which is linear) and must be sorted ascending for interpolation.

        """
        from tengri.utils.grid_interp import interp_nd_triweight

        grid = self.grid

        # Reverse axes if they were descending so the slice is always ascending
        log_line_slice = grid.log_line_per_qh
        if self._mass_descending:
            log_line_slice = log_line_slice[::-1, :, :, :, :, :, :]
        if self._edd_descending:
            log_line_slice = log_line_slice[:, ::-1, :, :, :, :, :]
        if self._cos_descending:
            log_line_slice = log_line_slice[:, :, ::-1, :, :, :, :]
        if self._met_descending:
            log_line_slice = log_line_slice[:, :, :, ::-1, :, :, :]
        if self._ionU_descending:
            log_line_slice = log_line_slice[:, :, :, :, ::-1, :, :]
        if self._nH_descending:
            log_line_slice = log_line_slice[:, :, :, :, :, ::-1, :]

        # Set up interpolation
        axes = (
            self._mass_sorted,
            self._edd_sorted,
            self._cos_sorted,
            self._met_sorted,
            self._ionU_sorted,
            self._nH_sorted,
        )
        edges = (
            self._edges_mass,
            self._edges_edd,
            self._edges_cos,
            self._edges_met,
            self._edges_ionU,
            self._edges_nH,
        )
        point = (log_bh_mass, log_eddington, cosine_inclination, log_metallicity, log_ionU, log_nH)

        log_line_per_qh_interp = interp_nd_triweight(log_line_slice, axes, edges, point)

        # Convert to linear luminosity and scale by Q_H and escape fraction
        # L_line = 10^(log_line_per_qh) × Q_H × (1 - fesc)  [L_sun]
        line_lum = (10.0**log_line_per_qh_interp) * (10.0**log_qh) * (1.0 - neb_fesc)

        return grid.line_wavelengths_aa, line_lum

    def interp_log_qh_specific(
        self,
        log_bh_mass: float = 8.0,
        log_eddington: float = -0.3,
        cosine_inclination: float = 0.2,
        log_metallicity: float = 0.0,
        log_ionU: float = -1.5,
        log_nH: float = 4.0,
    ) -> jnp.ndarray:
        r"""Interpolate the grid's own specific ionizing luminosity log10(Q_H / L_bol).

        This is the disc model's ionizing output baked into the grid (Q_H per unit
        bolometric luminosity, in photons/s per W). Recovering it lets a caller use
        the grid's own :math:`Q_H` normalization — the value Synthesizer itself
        uses — rather than assuming an ionizing-spectrum slope. The absolute
        :math:`\log_{10} Q_H` for a source of bolometric luminosity ``L_bol`` [erg/s]
        is ``interp_log_qh_specific(...) + log10(L_bol) - 7`` (the −7 converts
        erg/s to W).

        Parameters
        ----------
        log_bh_mass, log_eddington, cosine_inclination, log_metallicity, log_ionU, log_nH : float
            Grid coordinates (same convention as :meth:`predict_agn_nlr_lines`).

        Returns
        -------
        jnp.ndarray
            Scalar ``log10(Q_H / L_bol)`` [log10(photons/s/W)] at the point.
        """
        from tengri.utils.grid_interp import interp_nd_triweight

        grid = self.grid
        if grid.log_qh_specific is None:
            raise ValueError("grid carries no log_qh_specific; reload with the current loader")

        sl = grid.log_qh_specific
        if self._mass_descending:
            sl = sl[::-1]
        if self._edd_descending:
            sl = sl[:, ::-1]
        if self._cos_descending:
            sl = sl[:, :, ::-1]
        if self._met_descending:
            sl = sl[:, :, :, ::-1]
        if self._ionU_descending:
            sl = sl[:, :, :, :, ::-1]
        if self._nH_descending:
            sl = sl[:, :, :, :, :, ::-1]

        axes = (
            self._mass_sorted,
            self._edd_sorted,
            self._cos_sorted,
            self._met_sorted,
            self._ionU_sorted,
            self._nH_sorted,
        )
        edges = (
            self._edges_mass,
            self._edges_edd,
            self._edges_cos,
            self._edges_met,
            self._edges_ionU,
            self._edges_nH,
        )
        point = (log_bh_mass, log_eddington, cosine_inclination, log_metallicity, log_ionU, log_nH)
        return interp_nd_triweight(sl, axes, edges, point)

    def predict_agn_nebular_spectrum(
        self,
        wavelength_out: jnp.ndarray,
        l_bol_erg: float,
        covering_fraction: float = 0.1,
        log_bh_mass: float = 8.0,
        log_eddington: float = -0.3,
        log_metallicity: float = -2.0,
        log_ionU: float = -2.0,
        log_nH: float = 4.0,
    ) -> jnp.ndarray:
        r"""Reprocessed nebular :math:`L_\nu` reproducing Synthesizer's UnifiedAGN.

        Interpolates the grid's reprocessed ``/spectra/nebular`` array (continuum
        + lines) at the requested grid point — with ``cosine_inclination`` fixed
        at the grid's isotropic node (0.5), exactly as Synthesizer's UnifiedAGN
        extracts its NLR/BLR line-region emission — scales it to physical units,
        and resamples onto the caller's wavelength grid.

        Parameters
        ----------
        wavelength_out : array_like, shape (n_wave,)
            Output (rest-frame) wavelength grid [Angstrom].
        l_bol_erg : float
            Disc bolometric luminosity [erg/s].
        covering_fraction : float, optional
            Line-region covering fraction. Default 0.1.
        log_bh_mass, log_eddington, log_metallicity, log_ionU, log_nH : float
            Grid coordinates (``cosine_inclination`` is held at 0.5 internally).

        Returns
        -------
        ndarray, shape (n_wave,)
            Reprocessed nebular :math:`L_\nu` [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: yes. ``L_\nu = nebular\_per\_lbol \times L_{bol}
        \times f_{cov}`` (units verified against Synthesizer's UnifiedAGN ``nlr``
        component to ~2 %). Requires a grid carrying ``/spectra/nebular``; raises
        otherwise. Reproduces ``UnifiedAGN`` faithfully where re-broadening the
        scrambled ``/lines`` table cannot (issue #694). Implemented to match
        Synthesizer (Lovell et al. 2025; Roper et al. 2026).
        """
        from tengri.utils.grid_interp import interp_nd_triweight

        grid = self.grid
        if grid.nebular_per_lbol is None or grid.spectra_wavelengths_aa is None:
            raise ValueError(
                "grid carries no /spectra/nebular array; this grid cannot "
                "reproduce Synthesizer's UnifiedAGN NLR/BLR spectrum. Reload "
                "with the current loader or supply a grid that stores /spectra."
            )

        # Reverse leading (six parameter) axes to ascending, matching the line path.
        sl = grid.nebular_per_lbol  # (n_mass, n_edd, n_cos, n_met, n_ionU, n_nH, n_wave)
        if self._mass_descending:
            sl = sl[::-1]
        if self._edd_descending:
            sl = sl[:, ::-1]
        if self._cos_descending:
            sl = sl[:, :, ::-1]
        if self._met_descending:
            sl = sl[:, :, :, ::-1]
        if self._ionU_descending:
            sl = sl[:, :, :, :, ::-1]
        if self._nH_descending:
            sl = sl[:, :, :, :, :, ::-1]

        axes = (
            self._mass_sorted,
            self._edd_sorted,
            self._cos_sorted,
            self._met_sorted,
            self._ionU_sorted,
            self._nH_sorted,
        )
        edges = (
            self._edges_mass,
            self._edges_edd,
            self._edges_cos,
            self._edges_met,
            self._edges_ionU,
            self._edges_nH,
        )
        # cosine_inclination is held at 0.5 (Synthesizer's isotropic line-region
        # convention) so the line regions are inclination-independent.
        point = (log_bh_mass, log_eddington, 0.5, log_metallicity, log_ionU, log_nH)
        nebular_per_lbol = interp_nd_triweight(sl, axes, edges, point)  # (n_wave_grid,)

        l_nu_grid = nebular_per_lbol * l_bol_erg * covering_fraction
        return jnp.interp(
            jnp.asarray(wavelength_out),
            grid.spectra_wavelengths_aa,
            l_nu_grid,
            left=0.0,
            right=0.0,
        )


class SynthesizerBLRBackend(SynthesizerNLRBackend):
    """Synthesizer CLOUDY c23.01 AGN **broad**-line-region backend.

    The Synthesizer BLR grid shares the NLR grid's structure exactly — the same
    six axes (BH mass, Eddington ratio, cosine inclination, metallicity,
    ionization parameter, hydrogen density) and the same per-:math:`Q_H` line
    storage — differing only in the tabulated line luminosities (broad permitted
    lines from dense, high-ionization gas). So this backend reuses the NLR
    loader and interpolation wholesale; only the grid *file* and the line set
    differ. ``predict_agn_blr_lines`` is an alias of the inherited interpolation.

    **Grid data required**: ``test_grid_agn-blr.hdf5`` (downloadable test grid)
    or a production BLR grid.

    References
    ----------
    Lovell et al. 2025 (doi:10.33232/001c.145766) + Roper et al. 2026 (doi:10.21105/joss.09436)
    """

    name = "synthesizer_blr"

    def predict_agn_blr_lines(self, *args, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Alias for :meth:`SynthesizerNLRBackend.predict_agn_nlr_lines`.

        The grid interpolation is identical; only the underlying grid file (and
        therefore the line luminosities) differs. Returns ``(wavelengths [Å],
        luminosities [L_sun])``.
        """
        return self.predict_agn_nlr_lines(*args, **kwargs)


# ── Feltre+2016 NLR backend ───────────────────────────────────────


@dataclass
class FeltreGridData:
    """Container for the Feltre+2016 grid loaded from HDF5.

    Grid shape for ``logHB_per_logq`` and ``line_ratios`` leading dims:
    ``(n_alpha, n_logUs, n_logn, n_logZ, n_xi_d)``.

    Attributes
    ----------
    alpha_axis : ndarray, shape (n_alpha,)
        Ionizing EUV power-law slope values (discrete: -1.2, -1.4, -1.7, -2.0)
        [dimensionless].
    logUs_axis : ndarray, shape (n_logUs,)
        Ionization parameter log10(U_S) values [log10(U)]. May be in descending
        order.
    logn_axis : ndarray, shape (n_logn,)
        Hydrogen density log10(n_H / cm^-3) values [log10(cm^-3)].
    logZ_axis : ndarray, shape (n_logZ,)
        Absolute metallicity log10(Z) values [log10(Z_sun)].
    xi_d_axis : ndarray, shape (n_xi_d,)
        Dust-to-metal ratio values (discrete: 0.1, 0.3, 0.5)
        [dimensionless].
    line_wavelengths_aa : ndarray, shape (n_lines,)
        Emission line vacuum wavelengths [Angstrom].
    logHB_per_logq : ndarray, shape (n_alpha, n_logUs, n_logn, n_logZ, n_xi_d)
        log10(L_Hβ / Q_H) where Q_H is ionizing photon rate [photons/s]
        and L_Hβ is in erg/s [log10(erg/s·s/photons)].
    line_ratios : ndarray, shape (n_alpha, n_logUs, n_logn, n_logZ, n_xi_d, n_lines)
        Line-to-Hβ luminosity ratios L_line / L_Hβ [dimensionless].

    Notes
    -----
    Grid axes follow the Feltre et al. (2016) CLOUDY c13.03 photoionization
    calculations. Continuous axes (logUs, logn, logZ) can be interpolated
    smoothly; discrete axes (alpha, xi_d) use nearest-neighbor lookup.

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
        wavelengths : ndarray, shape (n_lines,)
            Emission line vacuum wavelengths [Angstrom].
        luminosities : ndarray, shape (n_lines,)
            Emission line luminosities [L_sun], scaled by ionizing photon
            rate and escape fraction.

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.

        Interpolation uses C²-continuous triweight on continuous axes
        (logU_S, logn, logZ) and nearest-neighbor on discrete axes
        (alpha, xi_d). The method first selects the nearest grid point for
        discrete axes, then interpolates smoothly on the 3-D continuous grid.

        """
        from tengri.utils.grid_interp import interp_nd_triweight

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


# ── Backend: analytic (existing) ──────────────────────────────────


# ── Unified dispatcher ────────────────────────────────────────────


def agn_nlr_emission(
    backend: str = "cue",
    cue_backend=None,
    feltre_backend: FeltreNLRBackend | None = None,
    synthesizer_nlr_backend: SynthesizerNLRBackend | None = None,
    l_acc_erg: float = 1e44,
    covering_fraction: float = 0.1,
    alpha_pl: float = -1.7,
    neb_logU: float = -3.0,
    # Differs from the declared gas_logn default (2.0) on purpose: that
    # declaration is the *galaxy* Cue HII-region density, while this is the AGN
    # narrow-line region, whose canonical density is ~1e3 (matching the separate
    # agn_nlr_logn declaration). Same parameter name, different physical region.
    gas_logn: float = 3.0,
    gas_logz: float = 0.0,
    gas_logno: float = 0.0,
    gas_logco: float = 0.0,
    ionspec_params: dict | None = None,
    neb_logZ_gas: float | None = None,
    xi_d: float = 0.3,
    log_qh: float = 53.0,
    neb_fesc: float = 0.0,
    log_bh_mass: float = 8.0,
    log_eddington: float = -0.3,
    cosine_inclination: float = 0.2,
    **kwargs,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Unified AGN NLR emission dispatcher.

    Routes to one of the available NLR backends depending on
    ``backend``.

    Parameters
    ----------
    backend : str
        Backend name: ``"cue"``, ``"feltre"``, or ``"synthesizer_nlr"``.
    cue_backend : CueBackend or None
        Required when ``backend="cue"``.
    feltre_backend : FeltreNLRBackend or None
        Required when ``backend="feltre"``. Initialize with
        ``FeltreNLRBackend(grid_path)`` before calling.
    synthesizer_nlr_backend : SynthesizerNLRBackend or None
        Required when ``backend="synthesizer_nlr"``. Initialize with
        ``SynthesizerNLRBackend(grid_path)`` before calling.
    l_acc_erg : float
        AGN accretion luminosity [erg s^-1]. Default 1e44.
    covering_fraction : float
        NLR covering fraction [dimensionless]. Default 0.1.
    alpha_pl : float
        AGN EUV power-law slope [dimensionless]. Default -1.7.
    neb_logU : float
        Gas ionization parameter log10(U) [log10(U)].
    gas_logn : float
        Gas density log10(n_e / cm^-3) (Cue backend) [log10(cm^-3)].
    gas_logz : float
        Gas metallicity log10(Z/Zsun) (Cue backend) [dimensionless].
    gas_logno : float
        Gas N/O offset (Cue backend) [dimensionless].
    gas_logco : float
        Gas C/O offset (Cue backend) [dimensionless].
    ionspec_params : dict or None
        Explicit Cue ionizing spectrum parameters (override alpha_pl).
        Keys: ``ionspec_index1..4``, ``ionspec_logLratio1..3`` [dimensionless].
    neb_logZ_gas : float or None
        Gas metallicity log10(Z) absolute (Feltre backend) [log10(Z)].
        If None, defaults to log10(Z_sun) = -1.8477.
    xi_d : float
        Dust-to-metal ratio (Feltre backend) [dimensionless]. Default 0.3.
    log_qh : float
        log10(Q_H) ionizing photon rate (all backends) [log10(photons/s)].
        Default 53.0.
    neb_fesc : float
        Ionizing photon escape fraction (all backends) [dimensionless].
        Default 0.0.
    log_bh_mass : float
        log10(BH mass [M_sun]) (Synthesizer backend) [log10(M_sun)].
        Default 8.0.
    log_eddington : float
        log10(accretion rate / L_Eddington) (Synthesizer backend)
        [dimensionless]. Default -0.3.
    cosine_inclination : float
        cos(inclination angle) (Synthesizer backend) [dimensionless].
        Default 0.2.
    **kwargs
        Additional keyword arguments (ignored).

    Returns
    -------
    tuple

        - line_wavelengths : ndarray, shape (n_lines,) — emission line vacuum
          wavelengths [Angstrom]
        - line_luminosities : ndarray, shape (n_lines,) — emission line
          luminosities [L_sun]

    Raises
    ------
    ValueError
        If ``backend`` is not recognized or required backend object is missing.

    References
    ----------
    .. [1] M. Li et al., "The Cue Nebular Emulator: Fast, Interpretable
       Predictions of Emission-Line Strengths from Stellar Populations,"
       ApJ, 986, 9 (2025). arXiv:2405.04598.
       https://doi.org/10.3847/1538-4357/ad7fe3
    .. [2] A. Feltre, S. Charlot, and J. Gutkin, "Updated photoionization
       models of the CLOUDY c13.03 code," MNRAS, 456, 3354 (2016).
       arXiv:1511.08217. https://doi.org/10.1093/mnras/stw2180
    .. [3] Lovell et al. 2025 (doi:10.33232/001c.145766);
           Roper et al. 2026 (doi:10.21105/joss.09436). Cite both (Synthesizer).

    Notes
    -----
    **JIT-compatible**: yes — dispatcher routes to backend-specific methods
    which are JIT-compatible. Gradient-safe for continuous parameters.

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
    elif backend == "synthesizer_nlr":
        if synthesizer_nlr_backend is None:
            raise ValueError(
                "synthesizer_nlr_backend must be provided when backend='synthesizer_nlr'. "
                "Initialize with SynthesizerNLRBackend(grid_path) where grid_path is the "
                "path to data/synthesizer_grids/test_grid_agn-nlr.hdf5."
            )
        _logZ = neb_logZ_gas if neb_logZ_gas is not None else _LOG10_ZSUN
        return synthesizer_nlr_backend.predict_agn_nlr_lines(
            log_bh_mass=log_bh_mass,
            log_eddington=log_eddington,
            cosine_inclination=cosine_inclination,
            log_metallicity=_logZ,
            log_ionU=neb_logU,
            log_nH=gas_logn,
            log_qh=log_qh,
            neb_fesc=neb_fesc,
        )
    else:
        raise ValueError(
            f"Unknown AGN NLR backend '{backend}'. "
            "Choose from: 'cue', 'feltre', 'synthesizer_nlr'."
        )
