# SPDX-License-Identifier: BSD-3-Clause
"""X-ray SED models: binaries, AGN corona, hot gas.

Predicts X-ray emission (0.1–10 keV, λ < 124 Å) from three physical components:

1. **X-ray binaries** (HMXB + LMXB): power-law + cutoff, SFR- and mass-dependent
2. **AGN corona**: power-law continuum with exponential high-energy cutoff,
   optionally tied to disc UV luminosity (self-consistent disc-corona coupling)
3. **Hot gas**: optional thermal bremsstrahlung from diffuse ISM/CGM

All functions are pure JAX, JIT-compatible, fully differentiable.

**Self-consistent disc-corona**: The function xray_agn_corona_from_disc computes
the X-ray photon index and normalization from disc UV luminosity using empirical
α_ox–L_2500 correlations (Just et al. 2007; Yang et al. 2022). This enforces
physical consistency between UV and X-ray SED components during inference.

**IRX-based X-ray (Lopez+2024)**: The function xray_agn_corona_lopez24 uses the
α_IRX parameter (12μm-to-X-ray ratio) instead of α_ox, which is more robust
for obscured and low-luminosity AGN where UV is unreliable.

**Design basis**: the same models as CIGALE's X-ray modules (Yang+2020,
Lopez+2024), implemented in JAX for gradient-based inference.
"""

import jax.numpy as jnp

from tengri._deprecated import deprecated_alias
from tengri.utils.physics_constants import (
    C_AA as _C_AA,
    H_PLANCK as _H_PLANCK,
    KEV_TO_ERG as _KEV_TO_ERG,
    KEV_TO_HZ as _KEV_TO_HZ,
)
from tengri.utils.scale import pow10 as _pow10

# Yang+2020 reference inclination for the AGN corona. The α_ox(L_2500)
# relations predict the L_2keV seen at 30°; ``xray_anisotropy`` satisfies
# f(cos 30°) = 1, so a corona evaluated at this inclination emits exactly
# the α_ox-predicted L_2keV (X-CIGALE yang20.py convention; #980).
COS_INC_REF_30DEG = 0.8660254037844387  # cos(30°)

# ── Morrison & McCammon (1983) photoelectric cross-section, Table 2 ──
# σ(E) · E³ = c0 + c1·E + c2·E² with σ in 10⁻²⁴ cm² and E in keV.
# Fit valid for 0.030 ≤ E ≤ 10 keV; above 10 keV the cross-section is
# negligible (drops faster than E⁻³) and we return transmission = 1.
_MM83_E_EDGES = jnp.array(
    [
        0.030,
        0.100,
        0.284,
        0.400,
        0.532,
        0.707,
        0.867,
        1.303,
        1.840,
        2.471,
        3.210,
        4.038,
        7.111,
        8.331,
        10.000,
    ]
)
_MM83_C0 = jnp.array(
    [
        17.3,
        34.6,
        78.1,
        71.4,
        95.5,
        308.9,
        120.6,
        141.3,
        202.7,
        342.7,
        352.2,
        433.9,
        629.0,
        701.2,
    ]
)
_MM83_C1 = jnp.array(
    [
        608.1,
        267.9,
        18.8,
        66.8,
        145.8,
        -380.6,
        169.3,
        146.8,
        104.7,
        18.7,
        18.7,
        -2.4,
        30.9,
        25.2,
    ]
)
_MM83_C2 = jnp.array(
    [
        -2150.0,
        -476.1,
        4.3,
        -51.4,
        -61.1,
        294.0,
        -47.7,
        -31.5,
        -17.0,
        0.0,
        0.0,
        0.75,
        0.0,
        0.0,
    ]
)


def tbabs_transmission(E_keV: jnp.ndarray, log_nh: float) -> jnp.ndarray:
    r"""Photoelectric absorption transmission ``T(E) = exp(−σ(E)·N_H)``.

    Implements the Morrison & McCammon (1983) ``wabs`` cross-section
    via the published polynomial fit:

    .. math::

        \sigma(E)\,E^3 = c_0 + c_1\,E + c_2\,E^2
        \quad
        (\sigma \,\,\mathrm{in}\,\,10^{-24}\,\mathrm{cm}^2,\;
         E \,\,\mathrm{in}\,\,\mathrm{keV})

    The fit is valid for 0.030 ≤ E ≤ 10 keV. Outside that range the
    cross-section is negligible (E ≫ 10 keV) or outside the X-ray band
    we model (E ≪ 0.1 keV), so transmission is set to 1 there.

    Parameters
    ----------
    E_keV : array_like, shape (n,)
        Photon energy. [keV]
    log_nh : float
        Equivalent hydrogen column density. [log10(cm⁻²)]
        Typical AGN range: 20 (unobscured) → 24 (Compton-thick).

    Returns
    -------
    ndarray, shape (n,)
        Transmission ``T(E) ∈ [0, 1]``. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — pure ``jnp`` primitives.

    **Gradient**: smooth with respect to ``log_nh`` (single ``exp``);
    bin-edge selection uses ``searchsorted`` which has zero gradient
    with respect to ``E_keV`` — adequate because ``E_keV`` is the
    wavelength grid, not a free parameter.

    **Convention**: matches XSPEC ``wabs``. The newer ``tbabs`` model
    of Wilms et al. (2000) gives 30–50 % higher cross-sections in the
    0.5–2 keV band due to updated metal abundances; we use ``wabs``
    here because its closed-form polynomial fit is exactly
    differentiable and the systematic is well below typical N_H
    posterior uncertainty.

    References
    ----------
    .. [1] R. Morrison and D. McCammon, "Interstellar photoelectric
       absorption cross-sections, 0.03–10 keV," ApJ, 270, 119 (1983),
       Table 2. https://doi.org/10.1086/161102
    .. [2] J. Wilms, A. Allen and R. McCray, "On the absorption of X-rays
       in the interstellar medium," ApJ, 542, 914 (2000). arXiv:astro-ph/0008425.
    """
    E = jnp.asarray(E_keV)
    # Only the lower bound is enforced strictly; above 10 keV we let the
    # last bin extrapolate (σ ∝ E⁻³ asymptotically, so τ → 0 quickly and
    # T → 1 naturally). A hard upper cutoff would create a spurious
    # discontinuity at exactly E = 10 keV under floating round-trip.
    in_range = E >= 0.030

    idx = jnp.clip(jnp.searchsorted(_MM83_E_EDGES, E, side="right") - 1, 0, 13)
    c0 = _MM83_C0[idx]
    c1 = _MM83_C1[idx]
    c2 = _MM83_C2[idx]
    sigma_e3 = c0 + c1 * E + c2 * E**2  # 10⁻²⁴ cm² · keV³
    sigma = sigma_e3 / jnp.maximum(E, 1e-30) ** 3 * 1e-24  # cm²

    tau = sigma * 10.0**log_nh
    return jnp.where(in_range, jnp.exp(-jnp.maximum(tau, 0.0)), 1.0)


# Thomson scattering cross-section per hydrogen atom (one free electron).
# σ_T = 6.6524587e-25 cm² (NIST 2018). XSPEC ``cabs`` applies exactly this
# as exp(−σ_T·N_H), an energy-independent attenuation that captures
# Compton down-scattering of photons out of the line of sight.
_SIGMA_THOMSON_CM2 = 6.6524587e-25


def compton_scattering_transmission(log_nh: float) -> float:
    r"""Energy-independent Compton (Thomson) attenuation ``exp(−σ_T·N_H)``.

    Matches XSPEC ``cabs``: photons removed from the line of sight by
    Thomson scattering off bound electrons in the absorber. Becomes the
    dominant suppression mechanism above the photoelectric edge once
    ``log_nh ≳ 24`` (Compton-thick regime).

    Parameters
    ----------
    log_nh : float
        Equivalent hydrogen column density. [log10(cm⁻²)]

    Returns
    -------
    float
        Transmission ``T ∈ [0, 1]``. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — single ``jnp.exp``.

    **Why this matters**: the photoelectric edge alone (Morrison &
    McCammon ``wabs``) underestimates the suppression of hard-band
    flux at log_nh ≥ 24. At log_nh = 24, σ_T·N_H ≈ 0.67, giving an
    extra factor exp(−0.67) ≈ 0.51 of attenuation that the
    photoelectric model misses entirely above ~10 keV.

    References
    ----------
    .. [1] C. Ricci et al., "The Close Environments of Accreting Massive
       Black Holes are Shaped by Radiative Feedback," Nature, 549, 488
       (2017). Underlying XSPEC spectral model used in obscured-AGN
       fits (Eq. B6 of Matsumoto+2026).
    .. [2] N. Matsumoto et al., "MIR Search for Luminous Heavily
       Obscured AGN at z > 3," ApJ submitted (2026), Appendix B.
    """
    tau_T = _SIGMA_THOMSON_CM2 * 10.0**log_nh
    return jnp.exp(-jnp.maximum(tau_T, 0.0))


# ── pexrav: cold-disc Compton reflection (Magdziarz & Zdziarski 1995) ──
# Cold disc surface column. Calibrates the photoelectric "ramp-up" of the
# reflection albedo: A(E) → 0 at E ≪ 1 keV (soft photons absorbed),
# A(E) → R at E ~ 30 keV (Compton-hump peak). N_disc = 1e24 cm⁻² gives
# τ_phabs(10 keV) ≈ 1, matching MZ95 Fig 1 where the albedo crosses 0.2
# around 10 keV. Represents the disc surface, not the LoS obscurer.
_PEXRAV_N_H_DISC_DEFAULT = 1.0e24

# Electron rest mass in keV. Sets the Klein-Nishina rolloff scale of
# the Compton kernel: g_KN(E) → 0 when E ≳ m_e c².
_M_E_C2_KEV = 511.0


def pexrav_reflection(
    wavelength: jnp.ndarray,
    l_primary: jnp.ndarray,
    R: float = 0.5,
    cos_inc: float = 0.5,
    n_h_disc: float = _PEXRAV_N_H_DISC_DEFAULT,
) -> jnp.ndarray:
    r"""Cold-disc Compton reflection (Magdziarz & Zdziarski 1995, pexrav).

    Computes the additive reflection component produced when the AGN
    corona's primary continuum reprocesses off the cold accretion disc.
    The spectral signature is the **Compton hump** peaking around
    30 keV — the feature that lets hard-X-ray surveys (NuSTAR,
    Swift/BAT) confirm an AGN even when the soft band is
    photoelectrically extinguished (Compton-thick, log N_H ≳ 24).

    Closed-form multiplicative approximation:

    .. math::

        L_\nu^{\rm refl}(E, \mu)
            = R \, \mu_{\rm fac}(\mu) \,
              g_{\rm phabs}(E) \, g_{\rm KN}(E) \,
              L_\nu^{\rm primary}(E)

    with

    .. math::

        g_{\rm phabs}(E) &= 1 - \exp\!\big[-\sigma_{\rm MM83}(E) \, N_H^{\rm disc}\big] \\
        g_{\rm KN}(E)    &= \max\!\big(\,1 - 2 x + \tfrac{5}{3} x^2,\; 0\big),
            \quad x = E / m_e c^2 \\
        \mu_{\rm fac}(\mu) &= (2 \mu + 1)/3

    Calibrated against MZ95 Fig. 1: the reflection albedo (= ratio of
    reflected to primary L_ν) rises from ~ 0 below 1 keV
    (photoelectric), peaks at A ≈ 0.45 around 30 keV for 60° viewing,
    and falls off above ~ 200 keV via Klein-Nishina. Reproduces the
    dominant spectral feature of pexrav without the full angle-dependent
    Green's-function convolution; the XSPEC ``pexrav`` would give an
    additional ~ 10–20 % shape difference in 50–100 keV that is below
    typical posterior uncertainty for SED fitting at z > 3.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    l_primary : array, shape (n_wave,)
        Primary AGN corona spectrum L_ν(E) — the unabsorbed
        ``xray_agn_corona`` output before line-of-sight obscuration.
        [erg/s/Hz]
    R : float, optional
        Reflection covering fraction Ω/2π. Range 0–2; default 0.5
        (typical of luminous local AGN, Ricci+2017; Matsumoto+2026 use
        R = 0.5 in their pexrav fits). R = 0 disables reflection.
    cos_inc : float, optional
        Cosine of disc inclination angle. Default 0.5 (≈ 60°), the
        canonical mean inclination assumed in pexrav fits. Range 0–1.
    n_h_disc : float, optional
        Cold-disc surface column density. Default 1e24 cm⁻²; should
        rarely be changed (represents the disc, not the LoS obscurer).
        [cm⁻²]

    Returns
    -------
    array, shape (n_wave,)
        Reflected spectrum L_ν^refl(E). [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — pure ``jnp`` primitives.

    **Approximation scope**: closed-form multiplicative model, not the
    full Green's-function convolution of MZ95 / XSPEC ``pexrav``.
    Reproduces the Compton-hump albedo to ~ 20 % across 5–100 keV;
    omits the 6.4 keV iron Kα line and weaker high-energy edges. For
    z ≥ 3 fits where the 6.4 keV line is redshifted to ≲ 1.6 keV and
    is unconstrained by the data, the approximation is adequate; for
    z < 1 or Fe-K studies, use a tabulated pexrav grid instead.

    **Usage**: combine additively with the primary corona, as in
    Ricci+2017 / Matsumoto+2026 Eq. B6:

    .. math::

        L_\nu^{\rm total}(E)
            = T_{\rm phabs}(E, N_H) T_{\rm cabs}(N_H) L_\nu^{\rm primary}(E)
            + L_\nu^{\rm refl}(E)
            + f_{\rm scat} L_\nu^{\rm primary}(E)

    The reflection is **not** line-of-sight absorbed (MZ95 treats the
    reflector as the absorber; the reflected photons emerge from the
    far side of the cold material with attenuation already baked into
    the Green's function).

    References
    ----------
    .. [1] P. Magdziarz and A. A. Zdziarski, "Angle-dependent Compton
       reflection of X-rays and gamma-rays," MNRAS, 273, 837 (1995).
       The XSPEC ``pexrav`` model.
       https://doi.org/10.1093/mnras/273.3.837
    .. [2] C. Ricci et al., "The Close Environments of Accreting Massive
       Black Holes are Shaped by Radiative Feedback," Nature, 549, 488
       (2017). R = 0.5 obscured-AGN spectral model adopted in
       Matsumoto+2026 Eq. B6.
    .. [3] A. P. Lightman and T. R. White, "Effects of cold matter in
       active galactic nuclei," ApJ, 335, 57 (1988). Original analytic
       cold-reflection treatment.
    """
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / _KEV_TO_ERG

    # Photoelectric vs. Compton branching ratio at the disc surface.
    # A photon hitting the cold disc is either photoelectrically
    # absorbed (lost to reflection) or Compton-scattered (reflected).
    # The reflected fraction is σ_T / (σ_T + σ_phabs(E)):
    #   * E ≪ few keV: σ_phabs ≫ σ_T → branching → 0 (photoelectric
    #     suppression, the well-known soft-band cutoff of reflection
    #     spectra).
    #   * E ~ 10 keV: σ_phabs ≈ σ_T → branching ≈ 0.5 (crossover).
    #   * E ≫ 10 keV: σ_phabs falls as E⁻³ → branching → 1.
    # Recover σ_phabs(E) from the tbabs transmission at a moderate
    # column. Use log_nh = 22 (N_H = 1e22 cm⁻²): low enough that the
    # transmission stays well above float underflow, high enough that
    # −ln T is a faithful sample of σ over the full X-ray band.
    _LOG_NH_PROBE = 22.0
    sigma_phabs = (
        -jnp.log(jnp.maximum(tbabs_transmission(E_keV, _LOG_NH_PROBE), 1e-300))
        / 10.0**_LOG_NH_PROBE
    )
    g_branching = _SIGMA_THOMSON_CM2 / (_SIGMA_THOMSON_CM2 + sigma_phabs)

    # Klein-Nishina rolloff for Compton single-scattering off cold
    # electrons. Taylor-expanded σ_KN/σ_T valid for x ≲ 1; hard cutoff
    # above x = 0.5 (E ≈ 256 keV) avoids the Taylor overshoot.
    x = E_keV / _M_E_C2_KEV
    g_kn = jnp.maximum(1.0 - 2.0 * x + (5.0 / 3.0) * x**2, 0.0)
    g_kn = jnp.where(x > 0.5, 0.0, g_kn)

    # Angle factor: (2μ+1)/3 normalizes to ~ 0.67 at the default
    # cos_inc = 0.5, matching MZ95 Fig. 1's 60° normalization.
    mu_factor = (2.0 * cos_inc + 1.0) / 3.0

    # n_h_disc is the cold disc surface column — currently absorbed
    # into the σ_phabs/σ_T branching ratio. Kept in the signature for
    # future calibration but does not affect the closed-form output;
    # log it so JAX trace-time use doesn't drop it as dead.
    _ = n_h_disc

    # X-ray band mask — reflection vanishes outside the X-ray range.
    in_band = wavelength < 124.0
    return jnp.where(in_band, R * mu_factor * g_branching * g_kn * l_primary, 0.0)


def xray_xrb_terms(
    wavelength: jnp.ndarray,
    sfr: float,
    stellar_mass: float,
    metallicity_z: float = 0.02,
    stellar_age_gyr: float = 1.0,
    gamma_hmxb: float = 2.0,
    gamma_lmxb: float = 1.6,
    E_cut: float = 100.0,
    log_L_hmxb_offset: float = 0.0,
    log_L_lmxb_offset: float = 0.0,
) -> dict[str, jnp.ndarray]:
    r"""Predict unsummed X-ray SED terms from accretion-powered binaries.

    Computes HMXB and LMXB X-ray emission as separate terms with different
    photon indices (Γ_HMXB = 2.0, Γ_LMXB = 1.6). Unlike :func:`xray_xrb`,
    returns the unsummed contributions so each can be precomputed independently
    at build time through broadband filters.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid in Å (rest-frame). [Å]
    sfr : float
        Star formation rate. [Msun/yr]
    stellar_mass : float
        Stellar mass. [Msun]
    metallicity_z : float, optional
        Metallicity (mass fraction, not log Z/Z_sun). Default: 0.02 (solar). []
    stellar_age_gyr : float, optional
        Stellar age in Gyr. Default: 1.0. [Gyr]
    gamma_hmxb : float, optional
        HMXB photon index (Γ, where F_ν ∝ ν^{−Γ}). Default: 2.0.
    gamma_lmxb : float, optional
        LMXB photon index. Default: 1.6.
    E_cut : float, optional
        Exponential cutoff energy for both populations. Default: 100 keV. [keV]
    log_L_hmxb_offset : float, optional
        Departure from mean SFR relation (dex). Default: 0.0. [dex]
    log_L_lmxb_offset : float, optional
        Departure from mean stellar-mass relation (dex). Default: 0.0. [dex]

    Returns
    -------
    dict with keys {"hmxb", "lmxb"}
        "hmxb" : ndarray, shape (n_wave,)
            High-mass X-ray binary spectral luminosity density. [erg/s/Hz]
        "lmxb" : ndarray, shape (n_wave,)
            Low-mass X-ray binary spectral luminosity density. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    **Why separate terms**: Each binary population carries a distinct photon
    index (Γ_HMXB = 2.0, Γ_LMXB = 1.6), so their sum is **not** a single
    amplitude-times-fixed-shape product. Each term separately *is* rank-1 in
    wavelength. Precomputation at build time can therefore integrate each
    through the filters independently, then sum at evaluation time. The
    summed :func:`xray_xrb` must return bit-identical results to ``xray_xrb_terms()["hmxb"]
    + xray_xrb_terms()["lmxb"]``.

    **HMXB luminosity scaling** (Lehmer et al. 2016, ApJ 825, 7, Eq. 15):
        HMXBs are young binary systems (age < 100 Myr) with massive companions,
        so their population follows the instantaneous SFR. The luminosity
        depends strongly on metallicity Z (mass fraction):

        .. math::

            \log(L_X^{\mathrm{HMXB}}(2\text{–}10\,\mathrm{keV})/\mathrm{SFR}) =
                40.28 - 62.12Z + 569.44Z^2 - 1833.80Z^3 + 1968.33Z^4
                \quad [\mathrm{erg\,s^{-1}\,(M_\odot\,yr^{-1})^{-1}}]

        At the module default Z = 0.02 this yields **1.78×10^39** erg/s per
        M_sun/yr (log = 39.251), and at the project's canonical solar
        Z = 0.0142 (Asplund 2009) it yields 3.22×10^39 — the relation is steep
        in Z, spanning 1.65×10^40 at Z = 0.001 down to 7.6×10^38 at Z = 0.03.

        This block previously claimed ≈2.6×10^39 at Z = 0.02 "consistent with
        Grimm et al. 2003". The polynomial does not give that — it is off by a
        factor of 1.46 — and Lehmer+2016 genuinely differs from
        Grimm+2003/Mineo+2012 (2.6×10^39 in this band) by ~30-45% rather than
        agreeing with them. Two crossval tests were written from that sentence
        rather than from the equation above it (#1755).

    **LMXB luminosity scaling** (Lehmer et al. 2016, ApJ 825, 7, Eq. 15):
        LMXBs are old systems (age > 1 Gyr), so their population traces
        stellar mass. The luminosity depends on stellar age t (Gyr):

        .. math::

            \log(L_X^{\mathrm{LMXB}}(2\text{–}10\,\mathrm{keV})/M_\star) =
                40.276 - 1.503\log t - 0.423(\log t)^2 + 0.425(\log t)^3 + 0.136(\log t)^4
                \quad [\mathrm{erg\,s^{-1}\,M_\odot^{-1}}]

        At t=1 Gyr, this yields ≈ 8.3×10^28 erg/s per M_sun, consistent with
        Gilfanov 2004.

    **Spectral shape**: Both HMXB and LMXB are modeled as power-laws with
    a high-energy exponential cutoff:

        .. math::

            F_\nu \propto \nu^{-\Gamma} \exp(-h\nu / E_{\mathrm{cut}})

        Each term is masked independently and separately to zero outside the
        X-ray band (E > 0.1 keV, λ < 124 Å).

    References
    ----------
    .. [1] B. D. Lehmer et al., "The evolution of the X-ray binary
       luminosity functions of nearby galaxies with the Chandra COSMOS
       survey," ApJ, 825, 7 (2016).
       https://doi.org/10.3847/0004-637X/825/1/7
    .. [2] H.-J. Grimm et al., "High-mass X-ray binaries as a star formation
       rate indicator in distant galaxies," MNRAS, 339, 793 (2003).
       https://doi.org/10.1046/j.1365-8711.2003.06224.x
    .. [3] M. Gilfanov, "Low-mass X-ray binaries as a stellar mass indicator
       for the host galaxy," MNRAS, 349, 146 (2004). arXiv:astro-ph/0309171.
       https://doi.org/10.1111/j.1365-2966.2004.07473.x
    .. [4] G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
       CIGALE and improvement of the code," ApJ, 927, 192 (2022).
       https://doi.org/10.3847/1538-4357/ac4971
    """
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / _KEV_TO_ERG  # convert to keV

    # Lehmer+2016 metallicity quartic for HMXB (yang20.py:207–214)
    # log(L_HMXB / SFR) = 33.28 - 62.12*Z + 569.44*Z^2 - 1833.80*Z^3 + 1968.33*Z^4
    # in W units. Convert to erg/s: +7.0 (log10 conversion)
    # Leading constant 40.28 = 33.28 + 7.0 makes the unit conversion explicit.
    log_l_hmxb_per_sfr = (
        40.28
        - 62.12 * metallicity_z
        + 569.44 * metallicity_z**2
        - 1833.80 * metallicity_z**3
        + 1968.33 * metallicity_z**4
    )

    # Lehmer+2014 / Yang+22 age quartic for LMXB (yang20.py:216–224).
    # Yang+22 normalizes *per 1e10 M_sun*, not per M_sun:
    #   log( L_LMXB(2-10) / (M_star/1e10 Msun) ) [W]
    #       = 33.276 - 1.503·logT - 0.423·logT² + 0.425·logT³ + 0.136·logT⁴
    # So in erg/s per Msun:
    #   L_LMXB = (M_star / 1e10) · 10^(quartic + 7)
    #          = (M_star / 1e10) · 10^40.276 · 10^(quartic_terms)
    # NOT  10^(40.276 + ...) · M_star, which was off by 10^10 (the original
    # bug surfaced by the salvaged regression tests, see
    # tests/physics/test_xray_yang22_scalings.py).
    log_t = jnp.log10(jnp.maximum(stellar_age_gyr, 1e-3))  # protect against log(0)
    log_l_lmxb_per_1e10 = (
        40.276 - 1.503 * log_t - 0.423 * log_t**2 + 0.425 * log_t**3 + 0.136 * log_t**4
    )

    # Power-law with exponential cutoff: L_nu ∝ (E/E_ref)^{-Γ+1} * exp(-E/E_cut)
    # Normalize by integrating the spectral shape over the 2-10 keV reference band
    # (not by a single-point bandwidth, which gives ~2-3x error in absolute luminosity).
    E_ref = 5.0  # keV (reference for spectral shape evaluation)

    spec_hmxb = (E_keV / E_ref) ** (-gamma_hmxb + 1) * jnp.exp(-E_keV / E_cut)
    spec_lmxb = (E_keV / E_ref) ** (-gamma_lmxb + 1) * jnp.exp(-E_keV / E_cut)

    # Compute band integral of each spectral shape over 2-10 keV on a fine grid.
    # ∫L_nu dnu = L_ref  → L_nu = L_ref * spec / ∫_band spec dnu
    E_fine = jnp.linspace(2.0, 10.0, 200)  # keV
    nu_fine = E_fine * _KEV_TO_HZ
    spec_hmxb_fine = (E_fine / E_ref) ** (-gamma_hmxb + 1) * jnp.exp(-E_fine / E_cut)
    spec_lmxb_fine = (E_fine / E_ref) ** (-gamma_lmxb + 1) * jnp.exp(-E_fine / E_cut)
    band_int_hmxb = jnp.maximum(jnp.trapezoid(spec_hmxb_fine, nu_fine), 1e-60)
    band_int_lmxb = jnp.maximum(jnp.trapezoid(spec_lmxb_fine, nu_fine), 1e-60)

    # Float32 (#1206): the XRB normalizations ``10**40.28`` (HMXB) and
    # ``10**40.276`` (LMXB) already exceed the float32 maximum (3.4e38) before
    # SFR / M_star are applied, so ``L_*_ref`` is ``inf`` and ``inf * spec`` is
    # ``nan`` wherever the spectrum underflows — although the result ~1e22
    # erg/s/Hz is representable. Fold the band integral into the exponent so no
    # out-of-range intermediate forms. Float64 keeps the literal expressions.
    if wavelength.dtype == jnp.float32:
        L_nu_hmxb = (
            _pow10(log_l_hmxb_per_sfr + log_L_hmxb_offset - jnp.log10(band_int_hmxb))
            * sfr
            * spec_hmxb
        )
        L_nu_lmxb = (
            _pow10(log_l_lmxb_per_1e10 + log_L_lmxb_offset - jnp.log10(band_int_lmxb))
            * (stellar_mass / 1.0e10)
            * spec_lmxb
        )
    else:
        L_hmxb_ref = 10.0**log_l_hmxb_per_sfr * sfr * 10.0**log_L_hmxb_offset
        L_lmxb_ref = 10.0**log_l_lmxb_per_1e10 * (stellar_mass / 1.0e10) * 10.0**log_L_lmxb_offset
        L_nu_hmxb = L_hmxb_ref / band_int_hmxb * spec_hmxb
        L_nu_lmxb = L_lmxb_ref / band_int_lmxb * spec_lmxb

    # X-ray only (E > 0.1 keV = lambda < 124 A); mask each term independently
    xray_mask = wavelength < 124.0
    return {
        "hmxb": jnp.where(xray_mask, L_nu_hmxb, 0.0),
        "lmxb": jnp.where(xray_mask, L_nu_lmxb, 0.0),
    }


def xray_xrb(
    wavelength: jnp.ndarray,
    sfr: float,
    stellar_mass: float,
    metallicity_z: float = 0.02,
    stellar_age_gyr: float = 1.0,
    gamma_hmxb: float = 2.0,
    gamma_lmxb: float = 1.6,
    E_cut: float = 100.0,
    log_L_hmxb_offset: float = 0.0,
    log_L_lmxb_offset: float = 0.0,
) -> jnp.ndarray:
    r"""Predict X-ray SED from accretion-powered binaries.

    Computes the combined X-ray emission from high-mass (HMXB) and low-mass
    (LMXB) X-ray binary populations. HMXB luminosity scales with SFR and
    metallicity (Lehmer et al. 2016). LMXB luminosity scales with stellar
    mass and age (Lehmer et al. 2016). Both are modeled as power-laws
    with exponential cutoff.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid in Å (rest-frame). [Å]
    sfr : float
        Star formation rate. [Msun/yr]
    stellar_mass : float
        Stellar mass. [Msun]
    metallicity_z : float, optional
        Metallicity (mass fraction, not log Z/Z_sun). Default: 0.02 (solar). []
    stellar_age_gyr : float, optional
        Stellar age in Gyr. Default: 1.0. [Gyr]
    gamma_hmxb : float, optional
        HMXB photon index (Γ, where F_ν ∝ ν^{−Γ}). Default: 2.0.
    gamma_lmxb : float, optional
        LMXB photon index. Default: 1.6.
    E_cut : float, optional
        Exponential cutoff energy for both populations. Default: 100 keV. [keV]
    log_L_hmxb_offset : float, optional
        Departure from mean SFR relation (dex). Allows scatter or evolution.
        Default: 0.0. [dex]
    log_L_lmxb_offset : float, optional
        Departure from mean stellar-mass relation (dex). Default: 0.0. [dex]

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density of X-ray binary populations.
        [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    This function computes the sum of HMXB and LMXB contributions.
    Use :func:`xray_xrb_terms` to access the individual unsummed terms.

    **HMXB luminosity scaling** (Lehmer et al. 2016, ApJ 825, 7, Eq. 15):
        HMXBs are young binary systems (age < 100 Myr) with massive companions,
        so their population follows the instantaneous SFR. The luminosity
        depends strongly on metallicity Z (mass fraction):

        .. math::

            \log(L_X^{\mathrm{HMXB}}(2\text{–}10\,\mathrm{keV})/\mathrm{SFR}) =
                40.28 - 62.12Z + 569.44Z^2 - 1833.80Z^3 + 1968.33Z^4
                \quad [\mathrm{erg\,s^{-1}\,(M_\odot\,yr^{-1})^{-1}}]

        At the module default Z = 0.02 this yields **1.78×10^39** erg/s per
        M_sun/yr (log = 39.251), and at the project's canonical solar
        Z = 0.0142 (Asplund 2009) it yields 3.22×10^39 — the relation is steep
        in Z, spanning 1.65×10^40 at Z = 0.001 down to 7.6×10^38 at Z = 0.03.

        This block previously claimed ≈2.6×10^39 at Z = 0.02 "consistent with
        Grimm et al. 2003". The polynomial does not give that — it is off by a
        factor of 1.46 — and Lehmer+2016 genuinely differs from
        Grimm+2003/Mineo+2012 (2.6×10^39 in this band) by ~30-45% rather than
        agreeing with them. Two crossval tests were written from that sentence
        rather than from the equation above it (#1755).

    **LMXB luminosity scaling** (Lehmer et al. 2016, ApJ 825, 7, Eq. 15):
        LMXBs are old systems (age > 1 Gyr), so their population traces
        stellar mass. The luminosity depends on stellar age t (Gyr):

        .. math::

            \log(L_X^{\mathrm{LMXB}}(2\text{–}10\,\mathrm{keV})/M_\star) =
                40.276 - 1.503\log t - 0.423(\log t)^2 + 0.425(\log t)^3 + 0.136(\log t)^4
                \quad [\mathrm{erg\,s^{-1}\,M_\odot^{-1}}]

        At t=1 Gyr, this yields ≈ 8.3×10^28 erg/s per M_sun, consistent with
        Gilfanov 2004.

    **Spectral shape**: Both HMXB and LMXB are modeled as power-laws with
    a high-energy exponential cutoff (photoelectric absorption, or intrinsic
    accretion torque limits):

        .. math::

            F_\nu \propto \nu^{-\Gamma} \exp(-h\nu / E_{\mathrm{cut}})

        Typical cutoffs: E_cut ≈ 100 keV (LMXB) to 200 keV (HMXB).
        The exponent E_cut controls the shape at high energies.

    **Wavelength coverage**: X-ray binaries emit primarily in 0.1–10 keV
    (λ ≈ 1.2 Å – 124 Å). Outside this range, flux is negligible.

    **Offsets**: The log_L_*_offset parameters allow captured intrinsic
    scatter (e.g., redshift-dependent evolution) in hierarchical models.

    References
    ----------
    .. [1] B. D. Lehmer et al., "The evolution of the X-ray binary
       luminosity functions of nearby galaxies with the Chandra COSMOS
       survey," ApJ, 825, 7 (2016).
       https://doi.org/10.3847/0004-637X/825/1/7
    .. [2] H.-J. Grimm et al., "High-mass X-ray binaries as a star formation
       rate indicator in distant galaxies," MNRAS, 339, 793 (2003).
       https://doi.org/10.1046/j.1365-8711.2003.06224.x
    .. [3] M. Gilfanov, "Low-mass X-ray binaries as a stellar mass indicator
       for the host galaxy," MNRAS, 349, 146 (2004). arXiv:astro-ph/0309171.
       https://doi.org/10.1111/j.1365-2966.2004.07473.x
    .. [4] G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
       CIGALE and improvement of the code," ApJ, 927, 192 (2022).
       https://doi.org/10.3847/1538-4357/ac4971
    """
    t = xray_xrb_terms(
        wavelength,
        sfr,
        stellar_mass,
        metallicity_z=metallicity_z,
        stellar_age_gyr=stellar_age_gyr,
        gamma_hmxb=gamma_hmxb,
        gamma_lmxb=gamma_lmxb,
        E_cut=E_cut,
        log_L_hmxb_offset=log_L_hmxb_offset,
        log_L_lmxb_offset=log_L_lmxb_offset,
    )
    return t["hmxb"] + t["lmxb"]


ALPHA_OX_RELATIONS: tuple[str, ...] = (
    "just2007",
    "lusso_risaliti_2016",
    "lusso_risaliti_2017",
)
"""Available empirical α_OX(L_2500) correlations.

* ``"just2007"`` — Just et al. 2007, ApJ 665, 1004 Eq. 3. CIGALE default
  (yang20.py:227). Derived from optically-bright AGN at low–intermediate
  luminosity.
* ``"lusso_risaliti_2016"`` — Lusso & Risaliti 2016, ApJ 819, 154 Eq. 3.
  Refit on 2685 SDSS+XMM quasars, extends to higher L_2500.
* ``"lusso_risaliti_2017"`` — Lusso & Risaliti 2017, A&A 602, A79 Eq. 2.
  High-z quasar sample, used by AGNfitter-rx.
"""

_ALPHA_OX_CALIB_RANGE: dict[str, tuple[float, float]] = {
    # Fitted log10(L_2500 / erg s^-1 Hz^-1) range for each α_OX correlation.
    # α_OX is clamped to these bounds (#861) so the anti-correlation is never
    # extrapolated into the unphysical regime where α_OX > 0 (L_2keV > L_2500).
    "just2007": (28.0, 33.0),  # Just+2007 optically-bright AGN
    "lusso_risaliti_2016": (27.0, 32.0),  # 2685 SDSS+XMM quasars
    "lusso_risaliti_2017": (27.0, 32.0),  # high-z quasar sample
}
"""Calibration ranges of the α_OX(L_2500) relations, log10(erg/s/Hz) (#861)."""


def alpha_ox_from_l2500(
    l_2500_erg_hz: float,
    relation: str = "just2007",
) -> float:
    r"""Compute alpha_ox from monochromatic 2500 A luminosity.

    Parameters
    ----------
    l_2500_erg_hz : float
        Monochromatic luminosity density at rest-frame 2500 A. [erg/s/Hz]
    relation : {"just2007", "lusso_risaliti_2016", "lusso_risaliti_2017"}
        Which empirical α_OX(L_2500) correlation to use. Default
        ``"just2007"`` matches X-CIGALE (yang20.py:227). The Lusso–Risaliti
        variants are used by AGNfitter-rx.

    Returns
    -------
    float
        Optical-to-X-ray spectral index alpha_ox, the slope between 2500 A
        and 2 keV monochromatic fluxes. Typical AGN range: -2.0 to -1.0.
        [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function. The string ``relation``
    argument is a Python-level dispatch (not traced); pass it statically.

    **Just+2007 [1]_ (Eq. 3):** derived from optically-bright AGN; valid for
    :math:`28 \lesssim \log_{10}(L_{2500}) \lesssim 33`.

    .. math::

        \alpha_{\mathrm{ox}} = -0.137 \, \log_{10}(L_{2500}) + 2.638

    **Lusso–Risaliti 2016 [2]_ (Eq. 3):** refit on 2685 SDSS+XMM quasars,
    extends usefully to high L_2500.

    .. math::

        \alpha_{\mathrm{ox}} = -0.137 \, \log_{10}(L_{2500}) + 2.594

    **Lusso–Risaliti 2017 [3]_ (Eq. 2):** high-z quasar sample, used by
    AGNfitter-rx.

    .. math::

        \alpha_{\mathrm{ox}} = -0.159 \, \log_{10}(L_{2500}) + 3.32

    More luminous AGN are X-ray weaker (steeper, more negative
    :math:`\alpha_{\mathrm{ox}}`). All three correlations agree to within
    ~0.05 at the median quasar L_2500 ≈ 10^30 erg/s/Hz and diverge by up
    to ~0.15 at the extremes.

    **Calibration-range clamp (#861).** The relations are anti-correlations, so
    :math:`\alpha_{\mathrm{ox}}` rises without bound as :math:`L_{2500}` falls
    and turns **positive** below :math:`\log_{10} L_{2500} \approx 19` — i.e.
    :math:`L_{2\,\mathrm{keV}} > L_{2500}`, an X-ray corona brighter than the
    disc that produced it, which pushes the total X-ray past :math:`L_{\rm bol}`.
    To avoid this unphysical extrapolation, :math:`\log_{10} L_{2500}` is clamped
    to each relation's fitted range (:data:`_ALPHA_OX_CALIB_RANGE`) before the
    slope is applied — below the range the boundary (faintest-calibrated)
    :math:`\alpha_{\mathrm{ox}}` is held, matching pcigale, which never
    extrapolates the relation (it takes a fixed ``alpha_ox`` bounded by
    ``max_dev_alpha_ox``).

    References
    ----------
    .. [1] Just, D. W. et al., 2007, ApJ, 665, 1004, Eq. 3.
    .. [2] Lusso, E. & Risaliti, G., 2016, ApJ, 819, 154, Eq. 3.
    .. [3] Lusso, E. & Risaliti, G., 2017, A&A, 602, A79, Eq. 2.
    """
    lo, hi = _ALPHA_OX_CALIB_RANGE.get(relation, (28.0, 33.0))
    log_l = jnp.clip(jnp.log10(l_2500_erg_hz), lo, hi)
    if relation == "just2007":
        return -0.137 * log_l + 2.638
    if relation == "lusso_risaliti_2016":
        return -0.137 * log_l + 2.594
    if relation == "lusso_risaliti_2017":
        return -0.159 * log_l + 3.32
    raise ValueError(f"Unknown alpha_ox relation {relation!r}. Choose from {ALPHA_OX_RELATIONS}.")


def xray_hotgas(
    wavelength: jnp.ndarray,
    sfr: float,
    gamma: float = 1.0,
    E_cut: float = 1.0,
) -> jnp.ndarray:
    r"""Predict X-ray SED from hot gas (diffuse ISM/CGM).

    Computes thermal X-ray emission from optically-thin hot plasma in the
    interstellar medium (ISM) and circumgalactic medium (CGM). The emission
    scales with SFR and is modeled as thermal bremsstrahlung.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Å (rest-frame). [Å]
    sfr : float
        Star formation rate. [Msun/yr]
    gamma : float, optional
        Photon index (Γ, where F_ν ∝ ν^{−Γ}). Default: 1.0 (thermal).
    E_cut : float, optional
        Exponential cutoff energy. Default: 1.0 keV (hot gas characteristic). [keV]

    Returns
    -------
    array, shape (n_wave,)
        Spectral luminosity density of hot gas X-ray emission.
        [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    **Hot gas luminosity scaling** (Yang et al. 2020, MNRAS 491, 740;
    Yang et al. 2022, ApJ 927, 192; Mineo et al. 2012, ApJ 745, 181):
        Hot gas X-ray emission scales with SFR because star-forming regions
        heat the ISM through supernovae and winds:

        .. math::

            \log(L_X^{\mathrm{hot\,gas}}(0.5\text{–}2\,\mathrm{keV})/\mathrm{SFR}) = 38.9
                \quad [\mathrm{erg\,s^{-1}\,(M_\odot\,yr^{-1})^{-1}}]

        This gives L_X^{hot gas} ≈ 7.94×10^38 erg/s per M_sun/yr SFR.

    **Spectral shape**: Hot gas is modeled as thermal bremsstrahlung from
    optically-thin plasma (Γ = 1; free-free and free-bound emission):

        .. math::

            F_\nu \propto \nu^{-\Gamma} \exp(-h\nu / E_{\mathrm{cut}})

        Typical cutoff: E_cut ≈ 1 keV (plasma temperature ~ 10^7 K).

    **Wavelength coverage**: Hot gas emits in soft X-rays (0.5–2 keV,
        λ ≈ 6–124 Å). Outside this range, flux is negligible.

    References
    ----------
    .. [1] G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
       CIGALE and improvement of the code," MNRAS, 491, 740 (2020).
       https://doi.org/10.1093/mnras/stz3001
    .. [2] G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
       CIGALE and improvement of the code," ApJ, 927, 192 (2022).
       https://doi.org/10.3847/1538-4357/ac4971
    .. [3] S. Mineo et al., "The hot and energetic universe: The X-ray binary
       populations in normal galaxies," ApJ, 745, 181 (2012).
       https://doi.org/10.1088/0004-637X/745/2/181
    """
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / _KEV_TO_ERG

    # Hot gas luminosity scaling (yang20.py:204)
    # L_0.5-2keV = 8.3e31 W * SFR. In erg/s: 8.3e38 = 10^38.919.
    # (yang20.py:204 shows L_hotgas_0p5to2keV = 8.3e31 * sfr)
    log_l_hotgas_per_sfr = 38.919

    # Thermal bremsstrahlung spectrum with exponential cutoff
    E_ref = 1.0  # keV (characteristic hot-gas energy)
    spec = (E_keV / E_ref) ** (-gamma + 1) * jnp.exp(-E_keV / E_cut)

    # Normalize by integrating spectral shape over 0.5-2 keV
    E_fine = jnp.linspace(0.5, 2.0, 200)  # keV
    nu_fine = E_fine * _KEV_TO_HZ
    spec_fine = (E_fine / E_ref) ** (-gamma + 1) * jnp.exp(-E_fine / E_cut)
    band_int = jnp.maximum(jnp.trapezoid(spec_fine, nu_fine), 1e-60)

    # Float32 (#1206): ``10**38.919 = 8.3e38`` already exceeds the float32
    # maximum (3.4e38) BEFORE ``sfr`` is applied, so ``L_hotgas_ref`` is ``inf``
    # and ``inf * spec`` is ``nan`` wherever the spectrum underflows to zero —
    # even though the result ``L_nu`` (~1e26 erg/s/Hz) is perfectly
    # representable. Divide the constant by the band integral in log space so no
    # out-of-range intermediate forms. Float64 keeps the literal expression.
    if wavelength.dtype == jnp.float32:
        L_nu = _pow10(log_l_hotgas_per_sfr - jnp.log10(band_int)) * sfr * spec
    else:
        L_hotgas_ref = 10.0**log_l_hotgas_per_sfr * sfr
        L_nu = L_hotgas_ref / band_int * spec

    # Soft X-ray only (0.5-2 keV => 6-124 A)
    # but allow slightly beyond for smoothness
    xray_mask = wavelength < 124.0
    return jnp.where(xray_mask, L_nu, 0.0)


def xray_anisotropy(
    l_x: jnp.ndarray,
    cos_inc: float,
    a1: float = 0.5,
    a2: float = 0.0,
) -> jnp.ndarray:
    r"""Apply viewing-angle anisotropy to X-ray luminosity (Yang+2022).

    Parameters
    ----------
    l_x : array_like, shape (n_wave,)
        Corona luminosity spectrum at the Yang+2020 30° reference
        inclination — i.e. the α_ox-predicted spectrum. [erg/s/Hz]
    cos_inc : float
        Cosine of inclination angle (1 = face-on, 0 = edge-on).
        [dimensionless]
    a1 : float, optional
        Linear anisotropy coefficient. Default 0.5. [dimensionless]
    a2 : float, optional
        Quadratic anisotropy coefficient. Default 0.0. [dimensionless]

    Returns
    -------
    ndarray, shape (n_wave,)
        Anisotropy-corrected L_X. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.

    **Empirical correction** (Yang et al. 2022 [1]_): polynomial in
    :math:`\mu \equiv \cos\theta`, anchored at the 30° reference
    inclination where the α_ox(L_2500) relations are defined.

    The anisotropic luminosity is computed as (X-CIGALE yang20.py:231–235):

    .. math::

        f(\mu) = \frac{a_1\,\mu + a_2\,\mu^2 + (1 - a_1 - a_2)}{1 - 0.13397\,a_1 - 0.25\,a_2},
        \qquad
        L_X^{\rm obs} = f(\mu)\, L_X^{(30^\circ)}

    The denominator is the numerator evaluated at :math:`\mu = \cos 30^\circ`
    (:math:`0.13397 = 1 - \cos 30^\circ`, :math:`0.25 = 1 - \cos^2 30^\circ`),
    so :math:`f(\cos 30^\circ) = 1`: the input spectrum is interpreted as the
    30° (α_ox-anchored) corona, matching CIGALE's ``*_30deg`` bookkeeping.
    Face-on (:math:`\mu = 1`) is *brighter* than the anchor —
    :math:`f(1) \approx 1.072` at the default :math:`a_1 = 0.5,\, a_2 = 0`
    (the "intermediate" obscuration solution adopted in X-CIGALE). See #980
    for the parity audit that pinned this convention against CIGALE 2025.1.

    References
    ----------
    .. [1] Yang, G. et al., 2022, ApJ, 927, 192.
       CIGALE implementation: yang20.py:231–235.
    """
    numerator = a1 * cos_inc + a2 * cos_inc**2 + (1.0 - a1 - a2)
    # Denominator = numerator at μ = cos 30° (yang20.py:231–235): anchors
    # f(cos 30°) = 1 so the α_ox-derived L_2keV is the 30° value (#980).
    denominator = 1.0 - 0.13397 * a1 - 0.25 * a2
    factor = numerator / denominator
    return l_x * factor


def xray_agn_corona_from_disc(
    wavelength: jnp.ndarray,
    l_2500_erg_hz: float,
    cos_inc: float = COS_INC_REF_30DEG,
    delta_alpha_ox: float = 0.0,
    gamma: float = 1.8,
    E_cut: float = 300.0,
    apply_anisotropy: bool = True,
    a1: float = 0.5,
    a2: float = 0.0,
    log_nh: float = 20.0,
    alpha_ox_relation: str = "just2007",
    pexrav_R: float = 0.0,
) -> jnp.ndarray:
    """Self-consistent AGN corona emission derived from disc UV luminosity.

    Computes alpha_ox from L_2500 via an empirical correlation (Just+2007
    by default; Lusso–Risaliti 2016/2017 selectable), derives L_2keV, builds
    the X-ray power-law spectrum, and optionally applies viewing-angle
    anisotropy (Yang+2022).

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength [Angstrom].
    l_2500_erg_hz : float
        Monochromatic luminosity density at 2500 A [erg/s/Hz].
    cos_inc : float
        Cosine of inclination (1 = face-on, 0 = edge-on). Default
        ``COS_INC_REF_30DEG`` — the Yang+2020 anchor where the anisotropy
        factor is exactly 1 (#980).
    delta_alpha_ox : float
        Additive offset to the Just+2007 alpha_ox. Default 0.0.
    gamma : float
        Photon index. Default 1.8. Range: 1.4-2.4.
    E_cut : float
        Exponential cutoff energy [keV]. Default 300.
    apply_anisotropy : bool
        Whether to apply Yang+2022 viewing-angle correction.
    a1 : float
        Linear anisotropy coefficient. Default 0.5.
    a2 : float
        Quadratic anisotropy coefficient. Default 0.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.
    """
    # alpha_ox from disc UV luminosity via the selected empirical correlation.
    # NaN guard for the L_2500=0 fallback (no AGN upstream): log10(0)=-inf
    # would propagate as 0 * inf = NaN through L_2keV below. Compute α_ox
    # from a tiny positive floor; the spectrum is masked back to 0 at the
    # end when l_2500_erg_hz ≤ 0.
    safe_l_2500 = jnp.maximum(l_2500_erg_hz, 1e-300)
    alpha_ox = alpha_ox_from_l2500(safe_l_2500, relation=alpha_ox_relation) + delta_alpha_ox

    # Derive L_2keV from alpha_ox definition (yang20.py:227):
    #   alpha_ox = 0.3838 * log10(L_2keV / L_2500)
    #   => L_2keV = L_2500 * 10^(alpha_ox / 0.3838)
    # The divisor 0.3838 = 1 / log10(nu_2keV / nu_2500A) is the exact
    # frequency ratio between 2 keV (λ ≈ 6.2 Å) and 2500 Å.
    l_2kev_erg_hz = safe_l_2500 * 10.0 ** (alpha_ox / 0.3838)

    # Build power-law spectrum with exponential cutoff
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / 1.6022e-9  # convert to keV
    E_ref = 2.0  # keV
    spec = (E_keV / E_ref) ** (-gamma + 1) * jnp.exp(-E_keV / E_cut)

    # Normalize at 2 keV. ``l_2kev_erg_hz`` is already L_nu(2 keV) in erg/s/Hz
    # (alpha_ox is defined on monochromatic L_nu values, Tananbaum+1979), so
    # multiplying by the dimensionless ``spec`` (=1 at E=E_ref) gives L_nu(E).
    l_nu = l_2kev_erg_hz * spec

    # Ricci+2017 / Matsumoto+2026 Eq. B6: photoelectric + Compton
    # scattering applied to the primary continuum, plus a small
    # constant scattered fraction (defaulting to 1 %).
    l_intr = l_nu
    l_nu = (
        tbabs_transmission(E_keV, log_nh) * compton_scattering_transmission(log_nh) * l_intr
        + 0.01 * l_intr
    )

    # Cold-disc Compton reflection (Magdziarz & Zdziarski 1995 pexrav).
    # Additive on top of the absorbed primary + scattered terms, using
    # the *unabsorbed* intrinsic continuum as the seed (MZ95 treats the
    # reflector as the absorber; reflection is not LoS-attenuated). The
    # contribution is gated by pexrav_R; default 0.0 = disabled.
    l_nu = l_nu + pexrav_reflection(wavelength, l_intr, R=pexrav_R, cos_inc=cos_inc)

    # X-ray mask (E > 0.1 keV => lambda < 124 A) AND mask to zero when
    # there is no AGN upstream (L_2500 ≤ 0). The safe_l_2500 floor above
    # keeps the math finite; this mask reverts the floor's effect on the
    # final spectrum so consumers see exactly zero, not 10^-300 noise.
    has_agn = l_2500_erg_hz > 0.0
    l_nu = jnp.where((wavelength < 124.0) & has_agn, l_nu, 0.0)

    # Optional anisotropy correction
    if apply_anisotropy:
        l_nu = xray_anisotropy(l_nu, cos_inc, a1=a1, a2=a2)

    return l_nu


def xray_agn_corona(
    wavelength: jnp.ndarray,
    l_2500_30deg_erg_hz: float,
    gamma: float = 1.8,
    E_cut: float = 300.0,
    delta_alpha_ox: float = 0.0,
    cos_inc: float = COS_INC_REF_30DEG,
    apply_anisotropy: bool = True,
    a1: float = 0.5,
    a2: float = 0.0,
    log_nh: float = 20.0,
    alpha_ox_relation: str = "just2007",
    pexrav_R: float = 0.0,
) -> jnp.ndarray:
    r"""X-ray emission from AGN corona (CIGALE / Yang+2020 canonical path).

    Self-consistent AGN corona emission derived from the disc UV luminosity
    at 30° intrinsic angle. This is the canonical path matching X-CIGALE
    (Yang+2020) and enforces physical consistency between UV and X-ray SEDs.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Å (rest-frame). [Å]
    l_2500_30deg_erg_hz : float
        Monochromatic luminosity density at 2500 Å from the AGN disc
        at 30° inclination angle (intrinsic). [erg/s/Hz]
    gamma : float, optional
        Photon index (Γ, where F_ν ∝ ν^{−Γ}). Default: 1.8. Range: 1.4–2.4.
    E_cut : float, optional
        Exponential cutoff energy. Default: 300 keV. [keV]
    delta_alpha_ox : float, optional
        Additive offset to the Just+2007 α_ox relation. Default: 0.0. [dex]
    cos_inc : float, optional
        Cosine of inclination angle (1 = face-on, 0 = edge-on). Default:
        ``COS_INC_REF_30DEG`` — the Yang+2020 anchor, factor exactly 1 (#980). []
    apply_anisotropy : bool, optional
        Whether to apply Yang+2022 viewing-angle correction. Default: True.
    a1 : float, optional
        Linear anisotropy coefficient. Default: 0.5. []
    a2 : float, optional
        Quadratic anisotropy coefficient. Default: 0.0. []

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.

    **Physical basis** (Yang et al. 2020, MNRAS 491, 740, §2.2.1):
    The α_OX parameter (defined as the SED slope between 2500 Å and 2 keV)
    is an observationally-calibrated relation from Just et al. (2007):

    .. math::

        \alpha_{\mathrm{OX}} = -0.137 \log_{10}(L_{2500}/[\mathrm{erg\,s^{-1}\,Hz^{-1}}])
            + 2.638

    The 2–10 keV luminosity then follows from the definition:

    .. math::

        \alpha_{\mathrm{OX}} = 0.3838 \log_{10}(L_{2\,\mathrm{keV}}/L_{2500})
            \quad \Rightarrow \quad
        L_{2\,\mathrm{keV}} = L_{2500} \times 10^{\alpha_{\mathrm{OX}}/0.3838}

    This driving from L_2500 (rather than L_bol) ensures consistency with
    the AGN disc model and avoids the ambiguity of bolometric corrections.

    **Anisotropy** (Yang et al. 2022, ApJ 927, 192):
    When apply_anisotropy=True, an inclination-dependent factor is applied:

    .. math::

        L_X(\theta) = L_X(0°) \times [a_1\cos\theta + a_2\cos^2\theta
            + (1 - a_1 - a_2)]

    Default (a1=0.5, a2=0.0) matches X-CIGALE's "intermediate" solution.

    References
    ----------
    .. [1] D. W. Just et al., "The X-ray luminosity and morphology
       dependence of narrow emission-line regions in nearby active galactic
       nuclei," ApJ, 665, 1004 (2007).
       https://doi.org/10.1086/519990
    .. [2] G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
       CIGALE and improvement of the code," MNRAS, 491, 740 (2020).
       https://doi.org/10.1093/mnras/stz3001
    .. [3] G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
       CIGALE and improvement of the code," ApJ, 927, 192 (2022).
       https://doi.org/10.3847/1538-4357/ac4971
    """
    return xray_agn_corona_from_disc(
        wavelength,
        l_2500_30deg_erg_hz,
        cos_inc=cos_inc,
        delta_alpha_ox=delta_alpha_ox,
        gamma=gamma,
        E_cut=E_cut,
        apply_anisotropy=apply_anisotropy,
        a1=a1,
        a2=a2,
        log_nh=log_nh,
        alpha_ox_relation=alpha_ox_relation,
        pexrav_R=pexrav_R,
    )


def _xray_agn_corona_bolometric(
    wavelength: jnp.ndarray,
    L_agn_bol: float,
    gamma: float = 1.8,
    E_cut: float = 300.0,
    alpha_ox: float = -1.4,
    log_nh: float = 20.0,
    scattered_frac: float = 0.01,
) -> jnp.ndarray:
    r"""**DEPRECATED**: AGN corona from bolometric luminosity (with N_H absorption).

    Use :func:`xray_agn_corona_from_disc` (which takes ``L_2500_30deg``
    directly) instead — that is the X-CIGALE-faithful path (Yang+2020
    yang20.py:227). This function converts from ``L_bol`` via the
    Hopkins+2007 bolometric correction (BC_2500 ≈ 5.15), which is
    ambiguous and inconsistent with the disc UV model.

    The N_H absorption + scattered-flux machinery from PR #325 is
    preserved here so that legacy callers keep working until the
    deprecation is removed:

    .. math::

        L_\nu(E) =
            T_{\rm phabs}(E, N_H)\,
            T_{\rm cabs}(N_H)\,
            L_\nu^{\rm intr}(E)
            + f_{\rm scat}\, L_\nu^{\rm intr}(E)

    where the line-of-sight absorber attenuates the primary continuum
    through photoelectric absorption (:func:`tbabs_transmission`) and
    Compton down-scattering (:func:`compton_scattering_transmission`),
    while a constant fraction ``scattered_frac`` of the intrinsic
    spectrum reaches the observer via warm-electron scattering on
    scales beyond the obscurer (Ricci+2017; Matsumoto+2026 Eq. B6).

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength [Angstrom].
    L_agn_bol : float
        AGN bolometric luminosity [erg/s].
    gamma : float
        Photon index. Default 1.8. Range: 1.4-2.4.
    E_cut : float
        Cutoff energy [keV]. Default 300.
    alpha_ox : float
        UV-to-X-ray slope. Default -1.4. Range: -2.0 to -1.0.
    log_nh : float
        Line-of-sight equivalent hydrogen column density. [log10(cm⁻²)]
        Default 20.0 (unobscured). Range 20.0 – 26.0.
    scattered_frac : float
        Fraction of the intrinsic continuum reaching the observer via
        warm-electron scattering on scales beyond the obscurer. [dimensionless]
        Default 0.01 (Ricci+2017, typical for type-2 AGN). Range 0.0 – 0.1.
        Set to 0 to disable.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz].
    """
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / (1.6022e-9)

    # Monochromatic luminosity density at 2500 A in erg/s/Hz.
    # L_bol = BC_2500 * nu_2500 * L_nu(2500) => L_nu = L_bol / (BC * nu)
    # BC_2500 ~ 5.15 (Hopkins+2007), nu_2500 = c / 2500 A = 1.199e15 Hz
    _NU_2500 = 1.199e15  # Hz
    _BC_2500 = 5.15  # Hopkins+2007 bolometric correction at 2500 A
    L_2500 = L_agn_bol / (_BC_2500 * _NU_2500)  # erg/s/Hz

    # alpha_ox = 0.3838 * log10(L_2keV / L_2500A)
    # => L_2keV = L_2500 * 10^(alpha_ox / 0.3838)
    L_2keV = L_2500 * 10.0 ** (alpha_ox / 0.3838)

    # Power-law spectrum
    E_ref = 2.0  # keV
    spec = (E_keV / E_ref) ** (-gamma + 1) * jnp.exp(-E_keV / E_cut)

    # Normalize at 2 keV. ``L_2keV`` is already L_nu(2 keV) in erg/s/Hz
    # (alpha_ox is defined on monochromatic L_nu values, Tananbaum+1979), so
    # multiplying by the dimensionless ``spec`` (=1 at E=E_ref) gives L_nu(E).
    L_intr = L_2keV * spec

    # Ricci+2017 / Matsumoto+2026 Eq. B6:
    #   primary = zphabs(N_H) × cabs(N_H) × intrinsic
    #   scattered = scattered_frac × intrinsic
    # Applied to the AGN corona only — XRBs and hot gas are outside the
    # torus line of sight and are unobscured by host N_H.
    T_phabs = tbabs_transmission(E_keV, log_nh)
    T_cabs = compton_scattering_transmission(log_nh)
    L_nu = T_phabs * T_cabs * L_intr + scattered_frac * L_intr

    xray_mask = wavelength < 124.0
    return jnp.where(xray_mask, L_nu, 0.0)


def xray_total_terms(
    wavelength: jnp.ndarray,
    sfr: float = 1.0,
    stellar_mass: float = 1e10,
    metallicity_z: float = 0.02,
    stellar_age_gyr: float = 1.0,
    l_2500_30deg: float = 0.0,
    gamma_hmxb: float = 2.0,
    gamma_lmxb: float = 1.6,
    gamma_agn: float = 1.8,
    E_cut: float = 300.0,
    delta_alpha_ox: float = 0.0,
    cos_inc: float = COS_INC_REF_30DEG,
    apply_anisotropy: bool = True,
    a1: float = 0.5,
    a2: float = 0.0,
    log_nh: float = 20.0,
    alpha_ox_relation: str = "just2007",
    pexrav_R: float = 0.0,
    log_L_hmxb_offset: float = 0.0,
    log_L_lmxb_offset: float = 0.0,
    **_kwargs,
) -> dict[str, jnp.ndarray]:
    """Unsummed X-ray SED terms: HMXB, LMXB, hot gas, AGN.

    Computes all four X-ray components as separate unsummed terms, enabling
    independent precomputation through broadband filters at build time. Unlike
    :func:`xray_total`, returns a dictionary with keys ``"hmxb"``, ``"lmxb"``,
    ``"hotgas"``, and ``"agn"``, each as a rank-1 spectral shape times amplitude.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength [Angstrom].
    sfr : float
        Star formation rate [Msun/yr]. Default: 1.0.
    stellar_mass : float
        Stellar mass [Msun]. Default: 1e10.
    metallicity_z : float
        Metallicity (mass fraction). Default: 0.02 (solar). []
    stellar_age_gyr : float
        Stellar age in Gyr. Default: 1.0. [Gyr]
    l_2500_30deg : float
        AGN monochromatic luminosity at 2500 Å at 30° inclination [erg/s/Hz].
        Default: 0.0 (no AGN X-ray).
    gamma_hmxb : float
        HMXB photon index. Default: 2.0.
    gamma_lmxb : float
        LMXB photon index. Default: 1.6.
    gamma_agn : float
        AGN X-ray photon index. Default: 1.8.
    E_cut : float
        Exponential cutoff energy [keV]. Default: 300.
    delta_alpha_ox : float
        Additive offset to Just+2007 α_ox relation [dex]. Default: 0.0.
    cos_inc : float
        Cosine of inclination angle (1 = face-on, 0 = edge-on). Default:
        ``COS_INC_REF_30DEG`` — the Yang+2020 anchor, factor exactly 1 (#980). []
    apply_anisotropy : bool
        Whether to apply Yang+2022 viewing-angle correction. Default: True.
    a1 : float
        Linear anisotropy coefficient. Default: 0.5. []
    a2 : float
        Quadratic anisotropy coefficient. Default: 0.0. []
    log_nh : float
        Line-of-sight equivalent hydrogen column density [log10(cm⁻²)].
        Default: 20.0. Range: 20.0–26.0.
    alpha_ox_relation : str
        Empirical α_OX relation. Default: "just2007". Options: "just2007",
        "lusso_risaliti_2016", "lusso_risaliti_2017".
    pexrav_R : float
        Cold-disc Compton reflection covering fraction. Default: 0.0 (disabled).
        [dimensionless]
    log_L_hmxb_offset : float
        Departure from expected HMXB log L_X [dex]. Default: 0.0. [dex]
    log_L_lmxb_offset : float
        Departure from expected LMXB log L_X [dex]. Default: 0.0. [dex]

    Returns
    -------
    dict with keys {"hmxb", "lmxb", "hotgas", "agn"}
        Each value is ndarray, shape (n_wave,), units [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.

    **Why separate terms**: HMXB and LMXB carry distinct photon indices
    (Γ_HMXB = 2.0, Γ_LMXB = 1.6), so their sum is not a single
    amplitude-times-fixed-shape product. Hot gas and AGN have different
    dependencies on physical parameters and spectral shapes. By returning
    unsummed terms, precompute mechanisms can integrate each through filters
    independently at build time, then sum at evaluation. The summed
    :func:`xray_total` must return bit-identical results to summing all
    four values from this dict.

    **Components**:

    - HMXB: Lehmer+2016 metallicity quartic, scaling with SFR
    - LMXB: Lehmer+2016 age quartic, scaling with M_star
    - Hot gas: Yang+2020, scaling with SFR
    - AGN corona: Just+2007 / Yang+2020, scaling with L_2500 and α_OX

    **XRB offsets** (``log_L_hmxb_offset``, ``log_L_lmxb_offset``):
    Multiplicative offsets in log space allowing intrinsic scatter or
    redshift-dependent evolution around the Lehmer+2016 empirical relations.
    """
    xrb_terms = xray_xrb_terms(
        wavelength,
        sfr,
        stellar_mass,
        metallicity_z=metallicity_z,
        stellar_age_gyr=stellar_age_gyr,
        gamma_hmxb=gamma_hmxb,
        gamma_lmxb=gamma_lmxb,
        E_cut=E_cut,
        log_L_hmxb_offset=log_L_hmxb_offset,
        log_L_lmxb_offset=log_L_lmxb_offset,
    )
    hotgas = xray_hotgas(wavelength, sfr, gamma=1.0, E_cut=1.0)
    # AGN corona path uses the X-CIGALE driver (L_2500_30deg) with the
    # PR #325 line-of-sight absorber (log_nh -> tbabs × cabs + scattered floor).
    agn = xray_agn_corona(
        wavelength,
        l_2500_30deg,
        gamma=gamma_agn,
        E_cut=E_cut,
        delta_alpha_ox=delta_alpha_ox,
        cos_inc=cos_inc,
        apply_anisotropy=apply_anisotropy,
        a1=a1,
        a2=a2,
        log_nh=log_nh,
        alpha_ox_relation=alpha_ox_relation,
        pexrav_R=pexrav_R,
    )
    return {
        "hmxb": xrb_terms["hmxb"],
        "lmxb": xrb_terms["lmxb"],
        "hotgas": hotgas,
        "agn": agn,
    }


def xray_total(
    wavelength: jnp.ndarray,
    sfr: float = 1.0,
    stellar_mass: float = 1e10,
    metallicity_z: float = 0.02,
    stellar_age_gyr: float = 1.0,
    l_2500_30deg: float = 0.0,
    gamma_hmxb: float = 2.0,
    gamma_lmxb: float = 1.6,
    gamma_agn: float = 1.8,
    E_cut: float = 300.0,
    delta_alpha_ox: float = 0.0,
    cos_inc: float = COS_INC_REF_30DEG,
    apply_anisotropy: bool = True,
    a1: float = 0.5,
    a2: float = 0.0,
    log_nh: float = 20.0,
    alpha_ox_relation: str = "just2007",
    pexrav_R: float = 0.0,
    log_L_hmxb_offset: float = 0.0,
    log_L_lmxb_offset: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Total X-ray emission (HMXB + LMXB + hot gas + AGN corona).

    Combines X-ray binaries (HMXB and LMXB), diffuse hot gas, and AGN
    corona emission into a single SED. Implements Lehmer et al. 2016
    (metallicity and age-dependent XRB scaling), Yang et al. 2020 (hot gas),
    and Just et al. 2007 / Yang et al. 2020 (AGN corona).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength [Angstrom].
    sfr : float
        Star formation rate [Msun/yr]. Default: 1.0.
    stellar_mass : float
        Stellar mass [Msun]. Default: 1e10.
    metallicity_z : float
        Metallicity (mass fraction). Default: 0.02 (solar). []
    stellar_age_gyr : float
        Stellar age in Gyr. Default: 1.0. [Gyr]
    l_2500_30deg : float
        AGN monochromatic luminosity at 2500 Å at 30° inclination [erg/s/Hz].
        Default: 0.0 (no AGN X-ray).
    gamma_hmxb : float
        HMXB photon index. Default: 2.0.
    gamma_lmxb : float
        LMXB photon index. Default: 1.6.
    gamma_agn : float
        AGN X-ray photon index. Default: 1.8.
    E_cut : float
        Exponential cutoff energy [keV]. Default: 300.
    delta_alpha_ox : float
        Additive offset to Just+2007 α_ox relation [dex]. Default: 0.0.
    cos_inc : float
        Cosine of inclination angle (1 = face-on, 0 = edge-on). Default:
        ``COS_INC_REF_30DEG`` — the Yang+2020 anchor, factor exactly 1 (#980). []
    apply_anisotropy : bool
        Whether to apply Yang+2022 viewing-angle correction. Default: True.
    a1 : float
        Linear anisotropy coefficient. Default: 0.5. []
    a2 : float
        Quadratic anisotropy coefficient. Default: 0.0. []
    log_nh : float
        Line-of-sight equivalent hydrogen column density [log10(cm⁻²)].
        Default: 20.0. Range: 20.0–26.0.
    alpha_ox_relation : str
        Empirical α_OX relation. Default: "just2007". Options: "just2007",
        "lusso_risaliti_2016", "lusso_risaliti_2017".
    pexrav_R : float
        Cold-disc Compton reflection covering fraction. Default: 0.0 (disabled).
        [dimensionless]
    log_L_hmxb_offset : float
        Departure from expected HMXB log L_X [dex]. Default: 0.0. [dex]
    log_L_lmxb_offset : float
        Departure from expected LMXB log L_X [dex]. Default: 0.0. [dex]

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.

    This function returns the sum of all four X-ray components. Use
    :func:`xray_total_terms` to access the individual unsummed terms for
    precomputation.

    **Components**:

    - HMXB: Lehmer+2016 metallicity quartic, scaling with SFR
    - LMXB: Lehmer+2016 age quartic, scaling with M_star
    - Hot gas: Yang+2020, scaling with SFR
    - AGN corona: Just+2007 / Yang+2020, scaling with L_2500 and α_OX

    **XRB offsets** (``log_L_hmxb_offset``, ``log_L_lmxb_offset``):
    Multiplicative offsets in log space allowing intrinsic scatter or
    redshift-dependent evolution around the Lehmer+2016 empirical relations.
    Implemented as additive terms in the log luminosity (Yang+2020 [1]_).
    """
    t = xray_total_terms(
        wavelength,
        sfr=sfr,
        stellar_mass=stellar_mass,
        metallicity_z=metallicity_z,
        stellar_age_gyr=stellar_age_gyr,
        l_2500_30deg=l_2500_30deg,
        gamma_hmxb=gamma_hmxb,
        gamma_lmxb=gamma_lmxb,
        gamma_agn=gamma_agn,
        E_cut=E_cut,
        delta_alpha_ox=delta_alpha_ox,
        cos_inc=cos_inc,
        apply_anisotropy=apply_anisotropy,
        a1=a1,
        a2=a2,
        log_nh=log_nh,
        alpha_ox_relation=alpha_ox_relation,
        pexrav_R=pexrav_R,
        log_L_hmxb_offset=log_L_hmxb_offset,
        log_L_lmxb_offset=log_L_lmxb_offset,
    )
    return t["hmxb"] + t["lmxb"] + t["hotgas"] + t["agn"]


# ── Lopez+2024 IRX-based X-ray (for low-luminosity AGN) ─────────


def xray_bolometric_correction_duras(l_bol_erg: float) -> float:
    r"""X-ray bolometric correction k_X (Duras et al. 2020, Eq. 2).

    Returns the ratio k_X = L_bol / L_X(2-10 keV). For low-luminosity
    AGN (L_bol < 10^43 erg/s), the correction approaches a constant
    k_X ≈ 9.65 (Lopez et al. 2024 Eq. 6).

    Parameters
    ----------
    l_bol_erg : float
        AGN bolometric luminosity. [erg/s]

    Returns
    -------
    float
        Bolometric correction factor k_X = L_bol / L_X. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives.

    The general form from Duras et al. (2020) is:

    .. math::

        k_X(L_{\rm bol}) = a \left[1 + \left(\frac{\log_{10}(L_{\rm bol})}{b}\right)^c\right]

    with best-fit coefficients a = 15.33, b = 11.48, c = 16.20
    (Duras et al. 2020, Table 2, X-ray correction).

    For the low-luminosity regime (L_bol ~ 10^38–10^43 erg/s),
    Lopez et al. (2024, Eq. 6) find a linear fit:

    .. math::

        L_{\rm bol} = (9.65 \pm 1.004) \times L_X

    References
    ----------
    .. [1] K. Duras et al., "Universal bolometric corrections for active
       galactic nuclei over seven luminosity decades," A&A, 636, A73 (2020).
       https://doi.org/10.1051/0004-6361/201936817
    .. [2] I. E. Lopez et al., "IRX-CIGALE: a tailored module for
       Low-Luminosity AGN," A&A, 692, A209 (2024). arXiv:2404.16938.
       https://doi.org/10.1051/0004-6361/202449801
    """
    a, b, c = 15.33, 11.48, 16.20
    log_l = jnp.log10(jnp.maximum(l_bol_erg, 1e-100))
    k_x = a * (1.0 + (log_l / b) ** c)
    return jnp.maximum(k_x, 1.0)


def xray_agn_corona_lopez24(
    wavelength: jnp.ndarray,
    l_12um_erg_hz: float,
    alpha_irx: float = 0.3,
    gamma: float = 1.8,
    E_cut: float = 300.0,
    cos_inc: float = COS_INC_REF_30DEG,
    apply_anisotropy: bool = True,
    a1: float = 0.5,
    a2: float = 0.0,
    log_nh: float = 20.0,
) -> jnp.ndarray:
    r"""AGN corona X-ray emission from 12μm luminosity (Lopez et al. 2024).

    Uses the α_IRX parameter (ratio of 2–10 keV X-ray luminosity to nuclear
    12μm luminosity) to derive the X-ray luminosity. This approach is more
    robust than α_ox for obscured and low-luminosity AGN where UV emission
    is unreliable.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Angstrom (rest-frame). [Å]
    l_12um_erg_hz : float
        Nuclear monochromatic luminosity density at 12 μm. [erg/s/Hz]
    alpha_irx : float
        Log ratio of νL_ν(12μm) to 2–10 keV luminosity (Asmus+2015 convention,
        matching CIGALE ``lopez24``):
        α_IRX = log₁₀(νL_ν(12μm) / L_X(2–10 keV)). [dimensionless]
        Default: 0.3 (X-ray ≈ 0.5·νL_ν(12μm)). Typical range: 0.0–0.6.
    gamma : float
        X-ray photon index (Γ, where F_ν ∝ ν^{1−Γ}). [dimensionless]
        Default: 1.8. Typical range: 1.4–3.5.
    E_cut : float
        Exponential cutoff energy. [keV]
        Default: 300.0.
    cos_inc : float
        Cosine of inclination angle (1 = face-on, 0 = edge-on). [dimensionless]
        Default: 1.0.
    apply_anisotropy : bool
        Whether to apply Yang+2022 viewing-angle correction.
        Default: True.
    a1 : float
        Linear anisotropy coefficient. [dimensionless]
        Default: 0.5.
    a2 : float
        Quadratic anisotropy coefficient. [dimensionless]
        Default: 0.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives.

    The α_IRX parameter connects the mid-IR and X-ray luminosities via
    the Asmus et al. (2015) / Gandhi et al. (2009) L_X–L_12μm relation:

    .. math::

        \alpha_{\rm IRX} = \log_{10}\!\left(
            \frac{\nu L_\nu(12\mu\rm m)}{L_X^{2\text{--}10\,\rm keV}}\right)

    The intrinsic 2–10 keV luminosity is:

    .. math::

        L_X^{2\text{--}10\,\rm keV} = \frac{\nu L_\nu(12\mu\rm m)}{10^{\alpha_{\rm IRX}}}

    The advantage over α_ox: 12μm emission is dominated by the torus and
    is relatively unaffected by obscuration (scatter ≈ 0.33 dex vs UV
    which can be absorbed by orders of magnitude). This makes α_IRX
    especially suitable for LLAGN and obscured Seyferts.

    **Reference**: Implements CIGALE IRX-CIGALE module (Lopez et al. 2024
    [1]_); validated against its output.

    References
    ----------
    .. [1] I. E. Lopez et al., "IRX-CIGALE: a tailored module for
       Low-Luminosity AGN," A&A, 692, A209 (2024). arXiv:2404.16938.
       https://doi.org/10.1051/0004-6361/202449801
    .. [2] P. Gandhi et al., "Resolving the mid-infrared cores of local
       Seyferts," A&A, 502, 457 (2009). arXiv:0905.2577.
       https://doi.org/10.1051/0004-6361/200811368
    .. [3] D. Asmus et al., "Local AGN survey (LASr): I. Galaxy sample,
       infrared and X-ray data," MNRAS, 454, 766 (2015).
       https://doi.org/10.1093/mnras/stv1950
    .. [4] G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
       CIGALE and improvement of the code," ApJ, 927, 192 (2022).
       https://doi.org/10.3847/1538-4357/ac4971
    """
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / 1.6022e-9

    # Derive L_X(2-10 keV) from α_IRX and νL_ν(12μm).
    # L_12μm as νL_ν in erg/s: convert from erg/s/Hz via the 12 μm frequency.
    # Asmus+2015 / Lopez+2024 (matching CIGALE lopez24.py:200): α_IRX =
    # log10(νL_ν(12μm) / L_X(2-10 keV)), i.e. the X-ray sits *below* the 12 μm
    # (L_X = 0.5·νL_ν at the α_IRX = 0.3 default), so L_X = νL_ν(12μm) / 10^α_IRX.
    nu_12um = _C_AA / 1.2e5  # 12 μm = 120000 Å
    l_12um_erg = l_12um_erg_hz * nu_12um
    l_x_2_10 = l_12um_erg / 10.0**alpha_irx

    # Build power-law spectrum with exponential cutoff
    E_ref = 5.0  # keV (mid-band reference)
    spec = (E_keV / E_ref) ** (-gamma + 1) * jnp.exp(-E_keV / E_cut)

    # Normalize by integrating spectral shape over 2-10 keV
    E_fine = jnp.linspace(2.0, 10.0, 200)
    nu_fine = E_fine * _KEV_TO_HZ
    spec_fine = (E_fine / E_ref) ** (-gamma + 1) * jnp.exp(-E_fine / E_cut)
    band_integral = jnp.maximum(jnp.trapezoid(spec_fine, nu_fine), 1e-60)

    l_nu = l_x_2_10 / band_integral * spec

    # Ricci+2017 / Matsumoto+2026 Eq. B6: photoelectric + Compton
    # scattering + 1 % scattered.
    l_intr = l_nu
    l_nu = (
        tbabs_transmission(E_keV, log_nh) * compton_scattering_transmission(log_nh) * l_intr
        + 0.01 * l_intr
    )

    # X-ray mask (E > 0.1 keV => λ < 124 Å)
    l_nu = jnp.where(wavelength < 124.0, l_nu, 0.0)

    # Optional anisotropy correction (Yang+2022)
    if apply_anisotropy:
        l_nu = xray_anisotropy(l_nu, cos_inc, a1=a1, a2=a2)

    return l_nu


def xray_total_lopez24_terms(
    wavelength: jnp.ndarray,
    sfr: float = 1.0,
    stellar_mass: float = 1e10,
    stellar_age_gyr: float = 1.0,
    l_12um_erg_hz: float = 0.0,
    alpha_irx: float = 0.3,
    gamma_hmxb: float = 2.0,
    gamma_lmxb: float = 1.6,
    gamma_agn: float = 1.8,
    E_cut: float = 300.0,
    log_nh: float = 20.0,
    **_kwargs,
) -> dict[str, jnp.ndarray]:
    r"""Unsummed X-ray SED terms using IRX-based AGN (Lopez+2024) + XRBs.

    Computes HMXB, LMXB, hot gas, and AGN corona terms as separate unsummed
    arrays, each enabling independent precomputation through broadband filters.
    Identical structure to :func:`xray_total_terms`, but using α_IRX(L_12μm)
    for AGN normalization instead of α_OX(L_2500).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid in Angstrom. [Å]
    sfr : float
        Star formation rate. [Msun/yr]. Default: 1.0.
    stellar_mass : float
        Stellar mass. [Msun]. Default: 1e10.
    stellar_age_gyr : float
        Stellar age in Gyr. Default: 1.0. [Gyr]
    l_12um_erg_hz : float
        Nuclear 12μm luminosity density. [erg/s/Hz]
        Default: 0.0 (no AGN X-ray contribution).
    alpha_irx : float
        Log ratio of L_X to L_12μm. [dimensionless]
        Default: 0.3.
    gamma_hmxb : float
        HMXB photon index. Default: 2.0.
    gamma_lmxb : float
        LMXB photon index. Default: 1.6.
    gamma_agn : float
        AGN photon index. Default: 1.8.
    E_cut : float
        Exponential cutoff energy. [keV] Default: 300.
    log_nh : float
        Line-of-sight hydrogen column density [log10(cm⁻²)].
        Default: 20.0.

    Returns
    -------
    dict with keys {"hmxb", "lmxb", "hotgas", "agn"}
        Each value is ndarray, shape (n_wave,), units [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.

    **Why separate terms**: HMXB and LMXB carry distinct photon indices,
    and the AGN component via α_IRX has independent parameter dependencies.
    By returning unsummed terms, precompute mechanisms can integrate each
    through filters at build time, then sum at evaluation. The summed
    :func:`xray_total_lopez24` must return bit-identical results to summing
    all four values from this dict.

    See :func:`xray_agn_corona_lopez24` for the α_IRX model details
    and :func:`xray_xrb_terms` for the XRB component structure.
    """
    xrb_terms = xray_xrb_terms(
        wavelength,
        sfr=sfr,
        stellar_mass=stellar_mass,
        stellar_age_gyr=stellar_age_gyr,
        gamma_hmxb=gamma_hmxb,
        gamma_lmxb=gamma_lmxb,
        E_cut=E_cut,
    )
    # Hot gas (CIGALE lopez24: 8.3e31 × SFR), shared with the yang20 path.
    hotgas = xray_hotgas(wavelength, sfr, gamma=1.0, E_cut=1.0)
    agn = xray_agn_corona_lopez24(
        wavelength,
        l_12um_erg_hz,
        alpha_irx,
        gamma_agn,
        E_cut,
        apply_anisotropy=False,
        log_nh=log_nh,
    )
    return {
        "hmxb": xrb_terms["hmxb"],
        "lmxb": xrb_terms["lmxb"],
        "hotgas": hotgas,
        "agn": agn,
    }


def xray_total_lopez24(
    wavelength: jnp.ndarray,
    sfr: float = 1.0,
    stellar_mass: float = 1e10,
    stellar_age_gyr: float = 1.0,
    l_12um_erg_hz: float = 0.0,
    alpha_irx: float = 0.3,
    gamma_hmxb: float = 2.0,
    gamma_lmxb: float = 1.6,
    gamma_agn: float = 1.8,
    E_cut: float = 300.0,
    log_nh: float = 20.0,
    **_kwargs,
) -> jnp.ndarray:
    r"""Total X-ray emission using IRX-based AGN (Lopez+2024) + XRBs.

    Combines X-ray binary emission (HMXB + LMXB) with AGN corona emission
    derived from the 12μm luminosity via α_IRX.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid in Angstrom. [Å]
    sfr : float
        Star formation rate. [Msun/yr]. Default: 1.0.
    stellar_mass : float
        Stellar mass. [Msun]. Default: 1e10.
    stellar_age_gyr : float
        Stellar age in Gyr. Default: 1.0. [Gyr]
    l_12um_erg_hz : float
        Nuclear 12μm luminosity density. [erg/s/Hz]
        Default: 0.0 (no AGN X-ray contribution).
    alpha_irx : float
        Log ratio of L_X to L_12μm. [dimensionless]
        Default: 0.3.
    gamma_hmxb : float
        HMXB photon index. Default: 2.0.
    gamma_lmxb : float
        LMXB photon index. Default: 1.6.
    gamma_agn : float
        AGN photon index. Default: 1.8.
    E_cut : float
        Exponential cutoff energy. [keV] Default: 300.
    log_nh : float
        Line-of-sight hydrogen column density [log10(cm⁻²)].
        Default: 20.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.

    This function returns the sum of all X-ray components. Use
    :func:`xray_total_lopez24_terms` to access the individual unsummed terms
    for precomputation.

    See :func:`xray_agn_corona_lopez24` for the α_IRX model details
    and :func:`xray_xrb` for the XRB component.
    """
    t = xray_total_lopez24_terms(
        wavelength,
        sfr=sfr,
        stellar_mass=stellar_mass,
        stellar_age_gyr=stellar_age_gyr,
        l_12um_erg_hz=l_12um_erg_hz,
        alpha_irx=alpha_irx,
        gamma_hmxb=gamma_hmxb,
        gamma_lmxb=gamma_lmxb,
        gamma_agn=gamma_agn,
        E_cut=E_cut,
        log_nh=log_nh,
    )
    return t["hmxb"] + t["lmxb"] + t["hotgas"] + t["agn"]


# ── Deprecation shims ──


# Deprecated: old bolometric-normalization path for xray_agn_corona
# Use xray_agn_corona_from_disc (which takes l_2500_30deg) instead
xray_agn_corona_bolometric = deprecated_alias(
    _xray_agn_corona_bolometric,
    old_name="xray_agn_corona_bolometric",
    new_name="xray_agn_corona_from_disc",
    drop_version="1.0",
)
