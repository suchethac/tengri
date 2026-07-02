# SPDX-License-Identifier: BSD-3-Clause
"""Analytic dust IR emission closures.

Pure JAX emission functions for the analytic models (modified blackbody,
Casey 2012, Drude-profile PAH, Schreiber 2016, and the two-temperature
energy-balance split). These are the physics kernels wrapped by the
``EmissionPort`` subclasses in this package; the ``emission`` facade imports
them and registers the grammar-dispatchable ones in ``DUST_EMISSION_MODELS``.

Leaf module: imports only ``jnp``, physical constants, and the shared
``_physics`` helpers, so it never forms an import cycle with the facade (#843).
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.components.dust.emission._physics import (
    cmb_contrast_factor,
    cmb_corrected_temperature,
    planck_bnu,
)
from tengri.utils.physics_constants import (
    AA_TO_CM as _AA_TO_CM,
    C_CGS as _C_CGS,
    H_PLANCK as _H_PLANCK,
    K_BOLTZ as _K_BOLTZMANN,
)


def modified_blackbody(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    dust_T: float = 30.0,
    dust_beta_ir: float = 1.8,
    redshift: float = 0.0,
    dust_epsilon_mbb: float = 1.0,
    **_kwargs,
) -> jnp.ndarray:
    """Optically-thin modified blackbody dust emission.

    The unnormalized spectrum is::

        S_nu ~ nu^beta * B_nu(T_dust)

    which is then normalized so that the frequency integral equals
    ``L_absorbed``.

    When ``redshift > 0``, the dust temperature is corrected for CMB
    heating (da Cunha et al. 2013) and the observed flux is reduced by
    the CMB contrast factor.

    Parameters
    ----------
    wavelength_aa : array, shape (n_wave,)
        Wavelength grid in Angstrom (sorted ascending). [Å]
    L_absorbed : float
        Total absorbed luminosity.  Unit-agnostic: the output L_nu will be
        in the same units per Hz (e.g. pass erg/s → get erg/s/Hz; pass
        Lsun → get Lsun/Hz).  In ``sed_pipeline.py`` the pipeline passes
        erg/s (from a frequency-integrated trapezoid) and receives erg/s/Hz.
    dust_T : float
        Dust temperature in Kelvin.  Typical range: 20--60 K. [K]
    dust_beta_ir : float
        Emissivity index.  Typical range: 1.5--2.0. [dimensionless]
    redshift : float
        Source redshift. When > 0, CMB heating correction is applied.
        Default 0 (no correction). [dimensionless]

    Returns
    -------
    array, shape (n_wave,)
        Dust emission L_nu in ``[L_absorbed units] / Hz``.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere.

    References
    ----------
    .. [1] B. T. Draine, "Physics of the Interstellar and Intergalactic Medium"
       (Princeton University Press, 2011). Chapter 22: thermal continuum
       emission from dust grains in the optically-thin limit.
       https://ui.adsabs.harvard.edu/abs/2011piim.book.....D

    """
    # epsilon_mbb (CIGALE mbb): fraction of L_dust carried by the MBB. 1.0 =
    # full energy balance (default); < 1.0 scales the MBB luminosity down.
    L_absorbed = L_absorbed * jnp.clip(dust_epsilon_mbb, 0.0, 1.0)

    # CMB correction: always applied. At z=0 this is a no-op since
    # T_cmb(z=0) terms cancel and B_nu(T_cmb)/B_nu(T_dust) ~ 0.
    T_eff = cmb_corrected_temperature(dust_T, redshift, dust_beta_ir)

    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm

    # Reference frequency at 250 um (convenient normalization pivot)
    nu_ref = _C_CGS / (250.0e-4)  # 250 um in cm
    emissivity = (nu / nu_ref) ** dust_beta_ir

    bnu = planck_bnu(wavelength_aa, T_eff)

    # Unnormalized SED shape (erg/s/cm^2/Hz/sr units cancel in ratio)
    shape = emissivity * bnu

    # Integrate shape over frequency for normalization.
    # nu is descending (wave ascending), so negate to get positive integral.
    integral = -jnp.trapezoid(shape, nu)

    # Guard against zero integral (e.g. wavelength grid entirely outside
    # the thermal peak) — return zeros instead of NaN
    norm = jnp.where(integral > 0.0, L_absorbed / integral, 0.0)

    result = norm * shape

    # CMB contrast: suppresses flux where dust is observed against CMB
    contrast = cmb_contrast_factor(wavelength_aa, T_eff, redshift)

    return result * contrast


# ── Model 1b: Casey (2012) modified blackbody + mid-IR power law ──

# Empirical coefficients for turnover wavelength (Casey 2012, Eq. 3, errata)
_CASEY_B1_UM = 26.68  # μm
_CASEY_B2_UM_PER_K = 6.246e-3  # μm / K


def _casey_transition_function(
    wavelength_cm: jnp.ndarray,
    T_eff: float,
) -> jnp.ndarray:
    """Compute Casey (2012) transition function between power law and MBB.

    Parameters
    ----------
    wavelength_cm : array, shape (n_wave,)
        Wavelength grid in cm. [cm]
    T_eff : float
        Effective dust temperature (already CMB-corrected). [K]

    Returns
    -------
    array, shape (n_wave,)
        Transition function f(λ) = 1 / (1 + (λ/λ_0)^2).
        f→1 at short λ (power law), f→0 at long λ (MBB). [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes.

    """
    # Empirical turnover wavelength (Casey 2012, Eq. 3 with errata)
    lambda0_cm = (_CASEY_B1_UM + _CASEY_B2_UM_PER_K * T_eff) * 1.0e-4  # μm -> cm

    # Transition function: f(λ) = 1 / (1 + (λ/λ_0)^2)
    # f→1 at short λ (mid-IR power law dominates), f→0 at long λ (MBB dominates)
    # Casey 2012 convention: power law for Wien side, MBB for Rayleigh-Jeans
    return 1.0 / (1.0 + (wavelength_cm / lambda0_cm) ** 2)


def _casey_powerlaw_component(
    nu: jnp.ndarray,
    nu_ref: float,
    f_transition: jnp.ndarray,
    x: jnp.ndarray,
    dust_alpha_mir: float,
    optically_thin: bool,
) -> jnp.ndarray:
    """Compute Casey (2012) mid-IR power-law component.

    Parameters
    ----------
    nu : array, shape (n_wave,)
        Frequency in Hz (descending). [Hz]
    nu_ref : float
        Reference frequency (100 μm pivot). [Hz]
    f_transition : array, shape (n_wave,)
        Transition function from _casey_transition_function. [dimensionless]
    x : array, shape (n_wave,)
        Planck argument h*nu/(k*T). [dimensionless]
    dust_alpha_mir : float
        Mid-IR power-law slope. [dimensionless]
    optically_thin : bool
        If True, return zero (power law suppressed). [dimensionless]

    Returns
    -------
    array, shape (n_wave,)
        Mid-IR power-law component S_pl(ν). [erg/s/Hz] (before normalization)

    Notes
    -----
    **JIT-compatible**: yes.

    The exponential cutoff prevents the power law from diverging at
    UV/optical wavelengths. This follows Casey (2012) Eq. 2 where the
    power law implicitly operates only in the IR regime.

    """
    # S_pl(ν) ~ ν^α_mid * f(ν) * exp(-hν/kT) [Wien cutoff]
    wien_cutoff = jnp.exp(-x)
    power_law = (nu / nu_ref) ** dust_alpha_mir * f_transition * wien_cutoff
    power_law = power_law * (1.0 - optically_thin)
    return power_law


def _casey_mbb_component(
    nu: jnp.ndarray,
    nu_ref: float,
    f_transition: jnp.ndarray,
    x: jnp.ndarray,
    dust_beta_ir: float,
) -> jnp.ndarray:
    """Compute Casey (2012) modified blackbody component.

    Parameters
    ----------
    nu : array, shape (n_wave,)
        Frequency in Hz (descending). [Hz]
    nu_ref : float
        Reference frequency (100 μm pivot). [Hz]
    f_transition : array, shape (n_wave,)
        Transition function from _casey_transition_function. [dimensionless]
    x : array, shape (n_wave,)
        Planck argument h*nu/(k*T). [dimensionless]
    dust_beta_ir : float
        Dust emissivity index. [dimensionless]

    Returns
    -------
    array, shape (n_wave,)
        Modified blackbody component S_bb(ν). [erg/s/Hz] (before normalization)

    Notes
    -----
    **JIT-compatible**: yes.

    """
    # S_bb(ν) ~ ν^(3+β) / (exp(hν/kT) - 1) * (1 - f(ν))
    mbb = (nu / nu_ref) ** (3.0 + dust_beta_ir) / (jnp.exp(x) - 1.0)
    mbb = mbb * (1.0 - f_transition)
    return mbb


def casey2012(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    dust_T: float = 35.0,
    dust_beta_ir: float = 1.8,
    dust_alpha_mir: float = 2.0,
    optically_thin: bool = False,
    redshift: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Casey (2012) modified blackbody + mid-IR power law dust emission.

    Combines a modified blackbody (FIR peak from cold/warm dust) with a
    mid-IR power law (Wien-side excess from warm dust continuum), joined
    by a smooth sigmoid transition function.

    When ``optically_thin=True``, the mid-IR power-law component is zeroed,
    leaving only the modified blackbody.  This variant is useful for cold
    dust-dominated galaxies where the power law is unphysical.

    .. note::

        The mid-IR power-law contribution is only significant for **warm/hot
        dust** (T ≳ 60 K).  For typical cold ISM dust (T = 25–60 K) the Wien
        cutoff exp(-hν/kT) kills the power-law component at 8–40 μm (x ≈ 10–51
        at those wavelengths), so the model produces *less* 8–40 μm flux than a
        pure MBB normalized to the same L_absorbed.  The 8–40 μm advantage
        described in Casey (2012) applies to warmer starburst / AGN-heated dust
        components where T ≳ 80–100 K.

    The implemented model uses the following convention (see code comments)::

        S(ν) = N_pl * ν^α_mid * f(λ)         [mid-IR power law, f→1 at short λ]
             + N_bb * ν^(3+β) / (exp(hν/kT) - 1) * (1 - f(λ))   [FIR MBB, 1-f→1 at long λ]

    where the transition function (f→1 selects power law at short λ) is::

        f(λ) = 1 / (1 + (λ / λ_0)^2)

    Note: Casey (2012, MNRAS 425 3094) Eq. 2 defines the carrier function differently;
    the code's convention has f→1 at short λ (mid-IR) and 1-f→1 at long λ (FIR).
    The shapes produced are equivalent; only the labeling of f vs (1-f) differs.

    The empirical turnover wavelength is (Eq. 3, with errata)::

        λ_0 = b1 + b2 * T[μm]

    with ``b1 = 26.68 μm``, ``b2 = 6.246e-3 μm/K``.

    Both components are normalized so that the total frequency integral
    equals ``L_absorbed``.

    When ``redshift > 0``, the dust temperature is corrected for CMB
    heating (da Cunha et al. 2013) and the observed flux is reduced by
    the CMB contrast factor.

    Parameters
    ----------
    wavelength_aa : array, shape (n_wave,)
        Wavelength grid in Angstrom (sorted ascending). [Å]
    L_absorbed : float
        Total absorbed luminosity in Lsun (sets the normalization). [L_sun]
    dust_T : float
        Dust temperature in Kelvin.  Typical range: 25--60 K. [K]
    dust_beta_ir : float
        Dust emissivity index for the MBB component.
        Typical range: 1.5--2.0. [dimensionless]
    dust_alpha_mir : float
        Mid-IR power-law slope.  Typical range: 1.5--2.5. [dimensionless]
    optically_thin : bool
        If True, zero the mid-IR power-law component, leaving only the
        modified blackbody.  Default: False. [dimensionless]
    redshift : float
        Source redshift. When > 0, CMB heating correction is applied.
        Default 0 (no correction). [dimensionless]

    Returns
    -------
    array, shape (n_wave,)
        Dust emission L_nu in Lsun/Hz.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere.

    References
    ----------
    Casey, C. M., 2012, MNRAS, 425, 3094.
    da Cunha, E. et al., 2013, ApJ, 766, 13 (CMB corrections).

    """
    # CMB correction (no-op at z=0)
    T_eff = cmb_corrected_temperature(dust_T, redshift, dust_beta_ir)

    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm  # Hz, descending

    # Planck argument (shared by both components)
    x = jnp.clip(_H_PLANCK * nu / (_K_BOLTZMANN * T_eff), 0.0, 500.0)
    nu_ref = _C_CGS / (100.0e-4)  # 100 μm pivot in Hz

    f_transition = _casey_transition_function(wavelength_cm, T_eff)
    power_law = _casey_powerlaw_component(
        nu, nu_ref, f_transition, x, dust_alpha_mir, optically_thin
    )
    mbb = _casey_mbb_component(nu, nu_ref, f_transition, x, dust_beta_ir)
    shape = power_law + mbb

    # Normalize so integral over frequency = L_absorbed
    # nu is descending (wave ascending), negate for positive integral
    integral = -jnp.trapezoid(shape, nu)
    norm = jnp.where(integral > 0.0, L_absorbed / integral, 0.0)

    result = norm * shape

    # CMB contrast suppression
    contrast = cmb_contrast_factor(wavelength_aa, T_eff, redshift)

    return result * contrast


# ── Model 1c: Schreiber et al. (2016) dust continuum + PAH ────────


def _drude_profile(
    wavelength_aa: jnp.ndarray,
    lambda0_aa: float,
    fwhm_um: float,
) -> jnp.ndarray:
    r"""Drude profile for PAH emission feature.

    Parameters
    ----------
    wavelength_aa : array_like, shape (n_wave,)
        Wavelength grid in Ångstrom.
    lambda0_aa : float
        Center wavelength in Ångstrom.
    fwhm_um : float
        FWHM in micrometers.

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized Drude profile.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    The Drude profile is:

    .. math::

        D(\lambda) = \frac{2}{\pi} \frac{\gamma}{
            (\lambda/\lambda_0 - \lambda_0/\lambda)^2 + \gamma^2}

    where :math:`\gamma = \text{FWHM} / \lambda_0`.

    """
    lambda0_um = lambda0_aa / 1e4
    fwhm_um_safe = jnp.maximum(fwhm_um, 1e-6)
    gamma = fwhm_um_safe / lambda0_um

    wavelength_um = wavelength_aa / 1e4
    ratio = wavelength_um / lambda0_um

    denominator = (ratio - 1.0 / ratio) ** 2 + gamma**2
    return 2.0 / jnp.pi * gamma / denominator


# ── Model 2: PAH Drude Profile Template ──


def pah_drude(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    redshift: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Smith et al. (2007) PAH Drude profiles — mid-IR PAH building block.

    A sum of 18 PAH Drude profiles (normalized to Smith+2007 SINGS median
    strengths). This is a **PAH-only building block**, not a standalone
    energy-balanced dust emitter: it carries the aromatic-feature forest only
    (no thermal continuum), so its frequency integral is *not* renormalized to
    ``L_absorbed`` — it is scaled by ``L_absorbed`` but deliberately leaves the
    bulk of the absorbed energy for a continuum component to carry. Select a
    full model (``dale2014``, ``draine_li2007/2014``, ``themis``,
    ``modified_blackbody``, ``casey2012``, ``schreiber2018``) for an
    energy-conserving dust SED; ``pah_drude`` is intended as a diagnostic /
    composition primitive (it backs the PAH term of ``schreiber2016``).

    Parameters
    ----------
    wavelength_aa : array, shape (n_wave,)
        Wavelength grid in Angstrom (sorted ascending). [Å]
    L_absorbed : float
        Total absorbed luminosity. [Lsun or erg/s, as passed]
    redshift : float
        Source redshift (unused for this model). [dimensionless]

    Returns
    -------
    array, shape (n_wave,)
        Dust emission L_nu in ``[L_absorbed units] / Hz``.

    Notes
    -----
    **JIT-compatible**: yes — pure ``jnp`` primitives (a precomputed lookup in
    :mod:`~tengri.components.dust.dust_analytic_precompute` is preferred in the
    hybrid kernel; this is the direct full-wavelength evaluation).

    **Gradient-safe**: yes.

    **Not energy-balanced standalone** — see the summary above; excluded from
    the cross-model energy-balance contract test for this reason.

    The PAH template is a pure shape (no free axes). Runtime evaluation uses the
    precomputed lookup from :mod:`~tengri.components.dust.dust_analytic_precompute`
    and skips the full-wavelength evaluation in the hybrid kernel.

    References
    ----------
    .. [1] Smith, J. D., et al., "The mid-infrared emission of ultraluminous
           infrared galaxies," ApJ, 656, 770 (2007). arXiv:astro-ph/0701042.
           https://doi.org/10.1086/510378

    """
    from tengri.components.dust.drude_profiles import compute_pah_template

    wavelength_um = wavelength_aa / 1e4
    pah_llam = compute_pah_template(wavelength_um, strengths=None)

    # L_nu = L_lambda * lambda^2 / c.
    wave_cm = wavelength_aa * 1.0e-8
    lnu = L_absorbed * pah_llam * (wave_cm**2) / _C_CGS

    return lnu


def schreiber2016(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    dust_T: float = 30.0,
    dust_f_pah: float = 0.05,
    redshift: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    r"""Schreiber et al. (2016) 2-parameter dust emission model (analytic).

    Mixes dust continuum and PAH emission by a fractional parameter.
    The dust continuum is a modified blackbody (modified_blackbody with
    beta=1.5). The PAH component is **approximated** as a sum of Drude profiles
    at standard wavelengths (not the full Schreiber+ mid-IR aromatic forest).

    For the CIGALE-faithful tabulated version with the real PAH feature forest,
    select ``schreiber2018`` (``data/schreiber2018_templates.h5``) instead —
    this analytic model is the lightweight, grid-free approximation.

    Parameters
    ----------
    wavelength_aa : array_like, shape (n_wave,)
        Wavelength grid in Ångstrom (sorted ascending).
    L_absorbed : float
        Total absorbed luminosity. Unit-agnostic: the output L_nu will be
        in the same units per Hz.
    dust_T : float
        Dust continuum temperature in Kelvin.
        Typical range: 15--60 K. Default: 30.0.
    dust_f_pah : float
        Fractional contribution from PAH emission in [0, 1].
        Default: 0.05.
    redshift : float
        Source redshift. When > 0, CMB heating correction is applied.
        Default: 0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_nu in ``[L_absorbed units] / Hz``.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    The model composition is:

    .. math::

        L_\nu = (1 - f_{\rm PAH}) L_\nu^{\rm continuum} + f_{\rm PAH} L_\nu^{\rm PAH}

    where:

    - Dust continuum: modified blackbody with temperature T_dust and emissivity
      index β = 1.5, using the same normalization as ``modified_blackbody``.
    - PAH: sum of Drude profiles at standard rest wavelengths (3.3, 6.2, 7.7,
      8.6, 11.3, 12.7 μm) with relative strengths from Smith et al. (2007).

    The total integral over frequency is normalized to ``L_absorbed``.

    References
    ----------
    .. [1] Schreiber, C., Elbaz, D., Sparre, M., et al., 2016,
           A&A, 589, A35 (https://doi.org/10.1051/0004-6361/201527923)

    .. [2] Smith, J. D. T., Draine, B. T., Dale, D. A., et al., 2007,
           ApJ, 656, 770 (PAH profile templates).

    """
    # CMB correction (no-op at z=0)
    T_eff = cmb_corrected_temperature(dust_T, redshift, 1.5)

    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm  # Hz, descending

    # ─ Dust continuum component (modified blackbody with beta=1.5) ─
    nu_ref = _C_CGS / (250.0e-4)
    emissivity = (nu / nu_ref) ** 1.5
    bnu = planck_bnu(wavelength_aa, T_eff)
    continuum = emissivity * bnu

    # ─ PAH component (sum of Drude profiles) ─
    # PAH features from Smith et al. (2007), with rest wavelengths and FWHM
    pah_features = [
        (33000.0, 0.05, 0.04),  # 3.3 μm, FWHM 0.05 μm, strength 0.04
        (62000.0, 0.19, 0.14),  # 6.2 μm, FWHM 0.19 μm, strength 0.14
        (77000.0, 0.47, 0.42),  # 7.7 μm, FWHM 0.47 μm, strength 0.42
        (86000.0, 0.27, 0.11),  # 8.6 μm, FWHM 0.27 μm, strength 0.11
        (113000.0, 0.18, 0.19),  # 11.3 μm, FWHM 0.18 μm, strength 0.19
        (127000.0, 0.32, 0.10),  # 12.7 μm, FWHM 0.32 μm, strength 0.10
    ]

    pah_sum = jnp.zeros_like(wavelength_aa)
    for lambda0_aa, fwhm_um, strength in pah_features:
        drude = _drude_profile(wavelength_aa, lambda0_aa, fwhm_um)
        pah_sum = pah_sum + strength * drude

    # Normalize PAH to unit integral
    pah_integral = -jnp.trapezoid(pah_sum, nu)
    pah_normalized = jnp.where(pah_integral > 0.0, pah_sum / pah_integral, 0.0)

    # ─ Mix components ─
    f_pah_clipped = jnp.clip(dust_f_pah, 0.0, 1.0)
    mixed_shape = (1.0 - f_pah_clipped) * continuum + f_pah_clipped * pah_normalized

    # ─ Normalize total to L_absorbed ─
    integral = -jnp.trapezoid(mixed_shape, nu)
    norm = jnp.where(integral > 0.0, L_absorbed / integral, 0.0)

    result = norm * mixed_shape

    # CMB contrast
    contrast = cmb_contrast_factor(wavelength_aa, T_eff, redshift)

    return result * contrast


# ── Backward-compatible module-level aliases for direct imports ───
# Tests and user code may do ``from tengri.components.dust.emission import draine_li2007``.
# These aliases point to the lazy wrappers (which auto-load templates on call).


def energy_balance_split(
    wavelength_aa: jnp.ndarray,
    L_absorbed_stellar: float,
    L_agn_ir: float = 0.0,
    eta_balance: float = 1.0,
    f_cold: float = 0.5,
    dust_T_warm: float = 45.0,
    dust_T_cold: float = 20.0,
    dust_beta_warm: float = 1.5,
    dust_beta_cold: float = 2.0,
    redshift: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    r"""Two-temperature energy balance with AGN contribution.

    Extends simple eta_balance by decomposing IR into warm (SF-heated)
    and cold (diffuse ISM) components, plus optional AGN IR contribution.

    Parameters
    ----------
    wavelength_aa : array_like, shape (n_wave,)
        Wavelength grid. [Å] Must be sorted ascending.
    L_absorbed_stellar : float
        Total absorbed stellar luminosity. [Lsun]
    L_agn_ir : float
        Additional AGN-heated IR luminosity. [Lsun] Default: 0.0.
    eta_balance : float
        Energy balance parameter: ratio of re-emitted to absorbed stellar luminosity.
        [dimensionless] Default: 1.0 (strict energy balance).
    f_cold : float
        Fraction of total IR luminosity in the cold component.
        [dimensionless, in [0, 1]] Default: 0.5.
    dust_T_warm : float
        Warm dust temperature. [K] Default: 45.0.
    dust_T_cold : float
        Cold dust temperature. [K] Default: 20.0.
    dust_beta_warm : float
        Warm component emissivity index. [dimensionless] Default: 1.5.
    dust_beta_cold : float
        Cold component emissivity index. [dimensionless] Default: 2.0.
    redshift : float
        Source redshift. [dimensionless] When > 0, CMB heating correction is applied
        to both components. Default: 0.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_ν. [Lsun Hz⁻¹]

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    The total IR luminosity budget is:

    .. math::

        L_{\rm IR,total} = \eta_{\rm balance} L_{\rm absorbed,\star} + L_{\rm AGN,IR}

        L_{\rm warm} = (1 - f_{\rm cold}) L_{\rm IR,total}

        L_{\rm cold} = f_{\rm cold} L_{\rm IR,total}

    Each component is a modified blackbody (via ``modified_blackbody``).

    References
    ----------
    .. [1] V. Kokorev et al., "STARDUST: Spectral Template Analysis and
       Recovery of Dust and Ultraviolet Spectral features,"
       ApJ, 921, 40 (2021). https://doi.org/10.3847/1538-4357/ac1aa7

    .. [2] E. da Cunha et al., "MAGPHYS: a new code to compute and interpret
       the Spectral Energy Distribution of the Galaxy," MNRAS, 388, 1595 (2008).
       https://doi.org/10.1111/j.1365-2966.2008.13535.x

    """
    L_ir_total = eta_balance * L_absorbed_stellar + L_agn_ir

    L_warm = (1.0 - f_cold) * L_ir_total
    L_cold = f_cold * L_ir_total

    # Each component is a modified blackbody
    sed_warm = modified_blackbody(
        wavelength_aa,
        L_absorbed=L_warm,
        dust_T=dust_T_warm,
        dust_beta_ir=dust_beta_warm,
        redshift=redshift,
    )
    sed_cold = modified_blackbody(
        wavelength_aa,
        L_absorbed=L_cold,
        dust_T=dust_T_cold,
        dust_beta_ir=dust_beta_cold,
        redshift=redshift,
    )

    return sed_warm + sed_cold


# ── Application layer: model dispatchers and utilities ────────────
