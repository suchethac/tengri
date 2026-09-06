# SPDX-License-Identifier: BSD-3-Clause
"""Analytic dust IR emission closures.

Pure JAX emission functions for the analytic models (modified blackbody,
Casey 2012, Drude-profile PAH, Schreiber 2016, and the two-temperature
energy-balance split). These are the physics kernels wrapped by the
``EmissionComponent`` subclasses in this package; the ``emission`` facade imports
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

# x = h*nu/(k_B*T) ceiling.  A plain constant, not a dtype-dependent one: the
# denominator below is ``-expm1(-x)`` in (0, 1], so nothing here can overflow
# at any x, and ``exp(-x)`` underflows to exactly 0.0 (the true Wien limit)
# rather than needing to be clamped short of it (#1439).
_X_MAX: float = 500.0

# ...and a floor, because the denominator vanishes at the OTHER end: the
# occupation number goes as 1/x for small x, so x = 0 is a division by zero
# (``exp(-0) / -expm1(-0)`` is ``1 / 0``) and the Rayleigh-Jeans limit comes
# back ``inf`` rather than large-but-finite. Same value and same reason as
# ``utils.blackbody._X_MIN``, which the sibling Planck closure has always
# carried; this one clipped at 0.0 and could reach the pole (#1439).
_X_MIN: float = 1e-10


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

    **Convention**: this is the *optically-thin* modified blackbody. CIGALE's
    ``mbb`` module instead uses the general-opacity form
    ``(1 - exp(-(200 µm/λ)^β)) ν³ B``-factor, which peaks ~35% redder at
    T = 35 K (124 vs 91 µm) and carries ~3× more submm flux at fixed
    bolometric output (parity sweep vs pcigale 2025.1; #1006 tracks an
    opt-in opacity toggle). Both forms are standard in the literature:
    for a general-opacity graybody today, use ``casey2012`` (its graybody
    term is exactly that form).

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
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes, differentiable everywhere.

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

    # CMB correction: always applied. At z=0 the CMB suppression is
    # wavelength-dependent: ~0.4% at 1 mm (contrast ~ 0.996) and ~8% at 1 cm
    # (contrast ~ 0.92) for T_dust = 25 K, becoming negligible only at much
    # shorter wavelengths in the FIR peak.
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
    # the thermal peak): return zeros instead of NaN
    norm = jnp.where(integral > 0.0, L_absorbed / integral, 0.0)

    result = norm * shape

    # CMB contrast: suppresses flux where dust is observed against CMB
    contrast = cmb_contrast_factor(wavelength_aa, T_eff, redshift)

    return result * contrast


# ── Model 1a: General-opacity graybody ──


def graybody(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    dust_T: float = 35.0,
    dust_beta_ir: float = 1.8,
    dust_lambda_0_um: float = 200.0,
    redshift: float = 0.0,
    dust_epsilon_mbb: float = 1.0,
    **_kwargs,
) -> jnp.ndarray:
    r"""General-opacity graybody dust emission.

    The unnormalized spectrum is::

        S_nu ~ (1 - exp(-(lam_0/lam)^beta)) * B_nu(T_dust)

    which is then normalized so that the frequency integral equals
    ``L_absorbed``.

    This is the **general-opacity** graybody form used by Synthesizer
    (``Greybody(..., optically_thin=False)``) and CIGALE's ``mbb`` module
    (Boquien et al. 2019). It differs from the optically-thin modified
    blackbody (which lacks the opacity factor) and from Casey 2012's graybody
    (which adds a mid-IR power law and fixes the pivot at 200 μm). For
    the optionally-thin form, use ``modified_blackbody``; for the graybody
    plus mid-IR power law, use ``casey2012``.

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
        Lsun → get Lsun/Hz).
    dust_T : float
        Dust temperature in Kelvin.  Typical range: 20--60 K. [K]
    dust_beta_ir : float
        Emissivity index.  Typical range: 1.5--2.0. [dimensionless]
    dust_lambda_0_um : float
        Graybody optical depth pivot wavelength (µm): the wavelength where
        optical depth τ = 1. Default 200 µm (Casey 2012); Synthesizer uses
        100 µm. [µm]
    redshift : float
        Source redshift. When > 0, CMB heating correction is applied.
        Default 0 (no correction). [dimensionless]
    dust_epsilon_mbb : float
        Fraction of L_absorbed carried by this graybody (CIGALE mbb epsilon_mbb;
        1.0 = full energy balance). [dimensionless, in [0, 1]]

    Returns
    -------
    array, shape (n_wave,)
        Dust emission L_nu in ``[L_absorbed units] / Hz``.

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes, differentiable everywhere.

    References
    ----------
    .. [1] Boquien, M., Burgarella, D., Roehlly, Y., et al. 2019,
       A&A 622, A103. CIGALE: Code Investigating GALaxy Emission.
       https://doi.org/10.1051/0004-6361/201834156

    .. [2] Casey, C. M., 2012, MNRAS, 425, 3094, Eqs. 1-2, 11-12.
       doi:10.1111/j.1365-2966.2012.21455.x, arXiv:1206.1595.

    .. [3] da Cunha, E. et al., 2013, ApJ, 766, 13 (CMB corrections).
       doi:10.1088/0004-637X/766/1/13, arXiv:1302.0844.

    """
    # epsilon_mbb (CIGALE mbb): fraction of L_dust carried by the MBB. 1.0 =
    # full energy balance (default); < 1.0 scales the MBB luminosity down.
    L_absorbed = L_absorbed * jnp.clip(dust_epsilon_mbb, 0.0, 1.0)

    # CMB correction: always applied.
    T_eff = cmb_corrected_temperature(dust_T, redshift, dust_beta_ir)

    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm

    # Reference frequency at 250 um (convenient normalization pivot)
    nu_ref = _C_CGS / (250.0e-4)  # 250 um in cm
    emissivity = (nu / nu_ref) ** dust_beta_ir

    bnu = planck_bnu(wavelength_aa, T_eff)

    # Opacity factor: (1 - exp(-(lam_0/lam)^beta))
    # Convert lambda_0_um to cm for consistent units
    lambda_0_cm = dust_lambda_0_um * 1.0e-4  # um to cm
    tau = (lambda_0_cm / wavelength_cm) ** dust_beta_ir
    opacity = -jnp.expm1(-tau)  # = 1 - exp(-tau), numerically stable

    # Unnormalized SED shape (erg/s/cm^2/Hz/sr units cancel in ratio)
    shape = opacity * emissivity * bnu

    # Integrate shape over frequency for normalization.
    # nu is descending (wave ascending), so negate to get positive integral.
    integral = -jnp.trapezoid(shape, nu)

    # Guard against zero integral (e.g. wavelength grid entirely outside
    # the thermal peak): return zeros instead of NaN
    norm = jnp.where(integral > 0.0, L_absorbed / integral, 0.0)

    result = norm * shape

    # CMB contrast: suppresses flux where dust is observed against CMB
    contrast = cmb_contrast_factor(wavelength_aa, T_eff, redshift)

    return result * contrast


# ── Model 1b: Casey (2012) modified blackbody + mid-IR power law ──

# Casey (2012) Eqs. 11-12 turnover-fit coefficients: λ_c = (3/4)·λ_turnover
# with λ_turnover(α, T) = 10³ nm / [(b1 + b2·α)⁻² + (b3 + b4·α)·T].
_CASEY_B1 = 26.68  # dimensionless
_CASEY_B2 = 6.246  # dimensionless (per unit α)
_CASEY_B3_PER_K = 1.905e-4  # 1/K
_CASEY_B4_PER_K = 7.243e-5  # 1/K (per unit α)
# Default opacity pivot of the general graybody, Casey (2012) Eq. 1.
# Now parametric via dust_lambda_0_um; this is the default.
_CASEY_LAMBDA0_CM_DEFAULT = 200.0e-4  # 200 µm [cm]


def _casey_lambda_c_cm(T_eff: float, dust_alpha_mir: float) -> jnp.ndarray:
    """Power-law turnover wavelength λ_c [cm]: Casey (2012) Eqs. 11-12.

    λ_c = (3/4)·λ_turnover, where the peak-wavelength fit is
    λ_turnover [nm] = 10³ / [(b1 + b2 α)⁻² + (b3 + b4 α) T].

    Notes
    -----
    **JIT-compatible**: yes.
    """
    denom = (_CASEY_B1 + _CASEY_B2 * dust_alpha_mir) ** -2.0 + (
        _CASEY_B3_PER_K + _CASEY_B4_PER_K * dust_alpha_mir
    ) * T_eff
    return (0.75e3 / denom) * 1.0e-7  # nm -> cm


def _casey_graybody_nu(
    wavelength_cm: jnp.ndarray,
    T_eff: float,
    dust_beta_ir: float,
    optically_thin: bool,
    lambda_0_cm: float = _CASEY_LAMBDA0_CM_DEFAULT,
) -> jnp.ndarray:
    r"""Graybody S_ν shape: the second term of Casey (2012) Eq. 1.

    .. math::

        S_\nu \propto \left(1 - e^{-(\lambda_0/\lambda)^\beta}\right)
        \frac{\nu^3}{e^{h\nu/kT} - 1},
        \qquad \lambda_0 = 200\,\mu m \text{ (default)}

    With ``optically_thin=True`` the opacity factor is replaced by its
    small-τ limit :math:`(\lambda_0/\lambda)^\beta`, giving the familiar
    :math:`\nu^{3+\beta} B_\nu` form (Casey 2012, §2).

    Parameters
    ----------
    wavelength_cm : array
        Wavelength in cm.
    T_eff : float
        Effective temperature in K (after CMB correction).
    dust_beta_ir : float
        Emissivity index.
    optically_thin : bool
        If True, use tau; if False, use (1 - exp(-tau)).
    lambda_0_cm : float
        Opacity pivot wavelength in cm. Default 200 µm.

    Notes
    -----
    **JIT-compatible**: yes, ``optically_thin`` is a static Python bool. Safe
    under ``grad`` and ``vmap`` in float32 as well as float64: both the exponent
    grouping and the ``1/expm1`` spelling below are chosen so that no squared
    denominator the reverse pass forms leaves the float32 range (#1439).
    """
    # Grouped as ``(h·c/k) / (lambda·T)``, NOT ``h·c / (lambda·k·T)`` (#1439).
    # Associativity holds for the value but not for the reverse pass: division's
    # derivative w.r.t. its denominator is ``-g·A/den**2``. Spelled with ``k``
    # in the denominator that square is ``(lambda·k·T)**2``; measured 2.3e-39
    # at the blue end of a UV-to-far-IR grid, *below* float32's smallest normal
    # 1.18e-38: so the reverse pass divided by zero and the gradient came back
    # NaN while the forward value stayed correct to seven digits. Folding the
    # tiny ``k`` into the numerator makes the denominator ``lambda·T``, whose
    # square is ~1e-7 at the same point. Measured: gradient NaN -> 9.9896e+04,
    # matching float64; float64 itself bit-identical.
    x = jnp.clip((_H_PLANCK * _C_CGS / _K_BOLTZMANN) / (wavelength_cm * T_eff), _X_MIN, _X_MAX)
    tau = (lambda_0_cm / wavelength_cm) ** dust_beta_ir
    opacity = tau if optically_thin else -jnp.expm1(-tau)
    # ``nu**3`` written out reaches ~2.7e49 on a UV-to-far-IR grid, eleven
    # decades past the float32 ceiling. Using (1/lambda)**3 [cm^-3] instead
    # drops a constant factor c**3, which cancels exactly: this is a *shape*,
    # normalized downstream by its own frequency integral, and both callers
    # (the graybody and the power-law amplitude tied to it at lambda_c) pick up
    # the same factor. Largest intermediate becomes ~1e18 (#1206).
    #
    # ``1/expm1(x)`` spelled ``exp(-x) / -expm1(-x)``: the same number, but the
    # denominator now lives in (0, 1] and its *square* is bounded by 1 in every
    # dtype. The raw form needs ``expm1(x)**2``, which passes float32's 3.4e38
    # at x ~ 44: half the clamp that guarded ``expm1``'s own forward overflow,
    # so that clamp could never have covered it (#1439).
    return opacity * (1.0 / wavelength_cm) ** 3 * jnp.exp(-x) / -jnp.expm1(-x)


def casey2012(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    dust_T: float = 35.0,
    dust_beta_ir: float = 1.8,
    dust_alpha_mir: float = 2.0,
    dust_lambda_0_um: float = 200.0,
    optically_thin: bool = False,
    redshift: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    r"""Casey (2012) graybody + mid-IR power law dust emission.

    Implements Casey (2012) [1]_ Eqs. 1-2: a general-opacity graybody
    (FIR peak) plus a mid-IR power law rising with wavelength up to the
    turnover :math:`\lambda_c`, where its amplitude is tied to the
    graybody (continuity) and a Gaussian cutoff hands over:

    .. math::

        S_\nu(\lambda) = N_{bb} \left[
            \left(1 - e^{-(\lambda_0/\lambda)^\beta}\right)
            \frac{\nu^3}{e^{h\nu/kT} - 1}
            + S^{gray}_\nu(\lambda_c)\,
              (\lambda/\lambda_c)^{\alpha}\, e^{-(\lambda/\lambda_c)^2}
        \right]

    with :math:`\lambda_0` from ``dust_lambda_0_um`` (default 200 µm per
    Casey 2012) and :math:`\lambda_c(\alpha, T)` from Eqs. 11-12. Every
    variable: :math:`T` = dust temperature [K], :math:`\beta` = emissivity
    index, :math:`\alpha` = mid-IR slope, :math:`\nu` = frequency [Hz],
    :math:`\lambda` = wavelength. The total frequency integral is
    normalized to ``L_absorbed``.

    This matches CIGALE's ``casey2012`` module term by term (parity
    verified against pcigale 2025.1; #1004: the previous closure carried
    a spurious Wien factor that annihilated the power law, an inverted
    power-law slope, and an optically-thin-only graybody).

    When ``redshift > 0``, the dust temperature is corrected for CMB
    heating (da Cunha et al. 2013 [2]_) and the observed flux is reduced
    by the CMB contrast factor.

    Parameters
    ----------
    wavelength_aa : array, shape (n_wave,)
        Wavelength grid in Angstrom (sorted ascending). [Å]
    L_absorbed : float
        Total absorbed luminosity (sets the normalization). Unit-agnostic:
        the output L_nu is in the same units per Hz.
    dust_T : float
        Dust temperature. Typical range: 20-60. [K]
    dust_beta_ir : float
        Dust emissivity index. Typical range: 1.5-2.0. [dimensionless]
    dust_alpha_mir : float
        Mid-IR power-law slope. Typical range: 1.5-2.5. [dimensionless]
    dust_lambda_0_um : float
        Graybody opacity pivot wavelength (µm). Default 200 µm (Casey 2012).
        [µm]
    optically_thin : bool
        If True, use the optically-thin graybody limit
        :math:`(\lambda_0/\lambda)^\beta \nu^3 / (e^{h\nu/kT}-1)`
        instead of the general form. The mid-IR power law is present in
        both variants. Default: False (Casey 2012 Eq. 1). [dimensionless]
    redshift : float
        Source redshift; > 0 applies the CMB corrections. Default 0.
        [dimensionless]

    Returns
    -------
    array, shape (n_wave,)
        Dust emission L_nu in ``[L_absorbed units] / Hz``.

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives;
    ``optically_thin`` is a static Python bool.

    **Gradient-safe**: yes, differentiable everywhere.

    References
    ----------
    .. [1] Casey, C. M., 2012, MNRAS, 425, 3094, Eqs. 1-2, 11-12.
       doi:10.1111/j.1365-2966.2012.21455.x, arXiv:1206.1595.
    .. [2] da Cunha, E. et al., 2013, ApJ, 766, 13 (CMB corrections).
       doi:10.1088/0004-637X/766/1/13, arXiv:1302.0844.
    """
    # CMB correction (no-op at z=0)
    T_eff = cmb_corrected_temperature(dust_T, redshift, dust_beta_ir)

    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm  # Hz, descending

    # Convert dust_lambda_0_um to cm
    lambda_0_cm = dust_lambda_0_um * 1.0e-4  # um to cm

    lambda_c_cm = _casey_lambda_c_cm(T_eff, dust_alpha_mir)
    graybody = _casey_graybody_nu(wavelength_cm, T_eff, dust_beta_ir, optically_thin, lambda_0_cm)
    # Power-law amplitude tied to the graybody at the turnover (Eq. 2).
    n_pl = _casey_graybody_nu(
        jnp.asarray(lambda_c_cm), T_eff, dust_beta_ir, optically_thin, lambda_0_cm
    )
    power_law = (
        n_pl
        * (wavelength_cm / lambda_c_cm) ** dust_alpha_mir
        * jnp.exp(-((wavelength_cm / lambda_c_cm) ** 2))
    )
    shape = graybody + power_law

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
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

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
    """Smith et al. (2007) PAH Drude profiles: mid-IR PAH building block.

    A sum of 18 PAH Drude profiles (normalized to Smith+2007 SINGS median
    strengths). This is a **PAH-only building block**, not a standalone
    energy-balanced dust emitter: it carries the aromatic-feature forest only
    (no thermal continuum), so its frequency integral is *not* renormalized to
    ``L_absorbed``: it is scaled by ``L_absorbed`` but deliberately leaves the
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
    **JIT-compatible**: yes, pure ``jnp`` primitives (a precomputed lookup in
    :mod:`~tengri.components.dust.dust_analytic_precompute` is preferred in the
    hybrid kernel; this is the direct full-wavelength evaluation).

    **Gradient-safe**: yes.

    **Not energy-balanced standalone**; see the summary above; excluded from
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
    select ``schreiber2018`` (``data/schreiber2018_templates.h5``) instead:
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
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

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
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

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
