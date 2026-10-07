# SPDX-License-Identifier: BSD-3-Clause
"""Polar dust extinction and graybody reemission for AGN.

Implements the X-CIGALE polar dust model (Yang et al. 2020, Section 2.2.2):
bi-conical polar dust with viewing-angle-independent absorption and
energy-conserving graybody FIR reemission. SMC extinction curve applied
to observer-frame disc attenuation (Type 1 sightlines only); absorption
is geometry-independent (Yang+2020 §2.2.2).

The Type 1/2 boundary uses a smooth sigmoid transition for differentiability.

References
----------

- Yang et al. 2020, MNRAS, 491, 740 (X-CIGALE polar dust § 2.2.2)
- Gordon et al. 2003, ApJ, 594, 279 (SMC extinction)
- Pei 1992, ApJ, 395, 130 (SMC parameterization used here)

"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn._phys import planck_lnu, wavelength_to_nu
from tengri.components.agn._polar_smc_opacity import SMC_OPACITY_EXT, SMC_OPACITY_WAVE_UM
from tengri.components.dust.attenuation import smc as smc_extinction_curve

# Physical constants (CGS / Angstrom-compatible)
from tengri.utils.physics_constants import C_AA as _C_AA

#: Wavelength grid [A] the graybody's normalization integral is taken on: 1000 A to
#: 10 cm, 1000 points per decade, which holds the Planck function of any dust
#: temperature the polar parameter allows (Wien side of a 2000 K body is below
#: 1e-40 of its peak at 1000 A).
_GRAYBODY_NORM_WAVE = np.geomspace(1.0e3, 1.0e9, 6001)

# SMC R_V from Pei (1992)
_RV_SMC = 2.93

#: The one width of the Type-1/2 transition, in :math:`\cos i` units, shared by
#: the polar-dust mask and the torus screen: both weights are the same logistic
#: in :math:`\cos i` about the same midpoint, so they sum to one at every
#: inclination. Narrow enough that a clearly face-on sightline is unscreened
#: even at a large torus optical depth, wide enough that the weight stays
#: differentiable across the boundary.
TYPE_TRANSITION_WIDTH = 0.025

#: Sigmoid steepness in :math:`\cos i`, the reciprocal of
#: :data:`TYPE_TRANSITION_WIDTH`.
_SIGMOID_SHARPNESS = 1.0 / TYPE_TRANSITION_WIDTH

#: Polar-cone geometries. ``"skirtor"``: the cone share is the integral of the
#: anisotropic disc law over the escape cone (Stalevski et al. 2012). ``"fritz"``:
#: the solid-angle fraction of the cone, :math:`1 - \cos\theta_c`, with
#: :math:`\theta_c` the dust-free half-angle (the Fritz et al. 2006 torus).
_POLAR_GEOMETRIES = ("skirtor", "fritz")


def type1_weight(
    cos_inc: float,
    opening_angle_deg: float,
    sharpness: float = _SIGMOID_SHARPNESS,
) -> jnp.ndarray:
    r"""Smooth sigmoid mask: 1 for Type 1 (face-on), 0 for Type 2 (edge-on).

    The sightline is Type 1 when it lies inside the dust-free polar cone, the
    cone of half-angle :math:`90^\circ - \Phi` about the axis, that is
    :math:`i < 90^\circ - \Phi`, :math:`\cos i > \sin\Phi`, with
    :math:`\Phi` the torus half-opening angle from the equatorial plane.

    Parameters
    ----------
    cos_inc : float
        Cosine of inclination angle. 1 = face-on, 0 = edge-on.
    opening_angle_deg : float
        Torus half-opening angle in degrees (measured from equator).
    sharpness : float
        Sigmoid steepness in :math:`\cos i`. Default is the reciprocal of
        :data:`TYPE_TRANSITION_WIDTH`, the width the torus screen uses.

    Returns
    -------
    mask : scalar
        Value in [0, 1]. ~1 for Type 1, ~0 for Type 2, ~0.5 at boundary.
    """
    cos_threshold = jnp.cos(jnp.radians(90.0 - opening_angle_deg))
    return jax.nn.sigmoid((cos_inc - cos_threshold) * sharpness)


#: Name the polar mask has carried; the screens and the generic tori call :func:`type1_weight`.
_type1_mask = type1_weight


def resolve_polar_opening_angle(agn_polar_oa: float, torus_opening_angle: float) -> jnp.ndarray:
    """The polar cone's opening angle: the explicit override, else the torus's own.

    Parameters
    ----------
    agn_polar_oa : float
        The ``agn_polar_oa`` override [deg]. A value ``<= 0`` (the declared
        default, ``0``) means "follow the torus".
    torus_opening_angle : float
        The selected torus's own opening angle, in the convention of
        ``agn_polar_oa`` (half-opening angle from the equatorial plane) [deg].

    Returns
    -------
    ndarray, scalar
        The angle the polar mask, the cone share and the torus screen all read.
        [deg]

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes, both branches are smooth
    and the select carries no gradient to the unselected one.
    """
    override = jnp.asarray(agn_polar_oa)
    return jnp.where(override > 0.0, override, jnp.asarray(torus_opening_angle))


#: Wavelength [nm] where the Calzetti et al. (2000) long-wavelength polynomial
#: ``2.659 (-1.857 + 1040/lambda_nm) + 4.05`` crosses zero (3115 nm).
_CALZETTI_ZERO_NM = 1040.0 / (1.857 - 4.05 / 2.659)


def calzetti2000_extinction_curve(wavelength: jnp.ndarray) -> jnp.ndarray:
    """Calzetti et al. (2000) dust extinction curve.

    Piecewise polynomial extinction law valid for 0.12–2.2 um (1200–22000 A).
    Widely used for star-forming galaxies and AGN optical/UV extinction.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength in Angstrom.

    Returns
    -------
    k_lambda : ndarray, shape (n_wave,)
        Extinction coefficient k(lambda) = A(lambda) / E(B-V).
        [dimensionless]

    Notes
    -----
    Calzetti et al. (2000) parameterization:

    For :math:`\\lambda < 630` nm (6300 A):

    .. math::

        k(\\lambda) = 2.659 \\times (-2.156 + 1509/\\lambda_{\\rm nm}
                    - 0.198 \\times 10^6/\\lambda_{\\rm nm}^2
                    + 0.011 \\times 10^9/\\lambda_{\\rm nm}^3) + 4.05

    For :math:`\\lambda \\geq 630` nm:

    .. math::

        k(\\lambda) = 2.659 \\times (-1.857 + 1040/\\lambda_{\\rm nm}) + 4.05

    where :math:`\\lambda_{\\rm nm}` is wavelength in nanometers. The fit is published for
    0.12 to 2.2 um; past the zero of the second polynomial (3115 nm) the curve is held at zero
    rather than going negative. CIGALE's ``skirtor2016`` law 1 leaves it unfloored, so the two
    differ beyond 3.1 um, which holds 7.9 % of the default disc's power: the absorbed polar
    power is 0.57 % higher here at E(B-V) = 0.1. The dust-attenuation Calzetti
    (:func:`tengri.components.dust.attenuation.calzetti`) is the same polynomial normalized
    to :math:`k(V) = 1` and is not reused here because that normalization moves ``k`` by
    5e-4.

    **JIT-compatible**: yes, uses ``jnp`` primitives.

    **Gradient-safe**: yes, fully differentiable.

    **Reference**: Implements CIGALE ``skirtor2016.py`` ``k_ext()``
    (Boquien et al. 2019 [2]_); validated against its output.

    References
    ----------
    .. [1] D. Calzetti et al., "The Dust Content and Opacity of Actively
       Star-forming Galaxies," ApJ, 533, 682 (2000).
       https://doi.org/10.1086/308692
    .. [2] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy
       Emission," A&A, 622, A103 (2019). arXiv:1811.03094.
       https://doi.org/10.1051/0004-6361/201834156
    """
    # Convert Angstrom to nanometers
    wave_nm = wavelength / 10.0

    # Split into two regimes
    x_short = 1.0 / wave_nm  # 1/lambda for short wavelengths

    # Short wavelength (lambda < 630 nm, x > 1/630)
    short_part = (
        2.659 * (-2.156 + 1509.0 * x_short - 0.198e6 * x_short**2 + 0.011e9 * x_short**3) + 4.05
    )

    # Long wavelength (lambda >= 630 nm)
    long_part = 2.659 * (-1.857 + 1040.0 * x_short) + 4.05

    # Select based on wavelength
    k_lambda = jnp.where(wave_nm < 630.0, short_part, long_part)

    # Beyond the zero crossing of the long-wavelength polynomial the curve is held at zero:
    # the fit is published for 0.12-2.2 um and goes negative past 3.1 um (min -0.80 at
    # 30 um), which would make the absorbed power negative there.
    return jnp.where(wave_nm < _CALZETTI_ZERO_NM, k_lambda, 0.0)


def gaskell2004_extinction_curve(wavelength: jnp.ndarray) -> jnp.ndarray:
    """Gaskell et al. (2004) dust extinction curve.

    Polynomial extinction law parameterized in inverse wavelength space.
    Developed from observations of AGN and suitable for AGN UV/optical
    extinction studies.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength in Angstrom.

    Returns
    -------
    k_lambda : ndarray, shape (n_wave,)
        Extinction coefficient k(lambda) = A(lambda) / E(B-V).
        [dimensionless]

    Notes
    -----
    Gaskell et al. (2004) parameterization in terms of :math:`x = 1000/\\lambda`
    (where :math:`\\lambda` is in nm):

    For :math:`x < 3.69` (wavelength > ~271 nm, 2710 A):

    .. math::

        A(\\lambda) / A_V = -0.8175 + 1.5848 x - 0.3774 x^2 + 0.0296 x^3

    For :math:`x \\geq 3.69`:

    .. math::

        A(\\lambda) / A_V = 1.3468 + 0.0087 x

    The result is then divided by :math:`A_B / A_V = 1.182` to convert from
    A(λ)/A_V to k(λ) = A(λ) / E(B-V).

    **JIT-compatible**: yes, uses ``jnp`` primitives.

    **Gradient-safe**: yes, fully differentiable.

    **Reference**: Implements CIGALE ``skirtor2016.py`` ``k_ext()``
    (Boquien et al. 2019 [2]_); validated against its output.

    References
    ----------
    .. [1] C. M. Gaskell et al., "A Redetermination of the Reddening of
       AGNs," ApJ, 616, 147 (2004).
       https://doi.org/10.1086/423885
    .. [2] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy
       Emission," A&A, 622, A103 (2019). arXiv:1811.03094.
       https://doi.org/10.1051/0004-6361/201834156
    """
    # Convert Angstrom to nanometers
    wave_nm = wavelength / 10.0

    # x = 1000 / lambda_nm
    x = 1000.0 / wave_nm

    # A(lambda) / A_V from polynomial
    a_av_short = -0.8175 + 1.5848 * x - 0.3774 * x**2 + 0.0296 * x**3
    a_av_long = 1.3468 + 0.0087 * x

    # Select regime
    a_av = jnp.where(x < 3.69, a_av_short, a_av_long)

    # Convert A(lambda)/A_V to k(lambda) = A(lambda)/E(B-V)
    # Using CIGALE's convention: k = a_av / 0.182
    # This ensures positivity for all wavelengths
    k_lambda = jnp.maximum(a_av / 0.182, 0.0)

    return k_lambda


def bongiorno2012_extinction_curve(wavelength: jnp.ndarray) -> jnp.ndarray:
    r"""Bongiorno et al. (2012) power-law extinction curve for AGN polar dust.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength. [Angstrom]

    Returns
    -------
    k_lambda : ndarray, shape (n_wave,)
        :math:`k(\lambda) = A(\lambda)/E(B-V)`. [dimensionless]

    Notes
    -----
    .. math::

        k(\lambda) = 1.39\,\lambda_{\mu{\rm m}}^{-1.2},

    the SMC-like power law of Bongiorno et al. (2012) [1]_ for
    :math:`\lambda \ge 100` nm. Below 100 nm the curve is the shape of the Weingartner &
    Draine (2001) SMC-bar dust-mixture extinction (Draine's public table, repackaged in
    ``tengri.components.agn._polar_smc_opacity`` with its provenance), rescaled to equal
    the power law at 100 nm and interpolated linearly in wavelength. This is the splice of
    CIGALE's ``extinction_law = 0``, which uses its own SMC-mixture table; the two shapes
    agree (shape relative to 100 nm) within 5.5 per cent above 35 nm, within 15.5 per cent between
    10 and 35 nm, and to 19 per cent at 1 nm. CIGALE rescales at the last grid point below
    100 nm rather than at 100 nm itself; for a dense grid the two differ by less than the
    opacity's change across one grid step. Wavelengths below the table's 10 A edge take its
    edge value.

    **JIT-compatible**: yes. **Gradient-safe**: yes.

    References
    ----------
    .. [1] A. Bongiorno et al., MNRAS, 427, 3103 (2012); the
       :math:`1.39\,\lambda_{\mu{\rm m}}^{-1.2}` form as used by CIGALE
       ``skirtor2016`` (Boquien et al., A&A, 622, A103, 2019,
       arXiv:1811.03094).
    """
    wave_um = jnp.asarray(wavelength) * 1.0e-4
    k_power = 1.39 * wave_um ** (-1.2)
    table_wave = jnp.asarray(SMC_OPACITY_WAVE_UM, dtype=wave_um.dtype)
    table_ext = jnp.asarray(SMC_OPACITY_EXT, dtype=wave_um.dtype)
    shape = jnp.interp(wave_um, table_wave, table_ext)
    shape_at_100nm = jnp.interp(0.1, table_wave, table_ext)
    k_at_100nm = 1.39 * 0.1 ** (-1.2)
    return jnp.where(wave_um < 0.1, shape * (k_at_100nm / shape_at_100nm), k_power)


#: Extinction laws the polar dust accepts. ``smc`` is :math:`A/A_V` (Pei 1992)
#: and is scaled by :math:`R_V`; the others are already per unit :math:`E(B-V)`.
POLAR_LAWS = ("smc", "calzetti", "gaskell", "bongiorno")


def polar_cone_covering_fraction(opening_angle_deg: float) -> jnp.ndarray:
    r"""Fraction of the disc's bolometric luminosity within the polar cone.

    X-CIGALE's polar dust (Yang et al. 2020 [1]_, section 2.2.2) sits in the
    bicone above and below the dusty torus -- the region a Type-1 (face-on)
    sightline escapes through. This function is that cone's *solid-angle*
    share of the disc's own anisotropic emission pattern, i.e. what fraction
    of the disc's total (one-hemisphere) luminosity the polar dust can ever
    intercept, independent of any particular observer's line of sight
    (:func:`polar_dust_extinction` already handles the *observed*,
    inclination-dependent transmission separately, via its Type-1/2 mask).

    Derivation. The disc's specific intensity as a function of polar angle
    :math:`\theta` (measured from the pole, :math:`\theta=0` face-on)
    follows the anisotropic thin-disc emission law used throughout the
    SKIRTOR torus model (Stalevski et al. 2012 [2]_, eq. 3):

    .. math::

        I(\theta) = I_0 \cos\theta\,(1 + 2\cos\theta), \qquad
        0 \le \theta \le \pi/2.

    Integrating over the one-sided hemisphere (:math:`\mu = \cos\theta`)
    gives the disc's total one-face luminosity

    .. math::

        L_{\rm bol} = 2\pi \int_0^{\pi/2} I(\theta)\,\sin\theta\,d\theta
                    = 2\pi I_0 \int_0^1 \mu(1+2\mu)\,d\mu
                    = \frac{7\pi}{3} I_0.

    The polar/escape cone spans polar angle :math:`\theta \in [0,
    \theta_{\rm max}]` with :math:`\theta_{\rm max} = 90^\circ - \Phi`
    (:math:`\Phi` = ``opening_angle_deg``, the torus half-opening angle
    measured from the equator -- the same convention
    :func:`polar_dust_extinction`'s Type-1/2 mask uses, where
    :math:`\mu_{\rm max} = \cos\theta_{\rm max} = \sin\Phi`). Integrating
    :math:`I(\theta)` over just that range and dividing by :math:`L_{\rm
    bol}` gives the cone's share:

    .. math::

        f_{\rm cone}(\Phi) = \frac{2\pi\int_0^{\theta_{\rm max}}
        I(\theta)\sin\theta\,d\theta}{L_{\rm bol}}
        = 1 - \frac{3}{7}\sin^2\Phi - \frac{4}{7}\sin^3\Phi.

    :math:`f_{\rm cone}(0^\circ) = 1` (an infinitesimally thin equatorial
    torus leaves the ENTIRE hemisphere as escape cone) and
    :math:`f_{\rm cone}(90^\circ) = 0` (the torus edge reaches the pole, no
    escape cone at all), monotonically decreasing in between.

    Parameters
    ----------
    opening_angle_deg : float
        Torus half-opening angle :math:`\Phi` [deg, measured from the
        equator] -- ``agn_polar_oa``.

    Returns
    -------
    f_cone : ndarray, scalar
        Fraction of the disc's bolometric luminosity within the polar cone,
        :math:`\in [0, 1]`. Dimensionless.

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` primitives only. **Gradient-safe**:
    yes, :math:`\sin\Phi` is smooth everywhere.

    :func:`polar_dust_reemission_lnu` multiplies the geometry-independent
    absorbed luminosity (:func:`polar_dust_extinction`'s ``l_absorbed``,
    integrated) by this factor before normalizing the re-emission graybody,
    so the re-emitted luminosity tracks ``agn_polar_oa`` as the declared
    parameter's description promises ("sets covering fraction").

    References
    ----------
    .. [1] Yang, A., et al. 2020, MNRAS, 491, 740 (X-CIGALE polar dust,
       section 2.2.2). https://doi.org/10.1093/mnras/stz3001
    .. [2] Stalevski, M. et al. 2012, MNRAS, 420, 2756 (disc anisotropic
       emission law). arXiv:1109.1286.
    """
    return polar_cone_covering_factor(opening_angle_deg, reference="bolometric")


#: The two disc reference luminosities a polar-cone factor can be taken against.
#:
#: ``"bolometric"`` -- the disc's hemisphere-integrated luminosity,
#: :math:`L_{\rm bol} = (7\pi/3) I_0`. ``"face_on"`` -- the face-on value
#: :math:`\int L(\theta{=}0)\,d\lambda` carried by a SKIRTOR flux table
#: (already multiplied by :math:`4\pi d^2`, so it needs the compensating
#: :math:`1/4\pi`), which is CIGALE's convention.
_POLAR_CONE_REFERENCES = ("bolometric", "face_on")


def polar_cone_covering_factor(
    opening_angle_deg: float, *, reference: str = "bolometric", geometry: str = "skirtor"
) -> jnp.ndarray:
    r"""Polar-cone factor against a named disc reference luminosity (R60).

    The geometry is one thing measured two ways. Integrating the SKIRTOR
    anisotropic disc law :math:`I(\theta) = I_0\cos\theta\,(1 + 2\cos\theta)`
    over the escape cone :math:`\theta \in [0, 90^\circ - \Phi]` gives one
    solid-angle share, but the number you multiply depends on which disc
    luminosity you are handed:

    .. math::

        f_{\rm cone}(\Phi) &= 1 - \tfrac{3}{7}\sin^2\Phi
                              - \tfrac{4}{7}\sin^3\Phi
        \qquad\text{(against } L_{\rm bol} = (7\pi/3)\,I_0\text{)} \\
        g(\Phi) &= \tfrac{7}{18} - \tfrac{1}{6}\sin^2\Phi
                   - \tfrac{2}{9}\sin^3\Phi
        \qquad\text{(against } \textstyle\int L(\theta{=}0)\,d\lambda\text{)}

    with :math:`\Phi` the torus half-opening angle from the equator. The two
    are **exactly proportional**:

    .. math::

        g(\Phi) \equiv \tfrac{7}{18}\,f_{\rm cone}(\Phi),
        \qquad f_{\rm cone}/g = 18/7 = 2.571428\ldots

    so they never disagree about geometry, only about the reference frame. The
    :math:`7/18` is the SKIRTOR flux-table bookkeeping: those models are given
    in flux already multiplied by :math:`4\pi d^2`, and an anisotropic source
    needs the :math:`1/4\pi` back, which CIGALE's ``skirtor2016`` folds into
    :math:`g` (see its ``l_ext`` derivation).

    **Which reference applies is a normalization-policy question, and the
    caller must state it** -- picking one silently moves the polar re-emission
    by 2.571x. Under ``agn_norm='cigale_joint'`` with the SKIRTOR torus the
    disc is tied to CIGALE's own inclination-specific ``disk`` template
    (``agn_power x R``), so ``reference='face_on'``. Under
    ``'independent'``/``'conserving'`` the disc carries the
    hemisphere-integrated bolometric :math:`10^{\rm agn\_log\_lbol}`, so
    ``reference='bolometric'``.

    Parameters
    ----------
    opening_angle_deg : float
        Torus half-opening angle :math:`\Phi` [deg, from the equator] --
        ``agn_polar_oa``.
    reference : {'bolometric', 'face_on'}, optional
        The disc reference luminosity the factor will multiply. Default
        ``'bolometric'``.
    geometry : {'skirtor', 'fritz'}, optional
        The torus whose cone this is. ``'skirtor'`` (default) is the form above.
        ``'fritz'`` is the solid-angle fraction of the dust-free cone of
        half-angle :math:`\theta_c = 90^\circ - \Phi`,
        :math:`1 - \cos\theta_c = 1 - \sin\Phi`, the form CIGALE's
        ``fritz2006`` uses (``l_ext = (1 - cos(half)) * int disk (1 - ext)``);
        it has no flux-table bookkeeping, so only ``reference='bolometric'``
        applies.

    Returns
    -------
    factor : ndarray, scalar
        Dimensionless. ``f_cone`` for ``'bolometric'`` (1 at
        :math:`\Phi = 0^\circ`, 0 at :math:`90^\circ`), ``g`` for
        ``'face_on'`` (:math:`7/18` at :math:`0^\circ`, 0 at
        :math:`90^\circ`).

    Raises
    ------
    ValueError
        On an unrecognized ``reference``. There is no defaulting here: an
        unknown frame is a 2.571x error with no tolerance at which it is small.

    Notes
    -----
    **JIT-compatible**: yes, ``jnp`` only (``reference`` is a static Python
    string). **Gradient-safe**: yes, :math:`\sin\Phi` is smooth everywhere.

    References
    ----------
    .. [1] Yang, A., et al. 2020, MNRAS, 491, 740 (X-CIGALE polar dust,
       section 2.2.2). https://doi.org/10.1093/mnras/stz3001
    .. [2] Stalevski, M. et al. 2012, MNRAS, 420, 2756 (the disc anisotropic
       emission law both forms integrate). arXiv:1109.1286.
    """
    if reference not in _POLAR_CONE_REFERENCES:
        raise ValueError(
            f"polar_cone_covering_factor: reference={reference!r} is not one of "
            f"{_POLAR_CONE_REFERENCES}. The two frames differ by exactly 18/7 "
            f"(2.5714x), so there is no safe default: state which disc "
            f"luminosity the factor multiplies. 'face_on' for CIGALE's "
            f"inclination-specific disk template (agn_norm='cigale_joint' with "
            f"the SKIRTOR torus), 'bolometric' for the hemisphere-integrated "
            f"10**agn_log_lbol ('independent'/'conserving')."
        )
    if geometry not in _POLAR_GEOMETRIES:
        raise ValueError(
            f"polar_cone_covering_factor: geometry={geometry!r} is not one of {_POLAR_GEOMETRIES}."
        )
    sin_phi = jnp.sin(jnp.radians(jnp.asarray(opening_angle_deg)))
    if geometry == "fritz":
        if reference != "bolometric":
            raise ValueError(
                "polar_cone_covering_factor: the Fritz cone share is referenced to the "
                "disc's own luminosity (reference='bolometric'); the 'face_on' frame is a "
                "SKIRTOR flux-table convention."
            )
        return 1.0 - sin_phi
    f_cone = 1.0 - (3.0 / 7.0) * sin_phi**2 - (4.0 / 7.0) * sin_phi**3
    return f_cone if reference == "bolometric" else (7.0 / 18.0) * f_cone


def _polar_law_curve(law: str, wavelength: jnp.ndarray) -> tuple[jnp.ndarray, float]:
    """The extinction curve and its :math:`R` for a named polar-dust law.

    Raises
    ------
    ValueError
        On a name outside :data:`POLAR_LAWS`; there is no fallback law.
    """
    if law == "smc":
        return smc_extinction_curve(wavelength), _RV_SMC
    if law == "calzetti":
        return calzetti2000_extinction_curve(wavelength), 1.0
    if law == "gaskell":
        return gaskell2004_extinction_curve(wavelength), 1.0
    if law == "bongiorno":
        return bongiorno2012_extinction_curve(wavelength), 1.0
    raise ValueError(
        f"agn_polar_law={law!r} is not a known polar-dust extinction law; "
        f"choose one of {POLAR_LAWS}. 'smc' is Pei (1992) SMC Bar scaled by "
        f"R_V = 2.93, 'bongiorno' the 1.39 lambda_um^-1.2 power law CIGALE "
        f"skirtor2016 uses by default."
    )


def polar_dust_extinction(
    l_nu: jnp.ndarray,
    wavelength: jnp.ndarray,
    cos_inc: float,
    opening_angle_deg: float,
    ebv: float,
    law: str = "smc",
    sharpness: float = _SIGMOID_SHARPNESS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply polar dust extinction to AGN luminosity.

    Extinction is applied only to Type 1 sightlines (face-on), with a
    smooth sigmoid transition at the Type 1/2 boundary.

    Parameters
    ----------
    l_nu : array, shape (n_wave,)
        Input luminosity density [Lsun/Hz or any consistent unit].
    wavelength : array, shape (n_wave,)
        Wavelength in Angstrom.
    cos_inc : float
        Cosine of inclination. 1 = face-on (Type 1), 0 = edge-on (Type 2).
        [dimensionless, 0–1]
    opening_angle_deg : float
        Torus half-opening angle in degrees (from equator). [degrees]
    ebv : float
        Color excess E(B-V) for the polar dust. 0 = no extinction.
        [dimensionless, mag]
    law : str
        Extinction law name: ``"smc"`` (Pei 1992), ``"calzetti"`` (Calzetti
        et al. 2000), ``"gaskell"`` (Gaskell et al. 2004) or ``"bongiorno"``
        (Bongiorno et al. 2012). Default: ``"smc"``. Any other name raises
        :class:`ValueError`.
    sharpness : float
        Sigmoid steepness at the Type 1/2 boundary. [dimensionless]

    Returns
    -------
    l_nu_attenuated : array, shape (n_wave,)
        Observer-frame disc luminosity after Type-1-masked attenuation.
        Same units as input l_nu. Type-2 sightlines are unchanged (mask ≈ 0).
    l_absorbed : array, shape (n_wave,)
        Absorbed luminosity density (per wavelength bin): bi-conical dust
        absorbs a fraction (1 - exp(-tau_lambda)) regardless of viewing angle.
        Always >= 0. Same units as input l_nu.

    Notes
    -----
    **Absorption vs. Attenuation:**

    - ``l_absorbed`` (geometry-independent) is the disc photon fraction intercepted
      by the bi-conical polar dust (Yang+2020 §2.2.2). This drives the
      graybody FIR reemission and is viewed isotropically.
    - ``l_nu_attenuated`` (Type-1-masked) is the observer-frame disc after
      passing through the near-cone geometry. Only face-on sightlines see
      attenuation; edge-on sightlines (Type 2) have the disc already screened
      by the equatorial torus (handled upstream), so ``l_nu_attenuated = l_nu``.

    **JIT-compatible**: yes, uses ``jnp`` primitives and smooth sigmoid.
    """
    k_lambda, r_v = _polar_law_curve(law, wavelength)

    # A(lambda) = E(B-V) * R_V * k(lambda)
    # Transmission: 10^{-0.4 * A(lambda)} = exp(-0.921 * A(lambda))
    tau_lambda = 0.921 * ebv * r_v * k_lambda
    extinction_factor = jnp.exp(-tau_lambda)  # fraction transmitted

    # Polar-dust absorption is geometry-independent: the bi-conical dust always
    # intercepts the same disc-photon fraction (set by E(B-V)) regardless of
    # observer viewing angle. Re-emission is isotropic, so observers at any
    # inclination see the FIR bump. No floor: for E(B-V) >= 0 and a curve
    # k >= 0, tau >= 0 and 1 - exp(-tau) >= 0, so the absorbed power is
    # non-negative by construction. A ``maximum(., 0)`` here ties its two
    # arguments at E(B-V) = 0, the lower edge of the prior, and JAX splits the
    # derivative evenly between them, halving the gradient at the edge.
    l_absorbed = l_nu * (-jnp.expm1(-tau_lambda))

    # Type 1 mask: 1 for face-on (extinct), 0 for edge-on (no effect)
    mask = _type1_mask(cos_inc, opening_angle_deg, sharpness)

    # Observed disc attenuation is gated by Type-1 mask: only face-on sight-lines
    # look through the near polar cone. Type-2 sightlines have the disc already
    # screened by the equatorial torus (handled upstream), so we leave l_nu
    # unchanged here.
    effective_transmission = 1.0 - mask * (1.0 - extinction_factor)
    l_nu_attenuated = l_nu * effective_transmission

    return l_nu_attenuated, l_absorbed


def polar_dust_emission(
    l_absorbed_total: float,
    wavelength: jnp.ndarray,
    temperature: float = 100.0,
    beta: float = 1.6,
    lambda_0: float = 2e6,
) -> jnp.ndarray:
    """Graybody reemission from polar dust.

    Energy-conserving: the integral of the reemitted spectrum equals the
    total absorbed luminosity.

    Parameters
    ----------
    l_absorbed_total : float
        Total absorbed luminosity (scalar, integrated over frequency).
        Same units as input l_nu * delta_nu.
    wavelength : array, shape (n_wave,)
        Wavelength grid [Angstrom].
    temperature : float
        Dust temperature [K]. Default 100.
    beta : float
        Dust emissivity index [dimensionless]. Default 1.6.
    lambda_0 : float
        Reference wavelength for optical depth [Angstrom].
        Default 2e6 (= 200 um).

    Returns
    -------
    l_nu_reemit : array, shape (n_wave,)
        Reemitted luminosity density [same units as input l_absorbed_total].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` primitives only.

    **Gradient-safe**: yes, fully differentiable.
    """
    # Graybody: L_nu proportional to (1 - exp(-(lambda_0/lambda)^beta)) * B_nu(T)
    opacity_factor = 1.0 - jnp.exp(-((lambda_0 / wavelength) ** beta))
    b_nu = planck_lnu(wavelength_to_nu(wavelength), temperature)
    unnormalized = opacity_factor * b_nu

    # Normalize so that integral(L_reemit * dnu) = l_absorbed_total. The
    # integral is taken over a FIXED frequency grid (``_GRAYBODY_NORM_WAVE``),
    # never over the caller's wavelength array, so the normalization does not
    # move with where that array starts or how it is sampled: the graybody's
    # shape at any wavelength, and its total, are properties of (T, beta) alone.
    wave_norm = jnp.asarray(_GRAYBODY_NORM_WAVE, dtype=unnormalized.dtype)
    nu_norm = _C_AA / wave_norm
    unnorm_norm = (1.0 - jnp.exp(-((lambda_0 / wave_norm) ** beta))) * planck_lnu(
        wavelength_to_nu(wave_norm), temperature
    )
    integral = -jnp.trapezoid(unnorm_norm, nu_norm)
    # Avoid division by zero when integral is tiny (e.g., all wavelengths
    # far from the emission peak)
    safe_integral = jnp.where(integral > 0.0, integral, 1.0)
    norm = l_absorbed_total / safe_integral

    return norm * unnormalized


def anisotropic_polar_luminosity(
    l_nu_disk: jnp.ndarray,
    wavelength: jnp.ndarray,
    opening_angle_deg: float,
    extinction_factor: jnp.ndarray,
) -> float:
    """Total extincted luminosity with anisotropic disc geometry.

    Computes the bolometric luminosity accounting for the anisotropic emission
    pattern of the disc as seen through the polar dust. The disc emission varies
    with inclination angle θ as L(θ,λ) ∝ A(λ) · cosθ · (1 + 2cosθ), and the
    observable luminosity is averaged over a solid angle.

    This models the viewing-angle dependent attenuation for an anisotropic
    accretion disc viewed through a clumpy torus with opening angle.

    Parameters
    ----------
    l_nu_disk : array, shape (n_wave,)
        Intrinsic disc luminosity density [erg/s/Hz].
    wavelength : array, shape (n_wave,)
        Wavelength in Angstrom.
    opening_angle_deg : float
        Torus half-opening angle in degrees (from equator). [degrees]
    extinction_factor : array, shape (n_wave,)
        Wavelength-dependent transmission through polar dust
        (i.e., exp(-tau_lambda) from :func:`polar_dust_extinction`).
        [dimensionless, 0–1]

    Returns
    -------
    l_total : float
        Total extincted luminosity integrated over frequency and solid angle
        [erg/s].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` primitives only.

    The anisotropic geometry factor is derived from CIGALE's SKIRTOR module.
    For a given opening angle (related to the torus geometry), the average
    over viewing angles of the anisotropic factor is:

    .. math::

        \\langle f_{\\rm aniso} \\rangle = \\frac{7}{18} - \\frac{\\sin^2 \\Phi}{6}
                                          - \\frac{2 \\sin^3 \\Phi}{9}

    where :math:`\\Phi` is the torus half-opening angle in degrees.

    The extincted luminosity is then:

    .. math::

        L_{\\rm ext} = f_{\\rm aniso} \\int L_\\nu (1 - A_\\nu) \\, d\\nu

    This accounts for both the varying disc brightness with inclination and
    the wavelength-dependent extinction through the polar dust.

    **Reference**: Implements CIGALE ``skirtor2016.py`` ``agn_lnu_ir``
    function (Boquien et al. 2019 [2]_); validated against its output.

    References
    ----------
    .. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
       torus around AGN, the influence of clumping," MNRAS, 420, 2756 (2012).
       arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x
    .. [2] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy
       Emission," A&A, 622, A103 (2019). arXiv:1811.03094.
       https://doi.org/10.1051/0004-6361/201834156
    """
    # Compute anisotropic geometry factor
    sin_oa = jnp.sin(jnp.radians(opening_angle_deg))
    aniso_factor = 7.0 / 18.0 - sin_oa**2 / 6.0 - (2.0 / 9.0) * sin_oa**3

    # ABSORBED disc flux per unit frequency: (1 - transmission) × L_nu.
    # ``extinction_factor`` is the wavelength-dependent transmission
    # (exp(-tau_lambda)); the polar dust absorbs the complementary
    # fraction. Matches CIGALE skirtor2016.py:368:
    # ``l_ext = ... × np.trapz(AGN1.disk * (1.0 - ext_fac), x=AGN1.wl)``.
    l_nu_absorbed = l_nu_disk * (1.0 - extinction_factor)

    # Integrate over frequency: convert wavelength integral to frequency integral
    # dnu = -c/lambda^2 dlambda, so |dnu| = c/lambda^2 |dlambda|
    nu = _C_AA / wavelength
    # ``nu`` is descending, so the trapezoid integral is negative; negate for
    # the physical absorbed power. Never reverse the operands: reversed-array
    # times broadcast scalar is silently zeroed under MLX compile on Apple
    # GPU (jax-mps#232, #2295), and ``jnp.trapezoid`` multiplies by 0.5.
    l_total = -jnp.trapezoid(l_nu_absorbed, nu)

    # Apply anisotropic geometry factor
    l_total = aniso_factor * l_total

    return jnp.maximum(l_total, 0.0)


def polar_dust_total(
    l_nu_disc: jnp.ndarray,
    wavelength: jnp.ndarray,
    cos_inc: float,
    opening_angle_deg: float,
    ebv: float,
    temperature: float = 100.0,
    beta: float = 1.6,
    lambda_0: float = 2e6,
    law: str = "smc",
    sharpness: float = _SIGMOID_SHARPNESS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply polar dust extinction and compute graybody reemission.

    Convenience function combining :func:`polar_dust_extinction` and
    :func:`polar_dust_emission`.

    Parameters
    ----------
    l_nu_disc : array, shape (n_wave,)
        Input AGN disc luminosity density.  Unit-agnostic: output units
        match input (e.g. erg/s/Hz in → erg/s/Hz out).
    wavelength : array, shape (n_wave,)
        Wavelength in Angstrom.
    cos_inc : float
        Cosine of inclination. 1 = face-on, 0 = edge-on.
    opening_angle_deg : float
        Torus half-opening angle in degrees.
    ebv : float
        Color excess E(B-V).
    temperature : float
        Polar dust temperature in Kelvin.
    beta : float
        Dust emissivity index.
    lambda_0 : float
        Reference wavelength for optical depth in Angstrom.
    law : str
        Extinction law name.
    sharpness : float
        Sigmoid steepness at the Type 1/2 boundary.

    Returns
    -------
    l_nu_attenuated : array, shape (n_wave,)
        Attenuated disc luminosity (same units as input).
    l_nu_reemit : array, shape (n_wave,)
        Graybody reemission from polar dust (same units as input).
    """
    l_nu_attenuated, l_absorbed = polar_dust_extinction(
        l_nu_disc, wavelength, cos_inc, opening_angle_deg, ebv, law, sharpness
    )

    # Total absorbed luminosity: integrate l_absorbed over frequency
    nu = _C_AA / wavelength
    delta_nu = jnp.abs(jnp.diff(nu))
    delta_nu = jnp.concatenate([delta_nu[:1], 0.5 * (delta_nu[:-1] + delta_nu[1:]), delta_nu[-1:]])
    l_absorbed_total = jnp.sum(l_absorbed * delta_nu)

    l_nu_reemit = polar_dust_emission(l_absorbed_total, wavelength, temperature, beta, lambda_0)

    return l_nu_attenuated, l_nu_reemit
