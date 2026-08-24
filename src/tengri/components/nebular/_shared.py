# SPDX-License-Identifier: BSD-3-Clause
"""Shared utilities for nebular emission backends.

Functions extracted from individual backends to eliminate duplication.
"""

from __future__ import annotations

import math as _math

import jax
import jax.numpy as jnp

from tengri.components.nebular._constants import (
    _C_CGS,
    _H_PLANCK,
    _LOG10_ZSUN,
    _LOG_OH_OFFSET,
    _LSUN_ERG,
    _LYMAN_LIMIT,
)
from tengri.utils.physics_constants import C_KM_S as _C_KM_S, K_BOLTZ as _K_BOLTZ
from tengri.utils.scale import pow10

#: ``log10`` of the two constants deferred out of the Q_H integrand (#1568).
#: Python floats, evaluated once at import in float64, so they enter the graph
#: as constants and never as a float32-rounded array.
_LOG10_LSUN_ERG = float(_math.log10(_LSUN_ERG))
_LOG10_H_PLANCK = float(_math.log10(_H_PLANCK))

# ── Line placement ────────────────────────────────────────────────


def _nu_quadrature_weights(obs_wavelengths: jnp.ndarray) -> jnp.ndarray:
    r"""Trapezoid weights in frequency for a wavelength grid.

    The single definition of "what the grid weighs each node by", shared by all
    three placement modes so they agree on flux by construction rather than by
    three separate arguments (#1836).

    Parameters
    ----------
    obs_wavelengths : ndarray, shape (n_wave,)
        Wavelength grid, increasing. [Å]

    Returns
    -------
    ndarray, shape (n_wave,)
        Weights :math:`w_i` with :math:`\sum_i w_i f_i = |\int f\,d\nu|` under
        the trapezoid rule. [Hz]

    Notes
    -----
    **JIT-compatible**: yes.
    """
    nu = _C_CGS / (obs_wavelengths * 1e-8)  # (n_wave,) [Hz]
    dnu = jnp.abs(jnp.diff(nu))
    return jnp.zeros_like(nu).at[:-1].add(0.5 * dnu).at[1:].add(0.5 * dnu)


def _grid_bracket(
    obs_wavelengths: jnp.ndarray, line_wavelengths: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Local grid spacing and nearest-node index for each line, from one search.

    Both facts come from the same bracketing interval, and both are needed on
    every render (#1836): the spacing floors the profile width, a *local*
    question, since the MILES SSP grid runs 0.9 Å inside 3500–7500 Å and 10 Å at
    Lyα, so a global statistic answers it wrongly: and the nearest node is where
    a line the grid cannot resolve gets placed.

    Returned together because they share the search, not because the search is
    expensive; measured, it is not: an ablation on an isolated (6185, 128) block
    put the whole of ``searchsorted`` + containment mask + scatter at **768**
    gradient FLOPs against 38,006,660 for the render. What costs is the discrete
    area itself (``quad_w @ profiles``, +41 % over the un-rescaled form), and
    that is the fix, not an accident of how it is written. Do not go looking for
    savings here; an earlier draft of this docstring blamed a second
    ``searchsorted`` call and the ablation refuted it.

    Parameters
    ----------
    obs_wavelengths : ndarray, shape (n_wave,)
        Wavelength grid, increasing. [Å]
    line_wavelengths : ndarray, shape (n_lines,)
        Line centers. [Å]

    Returns
    -------
    spacing : ndarray, shape (n_lines,)
        Width of the grid interval containing each line. [Å]
    nearest : ndarray, shape (n_lines,)
        Index of the closest grid node.

    Notes
    -----
    **JIT-compatible**: yes, ``searchsorted`` accepts traced operands.
    """
    hi = jnp.clip(
        jnp.searchsorted(obs_wavelengths, line_wavelengths), 1, obs_wavelengths.shape[0] - 1
    )
    lo = hi - 1
    left, right = obs_wavelengths[lo], obs_wavelengths[hi]
    nearest = jnp.where((line_wavelengths - left) <= (right - line_wavelengths), lo, hi)
    return right - left, nearest


def _render_conserving(
    profiles: jnp.ndarray,
    obs_wavelengths: jnp.ndarray,
    line_wavelengths: jnp.ndarray,
    line_luminosities: jnp.ndarray,
    support_half_width: jnp.ndarray,
    nearest: jnp.ndarray,
) -> jnp.ndarray:
    r"""Sum analytically-normalized line profiles, conserving flux exactly (#1836).

    Both profile shapes here (triweight, Gaussian) are normalized so that
    :math:`\int \phi\,d\nu = 1` in the continuum limit. The caller integrates on
    a discrete grid, where that holds only if the grid resolves the profile:
    and the SSP grids in use do not, outside their high-resolution window. Each
    profile is therefore divided by its own trapezoid area, making the rendered
    flux exact on whatever grid it was handed.

    A profile can still vanish on the grid: a sub-pixel line centered between two
    nodes puts both at :math:`|u| = 1`, where the compact-support triweight is
    *exactly* zero. Those lines are scattered into their nearest pixel instead:
    the grid cannot represent a width, but it must not lose the light. This is
    the one case the pre-#1836 code got wrong in the *quiet* direction, dropping
    the line with nothing raised.

    Parameters
    ----------
    profiles : ndarray, shape (n_wave, n_lines)
        Analytically normalized profiles [1/Hz].
    obs_wavelengths : ndarray, shape (n_wave,)
        Wavelength grid the profiles were rendered on, increasing. [Å]
    line_wavelengths : ndarray, shape (n_lines,)
        Line centers. [Å]
    line_luminosities : ndarray, shape (n_lines,)
        Integrated line luminosities. [erg/s] or [erg/s/Msun]
    support_half_width : ndarray, shape (n_lines,)
        Half-width beyond which each profile is negligible: exact for the
        compact-support triweight, a 4-sigma convention for the Gaussian. Used
        only to decide containment, never to shape the profile. [Å]
    nearest : ndarray, shape (n_lines,)
        Index of the grid node closest to each line, from :func:`_grid_bracket`.
        Where a profile vanishes on the grid entirely, its luminosity is
        scattered here instead.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density. For lines the grid fully contains,
        ``|∫ sed dν| == sum(line_luminosities)`` to rounding; a line straddling
        the grid edge keeps only the fraction that falls in range.
        [erg/s/Hz] or [erg/s/Hz/Msun]

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes, *both* divisors are
    guarded (the profile area and the quadrature weight), so no NaN reaches the
    tape even on a degenerate single-node grid.
    """
    quad_w = _nu_quadrature_weights(obs_wavelengths)  # (n_wave,) [Hz]
    area = quad_w @ profiles  # (n_lines,)

    # Rescale ONLY the profiles the grid fully contains. The rescale corrects
    # *quadrature* error (too few nodes under the profile) and a truncated
    # profile is a different thing that must not be corrected: a line near or
    # past the grid edge legitimately contributes only the part that lands in
    # range. Dividing its visible sliver by that sliver's own area would inflate
    # it to carry the line's entire luminosity, inventing flux at the boundary.
    # Measured when this guard was missing: [NII] 6583 A, outside a
    # 6550-6580 A grid, had its in-grid tail boosted until it out-peaked Halpha
    # and the rendered Halpha FWHM read 0.84 A instead of 4.71 A.
    #
    # Containment is exact for the compact-support triweight and a 4-sigma
    # convention for the Gaussian; ``support_half_width`` carries whichever.
    contained = (line_wavelengths - support_half_width >= obs_wavelengths[0]) & (
        line_wavelengths + support_half_width <= obs_wavelengths[-1]
    )
    rescale = contained & (area > 0.0)
    scaled = profiles / jnp.where(rescale, area, 1.0)[None, :]
    sed = scaled @ line_luminosities

    # Nearest-pixel placement for a contained line the grid still could not
    # represent: a sub-pixel profile centered between two nodes puts both at
    # |u| = 1, where the triweight is exactly zero. ``L / quad_w`` is the
    # density whose trapezoid integral is exactly ``L``, so this conserves flux
    # by the same rule the rescale uses. Its ``scaled`` column is all-zero, so
    # there is nothing to double-count.
    #
    # Both divisors are guarded the same way, and for the same reason: a
    # single-node grid has zero extent in ν, so ``quad_w`` is identically zero
    # there and an unguarded ``0/0`` puts a NaN on the tape. Guarding only
    # ``area`` left that hole: a 1-point grid returned ``nan`` where the
    # pre-#1836 code returned a finite value. Zero flux is the honest answer: a
    # grid of one point cannot carry a line, because it has no measure to
    # integrate over.
    w_near = quad_w[nearest]  # (n_lines,) [Hz]
    representable = w_near > 0.0
    dropped = jnp.where(contained & (area <= 0.0) & representable, line_luminosities, 0.0)
    return sed.at[nearest].add(dropped / jnp.where(representable, w_near, 1.0))


def place_line_profiles(
    line_wavelengths: jnp.ndarray,
    line_luminosities: jnp.ndarray,
    obs_wavelengths: jnp.ndarray,
    line_sigma_aa: float,
    line_sigma_kms: float = 0.0,
) -> jnp.ndarray:
    r"""Place emission lines onto a wavelength grid with an intrinsic line profile.

    Converts line luminosities (point-like) to spectral luminosity density on a
    wavelength grid using one of three modes, in priority order:

    1. **Velocity triweight** (``line_sigma_kms > 0``), the recommended path:
       each line is a triweight kernel whose width scales with wavelength,
       :math:`\sigma_\lambda = (\sigma_v / c)\,\lambda`, giving every line the
       same velocity dispersion (as a real emission line has). Mirrors
       Prospector's ``eline_sigma`` treatment (Johnson et al. 2021 [1]_).
    2. **Fixed-Å Gaussian** (``line_sigma_aa > 0``): legacy; one width in Å for
       every line.
    3. **Delta function** (both <= 0): nearest-pixel scatter-add.

    Parameters
    ----------
    line_wavelengths : array, shape (n_lines,)
        Rest-frame line centers in Å (vacuum wavelength). [Å]
    line_luminosities : array, shape (n_lines,)
        Line luminosities in consistent units [erg/s] or [erg/s/Msun].
    obs_wavelengths : array, shape (n_wave,)
        Output wavelength grid in Å (rest-frame, increasing). [Å]
    line_sigma_aa : float
        Fixed Gaussian line width in Å (legacy). Used only when
        ``line_sigma_kms <= 0``. [Å]
    line_sigma_kms : float, optional
        Velocity dispersion in km/s. When > 0, lines are rendered as triweight
        profiles with per-line width :math:`\sigma_\lambda=(\sigma_v/c)\lambda`.
        Default 0 (fall back to ``line_sigma_aa`` / delta). [km/s]

    Returns
    -------
    array, shape (n_wave,)
        Spectral luminosity density on ``obs_wavelengths``. [erg/s/Hz] or [erg/s/Hz/Msun]

    References
    ----------
    .. [1] B. D. Johnson, J. Leja, C. Conroy, J. S. Speagle 2021, "Stellar
       Population Inference with Prospector", ApJS, 254, 22.
       arXiv:2012.01426. DOI: 10.3847/1538-4365/abef67
    .. [2] Hearin, A. P., Chaves-Montero, J., Alarcon, A., Becker, M. R.,
       Benson, A. 2023, "DSPS: Differentiable stellar population synthesis",
       MNRAS, 521, 1741. arXiv:2112.06830. DOI: 10.1093/mnras/stad456

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives and are
    vectorized over lines and wavelengths.

    **Velocity triweight mode** (``line_sigma_kms > 0``):
        Each line is the compact-support triweight kernel
        :math:`K(u) = \tfrac{35}{32}(1-u^2)^3` for :math:`|u|<1`, with
        :math:`u=(\lambda-\lambda_i)/h_i` and half-width :math:`h_i = 3\sigma_{\lambda,i}`
        (the triweight kernel has variance :math:`h^2/9`, so :math:`h=3\sigma`).
        The profile is normalized to unit area in frequency, so the integrated
        line flux equals ``line_luminosity``. Preferred over a Gaussian: it is a
        polynomial (no ``exp``), has finite support, and is C²-continuous:
        gradient-safe for a fitted ``line_sigma_kms``.

    **Fixed-Å Gaussian mode** (``line_sigma_aa > 0``):
        Converts σ_λ to σ_ν via :math:`\sigma_\nu = \sigma_\lambda\,c/\lambda^2`
        and normalizes each line to unit flux.

    **Delta function mode** (both widths <= 0):
        Places each line in the nearest wavelength pixel by scatter-add,
        normalized by the local frequency spacing Δν. Carries no line width, and
        places the line up to half a pixel off center: negligible through a
        broadband filter, wrong for spectroscopy.

    **All three modes conserve flux** (#1836): each rescales its profiles to
    unit *discrete* area in ν, so ``|∫ sed dν| == sum(line_luminosities)`` on
    any grid. They therefore differ only in profile **shape**, which is the only
    reason to prefer one. Before #1836 they differed in flux too: on the MILES
    SSP grid the same 128 Cue lines integrated to 2.2489x (triweight), 1.5711x
    (Gaussian, σ=2 Å) and 1.0000x (delta) of their true total, and the resulting
    broadband photometry differed between modes by up to 3.04x.

    **Reference**: line-placement approach follows Prospector
    (Johnson et al. 2021 [1]_); triweight kernel after Hearin et al. 2023 [2]_.

    """
    n_wave = obs_wavelengths.shape[0]

    # NOTE: ``line_sigma_kms`` here is only honored for *concrete* values; the
    # ``if`` below is a Python branch. The forward model, where the velocity
    # width is a fittable (traced) parameter, must call
    # :func:`place_line_profiles_velocity` directly to stay JIT-safe.
    if line_sigma_kms > 0:
        sed = place_line_profiles_velocity(
            line_wavelengths, line_luminosities, obs_wavelengths, line_sigma_kms
        )
    elif line_sigma_aa > 0:
        # Vectorized Gaussian profiles: broadcast (n_wave, 1) × (n_lines,)
        # σ_ν = σ_λ[cm] × c / λ[cm]²
        # Widen to the local grid spacing where the requested width would fall
        # between nodes, for the same reason the triweight path does (#1836):
        # a 2 Å Gaussian on the 3100 Å-spaced far-IR section of the MILES grid
        # renders as all zeros, and the line is lost outright.
        dwave_local, nearest = _grid_bracket(obs_wavelengths, line_wavelengths)
        sigma_eff = jnp.maximum(jnp.asarray(line_sigma_aa), 0.5 * dwave_local)  # (n_lines,) [Å]
        sigma_nu = sigma_eff * 1e-8 * _C_CGS / (line_wavelengths * 1e-8) ** 2  # (n_lines,)
        dwave = obs_wavelengths[:, None] - line_wavelengths[None, :]  # (n_wave, n_lines)
        profiles = jnp.exp(-0.5 * (dwave / sigma_eff[None, :]) ** 2)
        profiles = profiles / (jnp.sqrt(2.0 * jnp.pi) * sigma_nu[None, :])
        # Same conserving render as the triweight path, so the two modes agree on
        # flux and differ only in profile shape (#1836). Unrescaled, this mode
        # recovered 1.5711x of the true total at sigma=2 Å, and lost the far-IR
        # lines outright where a 2 Å Gaussian underflowed on a 3100 Å grid.
        # 4 sigma: the Gaussian has no compact support, so containment is a
        # convention. Beyond 4 sigma it carries 6e-5 of its area.
        sed = _render_conserving(
            profiles,
            obs_wavelengths,
            line_wavelengths,
            line_luminosities,
            4.0 * sigma_eff,
            nearest,
        )
    else:
        # Vectorized delta functions: nearest-pixel placement via scatter-add.
        # Via searchsorted, not an (n_wave, n_lines) argmin: see
        # :func:`_grid_bracket` for why that matrix is worth avoiding.
        #
        # NOT clipped to [1, n_wave - 2]. That clip belonged to the old Δν,
        # which read ``obs[i ± 1]`` and so could not be evaluated at either end;
        # the quadrature weight below is defined at every node, so the clip is a
        # dead constraint that only displaces a line by a whole pixel when it
        # lands on the first or last one (#1836).
        indices = _grid_bracket(obs_wavelengths, line_wavelengths)[1]  # (n_lines,)
        # Divide by the SAME quadrature weight the caller will integrate with,
        # rather than a separately-derived Δν, so ``sum(w*sed) == sum(L)`` to
        # rounding instead of to the ~1e-4 the two spellings used to differ by
        # (#1836).
        quad_w = _nu_quadrature_weights(obs_wavelengths)  # (n_wave,) [Hz]
        sed = (
            jnp.zeros(n_wave, dtype=obs_wavelengths.dtype)
            .at[indices]
            .add(line_luminosities / quad_w[indices])
        )

    return sed


def place_line_profiles_velocity(
    line_wavelengths: jnp.ndarray,
    line_luminosities: jnp.ndarray,
    obs_wavelengths: jnp.ndarray,
    line_sigma_kms: float,
) -> jnp.ndarray:
    r"""Render emission lines as velocity-width triweight profiles (JIT/trace-safe).

    Each line is a compact-support triweight kernel
    :math:`K(u)=\tfrac{35}{32}(1-u^2)^3` (``|u|<1``) with a width that scales
    with wavelength, :math:`\sigma_\lambda=(\sigma_v/c)\,\lambda`, so every line
    shares the same velocity dispersion: the intrinsic nebular line width, as in
    Prospector's ``eline_sigma`` treatment (Johnson et al. 2021 [1]_). Each
    profile is normalized to unit area in frequency, conserving the integrated
    line flux.

    Unlike :func:`place_line_profiles`, this function has **no Python branch on
    the width**, so ``line_sigma_kms`` may be a *traced* (fittable) value under
    ``jax.jit`` / ``jax.grad`` / ``jax.vmap``. The support half-width is floored
    at the *local* grid spacing so a vanishing ``line_sigma_kms`` degrades to a
    ~1-pixel profile (rather than dividing by zero) instead of a true delta.

    The rendered flux is **exact on any grid**: each profile is rescaled by its
    own discrete integral in ν, so ``∫ sed dν`` returns ``line_luminosities``
    whether or not the grid resolves the line (#1836).

    Parameters
    ----------
    line_wavelengths : array, shape (n_lines,)
        Rest-frame line centers in Å (vacuum). [Å]
    line_luminosities : array, shape (n_lines,)
        Integrated line luminosities [erg/s] or [erg/s/Msun].
    obs_wavelengths : array, shape (n_wave,)
        Output wavelength grid in Å (increasing). [Å]
    line_sigma_kms : float
        Velocity dispersion. May be traced. [km/s]

    Returns
    -------
    array, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz] or [erg/s/Hz/Msun].

    References
    ----------
    .. [1] B. D. Johnson, J. Leja, C. Conroy, J. S. Speagle 2021, "Stellar
       Population Inference with Prospector", ApJS, 254, 22.
       arXiv:2012.01426. DOI: 10.3847/1538-4365/abef67
    .. [2] Hearin, A. P., Chaves-Montero, J., Alarcon, A., Becker, M. R.,
       Benson, A. 2023, "DSPS: Differentiable stellar population synthesis",
       MNRAS, 521, 1741. arXiv:2112.06830. DOI: 10.1093/mnras/stad456

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes, the triweight kernel is
    C²-continuous, so ``line_sigma_kms`` survives ``jax.grad``.

    **Flux is exact on any grid** (#1836). The analytic unit-area normalization
    is followed by a rescale to the profile's *discrete* area in ν, so
    ``|∫ sed dν| == sum(line_luminosities)`` to rounding for any grid, resolved
    or not. This also fixes the gradient's meaning: before the rescale a line's
    integrated flux varied with ``line_sigma_kms`` (Lyα on the MILES SSP grid
    ran 0.0000 → 3.0436 → 2.7072 of its true flux across σ_v = 50, 100,
    300 km/s), so ``d(flux)/d(sigma_v)`` was non-zero for a parameter that
    physically sets only the shape.
    """
    # Triweight variance is h²/9, so the support half-width h = 3σ.
    sigma_aa = (line_sigma_kms / _C_KM_S) * line_wavelengths  # (n_lines,) [Å]
    h_raw = 3.0 * sigma_aa
    # Floor h at the LOCAL grid spacing, per line (#1836). The floor exists so
    # σ→0 stays finite, but it also decides whether the grid samples the profile
    # at all: and a *global* statistic cannot answer a local question. The
    # previous floor was 0.5·median(diff(grid)): on the MILES SSP grid that
    # median is 0.9 Å, set by the 4423 points inside the 3500–7500 Å window,
    # while the same grid is 10 Å at Lyα and 29000 Å in the far-IR. So it never
    # fired where the grid was actually coarse: 0 of 128 Cue lines were floored
    # while 86 of them were rendered onto fewer than 4 points.
    # A window of half-width ≥ the local spacing always contains a grid node,
    # which is what makes the discrete renormalization below well-posed.
    # ``obs_wavelengths.shape`` is static, so the size guard is JIT-safe.
    if obs_wavelengths.shape[0] > 1:
        dwave_local, nearest = _grid_bracket(obs_wavelengths, line_wavelengths)
        h = jnp.maximum(h_raw, 0.5 * dwave_local)  # (n_lines,) [Å]
    else:
        h = jnp.maximum(h_raw, line_wavelengths * 1e-6)  # degenerate grid: tiny floor
        nearest = jnp.zeros(line_wavelengths.shape, dtype=jnp.int32)
    u = (obs_wavelengths[:, None] - line_wavelengths[None, :]) / h[None, :]
    kernel = jnp.where(jnp.abs(u) < 1.0, (35.0 / 32.0) * (1.0 - u**2) ** 3, 0.0)
    # Unit area in λ is K(u)/h; the λ²/c Jacobian converts to unit area in ν.
    lam2_over_c = (line_wavelengths * 1e-8) ** 2 / _C_CGS  # (n_lines,) [s·cm]
    profiles = kernel / (h[None, :] * 1e-8) * lam2_over_c[None, :]  # [1/Hz]
    # That normalization is ANALYTIC: ∫profile dν = 1 in the continuum limit.
    # On a grid that under-resolves the profile it is not, so make the flux
    # exact (#1836). Before this, recovered flux ran from 0.0026x (line silently
    # lost) to 3.04x (Lyα), and it was a function of ``line_sigma_kms``, so
    # fitting a width was partly fitting a flux.
    return _render_conserving(
        profiles, obs_wavelengths, line_wavelengths, line_luminosities, h, nearest
    )


def render_nebular_lines(
    line_wavelengths: jnp.ndarray,
    line_luminosities: jnp.ndarray,
    obs_wavelengths: jnp.ndarray,
    line_sigma_aa: float = 0.0,
    line_sigma_kms: float = 0.0,
) -> jnp.ndarray:
    """Render nebular emission lines, preferring the velocity-triweight profile.

    The dispatcher used by every nebular backend's ``predict_nebular_sed``:

    * ``line_sigma_aa > 0`` → legacy fixed-Å Gaussian via
      :func:`place_line_profiles`. ``line_sigma_aa`` is a *static* keyword (never
      a fittable parameter), so this Python branch is JIT-safe.
    * otherwise → velocity triweight via :func:`place_line_profiles_velocity`,
      which accepts a *traced* ``line_sigma_kms`` (the fittable
      ``neb_eline_sigma_kms``). A vanishing width floors to a ~1-pixel line.

    Replaces the previous nearest-pixel delta default so nebular lines carry an
    intrinsic velocity width in the rest-frame SED (Prospector-style).
    """
    if line_sigma_aa > 0:
        return place_line_profiles(
            line_wavelengths, line_luminosities, obs_wavelengths, line_sigma_aa
        )
    return place_line_profiles_velocity(
        line_wavelengths, line_luminosities, obs_wavelengths, line_sigma_kms
    )


# ── Ionizing photon rate ──────────────────────────────────────────


class QHTableOverflowError(ValueError):
    """The Q_H table cannot be represented in the working float dtype (#1491)."""


def sanitize_qh_table(qh_raw, *, backend_name: str):
    """Replace non-finite Q_H entries with zero: but only when zero is honest.

    Parameters
    ----------
    qh_raw : array_like, shape (n_met, n_age)
        Raw ionizing photon rate per SSP grid point. [1/s]
    backend_name : str
        Backend class name, for the error message.

    Returns
    -------
    ndarray, shape (n_met, n_age)
        ``qh_raw`` with non-finite entries set to 0.0.

    Raises
    ------
    QHTableOverflowError
        When the non-finite entries are dtype overflow rather than bad input.

    Notes
    -----
    The zeroing exists for SSP grids with incomplete UV coverage, where a
    non-finite Q_H really does mean "no usable ionizing flux here" and 0.0 is
    the honest answer.

    It is wrong for the *other* way an entry goes non-finite. Q_H reaches
    ~1e47 photons/s, and float32 tops out at 3.4e38, so in a float32 build the
    integral overflows on healthy input: and the guard then rewrites a real
    ionizing budget to exactly zero, i.e. no nebular emission, silently.
    Measured on ``fsps_prsc_miles_chabrier.h5``: **0 of 1395** entries
    non-finite in float64, **861 of 1395 (61.7%)** in float32 (#1491).

    Overflow is separable from bad input by its signature: it needs the working
    dtype to be too narrow to hold the *finite* entries that survived. A grid
    with patchy UV coverage loses a few bins and the survivors sit far below the
    dtype ceiling; an overflow leaves the survivors pressed against it. This
    checks that rather than guessing from the count.

    Float64 behavior is unchanged: the condition cannot fire when the finite
    entries are orders below 1.8e308.
    """
    finite = jnp.isfinite(qh_raw)
    n_bad = int(jnp.sum(~finite))
    if n_bad:
        largest_finite = float(jnp.max(jnp.where(finite, jnp.abs(qh_raw), 0.0)))
        ceiling = float(jnp.finfo(jnp.result_type(qh_raw)).max)
        # ``n_bad == size`` is the case the survivor test cannot see (#1568).
        # With nothing finite, ``largest_finite`` is 0.0 and the threshold below
        # is never crossed, so a *totally* overflowed table fell through and was
        # zeroed in full: the worst case, not an edge case. A grid with no
        # usable UV integrates to a finite 0.0 rather than a non-finite value,
        # so an all-non-finite table cannot be the honest-missing-data case.
        if largest_finite > 0.01 * ceiling or n_bad == qh_raw.size:
            raise QHTableOverflowError(
                f"{backend_name}: {n_bad} of {qh_raw.size} Q_H grid entries overflowed "
                f"{jnp.result_type(qh_raw)} (largest finite entry {largest_finite:.3e} sits "
                f"against the {ceiling:.3e} ceiling). Q_H reaches ~1e47 photons/s, which "
                "float32 cannot represent, so these are healthy SSP bins lost to dtype "
                "range: not a grid with missing UV coverage. Zeroing them would remove "
                "the ionizing budget from most of the grid and silently produce no "
                "nebular emission. Build this backend under float64 "
                "(the default; `jax.enable_x64(True)`). Pure-float32 support needs the "
                "line-luminosity unit change tracked as #1206 item 3, because the line "
                "luminosities this backend returns (~1e40 erg/s) are not float32 "
                "numbers either. See #1491."
            )
    return jnp.where(finite, qh_raw, 0.0)


@jax.jit
def compute_qh(ssp_wave: jnp.ndarray, ssp_flux: jnp.ndarray) -> float:
    r"""Compute hydrogen-ionizing photon production rate Q_H from an SSP spectrum.

    Integrates the far-UV SSP luminosity density, divided by photon energy, over
    the hydrogen-ionizing frequency range (ν > ν_LL ≈ 13.6 eV, λ < 911.76 Å).
    This rate is essential for all nebular emission models (HII regions, AGN NLR,
    DIG) that depend on ionizing photon supply.

    Parameters
    ----------
    ssp_wave : array, shape (n_wave,)
        SSP wavelength grid in Å (rest-frame, increasing).
    ssp_flux : array, shape (n_wave,)
        SSP spectral luminosity density. [erg/s/Hz/Msun]

    Returns
    -------
    float
        Hydrogen-ionizing photon production rate. [photons/s/Msun]

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives. Safe inside
    :func:`jax.jit`, :func:`jax.vmap`, and :func:`jax.grad`.

    **Frequency integration**:
        Q_H is computed as:

        .. math::

            Q_H = \int_0^{\nu_{\rm LL}} \frac{L_\nu}{h\nu} \, \mathrm{d}\nu

        where ν_LL = 13.6 eV / h ≈ 3.29 × 10^15 Hz (Lyman limit, λ < 911.76 Å),
        L_ν is the SSP flux [erg/s/Hz/Msun], and h is Planck's constant.

        The integral is computed via trapezoidal quadrature in frequency space
        (not wavelength space) to avoid nonlinear Jacobian effects.

    **Warning (wNE SSPs)**:
        Returns ~0 for "with Nebular Emission" (wNE) SSP spectra because CLOUDY
        consumes ionizing photons during SSP generation. If you see Q_H ≈ 0 for
        young SSPs (which should have Q_H > 1e50 photons/s), check that your
        SSP templates are non-nebular variants (BC03, FSPS/Conroy+Gunn models, etc.).

    **Numerical safety**:
        Clamps per-wavelength integrand to prevent float64 overflow during
        trapezoidal accumulation (only relevant for artificially young/pure SSPs
        with Q_H > 1e100). Does not affect physically realistic rates (~1e31).

    """
    return pow10(compute_qh_log10(ssp_wave, ssp_flux))


@jax.jit
def compute_qh_log10(ssp_wave: jnp.ndarray, ssp_flux: jnp.ndarray) -> float:
    r"""``log10(Q_H)`` [dex re photons/s/Msun]: the range-safe core integral.

    Same quantity as :func:`compute_qh`, computed so that no intermediate leaves
    float32 range. :func:`compute_qh` is a thin ``pow10`` wrapper over this, so
    the two cannot drift.

    .. math::

        \log_{10} Q_H = \log_{10}\!\left(-\int_{\lambda<912\,\mathrm{\AA}}
        \frac{\hat{L}_\nu}{\nu}\,d\nu\right)
        + \log_{10} \hat{L}_{\rm peak} + \log_{10} L_\odot - \log_{10} h

    where :math:`\hat{L}_\nu = L_\nu / \hat{L}_{\rm peak}` is the SSP flux
    normalized by its own peak [dimensionless] and :math:`h` is Planck's
    constant [erg s].

    Parameters
    ----------
    ssp_wave : array, shape (n_wave,)
        SSP wavelength grid [Angstrom], rest-frame, increasing.
    ssp_flux : array, shape (n_wave,)
        SSP spectral luminosity density [Lsun/Hz/Msun].

    Returns
    -------
    float
        ``log10`` of the ionizing photon rate [dex re photons/s/Msun].
        ``-inf`` when there is no ionizing flux: the exact-zero sentinel, so
        ``pow10`` returns exactly 0.0 as the linear form always did.

    Notes
    -----
    **JIT-compatible / gradient-safe**: yes. The where-dummy keeps the backward
    pass free of NaN where the integral vanishes.

    **Why the peak is factored out first (#1568)**: the division by
    :math:`h\nu` is the step that leaves range:
    :math:`h\nu \approx 2\times10^{-11}`, so a healthy
    :math:`L_\nu \sim 10^{30}` becomes :math:`\sim10^{41}` *before* the
    trapezoid, and the integral itself reaches :math:`\sim10^{46}` against a
    float32 ceiling of 3.4e38. Normalizing by the peak and deferring both
    constants to a log-space sum keeps the integrand at
    :math:`\sim10^{-16}` and the accumulated integral at order unity. Same
    treatment as ``_integrate_nion_log10`` on the stellar path.

    The float64 clamp that used to sit here (``finfo(float64).max / n_wave``,
    guarding trapezoid overflow for artificially young SSPs with
    :math:`Q_H > 10^{100}`) is gone rather than ported: it was a float64
    literal, hence ``inf`` and inert in float32, and in the normalized form
    there is nothing left for it to guard.
    """
    nu = _C_CGS / (ssp_wave * 1e-8)  # Hz
    mask = ssp_wave < _LYMAN_LIMIT

    # Factor the peak out BEFORE dividing by nu: that division is the one that
    # leaves float32 range, so it must act on an O(1) quantity.
    peak = jnp.max(jnp.abs(ssp_flux))
    peak_safe = jnp.where(peak > 0, peak, jnp.ones_like(peak))
    integrand = jnp.where(mask, (ssp_flux / peak_safe) / nu, 0.0)
    norm = -jnp.trapezoid(integrand, nu)

    # ``jnp.maximum(qh, 0.0)`` in the linear form mapped a negative integral to
    # zero; ``-inf`` here is the same statement, and pow10 reproduces the 0.0.
    positive = norm > 0
    safe = jnp.where(positive, norm, 1.0)
    log_norm = jnp.where(positive, jnp.log10(safe), -jnp.inf)

    return log_norm + jnp.log10(peak_safe) + _LOG10_LSUN_ERG - _LOG10_H_PLANCK


# Vectorized over (metallicity, age) grid dimensions
compute_qh_grid = jax.vmap(
    jax.vmap(compute_qh, in_axes=(None, 0)),
    in_axes=(None, 0),
)


# ── Grid interpolation: piecewise-linear ─────────────────────────


def _interp_index_weight(
    x: float,
    grid: jnp.ndarray,
) -> tuple[int, float]:
    """Find bracketing index and linear interpolation weight for a 1D grid.

    Parameters
    ----------
    x : float
        Query point.
    grid : array, shape (n_grid,)
        Sorted grid points.

    Returns
    -------
    idx : int
        Index of left bracket in ``grid``.
    w : float
        Linear interpolation weight [0, 1].

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.

    """
    x_clipped = jnp.clip(x, grid[0], grid[-1])
    idx = jnp.searchsorted(grid, x_clipped, side="right") - 1
    idx = jnp.clip(idx, 0, len(grid) - 2)
    dx = grid[idx + 1] - grid[idx]
    w = jnp.where(dx > 0, (x_clipped - grid[idx]) / dx, 0.0)
    return idx, w


def _qh_bilinear(
    qh_table,
    qh_log_met: jnp.ndarray,
    qh_log_age: jnp.ndarray,
    log_z: float,
    log_age_yr: float,
    *,
    missing: float,
) -> jnp.ndarray:
    r"""Bilinear interpolation of an ionizing photon rate table, floored at zero.

    The single implementation behind every nebular backend's ``_get_qh_at``.
    It was previously copied into three backends, and the copies diverged: the
    CB19 backend lost the non-negativity floor its siblings carried (#1405).

    Parameters
    ----------
    qh_table : array_like, shape (n_met, n_age) or None
        Ionizing photon rate Q_H on the (metallicity, age) grid [1/s].
        ``None`` selects the ``missing`` fallback.
    qh_log_met : array_like, shape (n_met,)
        Table metallicity axis, absolute ``log10(Z)``, sorted ascending.
    qh_log_age : array_like, shape (n_age,)
        Table age axis, ``log10(age/yr)``, sorted ascending.
    log_z : float
        Query metallicity, absolute ``log10(Z)``.
    log_age_yr : float
        Query age, ``log10(age/yr)``.
    missing : float
        Value returned when ``qh_table`` is ``None``. Backend-specific and
        deliberately not unified: Q_H is consumed multiplicatively, so ``1.0``
        means "no Q_H scaling" and ``0.0`` means "no ionizing photons".

    Returns
    -------
    ndarray, shape ()
        Interpolated Q_H [1/s], clamped to be non-negative.

    Notes
    -----
    .. math::

        Q_H = (1 - w_z)\,[(1 - w_a) Q_{00} + w_a Q_{01}]
            + w_z\,[(1 - w_a) Q_{10} + w_a Q_{11}]

    with :math:`w_z, w_a \in [0, 1]` the linear weights from
    :func:`_interp_index_weight` along the metallicity and age axes. That
    function clips its query to the grid, so this is a convex combination of
    four table entries and never extrapolates: a negative result is only
    reachable from negative table entries, and is unphysical.

    The result is floored with ``jnp.maximum(..., 0.0)``. **NaN is deliberately
    not removed**: ``jnp.maximum(nan, 0.0)`` is ``nan``, and a NaN Q_H means the
    table is broken upstream. Propagating it makes that visible where a silent
    zero would not.

    **JIT-compatible**: yes, the only Python-level branch is on ``qh_table
    is None``, which is structural, not traced.

    """
    if qh_table is None:
        return jnp.asarray(missing)

    iz, wz = _interp_index_weight(log_z, qh_log_met)
    ia, wa = _interp_index_weight(log_age_yr, qh_log_age)

    q00 = qh_table[iz, ia]
    q01 = qh_table[iz, ia + 1]
    q10 = qh_table[iz + 1, ia]
    q11 = qh_table[iz + 1, ia + 1]

    q0 = q00 * (1 - wa) + q01 * wa
    q1 = q10 * (1 - wa) + q11 * wa
    return jnp.maximum(q0 * (1 - wz) + q1 * wz, 0.0)


# ── Metallicity convention converters ─────────────────────────────


def neb_logzsol_to_log_z_abs(logzsol: jnp.ndarray) -> jnp.ndarray:
    """Convert gas metallicity from log10(Z/Zsun) to absolute log10(Z).

    Parameters
    ----------
    logzsol : array
        Gas metallicity relative to solar [log10(Z/Zsun)].

    Returns
    -------
    array
        Absolute gas metallicity [log10(Z)].

    Notes
    -----
    **JIT-compatible**: yes, simple addition.

    """
    return logzsol + _LOG10_ZSUN


def neb_logzsol_to_cloudy_logoh(logzsol: jnp.ndarray) -> jnp.ndarray:
    """Convert gas metallicity to CLOUDY c17.01 log10(O/H) scale.

    Parameters
    ----------
    logzsol : array
        Gas metallicity relative to solar [log10(Z/Zsun)].

    Returns
    -------
    array
        log10(O/H) on CLOUDY c17.01 solar scale.

    Notes
    -----
    **JIT-compatible**: yes, simple addition.

    """
    return logzsol + _LOG10_ZSUN - _LOG_OH_OFFSET


def neb_logzsol_to_mappings_zeta(logzsol: jnp.ndarray) -> jnp.ndarray:
    """Convert gas metallicity to MAPPINGS V solar-relative O abundance (zeta_O).

    Parameters
    ----------
    logzsol : array
        Gas metallicity relative to solar [log10(Z/Zsun)].

    Returns
    -------
    array
        Solar-relative O abundance [zeta_O = Z/Zsun].

    Notes
    -----
    **JIT-compatible**: yes, simple exponentiation.

    """
    return 10.0**logzsol


# ── Analytic nebular continuum ────────────────────────────────────

# Case B recombination coefficient at T=10^4 K [cm³/s]
# Storey & Hummer (1995) via pyNeb: alpha_B(1e4 K, ne=100) = 2.585e-13
# Power-law slope: T^{-0.847} (SH95 fit over 5e3–3e4 K, via pyNeb getTotRecombination)
_ALPHA_B_T4: float = 2.585e-13
_ALPHA_B_SLOPE: float = -0.847

# Lyman-α vacuum wavelength [Å]
_LYA_AA: float = 1216.0

# Free-free coefficient [erg cm³/s/Hz] from Osterbrock & Ferland eq 4.16
# ε_ff(ν) = _FF_COEFF * T^{-1/2} * Z² * n_e * n_i * g_ff * exp(-hν/kT)
_FF_COEFF: float = 6.8e-38

# Two-photon constants (Dopita & Osterbrock / pyNeb Continuum.two_photon):
#   α_eff_2s(T) = _ALPHA_EFF_2S_T4 × (T/1e4)^{-0.728}  [cm³/s]: effective recombination
#   A_2s = 8.226 s^{-1}: Einstein A coefficient for 2s→1s two-photon decay
# Reference: pyNeb v1.1.30 (Luridiana et al. 2015), Osterbrock & Ferland (2006) eq 4.29
_ALPHA_EFF_2S_T4: float = 0.838e-13
_ALPHA_EFF_2S_SLOPE: float = -0.728
_A_2S: float = 8.226


def compute_analytic_nebular_continuum(
    wave_aa: jnp.ndarray,
    q_h: float,
    log_z_abs: float,
    temperature: float = 1e4,
) -> jnp.ndarray:
    r"""Compute hydrogen nebular continuum: free-free + two-photon emission.

    Predicts the optically-thick, ionization-bounded HII region continuum from
    case B recombination, thermal bremsstrahlung (free-free), and the two-photon
    emission from the 2s metastable transition. This is a Tier 2 analytic
    approximation used as a fallback when full grids (Cue, Cloudy) are unavailable.

    Parameters
    ----------
    wave_aa : array, shape (n_wave,)
        Wavelength grid in Å (rest-frame, increasing).
    q_h : float
        Hydrogen-ionizing photon production rate. [photons/s]
    log_z_abs : float
        Absolute metallicity. [log10(Z/Z_sun)]
        Currently unused; included for forward compatibility (metallicity scaling
        of free-free via He/metal opacity is reserved for a future update).
    temperature : float, optional
        Electron temperature in K. Default: 10^4 K (typical HII region). [K]

    Returns
    -------
    array, shape (n_wave,)
        Nebular continuum spectral luminosity density. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives with no
    Python-level branching on traced values.

    **Case B recombination normalization** (Osterbrock & Ferland 2006, §4.3):
        The ionization balance is:

        .. math::

            Q_H = \alpha_B(T) \, n_e \, n_p \, V

        where α_B(T) is the case-B recombination coefficient and V is the HII
        region volume. Rearranging:

        .. math::

            n_e \, n_p \, V = \frac{Q_H}{\alpha_B(T)}

        Temperature dependence (Storey & Hummer 1995):

        .. math::

            \alpha_B(T) = \alpha_B(10^4\,\mathrm{K}) \, \left(\frac{T}{10^4}\right)^{-0.847}

        with α_B(10^4 K) ≈ 2.585 × 10^{-13} cm^3/s.

    **Free-free (bremsstrahlung) continuum** (OF06, §4.4, eq. 4.16):
        Thermal radiation from electron–ion collisions:

        .. math::

            L_{\mathrm{ff}}(\nu) = \frac{Q_H}{\alpha_B(T)} \, \epsilon_{\mathrm{ff}}(\nu,T)
                \quad \text{where} \quad
                \epsilon_{\mathrm{ff}} = k_{\mathrm{ff}} \, T^{-1/2} \,
                g_{\mathrm{ff}}(\nu,T) \, e^{-h\nu/(k_B T)}

        Gaunt factor approximation (Draine 2011, §10.4, eq. 10.9):

        .. math::

            g_{\mathrm{ff}}(u) = \max\left(1, \frac{\sqrt{3}}{\pi} \ln\frac{2}{u}\right)
                \quad \text{where} \quad u = \frac{h\nu}{k_B T}

    **Two-photon continuum** (Nussbaumer & Schmutz 1984, OF06 §4.5, eq. 4.29):
        Recombination to the 2s metastable level produces two-photon pairs
        (2s → 1s emission is forbidden as a single photon). Spectral shape:

        .. math::

            A_{2\gamma}(y) = 202 \, y(1-y) \,
                \left[(1-4y(1-y))^{0.8} + 0.88(y(1-y))^{1.53}\right]

        where y = ν/ν_{Ly\alpha} = λ_{Ly\alpha}/λ ∈ (0,1), defined only for
        λ > λ_{Ly\alpha} (≈ 1216 Å). Peak intensity near y ≈ 0.5 (λ ≈ 2432 Å,
        optical/NUV). Normalization:

        .. math::

            L_{2\gamma}(\nu) = \frac{Q_H}{\alpha_B(T)} \, \frac{\alpha_{\mathrm{eff}}^{2s}(T)}{A_{2s}}
                \, h \, y \, A_{2\gamma}(y)

        where α^{eff}_{2s}(T) = 0.838 × 10^{-13} (T/10^4)^{-0.728} cm^3/s is
        the effective recombination coefficient to 2s, and A_{2s} = 8.226 s^{-1}
        is the Einstein A coefficient for the forbidden 2s → 1s transition.

    **Approximation flags**:

        - **Missing continua**: Free-bound (recombination) continuum is omitted
          (contributes λ < 3646 Å, Balmer limit). This is acceptable for optical
          SEDs but underestimates UV flux shortward of the Balmer limit.
        - **Hydrogen only**: Neglects helium and metal free-free opacity (~10% at
          solar metallicity); reserved for a future metallicity-dependent update.
        - **Single temperature**: Assumes uniform electron temperature T=10^4 K.
          Physically, HII region temperature varies with ionization parameter and
          ISM conditions; this approximation is valid for log(U) ∈ [−3, −1].

    **Validity**: Use this function when Cloudy or Cue grids are unavailable.
    For science-grade nebular fitting, prefer CloudyGridBackend or CueBackend.

    References
    ----------
    .. [1] D. E. Osterbrock and G. J. Ferland, "Astrophysics of Gaseous Nebulae
       and Active Galactic Nuclei," 2nd edn. (University Science Books, 2006).
       Chapter 4: Nebular Continuum and Line Emission.
    .. [2] H. Nussbaumer and W. Schmutz, "The two-photon continuum of HeII and
       the He+ f-value problem," A&A, 138, 495 (1984).
    .. [3] P. J. Storey and D. G. Hummer, "Recombination coefficients for H II and
       HeII," MNRAS, 272, 41 (1995). https://doi.org/10.1093/mnras/272.1.41
    .. [4] B. T. Draine, "Physics of the Interstellar and Intergalactic Medium"
       (Princeton University Press, 2011). Section 10.4: Gaunt factors.
    .. [5] V. Luridiana, C. Morisset, and R. A. Shaw, "PyNeb: A Python Package
       for Analyzing Emission Lines from Ionized Nebulae," A&A, 573, A42 (2015).
       arXiv:1412.6345. https://doi.org/10.1051/0004-6361/201323152

    """
    nu = _C_CGS / (wave_aa * 1e-8)  # Hz

    # Case B recombination coefficient: α_B(T) = α_B(1e4) × (T/1e4)^{-0.847}
    # Slope -0.847 from Storey & Hummer (1995) fit via pyNeb over 5e3–3e4 K.
    alpha_b = _ALPHA_B_T4 * (temperature / 1.0e4) ** _ALPHA_B_SLOPE

    # n_e · n_p · V = Q_H / α_B
    q_over_alpha = q_h / jnp.maximum(alpha_b, 1.0e-40)

    # ─── Free-free ───────────────────────────────────────────────────────────
    # Osterbrock & Ferland (2006), eq 4.16
    x = _H_PLANCK * nu / (_K_BOLTZ * temperature)  # dimensionless hν/kT
    # Gaunt factor: Draine (2011) eq 10.9 approximation; clip to ≥ 1
    g_ff = jnp.maximum(1.0, jnp.sqrt(3.0) / jnp.pi * jnp.log(2.0 / jnp.maximum(x, 1e-30)))
    gamma_ff = _FF_COEFF * temperature ** (-0.5) * g_ff * jnp.exp(-x)
    L_ff = q_over_alpha * gamma_ff  # erg/s/Hz

    # ─── Two-photon ──────────────────────────────────────────────────────────
    # N&S 1984 shape: y = ν/ν_Lyα = λ_Lyα/λ, valid for 0 < y < 1 (λ > λ_Lyα).
    # Each 2s photon pair spans λ_Lyα < λ < ∞ (peak near λ ≈ 2 × λ_Lyα).
    # Mask y ≥ 1 (λ ≤ λ_Lyα) to zero: photons cannot exceed Lyα energy.
    # JAX NaN-safe pattern: supply y_safe=0.5 in masked pixels so the shape
    # formula evaluates to a finite value before the jnp.where mask is applied.
    y_raw = _LYA_AA / jnp.maximum(wave_aa, _LYA_AA + 1e-6)  # never > 1 (safe)
    uu = y_raw * (1.0 - y_raw)
    A2q_shape = 202.0 * (uu * (1.0 - 4.0 * uu) ** 0.8 + 0.88 * uu**1.53)
    A2q = jnp.where(wave_aa > _LYA_AA, A2q_shape, 0.0)

    # Normalization (pyNeb / OF06 §4.5):
    # Power emitted per atom per unit-y interval = h × ν_Lyα × A₂γ(y), so:
    #   dL/dν = n_2s V × h × ν_Lyα × A₂γ(y) / ν_Lyα = n_2s V × h × A₂γ(y)... but
    # more cleanly, with y = ν/ν_Lyα:
    #   L_2q(ν) = (Q_H/α_B) × (α_eff_2s/A_2s) × h × y × A₂γ(y)   [erg/s/Hz]
    # where y = _LYA_AA/wave_aa = ν/ν_Lyα (already computed as y_raw).
    alpha_eff_2s = _ALPHA_EFF_2S_T4 * (temperature / 1.0e4) ** _ALPHA_EFF_2S_SLOPE
    L_2q = q_over_alpha * _H_PLANCK * y_raw * A2q / _A_2S * alpha_eff_2s

    return L_ff + L_2q


# ── Continuum fallback ────────────────────────────────────────────


class NebularContinuumFallback:
    """Wrapper that adds nebular continuum to line-only nebular backends.

    Many nebular backends (CB19, MAPPINGS, Shock) produce emission lines only.
    This wrapper provides missing continuum via a prioritized fallback chain:
    (1) secondary physics backend, (2) analytic free-free + two-photon, (3) error
    or warning.

    Parameters
    ----------
    primary : NebularBackend
        Line-only nebular backend (has_continuum=False). Must implement
        ``predict_nebular_sed()``.
    fallback : NebularBackend, optional
        Continuum-capable backend (CueBackend, CloudyGridBackend) for Tier 1
        fallback. Default: None.
    fallback_mode : str, optional
        Fallback behavior if neither backend nor analytical continuum is
        available. One of "error" (raise NebularContinuumUnavailableError) or
        "warn" (emit warning, return lines only). Default: "error".

    Attributes
    ----------
    has_continuum : bool
        Always True; guarantees continuum provision via three-tier chain.
    has_free_params : bool
        Inherited from primary backend.
    name : str
        Identifier string (e.g., "fallback(CB19Backend)").

    Notes
    -----
    **JIT-compatible**: no, predict_nebular_sed may invoke non-JIT backends.

    **Continuum supply chain** (at prediction time):

    1. **Tier 1 (Secondary backend)**: If ``fallback`` provided and has
       ``predict_nebular_sed``, use it to compute full continuum.
    2. **Tier 2 (Analytic)**: If ``ssp_wave`` and ``gas_logqion`` in kwargs,
       compute analytic free-free + two-photon via
       ``compute_analytic_nebular_continuum()``. Requires ``ssp_wave`` [Angstrom]
       and ``gas_logqion`` [log10(Q_H)].
    3. **Tier 3 (Graceful degradation)**: If neither Tier 1 nor 2 available,
       either raise ``NebularContinuumUnavailableError`` (fallback_mode="error")
       or emit UserWarning and return lines only (fallback_mode="warn").

    """

    def __init__(
        self,
        primary,
        fallback=None,
        fallback_mode: str = "error",
    ) -> None:
        if fallback_mode not in ("error", "warn"):
            raise ValueError("fallback_mode must be 'error' or 'warn'")
        self.primary = primary
        self.fallback = fallback
        self.fallback_mode = fallback_mode
        # This wrapper always attempts to provide continuum: either via the
        # fallback backend (Tier 1), analytic approximation (Tier 2), or
        # graceful degradation per fallback_mode (Tier 3/4).
        self.has_continuum = True
        self.has_free_params = getattr(primary, "has_free_params", False)
        self.name = f"fallback({getattr(primary, 'name', type(primary).__name__)})"

    def __getattr__(self, name: str):
        """Delegate all unknown attributes and methods to the primary backend."""
        return getattr(self.primary, name)

    def predict_nebular_sed(self, *args, **kwargs) -> jnp.ndarray:
        """Predict nebular SED: lines + continuum via fallback chain.

        Retrieves emission lines from the primary backend and adds nebular
        continuum via the priority fallback chain (secondary backend, analytic,
        or graceful degradation).

        Parameters
        ----------
        *args
            Positional arguments passed to primary.predict_nebular_sed().
        **kwargs
            Keyword arguments passed to primary and fallback backends.
            Special keywords: ``ssp_wave`` [Angstrom], ``gas_logqion``
            [log10(Q_H)] for Tier 2 (analytic) continuum.

        Returns
        -------
        array, shape (n_wave,)
            Nebular spectral luminosity density on SSP wavelength grid
            [erg/s/Hz].

        Raises
        ------
        NebularContinuumUnavailableError
            If fallback_mode="error" and neither secondary backend nor
            analytic continuum (via ssp_wave + gas_logqion) is available.

        Warns
        -----
        UserWarning
            If fallback_mode="warn" and continuum is unavailable. Returns
            lines only in this case.

        Notes
        -----
        **JIT-compatible**: no, may invoke non-JIT backends.

        **Execution order**:
        1. Call primary.predict_nebular_sed(*args, **kwargs) → lines
        2. If fallback backend available → add its continuum
        3. Else if ssp_wave and gas_logqion in kwargs → analytic continuum
        4. Else apply fallback_mode (error or warn)

        """
        from tengri.components.nebular._protocol import NebularContinuumUnavailableError

        lines_sed = self.primary.predict_nebular_sed(*args, **kwargs)

        # Tier 1: secondary backend
        if self.fallback is not None and hasattr(self.fallback, "predict_nebular_sed"):
            cont_sed = self.fallback.predict_nebular_sed(*args, **kwargs)
            return lines_sed + cont_sed

        # Tier 2: analytic continuum (free-free + two-photon) when possible
        ssp_wave = kwargs.get("ssp_wave")
        gas_logqion = kwargs.get("gas_logqion")
        log_z = kwargs.get("log_z", _LOG10_ZSUN)
        if ssp_wave is not None and gas_logqion is not None:
            q_h = 10.0**gas_logqion
            cont_analytic = compute_analytic_nebular_continuum(ssp_wave, q_h, log_z_abs=log_z)
            return lines_sed + cont_analytic

        if self.fallback_mode == "error":
            raise NebularContinuumUnavailableError(
                f"{type(self.primary).__name__} provides no nebular continuum. "
                "Pass fallback=CueBackend(...) or fallback=CloudyGridBackend(...) "
                "to NebularContinuumFallback, or ensure ssp_wave + gas_logqion are "
                "passed as keyword arguments for analytic continuum."
            )
        # Tier 4: warn and return lines only
        import warnings

        warnings.warn(
            f"{type(self.primary).__name__} has no nebular continuum: returning "
            "lines only. Pass fallback= to NebularContinuumFallback to add continuum.",
            UserWarning,
            stacklevel=2,
        )
        return lines_sed
