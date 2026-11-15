# SPDX-License-Identifier: BSD-3-Clause
"""CIGALE's ``lambda_fracAGN`` window for ``agn_ir_frac`` (#2662).

CIGALE ``skirtor2016`` sets the AGN dust power from the fraction of the light in a
wavelength window (``pcigale/sed_modules/skirtor2016.py:473-498``)::

    scale     = int_w L_host dlam / AGNlumin        (window)
              = L_host(lam) / AGNlumin(lam)          (lam_min == lam_max, monochromatic)
    AGNlumin  = int_w L_agn dlam / D_unit            (per unit AGN dust luminosity)
    agn_power = scale * f / (1 - f)

so ``f`` is the AGN fraction of the total light in the window. The total-dust case
``lambda_fracAGN="0/0"`` is ``scale = L_host`` and is not handled here.

The functions below return ``log10(agn_power)`` in erg/s. The ratio is formed in log
space because the window luminosities of a host near 1e43 erg/s do not fit float32.

Tier: Tier 2 (scientific helper, ``components/``).

Notes
-----
JIT/grad/vmap compatible. The window integral is evaluated on the caller's rest grid with
the edges included by linear interpolation (``np.interp``) and the trapezoid rule, the same
node set as pcigale (``hstack([lam_min, grid_in_window, lam_max])``). It is written as a
clip of the grid, so no boolean indexing enters the trace and a zero-length segment
contributes nothing.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.utils.scale import representable_floor

#: Speed of light [Angstrom/s]; enters once, through the dust luminosity ``D``.
_C_AA_PER_S = 2.99792458e18

#: Floor for a luminosity integral before its logarithm (dtype-safe).
_FLOOR = representable_floor(1.0e-30)


def _log10_safe(value):
    return jnp.log10(jnp.maximum(jnp.asarray(value), _FLOOR))


def _window_q(wave, lnu, lam_min, lam_max):
    """``int_w (L_nu / lam^2) dlam`` over ``[lam_min, lam_max]``; the L_lambda integral / c.

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Rest wavelength [Angstrom], ascending.
    lnu : array_like, shape (n_wave,)
        Spectral luminosity [erg/s/Hz] on ``wave``.
    lam_min, lam_max : float
        Window edges [Angstrom].

    Returns
    -------
    ndarray, scalar
        The window integral of ``L_nu / lam^2`` [erg/s/Hz/Angstrom].
    """
    lam_c = jnp.clip(wave, lam_min, lam_max)
    f_c = jnp.interp(lam_c, wave, lnu / wave**2)
    return jnp.trapezoid(f_c, lam_c)


def _mono_ratio(wave, host_lnu, agn_lnu, lam):
    """``L_host(lam) / L_agn(lam)`` by linear interpolation on the rest grid."""
    h = jnp.interp(lam, wave, host_lnu / wave**2)
    w = jnp.interp(lam, wave, agn_lnu / wave**2)
    return h / w


def log10_agn_power_from_window(
    wave, host_lnu, agn_lnu, dust_lnu, frac, lam_min: float, lam_max: float
):
    """``log10`` of CIGALE's ``agn_power`` for a wavelength window (erg/s).

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Rest wavelength [Angstrom], ascending.
    host_lnu : array_like, shape (n_wave,)
        Host SED without the AGN (stellar, nebular, dust emission) [erg/s/Hz].
    agn_lnu : array_like, shape (n_wave,)
        AGN SED per unit dust luminosity, or at any finite normalization: only the
        ratio ``dust / agn`` enters [erg/s/Hz].
    dust_lnu : array_like, shape (n_wave,)
        AGN dust (torus plus polar) SED at the same normalization as ``agn_lnu``
        [erg/s/Hz].
    frac : float or array_like
        AGN fraction ``f`` of the light in the window, ``0 < f < 1``.
    lam_min, lam_max : float
        Window edges [Angstrom]. ``lam_min == lam_max != 0`` is monochromatic.

    Returns
    -------
    ndarray, scalar
        ``log10(agn_power / (erg/s))``.

    Notes
    -----
    Implements ``agn_power = f/(1-f) * H * D / W`` with ``H``, ``W`` the host and AGN
    window luminosities and ``D`` the AGN dust luminosity, the pcigale definition at
    ``skirtor2016.py:473-498``. The monochromatic case replaces ``H`` and ``W`` by the
    L_lambda values at the wavelength. JIT/grad/vmap compatible; ``lam_min`` and
    ``lam_max`` are static Python floats.
    """
    frac = jnp.asarray(frac)
    log_odds = _log10_safe(frac) - _log10_safe(1.0 - frac)
    # D = int L_nu dnu = c * int (L_nu / lam^2) dlam over the whole grid.
    log_dust = _log10_safe(_C_AA_PER_S) + _log10_safe(jnp.trapezoid(dust_lnu / wave**2, wave))
    if lam_min == lam_max:
        log_ratio = _log10_safe(_mono_ratio(wave, host_lnu, agn_lnu, lam_min))
    else:
        log_ratio = _log10_safe(_window_q(wave, host_lnu, lam_min, lam_max)) - _log10_safe(
            _window_q(wave, agn_lnu, lam_min, lam_max)
        )
    return log_odds + log_ratio + log_dust


#: Torus fraction at which the provisional AGN pass is evaluated; the runner is linear in it.
PROVISIONAL_TORUS_FRAC = 0.5


def log10_torus_frac_from_window(
    wave, host_lnu, provisional_lnu, frac, lam_min: float, lam_max: float
):
    """``log10`` of the torus fraction that gives CIGALE's window ratio, from a provisional pass.

    The composable AGN is linear in ``agn_torus_frac`` under the cigale_joint tie, so the
    provisional total ``W_p`` at ``PROVISIONAL_TORUS_FRAC`` scales with the final one. The
    final fraction is ``frac_p * f/(1-f) * H / W_p``, which makes the final window AGN
    luminosity exactly ``f/(1-f)`` times the host's. Using the grid dust integral ``D``
    instead would carry the torus template's own quadrature error (~8e-6 at the
    default grid), so the ratio is taken on the total window luminosity.

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Rest wavelength [Angstrom], ascending.
    host_lnu : array_like, shape (n_wave,)
        Host SED without the AGN [erg/s/Hz].
    provisional_lnu : array_like, shape (n_wave,)
        AGN total SED at ``PROVISIONAL_TORUS_FRAC`` [erg/s/Hz].
    frac : float or array_like
        AGN fraction ``f`` of the light in the window, ``0 < f < 1``.
    lam_min, lam_max : float
        Window edges [Angstrom]; equal edges give the monochromatic case.

    Returns
    -------
    ndarray, scalar
        ``log10(agn_torus_frac)``, dimensionless.

    Notes
    -----
    JIT/grad/vmap compatible; ``lam_min``/``lam_max`` are static floats.
    """
    frac = jnp.asarray(frac)
    log_odds = _log10_safe(frac) - _log10_safe(1.0 - frac)
    if lam_min == lam_max:
        log_ratio = _log10_safe(_mono_ratio(wave, host_lnu, provisional_lnu, lam_min))
    else:
        log_ratio = _log10_safe(_window_q(wave, host_lnu, lam_min, lam_max)) - _log10_safe(
            _window_q(wave, provisional_lnu, lam_min, lam_max)
        )
    return log_odds + log_ratio + jnp.log10(PROVISIONAL_TORUS_FRAC)


#: Torus fractions of the two provisional passes of the conserving window solve.
CONSERVING_PROVISIONAL_FRACS = (0.25, 0.75)


def _log10_window(wave, lnu, lam_min: float, lam_max: float):
    """``log10`` of the window integral of ``L_nu / lam^2``, or of its value at one wavelength."""
    if lam_min == lam_max:
        return _log10_safe(jnp.interp(lam_min, wave, lnu / wave**2))
    return _log10_safe(_window_q(wave, lnu, lam_min, lam_max))


def conserving_torus_frac_from_window(
    wave, host_lnu, lnu_a, lnu_b, frac, lam_min: float, lam_max: float
):
    r"""The covering factor that gives the window fraction under ``agn_norm='conserving'``.

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Rest wavelength [Angstrom], ascending.
    host_lnu : array_like, shape (n_wave,)
        Host SED without the AGN [erg/s/Hz].
    lnu_a, lnu_b : array_like, shape (n_wave,)
        AGN total SED at the torus fractions :data:`CONSERVING_PROVISIONAL_FRACS`
        [erg/s/Hz].
    frac : float or array_like
        AGN fraction ``f`` of the light in the window, ``0 < f < 1``.
    lam_min, lam_max : float
        Window edges [Angstrom]; equal edges give the monochromatic case.

    Returns
    -------
    ndarray, scalar
        ``agn_torus_frac`` in ``[0, 1]``, dimensionless.

    Notes
    -----
    Under ``'conserving'`` the AGN power is ``agn_lum_ratio 10**agn_log_lbol L_sun`` and the
    torus fraction ``t`` splits it: the disc carries ``(1 - t)`` of it and the torus ``t``. The
    window luminosity of the AGN is then affine in ``t``,

    .. math::

        W(t) = W_a + \frac{t - t_a}{t_b - t_a}\,(W_b - W_a),

    and the fraction is solved from :math:`W(t) = H\,f/(1 - f)`:

    .. math::

        t = t_a + (t_b - t_a)\,\frac{H f/(1-f) - W_a}{W_b - W_a},

    with :math:`W_a`, :math:`W_b` the AGN window luminosities at
    :math:`t_a, t_b` = :data:`CONSERVING_PROVISIONAL_FRACS` and :math:`H` the host's
    [erg/s/Angstrom, up to the common factor ``c``], each the integral of
    :math:`L_\nu/\lambda^2` over ``[lam_min, lam_max]`` (the value at the wavelength when the
    edges are equal). ``t`` is then clipped to ``[0, 1]``: a covering factor. Outside that range
    the requested fraction is not reachable at this accretion power (the disc alone outshines
    the target in the window, or the whole power re-emitted by the torus falls short of it),
    and the realized fraction is the one at the bound. The affine form is exact for every
    recipe without the ``polar_dust`` block. With it the polar re-emission's share of the AGN
    dust moves with ``t``, the window light is rational in ``t``, and this chord is only the
    start of :func:`refine_conserving_window_root`, which the component always applies. All
    ratios are formed in log space relative to :math:`W_b`, so nothing erg/s-scale forms
    in float32. JIT/grad/vmap compatible; ``lam_min``/``lam_max`` are static floats.
    """
    t_a, t_b = CONSERVING_PROVISIONAL_FRACS
    frac = jnp.asarray(frac)
    log_odds = _log10_safe(frac) - _log10_safe(1.0 - frac)
    log_wb = _log10_window(wave, lnu_b, lam_min, lam_max)
    s = jnp.power(10.0, _log10_window(wave, lnu_a, lam_min, lam_max) - log_wb)
    target = jnp.power(10.0, _log10_window(wave, host_lnu, lam_min, lam_max) + log_odds - log_wb)
    # W_b == W_a: the torus share leaves the window untouched, so no fraction can move it; the
    # selected denominator keeps the reverse pass finite there and the bound below decides.
    slope = 1.0 - s
    live = jnp.abs(slope) > 0.0
    t = t_a + (t_b - t_a) * (target - s) / jnp.where(live, slope, 1.0)
    return jnp.clip(jnp.where(live, t, 1.0), 0.0, 1.0)


def refine_conserving_window_root(
    agn_lnu_at,
    wave,
    host_lnu,
    frac,
    lam_min: float,
    lam_max: float,
    t_start,
    *,
    n_steps: int,
):
    r"""The exact covering factor of the conserving window solve, refined from a start.

    Parameters
    ----------
    agn_lnu_at : callable
        ``t -> L_nu`` [erg/s/Hz], shape (n_wave,): the AGN total SED at torus fraction ``t``.
    wave : array_like, shape (n_wave,)
        Rest wavelength [Angstrom], ascending.
    host_lnu : array_like, shape (n_wave,)
        Host SED without the AGN [erg/s/Hz].
    frac : float or array_like
        AGN fraction ``f`` of the light in the window, ``0 < f < 1``.
    lam_min, lam_max : float
        Window edges [Angstrom]; equal edges give the monochromatic case.
    t_start : array_like
        Starting covering factor in ``[0, 1]`` (the chord of
        :func:`conserving_torus_frac_from_window`).
    n_steps : int
        Newton steps (static), each one forward-mode evaluation of ``agn_lnu_at``.

    Returns
    -------
    ndarray, scalar
        ``agn_torus_frac`` in ``[0, 1]``, dimensionless.

    Notes
    -----
    Solves :math:`g(t) = \log_{10} W_{\rm agn}(t) - \log_{10}[H f/(1-f)] = 0` on
    :math:`[0, 1]`. Beside ``polar_dust`` the window light is rational in :math:`t` (the polar
    share of the dust budget moves with the disc it absorbs), so the chord through two passes
    is only a start. Each Newton step takes :math:`t \leftarrow t - g/g'` with :math:`g'` from a
    forward-mode derivative of one pass, and is clipped to :math:`[0, 1]`, so an unreachable
    fraction ends at the bound. Measured on the 5-20 um window at ``f = 0.3`` (synthetic SSP,
    skirtor and Fritz, ``agn_polar_ebv = 0.03``): the chord's -1.1% and -0.4% become -7.8e-5 and
    -1.0e-5 after one step, -3.9e-9 and -6.8e-11 after two, 6e-15 and 7e-16 after three. A
    secant rule reaches 2e-11 at the same cost (53.92 against 53.94 MFLOP for a value-and-gradient
    pass), but its last slope is a chord, so its gradient is not the root's.

    The iterates carry no gradient. The last step is
    :math:`t^\ast = \bar t - g(\bar t)/\mathrm{sg}(g'(\bar t))` with :math:`\bar t` the
    stopped iterate, whose derivative in any parameter :math:`\theta` is
    :math:`-\partial_\theta g/g'`: the implicit-function derivative of the root.
    JIT/grad/vmap compatible.
    """
    import jax

    frac = jnp.asarray(frac)
    log_target = _log10_window(wave, host_lnu, lam_min, lam_max) + (
        _log10_safe(frac) - _log10_safe(1.0 - frac)
    )

    def g(t):
        return _log10_window(wave, agn_lnu_at(t), lam_min, lam_max) - log_target

    def _step(t, value, slope):
        live = jnp.abs(slope) > 0.0
        return jnp.clip(t - value / jnp.where(live, slope, 1.0) * live, 0.0, 1.0)

    t = jax.lax.stop_gradient(jnp.asarray(t_start))
    for _ in range(n_steps - 1):
        value, slope = jax.jvp(g, (t,), (jnp.ones_like(t),))
        t = jax.lax.stop_gradient(_step(t, value, slope))
    value, slope = jax.jvp(g, (t,), (jnp.ones_like(t),))
    return _step(t, value, jax.lax.stop_gradient(slope))


def agn_power_from_window(wave, host_lnu, agn_lnu, dust_lnu, frac, lam_min, lam_max):
    """``agn_power`` [erg/s] for a wavelength window; the linear form of the log helper.

    Parameters
    ----------
    See :func:`log10_agn_power_from_window`.

    Returns
    -------
    ndarray, scalar
        ``agn_power`` [erg/s].

    Notes
    -----
    Exponentiates the log form, so it is only as accurate as float64 allows. The component
    uses the log form directly. JIT/grad/vmap compatible.
    """
    log_power = log10_agn_power_from_window(
        wave, host_lnu, agn_lnu, dust_lnu, frac, lam_min, lam_max
    )
    return jnp.power(10.0, log_power)


#: Newton steps of the conserving window solve beside ``polar_dust``
#: (:func:`refine_conserving_window_root`): three reach 1e-14 from the chord.
CONSERVING_WINDOW_REFINE_STEPS: int = 3
