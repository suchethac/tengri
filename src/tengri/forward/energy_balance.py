# SPDX-License-Identifier: BSD-3-Clause
r"""Canonical dust energy-balance integral (``L_absorbed``).

Single source of truth for the bolometric absorbed luminosity that feeds
dust IR re-emission (#922). Every exact-path computation of ``L_absorbed``
goes through :func:`_peak_factored_trapezoid` here, which is the
peak-factored (float32-safe) wrapper around
:func:`tengri.components.lyc.edge_trapezoid` -- the one step-at-the-edge
quadrature (see that module's docstring) -- rather than a hand-rolled
trapezoid; the build-time LUT in
:mod:`tengri.components.dust.energy_balance_precompute` is the precomputed
factorization of the *same* integral and must agree with it.

The two public spellings are **not** equally traveled. Measured 2026-08-04,
every call site in ``src/``, ``dust/two_component.py`` (twice),
``dust/component.py``, ``dust/wg00_model.py``, calls
:func:`bolometric_absorbed_log10`. :func:`bolometric_absorbed` has no live
caller and is not re-exported; it survives as the linear statement of the
contract the LUT is checked against, and is exercised only by tests. This
paragraph used to say the opposite, naming the linear form as the path
everything took, which is worth knowing when reading either function's guard
semantics, see :func:`_peak_factored_trapezoid` and #1527.

Physics convention: Lyman-continuum photons (:math:`\lambda <`
:data:`tengri.components.lyc.LYMAN_LIMIT_AA`, 911.76 Å) ionize hydrogen,
their energy re-emerges as nebular line and continuum emission, not as dust
heating, so they are excluded from the energy-balance integral, matching
CIGALE [1]_. The bracket grid cell straddling the edge is handled by the
step model, not linear interpolation (#537/#2447 generalized): see
:mod:`tengri.components.lyc`.

**Sign convention** (changed alongside the edge-aware quadrature): the
signed magnitude returned by :func:`bolometric_absorbed` /
:func:`bolometric_absorbed_log10` now follows the sign of
:math:`L_\nu^{\rm intr} - L_\nu^{\rm att}` itself
(:func:`tengri.components.lyc.edge_trapezoid` integrates with a
positive-oriented measure, ``|dx|``, regardless of whether the quadrature
variable is ascending or descending) rather than the grid orientation of
``nu``. Every caller either takes ``jnp.abs()`` of the result or combines
it with another term of the SAME (consistently reoriented) sign via
:func:`tengri.utils.scale.log10_add`, so the magnitude any caller observes
is unchanged; only the sign's own bookkeeping convention is.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp

from tengri.components.lyc import (
    LYMAN_LIMIT_AA,
    edge_trapezoid,
    ionizing_mask,
    log10_lyc_luminosity,
)


def warn_if_corrupt(log_l_absorbed: jnp.ndarray, *, component: str) -> None:
    """Attribute a ``+inf`` energy balance to its component, on the eager path.

    ``+inf`` is loud (it reaches ``L_ir`` and surfaces as a NaN fit) but on
    its own it says nothing about *where* the corruption entered. This supplies
    that, so the user is not left bisecting a NaN.

    Parameters
    ----------
    log_l_absorbed : array_like, shape ()
        The log10 absorbed luminosity just computed [dex].
    component : str
        Name of the calling component, quoted in the message.

    Notes
    -----
    **Not JIT-compatible by design, and safe to call from JIT-compatible code.**
    ``float()`` raises ``ConcretizationTypeError`` under any ``jit``/``grad``/
    ``vmap``, which is caught and treated as "nothing concrete to inspect",
    the same discipline as ``SFHBeforeBigBangWarning`` in the stellar
    component. Inference explores corrupt draws routinely and a per-sample
    warning would be unusable, so the ``+inf`` travels unannounced there.
    """
    try:
        value = float(log_l_absorbed)
    except (jax.errors.ConcretizationTypeError, TypeError):
        return  # tracing, no concrete value to inspect
    if value != float("inf"):
        return
    from tengri.config.exceptions import CorruptEnergyBalanceWarning

    warnings.warn(
        f"The dust energy balance in {component!r} received a non-finite SED, so "
        "L_absorbed is +inf and every quantity derived from it (L_ir, the dust IR "
        "emission, the FIR-radio correlation) will be inf or NaN. The intrinsic or "
        "attenuated SED reaching this component already contained Inf/NaN, check "
        "for an extreme metallicity driving an Inf*0 SSP flux, or an attenuation "
        "curve amplifying in the far UV. This is reported rather than silently "
        "clamped to zero absorption (#1527).",
        CorruptEnergyBalanceWarning,
        stacklevel=3,
    )


def absorbed_integrand(
    sed_intrinsic: jnp.ndarray,
    sed_attenuated: jnp.ndarray,
    wave: jnp.ndarray,
    lyman_cutoff_aa: float | None,
) -> jnp.ndarray:
    r"""LyC-masked absorbed integrand :math:`L_\nu^{\rm intr} - L_\nu^{\rm att}`.

    The single shared definition of "energy removed from the SED by dust
    attenuation" (#922): every canonical ``L_absorbed`` integral in this
    module, plus :func:`tengri.utils.sed_quantities.compute_l_dust_absorbed`,
    builds its integrand through this function so the two public spellings of
    the same quantity cannot silently disagree on the Lyman-continuum mask.

    Parameters
    ----------
    sed_intrinsic : array_like, shape (n_wave,)
        Intrinsic SED before dust attenuation [erg/s/Hz].
    sed_attenuated : array_like, shape (n_wave,)
        Dust-attenuated SED [erg/s/Hz].
    wave : array_like, shape (n_wave,)
        Wavelength grid [Angstrom]; used only for the Lyman-continuum mask.
    lyman_cutoff_aa : float or None
        Lyman-continuum cutoff [Angstrom]; energy absorbed at ionizing
        nodes (``wave < lyman_cutoff_aa``, :func:`tengri.components.lyc.
        ionizing_mask`) is excluded. ``None`` disables the mask (the full
        grid is integrated).

    Returns
    -------
    ndarray, shape (n_wave,)
        The (optionally masked) per-bin absorbed luminosity density
        [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes, pure ``jnp``; ``lyman_cutoff_aa`` is a static
    Python value, so the mask branch resolves at trace time.

    This per-node mask alone is NOT what makes the Lyman edge exact -- a
    plain trapezoid over its output would reintroduce the #537/#2447
    partial-bin ramp across the bracket cell. Every caller in this module
    pairs it with :func:`_peak_factored_trapezoid`'s ``side="nonionizing"``
    selection, which reads this array's ionizing-side nodes only through
    the exact step-model rectangle (never through a linear ramp), so the
    zeroing here is redundant-but-harmless rather than load-bearing for
    exactness; see :mod:`tengri.components.lyc`.
    """
    absorbed_lnu = sed_intrinsic - sed_attenuated
    if lyman_cutoff_aa is not None:
        absorbed_lnu = jnp.where(ionizing_mask(wave, edge_aa=lyman_cutoff_aa), 0.0, absorbed_lnu)
    return absorbed_lnu


def _peak_factored_trapezoid(
    integrand: jnp.ndarray,
    wave: jnp.ndarray,
    *,
    side: str = "all",
    edge_aa: float = LYMAN_LIMIT_AA,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Integrate ``integrand/peak`` over frequency, returning the factored pieces.

    The absorbed luminosity is a product of two individually representable
    factors (an integrand of ~1e28 erg/s/Hz and a frequency span of ~1e15 Hz)
    whose product (~1e43 erg/s) exceeds the float32 ceiling of 3.4e38. Dividing
    the integrand by its own peak makes the reduction O(1e15), so no
    intermediate leaves float32 range; the caller re-applies ``peak``, in log
    space where it must.

    The reduction itself is :func:`tengri.components.lyc.edge_trapezoid`
    (``variable="nu"``), not a plain ``jnp.trapezoid``: the one place this
    module's bracket-cell straddling the Lyman edge gets the step-model
    treatment instead of a linear ramp across it (do not re-derive the
    bracket weights here, call into :mod:`tengri.components.lyc`).

    Parameters
    ----------
    integrand : array_like, shape (n_wave,)
        Sampled integrand [erg/s/Hz].
    wave : array_like, shape (n_wave,)
        Wavelength grid, ascending [Angstrom].
    side : {"all", "ionizing", "nonionizing"}, optional
        Forwarded to :func:`tengri.components.lyc.edge_trapezoid`. Default
        ``"all"``.
    edge_aa : float, optional
        Lyman edge [Angstrom]. Default :data:`tengri.components.lyc.LYMAN_LIMIT_AA`.

    Returns
    -------
    signed_norm : ndarray, shape ()
        ``edge_trapezoid(integrand / peak, wave, variable="nu", side=side,
        edge_aa=edge_aa)``, signed. Positively oriented (follows the sign of
        ``integrand`` itself, not the grid orientation of ``nu`` -- see the
        module docstring's "Sign convention" note).
    peak : ndarray, shape ()
        The factored-out scale (1.0 when the integrand is zero or non-finite).
    ok : ndarray, shape (), bool
        True when the integral is a usable finite number. False for *both* an
        all-zero integrand and a corrupt one, use ``corrupt`` to tell which.
    corrupt : ndarray, shape (), bool
        True when the integrand contained ``Inf``/``NaN``, or when the reduction
        went non-finite despite a finite positive peak. Disjoint from the
        all-zero case, which leaves ``corrupt`` False.

    Notes
    -----
    ``ok`` alone merges two situations that are *not* the same answer: an
    all-zero integrand (nothing absorbed, a true zero) and a non-finite one
    (something upstream produced Inf or NaN). ``corrupt`` separates them, and
    the two callers **deliberately answer it differently** (#1527):

    * :func:`bolometric_absorbed_log10`: the live form, on every production
      path, reports ``+inf``, matching :func:`tengri.utils.scale.log10_add`,
      whose comment argues that folding a non-finite term into the zero
      sentinel "would report an overflowed term as exactly zero, a fail-open
      on precisely the axis this module exists to close".
    * :func:`bolometric_absorbed`: the linear form, with no caller in ``src/``
      keeps clamping to ``0.0``. That clamp is inherited, not chosen: #922's
      table lists it as a property of the retired compositional kernel,
      preserved to avoid changing behavior for a real artifact class (Inf·0
      from extreme-metallicity SSP fluxes, BUG-NSS-02), and it is pinned by
      ``tests/physics/conservation/test_lyc_mask_energy_balance.py::TestFiniteGuard``.

    The split is what makes both true at once, and it is only defensible
    because of the asymmetry: the test that pins the clamp guards the function
    nothing calls, so tightening the live path costs nothing there. An earlier
    attempt changed the shared flag for both and broke that test, which is what
    surfaced the asymmetry in the first place.
    """
    # stop_gradient: pure factorization constant (#1436). The caller re-applies this
    # peak to signed_norm, so the product is peak-independent and the peak's
    # derivative is analytically zero. Cancels in float64, not in float32.
    peak = jax.lax.stop_gradient(jnp.max(jnp.abs(integrand), initial=0.0))
    peak_finite = jnp.isfinite(peak)
    usable = peak_finite & (peak > 0)
    safe_peak = jnp.where(usable, peak, 1.0)
    signed_norm = edge_trapezoid(
        integrand / safe_peak, wave, variable="nu", side=side, edge_aa=edge_aa
    )
    norm_finite = jnp.isfinite(signed_norm)
    # A non-finite peak means the integrand itself carried Inf/NaN. A finite
    # positive peak with a non-finite reduction means the sum went bad on the
    # way. An all-zero integrand is neither: peak == 0 leaves both flags clear
    # of ``corrupt`` and yields signed_norm == 0, i.e. an honest "nothing
    # absorbed" rather than a failure.
    corrupt = ~peak_finite | (usable & ~norm_finite)
    return signed_norm, safe_peak, usable & norm_finite, corrupt


def _log10_signed_edge_integral(
    integrand: jnp.ndarray, wave: jnp.ndarray, *, side: str, edge_aa: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Shared log10-magnitude/sign wrapper around :func:`_peak_factored_trapezoid`.

    Factored out so :func:`bolometric_absorbed_log10` (``side="all"`` or
    ``"nonionizing"``) and :func:`bolometric_lyc_log10` (``side="ionizing"``)
    share one corrupt/zero-sentinel bookkeeping instead of each re-deriving
    it (#1527's split, see :func:`_peak_factored_trapezoid`).

    Returns
    -------
    log_magnitude, sign : ndarray, shape ()
        Same contract as :func:`bolometric_absorbed_log10`'s return value.
    """
    from tengri.utils.scale import log10_magnitude

    signed_norm, peak, ok, corrupt = _peak_factored_trapezoid(
        integrand, wave, side=side, edge_aa=edge_aa
    )
    log_norm = log10_magnitude(jnp.where(ok, signed_norm, 0.0))
    # Corrupt beats the -inf sentinel: -inf powers back to exactly 0.0, so
    # reporting it here would say "nothing absorbed" about an input nobody can
    # integrate. +inf survives log10_add and reaches L_ir, where it is visible.
    # The sign of an uncomputable integral is NaN, not 0.0; 0.0 already means
    # "no absorption" in this contract.
    log_magnitude = jnp.where(corrupt, jnp.inf, log_norm + jnp.log10(peak))
    sign = jnp.where(ok, jnp.sign(signed_norm), 0.0)
    return log_magnitude, jnp.where(corrupt, jnp.nan, sign)


def bolometric_lyc_log10(
    sed_lnu: jnp.ndarray,
    wave: jnp.ndarray,
    *,
    edge_aa: float = LYMAN_LIMIT_AA,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""log10 of the raw (pre-fesc/fdust) Lyman-continuum luminosity, ionizing side only.

    .. math::

        L_{\rm LyC} = \int_{\lambda < \lambda_{\rm edge}} L_\nu(\lambda)\, d\nu

    the credited-population LyC luminosity #2539's dust-heating credit
    (:func:`log10_fdust_lyc_credit`, :func:`log10_add_fdust_credit`) is built
    from. Computed via :func:`tengri.components.lyc.edge_trapezoid` with
    ``side="ionizing"``, so the bracket cell straddling ``edge_aa`` is held at
    the step model's last-ionizing-node rectangle rather than ramped -- the
    same primitive :func:`bolometric_absorbed_log10` uses for the
    complementary ``"nonionizing"`` side, so the credit and the dust-heating
    exclusion can never drift onto different bracket-cell conventions.

    Parameters
    ----------
    sed_lnu : array_like, shape (n_wave,)
        SED of the credited population [erg/s/Hz]. Typically the raw
        (un-fesc-masked) stellar or per-age-weighted SED; the caller decides
        which population is "credited" (#2539 item 2).
    wave : array_like, shape (n_wave,)
        Wavelength grid, ascending [Angstrom].
    edge_aa : float, optional
        Lyman edge [Angstrom]. Default :data:`tengri.components.lyc.LYMAN_LIMIT_AA`.

    Returns
    -------
    log_magnitude, sign : ndarray, shape ()
        Same sentinel contract as :func:`bolometric_absorbed_log10`: ``-inf``
        when the population has no LyC luminosity, ``+inf`` for a corrupt
        (non-finite) input, ``sign`` is ``NaN`` in that corrupt case.

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes, linear in ``sed_lnu``.

    Absorbed/credited luminosities are ~1e43 erg/s, six decades past the
    float32 ceiling (#1206); this log form is peak-factored the same way as
    :func:`bolometric_absorbed_log10`.

    A thin wrapper around :func:`tengri.components.lyc.log10_lyc_luminosity`
    (the ONE LyC-luminosity implementation, G1/G2) that also reports the
    integral's sign: ``sed_lnu`` is a physical :math:`L_\nu` (non-negative
    pointwise), so the sign is always 0.0 (exactly zero luminosity) or 1.0
    (some), never -1.0, unless the input is corrupt (``NaN``).
    """
    from tengri.utils.scale import _not_computable

    log_magnitude = log10_lyc_luminosity(sed_lnu, wave, edge_aa=edge_aa, axis=-1)
    corrupt = _not_computable(log_magnitude)
    sign = jnp.where(corrupt, jnp.nan, jnp.where(jnp.isneginf(log_magnitude), 0.0, 1.0))
    return log_magnitude, sign


def bolometric_absorbed_log10(
    sed_intrinsic: jnp.ndarray,
    sed_attenuated: jnp.ndarray,
    nu: jnp.ndarray,
    *,
    wave: jnp.ndarray,
    lyman_cutoff_aa: float | None = LYMAN_LIMIT_AA,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""log10 of the absorbed bolometric luminosity, the float32-safe contract.

    .. math::

        \log_{10} L_{\rm abs} = \log_{10} \left| \int_{\lambda \ge
            \lambda_{\rm LyC}} \left[ L_\nu^{\rm intr}(\lambda) -
            L_\nu^{\rm att}(\lambda) \right] d\nu \right|

    where :math:`L_\nu^{\rm intr}` is the intrinsic (unattenuated) SED
    [erg/s/Hz], :math:`L_\nu^{\rm att}` the dust-attenuated SED [erg/s/Hz],
    and :math:`\lambda_{\rm LyC}` the Lyman-continuum cutoff [Angstrom].

    Same integral, same LyC convention, and the same guard semantics as
    :func:`bolometric_absorbed`, only the output representation differs.
    Magnitude and sign are returned separately because that *is* what a
    signed quantity looks like in log space; callers that only need the
    energy (nearly all of them, the sign merely tracks whether the two SEDs
    canceled or reinforced -- see the module docstring's "Sign convention"
    note) discard the sign, while callers combining two absorbed terms need
    it to reproduce ``|a + b|`` rather than ``|a| + |b|``.

    Parameters
    ----------
    sed_intrinsic : array_like, shape (n_wave,)
        Intrinsic SED before dust attenuation [erg/s/Hz].
    sed_attenuated : array_like, shape (n_wave,)
        Dust-attenuated SED [erg/s/Hz].
    nu : array_like, shape (n_wave,)
        Unused (retained for call-site compatibility, #2570): every call
        site constructs this as ``C_AA / wave``, so the integral is instead
        computed directly from ``wave`` via
        :func:`tengri.components.lyc.edge_trapezoid`, which needs the
        wavelength grid (not just its frequency image) to place the
        step-model bracket cell at ``lyman_cutoff_aa``.
    wave : array_like, shape (n_wave,)
        Wavelength grid [Angstrom]; used for the Lyman-continuum mask AND
        the edge-aware quadrature.
    lyman_cutoff_aa : float or None, optional
        Lyman-continuum cutoff [Angstrom]. ``None`` disables the mask
        (integrates ``side="all"``); a value excludes the ionizing side
        (``side="nonionizing"``) with the step placed exactly at that value.
        Default :data:`tengri.components.lyc.LYMAN_LIMIT_AA` (911.76 Å).

    Returns
    -------
    log_magnitude : ndarray, shape ()
        :math:`\log_{10}(|L_{\rm abs}| / (\mathrm{erg/s}))` [dex]. ``-inf``
        when nothing is absorbed, which powers back to exactly 0.0. ``+inf``
        when the inputs are non-finite, a corrupt integrand is *not* folded
        into the zero sentinel (#1527); see
        :class:`tengri.config.exceptions.CorruptEnergyBalanceWarning`.
    sign : ndarray, shape ()
        Sign of the signed integral, for combining terms via
        :func:`tengri.utils.scale.log10_add`. 0.0 when nothing is absorbed,
        ``NaN`` when the integrand is corrupt, an uncomputable integral has no
        sign, and 0.0 is already spoken for.

    Notes
    -----
    **JIT-compatible**: yes, pure ``jnp``; ``lyman_cutoff_aa`` is a static
    Python value. Safe under ``grad`` and ``vmap``: the zero case takes the
    where-dummy path, so no NaN reaches the backward pass.

    The ``peak`` factored out of the integrand cancels analytically between
    the two log terms, so the gradient is that of the unfactored integral.

    Absorbed luminosities are ~1e43 erg/s, six decades past the float32
    ceiling, so this log form, not :func:`bolometric_absorbed`, is what a
    pure-float32 (JAX-Metal) forward pass must consume (#1206).
    """
    del nu  # unused; see the Parameters entry above (#2570 compatibility)
    integrand = absorbed_integrand(sed_intrinsic, sed_attenuated, wave, lyman_cutoff_aa)
    side = "all" if lyman_cutoff_aa is None else "nonionizing"
    edge_aa = LYMAN_LIMIT_AA if lyman_cutoff_aa is None else lyman_cutoff_aa
    return _log10_signed_edge_integral(integrand, wave, side=side, edge_aa=edge_aa)


def log10_fdust_lyc_credit(log_l_lyc: jnp.ndarray, f_dust: jnp.ndarray) -> jnp.ndarray:
    r"""``log10(f_dust * L_LyC)``, gradient-safe at ``f_dust == 0`` (#2539).

    ``f_dust`` is the fraction of Lyman-continuum photons that dust grains
    inside HII regions absorb (CIGALE convention:
    ``dust.luminosity = (lum_ly_young + lum_ly_old) * fdust``,
    ``pcigale/sed_modules/nebular.py:191-193``). That energy is credited to
    the dust IR budget as ``log10(f_dust) + log_L_lyc``, a plain log-add
    that is exact but has a singular derivative at ``f_dust == 0``
    (:math:`d/d(\mathrm{fdust})\,\log_{10}(\mathrm{fdust}) = 1/(\mathrm{fdust}
    \cdot \ln 10) \to \infty`) which a naive ``jnp.where`` around
    ``jnp.log10`` can turn into ``NaN`` under ``grad`` (``inf * 0``).

    This is the *double-where* idiom used elsewhere for a value with a
    removable singularity at a boundary (see :func:`tengri.components.stellar.
    sfh.mean_sfh.dpl`'s ``T_safe`` treatment): the argument to :func:`jnp.log10`
    is clamped to a finite dummy (1.0) *before* the log, so the log itself
    never sees zero, and the ``-inf`` sentinel for "no credit" is selected
    by a *second*, independent ``jnp.where`` whose off-branch is a bare
    constant (zero backward-pass contribution, not ``NaN``).

    Parameters
    ----------
    log_l_lyc : array_like, shape ()
        log10(L_LyC / (erg/s)) [dex]: the Lyman-continuum luminosity of
        whichever stellar population the nebular escape/dust factor was
        applied to (the *same* population, #2539 item 2). Independent of
        ``f_dust``.
    f_dust : array_like, shape ()
        Dust-absorption fraction of ionizing photons, in [0, 1].

    Returns
    -------
    ndarray, shape ()
        ``log10(f_dust) + log_l_lyc`` [dex] where ``f_dust > 0``,
        ``-inf`` (bit-identical to the exact zero-credit value) otherwise.

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes, including at
    ``f_dust == 0`` (grad is exactly 0.0 there, the same "flat at the
    dead boundary" choice :func:`~tengri.components.stellar.sfh.mean_sfh.dpl`
    makes, rather than blowing up); finite and growing for
    ``f_dust -> 0+`` (e.g. ``~4.3e7`` at ``1e-8``), moderate away from the
    boundary (e.g. ``~1.09`` at ``0.3``, when ``log_l_lyc`` does not itself
    depend on ``f_dust``).
    """
    safe_fdust = jnp.where(f_dust > 0.0, f_dust, 1.0)
    log_fdust = jnp.log10(safe_fdust)
    candidate = log_fdust + log_l_lyc
    return jnp.where(f_dust > 0.0, candidate, -jnp.inf)


def log10_add_fdust_credit(
    log_l_absorbed: jnp.ndarray, log_l_lyc: jnp.ndarray, f_dust: jnp.ndarray
) -> jnp.ndarray:
    r"""``log10(L_absorbed + f_dust * L_LyC)``, smooth in ``f_dust`` (#2539 item 3).

    Replaces the ``log10_fdust_lyc_credit(...)`` + ``log10_add(...)`` pairing
    with the fused, exact form the owner asked for:

    .. math::

        \log_{10}(L_{\rm abs} + f_{\rm dust} L_{\rm LyC}) = \log_{10} L_{\rm abs}
        + \log_{10}\!\left(1 + f_{\rm dust} \cdot 10^{\log_{10} L_{\rm LyC}
        - \log_{10} L_{\rm abs}}\right)

    computed with ``jnp.log1p`` so the argument to the log is never literally
    zero. ``log10_fdust_lyc_credit`` computes ``log10(fdust) + log_l_lyc`` in
    isolation and clamps its OWN gradient to exactly 0.0 at ``fdust == 0``
    (see its docstring) -- correct for that isolated quantity, but wrong once
    chained into this combine: the zero upstream gradient multiplies through
    and zeroes the gradient of the COMBINED ``log_l_absorbed`` too, even
    though ``L_absorbed`` is exactly LINEAR in ``fdust``
    (:math:`dL_{\rm abs}/d f_{\rm dust} = L_{\rm LyC}`, a finite nonzero
    constant at every ``fdust``, including 0). This form never computes
    ``log10(fdust)`` at all, so there is no singularity to clamp around.

    Parameters
    ----------
    log_l_absorbed : array_like, shape ()
        log10(L_absorbed / (erg/s)) [dex], the running absorbed-luminosity
        sum this component has accumulated so far. A plain (positively
        oriented) magnitude, not a signed quantity -- every call site on this
        seam reaches this function only after its own sum has already been
        reduced to a magnitude (#2539 item 2).
    log_l_lyc : array_like, shape ()
        log10(L_LyC / (erg/s)) [dex]: the RAW (pre-``fdust``) Lyman-continuum
        luminosity of the credited population. ``-inf`` if that population
        has no LyC luminosity.
    f_dust : array_like, shape ()
        Dust-absorption fraction of ionizing photons, in [0, 1].

    Returns
    -------
    ndarray, shape ()
        ``log10(L_absorbed + f_dust * L_LyC)`` [dex]. Bit-identical to
        ``log_l_absorbed`` at ``f_dust == 0`` (``jnp.log1p(0) == 0``
        exactly) and to ``log10_fdust_lyc_credit(log_l_lyc, f_dust)`` when
        ``log_l_absorbed`` is ``-inf`` (nothing else absorbed).

    Notes
    -----
    JIT/grad/vmap-safe. Robust to ``log_l_absorbed == -inf`` (e.g. a
    fully-transparent, ``tau == 0`` screen): the smooth ratio form would
    otherwise divide a zero base into a possibly-nonzero credit
    (``10**(log_l_lyc - (-inf)) == inf``, ``inf * 0 == NaN`` under naive
    evaluation), so that case is guarded by a where-dummy and falls back to
    :func:`log10_fdust_lyc_credit`'s own ``-inf``-safe value -- a condition
    that depends on ``tau``, not on ``f_dust``, so it does not reintroduce
    the singularity this function exists to avoid.
    """
    from tengri.utils.scale import LN10, pow10

    absorbed_is_zero = jnp.isneginf(log_l_absorbed)
    safe_log_l_absorbed = jnp.where(absorbed_is_zero, 0.0, log_l_absorbed)
    ratio = pow10(log_l_lyc - safe_log_l_absorbed)
    smooth = safe_log_l_absorbed + jnp.log1p(jnp.asarray(f_dust) * ratio) / LN10
    fallback = log10_fdust_lyc_credit(log_l_lyc, f_dust)
    return jnp.where(absorbed_is_zero, fallback, smooth)


def bolometric_absorbed(
    sed_intrinsic: jnp.ndarray,
    sed_attenuated: jnp.ndarray,
    nu: jnp.ndarray,
    *,
    wave: jnp.ndarray,
    lyman_cutoff_aa: float | None = LYMAN_LIMIT_AA,
) -> jnp.ndarray:
    r"""Signed bolometric luminosity absorbed by dust, LyC-masked.

    .. math::

        L_{\rm abs} = \int_{\lambda \ge \lambda_{\rm LyC}}
            \left[ L_\nu^{\rm intr}(\lambda) - L_\nu^{\rm att}(\lambda) \right]
            d\nu

    where :math:`L_\nu^{\rm intr}` is the intrinsic (unattenuated) SED
    [erg/s/Hz], :math:`L_\nu^{\rm att}` the dust-attenuated SED [erg/s/Hz],
    and :math:`\lambda_{\rm LyC}` the Lyman-continuum cutoff [Angstrom]. The
    grid cell straddling :math:`\lambda_{\rm LyC}` is handled by the step
    model (:mod:`tengri.components.lyc`), not linear interpolation.

    Parameters
    ----------
    sed_intrinsic : array_like, shape (n_wave,)
        Intrinsic SED before dust attenuation [erg/s/Hz].
    sed_attenuated : array_like, shape (n_wave,)
        Dust-attenuated SED [erg/s/Hz].
    nu : array_like, shape (n_wave,)
        Unused (retained for call-site compatibility, #2570) -- see
        :func:`bolometric_absorbed_log10`'s matching parameter.
    wave : array_like, shape (n_wave,)
        Wavelength grid [Angstrom]; used for the Lyman-continuum mask AND
        the edge-aware quadrature.
    lyman_cutoff_aa : float or None, optional
        Lyman-continuum cutoff [Angstrom]; energy absorbed at the ionizing
        side of ``lyman_cutoff_aa`` is excluded (those photons ionize H, they
        do not heat dust). ``None`` disables the mask (integrate the full
        grid, ``side="all"``). Default :data:`tengri.components.lyc.
        LYMAN_LIMIT_AA` (911.76 Å).

    Returns
    -------
    ndarray, shape ()
        Signed absorbed bolometric luminosity [erg/s]. Positively oriented
        (follows the sign of :math:`L_\nu^{\rm intr} - L_\nu^{\rm att}`
        itself; see the module docstring's "Sign convention" note) --
        callers apply ``jnp.abs`` regardless, for robustness, and any
        energy-balance relaxation factor (``dust_eta_balance``) themselves.
        Non-finite *inputs* (e.g. Inf·0 artifacts from extreme-metallicity
        SSP fluxes, BUG-NSS-02 era) are clamped to 0.0, the guard the
        retired compositional kernel carried; identity for finite inputs.

    Notes
    -----
    **JIT-compatible**: yes, pure ``jnp``; ``lyman_cutoff_aa`` is a static
    Python value, so the mask branch resolves at trace time. Safe under
    ``grad`` and ``vmap``.

    **Not float32-representable.** Absorbed luminosities are ~1e43 erg/s,
    six decades past the float32 ceiling, so this returns ``inf`` under pure
    float32 no matter how the reduction is arranged. Use
    :func:`bolometric_absorbed_log10` there (#1206). The integrand is
    peak-factored so that overflow is confined to that final re-scaling:
    previously the reduction itself overflowed and the non-finite guard
    turned the ``inf`` into **0.0**, silently switching dust IR emission off
    rather than failing loudly.

    The fast-path LUT (:func:`tengri.components.dust.
    energy_balance_precompute.lut_l_absorbed_stellar`) is the precomputed
    factorization of this integral over the SSP grid; the two must agree
    (contract test: ``tests/contract/test_energy_balance_lut.py``).

    Cross-code conventions: CIGALE zeroes its attenuation curves at
    λ ≤ 91.2 nm and Bagpipes masks the ionizing continuum via ``fesc``;
    both exclude LyC from dust heating, as here. FSPS does *not* mask the
    LyC, so ``L_dust`` comparisons against FSPS/Prospector carry this
    convention difference. The mask also protects the integral from
    attenuation laws whose far-UV extrapolation amplifies (k(λ) < 0 below
    the law's calibrated range, e.g. Calzetti).

    References
    ----------
    .. [1] Boquien, M., et al. 2019, A&A, 622, A103.
           https://doi.org/10.1051/0004-6361/201834156

    """
    del nu  # unused; see the Parameters entry above (#2570 compatibility)
    integrand = absorbed_integrand(sed_intrinsic, sed_attenuated, wave, lyman_cutoff_aa)
    side = "all" if lyman_cutoff_aa is None else "nonionizing"
    edge_aa = LYMAN_LIMIT_AA if lyman_cutoff_aa is None else lyman_cutoff_aa
    signed_norm, peak, ok, _corrupt = _peak_factored_trapezoid(
        integrand, wave, side=side, edge_aa=edge_aa
    )
    # ``_corrupt`` is deliberately discarded here while
    # ``bolometric_absorbed_log10`` acts on it (#1527). This is the linear form:
    # no caller in ``src/``, and its clamp is pinned by TestFiniteGuard as the
    # BUG-NSS-02 behavior #922 carried forward. Tightening it would change that
    # contract for no live benefit, so the two forms differ on purpose.
    return jnp.where(ok, signed_norm * peak, 0.0)
