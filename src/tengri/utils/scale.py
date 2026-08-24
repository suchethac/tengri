# SPDX-License-Identifier: BSD-3-Clause
"""Range-safe application of large log10 scale factors (float32 feasibility).

Physical scales in this code (``mass_scale`` ~ 1e42 erg/s, ``d_L²`` ~ 1e56 cm²,
``flux_scale`` ~ 1e-58) fall outside the float32 window ``[1.18e-38, 3.40e38]``.
This module applies such a scale to an array by carrying it as a ``log10``
offset and peak-normalizing the array, so no out-of-range intermediate is
materialized. In float64 the result equals the naive product to machine
precision; in float32 it stays finite whenever the *net* magnitude is in range.
See issue #1186.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy

LN10 = 2.302585092994046
LOG10_4PI = float(jnp.log10(4.0 * jnp.pi))  # ~1.09921


def representable_floor(value: float) -> float:
    """Raise a guard floor to the working dtype's smallest normal if it is below it.

    Parameters
    ----------
    value : float
        The intended floor, as written at the call site.

    Returns
    -------
    float
        ``max(value, finfo(working dtype).tiny)``: a static Python float, safe
        as a ``jnp.maximum`` / ``jnp.clip`` bound under JIT.

    Notes
    -----
    **JIT-compatible**: yes; resolved at trace time from
    ``jnp.result_type(float)``, so it is a compile-time constant.

    float32's smallest normal is 1.18e-38 and its smallest subnormal 1.4e-45, so
    a literal floor below that is **exactly 0.0** there; the guard reads as
    protection and provides none. The tree carries 36 such floors (``1e-50``,
    ``1e-60``, ``1e-100``, ``1e-300``), every one inert in the precision #1206
    exists to deliver (#1492).

    Perversely, the smaller the literal the worse it is: ``1e-30`` survives in
    float32, ``1e-100`` does not, and the latter reads as the more careful
    choice.

    **float64 is unchanged by construction.** ``finfo(float64).tiny`` is
    2.2e-308, below every floor the tree uses, so the ``max`` returns the
    original literal and float64 results are bit-identical. Only float32, where
    the literal was doing nothing at all, sees a different number.

    A floor sized for a *value* is still not sized for a *derivative*; division's
    VJP needs the denominator squared, so a derivative-safe bound is
    ``sqrt(tiny)`` (~1.1e-19 in float32). See #1397, #1436, #1439. That case has
    its own helper, :func:`representable_denominator`; use it whenever the
    floored quantity is a denominator. This function is **not** sufficient there
    and will silently pass such a site through: ``1e-30`` is above float32's
    ``tiny``, so it is returned unchanged while its VJP divides by zero (#1860).

    Examples
    --------
    >>> import jax
    >>> from tengri.utils.scale import representable_floor
    >>> with jax.enable_x64(True):
    ...     representable_floor(1e-50) == 1e-50  # float64: untouched
    True
    """
    return max(float(value), float(jnp.finfo(jnp.result_type(float)).tiny))


def representable_denominator(value: float) -> float:
    r"""Raise a guard floor to the smallest denominator whose *derivative* is finite.

    Parameters
    ----------
    value : float
        The intended floor, as written at the call site [dimensionless].

    Returns
    -------
    float
        ``max(value, sqrt(tiny))`` for the working dtype; a static Python float,
        safe as a ``jnp.maximum`` / ``jnp.clip`` bound under JIT.

    Notes
    -----
    **JIT-compatible**: yes, resolved at trace time from
    ``jnp.result_type(float)``, so it is a compile-time constant.

    The third member of the family, and the one :func:`representable_floor`'s
    Notes already specify: *a floor sized for a value is still not sized for a
    derivative*. Use this wherever the floored quantity is a **denominator**;
    use :func:`representable_floor` when it is merely an argument to a ``log``.

    A quotient's reverse-mode rule carries a term in the denominator squared,

    .. math::

        \frac{\partial}{\partial d}\left(\frac{n}{d}\right) = -\frac{n}{d^{2}}

    where :math:`n` is the numerator and :math:`d` the floored denominator (both
    dimensionless here). So the reverse pass needs :math:`1/d^{2}` to be
    representable, not merely :math:`d`. That is a strictly stronger requirement,
    and it bites at floors that :func:`representable_floor` passes through
    untouched:

    ==========  ====================  ====================  ==========
    floor       ``floor**2`` (f32)    ``1/floor**2`` (f32)  usable
    ==========  ====================  ====================  ==========
    ``1e-30``   ``0.0``               ``inf``               no
    ``1e-20``   ``1e-40``             ``1e40`` (> max)      no
    ``1e-18``   ``1e-36``             ``1e36``              yes
    ==========  ====================  ====================  ==========

    ``1e-30`` is *above* float32's ``tiny`` (1.175e-38), so
    ``representable_floor(1e-30)`` returns it unchanged and a floor census
    reports the site as clean; while its VJP divides by zero (#1860). ``1e-20``
    fails too, despite ``1e-40`` being a nominal subnormal, because the
    reciprocal overflows.

    The bound is analytic. Requiring :math:`1/d^{2} \le \mathrm{max}` gives
    :math:`d \ge 1/\sqrt{\mathrm{max}}`, and since
    :math:`1/\mathrm{max} \approx \mathrm{tiny}/4` the cliff sits at
    :math:`\sqrt{\mathrm{tiny}}/2`; 5.42e-20 in float32. ``sqrt(tiny)``
    (1.084e-19) is therefore one factor of two inside it, not on it.

    **float64 is unchanged for any floor at or above 1.5e-154**
    (``sqrt(finfo(float64).tiny)``), which covers the ``1e-30`` / ``1e-40``
    literals this guards, so those sites stay bit-identical in float64.

    Note the difference from :func:`representable_floor`, which is a no-op in
    float64 for *every* literal the tree uses. This one is not: a floor below
    1.5e-154; the tree carries ``1e-300``: is raised in float64 as well,
    because it is derivative-unsafe *there too* (``(1e-300)**2`` is ``0.0`` in
    float64, whose smallest normal is 2.2e-308). That is the intended behavior
    and not a float64 regression, but it does mean this helper cannot be applied
    blind: at such a site, float64 numbers move, and the move is the fix.

    References
    ----------
    .. [1] tengri issue #1860; the filter-integral guard floor NaNs the redshift
       gradient; ``(1e-30)**2`` flushes to zero in the VJP.

    Examples
    --------
    >>> import jax
    >>> from tengri.utils.scale import representable_denominator
    >>> with jax.enable_x64(True):
    ...     representable_denominator(1e-30) == 1e-30  # float64: untouched
    True
    """
    tiny = float(jnp.finfo(jnp.result_type(float)).tiny)
    return max(float(value), float(numpy.sqrt(tiny)))


def representable_exponent(value: float, *, base: float = 10.0) -> float:
    r"""Lower an exponent bound to the largest ``base**x`` the dtype can hold.

    The ceiling-side mirror of :func:`representable_floor`, and the same defect in
    the opposite direction: a saturating bound written for float64 does not merely
    stop protecting in float32, it **manufactures the** ``inf`` **it exists to
    prevent**.

    Parameters
    ----------
    value : float
        The intended exponent ceiling, as written at the call site [dex for
        ``base=10``, nats for ``base=e``].
    base : float, keyword-only, optional
        Base of the power the bound feeds. ``10.0`` (default) for
        ``10**clip(x, lo, hi)``; :data:`math.e` for ``exp(clip(x, lo, hi))``.
        One function rather than a per-base copy; the arithmetic is identical
        and only ``log(max)/log(base)`` changes.

    Returns
    -------
    float
        ``min(value, x_max)`` where ``x_max`` is the largest float for which
        ``base**x`` is finite in the working dtype; a static Python float, safe
        as a :func:`jax.numpy.clip` bound under JIT.

    Notes
    -----
    **JIT-compatible**: yes, resolved at trace time from
    ``jnp.result_type(float)``, so it is a compile-time constant.

    float32 tops out at 3.4028e38, i.e. :math:`10^{38.53}` or :math:`e^{88.72}`.
    A clip to ±50 dex therefore saturates to a value float32 cannot represent, so
    the guarded expression returns ``inf`` for every input the guard fires on.
    Measured: both Cue emission paths clipped to ±50 dex with the comment *"the
    clip is the only defense against NaN/inf poisoning a JAX gradient"*, and in
    pure float32 that clip was the sole reason the whole forward state went
    non-finite -- poisoning the dust energy balance, ``L_absorbed``, ``L_ir`` and
    every gradient through them (#1206).

    **The natural-exp form fails silently, which is worse.** ``qsogen``'s Planck
    terms are ``1 / (exp(clip(x, 0, 500)) - 1)``. In float32 ``exp(500)`` is
    ``inf``, so the *value* is ``0.0`` -- physically right for a Planck tail, so
    nothing announces it -- while the *gradient* is ``NaN``. Measured at ``x=600``:
    float64 gives value 7.1e-218 and derivative ``-0.0``; float32 gives ``0.0``
    and ``nan``.

    The bound is stepped strictly below ``log(max)/log(base)`` rather than set
    equal to it: ``base**that`` rounds *up* to ``inf`` in the last bits, so the
    exact value is not itself usable.

    **float64 is unchanged by construction.** Its ceiling is :math:`10^{308.25}`,
    above every exponent bound the tree writes, so the ``min`` returns the
    original literal and float64 results are bit-identical.

    This caps a *magnitude*, so it deliberately says nothing about ``-inf``, which
    remains the legitimate "this term is exactly zero" sentinel of
    :func:`log10_magnitude`.

    Examples
    --------
    >>> import jax
    >>> from tengri.utils.scale import representable_exponent
    >>> with jax.enable_x64(True):
    ...     representable_exponent(50.0) == 50.0  # float64: untouched
    True
    """
    # numpy, not jnp: this is called at trace time from inside jitted forwards, and
    # ``float()`` of a jnp array there is a ConcretizationTypeError. ``finfo.max``
    # alone is a numpy scalar (which is why representable_floor gets away with
    # ``jnp``), but taking a log of it is a JAX operation and the result traces.
    dtype = numpy.dtype(jnp.result_type(float))
    # The DERIVATIVE, not the value, sets the bound: d/dx base**x = base**x·ln(base),
    # so the headroom needed is ``max / ln(base)``. Capping the value alone leaves the
    # forward finite and the reverse pass NaN: clip contributes a zero gradient
    # there, and 0 * inf is NaN. Measured on Cue at the cap: forward 3.4028e38
    # (finite), gradient NaN, while float64 gives a clean 0.0.
    #
    # It costs 0.36 dex for base 10 and nothing for base e (ln(e) = 1), and
    # representable_floor's docstring already records the same value-vs-derivative
    # distinction for the floor side (#1397, #1436, #1439).
    headroom = numpy.finfo(dtype).max / max(numpy.log(base), 1.0)
    # One ULP *of the working dtype* below the limit: base**that rounds up to inf in
    # the last bits, so the exact value is not usable. Stepping in float64 would not
    # move it far enough to matter: float64's ULP at 38.5 is ~7e-15, float32's ~4e-6.
    limit = numpy.log(headroom) / numpy.log(base)
    ceiling = numpy.nextafter(dtype.type(limit), dtype.type(0.0))
    return min(float(value), float(ceiling))


def whiten(x, sigma):
    r""":math:`x / \sigma`, with the division made binding on the compiler.

    The single seam through which every noise-weighting in tengri passes:
    :math:`\chi^2` residuals, Gauss-Newton metrics, and the normal equations of
    the analytic emission-line marginalization.

    Dividing by :math:`\sigma` once is always representable; forming
    :math:`1/\sigma^2` never is, at the :math:`\sigma \sim 10^{-30}` of a real
    flux (:math:`1/\sigma^2 \sim 10^{59}` against a float32 ceiling of
    :math:`3.4\times10^{38}`). Applying this **twice** is therefore the only
    float32-safe spelling of :math:`N^{-1}`:

    .. math::

        J^\mathsf{T} N^{-1} J v \;=\; (J/\sigma)^\mathsf{T} (J/\sigma)\, v

    Writing it in that order is not sufficient. Under ``jax.jit``, when
    ``sigma`` is a compile-time constant, XLA re-associates
    :math:`(x/\sigma)/\sigma` into :math:`x \cdot (1/\sigma^2)` and
    constant-folds the reciprocal to ``inf``; ``0 * inf`` is ``NaN`` (#1535).
    Measured: without the barrier the double division is finite eagerly and
    NaN under ``jit``, with a literal ``inf`` in the compiled HLO. The
    ``optimization_barrier`` makes the intended order a *data dependency*,
    which the compiler must respect; a source-level grouping is only a
    suggestion.

    Parameters
    ----------
    x : array_like
        Quantity to whiten: a residual, a Jacobian-vector product, or a
        data-space vector.
    sigma : array_like
        1-σ uncertainty [same units as ``x``], broadcastable against ``x``.

    Returns
    -------
    ndarray
        ``x / sigma`` [dimensionless], shape of the broadcast inputs.

    Notes
    -----
    JIT/grad/vmap-safe, and it is only under JIT that the barrier does
    anything. ``optimization_barrier`` is semantically the identity, so values
    and derivatives are unchanged: verified bit-exact in float64.

    Lives here rather than beside the Gaussian likelihood so that
    ``observation/`` can use it without importing ``inference/`` (#1588).
    """
    return jax.lax.optimization_barrier(x / sigma)


def pow10(x):
    """10**x, computed as ``exp(x·ln10)`` to preserve the input dtype.

    Parameters
    ----------
    x : array_like
        Exponent [dimensionless].

    Returns
    -------
    ndarray
        ``10**x`` in the dtype of ``x``.

    Notes
    -----
    JIT/grad/vmap-safe. ``jnp.power(10.0, x)`` would promote to float64 under
    weak typing; ``exp(x·ln10)`` does not.
    """
    return jnp.exp(x * LN10)


def log10_four_pi_dl2(dl_cm):
    r"""``log10(4 pi d_L^2)`` [dex]; the luminosity-to-flux divisor, in log space.

    Parameters
    ----------
    dl_cm : array_like
        Luminosity distance :math:`d_L` [cm].

    Returns
    -------
    ndarray
        :math:`\log_{10}(4\pi d_L^2)` [dex]; ~57.0 at :math:`z=0.5`.

    Notes
    -----
    JIT/grad/vmap-safe.

    **The linear form has no safe distance.** :math:`4\pi d_L^2` is 1.1965e40 at
    the 10-pc :math:`z=0` convention and 8.12e58 at :math:`z=3`, against a float32
    ceiling of 3.4028e38; so it is ``inf`` at *every* distance, and a flux
    divided by it is ``nan``. Only the log form is representable, and only the
    *applied* result is meant to be: see :func:`apply_log10_scale`.

    Do not spell this ``10**log10_four_pi_dl2(...)`` to recover the linear value.
    That is a linear form wearing a log hat and is ``inf`` exactly as before
    (measured, #1859).

    See Also
    --------
    log10_flux_scale
        The same divisor with the :math:`(1+z)` k-correction folded in.
    """
    return LOG10_4PI + 2.0 * jnp.log10(jnp.asarray(dl_cm))


def log10_flux_scale(redshift, dl_cm):
    r"""``log10[(1+z) / (4 pi d_L^2)]`` [dex]; the cosmological dimming, in log space.

    The single spelling of a formula that was written longhand at twelve sites,
    seven correct and five not (#1859).

    .. math::

        \log_{10} \frac{1+z}{4\pi d_L^2}
        = \log_{10}(1+z) - \log_{10}(4\pi) - 2\log_{10} d_L

    where :math:`z` is redshift [dimensionless] and :math:`d_L` the luminosity
    distance [cm]; the result is [dex] relative to :math:`\mathrm{cm}^{-2}`.

    Parameters
    ----------
    redshift : array_like
        Redshift :math:`z` [dimensionless].
    dl_cm : array_like
        Luminosity distance :math:`d_L` [cm].

    Returns
    -------
    ndarray
        :math:`\log_{10}[(1+z)/(4\pi d_L^2)]` [dex]; ~-56.8 at :math:`z=0.5`.

    Notes
    -----
    JIT/grad/vmap-safe. Written in the association ``a - b - c`` rather than
    ``a - (b + c)`` so that migrating the sites which already spelled it out
    longhand is bit-exact, not merely close.

    **This is a scale to apply, never a value to hold.** The linear factor is
    1.4692e-57 at :math:`z=0.5` and 8.3577e-41 even at 10 pc, both below float32's
    smallest normal (1.1755e-38), so a standalone ``flux_scale`` scalar is exactly
    ``0.0`` in float32 at every distance. Pass the log offset to
    :func:`apply_log10_scale`; only the *net* product has to be representable,
    and it always is (~1e-29 for an ordinary galaxy).

    A stored table of such scales has the same problem one step later: built
    eagerly in float64 it is correct, and the cast to a float32 array zeroes
    every entry. Store the log (#1859).

    References
    ----------
    .. [1] Hogg, D. W. "Distance measures in cosmology." 1999,
       arXiv:astro-ph/9905116.

    See Also
    --------
    apply_log10_scale
        Applies this offset without materializing the factor.
    tengri.utils.conversions.lnu_to_fnu
        The immediate-application form, for callers holding the luminosity.
    """
    return jnp.log10(1.0 + jnp.asarray(redshift)) - LOG10_4PI - 2.0 * jnp.log10(jnp.asarray(dl_cm))


def _not_computable(log_value, sign=1.0):
    """True where a log-domain term is ``+inf``/``NaN`` rather than a usable value.

    ``-inf`` is excluded on purpose: it is the legitimate "this term is exactly
    zero" sentinel, not a failure.
    """
    bad_log = ~jnp.isfinite(log_value) & ~jnp.isneginf(log_value)
    return bad_log | ~jnp.isfinite(jnp.asarray(sign))


def log10_magnitude(value):
    r"""``log10|value|`` with "exactly zero" and "not computable" kept apart.

    The single spelling of the log-domain magnitude contract. Every producer of
    a ``log_*`` quantity re-derived this by hand, and four of them independently
    got the same half of it wrong (#1527).

    .. math::

        \mathrm{log10\_magnitude}(v) = \begin{cases}
            \log_{10}|v| & v \ne 0,\ v \ \mathrm{finite} \\
            -\infty      & v = 0 \\
            +\infty      & v \ \mathrm{non\text{-}finite}
        \end{cases}

    Parameters
    ----------
    value : array_like
        A signed or unsigned magnitude in linear space.

    Returns
    -------
    ndarray
        ``log10`` of the magnitude [dex], with the sentinels above.

    Notes
    -----
    JIT/grad/vmap-safe. The where-dummy keeps the backward pass free of NaN at
    ``value == 0``.

    **The two sentinels are not interchangeable, and that is the point.**
    ``-inf`` powers back through :func:`pow10` to exactly ``0.0``, so it is a
    *value*: "this quantity really is zero". ``+inf`` does not survive as a
    number and is not meant to; it says "no answer exists here". Folding a
    non-finite input into ``-inf`` reports a corrupt computation as a true zero,
    which is the failure mode that let one bad pixel silently switch off an
    entire dust IR budget or all nebular emission.

    ``+inf`` covers ``NaN`` as well as ``Inf`` deliberately: downstream there is
    nothing useful to do differently with the two, and one sentinel for "do not
    trust this" is far easier to test and to check for than two.

    See Also
    --------
    log10_add
        Sums two such quantities, preserving both sentinels.
    tengri.config.exceptions.CorruptEnergyBalanceWarning
        Attributes a ``+inf`` to the component that produced it.
    """
    value = jnp.asarray(value)
    magnitude = jnp.abs(value)
    finite = jnp.isfinite(value)
    positive = finite & (magnitude > 0)
    safe = jnp.where(positive, magnitude, 1.0)
    zero_or_log = jnp.where(positive, jnp.log10(safe), -jnp.inf)
    return jnp.where(finite, zero_or_log, jnp.inf)


def apply_log10_scale(arr, log10_scale):
    """Return ``arr * 10**log10_scale`` without out-of-range intermediates.

    Parameters
    ----------
    arr : array_like
        Values to scale (any magnitude within the dtype range).
    log10_scale : array_like, scalar
        Base-10 log of the multiplicative factor [dimensionless]. May be far
        outside the dtype range (e.g. -58); only the *net* result must be
        representable.

    Returns
    -------
    ndarray
        ``arr * 10**log10_scale``, equal to the naive product within ~1e-12 in
        float64 and finite in float32 when ``max|arr| * 10**log10_scale`` lies in
        the dtype's normal range.

    Notes
    -----
    JIT/grad/vmap-safe. Factors ``arr`` by its peak so the exponentiated scale
    is applied to an O(1) array; the peak's decades are folded into the exponent.

    The peak is held under ``stop_gradient`` (#1415). It is a pure factorization
    constant; ``(arr/p) * 10**(s + log10 p)`` equals ``arr * 10**s`` for *any*
    ``p``; so its derivative contributions cancel analytically. Left free, they
    are two separate autodiff paths (through the numerator, and through
    ``peak -> net -> pow10``) that must cancel numerically instead. They do in
    float64, but in float32 one side underflows while the other survives, the
    cancellation fails, and what is left is an uncancelled term the size of the
    main one; gradients exactly **2x** too large. Measured on a photometry fit:
    ``d(neg_log_posterior)/d(sfh_delayed_log_total_mass)`` was ``-5915.16``
    against a true ``-2957.58``. Stopping the gradient leaves the single correct
    path, ``d out/d arr = 10**log10_scale``, and float32 then tracks float64 to
    ~1e-7.

    Float64 **forward** values are untouched; ``stop_gradient`` is a no-op on the
    forward pass. Float64 **gradients** move by at most a few ulp: measured
    bit-identical where there is one scale seam (stellar, stellar+dust) and
    ``<= 1.5e-15`` relative where there are several (stellar+dust IR+AGN). That is
    the residue of a cancellation which was only ever exact to rounding, and it is
    three orders inside the ``rtol <= 1e-12`` no-behavioral-change bar for #1206.

    **Float32 reverse mode still underflows at large negative scales, and forward
    mode does not** (#1415, open as #1388). The remaining defect is a property of
    autodiff *mode*, not of this function. With ``log10_scale`` ~ -58 (the
    cosmological dimming), reverse mode has to form ``d out/d arr = 10**(-58)``
    explicitly; below float32's smallest subnormal (~1.4e-45); so the cotangent
    flushes to exactly zero, even when the gradient it was heading for (~1e-27) is
    perfectly representable. Forward mode carries the tangent instead, and because
    the tangent is divided by the same ``safe_peak`` as the primal it is O(1) when
    ``pow10(net)`` reaches it; measured correct to ~1e-6 in pure float32 where
    reverse mode returns 0.0.

    So: no local change here can fix reverse mode. The ratio follows from relating a
    ~1e30 quantity to a ~1e-28 one, which is what carrying the SED in scaled form
    (#1388) removes. In the meantime reverse mode is sound wherever the incoming
    cotangent is large enough to absorb the scale; notably the likelihood, whose
    ~1/sigma^2 keeps the product in range, which is why float32 inference works.
    """
    # initial=0.0 makes the peak of a zero-size array 0 (max over empty has no
    # identity and raises); the where() below then maps it to 1, so an empty
    # arr passes through as an empty result instead of a trace-time error.
    peak = jax.lax.stop_gradient(jnp.max(jnp.abs(arr), initial=0.0))
    usable = peak > 0
    safe_peak = jnp.where(usable, peak, jnp.ones_like(peak))
    # With no peak to fold in, the exponent would collapse to the raw
    # ``log10_scale``. That is fine in float64, but a dust-luminosity scale
    # (~43 dex) overflows float32, and ``0 * inf`` is NaN: so an all-zero
    # array would scale to NaN rather than to zero (#1206). Zeroing the
    # exponent keeps the identity ``0 * 10**s == 0`` at every scale, and costs
    # nothing when there is a peak.
    net = jnp.where(usable, log10_scale + jnp.log10(safe_peak), jnp.zeros_like(peak))
    return (arr / safe_peak) * pow10(net)


def log10_add(log_a, log_b, *, sign_a=1.0, sign_b=1.0):
    """Return ``log10|s_a·10**log_a + s_b·10**log_b|`` without leaving log space.

    A signed base-10 ``logaddexp``. Log-domain contracts (``log_nion``,
    ``log_L_ir``) are exact under multiplication but not under addition, so a
    seam that sums two such quantities would otherwise have to exponentiate
    both: reintroducing the very out-of-range intermediate the log form
    exists to avoid.

    Parameters
    ----------
    log_a, log_b : array_like
        Base-10 log magnitudes [dex]. ``-inf`` denotes an exactly zero term.
    sign_a, sign_b : array_like, optional
        Signs of the two terms (+1.0 or -1.0). Default +1.0. Cancellation
        between opposite signs is resolved at the precision of the larger
        term, as in any signed sum.

    Returns
    -------
    ndarray
        ``log10`` of the magnitude of the sum [dex]; ``-inf`` when the terms
        are both zero or cancel exactly.

    Notes
    -----
    JIT/grad/vmap-safe. Factors out the larger exponent so the exponentiated
    terms are O(1); the where-dummy keeps the backward pass free of NaN when
    the sum vanishes.
    """
    larger = jnp.maximum(log_a, log_b)
    finite = jnp.isfinite(larger)
    offset = jnp.where(finite, larger, 0.0)
    total = sign_a * pow10(log_a - offset) + sign_b * pow10(log_b - offset)
    magnitude = jnp.abs(total)
    positive = finite & (magnitude > 0)
    safe = jnp.where(positive, magnitude, 1.0)
    summed = jnp.where(positive, offset + jnp.log10(safe), -jnp.inf)
    # ``finite`` is False for BOTH infinities, but they mean opposite things:
    # -inf is "no term here", +inf is an overflow upstream. Folding the latter
    # into the -inf sentinel would report an overflowed term as exactly zero,
    # a fail-open on precisely the axis this module exists to close.
    #
    # This used to test ``isposinf(larger)`` alone, which caught +inf and missed
    # NaN entirely: ``maximum(43.0, nan)`` is NaN, ``isposinf(nan)`` is False,
    # and the result fell through to the -inf branch. A NaN term vanished as
    # though it were zero: the same fail-open, in the function whose comment
    # argues against it (#1527). Signs are checked too: a NaN sign with a finite
    # magnitude poisons ``total`` and lands in the same place.
    corrupt = _not_computable(log_a, sign_a) | _not_computable(log_b, sign_b)
    return jnp.where(corrupt, jnp.inf, summed)


def log10_weighted_sum(log_values, weights, axis=-1):
    r"""``log10(sum_i w_i * 10**log_i)``: a weighted sum without leaving log space.

    :func:`log10_add` for an arbitrary number of terms, with weights. The seam an
    *interpolator* over a log-domain table needs: linear interpolation and any
    kernel-weighted average are both weighted sums, and doing either in the linear
    domain would reintroduce the out-of-range factor the log form exists to avoid.

    .. math::

        \mathrm{log10\_weighted\_sum}(\ell, w)
        = \log_{10} \sum_i w_i \, 10^{\ell_i}

    where :math:`\ell_i` are log10 magnitudes [dex] and :math:`w_i` are
    non-negative weights [dimensionless].

    Parameters
    ----------
    log_values : array_like
        Base-10 log magnitudes [dex]. ``-inf`` denotes an exactly zero term.
    weights : array_like
        Non-negative weights, broadcastable against ``log_values``. A weight of
        exactly zero drops its term exactly, including when that term is ``-inf``.
    axis : int, optional
        Axis to reduce over. Default ``-1``.

    Returns
    -------
    ndarray
        ``log10`` of the weighted sum [dex]; ``-inf`` when every contributing
        term is zero.

    Notes
    -----
    JIT/grad/vmap-safe. Implemented as a base-10 ``logsumexp``, which factors out
    the largest exponent before summing, so no term is exponentiated at its own
    magnitude.

    **This is a re-spelling, not a re-model, and the distinction is the point.**
    Interpolating :math:`\log_{10} s` linearly is a *different function* from
    interpolating :math:`s` linearly; it is the geometric rather than the
    arithmetic mean at the midpoint. Migrating a stored table to log space with a
    naive ``lerp`` would therefore silently change float64 results. This
    reproduces the arithmetic weighted sum exactly (measured: rtol < 1e-12
    against the linear form it replaces), so the migration is invisible in
    float64 and merely finite in float32.

    Weights are assumed non-negative; a signed sum needs :func:`log10_add`'s
    ``sign_a`` / ``sign_b`` treatment, which resolves cancellation explicitly.

    See Also
    --------
    log10_add
        The two-term signed form.
    """
    from jax.scipy.special import logsumexp

    log_values = jnp.asarray(log_values)
    weights = jnp.asarray(weights)
    return logsumexp(log_values * LN10, b=weights, axis=axis) / LN10


# ``max_finite_exponent()`` lived here until 2026-07. It capped x at the
# dtype's own ``exp`` overflow (~87.7 in float32) for the two Planck-family
# closures that spelled the occupation number ``1/expm1(x)``. Both now spell it
# ``exp(-x) / -expm1(-x)``, whose denominator lies in (0, 1] and cannot
# overflow at any x in any dtype, so the cap became inert: measured identical
# values and gradients with and without it at x = 40, 60, 87, 90, 150, 400 in
# both dtypes. Removed rather than left as a no-op with a justification that no
# longer held (#1439). Restore it only alongside a caller that needs the raw
# ``expm1(x)`` form: and note the derivative needs ``sqrt`` of the limit.
