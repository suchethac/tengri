# SPDX-License-Identifier: BSD-3-Clause
"""Dust attenuation application: two-component and single-screen transmission.

Applies the ``DUST_LAWS`` k(lambda) curves to build the Charlot & Fall (2000)
two-component (birth-cloud + diffuse) and single-screen attenuation, the
age-weight precompute, birth/diffuse law-param resolution, and the Lyman-cutoff
mask. Pure JAX; resolves curves by name from :mod:`..laws._registry` at call
time, so it never imports the law functions directly (no import cycle).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax.numpy as jnp

from tengri.components.dust._age_mixture import (
    interval_fractions,
    interval_optical_depth,
    interval_transmission,
    mix_intervals,
)
from tengri.components.dust._params import (
    DEFAULT_DUST_BUMP_STRENGTH,
    DEFAULT_DUST_DELTA,
    DEFAULT_DUST_F_OBSCURATION,
    DEFAULT_DUST_RV,
    DEFAULT_DUST_SLOPE,
)
from tengri.components.dust.laws._registry import (
    reject_unread_law_kwargs,
    resolve_dust_law,
    select_law_kwargs,
)

#: Two-component attenuation-law parameters that may be set per-component.
#: Maps the law-function keyword to ``(flat_param_name, default)``. The
#: per-component flat names are ``<flat_param_name>_bc`` / ``_diff``. Defaults
#: are read off ``ATTENUATION_PARAMS`` (``components/dust/_params.py``) rather
#: than repeated as bare literals here.
_TWO_COMPONENT_LAW_PARAMS: tuple[tuple[str, str, float], ...] = (
    ("dust_slope", "dust_slope", DEFAULT_DUST_SLOPE),
    ("dust_bump_strength", "dust_bump_strength", DEFAULT_DUST_BUMP_STRENGTH),
    ("dust_delta", "dust_delta", DEFAULT_DUST_DELTA),
    ("dust_Rv", "dust_Rv", DEFAULT_DUST_RV),
)

#: User-facing per-component short name -> attenuation-law kwarg. Used by the
#: builder grammar to accept ``slope_bc`` / ``slope_diff`` / ``delta_bc`` etc.
#: in the ``dust`` group and route them onto the per-component overrides.
TWO_COMPONENT_OVERRIDE_KEYS: dict[str, str] = {
    "slope": "dust_slope",
    "bump_strength": "dust_bump_strength",
    "delta": "dust_delta",
    "Rv": "dust_Rv",
}

#: Every per-screen spelling (``dust_slope_bc``, ``dust_Rv_neb``, ...) of a
#: tabled parameter. The "law-specific parameter" loops in
#: :func:`resolve_bc_diff_law_params` and :func:`merge_neb_screen_live_overrides`
#: below must exclude these, not just the bare stems in ``tabled`` -- a tabled
#: parameter's per-screen name is not itself law-specific, and forwarding it
#: verbatim re-emits a key no law declares (``law_kwarg_names`` never lists a
#: ``_bc``/``_diff``/``_neb`` suffix) alongside the correctly-resolved bare
#: stem the first loop already produced.
_TABLED_SCREEN_SPELLINGS: frozenset[str] = frozenset(
    f"{flat}_{screen}"
    for _, flat, _ in _TWO_COMPONENT_LAW_PARAMS
    for screen in ("bc", "diff", "neb")
)


def resolve_bc_diff_law_params(
    params: Mapping,
    bc_overrides: Mapping | None = None,
    diff_overrides: Mapping | None = None,
    live_shape_params: frozenset[str] | None = None,
    bc_law: str | None = None,
    diff_law: str | None = None,
    redshift: jnp.ndarray | float | None = None,
) -> tuple[dict, dict]:
    """Split shared dust law parameters into birth-cloud and diffuse law dicts.

    For each two-component law parameter (slope, bump, delta, Rv), the value is
    the shared ``dust_<x>`` from ``params``, unless a per-component override is
    supplied in ``bc_overrides`` / ``diff_overrides`` (keyed by law-function
    kwarg, e.g. ``dust_slope``). The overrides are the static per-component
    settings carried on :class:`DustSEDComponentConfig`. This is the single
    source of truth shared by every stellar two-component attenuation path, so
    they cannot diverge.

    A parameter nobody asked for is **omitted** rather than defaulted, so the
    selected law's own published default stands; see ``live_shape_params``.

    Parameters
    ----------
    params : Mapping
        Flat ``dust_*`` parameter mapping (JAX scalars or floats).
    bc_overrides, diff_overrides : Mapping, optional
        Per-component law-kwarg overrides (e.g. ``{"dust_slope": -1.0}`` for the
        FSPS birth-cloud convention). Honored whenever the corresponding
        per-screen *declared* name (``dust_slope_bc``/``_diff``) is not itself
        live in ``params`` -- see the priority order below.
    live_shape_params : frozenset of str, optional
        Flat names a caller actually asked for, resolved from spec provenance
        by :meth:`SEDModel._requested_law_shape_params` (#1808). Names outside
        this set are left out of the returned dicts. ``None`` keeps the
        historical behavior of offering all four, for the direct callers that
        have no spec to ask. Also gates the per-screen live lookup (#2428):
        ``f"{flat_name}_{screen}"`` (``dust_slope_bc``, ``dust_Rv_diff``, ...)
        is read straight out of ``params`` -- not ``bc_overrides``/
        ``diff_overrides`` -- when both present in ``params`` AND listed here.
    bc_law, diff_law : str, optional
        Registry keys of the two screens' laws. When given, each dict is
        narrowed to the keywords *that* screen's law declares, so a parameter
        the other screen's law reads is not offered to a law that would have to
        discard it (#2185). ``None`` leaves the dict unnarrowed.
    redshift : jnp.ndarray or float, optional
        The model redshift [dimensionless], offered to both screens before the
        narrowing above. ``None`` offers nothing, for the direct callers that
        have no model to ask.

    Returns
    -------
    bc_params, diff_params : dict
        Keyword dicts ready to splat into an attenuation-law function (keys are
        law-function kwargs, e.g. ``dust_slope``).

    Notes
    -----
    **JIT-compatible**: yes, only dict construction and ``Mapping.get``; the
    values pass through untouched (traced arrays stay traced).

    Passing all four unconditionally was #1833. The spec declares ONE shared
    ``dust_bump_strength`` / ``dust_delta``, both ``Fixed(0.0)``, while each law
    carries its paper's value in its own signature (``kriek_conroy``
    ``dust_bump_strength=1.0``; ``narayanan_z`` and ``tea``
    ``dust_delta=-0.2``). Injecting the shared zero deleted the 2175 Å Drude
    bump that Kriek & Conroy (2013) Eqn 3 exists to add, so ``two_component``
    silently returned a different law from the one selected; measured at 128%
    on the SED against ``single_component``, which had already been fixed by
    #1808. This is that fix reaching its second caller.

    ``bc_law`` / ``diff_law`` are #2185. Both screens were handed the *union* of
    the two laws' parameters and the laws absorbed the surplus in a ``**kwargs``
    catch-all, so ``law_bc='calzetti', law_diff='cardelli'`` offered calzetti an
    ``dust_Rv`` it fixes internally at 4.05. The laws no longer take that
    catch-all, so the narrowing has to happen here, where the two laws are both
    known.

    ``redshift`` is #2199, and it deliberately skips the ``live_shape_params``
    gate the four tabled parameters pass through. That gate protects each law's
    own *published default* for a parameter the ``dust_attenuation`` grammar can
    set; ``redshift`` is neither -- it is a model-wide value with no per-law
    default to protect, and the grammar never accepts it as a dust key. The
    narrowing below is what keeps it away from the laws that do not read it, and
    ``narayanan_z`` is the only one that does.

    Per-screen declared parameters (#2428) are the HIGHEST-priority source,
    ahead of both the static overrides and the shared spelling. For each
    tabled parameter and screen, the resolution order is: (1) the live
    per-screen name ``params[f"{flat_name}_{screen}"]``, read only when it is
    both present in ``params`` and listed in ``live_shape_params`` (a
    declared-but-untouched per-screen name still carries its Fixed registry
    default in ``params`` on some call paths, and that default must not
    shadow the static override or the shared value nobody asked to bypass);
    (2) the static ``bc_overrides``/``diff_overrides`` entry; (3) the shared
    ``dust_<x>`` value, if requested. An override *is* a request whatever the
    shared parameter's own provenance, but a *live* per-screen name is a
    stronger request still: it is explicit-only by design, so its mere
    presence in ``live_shape_params`` means a caller named it by hand.
    """
    bc_overrides = bc_overrides or {}
    diff_overrides = diff_overrides or {}
    bc: dict = {}
    diff: dict = {}
    # Process tabled parameters first (slope, bump_strength, delta, Rv)
    tabled = {flat for _, flat, _ in _TWO_COMPONENT_LAW_PARAMS}
    for law_kw, flat_name, default in _TWO_COMPONENT_LAW_PARAMS:
        requested = live_shape_params is None or flat_name in live_shape_params
        shared = params.get(flat_name, default) if requested else None
        for target, overrides, screen in (
            (bc, bc_overrides, "bc"),
            (diff, diff_overrides, "diff"),
        ):
            live_key = f"{flat_name}_{screen}"
            if (
                live_shape_params is not None
                and live_key in params
                and live_key in live_shape_params
            ):
                target[law_kw] = params[live_key]
            elif law_kw in overrides:
                target[law_kw] = overrides[law_kw]
            elif requested:
                target[law_kw] = shared
    # Process law-specific parameters (dust_c1-c4, dust_bump_x0/gamma, dust_tea_scatter)
    # that live_shape_params may include but _TWO_COMPONENT_LAW_PARAMS does not (#2542).
    # These parameters do NOT support per-screen spelling (only shared); a tabled
    # parameter's own per-screen spelling (dust_slope_bc, ...) is excluded here --
    # the loop above already resolved it onto the correct bare-stem law kwarg.
    if live_shape_params is not None:
        for flat_name in live_shape_params - tabled - _TABLED_SCREEN_SPELLINGS:
            if flat_name in params:
                # These are law-specific: map flat_name -> law_kw (usually identical)
                law_kw = flat_name
                # Only shared spelling supported; no per-screen variants like dust_c1_bc
                for target in (bc, diff):
                    target[law_kw] = params[flat_name]
    if redshift is not None:
        bc["redshift"] = redshift
        diff["redshift"] = redshift
    if bc_law is not None:
        bc = select_law_kwargs(bc_law, bc)
    if diff_law is not None:
        diff = select_law_kwargs(diff_law, diff)
    return bc, diff


def merge_neb_screen_live_overrides(
    params: Mapping,
    neb_overrides: Mapping,
    live_shape_params: frozenset[str] | None,
) -> dict:
    """Layer live ``*_neb`` per-screen params on top of static neb overrides.

    The nebular birth-cloud screen (:class:`DustSEDComponent`'s two merge
    call sites -- a third consumer reuses the already-merged dict rather
    than calling this again) merges its static ``neb_law_overrides`` with
    the *live* per-screen names (``dust_slope_neb``, ``dust_Rv_neb``, ...)
    the same way :func:`resolve_bc_diff_law_params` does for ``bc``/``diff``
    (#2428) -- a ``params``-dict lookup gated on ``live_shape_params``,
    taking priority over the static override.

    Parameters
    ----------
    params : Mapping
        Flat ``dust_*`` parameter mapping (JAX scalars or floats).
    neb_overrides : Mapping
        The static per-nebular-screen overrides
        (``DustSEDComponentConfig.neb_law_overrides``), keyed by law-function
        kwarg (e.g. ``dust_Rv``).
    live_shape_params : frozenset of str, optional
        Flat names a caller actually asked for, from
        :meth:`SEDModel._requested_law_shape_params`. ``None`` means no spec
        to ask (direct callers), in which case no live override applies and
        ``neb_overrides`` is returned as-is.

    Returns
    -------
    dict
        ``neb_overrides`` merged with any live ``*_neb`` overrides, keyed by
        the correct law-function kwarg (``dust_slope``, not ``slope`` --
        ``law_kwarg_names`` never declares the bare stem).

    Notes
    -----
    **JIT-compatible**: yes, only dict construction and ``Mapping``
    membership checks on a fixed set of string keys; the values pass through
    untouched (traced arrays stay traced).

    Both nebular call sites in ``two_component.py`` used to derive the
    law-function keyword as ``stem.replace("dust_", "")`` (e.g. ``"Rv"`` for
    ``dust_Rv``), which no law's signature declares
    (``law_kwarg_names('cardelli') == ('dust_Rv',)``), so
    ``select_law_kwargs`` silently discarded every live ``*_neb`` value --
    ``Rv_neb`` swept 2 -> 6 left the nebular lines bit-identical, and a
    ``Fixed`` ``Rv_neb`` silently equaled the law's own default (#2428). This
    helper is the single, tested place that keyword is derived, off the same
    :data:`_TWO_COMPONENT_LAW_PARAMS` table :func:`resolve_bc_diff_law_params`
    already uses for ``bc``/``diff``, so the two screens cannot drift again.
    """
    result = dict(neb_overrides)
    if live_shape_params is None:
        return result
    tabled = {flat for _, flat, _ in _TWO_COMPONENT_LAW_PARAMS}
    # Process tabled parameters first (slope, bump_strength, delta, Rv)
    for law_kw, flat_name, _default in _TWO_COMPONENT_LAW_PARAMS:
        live_key = f"{flat_name}_neb"
        if live_key in params and live_key in live_shape_params:
            result[law_kw] = params[live_key]
    # Process law-specific parameters (dust_c1-c4, dust_bump_x0/gamma, dust_tea_scatter) (#2542).
    # These parameters do NOT support per-screen spelling (only shared); a tabled
    # parameter's own per-screen spelling (dust_slope_neb, ...) is excluded here --
    # the loop above already resolved it onto the correct bare-stem law kwarg.
    for flat_name in live_shape_params - tabled - _TABLED_SCREEN_SPELLINGS:
        if flat_name in params:
            law_kw = flat_name
            # Only shared spelling supported; no per-screen variants like dust_c1_neb
            result[law_kw] = params[flat_name]
    return result


def apply_lyman_cutoff(
    k: jnp.ndarray, wavelength: jnp.ndarray, cutoff_aa: float = 0.0
) -> jnp.ndarray:
    r"""Zero an attenuation curve below a wavelength cutoff (Lyman-limit clip).

    Sets :math:`k(\lambda) = 0` for :math:`\lambda < \lambda_{\rm cut}`, leaving
    the curve untouched elsewhere. The standard choice is the hydrogen Lyman
    limit (912 Å): far-UV photons are absorbed by H ionization before reaching
    dust grains, so CIGALE's ``dustatt_modified_starburst`` zeros its curve
    there (``a_vs_ebv`` clips at 91.2 nm). tengri's ``calzetti`` / ``leitherer02``
    polynomials instead *extrapolate* through the FUV by default; this helper
    is the opt-in that reproduces the CIGALE behavior.

    Parameters
    ----------
    k : ndarray, shape (n_wave,)
        Attenuation curve :math:`k(\lambda) = A_\lambda / A_V`. [dimensionless]
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    cutoff_aa : float, optional
        Cutoff wavelength. [Å] Default ``0.0`` -> no-op (``wavelength >= 0`` is
        always true), so passing ``0.0`` disables the clip without a Python
        branch.

    Returns
    -------
    ndarray, shape (n_wave,)
        Curve with values below ``cutoff_aa`` set to zero.

    Notes
    -----
    **JIT-compatible**: yes, a single ``jnp.where``; ``cutoff_aa`` is a static
    Python float (never a traced parameter), so no ``TracerBoolConversionError``
    risk. **Gradient-safe**: yes, ``jnp.where`` on a static mask.
    """
    return jnp.where(wavelength >= cutoff_aa, k, 0.0)


def two_component_dust(
    wavelength: jnp.ndarray,
    younger_fraction: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    law_bc: str = "power_law",
    law_diff: str = "power_law",
    f_obscuration: float = DEFAULT_DUST_F_OBSCURATION,
    bc_params: dict | None = None,
    diff_params: dict | None = None,
    lyman_cutoff_aa: float = 0.0,
    **law_params,
) -> jnp.ndarray:
    r"""Two-component dust transmission of Charlot & Fall (2000), a mixture of two stellar populations.

    Separates dust into a birth-cloud screen (stars younger than the cloud's
    lifetime) and a diffuse-ISM screen (all stars), with independent optical
    depths and attenuation curves.  A stellar-population node holds a mixture
    of the two populations, and its transmission is the mass-weighted mixture
    of theirs:

    .. math::

        T(\lambda, a) = y(a)\,T_{\rm young}(\lambda) + [1 - y(a)]\,T_{\rm old}(\lambda)

    with :math:`y(a)` the fraction of node :math:`a`'s formed mass younger than the
    birth-cloud lifetime (the stellar component's ``age_boundary_younger_fraction``),

    .. math::

        T_{\rm young} &= f_{\rm obs} + (1 - f_{\rm obs})
            \exp[-\tau_{\rm V,BC} k_{\rm BC} - \tau_{\rm V,ISM} k_{\rm ISM}], \\
        T_{\rm old}   &= f_{\rm obs} + (1 - f_{\rm obs})
            \exp[-\tau_{\rm V,ISM} k_{\rm ISM}].

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    younger_fraction : array_like, shape (n_ages,)
        Per-node formed-mass fraction younger than the birth-cloud lifetime
        [dimensionless, in [0, 1]]: the stellar component publishes it for the
        attenuator's boundary (see
        :mod:`tengri.components.stellar.age_boundary`).
    tau_v1 : float
        Birth-cloud V-band optical depth (at 5500 Å). [dimensionless]
    tau_v2 : float
        Diffuse ISM V-band optical depth. [dimensionless]
    law_bc : str, optional
        Attenuation curve name for birth cloud. Default: "power_law". Resolved from ``DUST_LAWS`` registry.
    law_diff : str, optional
        Attenuation curve name for diffuse ISM. Default: "power_law".
    f_obscuration : float, optional
        Fraction of unattenuated sightlines in clumpy geometry (Lower 2022). [dimensionless, in [0, 1]]
        Default: 0.0 (uniform screen).
    bc_params : dict, optional
        Per-component overrides for the **birth-cloud** law (e.g.
        ``{"dust_slope": -1.0}``). Merged on top of ``**law_params``, so any key
        absent here falls back to the shared value. Enables FSPS-style
        independent indices (birth cloud ``dust1_index`` != diffuse
        ``dust_index``). Default ``None`` -> shared parameters.
    diff_params : dict, optional
        Per-component overrides for the **diffuse ISM** law. Same merge
        semantics as ``bc_params``. Default ``None`` -> shared parameters.
    lyman_cutoff_aa : float, optional
        Zero both attenuation curves below this wavelength. [Å] Default ``0.0``
        -> disabled (the polynomial extrapolates through the FUV). Set to
        ``912.0`` to match CIGALE's Lyman-limit clip (see
        :func:`apply_lyman_cutoff`).
    **law_params
        Shared keyword arguments passed to both attenuation curve functions
        (e.g., ``dust_slope``, ``dust_bump_strength``, ``dust_delta``,
        ``dust_Rv``). ``bc_params`` / ``diff_params`` override these
        per-component.

    Returns
    -------
    ndarray, shape (n_ages, n_wave)
        Multiplicative transmission factor T(λ, t_age), where T in [0, 1]. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives and safe for ``jax.jit``.

    **Gradient-safe**: yes, differentiable in the optical depths, the law
    parameters and ``younger_fraction``.

    Implements the same model as Charlot & Fall (2000) [1]_, whose
    birth-cloud/diffuse split is a split of the *stellar populations* by age;
    the nodes of a discrete SSP grid carry the fraction of each population,
    rather than a single age.  ``y(a)`` in ``{0, 1}`` (a node wholly inside one
    population) reduces to that population's own transmission.  The one
    implementation of the mixture is
    ``_age_mixture``, shared with ``age_binned``.

    References
    ----------
    .. [1] S. Charlot and S. M. Fall, "A Simple Model for the Absorption of Starlight by
       Dust in Galaxies," ApJ, 539, 718 (2000).
       https://doi.org/10.1086/309250

    .. [2] S. Lower et al., "How Well Can We Measure Galaxy Dust Attenuation Curves?
       The Impact of the Assumed Star-dust Geometry Model in SED Fitting,"
       ApJ, 931, 14 (2022). arXiv:2203.00074.
       https://doi.org/10.3847/1538-4357/ac6959

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import two_component_dust
    >>> wave = jnp.linspace(1000.0, 30000.0, 300)
    >>> ages = jnp.logspace(6.0, 10.14, 64)
    >>> young = (ages < 1e7).astype(float)
    >>> T = two_component_dust(wave, young, tau_v1=1.0, tau_v2=0.3)
    >>> T.shape
    (64, 300)
    """
    t_int = two_component_interval_transmission(
        wavelength,
        tau_v1,
        tau_v2,
        law_bc=law_bc,
        law_diff=law_diff,
        f_obscuration=f_obscuration,
        bc_params=bc_params,
        diff_params=diff_params,
        lyman_cutoff_aa=lyman_cutoff_aa,
        **law_params,
    )
    fractions = interval_fractions(jnp.asarray(younger_fraction)[None, :])
    return mix_intervals(fractions, t_int)


def two_component_interval_transmission(
    wavelength: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    law_bc: str = "power_law",
    law_diff: str = "power_law",
    f_obscuration: float = DEFAULT_DUST_F_OBSCURATION,
    bc_params: dict | None = None,
    diff_params: dict | None = None,
    lyman_cutoff_aa: float = 0.0,
    **law_params,
) -> jnp.ndarray:
    r"""The two populations' transmissions: young (birth cloud + diffuse) and old (diffuse).

    The curve evaluation behind :func:`two_component_dust`; every parameter
    has the meaning given there.

    Returns
    -------
    ndarray, shape (2, n_wave)
        ``T_young`` (row 0) and ``T_old`` (row 1) in ``[0, 1]``, each with its
        own ``f_obscuration`` sightline mixture [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes.  **Gradient-safe**: yes.
    """
    # Per-component law parameters: shared ``law_params`` with optional
    # ``bc_params`` / ``diff_params`` overlays. Each overlay only replaces the
    # keys it names, so callers can steepen the birth cloud (FSPS
    # ``dust1_index=-1.0``) without touching the diffuse ISM.
    bc_kw = {**law_params, **(bc_params or {})}
    diff_kw = {**law_params, **(diff_params or {})}
    # The two screens can carry different laws, so a key that belongs to one is
    # foreign to the other. Offer each law only what it declares -- and refuse a
    # key NEITHER declares, which used to vanish into the laws' `**kwargs`
    # (#2185).
    reject_unread_law_kwargs({**bc_kw, **diff_kw}, (law_bc, law_diff), "two_component_dust")
    k_bc = resolve_dust_law(law_bc)(wavelength, **select_law_kwargs(law_bc, bc_kw))
    k_diff = resolve_dust_law(law_diff)(wavelength, **select_law_kwargs(law_diff, diff_kw))
    # Optional Lyman-limit clip: zero the curve below ``lyman_cutoff_aa`` (CIGALE
    # parity). ``cutoff_aa=0.0`` is a no-op, so the default leaves the FUV
    # extrapolation in place.
    k_bc = apply_lyman_cutoff(k_bc, wavelength, lyman_cutoff_aa)
    k_diff = apply_lyman_cutoff(k_diff, wavelength, lyman_cutoff_aa)
    return nested_two_screen_intervals(tau_v1 * k_bc, tau_v2 * k_diff, f_obscuration)


def nested_two_screen_intervals(
    tau_k_bc: jnp.ndarray,
    tau_k_diff: jnp.ndarray,
    f_obscuration=DEFAULT_DUST_F_OBSCURATION,
) -> jnp.ndarray:
    """Per-interval transmissions of the nested N = 2 case: young (bc + diff), old (diff).

    Parameters
    ----------
    tau_k_bc, tau_k_diff : ndarray, shape (n_wave,)
        ``tau * k(lambda)`` of the birth-cloud and diffuse screens
        [dimensionless].
    f_obscuration : float or ndarray, optional
        Unattenuated-sightline fraction [dimensionless].

    Returns
    -------
    ndarray, shape (2, n_wave)
        ``T_young`` (row 0) and ``T_old`` (row 1), each including its own
        ``f_obscuration`` sightline mixture.
    """
    cover = ((True, False), (True, True))
    tau_int = interval_optical_depth([tau_k_bc, tau_k_diff], cover)
    return interval_transmission(tau_int, f_obscuration)


def nested_two_screen_mixture(
    younger_fraction: jnp.ndarray,
    tau_k_bc: jnp.ndarray,
    tau_k_diff: jnp.ndarray,
    f_obscuration=DEFAULT_DUST_F_OBSCURATION,
) -> jnp.ndarray:
    """The nested N = 2 case of the age-interval mixture: ``[0, t_b)`` bc + diff, then diff.

    Parameters
    ----------
    younger_fraction : ndarray, shape (n_age,)
        Node fraction younger than the boundary [dimensionless].
    tau_k_bc, tau_k_diff : ndarray, shape (n_wave,)
        ``tau * k(lambda)`` of the birth-cloud and diffuse screens
        [dimensionless].
    f_obscuration : float or ndarray, optional
        Unattenuated-sightline fraction [dimensionless].

    Returns
    -------
    ndarray, shape (n_age, n_wave)
        Per-node transmission.
    """
    fractions = interval_fractions(jnp.asarray(younger_fraction)[None, :])
    return mix_intervals(
        fractions, nested_two_screen_intervals(tau_k_bc, tau_k_diff, f_obscuration)
    )


def two_component_dust_separable(
    wavelength: jnp.ndarray,
    younger_fraction: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    law_bc_fn: Callable,
    law_diff_fn: Callable,
    f_obscuration: float = DEFAULT_DUST_F_OBSCURATION,
    **law_params,
) -> jnp.ndarray:
    r"""Two-component dust transmission with pre-resolved law functions.

    The same young/old population mixture as :func:`two_component_dust`
    (``T = y T_young + (1 - y) T_old``), taking already-resolved law callables
    to avoid registry lookups in hot code.  The two ``exp`` calls run on
    ``(n_wave,)``; only the mixture itself is ``(n_ages, n_wave)``.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    younger_fraction : array_like, shape (n_ages,)
        Per-node formed-mass fraction younger than the birth-cloud lifetime
        [dimensionless, in [0, 1]].
    tau_v1 : float
        Birth-cloud V-band optical depth. [dimensionless]
    tau_v2 : float
        Diffuse ISM V-band optical depth. [dimensionless]
    law_bc_fn : Callable
        Pre-resolved birth-cloud attenuation function (e.g., ``resolve_dust_law("calzetti")``).
    law_diff_fn : Callable
        Pre-resolved diffuse ISM attenuation function.
    f_obscuration : float, optional
        Unattenuated sightline fraction. [dimensionless, in [0, 1]] Default: 0.0.
    **law_params
        Keyword arguments passed to both law functions.

    Returns
    -------
    ndarray, shape (n_ages, n_wave)
        Multiplicative transmission T(λ, t_age) in [0, 1]. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes.  **Gradient-safe**: yes.

    Implements the same model as Charlot & Fall (2000) [1]_.

    References
    ----------
    .. [1] S. Charlot and S. M. Fall, "A Simple Model for the Absorption of Starlight by
       Dust in Galaxies," ApJ, 539, 718 (2000).
       https://doi.org/10.1086/309250
    """
    reject_unread_law_kwargs(law_params, (law_bc_fn, law_diff_fn), "two_component_dust_separable")
    k_bc = law_bc_fn(wavelength, **select_law_kwargs(law_bc_fn, law_params))
    k_diff = law_diff_fn(wavelength, **select_law_kwargs(law_diff_fn, law_params))
    return nested_two_screen_mixture(
        younger_fraction, tau_v1 * k_bc, tau_v2 * k_diff, f_obscuration
    )


def two_component_dust_fast(
    wavelengths: jnp.ndarray,
    younger_fraction: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    law_bc: str = "power_law",
    law_diff: str = "power_law",
    f_obscuration: float = DEFAULT_DUST_F_OBSCURATION,
    **law_params,
) -> jnp.ndarray:
    r"""Two-component dust transmission from a precomputed per-node young fraction.

    The same population mixture as :func:`two_component_dust`; used by the
    fused kernel (at effective wavelengths) and the exact path (at the full
    wavelength grid).

    The output dtype follows the input ``wavelengths`` dtype, so passing float32
    arrays halves memory traffic on the ``(n_ages, n_wave)`` intermediates
    (~1.6x speedup on CPU). That is a property of this function, not of the
    model: nothing hands it float32 wavelengths unless the whole run is in pure
    float32 (``jax.enable_x64(False)``). In particular
    ``forward_dtype="float32"`` does not; it casts nothing (#1433).

    Parameters
    ----------
    wavelengths : array_like, shape (n_wave,)
        Evaluation wavelengths (rest-frame). [Å] Can be the full
        SSP grid or just the filter effective wavelengths.
    younger_fraction : array_like, shape (n_ages,)
        Per-node formed-mass fraction younger than the birth-cloud lifetime
        [dimensionless, in [0, 1]].
    tau_v1 : float
        Birth-cloud V-band optical depth. [dimensionless]
    tau_v2 : float
        Diffuse ISM V-band optical depth. [dimensionless]
    law_bc : str
        Attenuation curve name for birth cloud. [dimensionless] Default: "power_law".
        Looked up in ``DUST_LAWS`` registry.
    law_diff : str
        Attenuation curve name for diffuse ISM. Default: "power_law".
    f_obscuration : float
        Fraction of unattenuated sightlines. [dimensionless, in [0, 1]] Default: 0.0 (Lower 2022).
    **law_params
        Passed to curve functions: ``dust_slope``, ``dust_bump_strength``,
        ``dust_delta``, ``dust_Rv``, etc.

    Returns
    -------
    ndarray, shape (n_ages, n_wave)
        Multiplicative attenuation factor in [0, 1]. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes.  **Gradient-safe**: yes.
    """
    reject_unread_law_kwargs(law_params, (law_bc, law_diff), "two_component_dust_fast")
    k_bc = resolve_dust_law(law_bc)(wavelengths, **select_law_kwargs(law_bc, law_params))
    k_diff = resolve_dust_law(law_diff)(wavelengths, **select_law_kwargs(law_diff, law_params))
    return nested_two_screen_mixture(
        younger_fraction, tau_v1 * k_bc, tau_v2 * k_diff, f_obscuration
    )


# ── Single-component dust model (uniform screen) ──────────────────


def single_component_dust(
    wavelength: jnp.ndarray,
    tau_v: float,
    law: str = "power_law",
    f_obscuration: float = DEFAULT_DUST_F_OBSCURATION,
    **law_params,
) -> jnp.ndarray:
    r"""Single-component (uniform foreground screen) dust attenuation.

    Applies a single attenuation curve at uniform optical depth to all stellar ages.
    Age-independent, enabling factorization out of stellar population integration.
    Simpler but less realistic than two-component models; useful for low-precision fits
    or high-redshift galaxies where birth-cloud/ISM distinction is unresolved.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    tau_v : float
        V-band optical depth at 5500 Å. [dimensionless]
    law : str, optional
        Attenuation curve name, resolved from ``DUST_LAWS`` registry. Default: "power_law".
    f_obscuration : float, optional
        Unattenuated sightline fraction in clumpy geometry (Lower 2022). [dimensionless, in [0, 1]]
        Default: 0.0 (uniform foreground screen).
    **law_params
        Keyword arguments passed to the attenuation curve function
        (e.g., ``dust_slope``, ``dust_bump_strength``, ``dust_delta``, ``dust_Rv``).

    Returns
    -------
    ndarray, shape (n_wave,)
        Multiplicative transmission T(λ) ∈ [0, 1]. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes, differentiable everywhere.

    The transmission is:

    .. math::

        T(\lambda) = f_{\rm obs} + (1 - f_{\rm obs}) \cdot \exp[-\tau_V \, k(\lambda)]

    where :math:`k(\lambda)` is the normalized attenuation curve with :math:`k(5500 \, \text{\AA}) = 1`,
    :math:`\tau_V` is the V-band optical depth, and :math:`f_{\rm obs}` is the fraction of
    unattenuated sightlines (Lower 2022; default 0 = full screen).

    **Age independence**: Unlike two-component models, there is no age-dependence, so this
    transmission can be factored out of the stellar population age integration, enabling
    faster computation.

    **Geometry**: When :math:`f_{\rm obs} = 0`, this recovers the standard Beer-Lambert
    foreground screen. When :math:`f_{\rm obs} > 0`, it models a clumpy geometry where
    a fraction of photons are unattenuated (Lower 2022).

    References
    ----------
    .. [1] S. Lower et al., "How Well Can We Measure Galaxy Dust Attenuation Curves?
       The Impact of the Assumed Star-dust Geometry Model in SED Fitting,"
       ApJ, 931, 14 (2022). arXiv:2203.00074.
       https://doi.org/10.3847/1538-4357/ac6959
    """
    reject_unread_law_kwargs(law_params, (law,), "single_component_dust")
    k = resolve_dust_law(law)(wavelength, **select_law_kwargs(law, law_params))
    return f_obscuration + (1.0 - f_obscuration) * jnp.exp(-tau_v * k)


def single_component_dust_fast(
    wavelengths: jnp.ndarray,
    n_ages: int,
    tau_v: float,
    law: str = "power_law",
    f_obscuration: float = DEFAULT_DUST_F_OBSCURATION,
    **law_params,
) -> jnp.ndarray:
    r"""Single-component dust attenuation broadcast to (n_ages, n_wave).

    Computes ``exp()`` on the 1-D wavelength grid only, then broadcasts
    to ``(n_ages, n_wave)`` via ``jnp.broadcast_to`` (zero-copy in XLA).
    This is the production path used by the SED pipeline.

    Parameters
    ----------
    wavelengths : array_like, shape (n_wave,)
        Evaluation wavelengths (rest-frame). [Å]
    n_ages : int
        Number of SSP age bins (for output shape). [dimensionless]
    tau_v : float
        V-band optical depth. [dimensionless]
    law : str
        Attenuation curve name (from ``DUST_LAWS`` registry). Default: "power_law".
    f_obscuration : float
        Fraction of unattenuated sightlines. [dimensionless, in [0, 1]] Default: 0.0 (Lower 2022).
    **law_params
        Passed to curve function.

    Returns
    -------
    ndarray, shape (n_ages, n_wave)
        Multiplicative transmission factor in [0, 1]. [dimensionless]
        All age rows are identical (age-independent attenuation).

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes, differentiable everywhere.

    **Memory efficiency**: Using ``jnp.broadcast_to`` avoids materializing
    the full (n_ages, n_wave) grid in memory; the result is a zero-copy view.
    """
    trans_1d = single_component_dust(
        wavelengths, tau_v=tau_v, law=law, f_obscuration=f_obscuration, **law_params
    )
    return jnp.broadcast_to(trans_1d[None, :], (n_ages, wavelengths.shape[0]))


# ── Witt & Gordon (2000) dust geometry transmission functions ─────
#
# These functions compute the wavelength-dependent transmission T(lambda)
# for different star-dust geometries, given a V-band optical depth tau_V
# and an underlying extinction curve k(lambda).
#
# The key insight from Witt & Gordon (2000, ApJ, 528, 799) is that the
# EFFECTIVE attenuation depends strongly on the spatial distribution of
# dust relative to stars.  A uniform foreground screen (SHELL) produces
# the steepest wavelength dependence; a homogeneous mix (CLOUDY) is
# grayer because high-tau sightlines are self-shielded; a clumpy medium
# (DUSTY) is grayest because photons preferentially escape through
# low-tau channels.
#
# All functions are pure JAX and JIT-compatible.
