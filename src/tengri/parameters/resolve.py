# SPDX-License-Identifier: BSD-3-Clause
r"""Resolve a user parameter dict against a model's **Fixed** parameter values.

Fixed parameters are not required at predict time; that is what fixing them
*means*, so a user's dict legitimately omits them. The forward model resolves
them internally when it builds the SED, but anything that reads the **dict**
rather than the **state** does not. The exact projectors do exactly that: they
take the luminosity distance from ``params["redshift"]``.

That divergence is a silent physics error, and it is systemic rather than a
one-off: handing raw user params to an exact projector is the natural thing to
write, so the bug reappears in every new call site until the resolution is
shared. Use :func:`resolve_fixed_params` at any boundary where a user params
dict meets a projector.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

#: Failures that mean "this comparison cannot be resolved under trace", not
#: "the values differ". One entry per class rather than a scan of
#: ``jax.errors`` at import time (#2264: ``ConcretizationTypeError`` is not
#: the base of the ``Tracer*ConversionError`` family; a future jax release
#: reshuffling the hierarchy must not silently start swallowing an
#: unrelated error here).
_NOT_CONCRETE = (
    jax.errors.ConcretizationTypeError,
    jax.errors.TracerArrayConversionError,
    jax.errors.TracerBoolConversionError,
    jax.errors.TracerIntegerConversionError,
)


def _mirror_value_conflicts(spec, params):
    r"""Mirror targets present in ``params`` whose value differs from their
    tied source's resolved value.

    Returns a list of ``(target, source, supplied, resolved)`` tuples, empty
    when every present mirror target agrees with its source (or the
    comparison cannot be resolved under trace -- see Notes).

    Notes
    -----
    **Jit-safety.** Unlike the plain Fixed-key membership test, this compares
    *values*, which a traced pytree cannot always support (``bool(tracer)``
    raises rather than answering). A well-formed params dict reaching this
    point under ``jax.jit`` came from :meth:`Parameters.sample`'s own
    ``resolve_mirrors``, where target and source are equal by construction,
    so when the comparison cannot be resolved concretely this permissively
    treats it as equal (the pre-existing behavior) rather than raising a
    tracer error out of a caller that never asked for one.
    """
    mirrors = spec.mirrors
    conflicts = []
    for target in sorted(set(params) & set(mirrors)):
        source = mirrors[target]
        if source in params:
            resolved = params[source]
        else:
            resolved = spec.fixed_value(source)
        if resolved is None:
            continue  # cannot resolve the source's value; be permissive
        supplied = params[target]
        try:
            equal = bool(jnp.all(jnp.asarray(supplied) == jnp.asarray(resolved)))
        except _NOT_CONCRETE:
            continue
        if not equal:
            conflicts.append((target, source, supplied, resolved))
    return conflicts


def refuse_fixed_overrides(spec, params):
    r"""Refuse any params key the spec declared Fixed (#2296).

    Fixed parameters are not required at predict time; that is what fixing them
    *means*, so a user's dict legitimately omits them. However, a user **cannot**
    pass a key the spec declared Fixed at a different value: such an override is
    silently ignored on exact paths and honored on specialized paths (FeaturePrecomp),
    creating a silent physics error. This function raises loudly when a Fixed key
    is present, enforcing a single rule: rebuild the model if you want to change
    a Fixed parameter.

    The plain Fixed-key check is static and safe under ``jax.jit`` / ``jax.vmap``:
    a pure membership test on dict keys, with no value comparison. The mirror-
    target check below it does compare values (see :func:`_mirror_value_conflicts`
    for how it stays jit-safe).

    Parameters
    ----------
    spec : ParamSpec
        The model's ``spec``, carrying the fixed parameter declarations.
    params : dict
        User-supplied parameters.

    Raises
    ------
    ParameterError
        If any key in ``params`` is in ``spec.fixed_params`` and is not a
        mirror target (``spec.mirrors``); message names the key, the pinned
        value, and the remedy. Also raised if a mirror target IS present and
        its supplied value differs from its tied source's resolved value;
        message names the tie and the source to set instead.

    Notes
    -----
    **Mirrors are exempt, conditionally.** A mirror target (e.g.
    ``neb_logZ_gas="met_logzsol"``) is stored internally as ``Fixed(0.0)`` --
    a placeholder never meant to be read -- and :meth:`Parameters.resolve_mirrors`
    overwrites it with the tied source's actual sampled value. ``spec.sample()``
    calls ``resolve_mirrors`` before returning, so its output legitimately
    carries the target key at the CORRECT (tied) value, not the placeholder,
    and refusing that presence outright would make every mirrored spec's own
    ``sample()`` output unusable on any predict surface. But exempting
    presence unconditionally reopens the exact failure class #2296 closed
    for ordinary Fixed keys: a caller passing ``dust_slope=-99.0`` on a spec
    where ``dust_slope`` mirrors ``dust_delta`` (resolved value ``0.25``) had
    that ``-99.0`` **silently discarded** -- ``resolve_mirrors`` overwrites it
    with the source's value with no warning. The exemption therefore covers
    only a target whose supplied value equals the resolved source value (what
    ``sample()`` produces); a target present at a **different** value is
    refused, naming the tie and the source to set instead.
    """
    from tengri.config.exceptions import ParameterError

    offending = sorted((set(params) & set(spec.fixed_params)) - set(spec.mirrors))
    if offending:
        detail = "; ".join(f"{k!r} (pinned {spec.fixed_value(k)!r})" for k in offending)
        raise ParameterError(
            f"params overrides Fixed parameter(s): {detail}. "
            "Call-time overrides of a Fixed parameter are not supported (#2296); "
            "rebuild the model with this parameter FREE, or with a different "
            "Fixed value, instead."
        )

    conflicts = _mirror_value_conflicts(spec, params)
    if conflicts:
        detail = "; ".join(
            f"{target!r} mirrors {source!r} (supplied {supplied!r}, resolved {resolved!r})"
            for target, source, supplied, resolved in conflicts
        )
        raise ParameterError(
            f"params sets mirror target(s) at a value that differs from their "
            f"tied source's resolved value: {detail}. A mirror target's value "
            "is determined by its source, not by itself (#2296); set the "
            "source parameter instead."
        )


def merge_fixed_params(spec, params):
    r"""Refuse Fixed key overrides, then merge Fixed values into params.

    This is the one seam through which every caller's params dict meets the
    spec's fixed values. It refuses any Fixed key present (#2296), then fills in
    any omitted Fixed values, returning a complete dict ready for the forward model.

    Parameters
    ----------
    spec : ParamSpec
        The model's ``spec``, carrying the fixed parameter declarations.
    params : dict
        User-supplied parameters (free params only; see :func:`refuse_fixed_overrides`).

    Returns
    -------
    dict
        A **new** dict: ``params`` plus every numeric fixed value it omitted,
        with any mirror target overwritten by its tied source's value.

    Raises
    ------
    ParameterError
        If ``params`` contains a Fixed key (see :func:`refuse_fixed_overrides`);
        a mirror target is exempt (see that function's Notes).

    Notes
    -----
    A mirror target (``neb_logZ_gas="met_logzsol"``) is internally stored
    ``Fixed(0.0)`` -- a placeholder never meant to be read on its own -- so
    the fixed-value fill loop below fills it in at that placeholder like any
    other omitted Fixed name. :meth:`Parameters.resolve_mirrors` runs last,
    overwriting every mirror target with its tied source's actual value
    (already present in ``merged``, either from ``params`` if the source is
    free or from the fill loop above if the source is itself Fixed), so a
    caller of this function never sees the placeholder -- the same guarantee
    ``spec.sample()`` already gives its own free-only output.
    """
    refuse_fixed_overrides(spec, params)

    merged = dict(params)
    for name in spec.fixed_params:
        if name in merged:
            continue
        value = spec.fixed_value(name)
        if value is None or isinstance(value, (str, bool)):
            continue
        merged[name] = jnp.asarray(value)
    return spec.resolve_mirrors(merged)


def resolve_fixed_params(model, params):
    r"""Fill a user params dict with the model's **Fixed** parameter values.

    .. deprecated:: 1.0
        Use :func:`merge_fixed_params` with ``model.spec`` instead.
        This function is kept for compatibility only.

    Fixed parameters are not required at predict time; that is the whole point
    of fixing them, so a user's dict legitimately omits them. The forward model
    resolves them internally when it builds the SED, but anything that reads the
    *dict* rather than the *state* does not: notably the exact projectors, which
    take the luminosity distance from ``params["redshift"]``.

    That divergence is a silent physics error. With ``redshift=Fixed(0.5)`` and a
    params dict that (correctly) omits ``redshift``, ``project_photometry`` fell
    back to ``params.get("redshift", 0.0)``, computing the flux at 10 pc instead
    of at z = 0.5, ~16 orders of magnitude off, with no warning. The lean
    ``predict_photometry`` was right; ``pred.photometry()`` was wrong; nothing
    flagged the disagreement.

    Resolving once, here, closes it for every consumer of ``Prediction._params``
    at the same time (photometry, magnitudes, spectrum, obs_sed, the property
    catalog, and ``tengri.measure.from_prediction``) rather than patching each
    projector and waiting to discover the next one.

    Only **numeric** fixed values are injected. String-valued fixed settings
    (attenuation-law names and the like) are structural choices consumed at build
    time; they are not parameters and must never enter a dict that gets traced.

    Notes
    -----
    This function reads ``model.spec.fixed_params`` and ``spec.fixed_value``
    **directly**, with no ``getattr`` defaults and no blanket ``except``. That is
    deliberate. An earlier version fell back to an empty tuple if the attribute
    were missing, which meant a rename anywhere upstream would silently turn the
    resolver into a no-op and bring the 1e17 error back with no warning at all
    (#1127). A guard against a silent failure must not itself be able to fail
    silently: if the spec API moves, this raises ``AttributeError`` and someone
    fixes it.

    Parameters
    ----------
    model : SEDModel
        The model, whose ``spec`` carries the fixed values.
    params : dict
        User-supplied parameters (free params only; Fixed keys are refused).

    Returns
    -------
    dict
        A **new** dict: ``params`` plus every numeric fixed value it omitted.

    Raises
    ------
    AttributeError
        If ``model`` has no ``spec``, or the spec has no ``fixed_params`` /
        ``fixed_value``. Loud on purpose; see Notes.
    ParameterError
        If ``params`` contains a Fixed key (#2296).
    """
    spec = model.spec
    return merge_fixed_params(spec, params)


def require_redshift(params, where):
    r"""Read ``redshift`` from a params dict that is guaranteed to carry it.

    Parameters
    ----------
    params : dict
        Parameter dict. Must contain ``"redshift"``.
    where : str
        Caller identification for the error message, e.g.
        ``"observation.observation.project_photometry"``. Shown verbatim, so
        make it the thing a reader would grep for.

    Returns
    -------
    float or jnp.ndarray
        The redshift exactly as stored, no coercion, so a traced value stays
        traced and JIT/vmap are unaffected.

    Raises
    ------
    KeyError
        If ``redshift`` is absent, naming *where* and the boundaries that are
        supposed to guarantee it.

    Notes
    -----
    **JIT-compatible**: yes, a dict lookup on a static key, no tracing.

    Replaces ``params.get("redshift", 0.0)``. That idiom predates
    :func:`resolve_fixed_params` and is now a fossil: every dict reaching these
    call sites has already passed one of two boundaries that inject a ``Fixed``
    redshift:

    * :class:`~tengri.forward.prediction.Prediction`, which sets ``_params =
      resolve_fixed_params(model, params)``, and
    * the forward pipeline, which merges ``{**fixed_values, **params}`` before
      any component runs.

    So the ``0.0`` was unreachable; which is exactly why it was dangerous. A
    default that cannot be reached is not a safety net; it is a silencer for the
    one condition worth hearing about. Should a future caller bypass both
    boundaries, ``0.0`` places the galaxy at 10 pc and the flux is wrong by ~16
    orders of magnitude, silently. :func:`resolve_fixed_params` exists because
    precisely that shipped once. This raises instead.

    Verified before the conversion: with ``redshift=Fixed(0.5)`` and a params
    dict that correctly omits it, **no** site reached its default across
    ``predict_photometry``, ``predict``, ``photometry``, ``magnitudes``,
    ``spectrum``, ``obs_sed``, ``rest_sed``, the property catalog and
    ``measure.from_prediction``.

    Not every redshift lookup should use this. Two kinds legitimately have no
    key and keep an explicit fallback:

    * dicts from ``spec.get_fixed_values()``: ``redshift`` is absent whenever
      it is a *free* parameter, by construction;
    * caller-supplied precompute reference params, which want a documented
      reference redshift rather than an exception.
    """
    try:
        return params["redshift"]
    except KeyError:
        raise KeyError(
            f"{where}: 'redshift' is missing from the params dict.\n"
            "Every dict reaching here should already carry it: Prediction "
            "applies resolve_fixed_params(), and the forward pipeline merges "
            "{**fixed_values, **params} before components run.\n"
            "If you are calling this directly, pass a dict that includes "
            "'redshift' (resolve_fixed_params(model, params) fills a Fixed one "
            "in). Defaulting to 0.0 would put the galaxy at 10 pc, wrong by "
            "~16 orders of magnitude, with no warning."
        ) from None
