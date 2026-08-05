# SPDX-License-Identifier: BSD-3-Clause
r"""Resolve a user parameter dict against a model's **Fixed** parameter values.

Fixed parameters are not required at predict time — that is what fixing them
*means* — so a user's dict legitimately omits them. The forward model resolves
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

import jax.numpy as jnp


def resolve_fixed_params(model, params):
    r"""Fill a user params dict with the model's **Fixed** parameter values.

    Fixed parameters are not required at predict time — that is the whole point
    of fixing them — so a user's dict legitimately omits them. The forward model
    resolves them internally when it builds the SED, but anything that reads the
    *dict* rather than the *state* does not: notably the exact projectors, which
    take the luminosity distance from ``params["redshift"]``.

    That divergence is a silent physics error. With ``redshift=Fixed(0.5)`` and a
    params dict that (correctly) omits ``redshift``, ``project_photometry`` fell
    back to ``params.get("redshift", 0.0)`` — computing the flux at 10 pc instead
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
        User-supplied parameters (free params, and optionally some fixed ones).

    Returns
    -------
    dict
        A **new** dict: ``params`` plus every numeric fixed value it omitted.
        User-supplied values always win — an explicit override is never clobbered.

    Raises
    ------
    AttributeError
        If ``model`` has no ``spec``, or the spec has no ``fixed_params`` /
        ``fixed_value``. Loud on purpose — see Notes.
    """
    spec = model.spec

    resolved = dict(params)
    for name in spec.fixed_params:
        if name in resolved:
            continue  # the user passed it explicitly — never override
        value = spec.fixed_value(name)
        if value is None or isinstance(value, (str, bool)):
            # Structural choices (attenuation-law names, backend flags), not
            # parameters. They are consumed at build time and must never enter a
            # dict that gets traced.
            continue
        # Anything else is numeric. Note this deliberately does NOT gate on
        # ``isinstance(value, (int, float))``: a numpy float32 or a 0-d array is
        # not a Python float, and an allowlist of concrete types would silently
        # skip it — reintroducing the very drop this function exists to prevent.
        resolved[name] = jnp.asarray(value)
    return resolved


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
        The redshift exactly as stored — no coercion, so a traced value stays
        traced and JIT/vmap are unaffected.

    Raises
    ------
    KeyError
        If ``redshift`` is absent, naming *where* and the boundaries that are
        supposed to guarantee it.

    Notes
    -----
    **JIT-compatible**: yes — a dict lookup on a static key, no tracing.

    Replaces ``params.get("redshift", 0.0)``. That idiom predates
    :func:`resolve_fixed_params` and is now a fossil: every dict reaching these
    call sites has already passed one of two boundaries that inject a ``Fixed``
    redshift —

    * :class:`~tengri.forward.prediction.Prediction`, which sets ``_params =
      resolve_fixed_params(model, params)``, and
    * the forward pipeline, which merges ``{**fixed_values, **params}`` before
      any component runs.

    So the ``0.0`` was unreachable — which is exactly why it was dangerous. A
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

    * dicts from ``spec.get_fixed_values()`` — ``redshift`` is absent whenever
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
            "in). Defaulting to 0.0 would put the galaxy at 10 pc — wrong by "
            "~16 orders of magnitude, with no warning."
        ) from None
