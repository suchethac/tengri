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
    """
    spec = getattr(model, "spec", None)
    if spec is None:
        return dict(params)

    resolved = dict(params)
    for name in getattr(spec, "fixed_params", ()):
        if name in resolved:
            continue  # the user passed it explicitly — never override
        try:
            value = spec.fixed_value(name)
        except (KeyError, ValueError):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            resolved[name] = jnp.asarray(value)
    return resolved
