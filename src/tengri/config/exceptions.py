# SPDX-License-Identifier: BSD-3-Clause
"""Exception hierarchy for Tengri (Naming Contract §8).

All Tengri exceptions inherit from ``TengriError``.  Domain-specific
exceptions also inherit from the closest stdlib type so that existing
``except ValueError`` / ``except OSError`` handlers still work.
"""


class TengriError(Exception):
    """Base exception for all Tengri errors.

    Parameters
    ----------
    message : str
        Human-readable error description.
    """


class ParameterError(TengriError, ValueError):
    """Invalid parameter names, values, or conflicts.

    Parameters
    ----------
    message : str
        Human-readable error description.
    """


class ParameterMapError(ParameterError):
    """Parameter map validation failure during SEDModel construction.

    Raised when:

    - A free parameter in spec has no entry in the parameter map
    - Multiple components claim conflicting (scale, offset) for the same parameter
    - Other parameter map consistency violations

    Parameters
    ----------
    message : str
        Human-readable error description.
    """


class UnknownParameterError(ParameterError):
    """Unknown parameter name passed to a ``predict_*`` method.

    Raised by :meth:`SEDModel.predict_photometry`, ``predict_spectrum``,
    ``predict_rest_sed``, ``predict_emission_lines``, etc., when the user
    passes an override key that doesn't match any free or fixed parameter
    of the assembled model (typo, deleted param, or param from a component
    the spec doesn't use).

    Silent no-ops on overrides are the worst class of bug — they make
    plausibly-correct downstream plots/fits encode stale defaults. We raise
    instead, and include a "did you mean…" suggestion built from the live
    param-map.

    Parameters
    ----------
    message : str
        Human-readable error description, listing unknown keys and
        suggested matches.
    """


class MissingParameterError(ParameterError):
    """A free parameter was given no value in a ``predict_*`` call.

    Every non-``Fixed`` parameter needs a value at predict time. Without
    this check the missing key surfaces deep inside a component as a bare
    ``KeyError`` (e.g. ``'dust_tau_bc'``) with no hint about the cause —
    commonly hit by ``model.mock({})`` / ``model.predict_photometry({})``
    on a model whose default dust group carries free optical depths.

    Parameters
    ----------
    message : str
        Human-readable error description listing the missing names and how
        to supply or fix them.
    """


class ParameterDefaultMissingError(ParameterError):
    """A parameter was marked FIXED without a physically-motivated default.

    Raised by :func:`tengri.parameters.groups.parse_groups` when the
    ``'all_params': FIXED`` wildcard (or an explicit short-form ``FIXED``) is applied
    to a registry entry whose ``Distribution`` carries no ``default=``. Prior
    behavior silently fell back to the midpoint of the prior support, which
    was an implicit and often physically wrong choice (e.g. ``Uniform(0, 5)``
    for ``log10(n_H/cm^-3)`` collapsed to 2.5 — 316 cm^-3 — when the
    CIGALE-faithful value is 2.0).

    Fix: register a default at the declaration site, e.g.
    ``Uniform(0, 5, default=2.0)``.

    Parameters
    ----------
    message : str
        Human-readable error description naming the offending parameter and
        suggesting where to add the default.
    """


class ConfigError(TengriError, ValueError):
    """Invalid Config construction or missing fields.

    Parameters
    ----------
    message : str
        Human-readable error description.
    """


class BackendError(TengriError, RuntimeError):
    """Backend initialization or computation failure.

    Parameters
    ----------
    message : str
        Human-readable error description.
    """


class InferenceError(TengriError, RuntimeError):
    """Sampler/optimizer failures (convergence, NaN, etc.).

    Parameters
    ----------
    message : str
        Human-readable error description.
    """


class TengriIOError(TengriError, OSError):
    """File I/O, missing data files, format mismatch.

    Parameters
    ----------
    message : str
        Human-readable error description.
    """


class AGNDustDoubleCountWarning(UserWarning):
    """Composable AGN and Dale2014 ``dust_frac_agn`` both inject AGN IR.

    The composable AGN (``agn_ir_frac > 0``) and Dale2014's embedded quasar
    template (``dust_frac_agn > 0``) are two distinct AGN surfaces, both keyed
    off the same stellar ``L_absorbed``. Using both with positive values
    double-counts AGN mid/far-IR (ADR-0018 §5, issue #721). Pick one surface;
    filter this category if the overlap is deliberate.
    """


class WildcardPartialFreeWarning(UserWarning):
    """``all_params: FREE`` freed only some of the parameters it covered.

    ``FREE`` resolves each parameter to its declared ``free_prior``. A parameter
    with no ``free_prior`` falls back to its ``prior`` — a ``Fixed`` scalar — and
    stays pinned. When a group holds both kinds, the wildcard frees one subset
    and silently leaves the rest frozen, and the fit reports a posterior with
    that physics held constant (issue #1474).

    A partial result is not always a defect: ``dust_Rv`` is fixed by definition
    for a Calzetti law, and ``dust_delta`` applies only to the Noll-modified
    variant. Whether a parameter *should* be freeable is a per-parameter physics
    question — so this warns rather than raising. Filter this category when the
    partial free is deliberate.

    See Also
    --------
    tengri.config.exceptions.ParameterError
        Raised instead when the wildcard frees *nothing*, which is never
        intended.
    """
