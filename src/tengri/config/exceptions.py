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
