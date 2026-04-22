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
