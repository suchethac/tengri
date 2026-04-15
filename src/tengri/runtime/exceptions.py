"""Exception hierarchy for Tengri (Naming Contract §8).

All Tengri exceptions inherit from ``TengriError``.  Domain-specific
exceptions also inherit from the closest stdlib type so that existing
``except ValueError`` / ``except OSError`` handlers still work.
"""


class TengriError(Exception):
    """Base exception for all Tengri errors."""


class ParameterError(TengriError, ValueError):
    """Invalid parameter names, values, or conflicts."""


class ConfigError(TengriError, ValueError):
    """Invalid Config construction or missing fields."""


class BackendError(TengriError, RuntimeError):
    """Backend initialization or computation failure."""


class InferenceError(TengriError, RuntimeError):
    """Sampler/optimizer failures (convergence, NaN, etc.)."""


class TengriIOError(TengriError, OSError):
    """File I/O, missing data files, format mismatch."""
