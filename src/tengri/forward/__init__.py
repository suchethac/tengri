"""Forward-model orchestration: SEDModel, pipeline, kernels, precompute.

Key class: ``SEDModel`` — the main forward model class.

Usage::

    from tengri import SEDModel  # canonical import path
    from tengri.forward.result import SEDResult  # lightweight type
"""

from tengri.forward.result import SEDResult

__all__ = ["SEDResult"]
