"""Parameter specification, priors, and name translation.

Key classes:
- ``Parameters`` — define priors and fixed values for all model parameters
- ``Uniform``, ``Gaussian``, ``LogNormal``, ``LogUniform``, ``Fixed``, ``StudentT`` — priors

Usage::

    from tengri.parameters import Uniform, Gaussian, Fixed
    from tengri import Parameters  # canonical import path for Parameters
"""

from tengri.parameters.priors import Fixed, Gaussian, LogNormal, LogUniform, StudentT, Uniform

__all__ = [
    "Fixed",
    "Gaussian",
    "LogNormal",
    "LogUniform",
    "StudentT",
    "Uniform",
]
