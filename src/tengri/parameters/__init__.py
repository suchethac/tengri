"""Parameter specification, priors, and name translation.

Key classes:
- ``Parameters`` — define priors and fixed values for all model parameters
- ``Uniform``, ``Gaussian``, ``LogNormal``, ``LogUniform``, ``Fixed``, ``StudentT`` — priors

Sentinels for nested-dict builder API:
- ``FREE`` — use registry's default prior for a parameter
- ``FIXED`` — pin parameter to registry's default value

Usage::

    from tengri.parameters import Uniform, Gaussian, Fixed, FREE, FIXED
    from tengri import Parameters  # canonical import path for Parameters
"""

from tengri.parameters.groups import parse_groups
from tengri.parameters.priors import Fixed, Gaussian, LogNormal, LogUniform, StudentT, Uniform
from tengri.parameters.sentinels import FIXED, FREE

__all__ = [
    "FIXED",
    "FREE",
    "Fixed",
    "Gaussian",
    "LogNormal",
    "LogUniform",
    "StudentT",
    "Uniform",
    "parse_groups",
]
