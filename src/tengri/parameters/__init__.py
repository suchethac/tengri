# SPDX-License-Identifier: BSD-3-Clause
"""Parameter specification, priors, and name translation.

Key classes:

- ``Parameters``: define priors and fixed values for all model parameters
- ``Uniform``, ``Gaussian``, ``LogNormal``, ``LogUniform``, ``Fixed``, ``StudentT``: priors

Sentinels for nested-dict builder API:

- ``FREE``: use registry's default prior for a parameter
- ``DEFAULT``: legal only as ``Fixed(DEFAULT)``, an explicit per-parameter
  spelling of "pin at the registry default"

Usage::

    from tengri.parameters import Uniform, Gaussian, Fixed, FREE, DEFAULT
    from tengri import Parameters  # canonical import path for Parameters

Where parameter declarations live
---------------------------------

Each physics component owns its own parameter declarations. To add or edit
a parameter, open its component's ``_params.py``::

    components/agn/_params.py: agn_log_lbol, agn_lum_ratio, agn_a_spin, ...
    components/dust/_params.py: dust_tau_v, dust_beta_ir, ...
    components/stellar/_params.py: sfh_*, met_*, alpha_fe_*
    components/nebular/_params.py: neb_logU, neb_logZ_gas, ...
    components/igm/_params.py: igm_x_HI, ...
    components/radio/_params.py: radio_*, ...
    components/xray/_params.py: xray_*, ...
    observation/_params.py: noise_frac_cal, noise_dof
    parameters/_shared.py: redshift, met_logzsol, sigma_v_kms

``tengri.parameters.registry`` walks all of these and provides one queryable
view via ``tengri.list_parameters()`` and ``tengri.describe_parameter()``.
"""

from tengri.parameters.groups import parse_groups
from tengri.parameters.priors import Fixed, Gaussian, LogNormal, LogUniform, StudentT, Uniform
from tengri.parameters.sentinels import DEFAULT, FREE

__all__ = [
    "DEFAULT",
    "FREE",
    "Fixed",
    "Gaussian",
    "LogNormal",
    "LogUniform",
    "StudentT",
    "Uniform",
    "parse_groups",
]
