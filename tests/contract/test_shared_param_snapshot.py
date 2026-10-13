# SPDX-License-Identifier: BSD-3-Clause
"""Snapshot of the five globally shared parameters as the build path sees them.

``redshift``, ``sigma_v_kms``, ``lsf_scale``, ``line_flux_scaling`` (declared in
``tengri.parameters._shared``) and ``noise_frac_cal`` (declared in
``tengri.observation._params``) were once also described by a legacy 4-tuple
adapter with no ``free_prior``. This snapshot pins what the introspection
registry and the ``Parameters`` build path (``_build_param_registry``) report
for each name, so removing duplicate declarations cannot change behavior.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

from tengri.parameters._builders import _build_param_registry
from tengri.parameters.registry import describe_parameter

SHARED_NAMES = ("redshift", "sigma_v_kms", "lsf_scale", "line_flux_scaling", "noise_frac_cal")
_BOUND_PROBES = ((0.0, 1.0), (-1.0, 1.0), (0.5, 2.0), (1e-3, 5.0))

# Captured from the code before the legacy duplicates were removed.
EXPECTED = {
    "redshift": {
        "owner": "tengri.parameters._shared",
        "group": "PARAMS",
        "prior": "Fixed(0.1)",
        "free_prior": "Uniform(0.0, 20.0)",
        "default": "Fixed(0.1)",
        "bound_error": "must have lo >= 0",
        "probes": (True, False, True, True),
        "description": "Source redshift",
    },
    "sigma_v_kms": {
        "owner": "tengri.parameters._shared",
        "group": "PARAMS",
        "prior": "Fixed(0.0)",
        "free_prior": "Uniform(0.0, 2000.0)",
        "default": "Fixed(0.0)",
        "bound_error": "sigma_v_kms must be in [0, 2000]",
        "probes": (True, False, True, True),
        "description": (
            "Stellar velocity dispersion sigma_v [km/s], added in quadrature "
            "to the instrumental LSF when computing spectra"
        ),
    },
    "lsf_scale": {
        "owner": "tengri.parameters._shared",
        "group": "PARAMS",
        "prior": "Fixed(1.0)",
        "free_prior": "Uniform(0.8, 1.2)",
        "default": "Fixed(1.0)",
        "bound_error": "lsf_scale bounds must be strictly positive",
        "probes": (False, False, True, True),
        "description": (
            "Multiplicative scale on the instrument LSF sigma_inst(lambda), "
            "applied before quadrature combination with sigma_v and "
            "sigma_lib in the spectroscopy projection"
        ),
    },
    "line_flux_scaling": {
        "owner": "tengri.parameters._shared",
        "group": "PARAMS",
        "prior": "Fixed(1.0)",
        "free_prior": "LogNormal(mu=0.0, sigma=0.05)",
        "default": "Fixed(1.0)",
        "bound_error": "line_flux_scaling must be non-negative",
        "probes": (True, False, True, True),
        "description": (
            "Multiplicative calibration nuisance on the integrated line-flux data "
            "channel (Observation.line_fluxes), absorbing an aperture / absolute-"
            "flux-calibration mismatch between that channel and the rest of the "
            "SED (#2527). Multiplies every predicted line flux of that channel "
            "immediately before the likelihood comparison; does not change "
            "predict_line_fluxes, line ratios, or any other derived property."
        ),
    },
    "noise_frac_cal": {
        "owner": "tengri.observation._params",
        "group": "PARAMS",
        "prior": "Fixed(0.0)",
        "free_prior": "Uniform(0.0, 0.2)",
        "default": "Fixed(0.0)",
        "bound_error": "noise_frac_cal bounds must have lo >= 0",
        "probes": (True, False, True, True),
        "description": "Fractional calibration noise floor (added in quadrature with obs noise)",
    },
}


def _snapshot(name: str, build_registry: dict, build_defaults: dict) -> dict:
    rec = describe_parameter(name)
    _desc, check, bound_error = build_registry[name]
    return {
        "owner": rec.owner,
        "group": rec.group,
        "prior": repr(rec.prior),
        "free_prior": repr(rec.free_prior),
        "default": repr(build_defaults[name]),
        "bound_error": bound_error,
        "probes": tuple(bool(check(lo, hi)) for lo, hi in _BOUND_PROBES),
        "description": rec.description,
    }


@pytest.fixture(scope="module")
def build_path():
    return _build_param_registry("dpl")


@pytest.mark.parametrize("name", SHARED_NAMES)
def test_shared_param_matches_snapshot(name, build_path):
    registry, defaults = build_path
    assert _snapshot(name, registry, defaults) == EXPECTED[name]


@pytest.mark.parametrize("name", SHARED_NAMES)
def test_shared_param_is_not_owned_by_legacy_adapter(name):
    assert describe_parameter(name).owner != "tengri.parameters._builders"
