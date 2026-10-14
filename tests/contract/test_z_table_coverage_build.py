# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end: a narrow explicit z table warns at SEDModel build; the default table does not.

The helper that emits :class:`ZTableCoverageWarning` is unit-tested directly
elsewhere. This test builds a real model, so it catches a broken call site in
``SEDModel``: if the warning is not wired into the build, the narrow case fails.
"""

from __future__ import annotations

import warnings

import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.config.exceptions import ZTableCoverageWarning
from tengri.forward.sed_model import WavePrecomp

pytestmark = pytest.mark.contract


def _build(ssp_data_wne, obs, approx):
    return SEDModel.build(
        ssp_data=ssp_data_wne,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(0.5),
            "tau_diff": Fixed(0.3),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={"type": "ssp"},
        redshift=Uniform(0.0, 3.0),
        approx=approx,
    )


def test_narrow_explicit_z_table_warns_at_build(ssp_data_wne, synthetic_tophat_obs):
    """WavePrecomp(z_min=0.5, z_max=1.0) under a Uniform(0, 3) redshift prior warns."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build(ssp_data_wne, synthetic_tophat_obs, WavePrecomp(z_min=0.5, z_max=1.0, n_z=40))
    coverage = [w for w in caught if issubclass(w.category, ZTableCoverageWarning)]
    assert coverage, "a narrow z table under a wider prior must warn at build"
    message = str(coverage[0].message)
    assert "[0, 3]" in message and "[0.5, 1]" in message, message


def test_default_z_table_is_silent_under_werror(ssp_data_wne, synthetic_tophat_obs):
    """The default table is built from the prior, so the prior builds with no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZTableCoverageWarning)
        _build(ssp_data_wne, synthetic_tophat_obs, WavePrecomp(n_z=40))
