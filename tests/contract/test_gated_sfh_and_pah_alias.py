# SPDX-License-Identifier: BSD-3-Clause
"""Contract: advertised SFH/dust-emission types match what actually forward-models.

Two wiring-consistency guards (audit follow-up):

1. SFH types that are registered but NOT yet validated against the DSPS forward
   path (``_UNVALIDATED_SFH_TYPES``) must NOT be advertised by the grammar — a
   build with one raises a clear error instead of succeeding and then raising
   ``NotImplementedError`` at predict time (advertised-but-unusable footgun).

2. ``dust.emission='draine2021_pah'`` (a deprecated name the validator advertised
   but the resolver never registered, #693) now resolves to ``pah_drude``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel

pytestmark = pytest.mark.contract

_GATED_SFH = ["bursty_continuity", "gaussian_burst", "prospector_beta", "psb_wild2020", "top_hat"]


def test_unvalidated_sfh_not_advertised():
    """The gated SFHs are still registered but excluded from the public type set."""
    from tengri.components.stellar.sfh.registry import (
        SFH_REGISTRY,
        UNVALIDATED_SFH_TYPES,
    )
    from tengri.parameters.groups import _valid_sfh_types

    valid = _valid_sfh_types()
    for name in _GATED_SFH:
        assert name in SFH_REGISTRY, f"{name} should remain registered"
        assert name in UNVALIDATED_SFH_TYPES
        assert name not in valid, f"{name} must not be advertised as a usable sfh.type"


@pytest.mark.parametrize("sfh_type", _GATED_SFH)
def test_gated_sfh_raises_clear_error_at_build(synthetic_ssp_wide, sfh_type):
    """Building with a gated SFH raises a clear 'not yet validated' error at build,
    not a NotImplementedError surprise at predict time."""
    with pytest.raises(ValueError, match=r"not yet validated|not available|Unknown SFH"):
        SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": sfh_type, "all_params": FIXED, "log_total_mass": 10.0},
            redshift=Fixed(0.05),
        )


def test_draine2021_pah_resolves_to_pah_drude():
    """The deprecated 'draine2021_pah' emission name resolves (to pah_drude)."""
    from tengri.components.dust.emission import DUST_EMISSION_MODELS

    assert "draine2021_pah" in DUST_EMISSION_MODELS
    assert DUST_EMISSION_MODELS["draine2021_pah"] is DUST_EMISSION_MODELS["pah_drude"]


def test_draine2021_pah_builds_and_emits(synthetic_ssp_wide):
    """dust.emission='draine2021_pah' builds and produces finite far-IR re-emission."""
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        sfh={"type": "delayed", "all_params": FIXED, "log_total_mass": 10.0},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": FIXED,
            "tau_diff": Fixed(1.5),
            "tau_bc": Fixed(0.0),
        },
        dust_emission={"type": "draine2021_pah", "all_params": FIXED},
        redshift=Fixed(0.05),
    )
    state = model.predict_state({})
    sed = np.asarray(state.sed_intrinsic)
    assert np.all(np.isfinite(sed))
