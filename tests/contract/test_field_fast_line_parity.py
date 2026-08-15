# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the fast (window-LUT) line measurement matches the exact forward for a FIELD SFH.

The stochastic GP-field SFH modulates the SFR on the lookback grid; its only effect on a
line flux is a modulation of the (metallicity, age) SSP weights — exactly what the window-LUT
contraction consumes. So ``measure_line_fluxes(fast=True)`` must equal ``fast=False`` for a
field SFH, letting ``FeaturePrecomp`` serve the field instead of raising. This guards the
unified field-aware ``compute_joint_weights``.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

from tengri import (
    FREE,
    FeaturePrecomp,
    Fixed,
    NoiseModel,
    Observation,
    Photometry,
    SEDModel,
    builders,
    load_ssp_data,
)
from tengri.observation import LineFluxData
from tengri.observation.line_measurement import default_line_defs

_SSP = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)


@pytest.mark.skipif(not _SSP.exists(), reason="wNE SSP grid not available")
def test_field_fast_line_matches_exact():
    ssp = load_ssp_data(str(_SSP))
    lines = ["Halpha", "Hbeta", "OIII_5007"]
    lfd = LineFluxData.from_dict({nm: (1e-16, 1e-17) for nm in lines})
    obs = Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]),
        line_fluxes=lfd,
        noise=NoiseModel(),
    )
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": ["dpl", "field"], "*": FREE},
        met={"logzsol": Fixed(-0.3)},
        dust=builders.dust.two_component(defaults=FREE, law_bc="calzetti"),
        neb=builders.neb.ssp(),
        redshift=Fixed(0.1),
        apply_igm=False,
        n_grid=8,
        approx=FeaturePrecomp(),
    )
    ld = default_line_defs(np.asarray(lfd.wavelengths), tuple(lfd.names))

    # A star-forming field truth (fresh xi, moderate burstiness, modest dust).
    p = model.spec.sample(jax.random.PRNGKey(3))
    p = {
        **p,
        "sfh_dpl_alpha": jnp.array(2.0),
        "sfh_dpl_beta": jnp.array(1.5),
        "sfh_dpl_age_gyr": jnp.array(11.0),
        "sfh_dpl_tau_gyr": jnp.array(13.0),
        "sfh_field_psd_sigma": jnp.array(0.4),
        "sfh_field_psd_tau_myr": jnp.array(150.0),
        "met_logzsol": jnp.array(-0.3),
        "dust_tau_bc": jnp.array(0.3),
        "dust_tau_diff": jnp.array(0.15),
    }
    params = {**model.spec.get_fixed_values(), **p}

    fast = np.asarray(model.measure_line_fluxes(params, ld, fast=True))
    exact = np.asarray(model.measure_line_fluxes(params, ld, fast=False))
    assert np.all(np.isfinite(fast))
    # Same window-LUT-vs-exact agreement the non-field path already meets (#1152).
    np.testing.assert_allclose(fast, exact, rtol=3e-3, atol=0.0)
