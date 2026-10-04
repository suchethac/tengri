# SPDX-License-Identifier: BSD-3-Clause
"""The WavePrecomp energy-balance LUT follows the exact path for the hard and the smooth split.

The LUT carries a young and an old population family and mixes them at run
time with each node's young fraction, so the same table serves a hard step
and a smooth dispersal. The far-IR band is dominated by the re-emitted dust
luminosity and so measures the LUT's ``L_absorbed`` directly; its drift
against the exact path must stay inside the 2% the single-width LUT held.
"""

import jax
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, WavePrecomp, builders
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.conservation


def _obs():
    import jax.numpy as jnp

    def tophat(center, frac=0.16, n=40):
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    centers = (3500.0, 4800.0, 6200.0, 9000.0, 1.0e6)
    return Observation(photometry=Photometry(filters=tuple(tophat(c) for c in centers)))


def _build(ssp, width, reprocessed_by, approx):
    return SEDModel.build(
        ssp_data=ssp,
        observation=_obs(),
        approx=approx,
        sfh=builders.sfh.tsnorm(
            all_params=Fixed(DEFAULT),
            log_total_mass=9.75,
            peak_lbt_gyr=0.05,
            width_gyr=0.03,
            skew=0.0,
            trunc=5.5,
        ),
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": 0.8,
            "tau_diff": 0.3,
            "transition_width_dex": width,
            "lyc_reprocessed_by": reprocessed_by,
        },
        dust_emission={"type": "modified_blackbody", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Fixed(0.05),
    )


@pytest.mark.parametrize("reprocessed_by", ["young", "all"])
@pytest.mark.parametrize("width", [0.0, 0.3])
def test_lut_far_ir_tracks_exact(synthetic_ssp_wide, width, reprocessed_by):
    exact = _build(synthetic_ssp_wide, width, reprocessed_by, None)
    lut = _build(synthetic_ssp_wide, width, reprocessed_by, WavePrecomp())
    p_exact = dict(exact.spec.sample(jax.random.PRNGKey(0)))
    p_lut = dict(lut.spec.sample(jax.random.PRNGKey(0)))
    phot_exact = np.asarray(exact.predict_photometry(p_exact))
    phot_lut = np.asarray(lut.predict_photometry(p_lut))
    drift = abs(phot_lut[-1] - phot_exact[-1]) / abs(phot_exact[-1])
    assert drift < 0.02, f"far-IR LUT-vs-exact drift {drift:.3%} (> 2%)"
