# SPDX-License-Identifier: BSD-3-Clause
"""#3541: the IGM-folded sub-band tables read metallicity by the native-slope Hermite.

The exact IGM fold integrates the transmission inside each sub-band, so its tables are
linear in the native SSP flux. Read at an off-node metallicity they must equal the exact
path, which reads the native flux by PCHIP across Z first and folds the IGM afterwards.
The reference here is that exact path: the same model on a one-node grid whose flux is the
native PCHIP read at the galaxy's log Z. The three-node model must agree with it to round-off.

Controls: the IGM-free sub-band photometry, which the native-slope fix already covers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.parameters.translate import LOG10_ZSUN

pytestmark = pytest.mark.regression_bug

#: Relative tolerance of a table read against the exact read (round-off).
RTOL = 1e-12
PHOT_BANDS = ["galex_fuv", "sdss_u", "sdss_r"]
N_SUBBANDS = 3
MET_Z = LOG10_ZSUN  # met_logzsol = 0.0 is the solar node of the reference grid
KEYS = (
    "stellar_phot_lnu_per_age_subband_precomp",  # IGM-free control
    "stellar_phot_lnu_per_age_subband_igm_precomp",
    "stellar_subband_waves_rest_igm_precomp",
)


def _synthetic_ssp():
    """Three nodes that bracket MET_Z, so the query is off-node."""
    import jax

    return SSPData(
        ssp_wave=jnp.linspace(3000.0, 10000.0, 100),
        ssp_flux=jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (3, 20, 100))) * 1e-3 + 1e-5,
        ssp_lg_age_gyr=jnp.linspace(-1.0, 1.14, 20),
        ssp_lgmet=jnp.array([-2.0, -1.0, 0.0]),
    )


def _native_pchip_one_node(ssp):
    """The exact path's SSP: the native flux PCHIP-read across Z at MET_Z, as one node."""
    flux = PchipInterpolator(np.asarray(ssp.ssp_lgmet), np.asarray(ssp.ssp_flux), axis=0)(MET_Z)
    return SSPData(
        ssp_wave=ssp.ssp_wave,
        ssp_flux=jnp.asarray(flux[None]),
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        ssp_lgmet=jnp.array([MET_Z]),
    )


def _sfh():
    return {
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_frac": Fixed(0.6),
        "log_total_mass": Fixed(10.0),
        "all_params": Fixed(DEFAULT),
        "age_kernel": "cic",
    }


def _build(ssp, *, redshift):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(PHOT_BANDS)),
        approx=WavePrecomp(n_subbands=N_SUBBANDS, igm_fold="exact"),
        redshift=redshift,
        igm={"type": "inoue"},
        sfh=_sfh(),
        met={"type": "delta", "met_logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
    )


def _derived(model, params):
    derived = model.predict_state(params).derived
    return {k: np.asarray(derived[k], dtype=np.float64) for k in KEYS}


def _max_rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def _assert_matches_exact(lut, exact):
    # The IGM-free control must hold too; a failure there is not this bug.
    assert _max_rel(lut[KEYS[0]], exact[KEYS[0]]) <= RTOL, "IGM-free control"
    assert _max_rel(lut[KEYS[1]], exact[KEYS[1]]) <= RTOL, "IGM-folded sub-band photometry"
    assert _max_rel(lut[KEYS[2]], exact[KEYS[2]]) <= RTOL, "IGM-folded sub-band node"


def test_igm_subband_tables_match_exact_at_off_node_z_fixed_z():
    ssp = _synthetic_ssp()
    lut = _derived(_build(ssp, redshift=Fixed(0.1)), {})
    exact = _derived(_build(_native_pchip_one_node(ssp), redshift=Fixed(0.1)), {})
    _assert_matches_exact(lut, exact)


def test_igm_subband_tables_match_exact_at_off_node_z_free_z():
    """Free z: exact at a z-table node, where the z read is exact."""
    ssp = _synthetic_ssp()
    free = Uniform(0.05, 0.15)
    lut_model = _build(ssp, redshift=free)
    exact_model = _build(_native_pchip_one_node(ssp), redshift=free)
    z_grid = np.asarray(
        next(
            c for c in lut_model._build_component_chain() if type(c).__name__.startswith("Stellar")
        )._state.ssp_phot_ztable.z_grid
    )
    inside = z_grid[(z_grid > 0.05) & (z_grid < 0.15)]
    z_node = float(inside[len(inside) // 2])
    params = {"redshift": jnp.asarray(z_node)}
    _assert_matches_exact(_derived(lut_model, params), _derived(exact_model, params))
