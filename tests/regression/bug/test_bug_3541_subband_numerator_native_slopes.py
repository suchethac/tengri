# SPDX-License-Identifier: BSD-3-Clause
"""#3541: the sub-band wavelength node is the exact flux-weighted centroid off the Z nodes.

The node ``lambda_k = (sum w lambda phi) / (sum w phi)`` is a ratio. Before the fix the
numerator table was a product of two stored tables (``waves * phot``), and its native-slope
sibling was the product of the two slope tables, which is not the slope of the product. The
read then differed from the exact path (the PCHIP-read flux, then the same quadrature).

The reference is that exact path: the native SSP flux read by monotone PCHIP at the
off-node metallicity, collapsed to one metallicity node, and sub-band quadrature on that.
The node is per age (no SFH weight enters it), so the comparison is per age.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.components.stellar.sps.met_pchip import read_met
from tengri.components.stellar.sps.precompute import precompute_photometry
from tengri.observation.filters import load_filter_set
from tengri.parameters.translate import LOG10_ZSUN

pytestmark = pytest.mark.regression_bug

#: Relative tolerance of the table node against the exact path (round-off).
RTOL = 1e-12
PHOT_BANDS = ["galex_fuv", "sdss_u", "sdss_r"]
N_SUB = 8
_MET_LOGZSOL_OFF_NODE = 0.0  # absolute log Z -1.848, between the synthetic nodes -2 and -1
_Z = 0.1
_NODE_KEY = "stellar_subband_waves_rest_precomp"
_PHI_KEY = "stellar_phot_lnu_per_age_subband_precomp"


def _synthetic_ssp():
    rng = np.random.default_rng(3541)
    flux = np.abs(rng.normal(size=(3, 20, 100))) * 1e-3 + 1e-5
    # Shift the spectral shape with metallicity so the node moves across Z.
    flux = flux * (1.0 + np.array([0.0, 0.4, 1.1]))[:, None, None] * np.linspace(
        0.5, 2.0, 100
    )[None, None, :] ** np.array([0.0, 0.3, 0.8])[:, None, None]
    return SSPData(
        ssp_wave=jnp.linspace(3000.0, 10000.0, 100),
        ssp_flux=jnp.asarray(flux),
        ssp_lg_age_gyr=jnp.linspace(-1.0, 1.14, 20),
        ssp_lgmet=jnp.array([-2.0, -1.0, 0.0]),
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


def _build(ssp, *, redshift, approx):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(PHOT_BANDS)),
        approx=approx,
        redshift=redshift,
        sfh=_sfh(),
        met={"type": "delta", "met_logzsol": Fixed(_MET_LOGZSOL_OFF_NODE), "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
    )


def _exact_node(ssp, z):
    """Per-age sub-band node of the PCHIP-read flux at the off-node Z (the exact path)."""
    log_z_abs = _MET_LOGZSOL_OFF_NODE + LOG10_ZSUN
    flux_at_z = read_met(ssp.ssp_flux, ssp.ssp_lgmet, jnp.asarray(log_z_abs))[None]
    one_node = ssp._replace(ssp_flux=flux_at_z, ssp_lgmet=jnp.array([log_z_abs]))
    waves, trans, _ = load_filter_set(PHOT_BANDS)
    return precompute_photometry(
        ssp_data=one_node,
        filter_waves=waves,
        filter_trans=trans,
        redshift=z,
        dl_cm=1.0,
        taylor_correction=False,
        n_subbands=N_SUB,
        lyc_gate=False,
    )


def _max_rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def _live_node_pair(derived, ref):
    """(model node, exact node) on the sub-bands that carry weight.

    A sub-band with no weight (an age the SFH does not populate) takes the filter's
    effective wavelength as its node, by design; its node cannot change the photometry.
    The live mask is the model's own per-age sub-band weight.
    """
    got = np.asarray(derived[_NODE_KEY])
    live = np.abs(np.asarray(derived[_PHI_KEY])) > 0.0
    assert live.any() and live.sum() >= live.size // 2, "too few live sub-bands to test"
    return got[live], np.asarray(ref)[live]


def test_subband_node_matches_exact_at_off_node_z_fixed_z():
    ssp = _synthetic_ssp()
    model = _build(ssp, redshift=Fixed(_Z), approx=WavePrecomp(n_subbands=N_SUB))
    derived = model.predict_state({}).derived
    ref = np.asarray(_exact_node(ssp, _Z).ssp_subband_waves_rest)[0]
    got, ref = _live_node_pair(derived, ref)
    rel = _max_rel(got, ref)
    assert rel <= RTOL, f"fixed z: sub-band node vs exact path, max rel {rel:.3e}"


def test_subband_node_matches_exact_at_off_node_z_free_z():
    """Free z: exact at a z-table node (the z read is exact there), off-node in Z."""
    ssp = _synthetic_ssp()
    model = _build(ssp, redshift=Uniform(0.05, 0.15), approx=WavePrecomp(n_subbands=N_SUB))
    z_grid = np.asarray(_stellar_state(model).ssp_phot_ztable.z_grid)
    inside = z_grid[(z_grid > 0.05) & (z_grid < 0.15)]
    z_node = float(inside[len(inside) // 2])
    derived = model.predict_state({"redshift": jnp.asarray(z_node)}).derived
    ref = np.asarray(_exact_node(ssp, z_node).ssp_subband_waves_rest)[0]
    got, ref = _live_node_pair(derived, ref)
    rel = _max_rel(got, ref)
    assert rel <= RTOL, f"free z at node {z_node:.5f}: sub-band node vs exact path, max rel {rel:.3e}"


def _stellar_state(model):
    return next(
        c for c in model._build_component_chain() if type(c).__name__.startswith("Stellar")
    )._state
