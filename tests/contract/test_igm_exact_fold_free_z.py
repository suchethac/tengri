# SPDX-License-Identifier: BSD-3-Clause
"""The exact IGM fold on a free redshift, against the wavelength-grid integrator.

On a free redshift the sub-band tensor lives on a z-table and is interpolated to
the runtime redshift. The exact fold multiplies each node of that table by the
ratio of :mod:`tengri.components.igm.exact_fold`; the node fold multiplies by the
transmission at each sub-band's node wavelength. The two differ where Ly-alpha
sweeps across a sub-band, which on a z = 6.5-7.5 source is i, z and F090W.

Measured on the bare FSPS grid (dpl SFH, no nebular, inoue), worst over bands
still carrying more than 5 % of their flux, at z-table nodes and midpoints:

========  =========  ==========
``n_z``   node fold  exact fold
========  =========  ==========
8         81.8 %     9.9 %
16        86.6 %     1.6 %
32        104.8 %    0.42 %
========  =========  ==========

The exact fold converges with the z spacing; what is left is the triweight
z-interpolation shared by every z-table (which is why it is not zero at the
nodes either). The node fold does not converge: its error is the fold.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, SEDModel, WavePrecomp
from tengri.parameters import Fixed, Uniform

pytestmark = pytest.mark.contract

BANDS = ["sdss_i", "sdss_z", "JWST_NIRCam_F090W", "JWST_NIRCam_F115W"]
Z_RANGE = (6.5, 7.5)
N_Z = 32
#: Bands with less surviving flux than this are dark; their relative error is noise.
T_FLOOR = 0.05
#: Measured worst 0.42 % at N_Z = 32.
_EXACT_TOL = 0.01
#: Measured worst 104.8 % at N_Z = 32: the arms must be far apart for the test to mean anything.
_NODE_FLOOR = 0.5


@pytest.fixture(scope="module")
def ssp(ssp_data_fsps):
    return ssp_data_fsps


@pytest.fixture(scope="module")
def observation():
    return tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))


def _model(ssp, observation, *, igm="inoue", approx=None, redshift=None):
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Uniform(*Z_RANGE) if redshift is None else redshift,
        igm={"type": igm},
        approx=approx,
    )


def _photometry(model, z=None):
    params = {} if z is None else {"redshift": jnp.asarray(z)}
    return np.asarray(model.predict_photometry(params), dtype=np.float64)


def _ztable(model):
    return model._build_component_chain()[0]._state.ssp_phot_ztable


@pytest.fixture(scope="module")
def folds(ssp, observation):
    return {
        fold: _model(ssp, observation, approx=WavePrecomp(igm_fold=fold, n_z=N_Z))
        for fold in ("node", "exact", "auto")
    }


def test_the_exact_fold_tracks_the_integrator_across_the_z_table(ssp, observation, folds):
    reference = _model(ssp, observation)
    bare = _model(ssp, observation, igm="none")
    z_grid = np.asarray(_ztable(folds["exact"]).z_grid)
    nodes = z_grid[(z_grid > Z_RANGE[0]) & (z_grid < Z_RANGE[1])]
    probes = np.concatenate([nodes, 0.5 * (nodes[:-1] + nodes[1:])])

    worst = {"node": 0.0, "exact": 0.0}
    for z in probes:
        exact_flux = _photometry(reference, z)
        live = exact_flux / _photometry(bare, z) > T_FLOOR
        for fold in worst:
            error = np.abs(_photometry(folds[fold], z) / exact_flux - 1.0)
            worst[fold] = max(worst[fold], float(np.max(np.where(live, error, 0.0))))

    assert worst["exact"] < _EXACT_TOL, f"exact fold off the integrator by {worst['exact']:.3%}"
    assert worst["node"] > _NODE_FLOOR, (
        f"node fold off by only {worst['node']:.3%}: the probe no longer puts "
        "Ly-alpha inside a sub-band"
    )


def test_a_z_table_node_is_folded_as_a_fixed_redshift_would_be(ssp, observation, folds):
    """The free-z branch reads each node's own redshift and the table's partition."""
    table = _ztable(folds["exact"])
    k = int(np.argmin(np.abs(np.asarray(table.z_grid) - 7.0)))
    z_node = float(table.z_grid[k])
    fixed = (
        _model(ssp, observation, approx=WavePrecomp(igm_fold="exact"), redshift=Fixed(z_node))
        ._build_component_chain()[0]
        ._state.ssp_phot_lut
    )

    def _ratio(folded, bare):
        folded, bare = np.asarray(folded), np.asarray(bare)
        return np.where(bare != 0.0, folded / np.where(bare != 0.0, bare, 1.0), 0.0)

    np.testing.assert_allclose(
        _ratio(table.ssp_subband_phot_igm_table[k], table.ssp_subband_phot_table[k]),
        _ratio(fixed.ssp_subband_phot_igm, fixed.ssp_subband_phot),
        rtol=1e-12,
        atol=0,
    )


def test_auto_takes_the_exact_fold_on_a_free_redshift(folds):
    np.testing.assert_array_equal(
        np.asarray(_ztable(folds["auto"]).ssp_subband_phot_igm_table),
        np.asarray(_ztable(folds["exact"]).ssp_subband_phot_igm_table),
    )
