# SPDX-License-Identifier: BSD-3-Clause
"""Conservation: the per-Q_H nebular grid's dust channels reproduce the exact path.

Pins three facts a dust screen applied at runtime depends on:

* the sub-band channels sum to the whole-band channel (observed band and rest-band
  twin), so ``sum_k Phi_k T(lambda_k)`` factors the exact filter integral;
* every chunk's node lies inside its filter's support and the nodes ascend;
* the energy-balance channel equals the exact absorbed nebular luminosity of the
  materialized ``sed_nebular`` at every ``(tau_a, tau_b)`` node, and is exactly zero
  where the screen is unity.

Data-gated on the bare FSPS SSP and ``cue_weights.npz`` (skips where absent).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    load_ssp_data,
)
from tengri.components.nebular.component import NebularSEDComponent
from tengri.components.nebular.nebular_grid_precompute import (
    _nebular_screen_for,
    precompute_nebular_grid,
    reconstruct_nebular_eb_absorbed_per_qh,
    reconstruct_nebular_phot,
    reconstruct_nebular_phot_subband,
    reconstruct_nebular_restband,
    reconstruct_nebular_restband_subband,
    reconstruct_nebular_subband_waves,
)
from tengri.forward.energy_balance import bolometric_absorbed_log10
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.conservation

_BARE = "data/fsps_prsc_miles_chabrier.h5"
_BANDS = ["galex_nuv", "des_g", "des_r", "des_i", "des_z", "wise_w1"]
_K = 5
Z = 0.15
_LOG_NION = 50.0


def _require():
    for f in (_BARE, "data/cue_weights.npz"):
        if not Path(f).is_file():
            pytest.skip(f"missing {f}")


def _dusty_model(ssp, *, dust):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(_BANDS)),
        approx=WavePrecomp(),
        redshift=Fixed(Z),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(8, 12)},
        dust_attenuation=dust,
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT), "neb_logU": Uniform(-3.5, -2.0)},
    )


_TWO = {
    "type": "two_component",
    "law": "calzetti",
    "all_params": Fixed(DEFAULT),
    "tau_bc": Uniform(0.0, 2.0),
    "tau_diff": Uniform(0.0, 2.0),
}
_ONE = {
    "type": "single_component",
    "law": "calzetti",
    "all_params": Fixed(DEFAULT),
    "tau_v": Uniform(0.0, 2.0),
}


@pytest.fixture(scope="module")
def built():
    _require()
    ssp = load_ssp_data(_BARE)
    out = {}
    for name, dust in (("two", _TWO), ("one", _ONE)):
        m = _dusty_model(ssp, dust=dust)
        chain = m._build_component_chain()
        eb = m._energy_balance_lut(chain)
        assert eb is not None, "fixture must carry the stellar energy-balance LUT"
        dust_comp = next(c for c in chain if c.name in ("dust", "dust_attenuation"))
        table = precompute_nebular_grid(
            m,
            jnp.asarray([]),
            n_grid=5,
            dust_component=dust_comp,
            eb_tau_grids=(eb.tau_bc_grid, eb.tau_diff_grid),
            n_subbands=_K,
        )
        out[name] = (m, chain, eb, dust_comp, table)
    return out


def _resolved(m, key=0):
    p = dict(m.spec.get_fixed_values())
    p.update(m.spec.sample(jax.random.PRNGKey(key)))
    p["neb_dig_frac"] = 0.0
    return p


def _point(m, table, key=0):
    p = _resolved(m, key)
    return {k: p[k] for k in table.axis_names}


@pytest.mark.parametrize("which", ["two", "one"])
def test_subband_channels_sum_to_the_whole_band(built, which):
    m, _, _, _, table = built[which]
    assert table.serves_dust
    log_nion = jnp.asarray(_LOG_NION)
    # At a grid node the chunks partition the band exactly; between nodes each
    # channel is interpolated on its own, so the sum agrees to the interpolation error.
    node = {
        k: float(table.axes[i][len(table.axes[i]) // 2]) for i, k in enumerate(table.axis_names)
    }
    between = _point(m, table)
    for pt, rtol in ((node, 1e-8), (between, 5e-3)):
        parts = reconstruct_nebular_phot_subband(log_nion, pt, table)
        assert parts.shape == (len(_BANDS), _K)
        np.testing.assert_allclose(
            np.asarray(parts.sum(-1)),
            np.asarray(reconstruct_nebular_phot(log_nion, pt, table)),
            rtol=rtol,
        )
        rest_parts = reconstruct_nebular_restband_subband(log_nion, pt, table)
        assert rest_parts.shape == parts.shape
        np.testing.assert_allclose(
            np.asarray(rest_parts.sum(-1)),
            np.asarray(reconstruct_nebular_restband(log_nion, pt, table)),
            rtol=rtol,
        )


@pytest.mark.parametrize("which", ["two", "one"])
def test_subband_nodes_lie_inside_their_filters(built, which):
    m, chain, _, _, table = built[which]
    pt = _point(m, table)
    neb = next(c for c in chain if isinstance(c, NebularSEDComponent))
    lam_obs = np.asarray(reconstruct_nebular_subband_waves(pt, table, rest=False))
    lam_rest = np.asarray(reconstruct_nebular_subband_waves(pt, table, rest=True))
    assert lam_obs.shape == lam_rest.shape == (len(_BANDS), _K)
    for f, fw in enumerate(neb._state.filter_waves):
        lo, hi = float(np.min(fw)), float(np.max(fw))
        assert np.all(np.diff(lam_obs[f]) > 0) and np.all(np.diff(lam_rest[f]) > 0)
        assert np.all((lam_obs[f] * (1 + Z) >= lo) & (lam_obs[f] * (1 + Z) <= hi))
        assert np.all((lam_rest[f] >= lo) & (lam_rest[f] <= hi))


@pytest.mark.parametrize("which", ["two", "one"])
def test_energy_balance_channel_is_exact_at_every_tau_node(built, which):
    m, _, eb, dust_comp, table = built[which]
    tau_a, tau_b = np.asarray(eb.tau_bc_grid), np.asarray(eb.tau_diff_grid)
    n_ax = len(table.axis_names)
    grid_dims = tuple(len(a) for a in table.axes)
    assert table.eb_absorbed_per_qh.shape == (*grid_dims, tau_a.size, tau_b.size)
    np.testing.assert_array_equal(np.asarray(table.eb_tau_a_grid), tau_a)
    np.testing.assert_array_equal(np.asarray(table.eb_tau_b_grid), tau_b)
    # exact reference at the grid's first node: the materialized nebular SED
    p = _resolved(m)
    for i, name in enumerate(table.axis_names):
        p[name] = float(table.axes[i][0])
    # the attenuator re-publishes sed_nebular reddened: a unit screen gives the intrinsic one
    p.update({k: 0.0 for k in ("dust_tau_bc", "dust_tau_diff", "dust_tau_v") if k in p})
    state = m.predict_state(p, fixed_values={})
    sed_neb = jnp.asarray(state.derived["sed_nebular"])
    wave = state.wave
    nu = C_AA / wave
    log_nion = float(state.derived["log_nion"])
    cutoff = None if dust_comp.config.eb_include_lyc else 912.0
    got = np.asarray(table.eb_absorbed_per_qh)[(0,) * n_ax]
    assert got.min() >= 0.0 and got[-1, -1] > 0.0
    for a, ta in enumerate(tau_a):
        for b, tb in enumerate(tau_b):
            q = dict(p)
            if which == "two":
                q["dust_tau_bc"], q["dust_tau_diff"] = float(ta), float(tb)
            else:
                q["dust_tau_v"] = float(tb)
            t = _nebular_screen_for(dust_comp, q, wave)
            log_abs = float(
                bolometric_absorbed_log10(
                    sed_neb, sed_neb * t, nu, wave=wave, lyman_cutoff_aa=cutoff
                )[0]
            )
            want = 10.0 ** (log_abs - log_nion) if np.isfinite(log_abs) else 0.0
            np.testing.assert_allclose(got[a, b], want, rtol=1e-6, atol=1e-30)
    # a unit screen absorbs nothing
    assert got[0, 0] == 0.0
    # the reconstruction returns the stored node exactly
    node = {k: float(table.axes[i][0]) for i, k in enumerate(table.axis_names)}
    np.testing.assert_allclose(
        np.asarray(reconstruct_nebular_eb_absorbed_per_qh(node, table)), got, rtol=1e-12
    )


def test_without_dust_the_channels_stay_none(built):
    m, _, _, _, _ = built["two"]
    plain = precompute_nebular_grid(m, jnp.asarray([]), n_grid=5)
    assert not plain.serves_dust
    assert plain.log_phot_per_qh is not None
    assert plain.eb_absorbed_per_qh is None and plain.log_phot_subband_per_qh is None
