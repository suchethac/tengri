# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the fast nebular path engages on dusty models whose dust takes the
nebular continuum from the per-Q_H grid.

Pinned:
  * ``Fitter(..., approx="auto")`` on a dusty Cue model attaches a grid that
    serves the dust screen, flags the dust component ``nebular_from_grid``, leaves
    the nebular component free to zero ``sed_nebular`` (``must_materialize_sed``
    False), and ``fast_nebular_can_engage`` reports True;
  * the compiled gradient of that model costs at least 5x fewer FLOPs than the
    ``WavePrecomp``-only model. Exact equality is the signature of a config that
    never reaches the graph (#1748);
  * the full-state chain (``predict_state``) keeps the dust unflagged and
    publishes a non-zero ``sed_nebular``;
  * a table built without dust channels leaves the dust unflagged and the
    nebular continuum materialized;
  * a shape-free attenuation model has no energy-balance LUT, so the dust still
    counts as a continuum consumer and the shortcut stays disarmed.

Needs a bare SSP and ``data/cue_weights.npz``; every test skips when they are
absent.
"""

from __future__ import annotations

import warnings
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
from tengri.components.dust.component import DustAttenuationSEDComponent
from tengri.components.dust.two_component import DustSEDComponent
from tengri.components.nebular.component import NebularSEDComponent
from tengri.components.nebular.nebular_grid_precompute import precompute_nebular_grid
from tengri.forward.sed_model import _nebular_continuum_consumers
from tengri.inference import Fitter
from tengri.inference.fitter import fast_nebular_can_engage

pytestmark = pytest.mark.contract

_BARE = "data/fsps_prsc_miles_chabrier.h5"
_BANDS = ["galex_nuv", "des_g", "des_r", "des_i", "des_z", "wise_w1"]
_DUST_TYPES = (DustSEDComponent, DustAttenuationSEDComponent)
Z = 0.15

_TWO = {
    "type": "two_component",
    "law": "calzetti",
    "all_params": Fixed(DEFAULT),
    "tau_bc": Uniform(0.0, 2.0),
    "tau_diff": Uniform(0.0, 2.0),
}


def _require():
    for f in (_BARE, "data/cue_weights.npz"):
        if not Path(f).is_file():
            pytest.skip(f"missing {f}")


def _dusty_model(ssp, dust=_TWO):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


@pytest.fixture(scope="module")
def ssp():
    _require()
    return load_ssp_data(_BARE)


@pytest.fixture(scope="module")
def data():
    fnu = np.array([2.0e-6, 4.0e-6, 6.0e-6, 8.0e-6, 1.0e-5, 1.5e-5])
    return fnu, 0.1 * fnu


def _auto_fit(model, data):
    fnu, sigma = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Fitter(model, data=fnu, noise=sigma, data_type="photometry", approx="auto")


def _find(chain, types):
    return next(c for c in chain if isinstance(c, types))


def _params(model):
    values = {k: 0.5 for k in model.spec.free_params}
    values["sfh_dpl_log_total_mass"] = 10.0
    values["neb_logU"] = -2.7
    return values


def _grad_flops(model):
    p = _params(model)

    def loss(v):
        return jnp.sum(model.predict_photometry({**p, "sfh_dpl_log_total_mass": v}))

    compiled = jax.jit(jax.grad(loss)).lower(jnp.asarray(10.0)).compile()
    return float(compiled.cost_analysis()["flops"])


def test_auto_fit_attaches_a_grid_that_serves_the_dust(ssp, data):
    f = _auto_fit(_dusty_model(ssp), data)
    chain = f.model._cached_component_chain

    assert fast_nebular_can_engage(f.model) is True
    assert getattr(f.model, "_nebular_grid_table", None) is not None
    assert f.model._nebular_grid_table.serves_dust
    assert _find(chain, _DUST_TYPES).nebular_from_grid is True
    assert _find(chain, NebularSEDComponent).must_materialize_sed is False


def test_gradient_flops_drop_at_least_five_fold(ssp, data):
    baseline = _grad_flops(_dusty_model(ssp))
    auto = _grad_flops(_auto_fit(_dusty_model(ssp), data).model)
    print(f"gradient FLOPs: WavePrecomp only {baseline:,.0f}, auto {auto:,.0f}")

    assert auto < 0.2 * baseline, (
        f"gradient FLOPs {baseline:,.0f} (WavePrecomp only) -> {auto:,.0f} (auto): "
        f"{baseline / auto:.2f}x. Equality is the signature of a config that never "
        "reaches the graph (#1748)."
    )


def test_grid_served_photometry_matches_the_exact_path(ssp, data):
    exact = _dusty_model(ssp)
    fast = _auto_fit(_dusty_model(ssp), data).model
    worst = 0.0
    for tau_bc, tau_diff, logu in ((0.3, 0.2, -2.7), (1.2, 0.8, -3.2), (1.8, 1.5, -2.2)):
        p = {
            **_params(exact),
            "dust_tau_bc": tau_bc,
            "dust_tau_diff": tau_diff,
            "neb_logU": logu,
        }
        want = np.asarray(exact.predict_photometry(p))
        got = np.asarray(fast.predict_photometry(p))
        worst = max(worst, float(np.max(np.abs(got - want) / np.abs(want))))
    print(f"worst relative photometry difference, grid vs exact: {worst:.3e}")

    assert worst < 3e-2


def test_full_state_chain_keeps_the_exact_nebular_continuum(ssp, data):
    m = _auto_fit(_dusty_model(ssp), data).model

    assert _find(m._cached_component_chain, _DUST_TYPES).nebular_from_grid is True
    assert _find(m._full_state_chain(), _DUST_TYPES).nebular_from_grid is False

    state = m.predict_state(_params(m))
    assert float(jnp.max(jnp.abs(state.derived["sed_nebular"]))) > 0.0


def test_a_table_without_dust_channels_keeps_the_exact_path(ssp, monkeypatch):
    import tengri.components.nebular.nebular_grid_precompute as grid_module

    m = _dusty_model(ssp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old_style = precompute_nebular_grid(m, jnp.asarray([]), n_grid=4)
    assert not old_style.serves_dust
    monkeypatch.setattr(grid_module, "precompute_nebular_grid", lambda *a, **k: old_style)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.enable_fast_nebular(jnp.asarray([]), n_grid=4)
    chain = m._cached_component_chain

    assert _find(chain, _DUST_TYPES).nebular_from_grid is False
    assert _find(chain, NebularSEDComponent).must_materialize_sed is True


def test_shape_free_attenuation_stays_disarmed(ssp):
    free_shape = {**_TWO, "f_obscuration": Uniform(0.0, 0.5)}
    m = _dusty_model(ssp, dust=free_shape)
    chain = m._build_component_chain()

    assert "dust_f_obscuration" in (m.spec.free_params)
    assert m._energy_balance_lut(chain) is None
    assert fast_nebular_can_engage(m) is False
    assert any(isinstance(c, _DUST_TYPES) for c in _nebular_continuum_consumers(chain))
