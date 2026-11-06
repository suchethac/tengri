# SPDX-License-Identifier: BSD-3-Clause
r"""A dusty model with no IR consumer keeps the per-Q_H nebular grid (contract).

``FeaturePrecomp`` serves the nebular photometry from a per-Q_H grid. A dust component
can read that grid instead of the continuum only when the stellar energy-balance LUT
exists, and the LUT was gated on an IR consumer (dust emission or an ``L_ir`` reader).
A two-component dusty model with no dust emission therefore fell back to the exact
continuum path: 49.55 MFLOP and 2.65 ms per photometry evaluation, against 7.5 MFLOP and
1.38 ms for the same physics with dust emission. Adding less physics cost 7x more.

The LUT is needed for the grid's optical-depth axes and absorbed-energy channel, not
for the IR publish, so the gate now admits the no-IR case whenever the grid will serve.

Pinned here:

* the model reports ``nebular_grid_can_serve_photometry()`` True;
* its band photometry agrees with the exact (``approx=None``) path in float64 to
  3e-4 at the prior draw, and to ``_RTOL_GRID_EDGES`` over the prior edges (the
  worst-point grid below, measured 2026-11-06: 1.14e-3 at worst);
* the photometry gradient is finite and non-zero in float32;
* the forward photometry costs more than 3x fewer FLOPs than ``WavePrecomp`` alone on
  the same model (the dust-free control is the ``WavePrecomp`` arm, not a different
  physics block).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.forward.sed_model import FeaturePrecomp, WavePrecomp

pytestmark = pytest.mark.contract

_BANDS = ["des_g", "des_r", "des_z", "wise_w1", "wise_w2"]

#: Same bound as ``tests/regression/bug/test_approx_channel_isolation.py``.
_RTOL_GRID_SERVED = 3e-4

#: Required FLOP advantage of the grid pair over ``WavePrecomp`` alone.
_MIN_FLOP_RATIO = 3.0


def _build(ssp, approx):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(_BANDS)),
        sfh={"type": "continuity", "log_total_mass": Fixed(10.0), "all_params": FREE},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": Uniform(0.0, 2.0),
        },
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        approx=approx,
    )


def _point(model):
    """Full parameter dict and its free subset, at one prior draw."""
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    free = set(model.spec.free_params)
    return p, {k: v for k, v in p.items() if k in free}


def _forward_flops(model, p, q):
    compiled = jax.jit(lambda qq: model.predict_photometry({**p, **qq})).lower(q).compile()
    cost = compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    return float(cost["flops"])


@pytest.fixture(scope="module")
def pair_model(ssp_data_fsps):
    return _build(ssp_data_fsps, (WavePrecomp(), FeaturePrecomp()))


@pytest.fixture(scope="module")
def wave_model(ssp_data_fsps):
    return _build(ssp_data_fsps, WavePrecomp())


@pytest.fixture(scope="module")
def exact_model(ssp_data_fsps):
    return _build(ssp_data_fsps, None)


def test_dusty_model_without_ir_reports_grid_can_serve(pair_model):
    assert pair_model.nebular_grid_can_serve_photometry() is True


def test_grid_bands_match_exact_path_in_float64(pair_model, exact_model):
    p, _ = _point(pair_model)
    grid = np.asarray(pair_model.predict_photometry(p))
    exact = np.asarray(exact_model.predict_photometry(p))
    assert np.all(np.isfinite(grid))
    np.testing.assert_allclose(grid, exact, rtol=_RTOL_GRID_SERVED, atol=0.0)


def test_gradient_is_finite_and_nonzero_in_float32(pair_model):
    p, q = _point(pair_model)
    with jax.enable_x64(False):
        q32 = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in q.items()}
        p32 = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in p.items()}

        # log10 of the photometry, not the photometry: a cgs flux of a 1e10 Msun galaxy
        # is ~1e-12, and a float32 reverse-mode gradient of the raw flux underflows
        # to exactly zero (the gradient would be zero for every model, so the check
        # would say nothing about the grid).
        def loss(qq):
            return jnp.sum(jnp.log10(pair_model.predict_photometry({**p32, **qq})))

        grads = jax.jit(jax.grad(loss))(q32)
    flat = np.concatenate([np.ravel(np.asarray(g)) for g in grads.values()])
    assert np.all(np.isfinite(flat))
    assert np.any(flat != 0.0)


def test_grid_pair_costs_at_least_3x_fewer_flops_than_wavepre_alone(pair_model, wave_model):
    p, q = _point(pair_model)
    wave_flops = _forward_flops(wave_model, p, q)
    pair_flops = _forward_flops(pair_model, p, q)
    assert pair_flops > 0.0
    assert wave_flops / pair_flops > _MIN_FLOP_RATIO


_SFH = {"young": (0.1, 0.1), "old": (5.0, 0.9)}  # (tau_gyr, age_frac)

#: Worst points of the 162-point grid (tau_bc, tau_diff over the prior, young/old
#: SFH, logU, logZ), measured 2026-11-06 per redshift on the no-IR model. The first
#: row is the worst of all: des_r at z=0.1, max |ratio-1| 1.14e-3.
_GRID_WORST = {
    0.1: [
        (2.0, 2.0, "old", -1.25, -0.4),  # des_r, 1.14e-3
        (1.0, 2.0, "old", -3.25, -1.0),  # des_g, 4.4e-4
        (2.0, 2.0, "old", -3.25, -0.4),  # wise_w1, 5.3e-4
        (0.0, 1.0, "old", -1.25, -0.4),  # des_z
    ],
    2.0: [
        (1.0, 2.0, "old", -3.25, 0.2),  # des_r, 3.8e-4
        (0.0, 2.0, "old", -1.25, 0.2),  # des_z, 4.4e-4
        (0.0, 1.0, "young", -3.25, -0.4),  # des_g, 3.6e-4
    ],
}

#: Grid-served band error bound over the prior edges. Measured worst 1.14e-3 on the
#: no-IR model and identical to the eligible dale2014 model to about 1 %; the owner
#: accepted up to about 1.2e-3 from the nebular grid for this speed-up (2026-11-06).
#: The 3e-4 bound above still holds at the prior draw.
_RTOL_GRID_EDGES = 1.5e-3


def _build_grid(ssp, z, emission, approx):
    kw = dict(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(_BANDS)),
        sfh={"type": "delayed", "all_params": FREE, "log_total_mass": Fixed(10.0)},
        neb={"type": "cue", "neb_logU": Uniform(-3.5, -1.0), "all_params": Fixed(DEFAULT)},
        met={"logzsol": Uniform(-1.0, 0.2)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": Uniform(0.0, 2.0),
            "tau_diff": Uniform(0.0, 2.0),
        },
        redshift=Fixed(z),
        approx=approx,
    )
    if emission is not None:
        kw["dust_emission"] = {"type": emission, "all_params": Fixed(DEFAULT)}
    return SEDModel.build(**kw)


@pytest.mark.parametrize("z", [0.1, 2.0])
@pytest.mark.parametrize("emission", [None, "dale2014"], ids=["no_ir", "dale2014"])
def test_grid_error_over_prior_edges_stays_within_bound(ssp_data_fsps, z, emission):
    pair = _build_grid(ssp_data_fsps, z, emission, (WavePrecomp(), FeaturePrecomp()))
    exact = _build_grid(ssp_data_fsps, z, emission, None)
    assert pair.nebular_grid_can_serve_photometry()
    fp = jax.jit(pair.predict_photometry)
    fe = jax.jit(exact.predict_photometry)
    base = dict(pair.spec.sample(jax.random.PRNGKey(0)))
    worst, where = 0.0, None
    for tau_bc, tau_diff, sfh_key, log_u, log_z in _GRID_WORST[z]:
        p = dict(base)
        p.update(
            dust_tau_bc=tau_bc,
            dust_tau_diff=tau_diff,
            sfh_delayed_tau_gyr=_SFH[sfh_key][0],
            sfh_delayed_age_frac=_SFH[sfh_key][1],
            neb_logU=log_u,
            met_logzsol=log_z,
        )
        ratio = np.abs(np.asarray(fp(p)) / np.asarray(fe(p)) - 1.0)
        if ratio.max() > worst:
            worst = float(ratio.max())
            where = (tau_bc, tau_diff, sfh_key, log_u, log_z, _BANDS[int(ratio.argmax())])
    assert worst <= _RTOL_GRID_EDGES, (
        f"grid-served band error {worst:.3e} at {where} exceeds {_RTOL_GRID_EDGES:.1e}"
    )
