# SPDX-License-Identifier: BSD-3-Clause
r"""A dusty model with no IR consumer keeps the per-Q_H nebular grid (contract, #3267).

``FeaturePrecomp`` serves the nebular photometry from a per-Q_H grid. A dust component
can read that grid instead of the continuum only when the stellar energy-balance LUT
exists, and the LUT was gated on an IR consumer (dust emission or an ``L_ir`` reader).
A two-component dusty model with no dust emission therefore fell back to the exact
continuum path: 49.55 MFLOP per photometry evaluation, against 1.57 MFLOP with the grid.

The LUT is needed for the grid's optical-depth axes and absorbed-energy channel, not
for the IR publish, so the gate now admits the no-IR case whenever the grid will serve.

Pinned here, with the error split the owner accepted (2026-11-07):

* the model reports ``nebular_grid_can_serve_photometry()`` True;
* band photometry agrees with the exact (``approx=None``) path in float64 to 3e-4 at the
  prior draw (the grid and ``WavePrecomp`` errors together, at that point);
* grid-only error, pair against ``WavePrecomp`` alone, is bounded at
  ``_RTOL_GRID_ONLY`` at the worst points of the sweep (worst 1.94e-3 at des_g, z 0.5,
  tau_bc 0, tau_diff 4, old SFH, logU -1, logZ 0.2; mostly under 1e-3). The error of
  ``WavePrecomp`` itself against exact is not bounded here (tracked in #3268);
* the absorbed-energy keys ``l_dust_absorbed`` and ``l_tir`` are bit-identical to exact,
  because they come from the full-state path, where the dust flag resets;
* the photometry gradient is finite and non-zero in float32;
* the forward photometry costs more than 3x fewer FLOPs than ``WavePrecomp`` alone.
"""

import itertools

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

#: Grid-only bound: pair against ``WavePrecomp`` alone, at the worst sweep points.
#: Measured worst 1.94e-3 (des_g, z 0.5, tau_bc 0, tau_diff 4, old SFH, logU -1,
#: logZ 0.2) and 1.25e-3 (des_r, z 1, same tau and SFH); mostly under 1e-3 over the
#: 162-point grid at z 0.01 to 2. The owner accepted up to about 1.9e-3 (2026-11-07).
_RTOL_GRID_ONLY = 2.5e-3

#: Required FLOP advantage of the grid pair over ``WavePrecomp`` alone.
_MIN_FLOP_RATIO = 3.0

#: Worst grid-only points: (redshift, (tau_bc, tau_diff, sfh, logU, logZ)).
_GRID_ONLY_WORST = [
    (0.5, (0.0, 4.0, "old", -1.0, 0.2)),
    (1.0, (0.0, 4.0, "old", -1.0, 0.2)),
]

_SFH = {"young": (0.1, 0.1), "old": (5.0, 0.9)}  # (tau_gyr, age_frac)

#: Absorbed-energy keys read from the full-state path (``materialized()`` resets the
#: dust flag there), so the grid is not read and the pair equals exact to the bit.
_ABSORBED_KEYS = ("l_dust_absorbed", "l_tir")


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


def _build_grid(ssp, z, approx):
    """Delayed-SFH, two-component, Cue model with tau and logU/logZ as grid axes."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(_BANDS)),
        sfh={"type": "delayed", "all_params": FREE, "log_total_mass": Fixed(10.0)},
        neb={"type": "cue", "neb_logU": Uniform(-3.5, -1.0), "all_params": Fixed(DEFAULT)},
        met={"logzsol": Uniform(-1.0, 0.2)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": Uniform(0.0, 4.0),
            "tau_diff": Uniform(0.0, 4.0),
        },
        redshift=Fixed(z),
        approx=approx,
    )


def _grid_point(base, tau_bc, tau_diff, sfh_key, log_u, log_z):
    p = dict(base)
    p.update(
        dust_tau_bc=tau_bc,
        dust_tau_diff=tau_diff,
        sfh_delayed_tau_gyr=_SFH[sfh_key][0],
        sfh_delayed_age_frac=_SFH[sfh_key][1],
        neb_logU=log_u,
        met_logzsol=log_z,
    )
    return p


@pytest.mark.parametrize(("z", "point"), _GRID_ONLY_WORST)
def test_grid_only_error_against_wavepre_alone_stays_within_bound(ssp_data_fsps, z, point):
    pair = _build_grid(ssp_data_fsps, z, (WavePrecomp(), FeaturePrecomp()))
    wave = _build_grid(ssp_data_fsps, z, WavePrecomp())
    assert pair.nebular_grid_can_serve_photometry()
    fp = jax.jit(pair.predict_photometry)
    fw = jax.jit(wave.predict_photometry)
    base = dict(pair.spec.sample(jax.random.PRNGKey(0)))
    p = _grid_point(base, *point)
    ratio = np.abs(np.asarray(fp(p)) / np.asarray(fw(p)) - 1.0)
    worst_band = int(ratio.argmax())
    assert ratio.max() <= _RTOL_GRID_ONLY, (
        f"grid-only band error {ratio.max():.3e} ({_BANDS[worst_band]}, z={z}, point={point}) "
        f"exceeds {_RTOL_GRID_ONLY:.1e}"
    )


def test_absorbed_energy_keys_are_bit_identical_to_exact_on_the_no_ir_pair(ssp_data_fsps):
    pair = _build_grid(ssp_data_fsps, 0.5, (WavePrecomp(), FeaturePrecomp()))
    exact = _build_grid(ssp_data_fsps, 0.5, None)
    base = dict(pair.spec.sample(jax.random.PRNGKey(0)))
    for tau_bc, tau_diff, sfh_key in itertools.product((0.0, 2.0, 4.0), (0.0, 2.0, 4.0), _SFH):
        p = _grid_point(base, tau_bc, tau_diff, sfh_key, -2.25, -0.4)
        got = pair.predict_properties(p, names=_ABSORBED_KEYS)
        ref = exact.predict_properties(p, names=_ABSORBED_KEYS)
        for key in _ABSORBED_KEYS:
            np.testing.assert_array_equal(
                np.asarray(got[key]),
                np.asarray(ref[key]),
                err_msg=f"{key} differs from exact at tau=({tau_bc}, {tau_diff}), {sfh_key}",
            )
