# SPDX-License-Identifier: BSD-3-Clause
"""The loss function must THREAD the SSP grid, never BAKE it as an XLA constant.

Regression guard for the JIT data-threading bug: the loss-function builder had a
fast path that threaded ``ssp_data`` / ``template_data`` as arguments (so they
appear as XLA ``Parameter`` ops) only for **pure photometry** fits. Every other
configuration — spectroscopy, joint, and feature channels (line fluxes / ratios /
indices) — fell through to ``model.predict_spectrum(params)`` /
``model.predict_state(params)``, which closure-capture ``self.ssp_data``. Inlined
into the outer HMC/NUTS/VI/MAP ``jax.jit`` trace, that concrete grid becomes a
``Constant`` op (the SSP flux grid is ~8M floats on a real SSP) — ballooning cold
compile from ~5 s to ~40 s.

The contract these tests pin: for a stellar+dust model (no nebular, so the SSP
flux grid is the only large array), the SSP flux grid must NOT appear among the
constants of the loss-function jaxpr on ANY data channel. It rides in as an
invar via ``data_args["_jit_inputs"]`` instead.

Runs on the synthetic wide SSP (no ``data/ssp_*.h5`` needed, #613).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.inference.fitter import Fitter
from tengri.inference.loss_functions import build_loss_fn
from tengri.observation.photometry import FilterCurve
from tengri.observation.spectroscopy import Spectroscopy

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.contract


def _tophat(center, frac=0.16, n=40):
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


_PHOT = Photometry(filters=tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0)))
_SPEC_WAVE = jnp.linspace(3500.0, 9000.0, 200)


def _build_model(ssp, observation):
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={"type": "dpl", "*": FIXED, "log_total_mass": Uniform(8, 12)},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )


def _largest_baked_const(loss_fn, init, data_args):
    """Return the size of the largest array baked as a jaxpr constant."""
    jaxpr = jax.make_jaxpr(loss_fn)(init, data_args)
    sizes = [int(getattr(c, "size", 0)) for c in jaxpr.consts]
    return max(sizes, default=0)


@pytest.mark.parametrize(
    "channel",
    ["photometry", "spectroscopy", "joint"],
)
def test_ssp_grid_is_threaded_not_baked(synthetic_ssp_wide, channel):
    """On every data channel, the SSP flux grid is an invar, not a baked constant."""
    ssp = synthetic_ssp_wide
    ssp_size = int(np.asarray(ssp.ssp_flux).size)

    if channel == "photometry":
        obs = Observation(photometry=_PHOT)
        data = jnp.ones(len(_PHOT.filters))
    elif channel == "spectroscopy":
        obs = Observation(spectroscopy=Spectroscopy(wave_obs=_SPEC_WAVE))
        data = jnp.ones(_SPEC_WAVE.shape[0])
    else:  # joint
        obs = Observation(photometry=_PHOT, spectroscopy=Spectroscopy(wave_obs=_SPEC_WAVE))
        data = jnp.ones(len(_PHOT.filters) + _SPEC_WAVE.shape[0])

    model = _build_model(ssp, obs)
    noise = 0.1 * jnp.ones_like(data)
    fitter = Fitter(model, data, noise, data_type=channel)

    loss_fn = build_loss_fn(fitter)
    init = fitter._initialize_unbounded(jax.random.PRNGKey(0))

    biggest = _largest_baked_const(loss_fn, init, fitter._data_args)
    assert biggest < ssp_size, (
        f"[{channel}] a constant of size {biggest} >= SSP grid size {ssp_size} is baked "
        f"into the loss jaxpr — the SSP grid must be THREADED via data_args, not "
        f"closure-captured. This balloons cold compile."
    )


@pytest.mark.parametrize("channel", ["photometry", "spectroscopy", "joint"])
def test_threaded_and_baked_loss_agree_bit_for_bit(synthetic_ssp_wide, channel):
    """Threading changes WHERE the SSP grid enters the trace, never the physics.

    Evaluate the exact same loss with ``_jit_inputs`` present (threaded path) and
    with it stripped (the eager ``model.predict_*`` baked path). The two must be
    bit-identical — a guard against the threaded route silently computing a
    different number (the classic silent-no-op / wrong-array failure mode).
    """
    ssp = synthetic_ssp_wide
    if channel == "photometry":
        obs = Observation(photometry=_PHOT)
        data = jnp.ones(len(_PHOT.filters))
    elif channel == "spectroscopy":
        obs = Observation(spectroscopy=Spectroscopy(wave_obs=_SPEC_WAVE))
        data = jnp.ones(_SPEC_WAVE.shape[0])
    else:
        obs = Observation(photometry=_PHOT, spectroscopy=Spectroscopy(wave_obs=_SPEC_WAVE))
        data = jnp.ones(len(_PHOT.filters) + _SPEC_WAVE.shape[0])

    model = _build_model(ssp, obs)
    noise = 0.1 * jnp.ones_like(data)
    fitter = Fitter(model, data, noise, data_type=channel)

    loss_fn = build_loss_fn(fitter)
    init = fitter._initialize_unbounded(jax.random.PRNGKey(1))

    threaded_args = fitter._data_args
    baked_args = {k: v for k, v in threaded_args.items() if k != "_jit_inputs"}

    v_threaded = float(loss_fn(init, threaded_args))
    v_baked = float(loss_fn(init, baked_args))

    assert np.isfinite(v_threaded)
    assert v_threaded == pytest.approx(v_baked, rel=1e-12, abs=1e-9), (
        f"[{channel}] threaded loss {v_threaded} != baked loss {v_baked} — "
        f"threading changed the physics"
    )
