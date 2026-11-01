# SPDX-License-Identifier: BSD-3-Clause
"""Batched path: per-galaxy Milky Way E(B-V) on the foreground screen (#3140).

The template fixes the law and R_V; each galaxy carries its own E(B-V). A batch
of galaxies with different E(B-V) must reproduce, galaxy by galaxy, a per-galaxy
model built with that E(B-V), and one compiled predict must serve every E(B-V)
value of the same spec.

Runs on the synthetic narrow SSP (no ``data/ssp_*.h5`` needed).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, SEDModel, Spectroscopy, Uniform
from tengri.forward.batched_spectrum import make_batched_predict
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches

pytestmark = pytest.mark.contract

_N_DIAG = 21
_R_BANDED = 300.0
_LAW = "fitzpatrick99"


def _banded(wave):
    return gaussian_resolution_bands(jnp.asarray(wave), _R_BANDED, _N_DIAG)


def _template(ssp, wave, foreground=None):
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(wave),
        resample="conserving",
        resolution_matrix=_banded(wave),
    )
    kwargs = {} if foreground is None else {"foreground": foreground}
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=spectroscopy),
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "sfh_dpl_log_total_mass": Uniform(8.0, 12.0),
        },
        neb={"type": "none"},
        redshift=Uniform(0.0, 0.5),
        **kwargs,
    )


def _stack(params: dict, batch: int) -> dict:
    return {k: jnp.broadcast_to(jnp.asarray(v), (batch,)) for k, v in params.items()}


@pytest.fixture(scope="module")
def wave():
    return np.linspace(4000.0, 7000.0, 160)


@pytest.fixture(scope="module")
def template(synthetic_ssp, wave):
    return _template(synthetic_ssp, wave, foreground={"ebmv_mw": 0.0, "law": _LAW})


def test_two_galaxies_with_their_own_ebmv_match_per_galaxy_models(synthetic_ssp, wave, template):
    ebmv = [0.05, 0.2]
    redshifts = [0.05, 0.1]
    records = [
        GalaxySpectrum(
            wave=wave,
            flux=np.ones(wave.size),
            ivar=np.ones(wave.size),
            resolution=_banded(wave),
            z=z,
            ebmv_mw=e,
        )
        for z, e in zip(redshifts, ebmv)
    ]
    ((spec, batch, index),) = build_spectro_batches(records, model=template)
    np.testing.assert_array_equal(index, [0, 1])

    params = dict(template.spec.sample(jax.random.PRNGKey(0)))
    predict = make_batched_predict(template, spec)
    out = np.asarray(predict(_stack(params, 2), batch)["spec_fnu"])

    for i, (z, e) in enumerate(zip(redshifts, ebmv)):
        ref_model = _template(synthetic_ssp, wave, foreground={"ebmv_mw": e, "law": _LAW})
        ref = np.asarray(ref_model.predict_spectrum({**params, "redshift": z}))
        np.testing.assert_allclose(out[i], ref, rtol=1e-10, atol=0.0)


def test_one_compile_serves_every_ebmv_value_of_the_spec(synthetic_ssp, wave, template):
    from jax._src import test_util as jtu

    records = [
        GalaxySpectrum(
            wave=wave,
            flux=np.ones(wave.size),
            ivar=np.ones(wave.size),
            resolution=_banded(wave),
            z=0.05,
            ebmv_mw=e,
        )
        for e in (0.05, 0.2)
    ]
    ((spec, batch, _),) = build_spectro_batches(records, model=template)
    params = _stack(dict(template.spec.sample(jax.random.PRNGKey(0))), 2)
    predict = make_batched_predict(template, spec)
    predict(params, batch)
    other = dataclasses.replace(batch, ebmv_mw=jnp.asarray([0.1, 0.3]))
    with jtu.count_jit_compilation_cache_miss() as counter:
        predict(params, other)
    assert counter() == 0


def test_template_screen_with_galaxy_without_ebmv_raises(synthetic_ssp, wave):
    loaded = _template(synthetic_ssp, wave, foreground={"ebmv_mw": 0.1, "law": _LAW})
    record = GalaxySpectrum(
        wave=wave,
        flux=np.ones(wave.size),
        ivar=np.ones(wave.size),
        resolution=_banded(wave),
        z=0.05,
    )
    with pytest.raises(ValueError, match="E\\(B-V\\)"):
        build_spectro_batches([record], model=loaded)


def test_galaxy_without_ebmv_matches_a_model_with_no_screen(synthetic_ssp, wave):
    template = _template(synthetic_ssp, wave)
    record = GalaxySpectrum(
        wave=wave,
        flux=np.ones(wave.size),
        ivar=np.ones(wave.size),
        resolution=_banded(wave),
        z=0.05,
    )
    ((spec, batch, _),) = build_spectro_batches([record], model=template)
    assert not spec.has_foreground
    params = dict(template.spec.sample(jax.random.PRNGKey(0)))
    out = np.asarray(make_batched_predict(template, spec)(_stack(params, 1), batch)["spec_fnu"])
    ref = np.asarray(template.predict_spectrum({**params, "redshift": 0.05}))
    np.testing.assert_allclose(out[0], ref, rtol=1e-10, atol=0.0)


def test_float32_batched_matches_per_galaxy_and_gradient_is_finite_and_nonzero(
    synthetic_ssp_physical, wave
):
    """Float32 batched equals per-galaxy (rtol 1e-4); gradient finite, nonzero.

    Galaxy-scale log10 M = 10 on the physically normalized SSP (#3149).
    """
    mass, z, ebmv = 10.0, 0.05, 0.1
    with jax.enable_x64(False):
        template = _template(
            synthetic_ssp_physical, wave, foreground={"ebmv_mw": 0.0, "law": _LAW}
        )
        ref_model = _template(
            synthetic_ssp_physical, wave, foreground={"ebmv_mw": ebmv, "law": _LAW}
        )
        record = GalaxySpectrum(
            wave=wave,
            flux=np.ones(wave.size),
            ivar=np.ones(wave.size),
            resolution=_banded(wave),
            z=z,
            ebmv_mw=ebmv,
        )
        ((spec, batch, _),) = build_spectro_batches([record], model=template)
        name = next(n for n in template.spec.free_params if n.endswith("log_total_mass"))
        predict = make_batched_predict(template, spec)
        stacked = jnp.asarray([mass], dtype=jnp.float32)
        out = np.asarray(predict({name: stacked}, batch)["spec_fnu"])[0]
        ref = np.asarray(ref_model.predict_spectrum({name: mass, "redshift": z}))
        grad = jax.grad(lambda m: jnp.log(jnp.sum(predict({name: m}, batch)["spec_fnu"])))(stacked)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=0.0)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.all(np.asarray(grad) != 0.0)
