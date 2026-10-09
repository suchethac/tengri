# SPDX-License-Identifier: BSD-3-Clause
"""Padding in a batched spectrum is exact, or it is refused at build time (#2833).

Each galaxy is padded to a common pixel count and its real pixels are compared
with the same galaxy evaluated alone on its own grid. The sweep covers the three
resolution paths (banded, Gaussian on a linear grid, Gaussian on a log-uniform
grid, plus banded on a log-uniform grid), sigma_v fixed at 0 or free, the
template's resample mode, and IGM on or off.

The oracle does not depend on the padding rule. For each combination the
padding is forced through (no template, so no refusal), and the real-pixel
error is measured. Then:

* if the forced error is at most 1e-12 of the peak, the template-checked build
  must succeed and match;
* otherwise the template-checked build must raise ``ValueError`` naming the
  padding, which is only correct if the forced error is above 1e-12.

A combination that neither matches nor raises fails the test.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, SEDModel, Spectroscopy, Uniform
from tengri.forward.batched_spectrum import predict_batched_observables
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches

pytestmark = pytest.mark.contract

_TOL = 1e-12
_N_REAL = 300
_N_MAX = 512
_Z = 2.5
_KINDS = ("banded_linear", "banded_log", "gauss_linear", "gauss_log")


def _galaxy_grid(kind: str) -> np.ndarray:
    if kind.endswith("_log"):
        return np.geomspace(4000.0, 7000.0, _N_REAL)
    return np.linspace(4000.0, 7000.0, _N_REAL)


def _resolution(kind: str, wave: np.ndarray) -> dict:
    if kind.startswith("banded"):
        return {"resolution_matrix": gaussian_resolution_bands(jnp.asarray(wave), 300.0, 21)}
    return {"resolution": 1000.0}


def _build(ssp, wave, kind: str, *, sigma_v_free: bool, resample: str, igm: bool):
    """Free-redshift model on ``wave``; the sigma_v and IGM options as the sweep requires."""
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(wave), resample=resample, **_resolution(kind, wave)
    )
    observation = Observation(spectroscopy=spectroscopy)
    kwargs = dict(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "sfh_dpl_log_total_mass": Uniform(8.0, 12.0),
        },
        neb={"type": "none"},
        redshift=Uniform(0.0, 4.0),
    )
    if igm:
        kwargs["igm"] = {"type": "inoue14"}
    model = SEDModel.build(**kwargs)
    if sigma_v_free:
        merged = model.spec.merge_observation_params(sigma_v_kms=Uniform(0.0, 2000.0))
        model = SEDModel(merged, ssp, observation=observation)
    return model


def _params(model):
    p = dict(model.spec.sample(jax.random.PRNGKey(3)))
    p["sfh_dpl_log_total_mass"] = jnp.asarray(10.0)
    if "sigma_v_kms" in p:
        p["sigma_v_kms"] = jnp.asarray(250.0)
    return p


@pytest.mark.parametrize("igm", [False, True], ids=["igm_off", "igm_on"])
@pytest.mark.parametrize("sigma_v_free", [False, True], ids=["sigma_v_0", "sigma_v_free"])
@pytest.mark.parametrize("resample", ["point", "conserving"])
@pytest.mark.parametrize("kind", _KINDS)
def test_padding_matches_unpadded_or_is_refused(
    synthetic_ssp_wide, kind, resample, sigma_v_free, igm
):
    wave = _galaxy_grid(kind)
    conserving = resample == "conserving"
    res = _resolution(kind, wave)
    resolution = res["resolution_matrix"] if "resolution_matrix" in res else res["resolution"]
    own = _build(
        synthetic_ssp_wide, wave, kind, sigma_v_free=sigma_v_free, resample=resample, igm=igm
    )
    template = _build(
        synthetic_ssp_wide,
        np.linspace(4000.0, 7000.0, 100),
        kind,
        sigma_v_free=sigma_v_free,
        resample=resample,
        igm=igm,
    )
    params = _params(own)
    reference = np.asarray(own.predict_spectrum({**params, "redshift": _Z}))
    peak = np.max(np.abs(reference))
    record = GalaxySpectrum(
        wave=wave,
        flux=np.ones(_N_REAL),
        ivar=np.ones(_N_REAL),
        resolution=resolution,
        z=_Z,
    )

    # Oracle, independent of the padding rule: force the padded bucket through.
    [(forced_spec, forced_batch, _)] = build_spectro_batches([record], quantum=_N_MAX)
    forced_spec = dataclasses.replace(forced_spec, conserving=conserving)
    forced = predict_batched_observables(template, params, forced_batch.galaxy(0), forced_spec)
    forced_err = float(np.max(np.abs(np.asarray(forced["spec_fnu"])[:_N_REAL] - reference)) / peak)

    try:
        [(spec, batch, _)] = build_spectro_batches([record], quantum=_N_MAX, model=template)
    except ValueError as exc:
        assert "padding would change real pixels" in str(exc)
        assert f"n_real={_N_REAL}, n_max={_N_MAX}" in str(exc)
        assert forced_err > _TOL, f"refused, but the padded result matches ({forced_err:.2e})"
        return

    assert forced_err <= _TOL, f"built without refusal, but the padded error is {forced_err:.2e}"
    out = np.asarray(
        predict_batched_observables(template, params, batch.galaxy(0), spec)["spec_fnu"]
    )
    assert float(np.max(np.abs(out[:_N_REAL] - reference)) / peak) <= _TOL
    np.testing.assert_array_equal(out[_N_REAL:], 0.0)
