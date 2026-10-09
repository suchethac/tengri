# SPDX-License-Identifier: BSD-3-Clause
"""Batched spectroscopic forward pass: per-galaxy observation as a traced argument (#2833).

The batched path must reproduce, galaxy by galaxy, what a per-galaxy
:class:`SEDModel` built with that galaxy's own grid, resolution matrix and
redshift returns, including the gradient. Padded pixels must contribute
exactly zero.

Runs on the synthetic narrow SSP (no ``data/ssp_*.h5`` needed).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, SEDModel, Spectroscopy, Uniform
from tengri.forward.batched_spectrum import (
    check_template_model,
    make_batched_predict,
)
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches

pytestmark = pytest.mark.contract

_N_DIAG = 21
_R_BANDED = 300.0


def _grid(n_pix: int, lo: float, hi: float) -> np.ndarray:
    return np.linspace(lo, hi, n_pix)


def _template(ssp, wave, resolution, *, redshift):
    """Spectroscopic model built on ``wave`` with the given resolution and z prior."""
    spectroscopy = Spectroscopy(wave_obs=jnp.asarray(wave), resample="conserving", **resolution)
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=spectroscopy),
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "sfh_dpl_log_total_mass": Uniform(8.0, 12.0),
        },
        neb={"type": "none"},
        redshift=redshift,
    )


def _banded(wave):
    return {"resolution_matrix": gaussian_resolution_bands(jnp.asarray(wave), _R_BANDED, _N_DIAG)}


def _free_z_model(ssp, wave, resolution):
    return _template(ssp, wave, resolution, redshift=Uniform(0.0, 0.5))


def _base_params(model, key=0):
    """One sample of the free parameters, including a redshift the caller overrides."""
    return dict(model.spec.sample(jax.random.PRNGKey(key)))


def _stack(params: dict, batch: int) -> dict:
    return {k: jnp.broadcast_to(jnp.asarray(v), (batch,)) for k, v in params.items()}


@pytest.fixture(scope="module")
def single_model(synthetic_ssp):
    wave = _grid(160, 4000.0, 7000.0)
    return _free_z_model(synthetic_ssp, wave, _banded(wave))


@pytest.fixture(scope="module")
def galaxies():
    """Three linear grids with different pixel counts, redshifts and banded matrices."""
    specs = [(300, 4000.0, 7000.0, 0.05), (360, 4200.0, 7400.0, 0.1), (420, 4500.0, 7500.0, 0.2)]
    out = []
    for n_pix, lo, hi, z in specs:
        wave = _grid(n_pix, lo, hi)
        out.append(
            {
                "wave": wave,
                "z": z,
                "bm": gaussian_resolution_bands(jnp.asarray(wave), _R_BANDED, _N_DIAG),
            }
        )
    return out


def test_identity_single_galaxy_matches_own_model(synthetic_ssp, single_model):
    """(a) No padding: the batched spectrum equals the model's own prediction exactly."""
    from tengri.forward.batched_spectrum import predict_batched_observables

    wave = _grid(160, 4000.0, 7000.0)
    z = 0.05
    bm = _banded(wave)["resolution_matrix"]
    spec, sb, _ = build_spectro_batches(
        [GalaxySpectrum(wave=wave, flux=np.ones(160), ivar=np.ones(160), resolution=bm, z=z)]
    )[0]
    params = _base_params(single_model)
    out = predict_batched_observables(single_model, params, sb.galaxy(0), spec, conserving=True)
    ref = single_model.predict_spectrum({**params, "redshift": z})
    np.testing.assert_allclose(np.asarray(out["spec_fnu"]), np.asarray(ref), rtol=1e-12, atol=0.0)


def test_batch_matches_per_galaxy_reference_and_pads_to_zero(synthetic_ssp, galaxies):
    """(b) Three galaxies in one bucket: real pixels match per-galaxy models, padding is 0."""
    fluxes = [np.ones(g["wave"].size) for g in galaxies]
    records = [
        GalaxySpectrum(wave=g["wave"], flux=f, ivar=np.ones_like(f), resolution=g["bm"], z=g["z"])
        for g, f in zip(galaxies, fluxes)
    ]
    buckets = build_spectro_batches(records, quantum=420)
    assert len(buckets) == 1
    spec, batch, index = buckets[0]
    assert spec.n_max == 420
    np.testing.assert_array_equal(index, [0, 1, 2])

    template = _free_z_model(synthetic_ssp, galaxies[0]["wave"], _banded(galaxies[0]["wave"]))
    params = _base_params(template)
    predict = make_batched_predict(template, spec, conserving=True)
    out = np.asarray(predict(_stack(params, 3), batch)["spec_fnu"])
    assert out.shape == (3, 420)

    for i, g in enumerate(galaxies):
        n = g["wave"].size
        ref_model = _free_z_model(synthetic_ssp, g["wave"], {"resolution_matrix": g["bm"]})
        ref = np.asarray(ref_model.predict_spectrum({**params, "redshift": g["z"]}))
        np.testing.assert_allclose(out[i, :n], ref, rtol=1e-10, atol=0.0)
        np.testing.assert_array_equal(out[i, n:], 0.0)


def test_gaussian_resolution_uses_variable_path(synthetic_ssp):
    """(c) Scalar Gaussian R on a linear grid: the batched path matches the reference."""
    wave = _grid(200, 4000.0, 7000.0)
    resolution = 800.0
    records = [
        GalaxySpectrum(
            wave=wave, flux=np.ones(200), ivar=np.ones(200), resolution=resolution, z=0.05
        )
    ]
    spec, batch, _ = build_spectro_batches(records)[0]
    assert spec.grid_kind == "nonuniform"

    template = _free_z_model(synthetic_ssp, wave, {"resolution": resolution})
    params = _base_params(template)
    predict = make_batched_predict(template, spec, conserving=True)
    out = np.asarray(predict(_stack(params, 1), batch)["spec_fnu"])[0]

    ref = np.asarray(template.predict_spectrum({**params, "redshift": 0.05}))
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=0.0)


def test_gaussian_padded_pixels_are_exactly_zero(synthetic_ssp):
    """(c) Gaussian R with padding: the mask zeroes pixels the variable-R path leaves nonzero."""
    waves = [_grid(200, 4000.0, 7000.0), _grid(260, 4000.0, 7200.0)]
    records = [
        GalaxySpectrum(
            wave=w, flux=np.ones(w.size), ivar=np.ones(w.size), resolution=800.0, z=0.05
        )
        for w in waves
    ]
    spec, batch, _ = build_spectro_batches(records, quantum=260)[0]
    assert spec.n_max == 260 and spec.grid_kind == "nonuniform"

    template = _free_z_model(synthetic_ssp, waves[1], {"resolution": 800.0})
    params = _base_params(template)
    predict = make_batched_predict(template, spec, conserving=True)
    out = np.asarray(predict(_stack(params, 2), batch)["spec_fnu"])

    ref = np.asarray(
        _free_z_model(synthetic_ssp, waves[0], {"resolution": 800.0}).predict_spectrum(
            {**params, "redshift": 0.05}
        )
    )
    np.testing.assert_allclose(out[0, :200], ref, rtol=1e-10, atol=0.0)
    np.testing.assert_array_equal(out[0, 200:], 0.0)


def test_gradient_matches_per_galaxy_reference(synthetic_ssp, galaxies):
    """(d) d(sum spec)/d(log_total_mass) batched equals the per-galaxy gradient."""
    records = [
        GalaxySpectrum(
            wave=g["wave"],
            flux=np.ones(g["wave"].size),
            ivar=np.ones(g["wave"].size),
            resolution=g["bm"],
            z=g["z"],
        )
        for g in galaxies
    ]
    spec, batch, _ = build_spectro_batches(records, quantum=420)[0]
    template = _free_z_model(synthetic_ssp, galaxies[0]["wave"], _banded(galaxies[0]["wave"]))
    params = _base_params(template)
    predict = make_batched_predict(template, spec, conserving=True)

    def batched_loss(log_mass_per_galaxy):
        # One entry per galaxy, so each gradient entry is that galaxy's own
        # derivative; a shared scalar would return the sum over the batch.
        p = {**_stack(params, 3), "sfh_dpl_log_total_mass": log_mass_per_galaxy}
        return jnp.sum(predict(p, batch)["spec_fnu"])

    log_mass0 = float(params["sfh_dpl_log_total_mass"])
    grad_batched = np.asarray(jax.grad(batched_loss)(jnp.full((3,), log_mass0, dtype=jnp.float64)))

    for i, g in enumerate(galaxies):
        ref_model = _free_z_model(synthetic_ssp, g["wave"], {"resolution_matrix": g["bm"]})

        def ref_loss(log_mass, ref_model=ref_model, z=g["z"]):
            return jnp.sum(
                ref_model.predict_spectrum(
                    {**params, "sfh_dpl_log_total_mass": log_mass, "redshift": z}
                )
            )

        grad_ref = float(jax.grad(ref_loss)(jnp.asarray(params["sfh_dpl_log_total_mass"])))
        np.testing.assert_allclose(grad_batched[i], grad_ref, rtol=1e-9, atol=0.0)


def test_check_template_model_rejects_fixed_redshift(synthetic_ssp):
    """(e) A model with redshift baked in is refused before any batched compile."""
    wave = _grid(100, 4000.0, 7000.0)
    fixed_z = _template(synthetic_ssp, wave, _banded(wave), redshift=Fixed(0.05))
    with pytest.raises(ValueError, match="redshift"):
        check_template_model(fixed_z)


def test_check_template_model_accepts_free_redshift(single_model):
    check_template_model(single_model)
