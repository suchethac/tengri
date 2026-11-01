# SPDX-License-Identifier: BSD-3-Clause
"""Batched spectroscopic forward pass: per-galaxy observation as a traced argument (#2833).

The batched path must reproduce, galaxy by galaxy, what a per-galaxy
:class:`SEDModel` built with that galaxy's own grid, resolution matrix and
redshift returns, including the gradient. Padded pixels must contribute
exactly zero.

Runs on the synthetic narrow SSP (no ``data/ssp_*.h5`` needed).
"""

from __future__ import annotations

import dataclasses

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
_FFT_SLACK = 64


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
        [GalaxySpectrum(wave=wave, flux=np.ones(160), ivar=np.ones(160), resolution=bm, z=z)],
        model=single_model,
    )[0]
    params = _base_params(single_model)
    out = predict_batched_observables(single_model, params, sb.galaxy(0), spec)
    ref = single_model.predict_spectrum({**params, "redshift": z})
    np.testing.assert_allclose(np.asarray(out["spec_fnu"]), np.asarray(ref), rtol=1e-12, atol=0.0)


def test_batch_matches_per_galaxy_reference_and_pads_to_zero(synthetic_ssp, galaxies):
    """(b) Three galaxies in one bucket: real pixels match per-galaxy models, padding is 0."""
    fluxes = [np.ones(g["wave"].size) for g in galaxies]
    records = [
        GalaxySpectrum(wave=g["wave"], flux=f, ivar=np.ones_like(f), resolution=g["bm"], z=g["z"])
        for g, f in zip(galaxies, fluxes)
    ]
    template = _free_z_model(synthetic_ssp, galaxies[0]["wave"], _banded(galaxies[0]["wave"]))
    buckets = build_spectro_batches(records, quantum=420, model=template)
    assert len(buckets) == 1
    spec, batch, index = buckets[0]
    assert spec.n_max == 420
    np.testing.assert_array_equal(index, [0, 1, 2])

    params = _base_params(template)
    predict = make_batched_predict(template, spec)
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
    template = _free_z_model(synthetic_ssp, wave, {"resolution": resolution})
    spec, batch, _ = build_spectro_batches(records, model=template)[0]
    assert spec.grid_kind == "nonuniform"
    params = _base_params(template)
    predict = make_batched_predict(template, spec)
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
    template = _free_z_model(synthetic_ssp, waves[1], {"resolution": 800.0})
    # The template-checked build refuses this padding (the flux-conserving Gaussian
    # LSF covers a range the observed grid sets), so the mask check forces the bucket
    # through, as the padding-exactness oracle does, and only checks the mask.
    spec, batch, _ = build_spectro_batches(records, quantum=260)[0]
    spec = dataclasses.replace(spec, conserving=True)
    assert spec.n_max == 260 and spec.grid_kind == "nonuniform"
    params = _base_params(template)
    predict = make_batched_predict(template, spec)
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
    template = _free_z_model(synthetic_ssp, galaxies[0]["wave"], _banded(galaxies[0]["wave"]))
    spec, batch, _ = build_spectro_batches(records, quantum=420, model=template)[0]
    params = _base_params(template)
    predict = make_batched_predict(template, spec)

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


def _sigma_v_template(ssp, wave, resolution):
    """Free-redshift template with sigma_v free on ``[0, 2000]`` km/s."""
    base = _free_z_model(ssp, wave, resolution)
    merged = base.spec.merge_observation_params(sigma_v_kms=Uniform(0.0, 2000.0))
    return SEDModel(merged, ssp, observation=base.observation)


def test_sigma_v_free_banded_forward_matches_reference_with_static_pad(synthetic_ssp):
    """(a) sigma_v = 250 km/s on the banded path, no padding: the static LSF pad is exact."""
    from tengri.forward.batched_spectrum import predict_batched_observables

    wave = _grid(160, 4000.0, 7000.0)
    z = 0.05
    template = _sigma_v_template(synthetic_ssp, wave, _banded(wave))
    bm = _banded(wave)["resolution_matrix"]
    record = GalaxySpectrum(wave=wave, flux=np.ones(160), ivar=np.ones(160), resolution=bm, z=z)
    spec, sb, _ = build_spectro_batches([record], model=template)[0]
    assert spec.lsf_pad_pixels is not None
    params = {**_base_params(template), "sigma_v_kms": jnp.asarray(250.0)}
    out = predict_batched_observables(template, params, sb.galaxy(0), spec)
    ref = template.predict_spectrum({**params, "redshift": z})
    np.testing.assert_allclose(np.asarray(out["spec_fnu"]), np.asarray(ref), rtol=1e-10, atol=0.0)


def _fft_lengths(jaxpr) -> list[int]:
    """Length of every FFT in a jaxpr, descending into nested (jitted, mapped) jaxprs."""
    lengths = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "fft":
            lengths.append(int(eqn.params["fft_lengths"][-1]))
        for value in eqn.params.values():
            for sub in value if isinstance(value, (list, tuple)) else [value]:
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    lengths.extend(_fft_lengths(inner))
    return lengths


def test_static_lsf_pad_shortens_the_batched_fft(synthetic_ssp):
    """(b) Banded sigma_v > 0 at 600 pixels: the static pad keeps the batched FFTs short.

    Measured on this grid: the largest FFT is 576 with the pad (256 per side) and
    1280 with the traced-grid fallback (``lsf_pad_pixels=None``).
    """
    n = 600
    wave = _grid(n, 4000.0, 7000.0)
    bm = _banded(wave)["resolution_matrix"]
    template = _sigma_v_template(synthetic_ssp, wave, _banded(wave))
    record = GalaxySpectrum(wave=wave, flux=np.ones(n), ivar=np.ones(n), resolution=bm, z=0.05)
    spec, batch, _ = build_spectro_batches([record], model=template)[0]
    n_max, pad = spec.n_max, spec.lsf_pad_pixels
    assert n_max == n and pad is not None
    params = _stack(_base_params(template), 1)

    def longest_fft(sp):
        fn = make_batched_predict(template, sp)
        return max(_fft_lengths(jax.make_jaxpr(fn)(params, batch).jaxpr))

    with_pad = longest_fft(spec)
    without = longest_fft(dataclasses.replace(spec, lsf_pad_pixels=None))
    assert with_pad < n_max + 2 * pad + _FFT_SLACK
    assert with_pad < without / 2
    assert without >= 2 * n_max


def _batched_single(template, wave, resolution, z, params):
    """Return ``(spec, spec_fnu)`` of one galaxy run through the batched forward pass."""
    record = GalaxySpectrum(
        wave=wave, flux=np.ones(wave.size), ivar=np.ones(wave.size), resolution=resolution, z=z
    )
    spec, batch, _ = build_spectro_batches([record], model=template)[0]
    predict = make_batched_predict(template, spec)
    return spec, np.asarray(predict(_stack(params, 1), batch)["spec_fnu"])[0]


def test_varying_r_on_log_uniform_grid_is_not_replaced_by_its_first_pixel(synthetic_ssp):
    """A per-pixel R rising 1000 -> 2000 on a log-uniform grid is used pixel by pixel (#3145).

    The batched spectrum must differ from the scalar R = 1000 batch by far more than
    round-off, and must equal the per-galaxy model built with the same R array.
    """
    wave = np.geomspace(4000.0, 7000.0, 160)
    r_rising = np.linspace(1000.0, 2000.0, 160)
    z = 0.05
    template = _free_z_model(synthetic_ssp, wave, {"resolution": 1000.0})
    params = _base_params(template)

    spec_rising, out_rising = _batched_single(template, wave, r_rising, z, params)
    _, out_scalar = _batched_single(template, wave, 1000.0, z, params)
    assert spec_rising.grid_kind == "nonuniform"

    peak = np.max(np.abs(out_scalar))
    assert np.max(np.abs(out_rising - out_scalar)) / peak > 1e-6

    ref = _free_z_model(synthetic_ssp, wave, {"resolution": r_rising})
    ref_spec = np.asarray(ref.predict_spectrum({**params, "redshift": z}))
    np.testing.assert_allclose(out_rising, ref_spec, rtol=1e-10, atol=0.0)


def test_constant_per_pixel_r_on_log_uniform_grid_keeps_the_scalar_path(synthetic_ssp):
    """A constant per-pixel R array is classified log-uniform and matches the scalar R."""
    wave = np.geomspace(4000.0, 7000.0, 160)
    z = 0.05
    template = _free_z_model(synthetic_ssp, wave, {"resolution": 1500.0})
    params = _base_params(template)

    spec_arr, out_arr = _batched_single(template, wave, np.full(160, 1500.0), z, params)
    spec_sc, out_sc = _batched_single(template, wave, 1500.0, z, params)
    assert spec_arr.grid_kind == "log_uniform"
    assert spec_sc.grid_kind == "log_uniform"
    np.testing.assert_allclose(out_arr, out_sc, rtol=1e-12, atol=0.0)


def test_padding_refusal_names_the_reason_of_the_grid_kind(synthetic_ssp):
    """Padding a varying-R galaxy takes the nonuniform reason; a constant-R one, the log reason."""
    wave = np.geomspace(4000.0, 7000.0, 160)
    template = _free_z_model(synthetic_ssp, wave, {"resolution": 1000.0})
    rising = GalaxySpectrum(
        wave=wave,
        flux=np.ones(160),
        ivar=np.ones(160),
        resolution=np.linspace(1000.0, 2000.0, 160),
        z=0.05,
    )
    constant = GalaxySpectrum(
        wave=wave, flux=np.ones(160), ivar=np.ones(160), resolution=1500.0, z=0.05
    )
    with pytest.raises(ValueError, match="rest-frame range"):
        build_spectro_batches([rising], quantum=192, model=template)
    with pytest.raises(ValueError, match="geometrically"):
        build_spectro_batches([constant], quantum=192, model=template)
