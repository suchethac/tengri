# SPDX-License-Identifier: BSD-3-Clause
"""Emission lines on the batched spectroscopic path with ``eline_mode="fitted"``.

The batched likelihood must score the same model as the per-galaxy
``ELineFittedLikelihood``: the line design matrix is added to the calibrated
continuum with one amplitude per independent line. These tests pin that
parity, the mask and padding behavior, the one-compile-per-spec property, the
refusals, and the amplitude latents of the MAP fit.
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
    require_model_spec,
)
from tengri.inference.batched_spectro import (
    batched_log_likelihood,
    fit_spectra_map_vmap,
)
from tengri.inference.likelihoods.marginalized import ELineFittedLikelihood
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches
from tengri.observation.eline_marginalization import (
    apply_doublet_constraints,
    build_eline_design_matrix,
    eline_amplitude_names,
    eline_scalar_resolution,
)
from tengri.observation.line_list import LineList

pytestmark = pytest.mark.contract

_LINES = ["Hbeta", "OIII_4959", "OIII_5007", "Halpha"]
_R = 60.0  # 160 px over 3500 Angstrom: a line of sigma ~35 A spans several pixels
_SIGMA_KMS = 120.0  # pinned on the template through Parameters.with_fixed_value
_PRIOR_SIGMA = 1.0e-15  # eline_prior_sigma [F_nu]; the synthetic continuum median is 1.3e-15
# Line amplitudes are F_nu times Angstrom: the design columns are unit-area Gaussians.
_AMPS = {"eline_amp_Hbeta": 3.0e-16, "eline_amp_OIII_5007": 5.0e-16, "eline_amp_Halpha": 8.0e-16}


def _grid(n_pix: int, lo: float, hi: float) -> np.ndarray:
    return np.linspace(lo, hi, n_pix)


def _template(ssp, wave, *, eline_mode="fitted", eline_broad=False, lines=_LINES):
    """Free-redshift template with lines; the stellar mass is the only physical latent."""
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(wave),
        resolution=_R,
        resample="conserving",
        eline_mode=eline_mode,
        eline_catalog=LineList.from_names(list(lines)),
        eline_prior_sigma=_PRIOR_SIGMA,
        eline_broad=eline_broad,
    )
    model = SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=spectroscopy),
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "sfh_dpl_log_total_mass": Uniform(8.0, 12.0),
        },
        neb={"type": "none"},
        igm={"type": "inoue14"},
        redshift=Uniform(0.0, 0.5),
    )
    return model if eline_mode == "off" else _with_width(model, _SIGMA_KMS)


def _with_width(model, sigma_kms):
    """Pin ``eline_sigma_kms`` at ``sigma_kms`` by the same rebuild ``with_fixed_redshift`` uses."""
    spec = model.spec.with_fixed_value("eline_sigma_kms", float(sigma_kms))
    return SEDModel(spec, model.ssp_data, observation=model.observation)


def _galaxy(wave, z, flux, ivar):
    return GalaxySpectrum(wave=wave, flux=flux, ivar=ivar, z=z, resolution=_R)


def _stack(params: dict, batch: int) -> dict:
    return {k: jnp.broadcast_to(jnp.asarray(v), (batch,)) for k, v in params.items()}


def _stellar_params(model, mass: float = 10.0) -> dict:
    return {"sfh_dpl_log_total_mass": mass}


def _line_design(model, wave, z):
    """Doublet-constrained design matrix of the template's catalog on ``wave``."""
    catalog = model.observation.spectroscopy.effective_catalog
    G = build_eline_design_matrix(
        jnp.asarray(wave), jnp.asarray(catalog.wavelengths), _R, z, _SIGMA_KMS, 0.0
    )
    return apply_doublet_constraints(G, catalog.build_constraint_matrix())


def _amp_vector(names):
    return jnp.asarray([_AMPS[nm] for nm in names])


def test_eline_scalar_resolution_scalar_array_and_none():
    """A scalar passes through, an array reduces to its median, None falls back to 2000."""
    assert eline_scalar_resolution(1500.0) == 1500.0
    assert eline_scalar_resolution(np.array([1000.0, 3000.0, 2000.0])) == 2000.0
    assert eline_scalar_resolution(None) == 2000.0


def test_eline_amplitude_names_skip_doublet_secondaries():
    """One amplitude name per line that is not a doublet secondary, in catalog order."""
    catalog = LineList.from_names(_LINES)
    names = eline_amplitude_names(catalog)
    secondary = {dc.secondary_idx for dc in catalog.doublets}
    expected = [f"eline_amp_{nm}" for i, nm in enumerate(catalog.names) if i not in secondary]
    assert names == expected
    assert len(names) == catalog.n_lines - len(secondary)
    assert all(nm.startswith("eline_amp_") for nm in names)


def test_batched_spectrum_is_continuum_plus_line_model(synthetic_ssp):
    """Parity (a): spec_fnu equals the per-galaxy continuum plus G_eff @ amplitudes."""
    wave = _grid(160, 4000.0, 7000.0)
    z = 0.1
    template = _template(synthetic_ssp, wave)
    names = eline_amplitude_names(template.observation.spectroscopy.effective_catalog)
    n = wave.size
    flux = np.zeros(n) + 1.0
    ivar = np.ones(n)
    spec, batch, _ = build_spectro_batches([_galaxy(wave, z, flux, ivar)], model=template)[0]
    params = {**_stellar_params(template), **_AMPS}
    out = make_batched_predict(template, spec)(_stack(params, 1), batch)
    got = np.asarray(out["spec_fnu"])[0]

    stellar = {k: v for k, v in params.items() if not k.startswith("eline_amp_")}
    continuum = np.asarray(template.predict_spectrum({**stellar, "redshift": z}))
    ref = continuum + np.asarray(_line_design(template, wave, z) @ _amp_vector(names))
    np.testing.assert_allclose(got, ref, rtol=1e-10)


def test_batched_loglik_and_gradient_match_eline_fitted_likelihood(synthetic_ssp):
    """Parity (a): log-likelihood equals ELineFittedLikelihood; gradient is finite and nonzero."""
    wave = _grid(160, 4000.0, 7500.0)  # Halpha at z=0.1 sits near 7220 A: keep it on the grid
    z = 0.1
    template = _template(synthetic_ssp, wave)
    names = eline_amplitude_names(template.observation.spectroscopy.effective_catalog)
    rng = np.random.default_rng(0)
    flux = rng.normal(0.0, 0.05, wave.size)
    ivar = np.full(wave.size, 1.0 / 0.05**2)
    spec, batch, _ = build_spectro_batches([_galaxy(wave, z, flux, ivar)], model=template)[0]
    loglik = batched_log_likelihood(template, spec)
    params = {**_stellar_params(template), **_AMPS}
    obs = batch.galaxy(0)
    got = float(loglik(params, obs))

    stellar = {k: v for k, v in params.items() if not k.startswith("eline_amp_")}
    m_single = template.predict_spectrum({**stellar, "redshift": z})
    G_eff = _line_design(template, wave, z)
    fitted = ELineFittedLikelihood(
        jnp.asarray(flux),
        jnp.asarray(1.0 / np.sqrt(ivar)),
        design_matrix_builder=lambda _p: G_eff,
        amplitude_names=tuple(names),
    )
    ref = float(fitted.log_prob({"spec_fnu": m_single}, params))
    np.testing.assert_allclose(got, ref, rtol=1e-10)

    grad = jax.grad(lambda p: loglik(p, obs))(params)
    for key in params:
        g = float(grad[key])
        assert np.isfinite(g), key
        assert g != 0.0, key


def _spectra(wave, zs, flux, ivar):
    return [_galaxy(wave, z, flux, ivar) for z in zs]


def test_lines_outside_grid_are_bit_identical_to_lines_off(synthetic_ssp):
    """(b): with every line beyond the grid, the fitted spectrum equals the lines-off one."""
    wave = _grid(160, 4000.0, 7500.0)
    z = 1.5  # every catalog line lands past 7500 A
    on = _template(synthetic_ssp, wave)
    off = _template(synthetic_ssp, wave, eline_mode="off")
    flux = np.ones(wave.size)
    ivar = np.ones(wave.size)
    spec_on, batch_on, _ = build_spectro_batches(_spectra(wave, [z], flux, ivar), model=on)[0]
    spec_off, batch_off, _ = build_spectro_batches(_spectra(wave, [z], flux, ivar), model=off)[0]
    assert np.all(np.asarray(batch_on.line_mask) == 0.0)
    params = {**_stellar_params(on), **_AMPS}
    got = np.asarray(make_batched_predict(on, spec_on)(_stack(params, 1), batch_on)["spec_fnu"])
    ref = np.asarray(make_batched_predict(off, spec_off)(_stack(params, 1), batch_off)["spec_fnu"])
    assert np.any(ref != 0.0)
    assert np.array_equal(got, ref)


def test_one_compile_per_spec_for_predict_and_vmapped_loglik(synthetic_ssp):
    """(c): a second batch of the same spec, with different line masks, triggers no compile."""
    from jax._src import test_util as jtu

    wave = _grid(160, 4000.0, 7500.0)
    model = _template(synthetic_ssp, wave)
    flux = np.ones(wave.size)
    ivar = np.ones(wave.size)
    spec_a, batch_a, _ = build_spectro_batches(
        _spectra(wave, [0.1, 1.5], flux, ivar), model=model
    )[0]
    spec_b, batch_b, _ = build_spectro_batches(
        _spectra(wave, [0.2, 0.3], flux, ivar), model=model
    )[0]
    assert spec_a == spec_b
    assert not np.array_equal(np.asarray(batch_a.line_mask), np.asarray(batch_b.line_mask))

    params_a = _stack({**_stellar_params(model), **_AMPS}, 2)
    params_b = _stack({**_stellar_params(model), **_AMPS}, 2)

    predict = make_batched_predict(model, spec_a)
    jax.block_until_ready(predict(params_a, batch_a)["spec_fnu"])
    with jtu.count_jit_compilation_cache_miss() as counter:
        jax.block_until_ready(predict(params_b, batch_b)["spec_fnu"])
    assert counter() == 0

    loglik = jax.jit(jax.vmap(batched_log_likelihood(model, spec_a), in_axes=(0, 0)))
    jax.block_until_ready(loglik(params_a, batch_a))
    with jtu.count_jit_compilation_cache_miss() as counter:
        jax.block_until_ready(loglik(params_b, batch_b))
    assert counter() == 0


def test_padding_quantum_leaves_loglik_unchanged_with_lines(synthetic_ssp):
    """(d): quantum=None and quantum=64 give the same log-likelihood at rtol 1e-10."""
    wave = _grid(160, 4000.0, 7500.0)
    model = _template(synthetic_ssp, wave)
    rng = np.random.default_rng(1)
    flux = rng.normal(0.0, 0.05, wave.size)
    ivar = np.full(wave.size, 1.0 / 0.05**2)
    params = {**_stellar_params(model), **_AMPS}
    values = []
    for quantum in (None, 64):
        spec, batch, _ = build_spectro_batches(
            _spectra(wave, [0.1], flux, ivar), model=model, quantum=quantum
        )[0]
        values.append(float(batched_log_likelihood(model, spec)(params, batch.galaxy(0))))
    np.testing.assert_allclose(values[1], values[0], rtol=1e-10)


def test_refusals_marginalized_broad_and_line_count(synthetic_ssp):
    """(e): marginalized lines, eline_broad, and a bucket from a different line count are refused."""
    wave = _grid(160, 4000.0, 7500.0)
    marginalized = _template(synthetic_ssp, wave, eline_mode="marginalized")
    with pytest.raises(ValueError, match="marginalizes emission lines"):
        check_template_model(marginalized)

    broad = _template(synthetic_ssp, wave, eline_broad=True)
    with pytest.raises(ValueError, match="eline_broad=False"):
        check_template_model(broad)

    fitted = _template(synthetic_ssp, wave)
    three = _template(synthetic_ssp, wave, lines=("Hbeta", "OIII_5007", "Halpha"))
    flux = np.ones(wave.size)
    ivar = np.ones(wave.size)
    spec_three, _, _ = build_spectro_batches(_spectra(wave, [0.1], flux, ivar), model=three)[0]
    assert spec_three.n_lines == 3
    with pytest.raises(ValueError, match="rebuild the batch with model=template"):
        require_model_spec(fitted, spec_three)


def test_map_recovers_injected_line_amplitudes(synthetic_ssp):
    """(f): MAP recovers each injected amplitude within 10% after 300 Adam steps."""
    wave = _grid(160, 4000.0, 7500.0)
    z = 0.1
    model = _template(synthetic_ssp, wave)
    names = eline_amplitude_names(model.observation.spectroscopy.effective_catalog)
    cont = np.asarray(model.predict_spectrum({"sfh_dpl_log_total_mass": 10.0, "redshift": z}))
    G = np.asarray(_line_design(model, wave, z))
    a_true = np.asarray(_amp_vector(names))
    noise_sigma = 1.0e-17  # about 1 % of the continuum median
    rng = np.random.default_rng(2)
    flux = cont + G @ a_true + rng.normal(0.0, noise_sigma, wave.size)
    ivar = np.full(wave.size, 1.0 / noise_sigma**2)
    batches = build_spectro_batches(_spectra(wave, [z], flux, ivar), model=model)
    out = fit_spectra_map_vmap(model, batches, n_steps=300, learning_rate=0.05, seed=0)
    for nm, truth in zip(names, a_true):
        recovered = float(np.asarray(out["params"][nm])[0])
        print(f"{nm}: injected {truth:.4e} recovered {recovered:.4e}")
        assert abs(recovered - truth) <= 0.1 * abs(truth), nm


def test_float32_gradients_are_finite_and_nonzero(synthetic_ssp):
    """(g): with x64 off, the log-likelihood and its gradients are finite and nonzero."""
    wave = _grid(160, 4000.0, 7500.0)
    z = 0.1
    rng = np.random.default_rng(3)
    flux = rng.normal(0.0, 1.0e-16, wave.size)
    ivar = np.full(wave.size, 1.0 / (1.0e-16) ** 2)
    with jax.enable_x64(False):
        model = _template(synthetic_ssp, wave)
        spec, batch, _ = build_spectro_batches(_spectra(wave, [z], flux, ivar), model=model)[0]
        loglik = batched_log_likelihood(model, spec)
        obs = batch.galaxy(0)
        params = {**_stellar_params(model), **_AMPS}
        value = float(loglik(params, obs))
        grad = jax.grad(lambda p: loglik(p, obs))(params)
        values = {k: float(grad[k]) for k in ("sfh_dpl_log_total_mass", "eline_amp_Halpha")}
    print(f"float32 loglik {value:.6e} gradients {values}")
    assert np.isfinite(value)
    for key, g in values.items():
        assert np.isfinite(g), key
        assert g != 0.0, key
