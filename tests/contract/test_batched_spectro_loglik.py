# SPDX-License-Identifier: BSD-3-Clause
"""Batched spectroscopic log-likelihood and vmapped MAP across galaxies (#2833).

The batched data term must reproduce, galaxy by galaxy, the Gaussian likelihood
of a per-galaxy :class:`SEDModel` on that galaxy's real pixels, and the existing
Fitter likelihood up to a constant. Padded pixels must contribute exactly zero.

Runs on the synthetic narrow SSP (no ``data/ssp_*.h5`` needed).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, SEDModel, Spectroscopy, Uniform
from tengri.inference.batched_spectro import (
    batched_log_likelihood,
    batched_neg_log_posterior,
    fit_spectra_map_vmap,
    init_unbounded_batch,
)
from tengri.inference.fitter import Fitter
from tengri.inference.loss_functions import build_loglikelihood_fn
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches
from tests.contract.test_batched_spectrum_forward import (
    _N_DIAG,
    _R_BANDED,
    _banded,
    _free_z_model,
    _grid,
    _template,
)

pytestmark = pytest.mark.contract

_MASS = "sfh_dpl_log_total_mass"
# (n_pix, lo, hi, z) as in test_batched_spectrum_forward: linear grids, distinct coverage.
_SPECS = [(300, 4000.0, 7000.0, 0.05), (360, 4200.0, 7400.0, 0.1), (420, 4500.0, 7500.0, 0.2)]
_TRUE_MASS = (10.0, 10.3, 9.7)
# Per-pixel noise as a fraction of each galaxy's peak model flux: high S/N.
_REL_NOISE = 1e-3


def _mock_catalog(ssp):
    """Three DESI-like mocks: model at known mass plus seeded Gaussian noise."""
    rng = np.random.default_rng(20260909)
    galaxies = []
    for (n_pix, lo, hi, z), true_mass in zip(_SPECS, _TRUE_MASS):
        wave = _grid(n_pix, lo, hi)
        bm = gaussian_resolution_bands(jnp.asarray(wave), _R_BANDED, _N_DIAG)
        ref = _free_z_model(ssp, wave, {"resolution_matrix": bm})
        clean = np.asarray(ref.predict_spectrum({_MASS: true_mass, "redshift": z}))
        sigma = _REL_NOISE * float(np.max(clean))
        flux = clean + sigma * rng.standard_normal(n_pix)
        galaxies.append(
            {
                "wave": wave,
                "bm": bm,
                "z": z,
                "flux": flux,
                "sigma": np.full(n_pix, sigma),
                "ref": ref,
                "true_mass": true_mass,
            }
        )
    return galaxies


def _record(g) -> GalaxySpectrum:
    return GalaxySpectrum(
        wave=g["wave"],
        flux=g["flux"],
        ivar=1.0 / g["sigma"] ** 2,
        resolution=g["bm"],
        z=g["z"],
    )


@pytest.fixture(scope="module")
def mock(synthetic_ssp):
    return _mock_catalog(synthetic_ssp)


@pytest.fixture(scope="module")
def template(synthetic_ssp, mock):
    """Free-redshift template on the first galaxy's grid; it is only structural."""
    return _free_z_model(synthetic_ssp, mock[0]["wave"], _banded(mock[0]["wave"]))


@pytest.fixture(scope="module")
def bucket(mock):
    """All three galaxies padded into one bucket of 420 pixels."""
    [(spec, batch, index)] = build_spectro_batches([_record(g) for g in mock], quantum=420)
    np.testing.assert_array_equal(index, [0, 1, 2])
    return spec, batch


def _pad_mask(mock, n_max):
    """Boolean ``(B, n_max)`` array, True on padded pixels."""
    mask = np.zeros((len(mock), n_max), dtype=bool)
    for i, g in enumerate(mock):
        mask[i, g["wave"].size :] = True
    return mask


# ── (1) Key test: batched value and gradient equal the per-galaxy reference ─


def test_batched_value_and_grad_match_per_galaxy_reference(template, bucket, mock):
    spec, batch = bucket
    loglik = batched_log_likelihood(template, spec, conserving=True)
    # Offset from the truth so that the gradient is nonzero.
    masses = jnp.asarray([m + 0.05 for m in _TRUE_MASS])
    values, grads = jax.vmap(jax.value_and_grad(loglik))({_MASS: masses}, batch)

    for i, g in enumerate(mock):
        n = g["wave"].size
        flux, sigma = jnp.asarray(g["flux"]), jnp.asarray(g["sigma"])

        def ref(m, g=g, flux=flux, sigma=sigma, n=n):
            pred = g["ref"].predict_spectrum({_MASS: m, "redshift": g["z"]})
            r = (flux - pred[:n]) / sigma
            return -0.5 * jnp.sum(r * r)

        v_ref, g_ref = jax.value_and_grad(ref)(masses[i])
        np.testing.assert_allclose(np.asarray(values[i]), float(v_ref), rtol=1e-9, atol=0.0)
        np.testing.assert_allclose(np.asarray(grads[_MASS][i]), float(g_ref), rtol=1e-9, atol=0.0)


# ── (2) Same galaxy, existing Fitter likelihood: differences agree ─────────


def test_single_galaxy_loglik_differences_match_fitter(synthetic_ssp, template, bucket, mock):
    spec, batch = bucket
    g = mock[0]
    loglik = batched_log_likelihood(template, spec, conserving=True)
    obs = batch.galaxy(0)

    # The existing fit path needs a fixed redshift, so build the per-galaxy model that way.
    fixed_model = _template(
        synthetic_ssp, g["wave"], {"resolution_matrix": g["bm"]}, redshift=Fixed(g["z"])
    )
    fitter = Fitter(
        fixed_model,
        jnp.asarray(g["flux"]),
        jnp.asarray(g["sigma"]),
        data_type="spectroscopy",
        approx=None,
    )
    fitter_loglik = build_loglikelihood_fn(fitter)
    data_args = fitter._data_args

    m0, m1 = _TRUE_MASS[0] - 0.02, _TRUE_MASS[0] + 0.03
    d_fitter = fitter_loglik({_MASS: m1}, data_args) - fitter_loglik({_MASS: m0}, data_args)
    d_batch = loglik({_MASS: m1}, obs) - loglik({_MASS: m0}, obs)
    np.testing.assert_allclose(float(d_batch), float(d_fitter), rtol=1e-9, atol=0.0)


# ── (3) Padding is inert ───────────────────────────────────────────────────


def test_padded_pixels_do_not_change_value_or_gradient(template, bucket, mock):
    spec, batch = bucket
    loglik = batched_log_likelihood(template, spec, conserving=True)
    params = {_MASS: jnp.asarray(list(_TRUE_MASS))}
    pad = _pad_mask(mock, spec.n_max)

    perturbed = dataclasses.replace(
        batch, flux=batch.flux + 1e6 * jnp.asarray(pad, dtype=batch.flux.dtype)
    )
    base_values = jax.vmap(loglik)(params, batch)
    np.testing.assert_array_equal(
        np.asarray(jax.vmap(loglik)(params, perturbed)), np.asarray(base_values)
    )

    def total(flux):
        return jnp.sum(jax.vmap(loglik)(params, dataclasses.replace(batch, flux=flux)))

    grad_flux = np.asarray(jax.grad(total)(batch.flux))
    np.testing.assert_array_equal(grad_flux[pad], 0.0)
    assert np.any(grad_flux[~pad] != 0.0)


# ── (4) Negative log-posterior equals the hand-computed sum ────────────────


def test_neg_log_posterior_equals_minus_loglik_plus_prior(template, bucket):
    spec, batch = bucket
    loglik = batched_log_likelihood(template, spec, conserving=True)
    nlp = batched_neg_log_posterior(template, spec, conserving=True)
    obs = batch.galaxy(1)

    xi = jnp.asarray(0.37)
    theta = template.spec.get_distribution(_MASS).unstandardize(xi)
    expected = -loglik({_MASS: theta}, obs) + 0.5 * xi**2
    np.testing.assert_allclose(
        np.asarray(nlp({_MASS: xi}, obs)), np.asarray(expected), rtol=1e-12, atol=0.0
    )


# ── (5) Batched MAP: descent, catalog order, recovery ──────────────────────


def test_init_matches_fitter_initializer(synthetic_ssp, template, mock):
    """Row i of the batched init equals the Fitter's init for the key of galaxy i."""
    g = mock[0]
    # profile_mass=False keeps log_total_mass latent. The "auto" default profiles it out
    # of a free-redshift spectroscopic fit, which would drop it from the Fitter's init.
    fitter = Fitter(
        template,
        jnp.asarray(g["flux"]),
        jnp.asarray(g["sigma"]),
        data_type="spectroscopy",
        approx=None,
        profile_mass=False,
    )
    key = jax.random.PRNGKey(3)
    batched = init_unbounded_batch(template, 3, key)
    for i, gal_key in enumerate(jax.random.split(key, 3)):
        single = fitter._initialize_unbounded(gal_key)
        np.testing.assert_allclose(
            np.asarray(batched[_MASS][i]), np.asarray(single[_MASS]), rtol=0.0, atol=0.0
        )


def test_map_descends_and_recovers_mass_in_catalog_order(template, mock):
    # Input order [0, 2, 1] gives two buckets: 300 and 360 px at n_max 384 (catalog
    # indices 0 and 2), and 420 px at n_max 512 (index 1). Concatenating the buckets
    # is therefore not sorted, so a missing permutation returns the wrong galaxies.
    order = [0, 2, 1]
    records = [_record(mock[i]) for i in order]
    batches = build_spectro_batches(records, quantum=128)
    assert [spec.n_max for spec, _, _ in batches] == [384, 512]
    bucket_order = np.concatenate([np.asarray(index) for _, _, index in batches])
    assert not np.all(np.diff(bucket_order) > 0)

    result = fit_spectra_map_vmap(template, batches, conserving=True, n_steps=150)

    assert len(result["loss_history"]) == 2
    assert result["loss_history"][0].shape == (150, 2)
    assert result["loss_history"][1].shape == (150, 1)

    # Initial nlp per record position, read from the first step of each bucket's trace.
    init_nlp = np.empty(3)
    for (_, _, index), trace in zip(batches, result["loss_history"]):
        init_nlp[np.asarray(index)] = np.asarray(trace[0])
    final_nlp = np.asarray(result["nlp"])
    assert np.all(final_nlp < init_nlp), (final_nlp, init_nlp)

    recovered = np.asarray(result["params"][_MASS])
    truth = np.asarray([mock[i]["true_mass"] for i in order])
    assert recovered.shape == (3,)
    np.testing.assert_array_less(np.abs(recovered - truth), 0.1)


# ── (6) Stochastic SFH is refused ──────────────────────────────────────────


def test_stochastic_sfh_raises_not_implemented(synthetic_ssp, mock):
    wave = mock[0]["wave"]
    stochastic = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=Observation(
            spectroscopy=Spectroscopy(
                wave_obs=jnp.asarray(wave),
                resample="conserving",
                resolution_matrix=mock[0]["bm"],
            )
        ),
        sfh={
            "type": ["dpl", "field"],
            "all_params": Fixed(DEFAULT),
            _MASS: Uniform(8.0, 12.0),
        },
        neb={"type": "none"},
        redshift=Uniform(0.0, 0.5),
    )
    assert stochastic.spec.stochastic
    [(spec, batch, index)] = build_spectro_batches([_record(mock[0])])

    with pytest.raises(NotImplementedError, match="stochastic"):
        batched_log_likelihood(stochastic, spec, conserving=True)
    with pytest.raises(NotImplementedError, match="stochastic"):
        fit_spectra_map_vmap(stochastic, [(spec, batch, index)], conserving=True, n_steps=2)
