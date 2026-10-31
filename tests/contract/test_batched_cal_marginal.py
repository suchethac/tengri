# SPDX-License-Identifier: BSD-3-Clause
"""Analytic calibration marginalization on the batched spectroscopic path (#3116).

With ``calibration_marginalize=True`` the batched spectral term is the marginal
likelihood of a Chebyshev calibration polynomial, and it must reproduce the
per-galaxy :class:`CalibrationMarginalizedLikelihood` galaxy by galaxy. Padding,
the compile cache and float32 behave as they do on the plain Gaussian path.

Runs on the synthetic narrow SSP (no ``data/ssp_*.h5`` needed).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import test_util as jtu

from tengri import DEFAULT, Fixed, Observation, SEDModel, Spectroscopy, Uniform
from tengri.config.exceptions import ConfigError
from tengri.inference.batched_spectro import (
    batched_log_likelihood,
    fit_spectra_map_vmap,
)
from tengri.inference.likelihoods.marginalized import CalibrationMarginalizedLikelihood
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches
from tests.contract.test_batched_spectrum_forward import (
    _N_DIAG,
    _R_BANDED,
    _banded,
    _free_z_model,
    _grid,
)

pytestmark = pytest.mark.contract

_MASS = "sfh_dpl_log_total_mass"
_SPECS = [(300, 4000.0, 7000.0, 0.05), (360, 4200.0, 7400.0, 0.1), (420, 4500.0, 7500.0, 0.2)]
_TRUE_MASS = (10.0, 10.3, 9.7)
_REL_NOISE = 1e-3
_N_POLY = 3
_TILT = 0.05


def _mock_catalog(ssp, tilt: float = 0.0):
    """Three mocks at known mass; ``tilt`` multiplies the model by ``1 + tilt * x``."""
    rng = np.random.default_rng(20260909)
    galaxies = []
    for (n_pix, lo, hi, z), true_mass in zip(_SPECS, _TRUE_MASS):
        wave = _grid(n_pix, lo, hi)
        bm = gaussian_resolution_bands(jnp.asarray(wave), _R_BANDED, _N_DIAG)
        ref = _free_z_model(ssp, wave, {"resolution_matrix": bm})
        clean = np.asarray(ref.predict_spectrum({_MASS: true_mass, "redshift": z}))
        sigma = _REL_NOISE * float(np.max(clean))
        flux = clean + sigma * rng.standard_normal(n_pix)
        x = 2.0 * (wave - wave.min()) / (wave.max() - wave.min()) - 1.0
        flux = flux * (1.0 + tilt * x)
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


def _calibrated_template(ssp, wave, resolution, *, calibration_order: int):
    """Free-redshift template whose spectroscopy carries ``calibration_order``."""
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(wave),
        resample="conserving",
        calibration_order=calibration_order,
        **resolution,
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=spectroscopy),
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            _MASS: Uniform(8.0, 12.0),
        },
        neb={"type": "none"},
        redshift=Uniform(0.0, 0.5),
    )


@pytest.fixture(scope="module")
def mock(synthetic_ssp):
    return _mock_catalog(synthetic_ssp)


@pytest.fixture(scope="module")
def tilted_mock(synthetic_ssp):
    return _mock_catalog(synthetic_ssp, tilt=_TILT)


@pytest.fixture(scope="module")
def template(synthetic_ssp, mock):
    return _calibrated_template(
        synthetic_ssp, mock[0]["wave"], _banded(mock[0]["wave"]), calibration_order=0
    )


@pytest.fixture(scope="module")
def bucket(mock, template):
    [(spec, batch, index)] = build_spectro_batches(
        [_record(g) for g in mock], quantum=420, model=template
    )
    np.testing.assert_array_equal(index, [0, 1, 2])
    return spec, batch


# ── (a) Single galaxy equals the per-galaxy calibration-marginal likelihood ─


def test_single_galaxy_matches_calibration_marginalized_likelihood(template, mock):
    [(spec, batch, _)] = build_spectro_batches([_record(mock[0])], quantum=None, model=template)
    loglik = batched_log_likelihood(
        template, spec, calibration_marginalize=True, cal_n_poly=_N_POLY, cal_prior_sigma=1.0
    )
    g = mock[0]
    n = g["wave"].size
    obs = batch.galaxy(0)

    def reference(mass):
        m_single = g["ref"].predict_spectrum({_MASS: mass, "redshift": g["z"]})[:n]
        lik = CalibrationMarginalizedLikelihood(
            jnp.asarray(g["flux"]),
            jnp.asarray(g["sigma"]),
            wavelength=jnp.asarray(g["wave"]),
            n_poly=_N_POLY,
            prior_sigma=1.0,
        )
        return lik.log_prob({"spec_fnu": m_single})

    mass = jnp.asarray(_TRUE_MASS[0] + 0.05)
    value = loglik({_MASS: mass}, obs)
    np.testing.assert_allclose(np.asarray(value), np.asarray(reference(mass)), rtol=1e-10)

    grad = jax.grad(lambda m: loglik({_MASS: m}, obs))(mass)
    grad_ref = jax.grad(reference)(mass)
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0
    np.testing.assert_allclose(np.asarray(grad), np.asarray(grad_ref), rtol=1e-8)


# ── (b) Padding does not change the marginal value ─────────────────────────


def test_padding_quantum_does_not_change_marginal_value(template, mock):
    params = {_MASS: jnp.asarray(list(_TRUE_MASS))}
    records = [_record(g) for g in mock]

    values = {}
    for quantum in (64, None):
        for spec, batch, index in build_spectro_batches(records, quantum=quantum, model=template):
            loglik = batched_log_likelihood(template, spec, calibration_marginalize=True)
            vals = np.asarray(jax.vmap(loglik)({_MASS: params[_MASS][np.asarray(index)]}, batch))
            for pos, value in zip(np.asarray(index), vals):
                values.setdefault(quantum, {})[int(pos)] = float(value)

    for pos in range(len(mock)):
        np.testing.assert_allclose(values[64][pos], values[None][pos], rtol=1e-10)


# ── (c) One compile per spec across two batches with different masks ───────


def test_second_batch_with_new_mask_does_not_recompile(template, mock):
    records = [_record(g) for g in mock]
    [(spec, batch, _)] = build_spectro_batches(records, quantum=420, model=template)
    loglik = jax.jit(
        jax.vmap(batched_log_likelihood(template, spec, calibration_marginalize=True))
    )
    params = {_MASS: jnp.asarray(list(_TRUE_MASS))}
    loglik(params, batch).block_until_ready()

    new_mask = np.asarray(batch.pix_mask).copy()
    new_mask[:, 10:20] = 0.0
    other = dataclasses.replace(batch, pix_mask=jnp.asarray(new_mask))
    with jtu.count_jit_compilation_cache_miss() as count:
        loglik(params, other).block_until_ready()
    assert count() == 0


# ── (d) Float32 mode: finite value and finite, nonzero gradient ────────────


def test_float32_value_and_gradient_are_finite_and_nonzero(template, bucket):
    spec, batch = bucket
    with jax.enable_x64(False):
        batch32 = jax.tree_util.tree_map(
            lambda a: (
                jnp.asarray(a, dtype=jnp.float32) if jnp.issubdtype(a.dtype, jnp.floating) else a
            ),
            batch,
        )
        loglik = batched_log_likelihood(template, spec, calibration_marginalize=True)
        mass = jnp.asarray([_TRUE_MASS[0] + 0.05] * 3, dtype=jnp.float32)
        values, grads = jax.vmap(jax.value_and_grad(loglik))({_MASS: mass}, batch32)
        assert values.dtype == jnp.float32
        assert np.all(np.isfinite(np.asarray(values)))
        assert np.all(np.isfinite(np.asarray(grads[_MASS])))
        assert np.any(np.asarray(grads[_MASS]) != 0.0)


# ── (e) Explicit calibration polynomial is refused ─────────────────────────


def test_calibration_order_above_zero_is_refused(synthetic_ssp, mock):
    wave = mock[0]["wave"]
    model = _calibrated_template(synthetic_ssp, wave, _banded(wave), calibration_order=1)
    [(spec, _, _)] = build_spectro_batches([_record(mock[0])], quantum=None, model=model)
    with pytest.raises(ConfigError, match="double-count"):
        batched_log_likelihood(model, spec, calibration_marginalize=True)


# ── (f) MAP with marginalization absorbs a calibration tilt ────────────────


def _map_masses(model, galaxies, *, marginalize: bool, n_steps: int = 400):
    batches = build_spectro_batches([_record(g) for g in galaxies], quantum=420, model=model)
    result = fit_spectra_map_vmap(
        model, batches, n_steps=n_steps, calibration_marginalize=marginalize
    )
    return np.asarray(result["params"][_MASS])


def test_map_with_marginalization_absorbs_calibration_tilt(synthetic_ssp, mock, tilted_mock):
    wave = mock[0]["wave"]
    model = _calibrated_template(synthetic_ssp, wave, _banded(wave), calibration_order=0)
    truth = np.asarray(_TRUE_MASS)

    clean = _map_masses(model, mock, marginalize=False)
    marg = _map_masses(model, tilted_mock, marginalize=True)
    plain = _map_masses(model, tilted_mock, marginalize=False)

    bias_marg = np.max(np.abs(marg - clean))
    bias_plain = np.max(np.abs(plain - clean))
    assert bias_marg < 0.02, (marg, clean)
    assert bias_plain > bias_marg, (plain, marg)
    np.testing.assert_array_less(np.abs(marg - truth), 0.1)
