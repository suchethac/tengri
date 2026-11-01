# SPDX-License-Identifier: BSD-3-Clause
"""Banded pixel covariance on the batched spectroscopic path (#3133).

A galaxy that carries a banded covariance is whitened by its banded Cholesky factor,
built on the host with that galaxy's pixel mask. The batched value must reproduce
the per-galaxy dense-covariance value galaxy by galaxy, with masked pixels equal to
deleted pixels, padding exact, one compile per spec, and a finite nonzero float32
gradient.

Runs on the synthetic narrow SSP (no ``data/ssp_*.h5`` needed).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import test_util as jtu

from tengri.inference.batched_spectro import batched_log_likelihood
from tengri.inference.likelihoods.marginalized import CalibrationMarginalizedLikelihood
from tengri.observation.banded import BandedMatrix, gaussian_resolution_bands
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches
from tengri.observation.spectroscopy import Spectroscopy
from tests.contract.test_batched_cal_marginal import _calibrated_template
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
_F32_MASS = (10.0, 10.3, 9.7)
_HALF_BAND = 3
_REL_NOISE = 1e-3
_MASKED = ((40, 55), (120, 130), (200, 215))


def _band_covariance(seed: int, n: int, sigma: float, p: int = _HALF_BAND) -> np.ndarray:
    """Dense SPD covariance of half-bandwidth p; the correlations differ per seed."""
    rng = np.random.default_rng(seed)
    corr = np.eye(n)
    for lag in range(1, p + 1):
        c = rng.uniform(0.05, 0.18) / lag
        corr += c * (np.eye(n, k=lag) + np.eye(n, k=-lag))
    scale = sigma * (1.0 + 0.2 * rng.random(n))
    return corr * scale[:, None] * scale[None, :]


def _to_banded(a: np.ndarray, p: int = _HALF_BAND) -> BandedMatrix:
    """Symmetric diagonal-offset storage, data[k, i] = a[i, i + offsets[k]]."""
    n = a.shape[0]
    offsets = np.arange(-p, p + 1)
    data = np.zeros((offsets.size, n))
    for k, off in enumerate(offsets):
        rows = np.arange(max(0, -off), min(n, n - off))
        data[k, rows] = a[rows, rows + off]
    return BandedMatrix(offsets=offsets, data=jnp.asarray(data))


def _mock_catalog(ssp, masses=_TRUE_MASS):
    """Three mocks with their own banded covariance and pixel mask."""
    galaxies = []
    for i, ((n_pix, lo, hi, z), mass) in enumerate(zip(_SPECS, masses)):
        wave = _grid(n_pix, lo, hi)
        bm = gaussian_resolution_bands(jnp.asarray(wave), _R_BANDED, _N_DIAG)
        ref = _free_z_model(ssp, wave, {"resolution_matrix": bm})
        clean = np.asarray(ref.predict_spectrum({_MASS: mass, "redshift": z}))
        sigma = _REL_NOISE * float(np.max(clean))
        cov = _band_covariance(100 + i, n_pix, sigma)
        draw = np.linalg.cholesky(cov) @ np.random.default_rng(7 + i).standard_normal(n_pix)
        keep = np.ones(n_pix, dtype=bool)
        lo_m, hi_m = _MASKED[i]
        keep[lo_m:hi_m] = False
        galaxies.append(
            {
                "wave": wave,
                "bm": bm,
                "z": z,
                "flux": clean + draw,
                "cov": cov,
                "keep": keep,
                "ref": ref,
                "mass": mass,
            }
        )
    return galaxies


def _record(g) -> GalaxySpectrum:
    return GalaxySpectrum(
        wave=g["wave"],
        flux=g["flux"],
        ivar=1.0 / np.diag(g["cov"]),
        resolution=g["bm"],
        z=g["z"],
        mask=g["keep"],
        covariance=_to_banded(g["cov"]),
    )


@pytest.fixture(scope="module")
def mock(synthetic_ssp):
    return _mock_catalog(synthetic_ssp)


@pytest.fixture(scope="module")
def template(synthetic_ssp, mock):
    return _calibrated_template(
        synthetic_ssp, mock[0]["wave"], _banded(mock[0]["wave"]), calibration_order=0
    )


def _batched_values(template, mock, quantum, *, marginalize):
    values = {}
    for spec, batch, index in build_spectro_batches(
        [_record(g) for g in mock], quantum=quantum, model=template
    ):
        loglik = batched_log_likelihood(template, spec, calibration_marginalize=marginalize)
        mass = jnp.asarray([mock[int(i)]["mass"] + 0.05 for i in index])
        for pos, v in zip(index, np.asarray(jax.vmap(loglik)({_MASS: mass}, batch))):
            values[int(pos)] = float(v)
    return values


def _kept_model(g, mass):
    n = g["wave"].size
    return np.asarray(g["ref"].predict_spectrum({_MASS: mass, "redshift": g["z"]}))[:n]


# ── (f) Batched equals the per-galaxy dense value, masked pixels deleted ──


def test_batched_gaussian_term_equals_per_galaxy_dense_covariance(template, mock):
    values = _batched_values(template, mock, 420, marginalize=False)
    for i, g in enumerate(mock):
        keep = g["keep"]
        diff = (g["flux"] - _kept_model(g, g["mass"] + 0.05))[keep]
        dense = g["cov"][np.ix_(keep, keep)]
        expected = -0.5 * diff @ np.linalg.solve(dense, diff)
        np.testing.assert_allclose(values[i], expected, rtol=1e-10)


def test_batched_calibration_marginal_equals_per_galaxy_factor(template, mock):
    values = _batched_values(template, mock, 420, marginalize=True)
    for i, g in enumerate(mock):
        keep = g["keep"]
        spectroscopy = Spectroscopy(
            wave_obs=g["wave"][keep], covariance=g["cov"][np.ix_(keep, keep)]
        )
        likelihood = CalibrationMarginalizedLikelihood(
            jnp.asarray(g["flux"][keep]),
            jnp.asarray(np.sqrt(np.diag(g["cov"]))[keep]),
            wavelength=jnp.asarray(g["wave"][keep]),
            n_poly=3,
            prior_sigma=1.0,
            chol_data=spectroscopy.chol_data,
            chol_offsets=spectroscopy.chol_offsets,
        )
        model = jnp.asarray(_kept_model(g, g["mass"] + 0.05)[keep])
        np.testing.assert_allclose(
            values[i], float(likelihood.log_prob({"spec_fnu": model})), rtol=1e-10
        )


def test_covariance_value_and_gradient_are_finite_and_nonzero(template, mock):
    [(spec, batch, _)] = build_spectro_batches(
        [_record(g) for g in mock], quantum=420, model=template
    )
    loglik = batched_log_likelihood(template, spec)
    mass = jnp.asarray([g["mass"] + 0.05 for g in mock])
    values, grads = jax.vmap(jax.value_and_grad(loglik))({_MASS: mass}, batch)
    # grad-assert: finite-only — the value is a log-likelihood, not a gradient
    assert np.all(np.isfinite(np.asarray(values)))
    assert np.all(np.isfinite(np.asarray(grads[_MASS])))
    assert np.all(np.asarray(grads[_MASS]) != 0.0)


# ── Padding is exact ──────────────────────────────────────────────────────


@pytest.mark.parametrize("marginalize", [False, True])
def test_padding_quantum_does_not_change_the_value(template, mock, marginalize):
    padded = _batched_values(template, mock, 64, marginalize=marginalize)
    exact = _batched_values(template, mock, None, marginalize=marginalize)
    for i in range(len(mock)):
        np.testing.assert_allclose(padded[i], exact[i], rtol=1e-12)


# ── One compile per spec across galaxies ──────────────────────────────────


def test_second_batch_with_new_covariance_does_not_recompile(template, mock):
    [(spec, batch, _)] = build_spectro_batches(
        [_record(g) for g in mock], quantum=420, model=template
    )
    loglik = jax.jit(jax.vmap(batched_log_likelihood(template, spec)))
    params = {_MASS: jnp.asarray([g["mass"] for g in mock])}
    loglik(params, batch).block_until_ready()

    other_records = [
        _record({**g, "cov": _band_covariance(900 + i, g["wave"].size, 1e-30)})
        for i, g in enumerate(mock)
    ]
    [(_, other, _)] = build_spectro_batches(other_records, quantum=420, model=template)
    with jtu.count_jit_compilation_cache_miss() as count:
        loglik(params, other).block_until_ready()
    assert count() == 0


# ── Offsets mismatch is refused ───────────────────────────────────────────


def test_galaxies_with_different_covariance_offsets_are_refused(template, mock):
    records = [_record(g) for g in mock]
    narrow = _to_banded(mock[1]["cov"], p=2)
    records[1] = records[1]._replace(covariance=narrow)
    with pytest.raises(ValueError, match=r"\(-2, -1, 0, 1, 2\).*\(-3, -2, -1, 0, 1, 2, 3\)"):
        build_spectro_batches(records, quantum=420, model=template)


# ── (g) Float32: finite value, finite nonzero gradient ────────────────────


def test_float32_value_and_gradient_are_finite_and_nonzero(synthetic_ssp_physical):
    """Finite value and nonzero gradient in float32 at galaxy-scale log10 M = 10.

    The SSP is physically normalized (see ``synthetic_ssp_physical``), so the float32
    forward is in range at a realistic mass (#3149).
    """
    galaxies = _mock_catalog(synthetic_ssp_physical, masses=_F32_MASS)
    template32 = _calibrated_template(
        synthetic_ssp_physical,
        galaxies[0]["wave"],
        _banded(galaxies[0]["wave"]),
        calibration_order=0,
    )
    [(spec, batch, _)] = build_spectro_batches(
        [_record(g) for g in galaxies], quantum=420, model=template32
    )
    with jax.enable_x64(False):
        batch32 = jax.tree_util.tree_map(
            lambda a: (
                jnp.asarray(a, dtype=jnp.float32) if jnp.issubdtype(a.dtype, jnp.floating) else a
            ),
            batch,
        )
        mass = jnp.asarray([m + 0.05 for m in _F32_MASS], dtype=jnp.float32)
        for marginalize in (False, True):
            loglik = batched_log_likelihood(template32, spec, calibration_marginalize=marginalize)
            values, grads = jax.vmap(jax.value_and_grad(loglik))({_MASS: mass}, batch32)
            assert values.dtype == jnp.float32
            # grad-assert: finite-only — the value is a log-likelihood, not a gradient
            assert np.all(np.isfinite(np.asarray(values)))
            assert np.all(np.isfinite(np.asarray(grads[_MASS])))
            assert np.any(np.asarray(grads[_MASS]) != 0.0)


# ── Line marginals whiten with the same factor ────────────────────────────


@pytest.mark.parametrize("calibrated", [False, True])
def test_batched_line_marginals_equal_per_galaxy_classes_with_the_factor(
    synthetic_ssp, calibrated
):
    from tengri.inference.likelihoods.marginalized import (
        CalibrationELineMarginalizedLikelihood,
        ELineMarginalizedLikelihood,
    )
    from tests.contract.test_batched_eline_marginal import (
        _MARG_PRIOR_SIGMA,
        _MARG_R,
        _MARG_WAVE,
        _marg_data,
        _marg_line_design,
        _marg_template,
    )

    template = _marg_template(synthetic_ssp)
    flux, sigma = _marg_data()
    cov = _band_covariance(5, _MARG_WAVE.size, float(sigma[0]))
    galaxy = GalaxySpectrum(
        wave=_MARG_WAVE,
        flux=flux,
        ivar=1.0 / np.diag(cov),
        z=0.1,
        resolution=_MARG_R,
        covariance=_to_banded(cov),
    )
    spec, batch, _ = build_spectro_batches([galaxy], model=template)[0]
    params = {_MASS: 10.0}
    loglik = batched_log_likelihood(
        template, spec, calibration_marginalize=calibrated, cal_n_poly=2, cal_prior_sigma=1.0
    )
    value = float(loglik(params, batch.galaxy(0)))

    spectroscopy = Spectroscopy(wave_obs=_MARG_WAVE, covariance=cov)
    model = template.predict_spectrum({**params, "redshift": 0.1})
    design = _marg_line_design(template, 0.1)
    common = {
        "fnu_obs": jnp.asarray(flux),
        "fnu_err": jnp.asarray(np.sqrt(np.diag(cov))),
        "chol_data": spectroscopy.chol_data,
        "chol_offsets": spectroscopy.chol_offsets,
    }
    if calibrated:
        reference = CalibrationELineMarginalizedLikelihood(
            **common,
            wavelength=jnp.asarray(_MARG_WAVE),
            design_matrix_builder=lambda _p: design,
            n_poly=2,
            prior_sigma=1.0,
            eline_prior_sigma=_MARG_PRIOR_SIGMA,
        )
    else:
        reference = ELineMarginalizedLikelihood(
            **common, design_matrix=design, prior_variance=_MARG_PRIOR_SIGMA**2
        )
    expected = float(reference.log_prob({"spec_fnu": model}, params={}))
    np.testing.assert_allclose(value, expected, rtol=1e-10)
