# SPDX-License-Identifier: BSD-3-Clause
"""Guards on the batched spectroscopic path: template, spec and data validation (#2833).

Covers the rules that keep a batch from silently disagreeing with its template:
the resample mode resolved from the template, the LSF bin count and photometry
width checked against it, the template settings the batched likelihood cannot
evaluate, non-finite pixels, and invariance of the chunked MAP to its chunk size.
"""

from __future__ import annotations

import dataclasses
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Spectroscopy,
    Uniform,
    WavePrecomp,
)
from tengri.forward.batched_spectrum import (
    check_template_model,
    make_batched_predict,
    predict_batched_observables,
)
from tengri.inference.batched_spectro import (
    batched_log_likelihood,
    batched_neg_log_posterior,
    fit_spectra_map_vmap,
)
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.noise_model import NoiseModel
from tengri.observation.photometry import FilterCurve
from tengri.observation.spectroscopy import resolve_resample_mode
from tests.contract.test_batched_spectro_loglik import _mock_catalog, _record
from tests.contract.test_batched_spectrum_forward import (
    _N_DIAG,
    _R_BANDED,
    _banded,
    _free_z_model,
    _grid,
)

pytestmark = pytest.mark.contract

_MASS = "sfh_dpl_log_total_mass"


def _model(
    ssp,
    wave,
    *,
    resample="conserving",
    lsf_n_bins=16,
    obs_kw=None,
    spectro_kw=None,
    redshift=None,
    photometry=None,
    kw=None,
):
    """Free-redshift template with a 1000 R Gaussian LSF unless overridden."""
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(wave),
        resample=resample,
        lsf_n_bins=lsf_n_bins,
        **(spectro_kw or {"resolution": 1000.0}),
    )
    observation = Observation(spectroscopy=spectroscopy, photometry=photometry, **(obs_kw or {}))
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            _MASS: Uniform(8.0, 12.0),
        },
        neb={"type": "none"},
        redshift=redshift if redshift is not None else Uniform(0.0, 0.5),
        **(kw or {}),
    )


def _record_for(wave, *, resolution=1000.0, z=0.1, **phot):
    n = np.asarray(wave).size
    return GalaxySpectrum(
        wave=wave, flux=np.ones(n), ivar=np.ones(n), resolution=resolution, z=z, **phot
    )


# ── Resample mode is the template's, per galaxy ───────────────────────────


@pytest.mark.parametrize("mode", ["point", "conserving", "auto"])
def test_resample_mode_is_resolved_from_the_template(synthetic_ssp, mode):
    wave = _grid(300, 4000.0, 7000.0)
    template = _model(synthetic_ssp, wave, resample=mode)
    [(spec, _, _)] = build_spectro_batches([_record_for(wave)], model=template)
    expected = resolve_resample_mode(mode, wave, template.wavelengths, 0.1)
    assert spec.conserving is expected


def test_forward_entry_points_take_no_conserving_keyword(synthetic_ssp):
    wave = _grid(300, 4000.0, 7000.0)
    template = _model(synthetic_ssp, wave)
    for fn in (
        predict_batched_observables,
        make_batched_predict,
        batched_log_likelihood,
        batched_neg_log_posterior,
    ):
        assert "conserving" not in inspect.signature(fn).parameters, fn.__name__


def test_unresolved_bucket_is_refused_by_the_likelihood(synthetic_ssp):
    wave = _grid(300, 4000.0, 7000.0)
    template = _model(synthetic_ssp, wave)
    [(spec, _, _)] = build_spectro_batches([_record_for(wave)])
    assert spec.conserving is None
    with pytest.raises(ValueError, match="model=template"):
        batched_log_likelihood(template, spec)


# ── LSF bins and photometry width are the template's ──────────────────────


def test_lsf_bins_come_from_the_template_by_default(synthetic_ssp):
    wave = _grid(300, 4000.0, 7000.0)
    template = _model(synthetic_ssp, wave, lsf_n_bins=4)
    [(spec, _, _)] = build_spectro_batches([_record_for(wave)], model=template)
    assert spec.lsf_n_bins == 4
    with pytest.raises(ValueError, match="lsf_n_bins=16 disagrees"):
        build_spectro_batches([_record_for(wave)], model=template, lsf_n_bins=16)


def test_forward_refuses_a_bucket_with_other_lsf_bins(synthetic_ssp):
    wave = _grid(300, 4000.0, 7000.0)
    template = _model(synthetic_ssp, wave, lsf_n_bins=4)
    [(spec, batch, _)] = build_spectro_batches([_record_for(wave)])  # default 16 bins
    spec = dataclasses.replace(spec, conserving=True)
    with pytest.raises(ValueError, match="lsf_n_bins"):
        predict_batched_observables(template, {}, batch.galaxy(0), spec)


def _phot_template(ssp, wave, n_filters):
    filters = tuple(
        FilterCurve(
            wave=jnp.linspace(c * 0.84, c * 1.16, 40),
            trans=jnp.sin(jnp.linspace(0.0, np.pi, 40)) * 0.6,
            name=f"b{k}",
        )
        for k, c in enumerate(np.linspace(4800.0, 9000.0, n_filters))
    )
    return _model(ssp, wave, photometry=Photometry(filters=filters))


@pytest.mark.parametrize("n_data", [1, 2, 5])
def test_photometry_filter_count_mismatch_is_refused(synthetic_ssp, n_data):
    wave = _grid(300, 4000.0, 7000.0)
    template = _phot_template(synthetic_ssp, wave, 4)
    rec = _record_for(
        wave,
        phot_flux=np.ones(n_data) * 1e-30,
        phot_err=np.ones(n_data) * 1e-31,
    )
    [(spec, _, _)] = build_spectro_batches([rec], model=template)
    assert spec.n_filt == n_data
    with pytest.raises(ValueError, match=f"{n_data} photometric filters but the template has 4"):
        batched_log_likelihood(template, spec)


def test_photometry_bucket_matches_the_template_filter_count(synthetic_ssp):
    wave = _grid(300, 4000.0, 7000.0)
    template = _phot_template(synthetic_ssp, wave, 4)
    rec = _record_for(wave, phot_flux=np.ones(4) * 1e-30, phot_err=np.ones(4) * 1e-31)
    [(spec, batch, _)] = build_spectro_batches([rec], model=template)
    loglik = batched_log_likelihood(template, spec)
    assert np.isfinite(float(loglik({_MASS: jnp.asarray(10.0)}, batch.galaxy(0))))


# ── Template settings the batched likelihood cannot evaluate ─────────────


def _refusal_cases(ssp):
    wave = _grid(100, 4000.0, 7000.0)
    lines = LineFluxData(
        names=("Ha",),
        fluxes=jnp.array([1e-16]),
        errors=jnp.array([1e-17]),
        wavelengths=jnp.array([6564.61]),
    )
    return {
        "catalog_z_range": lambda: _model(
            ssp,
            wave,
            redshift=Uniform(0.05, 0.5),
            spectro_kw={"resolution": 1000.0},
            kw={"approx": WavePrecomp(catalog_z_range=(0.05, 0.5))},
        ),
        "covariance": lambda: _model(
            ssp, wave, spectro_kw={"resolution": 1000.0, "covariance": jnp.eye(100) * 1e-3}
        ),
        "student_t_noise": lambda: _model(
            ssp, wave, obs_kw={"noise": NoiseModel(student_t_dof=4.0)}
        ),
        "calibration_floor": lambda: _model(
            ssp, wave, obs_kw={"noise": NoiseModel(calibration_floor=0.05)}
        ),
        "line_fluxes": lambda: _model(ssp, wave, obs_kw={"line_fluxes": lines}),
    }


@pytest.mark.parametrize(
    "case",
    ["catalog_z_range", "covariance", "student_t_noise", "calibration_floor", "line_fluxes"],
)
def test_template_with_unsupported_setting_is_refused(synthetic_ssp, case):
    template = _refusal_cases(synthetic_ssp)[case]()
    with pytest.raises(ValueError):
        check_template_model(template)


def test_plain_free_redshift_template_is_accepted(synthetic_ssp):
    wave = _grid(100, 4000.0, 7000.0)
    check_template_model(_model(synthetic_ssp, wave, obs_kw={"noise": NoiseModel()}))


# ── Non-finite data is excluded, and the gradient stays finite ────────────


@pytest.mark.parametrize("bad", ["inf_ivar", "nan_flux", "nan_ivar"])
def test_non_finite_pixel_is_excluded_with_finite_value_and_gradient(synthetic_ssp, bad):
    n, idx = 300, 40
    wave = _grid(n, 4000.0, 7000.0)
    bm = gaussian_resolution_bands(jnp.asarray(wave), _R_BANDED, _N_DIAG)
    template = _free_z_model(synthetic_ssp, wave, _banded(wave))
    truth = np.asarray(template.predict_spectrum({_MASS: 10.0, "redshift": 0.1}))
    ivar = np.full(n, 1.0 / (0.01 * truth.max()) ** 2)
    flux = truth * 1.01

    bad_flux, bad_ivar = flux.copy(), ivar.copy()
    if bad == "inf_ivar":
        bad_ivar[idx] = np.inf
    elif bad == "nan_flux":
        bad_flux[idx] = np.nan
    else:
        bad_ivar[idx] = np.nan
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    ref_flux, ref_ivar = flux.copy(), ivar.copy()
    ref_ivar[idx] = 0.0

    def nlp_and_grad(f, iv, m=None):
        rec = GalaxySpectrum(wave=wave, flux=f, ivar=iv, resolution=bm, z=0.1, mask=m)
        [(spec, batch, _)] = build_spectro_batches([rec], model=template)
        nlp = batched_neg_log_posterior(template, spec)
        xi = {_MASS: jnp.asarray(0.1)}
        return jax.value_and_grad(nlp)(xi, batch.galaxy(0))

    v_bad, g_bad = nlp_and_grad(bad_flux, bad_ivar)
    v_ref, g_ref = nlp_and_grad(ref_flux, ref_ivar)
    assert np.isfinite(float(v_bad))
    assert np.isfinite(float(g_bad[_MASS]))
    np.testing.assert_allclose(float(v_bad), float(v_ref), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(float(g_bad[_MASS]), float(g_ref[_MASS]), rtol=1e-10, atol=0.0)


# ── Chunked MAP: the chunk size bounds memory, it does not change the fit ──


def test_chunked_map_matches_the_unchunked_fit(synthetic_ssp):
    mock = _mock_catalog(synthetic_ssp)
    template = _free_z_model(synthetic_ssp, mock[0]["wave"], _banded(mock[0]["wave"]))
    batches = build_spectro_batches([_record(g) for g in mock], quantum=420, model=template)
    whole = fit_spectra_map_vmap(template, batches, n_steps=25)
    chunked = fit_spectra_map_vmap(template, batches, n_steps=25, chunk_size=1)
    np.testing.assert_allclose(
        np.asarray(chunked["params"][_MASS]),
        np.asarray(whole["params"][_MASS]),
        rtol=1e-9,
        atol=0.0,
    )
    np.testing.assert_allclose(np.asarray(chunked["nlp"]), np.asarray(whole["nlp"]), rtol=1e-9)
    assert chunked["loss_history"][0].shape == whole["loss_history"][0].shape


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_chunk_size_must_be_a_positive_integer(synthetic_ssp, bad):
    mock = _mock_catalog(synthetic_ssp)
    template = _free_z_model(synthetic_ssp, mock[0]["wave"], _banded(mock[0]["wave"]))
    batches = build_spectro_batches([_record(mock[0])], model=template)
    with pytest.raises(ValueError, match="chunk_size"):
        fit_spectra_map_vmap(template, batches, n_steps=2, chunk_size=bad)


def test_chunk_size_splits_the_bucket_into_vmapped_chunks(synthetic_ssp, monkeypatch):
    """chunk_size must reach the vmap: three galaxies in chunks of one run three calls."""
    import tengri.inference.batched_spectro as module

    mock = _mock_catalog(synthetic_ssp)
    template = _free_z_model(synthetic_ssp, mock[0]["wave"], _banded(mock[0]["wave"]))
    batches = build_spectro_batches([_record(g) for g in mock], quantum=420, model=template)
    calls = []
    original = module._fit_chunk

    def counting(*args, **kwargs):
        calls.append(int(args[1].z.shape[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_fit_chunk", counting)
    fit_spectra_map_vmap(template, batches, n_steps=2, chunk_size=1)
    assert calls == [1, 1, 1]
    calls.clear()
    fit_spectra_map_vmap(template, batches, n_steps=2)
    assert calls == [3]
