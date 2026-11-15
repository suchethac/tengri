# SPDX-License-Identifier: BSD-3-Clause
"""Batched spectroscopic path projects photometry through the WavePrecomp LUT (#3505, A1).

A template built with ``approx=WavePrecomp()`` publishes the SSP x filter LUT. The
per-galaxy joint path projects its photometry through that LUT beside an exact
spectrum; the batched path must do the same, not the exact projector. The batched
photometry must equal the per-galaxy model's photometry, differ from the exact
projector by more than round-off, agree with it within the documented LUT accuracy,
compile once per spec across redshifts, and carry a finite non-zero gradient.

Runs on the synthetic narrow SSP (no ``data/ssp_*.h5`` needed).
"""

from __future__ import annotations

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
from tengri.forward.batched_spectrum import make_batched_predict
from tengri.inference.batched_spectro import batched_log_likelihood
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.contract

_MASS = "sfh_dpl_log_total_mass"
_N_DIAG = 21
_R_BANDED = 300.0
_WAVE = np.linspace(4000.0, 7000.0, 160)  # [Angstrom]
_N_FILT = 4
# Documented worst-case LUT error of the photometric projection (predict_via_precomp
# docstring: K=5 sub-band quadrature, <~0.6 % in GALEX FUV, far smaller on griz).
_LUT_RTOL = 6e-3
# z table past the redshift prior [0, 0.5], so the LUT covers every galaxy.
_LUT = WavePrecomp(z_min=0.0, z_max=0.6)


def _filters():
    centers = np.linspace(4800.0, 9000.0, _N_FILT)
    return tuple(
        FilterCurve(
            wave=jnp.linspace(c * 0.84, c * 1.16, 40),
            trans=jnp.sin(jnp.linspace(0.0, np.pi, 40)) * 0.6,
            name=f"b{k}",
        )
        for k, c in enumerate(centers)
    )


def _template(ssp, *, approx=None):
    """Free-redshift spectroscopic template with photometry, optionally on the LUT."""
    observation = Observation(
        spectroscopy=Spectroscopy(
            wave_obs=jnp.asarray(_WAVE),
            resample="conserving",
            resolution_matrix=gaussian_resolution_bands(jnp.asarray(_WAVE), _R_BANDED, _N_DIAG),
        ),
        photometry=Photometry(filters=_filters()),
    )
    kwargs = {} if approx is None else {"approx": approx}
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), _MASS: Uniform(8.0, 12.0)},
        neb={"type": "none"},
        redshift=Uniform(0.0, 0.5),
        **kwargs,
    )


def _record(z, phot_flux=None):
    """One galaxy; photometry is carried so the bucket has ``has_phot`` set."""
    n = _WAVE.size
    if phot_flux is None:
        phot_flux = np.full(_N_FILT, 1e-30)
    phot = {"phot_flux": phot_flux, "phot_err": 0.05 * phot_flux}
    return GalaxySpectrum(
        wave=_WAVE,
        flux=np.ones(n),
        ivar=np.ones(n),
        resolution=gaussian_resolution_bands(jnp.asarray(_WAVE), _R_BANDED, _N_DIAG),
        z=z,
        **phot,
    )


def _stack(params: dict, batch: int) -> dict:
    return {k: jnp.broadcast_to(jnp.asarray(v), (batch,)) for k, v in params.items()}


def _params(model):
    return dict(model.spec.sample(jax.random.PRNGKey(0)))


def _batched_phot(template, z, params):
    """Batched ``phot_fnu`` of one galaxy at redshift ``z`` under ``template``."""
    [(spec, batch, _)] = build_spectro_batches([_record(z)], model=template)
    out = make_batched_predict(template, spec)(_stack(params, 1), batch)
    return np.asarray(out["phot_fnu"])[0]


def test_batched_photometry_equals_per_galaxy_lut_model(synthetic_ssp):
    """(a) One galaxy: batched phot_fnu equals the per-galaxy LUT model's photometry."""
    template = _template(synthetic_ssp, approx=_LUT)
    z = 0.1
    params = _params(template)
    batched = _batched_phot(template, z, params)
    per_galaxy = np.asarray(template.predict_photometry({**params, "redshift": z}))
    np.testing.assert_allclose(batched, per_galaxy, rtol=1e-10, atol=0.0)


def test_batched_lut_photometry_differs_from_exact_within_lut_accuracy(synthetic_ssp):
    """(b) The LUT is used: it differs from the exact projector beyond round-off, within rtol."""
    z = 0.1
    lut_template = _template(synthetic_ssp, approx=_LUT)
    exact_template = _template(synthetic_ssp)
    params = _params(lut_template)
    lut = _batched_phot(lut_template, z, params)
    exact = _batched_phot(exact_template, z, params)
    assert np.max(np.abs(lut / exact - 1.0)) > 1e-10
    np.testing.assert_allclose(lut, exact, rtol=_LUT_RTOL, atol=0.0)


def test_second_redshift_batch_compiles_nothing(synthetic_ssp):
    """(c) Two batches at different redshifts under one spec share one compile."""
    from jax._src import test_util as jtu

    template = _template(synthetic_ssp, approx=_LUT)
    params = _params(template)
    [(spec, batch_a, _)] = build_spectro_batches([_record(0.05)], model=template)
    [(spec_b, batch_b, _)] = build_spectro_batches([_record(0.2)], model=template)
    assert spec == spec_b
    predict = make_batched_predict(template, spec)
    predict(_stack(params, 1), batch_a)
    with jtu.count_jit_compilation_cache_miss() as counter:
        predict(_stack(params, 1), batch_b)
    assert counter() == 0


def test_batched_loglik_gradient_is_finite_and_nonzero(synthetic_ssp):
    """(d) float64 gradient of the batched log-likelihood, photometry included, is finite."""
    template = _template(synthetic_ssp, approx=_LUT)
    z = 0.1
    ref = _params(template)
    flux = np.asarray(template.predict_photometry({**ref, "redshift": z}))
    records = [_record(z, phot_flux=flux), _record(0.2, phot_flux=flux)]
    [(spec, batch, _)] = build_spectro_batches(records, model=template)
    loglik = batched_log_likelihood(template, spec)
    masses = jnp.asarray([10.0, 10.3])
    values, grads = jax.vmap(jax.value_and_grad(loglik))({_MASS: masses}, batch)
    g = np.asarray(grads[_MASS])
    assert np.asarray(values).dtype == np.float64
    assert np.all(np.isfinite(g))
    assert np.any(g != 0.0)
