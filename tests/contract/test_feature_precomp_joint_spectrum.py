# SPDX-License-Identifier: BSD-3-Clause
"""A joint photometry+spectrum model with a nebular grid keeps the photometry LUT (#3505, B9).

``approx=(WavePrecomp(), FeaturePrecomp())`` on a joint observation used to turn the
photometry LUT off in the per-galaxy kernel (``SEDModel._forward``, ``mixed_phot_lut``),
so photometry was projected exactly while the batched path projected it through the LUT.
The nebular grid does not require that exclusion: a spectrum keeps ``sed_nebular``
materialized (``must_materialize_sed``), so the grid never zeroes the continuum the
spectrum reads, and the LUT photometry reads the nebular filter photometry the grid
serves.

Pinned here, on a template with a spectrum, a Cue nebular and a free redshift:

* the per-galaxy joint photometry equals the batched path's photometry for one galaxy;
* it differs from the exact projector beyond round-off and agrees with it within the
  documented LUT accuracy;
* the joint spectrum equals the spectrum of the same model with exact photometry, and the
  nebular continuum it reads is non-zero;
* the float64 gradient of the joint photometry is finite and non-zero.

Runs on the bare-stellar FSPS SSP (``ssp_data_fsps``), which Cue requires.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FeaturePrecomp,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Spectroscopy,
    Uniform,
    WavePrecomp,
)
from tengri.forward.batched_spectrum import make_batched_predict
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.contract

_MASS = "sfh_delayed_log_total_mass"
_N_DIAG = 21
_R_BANDED = 300.0
_WAVE = np.linspace(4000.0, 7000.0, 160)  # [Angstrom]
_N_FILT = 4
# Documented worst-case LUT error of the photometric projection (predict_via_precomp
# docstring), with the grid's own band error on top; the sibling batched test uses the same bound.
_LUT_RTOL = 6e-3
_LUT = WavePrecomp(z_min=0.0, z_max=0.6)
_GRID = (_LUT, FeaturePrecomp())
_Z = 0.1


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
    """Joint photometry+spectrum model with a Cue nebular and a free redshift."""
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
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT), _MASS: Uniform(9.0, 11.0)},
        dust_attenuation={"type": "none"},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Uniform(0.0, 0.5),
        **kwargs,
    )


@pytest.fixture(scope="module")
def models(ssp_data_fsps):
    grid = _template(ssp_data_fsps, approx=_GRID)
    assert grid._nebular_grid_table is not None, "setup: the nebular grid is not attached"
    return {"grid": grid, "exact": _template(ssp_data_fsps)}


@pytest.fixture(scope="module")
def truth(models):
    params = dict(models["grid"].spec.sample(jax.random.PRNGKey(0)))
    params["redshift"] = _Z
    return params


def _batched_phot(template, z, params):
    """Batched ``phot_fnu`` of one galaxy at redshift ``z`` under ``template``."""
    record = GalaxySpectrum(
        wave=_WAVE,
        flux=np.ones(_WAVE.size),
        ivar=np.ones(_WAVE.size),
        resolution=gaussian_resolution_bands(jnp.asarray(_WAVE), _R_BANDED, _N_DIAG),
        z=z,
        phot_flux=np.full(_N_FILT, 1e-30),
        phot_err=np.full(_N_FILT, 5e-32),
    )
    [(spec, batch, _)] = build_spectro_batches([record], model=template)
    stacked = {k: jnp.broadcast_to(jnp.asarray(v), (1,)) for k, v in params.items()}
    out = make_batched_predict(template, spec)(stacked, batch)
    return np.asarray(out["phot_fnu"])[0]


def test_joint_photometry_equals_the_batched_path(models, truth):
    per_galaxy = np.asarray(models["grid"].predict_photometry(truth))
    batched = _batched_phot(models["grid"], _Z, truth)
    np.testing.assert_allclose(per_galaxy, batched, rtol=1e-10, atol=0.0)


def test_joint_photometry_is_the_lut_not_the_exact_projector(models, truth):
    lut = np.asarray(models["grid"].predict_photometry(truth))
    exact = np.asarray(models["exact"].predict_photometry(truth))
    assert np.max(np.abs(lut / exact - 1.0)) > 1e-10
    np.testing.assert_allclose(lut, exact, rtol=_LUT_RTOL, atol=0.0)


def test_joint_spectrum_equals_the_exact_photometry_model(models, truth):
    np.testing.assert_allclose(
        np.asarray(models["grid"].predict_spectrum(truth)),
        np.asarray(models["exact"].predict_spectrum(truth)),
        rtol=1e-10,
        atol=0.0,
    )


def test_joint_spectrum_reads_a_nonzero_nebular_continuum(models, truth):
    state = models["grid"].predict_state(truth)
    assert float(jnp.sum(jnp.abs(state.derived["sed_nebular"]))) > 0.0


def test_joint_photometry_gradient_is_finite_and_nonzero(models, truth):
    free = list(models["grid"].spec.free_params)
    v0 = jnp.asarray([float(truth[k]) for k in free])

    def loss(v):
        params = dict(truth)
        params.update({k: v[i] for i, k in enumerate(free)})
        return jnp.sum(models["grid"].predict_photometry(params))

    grad = np.asarray(jax.grad(loss)(v0))
    assert grad.dtype == np.float64
    assert np.all(np.isfinite(grad))
    assert np.any(grad != 0.0)
