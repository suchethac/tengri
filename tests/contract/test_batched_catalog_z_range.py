# SPDX-License-Identifier: BSD-3-Clause
"""Batched path accepts a ``catalog_z_range`` template and a fixed-z template (#3505, A2/A4).

Each galaxy's ``obs.z`` is the redshift the batched forward uses. A template with a
per-galaxy redshift window (``WavePrecomp(catalog_z_range=...)``) or a fixed redshift
must reproduce, galaxy by galaxy, the per-galaxy model at that galaxy's own z, and a
second batch of a new redshift under the same spec must compile nothing.

Runs on the synthetic narrow SSP (no ``data/ssp_*.h5`` needed).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import test_util as jtu

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
from tests.contract.test_batched_spectrum_forward import _N_DIAG, _R_BANDED, _banded, _grid

pytestmark = pytest.mark.contract

_MASS = "sfh_dpl_log_total_mass"
_N_PIX = 240
_LO, _HI = 4000.0, 7000.0
_CATALOG_RANGE = (0.02, 0.4)
_N_Z = 40
_Z_PAIR_A = (0.05, 0.2)
_Z_PAIR_B = (0.11, 0.33)
_MASS_VALUE = 10.1


def _window_template(ssp, wave):
    """Template with a per-galaxy redshift window; redshift free in the spec."""
    spectroscopy = Spectroscopy(wave_obs=jnp.asarray(wave), resample="conserving", **_banded(wave))
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=spectroscopy),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), _MASS: Uniform(8.0, 12.0)},
        neb={"type": "none"},
        redshift=Uniform(*_CATALOG_RANGE),
        approx=WavePrecomp(catalog_z_range=_CATALOG_RANGE, n_z=_N_Z),
    )


def _fixed_template(ssp, wave, z, approx=None):
    """Template with a fixed redshift ``z``; ``approx`` selects the optional LUT path."""
    spectroscopy = Spectroscopy(wave_obs=jnp.asarray(wave), resample="conserving", **_banded(wave))
    kwargs = {} if approx is None else {"approx": approx}
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=spectroscopy),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), _MASS: Uniform(8.0, 12.0)},
        neb={"type": "none"},
        redshift=Fixed(z),
        **kwargs,
    )


def _records(ssp_wave, redshifts, ssp_model):
    """One GalaxySpectrum per redshift, flux = 1.05 x the model's own prediction."""
    bm = gaussian_resolution_bands(jnp.asarray(ssp_wave), _R_BANDED, _N_DIAG)
    records = []
    for z in redshifts:
        clean = np.asarray(ssp_model.predict_spectrum({_MASS: _MASS_VALUE, "redshift": z}))
        records.append(
            GalaxySpectrum(
                wave=ssp_wave,
                flux=1.05 * clean,
                ivar=np.full(ssp_wave.size, 1.0 / (1e-3 * float(np.max(clean))) ** 2),
                resolution=bm,
                z=z,
            )
        )
    return records


def _reference_loglik(model, record, mass, *, redshift_is_fixed=False):
    """Per-galaxy Gaussian log-likelihood at the galaxy's own z (the reference).

    A fixed-z reference model already sits at ``record.z``, so it takes no override.
    """
    params = {_MASS: mass} if redshift_is_fixed else {_MASS: mass, "redshift": record.z}
    pred = model.predict_spectrum(params)
    r = (jnp.asarray(record.flux) - pred) * jnp.sqrt(jnp.asarray(record.ivar))
    return -0.5 * jnp.sum(r * r)


@pytest.fixture(scope="module")
def grid():
    return _grid(_N_PIX, _LO, _HI)


@pytest.fixture(scope="module")
def window_model(synthetic_ssp, grid):
    return _window_template(synthetic_ssp, grid)


# ── (1) catalog_z_range template: one galaxy, value and gradient ───────────


def test_catalog_z_range_template_matches_per_galaxy_at_its_z(window_model, grid):
    records = _records(grid, [0.2], window_model)
    [(spec, batch, _)] = build_spectro_batches(records, model=window_model)
    loglik = batched_log_likelihood(window_model, spec)
    obs = batch.galaxy(0)

    value, grad = jax.value_and_grad(lambda m: loglik({_MASS: m}, obs))(_MASS_VALUE)
    v_ref, g_ref = jax.value_and_grad(lambda m: _reference_loglik(window_model, records[0], m))(
        _MASS_VALUE
    )
    np.testing.assert_allclose(float(value), float(v_ref), rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(float(grad), float(g_ref), rtol=1e-10, atol=0.0)


# ── (2) fixed-z template: obs.z overrides the fixed value ──────────────────


@pytest.mark.parametrize("approx", [None, WavePrecomp()], ids=["exact", "wave_precomp"])
def test_fixed_z_template_uses_each_galaxys_z(synthetic_ssp, grid, window_model, approx):
    """A fixed value (0.05) baked into the template must not reach the galaxies at 0.05/0.2."""
    template = _fixed_template(synthetic_ssp, grid, 0.05, approx=approx)
    records = _records(grid, list(_Z_PAIR_A), window_model)
    batches = build_spectro_batches(records, model=template)
    [(spec, batch, index)] = batches
    np.testing.assert_array_equal(index, [0, 1])
    loglik = batched_log_likelihood(template, spec)

    for i, record in enumerate(records):
        per_galaxy = _fixed_template(synthetic_ssp, grid, float(record.z), approx=approx)
        mass = jnp.asarray(_MASS_VALUE)
        value, grad = jax.value_and_grad(lambda m, i=i: loglik({_MASS: m}, batch.galaxy(i)))(mass)
        v_ref, g_ref = jax.value_and_grad(
            lambda m, model=per_galaxy, record=record: _reference_loglik(
                model, record, m, redshift_is_fixed=True
            )
        )(mass)
        np.testing.assert_allclose(float(value), float(v_ref), rtol=1e-10, atol=0.0)
        np.testing.assert_allclose(float(grad), float(g_ref), rtol=1e-10, atol=0.0)


# ── (3) build-time validation of galaxy z against the catalog range ────────


def test_galaxy_z_outside_catalog_range_raises_at_build(window_model, grid):
    records = _records(grid, [0.2], window_model)
    bad = [records[0], _records(grid, [0.7], window_model)[0]]
    with pytest.raises(ValueError, match=r"galaxy 1.*z=0\.7.*0\.02.*0\.4"):
        build_spectro_batches(bad, model=window_model)


# ── (4) a second batch at new redshifts, same spec: no recompile ───────────


def test_second_batch_with_new_redshifts_compiles_nothing(window_model, grid):
    batch_a = build_spectro_batches(
        _records(grid, list(_Z_PAIR_A), window_model), model=window_model
    )
    batch_b = build_spectro_batches(
        _records(grid, list(_Z_PAIR_B), window_model), model=window_model
    )
    [(spec_a, data_a, _)] = batch_a
    [(spec_b, data_b, _)] = batch_b
    assert spec_a == spec_b, "the two batches must share one bucket spec"

    loglik = batched_log_likelihood(window_model, spec_a)
    fn = jax.jit(jax.vmap(loglik))
    params = {_MASS: jnp.full((2,), _MASS_VALUE)}
    fn(params, data_a).block_until_ready()
    with jtu.count_jit_compilation_cache_miss() as misses:
        fn(params, data_b).block_until_ready()
    assert misses() == 0, f"second batch recompiled {misses()} programs"


# ── (5) photometry at the galaxy's own z for fixed-z and catalog_z_range templates ──

_N_FILT = 3
_PHOT_LUT = WavePrecomp(z_min=0.0, z_max=0.6, n_z=_N_Z)


def _filters():
    centers = np.linspace(4800.0, 8000.0, _N_FILT)
    return tuple(
        FilterCurve(
            wave=jnp.linspace(c * 0.84, c * 1.16, 40),
            trans=jnp.sin(jnp.linspace(0.0, np.pi, 40)) * 0.6,
            name=f"b{k}",
        )
        for k, c in enumerate(centers)
    )


def _phot_model(ssp, wave, *, redshift, approx=None):
    """Spectroscopic template with photometry; ``approx`` selects the optional LUT path."""
    spectroscopy = Spectroscopy(wave_obs=jnp.asarray(wave), resample="conserving", **_banded(wave))
    observation = Observation(spectroscopy=spectroscopy, photometry=Photometry(filters=_filters()))
    kwargs = {} if approx is None else {"approx": approx}
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), _MASS: Uniform(8.0, 12.0)},
        neb={"type": "none"},
        redshift=redshift,
        **kwargs,
    )


def _phot_record(wave, z):
    """One galaxy with photometry; the photometry values are inert for the phot_fnu check."""
    phot_flux = np.full(_N_FILT, 1e-30)
    return GalaxySpectrum(
        wave=wave,
        flux=np.ones(wave.size),
        ivar=np.ones(wave.size),
        resolution=gaussian_resolution_bands(jnp.asarray(wave), _R_BANDED, _N_DIAG),
        z=z,
        phot_flux=phot_flux,
        phot_err=0.05 * phot_flux,
    )


def _batched_phot(template, wave, z):
    """Batched ``phot_fnu`` of one galaxy at its own ``z`` under ``template``."""
    [(spec, batch, _)] = build_spectro_batches([_phot_record(wave, z)], model=template)
    out = make_batched_predict(template, spec)({_MASS: jnp.asarray([_MASS_VALUE])}, batch)
    return np.asarray(out["phot_fnu"])[0]


_FIXED_LUT_XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "fixed-z WavePrecomp bakes stellar_phot_lnu_precomp at the template's fixed z "
        "(components/stellar/component.py, fixed-z LUT branch, z_source = fixed value); "
        "the batched path reads that LUT for every galaxy, so obs.z is ignored in photometry"
    ),
)


@pytest.mark.parametrize(
    "approx",
    [None, pytest.param(_PHOT_LUT, marks=_FIXED_LUT_XFAIL)],
    ids=["exact", "wave_precomp"],
)
def test_fixed_z_photometry_uses_obs_z(synthetic_ssp, grid, approx):
    """A template fixed at 0.05 gives galaxy photometry at its own z (0.2), not the fixed one."""
    template = _phot_model(synthetic_ssp, grid, redshift=Fixed(0.05), approx=approx)
    per_galaxy = _phot_model(synthetic_ssp, grid, redshift=Fixed(0.2), approx=approx)
    batched = _batched_phot(template, grid, 0.2)
    reference = np.asarray(per_galaxy.predict_photometry({_MASS: _MASS_VALUE}))
    np.testing.assert_allclose(batched, reference, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("z", [0.05, 0.2])
def test_catalog_z_range_photometry_uses_obs_z(synthetic_ssp, grid, z):
    """A catalog_z_range template gives photometry at each galaxy's z, on the LUT."""
    template = _phot_model(
        synthetic_ssp,
        grid,
        redshift=Uniform(*_CATALOG_RANGE),
        approx=WavePrecomp(catalog_z_range=_CATALOG_RANGE, n_z=_N_Z),
    )
    batched = _batched_phot(template, grid, z)
    reference = np.asarray(template.predict_photometry({_MASS: _MASS_VALUE, "redshift": z}))
    np.testing.assert_allclose(batched, reference, rtol=1e-10, atol=0.0)
