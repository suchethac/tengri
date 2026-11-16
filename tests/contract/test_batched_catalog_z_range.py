# SPDX-License-Identifier: BSD-3-Clause
"""Batched path accepts a ``catalog_z_range`` template and a fixed-z template (#3505, A2/A4).

Each galaxy's ``obs.z`` is the redshift the batched forward uses. A template with a
per-galaxy redshift window (``WavePrecomp(catalog_z_range=...)``) or a fixed redshift
must reproduce, galaxy by galaxy, the per-galaxy model at that galaxy's own z, and a
second batch of a new redshift under the same spec must compile nothing.

Runs on the synthetic narrow SSP (no ``data/ssp_*.h5`` needed).
"""

from __future__ import annotations

import gc
import weakref

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
from tengri.forward.batched_spectrum import batched_template, make_batched_predict
from tengri.inference.batched_spectro import batched_log_likelihood
from tengri.inference.fitter import catalog_redshift_span, promote_to_runtime_redshift
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
    with pytest.raises(ValueError, match=r"row\(s\) \[1\].*z=\[0\.7\]"):
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


def test_fixed_z_exact_photometry_uses_obs_z(synthetic_ssp, grid):
    """A template fixed at 0.05 gives galaxy photometry at its own z (0.2), not the fixed one."""
    template = _phot_model(synthetic_ssp, grid, redshift=Fixed(0.05))
    per_galaxy = _phot_model(synthetic_ssp, grid, redshift=Fixed(0.2))
    batched = _batched_phot(template, grid, 0.2)
    reference = np.asarray(per_galaxy.predict_photometry({_MASS: _MASS_VALUE}))
    np.testing.assert_allclose(batched, reference, rtol=1e-10, atol=0.0)


_PHOT_ZS = (0.05, 0.2)
# Documented worst-case error of the photometric LUT (predict_via_precomp), as in
# tests/contract/test_batched_wave_precomp_photometry.py.
_LUT_RTOL = 6e-3


def _batched_phot_many(template, wave, zs):
    """Batched ``phot_fnu`` of galaxies at ``zs`` (one call, so one span) under ``template``."""
    [(spec, batch, _)] = build_spectro_batches([_phot_record(wave, z) for z in zs], model=template)
    params = {_MASS: jnp.full((len(zs),), _MASS_VALUE)}
    return np.asarray(make_batched_predict(template, spec)(params, batch)["phot_fnu"])


def test_fixed_z_wave_precomp_photometry_uses_each_galaxys_z(synthetic_ssp, grid):
    """Fixed-z LUT template: batched photometry equals the catalog twin at each obs.z.

    The template is fixed at 0.05 with its photometry LUT baked there. The batched
    path projects each galaxy on the catalog twin over the batch span, so every
    galaxy matches the per-galaxy catalog model with that span, and sits within the
    LUT accuracy of the exact per-galaxy photometry.
    """
    template = _phot_model(synthetic_ssp, grid, redshift=Fixed(0.05), approx=_PHOT_LUT)
    batched = _batched_phot_many(template, grid, _PHOT_ZS)
    twin = promote_to_runtime_redshift(template, catalog_redshift_span(_PHOT_ZS))
    for i, z in enumerate(_PHOT_ZS):
        reference = np.asarray(twin.predict_photometry({_MASS: _MASS_VALUE, "redshift": z}))
        np.testing.assert_allclose(batched[i], reference, rtol=1e-10, atol=0.0)
        exact = _phot_model(synthetic_ssp, grid, redshift=Fixed(z))
        exact_ref = np.asarray(exact.predict_photometry({_MASS: _MASS_VALUE}))
        np.testing.assert_allclose(batched[i], exact_ref, rtol=_LUT_RTOL, atol=0.0)


def test_fixed_z_wave_precomp_batches_compile_once(synthetic_ssp, grid):
    """Two batches of one span, at different redshifts, add no compilation-cache miss."""
    template = _phot_model(synthetic_ssp, grid, redshift=Fixed(0.05), approx=_PHOT_LUT)
    [(spec_a, data_a, _)] = build_spectro_batches(
        [_phot_record(grid, z) for z in (0.05, 0.12, 0.2)], model=template
    )
    [(spec_b, data_b, _)] = build_spectro_batches(
        [_phot_record(grid, z) for z in (0.05, 0.1, 0.2)], model=template
    )
    assert spec_a == spec_b, "the two batches must share one bucket spec"
    predict = make_batched_predict(template, spec_a)
    params = {_MASS: jnp.full((3,), _MASS_VALUE)}
    predict(params, data_a)["phot_fnu"].block_until_ready()
    with jtu.count_jit_compilation_cache_miss() as misses:
        predict(params, data_b)["phot_fnu"].block_until_ready()
    assert misses() == 0, f"second batch recompiled {misses()} programs"


def test_catalog_twin_dies_with_its_template(synthetic_ssp, grid):
    """The twin is cached on the template, so freeing the template frees the twin."""
    template = _phot_model(synthetic_ssp, grid, redshift=Fixed(0.05), approx=_PHOT_LUT)
    [(spec, _, _)] = build_spectro_batches(
        [_phot_record(grid, z) for z in _PHOT_ZS], model=template
    )
    twin_ref = weakref.ref(batched_template(template, spec))
    assert twin_ref() is not None
    assert batched_template(template, spec) is twin_ref()
    assert batched_template(twin_ref(), spec) is twin_ref(), "a twin must return itself"
    del template, spec
    gc.collect()
    assert twin_ref() is None, "the twin outlived its template"


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


# Bounds on the max relative photometry error of the LUT twin against exact per-galaxy
# photometry at five interior redshifts. Measured: span 0.0-0.6 with n_z=40 1.725e-2,
# with the default n_z=250 3.274e-4; span 0.0-0.15 with n_z=40 7.400e-4,
# with n_z=250 5.511e-5.
@pytest.mark.parametrize(
    ("span", "n_z", "bound"),
    [
        ((0.0, 0.6), 40, 2.5e-2),
        ((0.0, 0.15), 40, 1e-3),
        ((0.0, 0.6), 250, 1e-3),
        ((0.0, 0.15), 250, 1e-4),
    ],
    ids=["wide_nz40", "dwarf_nz40", "wide_nz250", "dwarf_nz250"],
)
def test_fixed_z_wave_precomp_photometry_accuracy(synthetic_ssp, grid, span, n_z, bound):
    """The LUT twin's photometry stays within a measured bound of exact per-galaxy photometry."""
    lut = WavePrecomp(z_min=0.0, z_max=0.6, n_z=n_z)
    template = _phot_model(synthetic_ssp, grid, redshift=Fixed(0.05), approx=lut)
    twin = template.with_runtime_redshift(span)
    worst = 0.0
    for z in np.linspace(span[0], span[1], 7)[1:-1]:
        exact = _phot_model(synthetic_ssp, grid, redshift=Fixed(float(z)))
        exact_flux = np.asarray(exact.predict_photometry({_MASS: _MASS_VALUE}))
        lut_flux = np.asarray(twin.predict_photometry({_MASS: _MASS_VALUE, "redshift": float(z)}))
        worst = max(worst, float(np.max(np.abs(lut_flux - exact_flux) / np.abs(exact_flux))))
    print(f"MEASURED span={span} n_z={n_z} max_rel_err={worst:.6e}")
    assert worst < bound
