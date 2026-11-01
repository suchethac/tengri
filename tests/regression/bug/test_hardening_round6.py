# SPDX-License-Identifier: BSD-3-Clause
"""Regression: hardening round 6 (resample redshift decision, SKIRTOR load except).

Item 1 (``resample="auto"`` redshift). The decision is taken at a reference
redshift. With a free redshift and no finite lower bound it used to fall back
to z=0 silently, which point-samples or integrates the spectrum at the wrong
rest frame. The contract now is: fixed z uses the fixed value, a free z with a
finite ``lo`` uses ``lo``, and anything else raises ``ParameterError`` under
``resample="auto"`` only. Explicit modes never consult the reference, so they
are not refused.

Item 3 (SKIRTOR template load). ``build`` used to swallow every exception from
the SKIRTOR template load. Only the missing-file case may be silent: the lazy
``apply()`` path raises the clear not-found error itself. A corrupt grid must
propagate.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

from tengri import DEFAULT, Fixed, Observation, Spectroscopy, Uniform
from tengri.config.exceptions import ParameterError
from tengri.forward.forward_model import ForwardModel
from tengri.forward.population_sed_model import PopulationSEDModel
from tengri.forward.sed_model import SEDModel, SpectrumPrecomp, WavePrecomp
from tengri.inference.batched_spectro import batched_log_likelihood, fit_spectra_map_vmap
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches


def _redshift_model(distribution, *, fixed_z=None, resample="auto", spectroscopy=True):
    """Stand-in carrying only the attributes ``_resample_z_ref`` reads."""
    spec = SimpleNamespace(get_distribution=lambda name: distribution)
    spectro = SimpleNamespace(resample=resample) if spectroscopy else None
    return SimpleNamespace(
        spec=spec,
        observation=SimpleNamespace(spectroscopy=spectro),
        _z_fixed=fixed_z,
    )


def _free(lo):
    return SimpleNamespace(is_fixed=False, lo=lo, hi=3.0)


def _fixed(value):
    return SimpleNamespace(is_fixed=True, value=value, lo=value, hi=value)


def _ref(model):
    return SEDModel._resample_z_ref(model)


def test_free_unbounded_redshift_under_auto_raises():
    """A free z with no lower bound must not silently decide at z=0."""
    model = _redshift_model(_free(lo=None))
    with pytest.raises(ParameterError, match="bounded redshift"):
        _ref(model)


def test_free_non_finite_lower_bound_under_auto_raises():
    """An infinite ``lo`` is no reference redshift either."""
    model = _redshift_model(_free(lo=float("-inf")))
    with pytest.raises(ParameterError, match='resample="auto"'):
        _ref(model)


def test_missing_distribution_without_fixed_z_under_auto_raises():
    """No redshift distribution and no fixed z leaves nothing to decide at."""

    def _missing(name):
        raise KeyError(name)

    model = _redshift_model(None)
    model.spec = SimpleNamespace(get_distribution=_missing)
    with pytest.raises(ParameterError, match="bounded redshift"):
        _ref(model)


def test_free_bounded_redshift_uses_its_lower_bound():
    """A finite prior lower bound is the reference, clipped at zero."""
    assert _ref(_redshift_model(_free(lo=0.4))) == pytest.approx(0.4)
    assert _ref(_redshift_model(_free(lo=-0.2))) == 0.0


def test_fixed_redshift_uses_the_fixed_value():
    """Fixed z keeps its value; the free-z guard never fires for it."""
    assert _ref(_redshift_model(_fixed(0.3), fixed_z=0.3)) == pytest.approx(0.3)


def test_explicit_resample_mode_is_not_refused():
    """With ``resample='point'`` or ``'conserving'`` the reference is unused, so no raise."""
    for mode in ("point", "conserving"):
        model = _redshift_model(_free(lo=None), resample=mode)
        assert _ref(model) == 0.0


def test_no_spectroscopy_is_not_refused():
    """A photometry-only model has no resample decision to make."""
    model = _redshift_model(_free(lo=None), spectroscopy=False)
    assert _ref(model) == 0.0


def test_skirtor_template_load_propagates_non_missing_errors(monkeypatch):
    """A corrupt SKIRTOR grid is an error, not a silent fallback to lazy loading."""
    import tengri.components.agn.skirtor as skirtor

    def _corrupt():
        raise RuntimeError("corrupt SKIRTOR grid")

    monkeypatch.setattr(skirtor, "_load_skirtor_default_grid", _corrupt)
    from tengri.components.agn.component import AGNSEDComponent

    component = AGNSEDComponent.__new__(AGNSEDComponent)
    object.__setattr__(component, "config", SimpleNamespace(model="skirtor"))
    with pytest.raises(RuntimeError, match="corrupt SKIRTOR grid"):
        AGNSEDComponent.precompute(component, wave_grid=None)


# ── Real models under the default resample="auto" ─────────────────────
#
# The stand-ins above pin the decision rule. These build real spectroscopic
# models on the synthetic wide SSP (no data/ grid needed) and assert that
# nothing legitimate raises. Each one is a reviewer-facing claim: a free
# bounded redshift, a catalog runtime redshift, the batched per-galaxy path,
# the population caller of ``_predict_observation``, and a fixed redshift.

_WAVE = np.linspace(4000.0, 7000.0, 300)


def _build_model(ssp, redshift, *, approx=None):
    """Real spectroscopic SEDModel with the default ``resample="auto"``."""
    kwargs = {} if approx is None else {"approx": approx}
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=Spectroscopy(wave_obs=_WAVE, resolution=2000.0)),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": Fixed(0.3)},
        neb={"type": "none"},
        redshift=redshift,
        **kwargs,
    )


def test_real_free_bounded_redshift_decides_at_prior_lower_bound(synthetic_ssp_wide):
    """(a) Free z in [0.5, 2.0] under the default auto: decides at lo=0.5 and predicts.

    The spectrum is a flux density, so it must be finite and non-negative.
    """
    model = _build_model(synthetic_ssp_wide, Uniform(0.5, 2.0))
    assert model.observation.spectroscopy.resample == "auto"
    assert _ref(model) == pytest.approx(0.5)
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    spectrum = np.asarray(model.predict_spectrum(params))
    assert spectrum.shape == _WAVE.shape
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum >= 0.0)


def test_real_catalog_model_predicts_at_runtime_redshift(synthetic_ssp_wide):
    """(b) ``catalog_z_range`` model: the redshift is supplied at runtime.

    A free prior bounded by the catalog range builds and predicts the spectrum
    at a runtime z of 0.3 without raising. A ``Fixed`` placeholder is not used
    here: call-time overrides of a Fixed parameter are refused (#2296), and the
    per-galaxy Fitter route for that case is covered by test_bug_2858.
    """
    approx = (WavePrecomp(catalog_z_range=(0.01, 1.0)), SpectrumPrecomp())
    model = _build_model(synthetic_ssp_wide, Uniform(0.01, 1.0), approx=approx)
    assert model._catalog_z_range == (0.01, 1.0)
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    params["redshift"] = 0.3
    spectrum = np.asarray(model.predict_spectrum(params))
    assert spectrum.shape == _WAVE.shape
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum >= 0.0)


def test_real_batched_spectroscopy_per_galaxy_redshift_runs(synthetic_ssp_wide):
    """(c) Batched per-galaxy redshifts under the default auto: no raise end to end.

    Three galaxies at distinct redshifts go through ``build_spectro_batches``,
    ``batched_log_likelihood`` and a short batched MAP fit. The Gaussian
    log-likelihood is a sum of ``-0.5 r^2`` terms, so it is never positive.
    """
    specs = [(300, 4000.0, 7000.0, 0.05), (360, 4200.0, 7400.0, 0.1), (420, 4500.0, 7500.0, 0.2)]
    records = []
    for n_pix, lo, hi, z in specs:
        wave = np.linspace(lo, hi, n_pix)
        bm = gaussian_resolution_bands(wave, 2000.0, 3)
        records.append(
            GalaxySpectrum(
                wave=wave,
                flux=np.full(n_pix, 1e-3),
                ivar=np.full(n_pix, 1e6),
                resolution=bm,
                z=z,
            )
        )
    template = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=Observation(spectroscopy=Spectroscopy(wave_obs=_WAVE, resolution=2000.0)),
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "sfh_dpl_log_total_mass": Uniform(8.0, 12.0),
        },
        neb={"type": "none"},
        redshift=Uniform(0.0, 0.5),
    )
    buckets = build_spectro_batches(records, quantum=420, model=template)
    spec, batch, index = buckets[0]
    np.testing.assert_array_equal(index, [0, 1, 2])
    loglik = batched_log_likelihood(template, spec)
    values = np.asarray(jax.vmap(loglik)({"sfh_dpl_log_total_mass": np.full(3, 10.0)}, batch))
    assert np.all(np.isfinite(values))
    assert np.all(values <= 0.0)
    fit = fit_spectra_map_vmap(template, buckets, n_steps=2)
    assert np.all(np.isfinite(np.asarray(fit["nlp"])))


def test_real_population_caller_reaches_predict_observation(synthetic_ssp_wide):
    """(d) ``ForwardModel.build`` with a population: the ``_predict_observation`` caller.

    A population publishes ``batched_axes={"galaxy": 0}``, which routes through the
    ``_resample_z_ref`` call at forward_model.py ~1476. The free bounded redshift
    decides at its lower bound, and the batched spectrum is finite and non-negative.
    """
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=_WAVE, resolution=2000.0))
    sed = _build_model(synthetic_ssp_wide, Uniform(0.5, 2.0))
    galaxy = {"flux_obs": np.zeros(_WAVE.size), "noise": np.ones(_WAVE.size)}
    population = PopulationSEDModel(sed=sed, galaxies=[galaxy, galaxy], data_type="spectroscopy")
    assert population.batched_axes == {"galaxy": 0}
    forward = ForwardModel.build(population=population, observation=obs)
    params = {
        k: np.broadcast_to(np.asarray(v), (2,))
        for k, v in sed.spec.sample(jax.random.PRNGKey(0)).items()
    }
    prediction = forward.predict_observables(params)
    spectrum = np.asarray(prediction["spec_fnu"])
    assert spectrum.shape[-1] == _WAVE.size
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum >= 0.0)
    assert sed._resample_z_ref() == pytest.approx(0.5)


def test_real_fixed_redshift_is_unchanged(synthetic_ssp_wide):
    """(e) Fixed z keeps its value under the default auto, and predicts finite flux."""
    model = _build_model(synthetic_ssp_wide, Fixed(0.3))
    assert _ref(model) == pytest.approx(0.3)
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    spectrum = np.asarray(model.predict_spectrum(params))
    assert spectrum.shape == _WAVE.shape
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum >= 0.0)


def test_real_catalog_placeholder_fixed_z_decides_resample_at_catalog_lower_end(
    synthetic_ssp_wide,
):
    """(f) #3171: a catalog model decides ``resample="auto"`` at the catalog range's lower end.

    The spec carries a ``Fixed`` placeholder redshift, and the real redshift arrives at
    runtime within ``catalog_z_range``. The coarsest rest-frame pixels sit at the lowest
    redshift, so the decision must be taken at ``catalog_z_range[0]``, not the placeholder.
    """
    placeholder = 0.1
    approx = (WavePrecomp(catalog_z_range=(0.01, 1.0)), SpectrumPrecomp())
    model = _build_model(synthetic_ssp_wide, Fixed(placeholder), approx=approx)
    assert model._catalog_z_range == (0.01, 1.0)
    assert model.observation.spectroscopy.resample == "auto"
    assert _ref(model) == pytest.approx(0.01)
