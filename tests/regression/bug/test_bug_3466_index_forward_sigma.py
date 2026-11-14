# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #3466: model indices measured on the model velocity dispersion.

Lick indices are defined at sigma = 0 (Trager et al. 1998), but the model index used to
be measured on the unbroadened rest-frame SED while observed indices carry their measured
sigma. The model index is now measured on the SED broadened by the model's own sigma_v
(the same Gaussian-in-ln(lambda) kernel as the spectrum path), so the two are directly
comparable. ``broaden=False`` keeps the unbroadened measurement for users who sigma-correct
their data. The numpy reference below is the kernel of the issue's repro, written
independently of the library.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    Observation,
    SEDModel,
    SpectralIndexData,
    Uniform,
    load_ssp_data,
)
from tengri.observation.spectral_indices import STANDARD_INDICES, measure_index_jax

pytestmark = pytest.mark.regression_bug

C_KMS = 299792.458
SOLAR_LGMET = np.log10(0.0142)
TEN_GYR_LGAGE = np.log10(10.0)

INDEX_NAMES = ("Fe5335", "Fe5270", "Mgb", "Hbeta", "Dn4000")
DN4000_DROP_BOUND = 0.01  # Dn4000 moves by < 1 % from sigma = 0 to 300 km/s (#3466)
FE5335_DROP_RANGE = (0.30, 0.40)  # the issue's measured -35 % at 10 Gyr, with margin


@pytest.fixture(scope="module")
def ssp():
    return load_ssp_data("data/fsps_prsc_miles_chabrier.h5")


def _gaussian_ln_lambda(flux, wave, sigma_kms):
    """Independent numpy reference: Gaussian in ln(lambda) on a uniform ln(lambda) grid."""
    lnl = np.log(wave)
    dl = lnl[1] - lnl[0]
    s = sigma_kms / C_KMS / dl
    half = int(np.ceil(6 * s))
    x = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (x / s) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(flux, half, mode="edge"), k, mode="valid")


@pytest.fixture(scope="module")
def ten_gyr_solar(ssp):
    """Solar-metallicity 10 Gyr SSP spectrum on a uniform ln(lambda) grid, as F_lambda."""
    wave = np.asarray(ssp.ssp_wave, dtype=np.float64)
    lgmet = np.asarray(ssp.ssp_lgmet)
    lgage = np.asarray(ssp.ssp_lg_age_gyr)
    im = int(np.argmin(np.abs(lgmet - SOLAR_LGMET)))
    ia = int(np.argmin(np.abs(lgage - TEN_GYR_LGAGE)))
    lnl = np.arange(np.log(3000.0), np.log(6500.0), 2e-5)
    lam = np.exp(lnl)
    flam = np.interp(lam, wave, np.asarray(ssp.ssp_flux[im, ia, :]) / wave**2)
    return lam, flam


def _index_model(ssp, sigma_free=True):
    """Index-only model whose sigma_v is free, declared the way the spectrum path does."""
    obs = Observation(
        spectral_indices=SpectralIndexData.from_names(
            names=list(INDEX_NAMES),
            values=[1.0] * len(INDEX_NAMES),
            errors=[0.1] * len(INDEX_NAMES),
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_diff": Fixed(0.0),
                "tau_bc": Fixed(0.0),
            },
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )
    if not sigma_free:
        return base
    spec = base.spec.merge_observation_params(sigma_v_kms=Uniform(0.0, 2000.0))
    return SEDModel(spec, ssp, observation=obs)


def _defs():
    return tuple(STANDARD_INDICES[n] for n in INDEX_NAMES)


def _params(model, sigma_kms):
    p = {name: 0.5 for name in model.spec.free_params}
    p["sigma_v_kms"] = jnp.asarray(sigma_kms, dtype=jnp.float64)
    return p


# ── the kernel and the issue's measured effect ───────────────────


def test_helper_matches_the_independent_numpy_kernel(ten_gyr_solar):
    from tengri.observation.spectral_indices import broaden_index_sed

    lam, flam = ten_gyr_solar
    fnu = jnp.asarray(flam * lam**2)
    wave = jnp.asarray(lam)
    got = np.asarray(broaden_index_sed(wave, fnu, 300.0))
    ref = _gaussian_ln_lambda(flam * lam**2, lam, 300.0)
    np.testing.assert_allclose(got, ref, rtol=5e-3, atol=0.0)


def test_fe5335_at_10gyr_drops_by_the_issue_amount_from_sigma_0_to_300(ten_gyr_solar):
    from tengri.observation.spectral_indices import broaden_index_sed

    lam, flam = ten_gyr_solar
    d = STANDARD_INDICES["Fe5335"]
    base = float(measure_index_jax(jnp.asarray(lam), jnp.asarray(flam * lam**2), d))
    broad = float(
        measure_index_jax(
            jnp.asarray(lam),
            broaden_index_sed(jnp.asarray(lam), jnp.asarray(flam * lam**2), 300.0),
            d,
        )
    )
    drop = (broad - base) / base
    lo, hi = FE5335_DROP_RANGE
    assert -hi < drop < -lo, f"Fe5335 changed by {100 * drop:.1f} % (issue: about -35 %)"


def test_dn4000_moves_by_less_than_one_percent(ten_gyr_solar):
    from tengri.observation.spectral_indices import broaden_index_sed

    lam, flam = ten_gyr_solar
    d = STANDARD_INDICES["Dn4000"]
    fnu = jnp.asarray(flam * lam**2)
    base = float(measure_index_jax(jnp.asarray(lam), fnu, d))
    broad = float(
        measure_index_jax(jnp.asarray(lam), broaden_index_sed(jnp.asarray(lam), fnu, 300.0), d)
    )
    assert abs(broad - base) / abs(base) < DN4000_DROP_BOUND


# ── the model ────────────────────────────────────────────────────


def test_sigma_zero_is_bit_identical_to_the_unbroadened_measurement(ssp):
    model = _index_model(ssp)
    p = _params(model, 0.0)
    on = np.asarray(model.predict_spectral_indices(p, _defs()))
    off = np.asarray(model.predict_spectral_indices(p, _defs(), broaden=False))
    np.testing.assert_allclose(on, off, rtol=0.0, atol=1e-10)


def test_model_index_depends_on_sigma_v_and_broaden_false_is_the_old_value(ssp):
    model = _index_model(ssp)
    fe_300 = np.asarray(model.predict_spectral_indices(_params(model, 300.0), _defs()))
    fe_0 = np.asarray(model.predict_spectral_indices(_params(model, 0.0), _defs()))
    unbroadened = np.asarray(
        model.predict_spectral_indices(_params(model, 300.0), _defs(), broaden=False)
    )
    assert not np.allclose(fe_300, fe_0, rtol=1e-6)
    np.testing.assert_allclose(unbroadened, fe_0, rtol=0.0, atol=1e-10)


def test_model_index_is_the_broadened_rest_sed_measurement(ssp):
    model = _index_model(ssp)
    p = _params(model, 300.0)
    rest = model.predict_rest_sed({k: v for k, v in p.items()})
    wave = np.asarray(rest.wavelength, float)
    lnu = np.asarray(rest.sed, float)
    # The rest grid is not uniform in ln(lambda) everywhere, so the numpy reference
    # runs on a uniform ln(lambda) resampling of the same SED.
    lnl = np.arange(np.log(wave[0]), np.log(wave[-1]), np.median(np.diff(np.log(wave))))
    lam = np.exp(lnl)
    flam = np.interp(lam, wave, lnu / wave**2)
    got = np.asarray(model.predict_spectral_indices(p, _defs()))
    for i, n in enumerate(INDEX_NAMES):
        ref_flam = _gaussian_ln_lambda(flam, lam, 300.0)
        ref = float(
            measure_index_jax(
                jnp.asarray(lam), jnp.asarray(ref_flam * lam**2), STANDARD_INDICES[n]
            )
        )
        assert got[i] == pytest.approx(ref, rel=2e-2), n


def test_gradient_wrt_sigma_v_is_finite_at_zero_and_at_300(ssp):
    model = _index_model(ssp)
    defs = _defs()

    def f(s, k):
        p = _params(model, 0.0)
        p["sigma_v_kms"] = s
        return model.predict_spectral_indices(p, defs)[k]

    for s in (0.0, 300.0):
        for k in range(len(INDEX_NAMES)):
            g = float(jax.grad(f)(jnp.asarray(s), k))
            assert np.isfinite(g), (s, INDEX_NAMES[k])


def test_approx_true_refuses_a_broadened_measurement(ssp):
    model = _index_model(ssp)
    with pytest.raises(ValueError, match="broaden"):
        model.predict_spectral_indices(_params(model, 300.0), _defs(), approx=True)


def test_index_only_fit_recovers_sigma_v_within_its_posterior_sd(ssp):
    """Laplace on sigma_v alone: MAP of the index chi^2, sd from its curvature.

    The Fitter constructor requires a photometry or spectrum data array, so the index
    chi^2 is evaluated directly on the model's own index prediction, which is the
    term the index likelihood adds (``inference/data_term.py``).
    """
    truth = 200.0
    base = _index_model(ssp)
    defs = _defs()
    pred_true = np.asarray(base.predict_spectral_indices(_params(base, truth), defs))
    errors = jnp.asarray(0.02 * np.maximum(np.abs(pred_true), 1.0))
    rng = np.random.default_rng(3466)
    values = jnp.asarray(pred_true + np.asarray(errors) * rng.standard_normal(pred_true.shape))

    def chi2(s):
        p = _params(base, 0.0)
        p["sigma_v_kms"] = s
        pred = base.predict_spectral_indices(p, defs)
        return jnp.sum(((values - pred) / errors) ** 2)

    grid = np.linspace(0.0, 600.0, 61)
    s_map = float(grid[int(np.argmin([float(chi2(jnp.asarray(g))) for g in grid]))])
    grad = jax.grad(chi2)
    hess = jax.grad(lambda s: grad(s))
    assert float(hess(jnp.asarray(s_map))) > 0.0, "index chi^2 is flat in sigma_v"
    for _ in range(20):
        s_map = s_map - float(grad(jnp.asarray(s_map))) / float(hess(jnp.asarray(s_map)))
    # nll = chi2 / 2, so the posterior sd is 1 / sqrt(d2 nll / ds2) = sqrt(2 / d2 chi2 / ds2).
    sd = np.sqrt(2.0 / float(hess(jnp.asarray(s_map))))
    assert np.isfinite(sd) and sd > 0
    assert abs(s_map - truth) < sd, (s_map, sd)
