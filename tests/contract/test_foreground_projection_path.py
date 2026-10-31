# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests: the Milky Way foreground screen on the observed-frame projection path (#3140).

The screen multiplies the observed SED by ``10**(-0.4 R_V E(B-V) k(lambda))``. It is
applied in three places that must agree: the observed-frame SED (``predict_obs_sed``),
the photometric projection (``project_photometry``, which covers the photometric block
of ``Observation.predict``), and the spectroscopic projection (through the observed
transmission passed to ``project_spectrum_kernel_split``).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri.components.dust.attenuation import cardelli
from tengri.components.dust.foreground import FOREGROUND_LAWS, foreground_transmission
from tengri.observation.photometry import project_photometry
from tengri.parameters.groups import _VALID_FOREGROUND_LAWS, parse_groups

pytestmark = pytest.mark.contract

_EBMV = 0.1
_BANDS = ["des_g", "des_r", "des_i"]


def _build(ssp, foreground=None, sfh_all=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kwargs = {}
        if foreground is not None:
            kwargs["foreground"] = foreground
        sfh = {"type": "tsnorm", "all_params": sfh_all or tengri.Fixed(tengri.DEFAULT)}
        return tengri.SEDModel.build(
            ssp,
            observation=tengri.Observation(photometry=tengri.Photometry.from_names(_BANDS)),
            sfh=sfh,
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            redshift=tengri.Fixed(0.05),
            **kwargs,
        )


@pytest.fixture(scope="module")
def ssp(synthetic_ssp_wide):
    return synthetic_ssp_wide


@pytest.fixture(scope="module")
def baseline(ssp):
    return _build(ssp)


@pytest.fixture(scope="module")
def cardelli_model(ssp):
    return _build(ssp, foreground={"ebmv_mw": _EBMV, "law": "cardelli", "rv": 3.1})


@pytest.fixture(scope="module")
def fitz_model(ssp):
    return _build(ssp, foreground={"ebmv_mw": _EBMV, "law": "fitzpatrick99", "rv": 3.1})


@pytest.fixture(scope="module")
def sample(baseline):
    return dict(baseline.spec.sample(jax.random.PRNGKey(0)))


def _reference_transmission(wave_obs, ebmv, rv):
    """The pre-refactor inline screen expression, reproduced here for bit-level comparison."""
    k_lambda = cardelli(wave_obs, dust_Rv=rv)
    a_v = rv * ebmv
    return jnp.power(10.0, -0.4 * a_v * k_lambda)


class TestScreenFunction:
    def test_cardelli_matches_inline_expression_bit_for_bit(self):
        wave = jnp.asarray(np.geomspace(1000.0, 3.0e4, 257))
        got = foreground_transmission(wave, _EBMV, 3.1, "cardelli")
        expected = _reference_transmission(wave, _EBMV, 3.1)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))

    def test_zero_ebmv_is_exactly_one(self):
        wave = jnp.asarray(np.geomspace(1000.0, 3.0e4, 64))
        got = foreground_transmission(wave, 0.0, 3.1, "fitzpatrick99")
        np.testing.assert_array_equal(np.asarray(got), np.ones(64))

    def test_fitzpatrick99_transmission_is_below_one_and_rises_to_red(self):
        wave = jnp.asarray(np.geomspace(1500.0, 2.0e4, 200))
        t = np.asarray(foreground_transmission(wave, _EBMV, 3.1, "fitzpatrick99"))
        assert np.all(t <= 1.0)
        assert np.all(t > 0.0)
        assert t[0] < t[-1]

    def test_unknown_law_raises(self):
        with pytest.raises(ValueError, match="foreground law"):
            foreground_transmission(jnp.asarray([5000.0]), _EBMV, 3.1, "not-a-law")


class TestForegroundLawRegistry:
    def test_validated_names_equal_dispatch_table(self):
        assert frozenset(FOREGROUND_LAWS) == _VALID_FOREGROUND_LAWS

    def test_fitzpatrick99_accepted_by_grammar(self):
        params = parse_groups(
            foreground={"ebmv_mw": 0.05, "law": "fitzpatrick99"},
            redshift=tengri.Fixed(0.05),
        )
        assert params.foreground_law == "fitzpatrick99"


class TestObservedSedPath:
    def test_cardelli_observed_sed_is_baseline_times_reference_transmission(
        self, cardelli_model, baseline, sample
    ):
        base = baseline.predict_obs_sed(sample)
        fg = cardelli_model.predict_obs_sed(sample)
        expected = np.asarray(base.sed) * np.asarray(
            _reference_transmission(base.wavelength, _EBMV, 3.1)
        )
        np.testing.assert_allclose(np.asarray(fg.sed), expected, rtol=1e-12, atol=0.0)

    def test_fitzpatrick99_observed_sed_is_baseline_times_transmission(
        self, fitz_model, baseline, sample
    ):
        base = baseline.predict_obs_sed(sample)
        fg = fitz_model.predict_obs_sed(sample)
        t = np.asarray(foreground_transmission(base.wavelength, _EBMV, 3.1, "fitzpatrick99"))
        np.testing.assert_allclose(np.asarray(fg.sed), np.asarray(base.sed) * t, rtol=1e-12)


class TestPhotometricPath:
    def test_band_flux_equals_screen_applied_once_to_the_sed(self, fitz_model, sample):
        """Band fluxes equal the unreddened SED times the screen, applied exactly once."""
        full = {**sample, "redshift": fitz_model._get_redshift(sample)}
        z = full["redshift"]
        state = fitz_model.predict_state(sample)
        screen = foreground_transmission(state.wave * (1.0 + z), _EBMV, 3.1, "fitzpatrick99")
        state_ref = state.with_(
            sed_intrinsic=state.sed_intrinsic * screen,
            derived=state.derived.with_(foreground_transmission=None),
        )
        expected = project_photometry(state_ref, full, fitz_model.observation.photometry)
        got = fitz_model.observation.predict(state, full)["phot_fnu"]
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-10, atol=0.0)

    def test_band_flux_is_reddened_relative_to_baseline(self, fitz_model, baseline, sample):
        full_fg = {**sample, "redshift": fitz_model._get_redshift(sample)}
        full_base = {**sample, "redshift": baseline._get_redshift(sample)}
        got = np.asarray(
            fitz_model.observation.predict(fitz_model.predict_state(sample), full_fg)["phot_fnu"]
        )
        base = np.asarray(
            baseline.observation.predict(baseline.predict_state(sample), full_base)["phot_fnu"]
        )
        ratio = got / base
        assert np.all(ratio > 0.0)
        assert np.all(ratio <= 1.0 + 1e-12)
        assert np.any(ratio < 0.999)


class TestGradient:
    @pytest.mark.parametrize("x64", [True, False])
    def test_band_flux_gradient_wrt_log_total_mass_is_finite_and_nonzero(self, ssp, x64):
        free_model = _build(
            ssp,
            foreground={"ebmv_mw": _EBMV, "law": "fitzpatrick99", "rv": 3.1},
            sfh_all=tengri.FREE,
        )
        name = next(n for n in free_model.spec.free_params if n.endswith("log_total_mass"))
        base_sample = dict(free_model.spec.sample(jax.random.PRNGKey(1)))

        def band_sum(log_mass):
            params = {**base_sample, name: log_mass}
            state = free_model.predict_state(params)
            full = {**params, "redshift": free_model._get_redshift(params)}
            return jnp.sum(free_model.observation.predict(state, full)["phot_fnu"])

        with jax.enable_x64(x64):
            dtype = jnp.float64 if x64 else jnp.float32
            grad = jax.grad(band_sum)(jnp.asarray(base_sample[name], dtype=dtype))
        assert np.isfinite(float(grad))
        assert float(grad) != 0.0
