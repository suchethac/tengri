# SPDX-License-Identifier: BSD-3-Clause
"""The measured line-flux likelihood carries the nebular model's per-line error (#2941).

``sigma_eff = hypot(sigma_obs, f_cal |F_model|, s_line |F_model|)`` for detected
lines, with the ``ln sigma_eff`` term kept, and ``hypot(sigma_obs, s_line |F_model|)``
in the CDF of an upper or lower limit. The model error is a property of the model,
so it applies to limits as well as detections; ``f_cal`` stays detections-only.

Every expected value is a closed form written out here, not read back from the
code under test.
"""

from __future__ import annotations

import dataclasses
import types
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from tengri.config.exceptions import AdvisoryWarning
from tengri.inference.fitter import Fitter
from tengri.inference.likelihood import build_likelihood_extras
from tengri.inference.likelihoods import CensoredLikelihood, GaussianLikelihood
from tengri.inference.likelihoods.protocol import _CENSORED_LIKELIHOOD_CACHE_KEY_POLICY
from tengri.observation import LineFluxData, Observation, Photometry
from tengri.observation.noise import censored_neg_log_likelihood, compute_effective_noise
from tengri.observation.photometry import FilterCurve
from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.contract

# Normalized units: the functions are scale-free, and O(1) values keep the
# finite-difference check well conditioned.
D = np.array([1.2, 0.35, 0.2])  # observed fluxes
E = np.array([0.10, 0.05, 0.04])  # observed 1-sigma errors
M = np.array([1.1, 0.40, 0.15])  # model fluxes
F_CAL = 0.05
S = np.array([0.01, 0.045, 1.4])  # per-line fractional model sigma (Cue-like scale)
MASK_DET = np.zeros(3, dtype=np.int32)


def _det_energy(d, e, m, f_cal, s):
    """Closed form of the detected-line energy: sum[ 0.5 r^2 + ln sigma_eff ]."""
    se = np.sqrt(e**2 + (f_cal * m) ** 2 + (s * m) ** 2)
    r = (d - m) / se
    return float(np.sum(0.5 * r**2 + np.log(se)))


def _limit_energy_upper(d, e, m, s):
    """Closed form of an upper-limit energy: -ln Phi((d - m) / hypot(e, s |m|))."""
    sig = np.hypot(e, s * np.abs(m))
    return float(-stats.norm.logcdf((d - m) / sig).sum())


def _limit_energy_lower(d, e, m, s):
    sig = np.hypot(e, s * np.abs(m))
    return float(-stats.norm.logcdf((m - d) / sig).sum())


# ── (a) model_sigma_frac=None is the old behavior, bit for bit ───────────


class TestNoneIsOldBehavior:
    def test_none_equals_call_without_argument(self):
        mask = np.array([0, 1, -1], dtype=np.int32)
        d, e, m = D, E, M
        old = censored_neg_log_likelihood(
            jnp.asarray(d), jnp.asarray(e), jnp.asarray(m), jnp.asarray(mask), F_CAL
        )
        new = censored_neg_log_likelihood(
            jnp.asarray(d),
            jnp.asarray(e),
            jnp.asarray(m),
            jnp.asarray(mask),
            F_CAL,
            model_sigma_frac=None,
        )
        assert float(old) == float(new)

    def test_none_equals_pre_change_formula_exactly(self):
        """The detected branch is the same expression the function always used."""
        sigma_eff = compute_effective_noise(jnp.asarray(E), jnp.asarray(M), F_CAL)
        expected = float(
            jnp.sum(
                0.5 * ((jnp.asarray(D) - jnp.asarray(M)) / sigma_eff) ** 2 + jnp.log(sigma_eff)
            )
        )
        got = censored_neg_log_likelihood(
            jnp.asarray(D),
            jnp.asarray(E),
            jnp.asarray(M),
            jnp.asarray(MASK_DET),
            F_CAL,
            model_sigma_frac=None,
        )
        assert float(got) == expected


# ── (b) detections: exact closed form including ln sigma_eff ─────────────


class TestDetectionClosedForm:
    @pytest.mark.parametrize("s", [0.0, 0.045, S])
    def test_matches_closed_form(self, s):
        got = censored_neg_log_likelihood(
            jnp.asarray(D),
            jnp.asarray(E),
            jnp.asarray(M),
            jnp.asarray(MASK_DET),
            F_CAL,
            model_sigma_frac=jnp.asarray(s) * jnp.ones(3),
        )
        np.testing.assert_allclose(float(got), _det_energy(D, E, M, F_CAL, s), rtol=1e-12)

    def test_scalar_sigma_equals_array_sigma(self):
        args = (jnp.asarray(D), jnp.asarray(E), jnp.asarray(M), jnp.asarray(MASK_DET), F_CAL)
        scalar = censored_neg_log_likelihood(*args, model_sigma_frac=0.045)
        array = censored_neg_log_likelihood(*args, model_sigma_frac=jnp.full(3, 0.045))
        assert float(scalar) == float(array)

    def test_ln_sigma_term_is_present(self):
        """Without ln sigma_eff the energy would differ by sum(ln sigma_eff)."""
        se = np.sqrt(E**2 + (F_CAL * M) ** 2 + (S * M) ** 2)
        chi2_only = float(np.sum(0.5 * ((D - M) / se) ** 2))
        got = float(
            censored_neg_log_likelihood(
                jnp.asarray(D),
                jnp.asarray(E),
                jnp.asarray(M),
                jnp.asarray(MASK_DET),
                F_CAL,
                model_sigma_frac=jnp.asarray(S),
            )
        )
        np.testing.assert_allclose(got - chi2_only, float(np.sum(np.log(se))), rtol=1e-12)


# ── (c) limits: CDF uses hypot(sigma_obs, s |m|), f_cal excluded ─────────


class TestLimitClosedForm:
    def test_upper_limit_uses_model_sigma_in_cdf(self):
        d, e, m, s = D[:1], E[:1], M[:1], np.array([0.3])
        got = censored_neg_log_likelihood(
            jnp.asarray(d),
            jnp.asarray(e),
            jnp.asarray(m),
            jnp.asarray(np.array([1], dtype=np.int32)),
            0.2,  # f_cal: must NOT enter the limit
            model_sigma_frac=jnp.asarray(s),
        )
        np.testing.assert_allclose(float(got), _limit_energy_upper(d, e, m, s), rtol=1e-12)

    def test_upper_limit_ignores_f_cal(self):
        args = (
            jnp.asarray(D[:1]),
            jnp.asarray(E[:1]),
            jnp.asarray(M[:1]),
            jnp.asarray(np.array([1], dtype=np.int32)),
        )
        a = censored_neg_log_likelihood(*args, 0.0, model_sigma_frac=jnp.asarray([0.3]))
        b = censored_neg_log_likelihood(*args, 0.5, model_sigma_frac=jnp.asarray([0.3]))
        assert float(a) == float(b)

    def test_lower_limit_uses_model_sigma_in_cdf(self):
        d, e, m, s = D[1:2], E[1:2], M[1:2], np.array([0.7])
        got = censored_neg_log_likelihood(
            jnp.asarray(d),
            jnp.asarray(e),
            jnp.asarray(m),
            jnp.asarray(np.array([-1], dtype=np.int32)),
            F_CAL,
            model_sigma_frac=jnp.asarray(s),
        )
        np.testing.assert_allclose(float(got), _limit_energy_lower(d, e, m, s), rtol=1e-12)

    def test_mixed_detections_and_limits(self):
        mask = np.array([0, 1, -1], dtype=np.int32)
        got = censored_neg_log_likelihood(
            jnp.asarray(D),
            jnp.asarray(E),
            jnp.asarray(M),
            jnp.asarray(mask),
            F_CAL,
            model_sigma_frac=jnp.asarray(S),
        )
        det = _det_energy(D[:1], E[:1], M[:1], F_CAL, S[:1])
        up = _limit_energy_upper(D[1:2], E[1:2], M[1:2], S[1:2])
        lo = _limit_energy_lower(D[2:], E[2:], M[2:], S[2:])
        np.testing.assert_allclose(float(got), det + up + lo, rtol=1e-12)


# ── (d) gradient with respect to the model flux ──────────────────────────


class TestGradient:
    def test_gradient_wrt_model_flux_matches_finite_difference(self):
        mask = jnp.asarray(np.array([0, 1, -1], dtype=np.int32))

        def energy(m):
            return censored_neg_log_likelihood(
                jnp.asarray(D),
                jnp.asarray(E),
                m,
                mask,
                F_CAL,
                model_sigma_frac=jnp.asarray(S),
            )

        grad = assert_grad_matches_fd(energy, jnp.asarray(M))
        assert np.all(np.isfinite(np.asarray(grad))), (
            "gradient of the model-error likelihood wrt model flux is non-finite"
        )
        assert np.any(np.asarray(grad) != 0.0), (
            "gradient of the model-error likelihood wrt model flux is identically "
            "zero; the censored likelihood must respond to the model flux"
        )

    def test_gradient_of_detection_energy_is_exact(self):
        """Analytic check of d/dm of the detected energy, one line."""

        def energy(m):
            return censored_neg_log_likelihood(
                jnp.asarray(D[:1]),
                jnp.asarray(E[:1]),
                m,
                jnp.asarray(MASK_DET[:1]),
                F_CAL,
                model_sigma_frac=jnp.asarray(S[:1]),
            )

        m0 = float(M[0])
        got = float(jax.grad(energy)(jnp.asarray([m0]))[0])
        h = 1e-6
        fd = (
            _det_energy(D[:1], E[:1], np.array([m0 + h]), F_CAL, S[:1])
            - _det_energy(D[:1], E[:1], np.array([m0 - h]), F_CAL, S[:1])
        ) / (2 * h)
        np.testing.assert_allclose(got, fd, rtol=1e-6)


# ── (e) CensoredLikelihood: the data_args key and the cache-key policy ───


def _line_adapter(**kw):
    return CensoredLikelihood(
        obs=jnp.asarray(D),
        err=jnp.asarray(E),
        mask=jnp.asarray(MASK_DET),
        channel="line_fluxes",
        obs_key="line_flux_obs",
        err_key="line_flux_err",
        mask_key=None,
        **kw,
    )


class TestCensoredLikelihoodRouting:
    def test_key_reads_model_sigma_from_data_args(self):
        adapter = _line_adapter(model_sigma_frac_key="line_flux_model_sigma_frac")
        data_args = {
            "line_flux_obs": jnp.asarray(D),
            "line_flux_err": jnp.asarray(E),
            "line_flux_model_sigma_frac": jnp.asarray(S),
        }
        got = adapter.log_prob({"line_fluxes": jnp.asarray(M)}, None, data_args)
        np.testing.assert_allclose(float(got), -_det_energy(D, E, M, 0.0, S), rtol=1e-12)

    def test_baked_array_is_used_when_no_key(self):
        adapter = _line_adapter(model_sigma_frac=jnp.asarray(S))
        got = adapter.log_prob({"line_fluxes": jnp.asarray(M)})
        np.testing.assert_allclose(float(got), -_det_energy(D, E, M, 0.0, S), rtol=1e-12)

    def test_no_model_sigma_is_the_old_adapter(self):
        adapter = _line_adapter()
        got = adapter.log_prob({"line_fluxes": jnp.asarray(M)})
        np.testing.assert_allclose(float(got), -_det_energy(D, E, M, 0.0, 0.0), rtol=1e-12)

    def test_policy_covers_every_field(self):
        names = {f.name for f in dataclasses.fields(CensoredLikelihood)}
        assert names == set(_CENSORED_LIKELIHOOD_CACHE_KEY_POLICY)
        assert {"model_sigma_frac", "model_sigma_frac_key"} <= names

    def test_cache_key_differs_with_baked_values(self):
        a = _line_adapter(model_sigma_frac=jnp.asarray(S))
        b = _line_adapter(model_sigma_frac=jnp.asarray(S * 2.0))
        assert a.cache_key() != b.cache_key()

    def test_cache_key_ignores_data_args_values_behind_a_key(self):
        """Values behind the key are traced inputs: same program, same cache key."""
        a = _line_adapter(model_sigma_frac_key="line_flux_model_sigma_frac")
        b = _line_adapter(model_sigma_frac_key="line_flux_model_sigma_frac")
        assert a.cache_key() == b.cache_key()
        base = {"line_flux_obs": jnp.asarray(D), "line_flux_err": jnp.asarray(E)}
        lo = a.log_prob(
            {"line_fluxes": jnp.asarray(M)},
            None,
            {**base, "line_flux_model_sigma_frac": jnp.asarray(S)},
        )
        hi = b.log_prob(
            {"line_fluxes": jnp.asarray(M)},
            None,
            {**base, "line_flux_model_sigma_frac": jnp.asarray(2.0 * S)},
        )
        assert float(lo) != float(hi)
        np.testing.assert_allclose(float(hi), -_det_energy(D, E, M, 0.0, 2.0 * S), rtol=1e-12)


# ── (g) build_likelihood_extras routing ──────────────────────────────────


def _context(data_args):
    return types.SimpleNamespace(data_args=data_args)


class TestBuildLikelihoodExtras:
    def test_key_without_limit_mask_gives_censored_adapter(self):
        data_args = {
            "line_flux_waves": jnp.asarray([6564.61, 4862.68, 5008.24]),
            "line_flux_obs": jnp.asarray(D),
            "line_flux_err": jnp.asarray(E),
            "line_flux_model_sigma_frac": jnp.asarray(S),
        }
        (adapter,) = build_likelihood_extras(_context(data_args))
        assert isinstance(adapter, CensoredLikelihood)
        got = adapter.log_prob({"line_fluxes": jnp.asarray(M)}, None, data_args)
        np.testing.assert_allclose(float(got), -_det_energy(D, E, M, 0.0, S), rtol=1e-12)

    def test_key_with_limit_mask_scores_limits_too(self):
        mask = np.array([0, 1, -1], dtype=np.int32)
        data_args = {
            "line_flux_waves": jnp.asarray([6564.61, 4862.68, 5008.24]),
            "line_flux_obs": jnp.asarray(D),
            "line_flux_err": jnp.asarray(E),
            "line_flux_limit_mask": jnp.asarray(mask),
            "line_flux_model_sigma_frac": jnp.asarray(S),
        }
        (adapter,) = build_likelihood_extras(_context(data_args))
        got = adapter.log_prob({"line_fluxes": jnp.asarray(M)}, None, data_args)
        expected = (
            _det_energy(D[:1], E[:1], M[:1], 0.0, S[:1])
            + _limit_energy_upper(D[1:2], E[1:2], M[1:2], S[1:2])
            + _limit_energy_lower(D[2:], E[2:], M[2:], S[2:])
        )
        np.testing.assert_allclose(float(got), -expected, rtol=1e-12)

    def test_without_key_gaussian_is_unchanged(self):
        data_args = {
            "line_flux_waves": jnp.asarray([6564.61, 4862.68, 5008.24]),
            "line_flux_obs": jnp.asarray(D),
            "line_flux_err": jnp.asarray(E),
        }
        (adapter,) = build_likelihood_extras(_context(data_args))
        assert isinstance(adapter, GaussianLikelihood)
        got = adapter.log_prob({"line_fluxes": jnp.asarray(M)}, None, data_args)
        np.testing.assert_allclose(float(got), -0.5 * np.sum(((D - M) / E) ** 2), rtol=1e-12)


# ── (f) SEDModel.line_model_sigma_frac: host-side matching and gating ────


_FULL_WAVES = np.array([4862.68, 5008.24, 6564.61, 4000.0])
_FULL_SIGMA = np.array([0.010, 0.045, 0.010, 0.200])
_SUBSET_WAVES = np.array([4862.68, 6564.61])
_SUBSET_SIGMA = np.array([0.011, 0.020])


class _FakeCueLike:
    """Backend with a per-flag catalog and a per-flag model sigma."""

    def __init__(self, *, hook_returns=("array",)):
        self.flags = []
        self._hook_returns = hook_returns

    def published_line_wavelengths(self, *, cloudyfsps_only):
        return _SUBSET_WAVES if cloudyfsps_only else _FULL_WAVES

    def published_line_model_sigma(self, *, cloudyfsps_only):
        self.flags.append(cloudyfsps_only)
        if self._hook_returns == ("none",):
            return None
        return _SUBSET_SIGMA if cloudyfsps_only else _FULL_SIGMA


class _NoHookBackend:
    def published_line_wavelengths(self, *, cloudyfsps_only):
        return _FULL_WAVES


def _stub_model(backend, cue_full_catalog=True):
    from tengri.forward.sed_model import SEDModel

    stub = types.SimpleNamespace(
        _nebular_backend=backend,
        spec=types.SimpleNamespace(cue_full_catalog=cue_full_catalog),
    )
    return stub, SEDModel.line_model_sigma_frac


class TestLineModelSigmaFrac:
    def test_nearest_match_returns_matched_catalog_sigma(self):
        backend = _FakeCueLike()
        stub, method = _stub_model(backend, cue_full_catalog=True)
        got = method(stub, np.array([4861.9, 5007.9, 6563.5]))
        np.testing.assert_array_equal(got, np.array([0.010, 0.045, 0.010]))
        assert got.dtype == np.float64

    def test_cloudyfsps_flag_follows_cue_full_catalog(self):
        backend = _FakeCueLike()
        # cue_full_catalog=False means the legacy subset (cloudyfsps_only=True).
        stub, method = _stub_model(backend, cue_full_catalog=False)
        got = method(stub, np.array([4861.9, 6563.5]))
        np.testing.assert_array_equal(got, np.array([0.011, 0.020]))
        assert backend.flags == [True]

    def test_full_catalog_flag_selects_full_arrays(self):
        backend = _FakeCueLike()
        stub, method = _stub_model(backend, cue_full_catalog=True)
        got = method(stub, np.array([5007.9]))
        np.testing.assert_array_equal(got, np.array([0.045]))
        assert backend.flags == [False]

    def test_subset_flag_cannot_reach_a_line_only_full_catalog_has(self):
        backend = _FakeCueLike()
        stub, method = _stub_model(backend, cue_full_catalog=False)
        with pytest.raises(ValueError, match=r"tolerance_aa=5\.0"):
            method(stub, np.array([5007.9]))

    def test_zeros_without_hook(self):
        stub, method = _stub_model(_NoHookBackend())
        np.testing.assert_array_equal(method(stub, np.array([4862.68, 6564.61])), np.zeros(2))

    def test_zeros_and_advisory_when_hook_returns_none(self):
        stub, method = _stub_model(_FakeCueLike(hook_returns=("none",)))
        with pytest.warns(AdvisoryWarning, match="emulator"):
            got = method(stub, np.array([4862.68, 6564.61]))
        np.testing.assert_array_equal(got, np.zeros(2))

    def test_raises_beyond_tolerance(self):
        stub, method = _stub_model(_FakeCueLike())
        with pytest.raises(ValueError, match=r"tolerance_aa=5\.0"):
            method(stub, np.array([5500.0]))

    def test_tolerance_none_disables_the_check(self):
        stub, method = _stub_model(_FakeCueLike())
        got = method(stub, np.array([5500.0]), tolerance_aa=None)
        np.testing.assert_array_equal(got, np.array([0.045]))

    def test_misaligned_hook_array_is_refused(self):
        class _Bad(_FakeCueLike):
            def published_line_model_sigma(self, *, cloudyfsps_only):
                return np.array([0.1, 0.2])

        stub, method = _stub_model(_Bad())
        with pytest.raises(ValueError, match="shape"):
            method(stub, np.array([4862.68]))


# ── (h) Fitter packs the key only when it should ─────────────────────────


def _mock_line_model(sigma_return, *, with_lines=True):
    lf = LineFluxData.from_dict({"Halpha": (1.2e-16, 0.1e-16), "Hbeta": (3.5e-17, 0.5e-17)})
    wave = jnp.linspace(6200.0 - 1000.0, 6200.0 + 1000.0, 50)
    filt = FilterCurve(
        wave=wave, trans=jnp.exp(-0.5 * ((wave - 6200.0) / 500.0) ** 2), name="test_r"
    )
    obs = Observation(
        photometry=Photometry(filters=(filt,), names=("test_r",)),
        line_fluxes=lf if with_lines else None,
    )
    model = MagicMock()
    model.observation = obs
    model.spec = MagicMock()
    model.spec.free_params = []
    model.spec.get_fixed_values.return_value = {}
    model.spec.stochastic = False
    model.spec.resolve_mirrors = lambda x: x
    model._precomputed = MagicMock()
    model._precomputed.photometry = None
    model._z_fixed = None
    model.line_model_sigma_frac = MagicMock(return_value=sigma_return)
    return model


def _build(model, **kw):
    return Fitter(
        model,
        data=jnp.array([1.0]),
        noise=jnp.array([0.1]),
        data_type="photometry",
        **kw,
    )


class TestFitterPacking:
    def test_nonzero_sigma_is_packed_exactly(self):
        model = _mock_line_model(np.array([0.01, 0.045]))
        fitter = _build(model)
        np.testing.assert_array_equal(
            np.asarray(fitter._data_args["line_flux_model_sigma_frac"]), [0.01, 0.045]
        )
        waves = fitter._resolved_line_fluxes().wavelengths
        np.testing.assert_array_equal(model.line_model_sigma_frac.call_args.args[0], waves)

    def test_all_zero_sigma_is_not_packed(self):
        fitter = _build(_mock_line_model(np.zeros(2)))
        assert "line_flux_model_sigma_frac" not in fitter._data_args

    def test_opt_out_never_packs_and_never_asks(self):
        model = _mock_line_model(np.array([0.01, 0.045]))
        fitter = _build(model, line_model_sigma=False)
        assert "line_flux_model_sigma_frac" not in fitter._data_args
        assert not model.line_model_sigma_frac.called

    def test_no_line_fluxes_no_key(self):
        model = _mock_line_model(np.array([0.01, 0.045]), with_lines=False)
        fitter = _build(model)
        assert "line_flux_model_sigma_frac" not in fitter._data_args
        assert not model.line_model_sigma_frac.called


# ── (i) profile_mass refuses a nonzero model error ───────────────────────


def _real_fitter(ssp_data, monkeypatch, *, sigma=None):
    from tengri import ForwardModel, SEDModel, builders, generate_mock, recipes

    filters = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "des_g", "des_r", "des_i"]
    lines = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in ("Halpha", "Hbeta", "OIII_5007")})
    obs = Observation(photometry=Photometry.from_names(filters), line_fluxes=lines)
    recipe = recipes.mock_recovery_minimal()
    recipe["neb"] = builders.neb.ssp()
    model = SEDModel.build(ssp_data=ssp_data, observation=obs, **recipe)
    if sigma is not None:
        # Patched on the class: Fitter's approx="auto" rebuilds the SEDModel, so
        # an instance-level patch would not reach the object the guard reads.
        monkeypatch.setattr(
            SEDModel,
            "line_model_sigma_frac",
            lambda self, w, **kw: np.full(len(np.asarray(w)), sigma),
        )
    key_truth, key_mock = jax.random.split(jax.random.PRNGKey(0))
    truth = model.spec.sample(key_truth)
    mock = generate_mock(model, truth, key=key_mock, snr=30.0)
    return Fitter(
        ForwardModel.build(sed=model, observation=obs),
        jnp.asarray(mock["flux_obs"]),
        jnp.asarray(mock["noise"]),
        profile_mass="auto",
    )


class TestProfileMassRefusal:
    def test_control_without_model_error_engages(self, ssp_data_wne, monkeypatch):
        fitter = _real_fitter(ssp_data_wne, monkeypatch, sigma=None)
        assert fitter._profile_mass, fitter._profile_mass_reason

    def test_nonzero_model_error_is_refused_by_name(self, ssp_data_wne, monkeypatch):
        fitter = _real_fitter(ssp_data_wne, monkeypatch, sigma=0.045)
        assert not fitter._profile_mass
        assert "line_model_sigma" in (fitter._profile_mass_reason or "")
