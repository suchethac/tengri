# SPDX-License-Identifier: BSD-3-Clause
"""Tests for emission line flux fitting as direct observables.

Covers LineFluxData construction/validation, Observation integration,
and the additive chi2 contribution in the loss/loglikelihood functions.
No SSP data needed — uses mocks for the forward model.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds
jax.config.update("jax_enable_x64", True)

from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.observation import Observation
from tengri.observation.photometry import FilterCurve
from tengri.observation.photometry_config import Photometry
from tests._grad_parity import assert_grad_matches_fd

# ── Helpers ───────────────────────────────────────────────────────


def _make_filter(name="test_r", center=6200.0, width=500.0, n=50):
    wave = jnp.linspace(center - 2 * width, center + 2 * width, n)
    trans = jnp.exp(-0.5 * ((wave - center) / width) ** 2)
    return FilterCurve(wave=wave, trans=trans, name=name)


# ── LineFluxData ──────────────────────────────────────────────────


class TestLineFluxData:
    def test_from_dict_basic(self):
        lf = LineFluxData.from_dict(
            {
                "Halpha": (1.2e-16, 0.1e-16),
                "Hbeta": (3.5e-17, 0.5e-17),
            }
        )
        assert lf.n_lines == 2
        assert lf.names == ("Halpha", "Hbeta")
        np.testing.assert_allclose(lf.fluxes, [1.2e-16, 3.5e-17])
        np.testing.assert_allclose(lf.errors, [0.1e-16, 0.5e-17])

    def test_from_dict_wavelength_lookup(self):
        lf = LineFluxData.from_dict(
            {
                "Halpha": (1.0e-16, 0.1e-16),
                "OIII_5007": (5.0e-17, 0.5e-17),
            }
        )
        np.testing.assert_allclose(lf.wavelengths[0], 6564.61, atol=0.1)
        np.testing.assert_allclose(lf.wavelengths[1], 5008.24, atol=0.1)

    def test_from_dict_unknown_line_raises(self):
        with pytest.raises(ValueError, match="Unknown line name"):
            LineFluxData.from_dict({"NotARealLine": (1.0, 0.1)})

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one line"):
            LineFluxData(
                names=(),
                fluxes=jnp.array([]),
                errors=jnp.array([]),
                wavelengths=jnp.array([]),
            )

    def test_shape_mismatch_fluxes(self):
        with pytest.raises(ValueError, match="fluxes shape"):
            LineFluxData(
                names=("Halpha",),
                fluxes=jnp.array([1.0, 2.0]),
                errors=jnp.array([0.1]),
                wavelengths=jnp.array([6564.61]),
            )

    def test_shape_mismatch_errors(self):
        with pytest.raises(ValueError, match="errors shape"):
            LineFluxData(
                names=("Halpha",),
                fluxes=jnp.array([1.0]),
                errors=jnp.array([0.1, 0.2]),
                wavelengths=jnp.array([6564.61]),
            )

    def test_shape_mismatch_wavelengths(self):
        with pytest.raises(ValueError, match="wavelengths shape"):
            LineFluxData(
                names=("Halpha",),
                fluxes=jnp.array([1.0]),
                errors=jnp.array([0.1]),
                wavelengths=jnp.array([6564.61, 5008.24]),
            )

    def test_frozen(self):
        lf = LineFluxData.from_dict({"Halpha": (1.0e-16, 0.1e-16)})
        with pytest.raises(AttributeError):
            lf.names = ("Hbeta",)

    def test_summary(self):
        lf = LineFluxData.from_dict(
            {
                "Halpha": (1.0e-16, 0.1e-16),
                "Hbeta": (3.5e-17, 0.5e-17),
            }
        )
        s = lf.summary()
        assert "2 lines" in s
        assert "Halpha" in s
        assert "Hbeta" in s


# ── Observation integration ───────────────────────────────────────


class TestTopLevelExport:
    def test_line_flux_data_importable(self):
        from tengri.observation import LineFluxData as LFD

        assert LFD is LineFluxData


class TestObservationLineFluxes:
    def test_photometry_plus_line_fluxes(self):
        filt = _make_filter()
        phot = Photometry(
            filters=(filt,),
            names=("test_r",),
        )
        lf = LineFluxData.from_dict({"Halpha": (1.0e-16, 0.1e-16)})
        obs = Observation(photometry=phot, line_fluxes=lf)
        assert obs.has_line_fluxes
        assert obs.n_data_lines == 1
        assert obs.n_data == 1 + 1  # 1 filter + 1 line
        assert obs.data_type == "photometry"

    def test_line_fluxes_only(self):
        lf = LineFluxData.from_dict({"Halpha": (1.0e-16, 0.1e-16)})
        obs = Observation(line_fluxes=lf)
        assert obs.has_line_fluxes
        assert obs.n_data_lines == 1
        assert obs.n_data == 1

    def test_no_line_fluxes(self):
        filt = _make_filter()
        phot = Photometry(
            filters=(filt,),
            names=("test_r",),
        )
        obs = Observation(photometry=phot)
        assert not obs.has_line_fluxes
        assert obs.n_data_lines == 0

    def test_summary_includes_lines(self):
        filt = _make_filter()
        phot = Photometry(
            filters=(filt,),
            names=("test_r",),
        )
        lf = LineFluxData.from_dict(
            {
                "Halpha": (1.0e-16, 0.1e-16),
                "OIII_5007": (5.0e-17, 0.5e-17),
            }
        )
        obs = Observation(photometry=phot, line_fluxes=lf)
        s = obs.summary()
        assert "line fluxes" in s.lower() or "Line fluxes" in s

    def test_validation_none_of_three_raises(self):
        with pytest.raises(ValueError):
            Observation()


# ── Chi2 computation (direct unit test) ───────────────────────────


class TestLineFluxChi2:
    def test_chi2_perfect_match(self):
        """When model == observed, chi2 should be 0."""
        lf = LineFluxData(
            names=("Halpha", "Hbeta"),
            fluxes=jnp.array([1.0e-16, 5.0e-17]),
            errors=jnp.array([0.1e-16, 0.5e-17]),
            wavelengths=jnp.array([6564.61, 4862.68]),
        )
        model = jnp.array([1.0e-16, 5.0e-17])
        assert lf.chi2(model) == pytest.approx(0.0, abs=1e-30)

    def test_chi2_one_sigma_offset(self):
        """One-sigma offset on each line → chi2 = n_lines."""
        lf = LineFluxData(
            names=("Halpha", "Hbeta"),
            fluxes=jnp.array([1.0e-16, 5.0e-17]),
            errors=jnp.array([0.1e-16, 0.5e-17]),
            wavelengths=jnp.array([6564.61, 4862.68]),
        )
        model = lf.fluxes + lf.errors
        assert lf.chi2(model) == pytest.approx(2.0, abs=1e-10)

    def test_chi2_differentiable(self):
        """chi2() is differentiable w.r.t. model fluxes."""
        lf = LineFluxData(
            names=("Halpha",),
            fluxes=jnp.array([1.0e-16]),
            errors=jnp.array([0.1e-16]),
            wavelengths=jnp.array([6564.61]),
        )

        grad = assert_grad_matches_fd(lf.chi2, jnp.array([0.9e-16]))
        assert jnp.isfinite(grad).all()
        assert grad[0] != 0.0

    def test_log_likelihood_perfect_match(self):
        """Log-likelihood at exact match equals the normalization term."""
        lf = LineFluxData(
            names=("Halpha",),
            fluxes=jnp.array([1.0e-16]),
            errors=jnp.array([0.1e-16]),
            wavelengths=jnp.array([6564.61]),
        )
        ll = lf.log_likelihood(lf.fluxes)
        expected = -jnp.log(lf.errors[0]) - 0.5 * jnp.log(2.0 * jnp.pi)
        np.testing.assert_allclose(float(ll), float(expected), atol=1e-10)

    def test_log_likelihood_decreases_with_offset(self):
        """Log-likelihood decreases as model moves away from observed."""
        lf = LineFluxData(
            names=("Halpha",),
            fluxes=jnp.array([1.0e-16]),
            errors=jnp.array([0.1e-16]),
            wavelengths=jnp.array([6564.61]),
        )
        ll_exact = lf.log_likelihood(lf.fluxes)
        ll_offset = lf.log_likelihood(lf.fluxes + 2 * lf.errors)
        assert float(ll_exact) > float(ll_offset)

    def test_log_likelihood_differentiable(self):
        """log_likelihood() is differentiable."""
        lf = LineFluxData(
            names=("Halpha",),
            fluxes=jnp.array([1.0e-16]),
            errors=jnp.array([0.1e-16]),
            wavelengths=jnp.array([6564.61]),
        )
        grad = assert_grad_matches_fd(lf.log_likelihood, jnp.array([0.9e-16]))
        assert jnp.isfinite(grad).all()


class TestLineFluxUpperLimits:
    def test_upper_limit_field(self):
        lf = LineFluxData(
            names=("Halpha", "NII_6583"),
            fluxes=jnp.array([1.0e-16, 3.0e-17]),
            errors=jnp.array([0.1e-16, 1.0e-17]),
            wavelengths=jnp.array([6564.61, 6585.27]),
            is_upper_limit=jnp.array([False, True]),
        )
        assert lf.n_lines == 2

    def test_upper_limit_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="is_upper_limit shape"):
            LineFluxData(
                names=("Halpha",),
                fluxes=jnp.array([1.0e-16]),
                errors=jnp.array([0.1e-16]),
                wavelengths=jnp.array([6564.61]),
                is_upper_limit=jnp.array([False, True]),
            )

    def test_chi2_excludes_upper_limits(self):
        lf = LineFluxData(
            names=("Halpha", "NII_6583"),
            fluxes=jnp.array([1.0e-16, 3.0e-17]),
            errors=jnp.array([0.1e-16, 1.0e-17]),
            wavelengths=jnp.array([6564.61, 6585.27]),
            is_upper_limit=jnp.array([False, True]),
        )
        model = lf.fluxes + lf.errors
        chi2 = lf.chi2(model)
        assert chi2 == pytest.approx(1.0, abs=1e-10)

    def test_log_likelihood_upper_limit_below_limit(self):
        """SEDModel flux below upper limit → high likelihood (close to 0)."""
        lf = LineFluxData(
            names=("NII_6583",),
            fluxes=jnp.array([3.0e-17]),
            errors=jnp.array([1.0e-17]),
            wavelengths=jnp.array([6585.27]),
            is_upper_limit=jnp.array([True]),
        )
        ll_below = lf.log_likelihood(jnp.array([0.0]))
        ll_above = lf.log_likelihood(jnp.array([6.0e-17]))
        assert float(ll_below) > float(ll_above)

    def test_log_likelihood_mixed(self):
        """Mixed detected + upper limit log-likelihood is differentiable."""
        lf = LineFluxData(
            names=("Halpha", "NII_6583"),
            fluxes=jnp.array([1.0e-16, 3.0e-17]),
            errors=jnp.array([0.1e-16, 1.0e-17]),
            wavelengths=jnp.array([6564.61, 6585.27]),
            is_upper_limit=jnp.array([False, True]),
        )
        model = jnp.array([1.0e-16, 1.0e-17])
        ll = lf.log_likelihood(model)
        assert jnp.isfinite(ll)

        grad = assert_grad_matches_fd(lf.log_likelihood, model)
        assert jnp.isfinite(grad).all()


# ── Fitter data_args packing ──────────────────────────────────────


class TestFitterDataArgsPacking:
    def test_line_flux_data_in_data_args(self):
        """Verify that line flux arrays are packed into _data_args."""
        from unittest.mock import MagicMock

        filt = _make_filter()
        phot = Photometry(
            filters=(filt,),
            names=("test_r",),
        )
        lf = LineFluxData.from_dict(
            {
                "Halpha": (1.2e-16, 0.1e-16),
                "Hbeta": (3.5e-17, 0.5e-17),
            }
        )
        obs = Observation(photometry=phot, line_fluxes=lf)

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

        from tengri.inference.fitter import Fitter

        fitter = Fitter(
            model,
            data=jnp.array([1.0]),
            noise=jnp.array([0.1]),
            data_type="photometry",
        )

        assert "line_flux_obs" in fitter._data_args
        assert "line_flux_err" in fitter._data_args
        assert "line_flux_waves" in fitter._data_args
        np.testing.assert_allclose(fitter._data_args["line_flux_obs"], [1.2e-16, 3.5e-17])
        np.testing.assert_allclose(fitter._data_args["line_flux_err"], [0.1e-16, 0.5e-17])

    def test_no_line_flux_not_in_data_args(self):
        """Without line_fluxes, keys should not be in _data_args."""
        from unittest.mock import MagicMock

        filt = _make_filter()
        phot = Photometry(
            filters=(filt,),
            names=("test_r",),
        )
        obs = Observation(photometry=phot)

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

        from tengri.inference.fitter import Fitter

        fitter = Fitter(
            model,
            data=jnp.array([1.0]),
            noise=jnp.array([0.1]),
            data_type="photometry",
        )

        assert "line_flux_obs" not in fitter._data_args
        assert "line_flux_err" not in fitter._data_args
        assert "line_flux_waves" not in fitter._data_args
