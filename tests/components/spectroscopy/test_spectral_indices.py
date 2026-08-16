# SPDX-License-Identifier: BSD-3-Clause
"""Tests for spectral index fitting.

Covers SpectralIndexDef validation, SpectralIndexData construction,
EW and break measurement on synthetic spectra, Observation integration,
and fitter data_args packing. No SSP data needed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds

from tengri.observation.photometry import FilterCurve
from tengri.observation.photometry_config import Photometry
from tengri.observation.spectral_indices import (
    STANDARD_INDICES,
    SpectralIndexData,
    SpectralIndexDef,
    measure_index_jax,
)
from tests._grad_parity import assert_grad_matches_fd

# ── Helpers ───────────────────────────────────────────────────────


def _make_filter(name="test_r", center=6200.0, width=500.0, n=50):
    wave = jnp.linspace(center - 2 * width, center + 2 * width, n)
    trans = jnp.exp(-0.5 * ((wave - center) / width) ** 2)
    return FilterCurve(wave=wave, trans=trans, name=name)


# ── SpectralIndexDef ──────────────────────────────────────────────


class TestSpectralIndexDef:
    def test_break_definition(self):
        idx = SpectralIndexDef(
            name="test_break",
            index_type="break",
            continuum=((3850.0, 3950.0), (4000.0, 4100.0)),
        )
        assert idx.wave_min == 3850.0
        assert idx.wave_max == 4100.0

    def test_ew_definition(self):
        idx = SpectralIndexDef(
            name="test_ew",
            index_type="EW",
            continuum=((4041.0, 4080.0), (4128.0, 4161.0)),
            feature=(4083.0, 4122.0),
        )
        assert idx.wave_min == 4041.0
        assert idx.wave_max == 4161.0

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="index_type must be"):
            SpectralIndexDef(
                name="bad",
                index_type="invalid",
                continuum=((1.0, 2.0),),
            )

    def test_ew_without_feature_raises(self):
        with pytest.raises(ValueError, match="feature window"):
            SpectralIndexDef(
                name="bad",
                index_type="EW",
                continuum=((1.0, 2.0), (3.0, 4.0)),
            )

    def test_break_wrong_windows_raises(self):
        with pytest.raises(ValueError, match="2 continuum windows"):
            SpectralIndexDef(
                name="bad",
                index_type="break",
                continuum=((1.0, 2.0),),
            )

    def test_frozen(self):
        idx = STANDARD_INDICES["Dn4000"]
        with pytest.raises(AttributeError):
            idx.name = "modified"


# ── SpectralIndexData ─────────────────────────────────────────────


class TestSpectralIndexData:
    def test_from_names_basic(self):
        sid = SpectralIndexData.from_names(
            names=["Dn4000", "HdA"],
            values=[1.8, -1.2],
            errors=[0.05, 0.3],
        )
        assert sid.n_indices == 2
        assert sid.names == ("Dn4000", "HdA")
        np.testing.assert_allclose(sid.values, [1.8, -1.2])

    def test_from_names_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown index name"):
            SpectralIndexData.from_names(
                names=["NotAnIndex"],
                values=[1.0],
                errors=[0.1],
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one index"):
            SpectralIndexData(
                index_defs=(),
                values=jnp.array([]),
                errors=jnp.array([]),
            )

    def test_shape_mismatch_values(self):
        dn = STANDARD_INDICES["Dn4000"]
        with pytest.raises(ValueError, match="values shape"):
            SpectralIndexData(
                index_defs=(dn,),
                values=jnp.array([1.0, 2.0]),
                errors=jnp.array([0.1]),
            )

    def test_shape_mismatch_errors(self):
        dn = STANDARD_INDICES["Dn4000"]
        with pytest.raises(ValueError, match="errors shape"):
            SpectralIndexData(
                index_defs=(dn,),
                values=jnp.array([1.0]),
                errors=jnp.array([0.1, 0.2]),
            )

    def test_wave_range(self):
        sid = SpectralIndexData.from_names(
            names=["Dn4000", "Mgb"],
            values=[1.8, 3.0],
            errors=[0.05, 0.3],
        )
        lo, hi = sid.wave_range
        assert lo < 3900.0
        assert hi > 5200.0

    def test_summary(self):
        sid = SpectralIndexData.from_names(
            names=["Dn4000"],
            values=[1.8],
            errors=[0.05],
        )
        s = sid.summary()
        assert "1 indices" in s
        assert "Dn4000" in s

    def test_chi2_perfect_match(self):
        sid = SpectralIndexData.from_names(
            names=["Dn4000", "HdA"],
            values=[1.8, -1.2],
            errors=[0.05, 0.3],
        )
        chi2 = sid.chi2(sid.values)
        assert chi2 == pytest.approx(0.0, abs=1e-30)

    def test_chi2_one_sigma(self):
        sid = SpectralIndexData.from_names(
            names=["Dn4000", "HdA"],
            values=[1.8, -1.2],
            errors=[0.05, 0.3],
        )
        model = sid.values + sid.errors
        assert sid.chi2(model) == pytest.approx(2.0, abs=1e-10)

    def test_chi2_differentiable(self):
        sid = SpectralIndexData.from_names(
            names=["Dn4000"],
            values=[1.8],
            errors=[0.05],
        )
        grad = assert_grad_matches_fd(sid.chi2, jnp.array([1.9]))
        assert jnp.isfinite(grad).all()
        assert grad[0] != 0.0

    def test_log_likelihood_perfect_match(self):
        sid = SpectralIndexData.from_names(
            names=["Dn4000"],
            values=[1.8],
            errors=[0.05],
        )
        ll = sid.log_likelihood(sid.values)
        expected = -jnp.log(0.05) - 0.5 * jnp.log(2.0 * jnp.pi)
        np.testing.assert_allclose(float(ll), float(expected), atol=1e-10)

    def test_log_likelihood_decreases_with_offset(self):
        sid = SpectralIndexData.from_names(
            names=["Dn4000"],
            values=[1.8],
            errors=[0.05],
        )
        ll_exact = sid.log_likelihood(sid.values)
        ll_offset = sid.log_likelihood(sid.values + 3 * sid.errors)
        assert float(ll_exact) > float(ll_offset)

    def test_log_likelihood_differentiable(self):
        sid = SpectralIndexData.from_names(
            names=["Dn4000"],
            values=[1.8],
            errors=[0.05],
        )
        grad = assert_grad_matches_fd(sid.log_likelihood, jnp.array([1.9]))
        assert jnp.isfinite(grad).all()


# ── Standard catalog completeness ─────────────────────────────────


class TestStandardCatalog:
    def test_dn4000_exists(self):
        assert "Dn4000" in STANDARD_INDICES

    def test_d4000_wide_exists(self):
        assert "D4000" in STANDARD_INDICES
        d4000 = STANDARD_INDICES["D4000"]
        assert d4000.continuum[0] == (3750.0, 3950.0)
        assert d4000.continuum[1] == (4050.0, 4250.0)

    def test_dn4000_narrow_removed(self):
        assert "Dn4000_narrow" not in STANDARD_INDICES

    def test_dn4000_and_d4000_differ(self):
        dn = STANDARD_INDICES["Dn4000"]
        d = STANDARD_INDICES["D4000"]
        assert dn.continuum != d.continuum

    def test_lick_indices_exist(self):
        for name in ["HdA", "HdF", "HgA", "Mgb", "Fe5270", "Fe5335", "Hbeta"]:
            assert name in STANDARD_INDICES, f"Missing {name}"

    def test_all_valid(self):
        for name, idx in STANDARD_INDICES.items():
            assert idx.name == name
            assert idx.index_type in ("EW", "break", "slope")


# ── Index measurement ─────────────────────────────────────────────


class TestMeasureIndex:
    def test_break_flat_spectrum(self):
        """Flat spectrum should give break ratio = 1.0."""
        wave = jnp.linspace(3700.0, 4200.0, 1000)
        flux = jnp.ones_like(wave)
        dn4000 = STANDARD_INDICES["Dn4000"]
        val = measure_index_jax(wave, flux, dn4000)
        np.testing.assert_allclose(float(val), 1.0, atol=0.01)

    def test_break_red_spectrum(self):
        """Linearly increasing flux should give break > 1."""
        wave = jnp.linspace(3700.0, 4200.0, 1000)
        flux = wave / 3700.0
        dn4000 = STANDARD_INDICES["Dn4000"]
        val = measure_index_jax(wave, flux, dn4000)
        assert float(val) > 1.0

    def test_ew_flat_spectrum(self):
        """Flat spectrum has no absorption → EW = 0."""
        wave = jnp.linspace(3900.0, 4300.0, 1000)
        flux = jnp.ones_like(wave)
        hda = STANDARD_INDICES["HdA"]
        val = measure_index_jax(wave, flux, hda)
        np.testing.assert_allclose(float(val), 0.0, atol=0.1)

    def test_ew_absorption_positive(self):
        """Absorption line (dip in feature window) → positive EW."""
        wave = jnp.linspace(3900.0, 4300.0, 2000)
        hda = STANDARD_INDICES["HdA"]
        feat_lo, feat_hi = hda.feature
        feat_center = (feat_lo + feat_hi) / 2.0
        flux = jnp.ones_like(wave) - 0.5 * jnp.exp(-0.5 * ((wave - feat_center) / 5.0) ** 2)
        val = measure_index_jax(wave, flux, hda)
        assert float(val) > 0.0

    def test_differentiable(self):
        """Index measurement is differentiable w.r.t. flux."""
        wave = jnp.linspace(3700.0, 4200.0, 500)
        dn4000 = STANDARD_INDICES["Dn4000"]

        def measure_fn(flux):
            return measure_index_jax(wave, flux, dn4000)

        flux = jnp.ones(500)
        grad = assert_grad_matches_fd(measure_fn, flux)
        assert jnp.isfinite(grad).all()

    def test_jit_compatible(self):
        """Index measurement works under JAX JIT."""
        wave = jnp.linspace(3700.0, 4200.0, 500)
        flux = jnp.ones(500)
        dn4000 = STANDARD_INDICES["Dn4000"]

        @jax.jit
        def fn(f):
            return measure_index_jax(wave, f, dn4000)

        val = fn(flux)
        np.testing.assert_allclose(float(val), 1.0, atol=0.01)

    def test_soft_sigmoid_smooth_gradient(self):
        """Gradient of index measurement is smooth (no NaN near window edges)."""
        wave = jnp.linspace(3700.0, 4200.0, 1000)
        dn4000 = STANDARD_INDICES["Dn4000"]

        def measure_fn(flux):
            return measure_index_jax(wave, flux, dn4000)

        flux = 1.0 + 0.1 * jnp.sin(wave / 50.0)
        grad = assert_grad_matches_fd(measure_fn, flux)
        assert jnp.isfinite(grad).all()
        assert not jnp.all(grad == 0.0)

    def test_ew_second_derivative(self):
        """EW measurement supports second-order differentiation (Hessian)."""
        wave = jnp.linspace(3900.0, 4300.0, 500)
        hda = STANDARD_INDICES["HdA"]

        def measure_fn(flux):
            return measure_index_jax(wave, flux, hda)

        flux = jnp.ones(500)
        hess_diag = assert_grad_matches_fd(lambda f: jnp.sum(jax.grad(measure_fn)(f)), flux)
        assert jnp.isfinite(hess_diag).all()


# ── Observation integration ───────────────────────────────────────


class TestTopLevelExports:
    def test_spectral_index_data_importable(self):
        from tengri.observation import SpectralIndexData as SID

        assert SID is SpectralIndexData

    def test_spectral_index_def_importable(self):
        from tengri.observation import SpectralIndexDef as SIDef

        assert SIDef is SpectralIndexDef


class TestObservationSpectralIndices:
    def test_photometry_plus_indices(self):
        from tengri.observation.observation import Observation

        filt = _make_filter()
        phot = Photometry(filters=(filt,), names=("test_r",))
        sid = SpectralIndexData.from_names(
            names=["Dn4000", "HdA"],
            values=[1.8, -1.2],
            errors=[0.05, 0.3],
        )
        obs = Observation(photometry=phot, spectral_indices=sid)
        assert obs.has_spectral_indices
        assert obs.n_data_indices == 2
        assert obs.n_data == 1 + 2

    def test_indices_only(self):
        from tengri.observation.observation import Observation

        sid = SpectralIndexData.from_names(
            names=["Dn4000"],
            values=[1.8],
            errors=[0.05],
        )
        obs = Observation(spectral_indices=sid)
        assert obs.has_spectral_indices
        assert obs.n_data_indices == 1

    def test_summary_includes_indices(self):
        from tengri.observation.observation import Observation

        filt = _make_filter()
        phot = Photometry(filters=(filt,), names=("test_r",))
        sid = SpectralIndexData.from_names(
            names=["Dn4000"],
            values=[1.8],
            errors=[0.05],
        )
        obs = Observation(photometry=phot, spectral_indices=sid)
        s = obs.summary()
        assert "Indices" in s or "indices" in s


# ── Fitter data_args packing ──────────────────────────────────────


class TestFitterIndexPacking:
    def test_index_data_in_data_args(self):
        from unittest.mock import MagicMock

        from tengri.observation.observation import Observation

        filt = _make_filter()
        phot = Photometry(filters=(filt,), names=("test_r",))
        sid = SpectralIndexData.from_names(
            names=["Dn4000", "HdA"],
            values=[1.8, -1.2],
            errors=[0.05, 0.3],
        )
        obs = Observation(photometry=phot, spectral_indices=sid)

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

        assert "index_obs" in fitter._data_args
        assert "index_err" in fitter._data_args
        np.testing.assert_allclose(fitter._data_args["index_obs"], [1.8, -1.2])
        np.testing.assert_allclose(fitter._data_args["index_err"], [0.05, 0.3])
