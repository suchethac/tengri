# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for guards in ``forward/sed_model.py`` that could be bypassed.

Hardening pass, items B1, B2 and C1:

* **B1** (AGN flatness refusal): the relative-difference test divided by
  ``max|sed_lo| + 1e-300``. Under float32 that addend is 0, so two all-zero
  SEDs produced ``0 / 0 = NaN``, ``NaN < 1e-10`` was False, and the refusal
  of a flat ``agn_log_lbol`` never fired.
* **B2** (law-shape lookup): ``except Exception`` around
  ``_law_shape_params`` swallowed every error, not only the unregistered-law
  case that the function already answers with an empty set.
* **C1** (ignored kwargs): ``SEDModel.__init__`` deleted ``precompute`` and
  ``strategy`` without a word, so callers passing them believed they had an
  effect. Passing either now emits a ``DeprecationWarning``.

Notes
-----
Physics cross-check: flatness is a measurement, not a derivation. A zero SED
is trivially identical to itself, so two zero readings must count as flat, and
a genuinely different reading must not.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from tengri.config.exceptions import ConfigError
from tengri.forward.sed_model import SEDModel
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform

pytestmark = pytest.mark.regression_bug


def _probe_model(sed_lo_fn, sed_hi_fn, *, lbol_fixed=False):
    """Stand-in carrying the attributes the flatness probe reads.

    ``predict_state`` returns the SED the test dictates at each bound, so the
    probe's own comparison and refusal logic is what runs.
    """

    def predict_state(params, fixed_values=None):
        lbol = params["agn_log_lbol"]
        sed = sed_lo_fn() if lbol < 45.0 else sed_hi_fn()
        return SimpleNamespace(wave=jnp.linspace(1000.0, 2000.0, sed.shape[0]), sed_intrinsic=sed)

    return SimpleNamespace(
        spec=SimpleNamespace(get_fixed_values=lambda: {}),
        _agn_lbol_dist=SimpleNamespace(is_fixed=lbol_fixed, bounds=(44.0, 46.0)),
        _agn_ir_frac_dist=SimpleNamespace(is_fixed=False, bounds=(0.1, 0.9)),
        _agn_lbol_is_user_fixed=lbol_fixed,
        predict_state=predict_state,
    )


def _spec_stub():
    return SimpleNamespace(sample=lambda key: {})


class TestAgnFlatnessRefusalFloat32:
    """B1: the flatness refusal must hold in float32, where 1e-300 underflows."""

    def test_two_zero_seds_are_flat_in_float32(self):
        with jax.enable_x64(False):
            probe = _probe_model(
                lambda: jnp.zeros(16, dtype=jnp.float32),
                lambda: jnp.zeros(16, dtype=jnp.float32),
            )
            with pytest.raises(ConfigError, match="identical to within"):
                SEDModel._check_agn_lbol_flat_direction(probe, _spec_stub())

    def test_genuinely_different_seds_are_not_refused(self):
        with jax.enable_x64(False):
            probe = _probe_model(
                lambda: jnp.ones(16, dtype=jnp.float32),
                lambda: 2.0 * jnp.ones(16, dtype=jnp.float32),
            )
            assert SEDModel._check_agn_lbol_flat_direction(probe, _spec_stub()) is None


class TestLawShapeLookupPropagatesErrors:
    """B2: only an unregistered law may be answered with an empty set."""

    @staticmethod
    def _model():
        model = object.__new__(SEDModel)
        model.spec = SimpleNamespace(_group_provenance={})
        return model

    def test_unregistered_law_reads_nothing(self):
        assert self._model()._requested_law_shape_params("no_such_law") == frozenset()

    def test_unexpected_error_propagates(self, monkeypatch):
        import tengri.parameters.groups as groups

        def boom(law_name):
            raise RuntimeError("signature introspection failed")

        monkeypatch.setattr(groups, "_law_shape_params", boom)
        with pytest.raises(RuntimeError, match="signature introspection failed"):
            self._model()._requested_law_shape_params("calzetti")


@pytest.fixture(scope="module")
def minimal_spec():
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 4.0),
        sfh_dpl_beta=Uniform(0.5, 4.0),
        met_logzsol=Fixed(1.0),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="module")
def synthetic_ssp():
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    n_met, n_age, n_wave = 3, 20, 100
    wave = jnp.linspace(3000.0, 10000.0, n_wave)
    ages_gyr = jnp.linspace(-1.0, 1.14, n_age)
    flux = jnp.full((n_met, n_age, n_wave), 1e-3)
    lgmet = jnp.array([-1.5, -0.5, 0.0])
    return SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)


class TestDeprecatedConstructorKwargs:
    """C1: ``precompute`` and ``strategy`` warn when passed and do nothing."""

    def test_passing_precompute_warns(self, minimal_spec, synthetic_ssp):
        with pytest.warns(DeprecationWarning, match="precompute"):
            SEDModel(minimal_spec, synthetic_ssp, precompute=False)

    def test_passing_strategy_warns(self, minimal_spec, synthetic_ssp):
        with pytest.warns(DeprecationWarning, match="strategy"):
            SEDModel(minimal_spec, synthetic_ssp, strategy="fast")

    def test_omitting_both_does_not_warn(self, minimal_spec, synthetic_ssp):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            SEDModel(minimal_spec, synthetic_ssp)
        flagged = [
            w
            for w in record
            if issubclass(w.category, DeprecationWarning)
            and ("precompute" in str(w.message) or "strategy" in str(w.message))
        ]
        assert flagged == []
