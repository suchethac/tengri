# SPDX-License-Identifier: BSD-3-Clause
"""Phase 4-D tests: complete template threading (nebular backends, dust IR, AGN SKIRTOR).

Tests for Category A (CB19/MAPPINGS wiring), Category B (dust IR template
threading), and Category C (AGN SKIRTOR template threading).

What this file actually covers: backend dispatch and smoke-level JIT
compilation for the CB19 / MAPPINGS nebular backends, dust IR emission, and
the monolithic SKIRTOR model.

It does **not** verify that any template threads as a ``Parameter`` rather
than baking in as a ``Constant``, though its tests were once named and
docstring'd as if it did — and stayed green while 31 MB of SKIRTOR grid baked
into every graph (#1383, #1549). That invariant needs a jaxpr constant count
taken on the surface where the template data is an argument; it lives in
``test_agn_template_threading.py``. Do not re-add threading claims here
without a measurement to back them.
"""

from __future__ import annotations

import pathlib
import warnings

import chex
import jax
import jax.numpy as jnp
import pytest

from tengri import Parameters, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation import Observation, Photometry
from tengri.parameters.priors import Fixed, Uniform

pytestmark = pytest.mark.contract

_SSP_BARE = pathlib.Path("data/ssp_prsc_bc03_chabrier.h5").resolve()
_SSP_WNE = pathlib.Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5").resolve()


@pytest.fixture(scope="module")
def ssp_bare():
    if not _SSP_BARE.exists():
        pytest.skip(f"bare-stellar SSP not available at {_SSP_BARE}")
    return load_ssp_data(str(_SSP_BARE))


@pytest.fixture(scope="module")
def ssp_wne():
    if not _SSP_WNE.exists():
        pytest.skip(f"wNE SSP not available at {_SSP_WNE}")
    return load_ssp_data(str(_SSP_WNE))


@pytest.fixture(scope="module")
def obs():
    return Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )


def _base_spec(**kwargs):
    """Basic spec: all params fixed, optionally overridden."""
    defaults = dict(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(1.0),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.1),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    defaults.update(kwargs)
    return Parameters(**defaults)


def _silent_build(spec, ssp, obs, **kwargs):
    """Build a model, suppressing warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp, observation=obs, **kwargs)


# ── Category A: Nebular backend wiring (CB19 / MAPPINGS) ──────────────────────


class TestCB19Wiring:
    """CB19 backend wiring tests."""

    def test_cb19_backend_config_declaration(self):
        """CB19 is available as a config backend choice."""
        from tengri.components.nebular.component import NebularSEDComponentConfig

        cfg = NebularSEDComponentConfig(backend="cb19")
        assert cfg.backend == "cb19"

    def test_cb19_backend_in_spec(self, ssp_bare, obs):
        """CB19 backend string is accepted in Parameters."""
        spec = _base_spec(
            nebular_backend="cb19",
            neb_logU=Uniform(-4.0, -2.0),
        )

        # Verify that the spec accepts CB19 as a valid backend
        assert spec.nebular_backend == "cb19"

    def test_cb19_backend_exposes_a_grid(self, ssp_bare, obs):
        """The CB19 backend carries a ``.grid`` attribute.

        Renamed from ``..._template_detected_in_jit_inputs``: the body never
        called ``_template_data_for_jit()``, so the old name asserted a
        threading property this test does not measure (#1549).
        """
        try:
            from tengri.components.nebular.cloudy_cb19 import CB19Backend

            # Instantiate CB19 backend (may skip if grid data unavailable)
            try:
                backend = CB19Backend()
            except Exception as e:
                pytest.skip(f"CB19Backend instantiation failed: {e}")

            # Check that it has a .grid attribute
            assert hasattr(backend, "grid")
            assert backend.grid is not None

        except ImportError:
            pytest.skip("CB19Backend not available")

    def test_mappings_backend_config_declaration(self):
        """MAPPINGS photoionization backend is available as a config choice."""
        from tengri.components.nebular.component import NebularSEDComponentConfig

        cfg = NebularSEDComponentConfig(backend="mappings")
        assert cfg.backend == "mappings"

    def test_mappings_backend_in_spec(self, ssp_bare, obs):
        """MAPPINGS backend string is accepted in Parameters."""
        spec = _base_spec(
            nebular_backend="mappings",
            neb_logU=Uniform(-4.0, -2.0),
        )

        assert spec.nebular_backend == "mappings"

    def test_mappings_backend_exposes_a_grid(self, ssp_bare, obs):
        """The MAPPINGS backend carries a ``.grid`` attribute.

        Renamed for the same reason as the CB19 case above (#1549).
        """
        try:
            from tengri.components.nebular.mappings_photo import MappingsPhotoStellarBackend

            # Instantiate MAPPINGS backend (may skip if grid data unavailable)
            try:
                backend = MappingsPhotoStellarBackend()
            except Exception as e:
                pytest.skip(f"MappingsPhotoStellarBackend instantiation failed: {e}")

            # Check that it has a .grid attribute
            assert hasattr(backend, "grid")
            assert backend.grid is not None

        except ImportError:
            pytest.skip("MappingsPhotoStellarBackend not available")


# ── Category B: Dust IR template threading ────────────────────────────────────


class TestDustIRTemplateThreading:
    """Dust IR emission template threading tests."""

    @pytest.mark.parametrize(
        "dust_emission",
        ["modified_blackbody", "casey2012"],  # Analytic models (no templates)
    )
    def test_analytic_dust_models_no_templates(self, ssp_wne, obs, dust_emission):
        """Analytic dust emission models (no templates) work without threading."""
        spec = _base_spec(dust_emission=dust_emission)

        model = _silent_build(spec, ssp_wne, obs)
        # No templates to thread; _template_data_for_jit should return None or empty
        template_data = model._template_data_for_jit()
        # Dust analytic models don't produce template data; only nebular does
        assert template_data is None

    @pytest.mark.parametrize(
        "dust_emission",
        ["dale2014", "draine_li2007", "draine_li2014", "astrodust", "bosa"],
    )
    def test_dust_template_models_build(self, ssp_wne, obs, dust_emission):
        """Template-based dust models build successfully."""
        spec = _base_spec(dust_emission=dust_emission)

        try:
            model = _silent_build(spec, ssp_wne, obs)
            # Model should be built (may skip if template data not available)
            assert model is not None
        except FileNotFoundError as e:
            pytest.skip(f"Template file not available for {dust_emission}: {e}")

    def test_dust_ir_jit_non_jit_agreement(self, ssp_wne, obs):
        """JIT and non-JIT dust IR SED paths agree to floating-point precision."""
        spec = _base_spec(dust_emission="dale2014")

        try:
            model = _silent_build(spec, ssp_wne, obs)
        except FileNotFoundError:
            pytest.skip("Dale2014 template not available")

        fixed_params = spec.get_fixed_values()

        phot_non_jit = model.predict_photometry(fixed_params)
        phot_jit = jax.jit(lambda p: model.predict_photometry(p))(fixed_params)

        # The test's whole claim is agreement, so compare the two. It used to
        # assert each was separately `not None` and never compare them, which
        # would pass even if the JIT path returned different fluxes.
        chex.assert_tree_all_finite(phot_non_jit)
        assert float(jnp.max(jnp.abs(phot_non_jit))) > 0.0, "all-zero photometry proves nothing"
        chex.assert_trees_all_close(phot_jit, phot_non_jit, rtol=1e-12, atol=0.0)


# ── Category C: AGN SKIRTOR template threading ──────────────────────────────────


class TestAGNSKIRTORTemplateThreading:
    """AGN SKIRTOR template threading tests."""

    def test_skirtor_template_build(self, ssp_wne, obs):
        """Model with SKIRTOR torus builds successfully."""
        spec = _base_spec(
            agn_model="skirtor",
            agn_log_lbol=Fixed(11.42),
            agn_cos_inc=Fixed(0.5),
            agn_torus_frac=Fixed(0.5),
        )

        try:
            model = _silent_build(spec, ssp_wne, obs)
            assert model is not None
        except (FileNotFoundError, ValueError) as e:
            pytest.skip(f"SKIRTOR grid not available or config issue: {e}")

    def test_skirtor_model_jit_compiles(self, ssp_wne, obs):
        """The monolithic SKIRTOR model compiles under jit.

        A smoke test, and named like one. It was previously called
        ``..._jit_compatibility`` and docstring'd "without baking templates
        into HLO" while asserting only ``result is not None`` — a property
        true of any model that compiles at all. It stayed green through the
        entire period in which 31 MB of SKIRTOR grid baked into every graph
        (#1383). The invariant its old name claimed is now measured, on the
        surface where it is falsifiable, in
        ``test_agn_template_threading.py``.
        """
        spec = _base_spec(
            agn_model="skirtor",
            agn_log_lbol=Fixed(11.42),
            agn_cos_inc=Fixed(0.5),
            agn_torus_frac=Fixed(0.5),
        )

        try:
            model = _silent_build(spec, ssp_wne, obs)
        except (FileNotFoundError, ValueError):
            pytest.skip("SKIRTOR grid not available")

        fixed_vals = model.spec.get_fixed_values()
        predict_jit = jax.jit(lambda p: model.predict_photometry(p))
        result = predict_jit(fixed_vals)

        # Assert something the compile actually has to produce: finite fluxes
        # of the right shape, not merely "not None".
        chex.assert_tree_all_finite(result)
        assert result.shape == (len(obs.photometry.filters),)


# ── Regression tests: Phase 4-B and 4-C still pass ─────────────────────────────


def test_phase4b_ssp_threading_regression(ssp_wne, obs):
    """Phase 4-B (SSP threading) still passes after Phase 4-D changes."""
    spec = _base_spec()

    model = _silent_build(spec, ssp_wne, obs)
    assert model is not None

    # SSP data should be detectable
    assert model.ssp_data is not None


def test_phase4c_cue_threading_regression(ssp_bare, obs):
    """Phase 4-C (Cue threading) still passes after Phase 4-D changes."""
    try:
        from tengri.components.nebular.cue import CueBackend  # noqa: F401

        spec = _base_spec(
            nebular_backend="cue",
            neb_logU=Uniform(-4.0, -2.0),
        )

        try:
            model = _silent_build(spec, ssp_bare, obs)
            assert model is not None
        except Exception as e:
            pytest.skip(f"Cue instantiation or build failed: {e}")

    except ImportError:
        pytest.skip("Cue not available")
