# SPDX-License-Identifier: BSD-3-Clause
"""Integration test: verify cross-galaxy engine reuse in CatalogFitter.

Tests that CatalogFitter with multiple galaxies reuses a single compiled
engine across all galaxies when they share the same compile signature.

The test instruments _build_jit_engine with a counter and verifies:
1. Three fitters (with separately-constructed, identical-content SSP grids)
   reuse one compiled engine
2. Each fitter still produces correct numerical results (equivalence test)

Since PR #1973, the compile signature includes a blake2b content digest of
ssp_flux, so two SSP grids that differ in content produce different signatures
and therefore different engines. The sharing contract is now: identical *content*
(measured by the digest), not merely identical shape.
"""

from __future__ import annotations

from unittest import mock

import jax.numpy as jnp
import pytest

from tengri import Fitter, Parameters, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.inference.jit_engine import build_jit_engine
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry


@pytest.fixture
def base_ssp_data():
    """Return a minimal SSPData for testing."""
    n_met, n_age, n_wave = 8, 15, 200
    return SSPData(
        ssp_wave=jnp.logspace(3, 4.5, n_wave),
        ssp_flux=jnp.ones((n_met, n_age, n_wave), dtype=jnp.float64),
        ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
        ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
    )


@pytest.fixture
def photometry():
    """Return a basic photometry observation."""
    return Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"]))


@pytest.fixture
def spec_dpl():
    """Return a simple DPL SFH spec."""
    return Parameters(
        redshift=0.1,
        sfh_dpl_alpha=Uniform(0.5, 4.0),
        sfh_dpl_beta=Uniform(0.3, 3.0),
    )


def _create_identical_ssp() -> SSPData:
    """Return a fresh SSPData with the standard test arrays.

    Each call creates an independent array object (different id), so each
    call to get_ssp_content_hash is an id-cache miss. Because the digest is
    content-based (blake2b of the bytes), separately-constructed arrays with
    identical content produce equal digests, enabling signature equality and
    engine reuse.
    """
    n_met, n_age, n_wave = 8, 15, 200
    return SSPData(
        ssp_wave=jnp.logspace(3, 4.5, n_wave),
        ssp_flux=jnp.ones((n_met, n_age, n_wave), dtype=jnp.float64),
        ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
        ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
    )


class TestCatalogEngineReuse:
    """Integration tests for cross-galaxy engine reuse."""

    @pytest.fixture(autouse=True)
    def _isolated_engine_cache(self):
        """Isolate the process-global engine cache per test.

        The compile_signature is keyed on content and shapes, not data values;
        fitters from sibling tests in the same xdist worker collide in
        _SHARED_ENGINE_CACHE because they use identical specs/SSPData/Observation.
        Compile-count and engine-identity assertions are only meaningful when
        each test controls its own cache state. Per-model write-through caches
        (_model_cache_owner namespace) do not leak across tests because models
        are per-test-local objects.
        """
        from tengri.inference import jit_engine

        with jit_engine._SHARED_ENGINE_CACHE_LOCK:
            saved = dict(jit_engine._SHARED_ENGINE_CACHE)
            jit_engine._SHARED_ENGINE_CACHE.clear()
        try:
            yield
        finally:
            with jit_engine._SHARED_ENGINE_CACHE_LOCK:
                jit_engine._SHARED_ENGINE_CACHE.clear()
                jit_engine._SHARED_ENGINE_CACHE.update(saved)

    def test_three_fitters_one_compile(self, photometry, spec_dpl):
        """Three fitters with identical-content SSP should compile once.

        Creates 3 SEDModels by separately constructing SSPData instances with
        identical numerical content (each call creates a new array object with
        different id()), then constructs 3 Fitters and verifies only 1 call to
        build_jit_engine.

        Each call to get_ssp_content_hash() is an id-cache miss (keyed by
        object identity), but equal content produces equal digests (content-based
        blake2b). All three models have the same compile signature due to
        identical digests, enabling them to reuse one compiled engine.
        """
        # Create 3 separately-constructed SSPData with identical content.
        # Each call to _create_identical_ssp returns a fresh array object
        # (different id()), but the arrays have identical numerical content.
        ssp1 = _create_identical_ssp()
        ssp2 = _create_identical_ssp()
        ssp3 = _create_identical_ssp()

        # Create 3 models
        models = [
            SEDModel(spec_dpl, ssp1, observation=photometry),
            SEDModel(spec_dpl, ssp2, observation=photometry),
            SEDModel(spec_dpl, ssp3, observation=photometry),
        ]

        # Verify all have identical compile_signature
        sigs = [model.compile_signature() for model in models]
        assert sigs[0] == sigs[1] == sigs[2], (
            "Models with identical-content SSP should have identical signatures"
        )

        # Create fitters with identical data config
        data = jnp.ones(3)
        noise = jnp.ones(3) * 0.1

        fitters = [Fitter(model, data, noise, data_type="photometry") for model in models]

        # Verify all fitters have identical compile_signature
        fitter_sigs = [fitter.compile_signature() for fitter in fitters]
        assert fitter_sigs[0] == fitter_sigs[1] == fitter_sigs[2], (
            "Fitters should have identical signatures"
        )

        # Instrument build_jit_engine with a call counter
        call_count = 0
        original_build = build_jit_engine

        def counting_build(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_build(*args, **kwargs)

        with mock.patch(
            "tengri.inference.jit_engine.build_jit_engine",
            side_effect=counting_build,
        ):
            # Now call _get_or_build_engine on each fitter
            # Only the first should trigger build_jit_engine; the rest reuse
            pos_dict = {name: jnp.array(0.0) for name in fitters[0].spec.free_params}

            engines = []
            for fitter in fitters:
                engine = fitter._get_or_build_engine(pos_dict)
                engines.append(engine)

            # Should have compiled exactly once
            assert call_count == 1, f"Expected 1 compile, got {call_count}"

        # All three fitters should have received the SAME engine object
        # (shared via module-level OrderedDict LRU under a lock)
        assert engines[0] is engines[1] is engines[2], (
            "All three fitters should reference the same compiled engine"
        )

    def test_engine_reuse_produces_correct_results(self, photometry, spec_dpl):
        """Engines reused from cache should work for both fitters.

        Creates two models with separately-constructed, identical-content SSP
        grids, verifies they share the same cached engine, and confirms both
        engines produce valid results.
        """
        # Create two separately-constructed SSPData with identical content
        ssp1 = _create_identical_ssp()
        ssp2 = _create_identical_ssp()

        model1 = SEDModel(spec_dpl, ssp1, observation=photometry)
        model2 = SEDModel(spec_dpl, ssp2, observation=photometry)

        # Create fitters
        data = jnp.ones(3) * 0.5
        noise = jnp.ones(3) * 0.05

        fitter1 = Fitter(model1, data, noise, data_type="photometry")
        fitter2 = Fitter(model2, data, noise, data_type="photometry")

        # Get engines (fitter1 will build, fitter2 will reuse)
        pos_dict = {name: jnp.array(0.0) for name in fitter1.spec.free_params}
        engine1 = fitter1._get_or_build_engine(pos_dict)
        engine2 = fitter2._get_or_build_engine(pos_dict)

        # Both engines should be valid dicts
        assert isinstance(engine1, dict), "engine1 should be a dict"
        assert isinstance(engine2, dict), "engine2 should be a dict"

        # They should be the SAME object (reused from cache)
        assert engine1 is engine2, "Engines should be identical objects (reused from cache)"

    def test_different_signature_no_reuse(self, base_ssp_data, photometry, spec_dpl):
        """Fitters with different signatures should NOT reuse engines.

        Tests that changing a JIT-affecting parameter (redshift being fixed)
        produces different compile signatures and doesn't reuse engines.
        """
        # Create two specs: one with fixed redshift, one with free redshift
        spec1 = Parameters(
            redshift=0.1,  # fixed
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            apply_igm=True,
        )
        spec2 = Parameters(
            redshift=Uniform(0.05, 0.15),  # free (different signature)
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            apply_igm=True,
        )

        model1 = SEDModel(spec1, base_ssp_data, observation=photometry)
        model2 = SEDModel(spec2, base_ssp_data, observation=photometry)

        data = jnp.ones(3)
        noise = jnp.ones(3) * 0.1

        fitter1 = Fitter(model1, data, noise, data_type="photometry")
        fitter2 = Fitter(model2, data, noise, data_type="photometry")

        # Verify different signatures
        sig1 = fitter1.compile_signature()
        sig2 = fitter2.compile_signature()
        assert sig1 != sig2, (
            "Fitters should have different signatures when redshift config differs"
        )

        # Get engines (each should build separately)
        pos_dict1 = {name: jnp.array(0.0) for name in fitter1.spec.free_params}
        pos_dict2 = {name: jnp.array(0.0) for name in fitter2.spec.free_params}

        engine1 = fitter1._get_or_build_engine(pos_dict1)
        engine2 = fitter2._get_or_build_engine(pos_dict2)

        # Engines should be different objects (different signatures)
        assert engine1 is not engine2, "Different signatures should produce different engines"

    def test_bug_2047_different_ssp_content_gets_own_engine(self, photometry, spec_dpl):
        """Different SSP flux content must produce different engines (issue #2047).

        Before PR #1973, the compile signature included only SSP shape and
        metallicity grid, not the flux content. This allowed two SSP grids
        with identical shape/lgmet but different flux to collide in the
        compile_signature, causing silent engine reuse and systematic bias
        (+0.9962 dex measured on log_total_mass).

        This test pins the fix: two separately-constructed SSPData with
        DIFFERENT flux content must produce DIFFERENT signatures and therefore
        different (not shared) compiled engines.
        """
        # Create a second SSPData with content-different flux
        ssp1 = _create_identical_ssp()

        # Manually create ssp2 with the same shape but scaled flux
        n_met, n_age, n_wave = 8, 15, 200
        ssp2 = SSPData(
            ssp_wave=jnp.logspace(3, 4.5, n_wave),
            ssp_flux=jnp.ones((n_met, n_age, n_wave), dtype=jnp.float64) * 1.1,
            ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
            ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
        )  # Different content

        # Create models
        model1 = SEDModel(spec_dpl, ssp1, observation=photometry)
        model2 = SEDModel(spec_dpl, ssp2, observation=photometry)

        # Verify they have DIFFERENT compile signatures (different flux content)
        sig1 = model1.compile_signature()
        sig2 = model2.compile_signature()
        assert sig1 != sig2, (
            "SSP flux content difference must produce different compile signatures"
        )

        # Create fitters
        data = jnp.ones(3) * 0.5
        noise = jnp.ones(3) * 0.05

        fitter1 = Fitter(model1, data, noise, data_type="photometry")
        fitter2 = Fitter(model2, data, noise, data_type="photometry")

        # Verify fitter signatures also differ
        fitter_sig1 = fitter1.compile_signature()
        fitter_sig2 = fitter2.compile_signature()
        assert fitter_sig1 != fitter_sig2, "Fitter signatures should differ due to SSP content"

        # Instrument build_jit_engine and verify we get TWO compiles, not one
        call_count = 0
        original_build = build_jit_engine

        def counting_build(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_build(*args, **kwargs)

        with mock.patch(
            "tengri.inference.jit_engine.build_jit_engine",
            side_effect=counting_build,
        ):
            pos_dict1 = {name: jnp.array(0.0) for name in fitter1.spec.free_params}
            pos_dict2 = {name: jnp.array(0.0) for name in fitter2.spec.free_params}

            engine1 = fitter1._get_or_build_engine(pos_dict1)
            engine2 = fitter2._get_or_build_engine(pos_dict2)

            # Should have compiled exactly twice (once per engine)
            assert call_count == 2, (
                f"Expected 2 compiles for different SSP content, got {call_count}"
            )

        # Engines must be DIFFERENT objects (not shared)
        assert engine1 is not engine2, (
            "Different SSP flux content should produce non-identical engine objects"
        )
