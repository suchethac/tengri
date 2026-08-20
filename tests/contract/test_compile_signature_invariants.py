# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Fitter.compile_signature() invariants.

Ensures that:
1. memory_mode changes do not affect compile_signature (no spurious recompile)
2. The field count is pinned to catch accidental additions
3. _engine_cache_key and compile_signature agree on the JIT-invariant fields
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri import Fitter, Parameters, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry

pytestmark = pytest.mark.contract


@pytest.fixture
def mock_ssp_data():
    """Return a minimal SSPData for testing."""
    n_met, n_age, n_wave = 8, 15, 200
    ssp = SSPData(
        ssp_wave=jnp.logspace(3, 4.5, n_wave),
        ssp_flux=jnp.ones((n_met, n_age, n_wave), dtype=jnp.float64),
        ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
        ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
    )
    return ssp


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


class TestCompileSignatureInvariants:
    """Tests for memory_mode exclusion from compile_signature."""

    def test_memory_mode_does_not_affect_signature(self, mock_ssp_data, photometry, spec_dpl):
        """Two Fitters identical except memory_mode must have identical compile_signature.

        memory_mode only affects posterior-chunking behavior in the analysis layer,
        not the compiled HLO graph. Toggling it should NOT cause recompilation.
        """
        model = SEDModel(spec_dpl, mock_ssp_data, observation=photometry)
        data = jnp.ones(3)
        noise = jnp.ones(3) * 0.1

        fitter_fast = Fitter(model, data, noise, data_type="photometry")
        fitter_low = Fitter(model, data, noise, data_type="photometry")

        # Set different memory modes (simulating what run() does)
        fitter_fast._memory_mode = "fast"
        fitter_low._memory_mode = "low"

        sig_fast = fitter_fast.compile_signature()
        sig_low = fitter_low.compile_signature()

        assert sig_fast == sig_low, "compile_signature must be identical regardless of memory_mode"

    def test_compile_signature_field_count_pinned(self, mock_ssp_data, photometry, spec_dpl):
        """Pin the field count of compile_signature to catch accidental additions.

        If this test fails, it means compile_signature() was changed.
        Verify the change is intentional (affects HLO), then update this assertion.
        """
        model = SEDModel(spec_dpl, mock_ssp_data, observation=photometry)
        data = jnp.ones(3)
        noise = jnp.ones(3) * 0.1

        fitter = Fitter(model, data, noise, data_type="photometry")
        sig = fitter.compile_signature()

        # sig is a tuple of (model_sig, fitter_sig)
        model_sig, fitter_sig = sig

        # model_sig is SEDModel.compile_signature()'s tuple; its per-field
        # ledger lives in that method's own comments. Pinned here so a field
        # cannot vanish silently — removing one is how #1973's collision
        # shipped. Was 64 before #1973 added ssp_flux_id (grid CONTENT, not
        # just shape/lgmet).
        assert len(model_sig) == 65, (
            f"model_sig field count changed from 65 to {len(model_sig)}. "
            "If intentional, update this assertion and the ledger in "
            "SEDModel.compile_signature."
        )

        # fitter_sig should have exactly 10 fields:
        # 1. data_type
        # 2. stochastic
        # 3. n_grid
        # 4. len(data)
        # 5. sorted free names
        # 6. has_noise_model
        # 7. _eline_marginalize
        # 8. _eline_fitted
        # 9. _calibration_marginalize
        # 10. _eline_prior_type
        # (memory_mode was removed; it was the 11th)
        # 11. line_flux_key (wavelengths + limit-mask presence)
        # 12. line_ratios present
        # 13. spectral_indices present
        # 14. data_mask present
        # (11-14 added 2026-07: observation feature channels are baked into
        # the loss closure, so they must key the engine/loss cache — else a
        # joint phot+lines Fitter reuses a photometry-only compiled loss.)
        # 15. params_override key (#1329): the per-fit fixed-value override is
        # baked into the loss closure via fitter._fixed_values, so two fits
        # differing only by override must compile distinct losses — else fit #2
        # silently reuses fit #1's baked redshift. None when no override.
        # 16. free-parameter prior identity: _primals_to_params calls
        # dist.unstandardize(xi), which reads the distribution's Python floats
        # at trace time, so the priors are baked constants. Without this entry
        # two models differing only in a prior's bounds share one engine and
        # fit #2's latent is decoded through fit #1's interval — a shift of
        # order the prior width (measured 1.53 dex on log_total_mass). Free
        # NAMES (field 5) do not cover it: changing Uniform(9.6, 11.1) to
        # Uniform(7, 13) alters no name, shape, dtype or control flow. See
        # tests/regression/bug/test_prior_bounds_key_the_engine_cache.py.
        # 17. spec fixed VALUES (#1972 instance 2): _primals_to_params also
        # bakes fitter._fixed_values, so two models differing only in a fixed
        # scalar shared one engine — measured -0.18 dex on mass via dust_slope.
        # 18. mirror map (#1972 instance 3): spec.resolve_mirrors bakes
        # target -> source, so two specs sharing every name and prior but tying
        # to different sources silently tied to the same one.
        assert len(fitter_sig) == 18, (
            f"fitter_sig field count changed from 18 to {len(fitter_sig)}. "
            "If intentional, update this assertion and the docstring."
        )

    def test_engine_cache_key_matches_compile_signature_fields(
        self, mock_ssp_data, photometry, spec_dpl
    ):
        """Verify _engine_cache_key() and compile_signature() agree on JIT-invariant fields.

        Both methods should use the same fields (in the same order) to ensure that
        smart-lean's cache key logic remains correct. _engine_cache_key is the
        source of truth for which fields affect the compiled HLO; compile_signature
        wraps it with the model signature.
        """
        model = SEDModel(spec_dpl, mock_ssp_data, observation=photometry)
        data = jnp.ones(3)
        noise = jnp.ones(3) * 0.1

        fitter = Fitter(model, data, noise, data_type="photometry")

        engine_key = fitter._engine_cache_key()
        _, fitter_sig = fitter.compile_signature()

        # _engine_cache_key returns the fitter_sig component (no model_sig prefix)
        assert engine_key == fitter_sig, (
            "engine_key and fitter_sig must be identical; smart-lean relies on this"
        )

    def test_different_memory_modes_reuse_same_engine(self, mock_ssp_data, photometry, spec_dpl):
        """Verify that different memory_mode settings would use the same cached engine.

        This is a white-box test: it checks that the cache key derivation
        (compile_signature) is consistent even when memory_mode is toggled.
        The actual engine reuse happens in _get_or_build_engine.
        """
        model = SEDModel(spec_dpl, mock_ssp_data, observation=photometry)
        data = jnp.ones(3)
        noise = jnp.ones(3) * 0.1

        fitter = Fitter(model, data, noise, data_type="photometry")

        # Cache the signature before setting memory_mode
        sig_before = fitter.compile_signature()

        # Simulate what run() does: toggle memory_mode
        fitter._memory_mode = "low"

        # Signature must remain unchanged
        sig_after = fitter.compile_signature()

        assert sig_before == sig_after, "Toggling memory_mode must not invalidate the engine cache"
