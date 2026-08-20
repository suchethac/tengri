# SPDX-License-Identifier: BSD-3-Clause
"""SSP grids with different flux values must have different compile signatures.

Two SSP grids with identical shapes and identical ssp_lgmet but different
ssp_flux values must produce different compile_signature() values. Without
this, the second model silently runs the first model's stellar physics,
producing +1 dex errors in stellar mass when fit order changes (#1973).

Before the fix::

    model_signatures_equal: True
    truth log_total_mass          = 10.5000
    B fitted alone                = 10.5105
    B fitted after A (10x grid)   = 11.5066  <- A's physics
    delta                         = +0.9962 dex

After the fix, compile_signature includes ssp_flux content, so B gets its
own compiled kernel and the delta is zero (within optimizer tolerance).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fitter, Fixed, Parameters, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData, get_ssp_content_hash
from tengri.inference.jit_engine import clear_shared_caches, get_or_build_signal_response
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry

pytestmark = pytest.mark.regression_bug


@pytest.fixture
def ssp_base():
    """Return a minimal SSPData: small enough to fit fast, real enough to compile."""
    n_met, n_age, n_wave = 8, 15, 200
    rng = np.random.default_rng(0)
    return SSPData(
        ssp_wave=jnp.logspace(3, 4.5, n_wave),
        ssp_flux=jnp.asarray(rng.uniform(0.5, 1.5, (n_met, n_age, n_wave)), dtype=jnp.float64),
        ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
        ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
    )


@pytest.fixture
def obs():
    return Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"]))


# ── Content hash is computed once and cached ────────────────────────────────


def test_ssp_content_hash_is_cached(ssp_base):
    """The digest is cached per flux-array object, weakref-validated.

    SSPData is a NamedTuple — nothing can be cached on the instance (tuples
    reject both ``__setattr__`` and weakrefs) — so the cache lives at module
    level keyed by ``id(ssp_flux)`` and anchored by a weakref to the array.
    First call pays the digest (~30 ms on the default grid); repeats are O(1).
    """
    from tengri.components.stellar.sps import dsps_wrapper

    hash1 = get_ssp_content_hash(ssp_base)
    key = id(ssp_base.ssp_flux)
    entry = dsps_wrapper._SSP_CONTENT_HASH_CACHE.get(key)
    assert entry is not None, "digest must be cached after the first call"
    ref, cached_digest = entry
    assert ref() is ssp_base.ssp_flux, "cache entry must anchor the live array"
    assert cached_digest == hash1

    # A poisoned cache entry coming back proves the repeat call is a cache
    # hit rather than a recompute (identity check passes, no digest runs).
    dsps_wrapper._SSP_CONTENT_HASH_CACHE[key] = (ref, hash1 + 1)
    try:
        assert get_ssp_content_hash(ssp_base) == hash1 + 1, (
            "second call must come from the cache, not a recompute"
        )
    finally:
        dsps_wrapper._SSP_CONTENT_HASH_CACHE[key] = (ref, hash1)
    assert get_ssp_content_hash(ssp_base) == hash1


def test_identical_content_produces_identical_hash(ssp_base):
    """Two SSPData with identical arrays produce identical hashes.

    Tests content-based equality (not identity-based). If the same SSP is
    reloaded from disk, the two instances should produce equal hashes.
    """
    # Create a second SSPData with identical arrays
    ssp_copy = SSPData(
        ssp_wave=ssp_base.ssp_wave,
        ssp_flux=ssp_base.ssp_flux,
        ssp_lg_age_gyr=ssp_base.ssp_lg_age_gyr,
        ssp_lgmet=ssp_base.ssp_lgmet,
        ssp_mass_remaining=ssp_base.ssp_mass_remaining,
        ssp_alpha_fe=ssp_base.ssp_alpha_fe,
        imf=ssp_base.imf,
        source=ssp_base.source,
        nebular=ssp_base.nebular,
    )

    hash_base = get_ssp_content_hash(ssp_base)
    hash_copy = get_ssp_content_hash(ssp_copy)

    assert hash_base == hash_copy, "identical SSP arrays must produce identical content hashes"


def test_different_ssp_flux_produces_different_hash(ssp_base):
    """Two SSPData differing only in ssp_flux produce different hashes.

    This is the core guard: the hash must detect flux differences, not just
    shape differences. A 10x-scaled grid should produce a different hash.
    """
    ssp_scaled = SSPData(
        ssp_wave=ssp_base.ssp_wave,
        ssp_flux=ssp_base.ssp_flux * 10.0,
        ssp_lg_age_gyr=ssp_base.ssp_lg_age_gyr,
        ssp_lgmet=ssp_base.ssp_lgmet,
        ssp_mass_remaining=ssp_base.ssp_mass_remaining,
        ssp_alpha_fe=ssp_base.ssp_alpha_fe,
        imf=ssp_base.imf,
        source=ssp_base.source,
        nebular=ssp_base.nebular,
    )

    hash_base = get_ssp_content_hash(ssp_base)
    hash_scaled = get_ssp_content_hash(ssp_scaled)

    assert hash_base != hash_scaled, "SSP flux scaling should produce a different content hash"


# ── Compile signature reflects SSP content differences ──────────────────────


def test_different_ssp_flux_get_different_compile_signatures(ssp_base, obs):
    """Two models differing only in ssp_flux must have different signatures.

    This is the fix: compile_signature must include ssp_flux content so the
    cache distinguishes between SSP libraries.
    """
    ssp_scaled = SSPData(
        ssp_wave=ssp_base.ssp_wave,
        ssp_flux=ssp_base.ssp_flux * 10.0,
        ssp_lg_age_gyr=ssp_base.ssp_lg_age_gyr,
        ssp_lgmet=ssp_base.ssp_lgmet,
        ssp_mass_remaining=ssp_base.ssp_mass_remaining,
        ssp_alpha_fe=ssp_base.ssp_alpha_fe,
        imf=ssp_base.imf,
        source=ssp_base.source,
        nebular=ssp_base.nebular,
    )

    spec = Parameters(
        redshift=Fixed(0.1),
        sfh_dpl_alpha=Uniform(0.5, 4.0),
        sfh_dpl_beta=Uniform(0.3, 3.0),
    )

    model_base = SEDModel(spec, ssp_base, observation=obs)
    model_scaled = SEDModel(spec, ssp_scaled, observation=obs)

    sig_base = model_base.compile_signature()
    sig_scaled = model_scaled.compile_signature()

    assert sig_base != sig_scaled, (
        "models with different ssp_flux must have different compile signatures"
    )


def test_identical_ssp_content_still_shares_signature(ssp_base, obs):
    """Guard against over-keying: identical content must still share a signature.

    Cross-galaxy reuse is the whole point of the shared cache. Two models with
    equal specs must remain cache-identical, or catalog fits recompile per row.
    """
    # Create a copy with identical content
    ssp_copy = SSPData(
        ssp_wave=ssp_base.ssp_wave,
        ssp_flux=ssp_base.ssp_flux,
        ssp_lg_age_gyr=ssp_base.ssp_lg_age_gyr,
        ssp_lgmet=ssp_base.ssp_lgmet,
        ssp_mass_remaining=ssp_base.ssp_mass_remaining,
        ssp_alpha_fe=ssp_base.ssp_alpha_fe,
        imf=ssp_base.imf,
        source=ssp_base.source,
        nebular=ssp_base.nebular,
    )

    spec = Parameters(
        redshift=Fixed(0.1),
        sfh_dpl_alpha=Uniform(0.5, 4.0),
        sfh_dpl_beta=Uniform(0.3, 3.0),
    )

    model_base = SEDModel(spec, ssp_base, observation=obs)
    model_copy = SEDModel(spec, ssp_copy, observation=obs)

    sig_base = model_base.compile_signature()
    sig_copy = model_copy.compile_signature()

    assert sig_base == sig_copy, (
        "identical SSP content must produce identical signatures (no over-keying)"
    )


# ── Inference closures reflect signature differences ──────────────────────


def test_different_ssp_flux_get_different_signal_response_closures(ssp_base, obs):
    """Two models with different ssp_flux must not share a signal_response.

    The leak site itself: get_or_build_signal_response must not alias closures
    for models whose compile_signature differs due to ssp_flux.
    """
    ssp_scaled = SSPData(
        ssp_wave=ssp_base.ssp_wave,
        ssp_flux=ssp_base.ssp_flux * 10.0,
        ssp_lg_age_gyr=ssp_base.ssp_lg_age_gyr,
        ssp_lgmet=ssp_base.ssp_lgmet,
        ssp_mass_remaining=ssp_base.ssp_mass_remaining,
        ssp_alpha_fe=ssp_base.ssp_alpha_fe,
        imf=ssp_base.imf,
        source=ssp_base.source,
        nebular=ssp_base.nebular,
    )

    spec = Parameters(
        redshift=Fixed(0.1),
        sfh_dpl_alpha=Uniform(0.5, 4.0),
        sfh_dpl_beta=Uniform(0.3, 3.0),
    )

    model_base = SEDModel(spec, ssp_base, observation=obs)
    model_scaled = SEDModel(spec, ssp_scaled, observation=obs)

    fitter_base = Fitter(model_base, jnp.ones(3), jnp.ones(3) * 0.1, data_type="photometry")
    fitter_scaled = Fitter(model_scaled, jnp.ones(3), jnp.ones(3) * 0.1, data_type="photometry")

    clear_shared_caches()
    base_response, _ = get_or_build_signal_response(fitter_base)
    scaled_response, _ = get_or_build_signal_response(fitter_scaled)

    assert base_response is not scaled_response, (
        "SSP-content siblings must receive different cached signal_response closures; "
        "the scaled model would otherwise decode its latent through the base model's physics"
    )


def test_identical_ssp_content_still_shares_signal_response(ssp_base, obs):
    """Guard the other direction: equal SSP content must still hit the cache."""
    ssp_copy = SSPData(
        ssp_wave=ssp_base.ssp_wave,
        ssp_flux=ssp_base.ssp_flux,
        ssp_lg_age_gyr=ssp_base.ssp_lg_age_gyr,
        ssp_lgmet=ssp_base.ssp_lgmet,
        ssp_mass_remaining=ssp_base.ssp_mass_remaining,
        ssp_alpha_fe=ssp_base.ssp_alpha_fe,
        imf=ssp_base.imf,
        source=ssp_base.source,
        nebular=ssp_base.nebular,
    )

    spec = Parameters(
        redshift=Fixed(0.1),
        sfh_dpl_alpha=Uniform(0.5, 4.0),
        sfh_dpl_beta=Uniform(0.3, 3.0),
    )

    model_base = SEDModel(spec, ssp_base, observation=obs)
    model_copy = SEDModel(spec, ssp_copy, observation=obs)

    fitter_base = Fitter(model_base, jnp.ones(3), jnp.ones(3) * 0.1, data_type="photometry")
    fitter_copy = Fitter(model_copy, jnp.ones(3), jnp.ones(3) * 0.1, data_type="photometry")

    clear_shared_caches()
    base_response, _ = get_or_build_signal_response(fitter_base)
    copy_response, _ = get_or_build_signal_response(fitter_copy)

    assert base_response is copy_response, (
        "identical SSP content must reuse one cached signal_response, "
        "or catalog fits recompile per row"
    )
