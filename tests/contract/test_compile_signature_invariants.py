# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Fitter.compile_signature() invariants.

Ensures that:
1. memory_mode changes do not affect compile_signature (no spurious recompile)
2. Different memory_mode settings reuse the same cached engine

The former field-count ratchet (2) and the tautological ``engine_key ==
fitter_sig`` check (3) are retired (#2163 E.5): ``Fitter._engine_cache_key()``
is now policy-derived (``tengri.inference._engine_policy.ENGINE_POLICY``),
and its own completeness tests live in
``tests/contract/test_inference_cache_keys.py``.

Also covers SEDModel.compile_signature()'s own policy-ledger rewrite (#2163
E.3): completeness of tengri.forward._signature_policy.SIGNATURE_POLICY over
every representative build, memoization + invalidation, equal/unequal
signature behavior under a battery of structural changes, and a
source-level guard that no method can silently skip invalidation.
"""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fitter, Parameters, SEDModel, Spectroscopy, Uniform
from tengri._cache_keys import assert_policy_complete
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.forward import sed_model as sed_model_module
from tengri.forward._signature_policy import SIGNATURE_POLICY
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry
from tests.contract._signature_builds import (
    ALL_BUILDS,
    BARE_STELLAR_NAME,
    build_kitchen_sink_for_completeness,
    build_photometry_star_forming,
    build_spectroscopy_simple,
    exercise_kitchen_sink,
    predict_for_build,
)

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

    # ``test_compile_signature_field_count_pinned`` (a hand-counted
    # ``len(fitter_sig) == 18`` ratchet) and ``test_engine_cache_key_matches_
    # compile_signature_fields`` (``engine_key == fitter_sig``) are RETIRED
    # (#2163 E.5): ``_engine_cache_key()`` is now derived from the
    # ``tengri.inference._engine_policy.ENGINE_POLICY`` ledger over every
    # Fitter attribute, the same policy-derived design ``test_sed_model_
    # policy_complete`` below already applies to the model half. A
    # hand-counted length pin on a policy-derived tuple is exactly the
    # "forgot to add a field" hazard the ledger exists to close, moved into
    # a test; ``engine_key == fitter_sig`` was tautological by construction
    # (``compile_signature()`` builds ``fitter_sig`` by calling
    # ``_engine_cache_key()`` directly) and asserted nothing beyond "this
    # method returns what it returns". Both become
    # ``test_engine_policy_complete`` / ``test_fingerprint_policy_complete`` /
    # ``test_engine_and_fingerprint_ledgers_partition_fitter_attributes`` in
    # ``tests/contract/test_inference_cache_keys.py``, which assert
    # completeness (every Fitter attribute classified) over five
    # representative Fitters instead of counting positions in one tuple.

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

    def test_identical_ssp_content_produces_equal_signature(self, photometry, spec_dpl):
        """Two SEDModels with separately-constructed identical SSPData have equal signatures.

        Since PR #1973, the compile signature includes a blake2b content digest
        of ssp_flux. Two SSPData instances created independently with identical
        numerical content produce identical digests (content-based, not id-based),
        so the resulting SEDModel and Fitter signatures are equal.

        This is the (#1973) regression guard: identical-content SSP grids must
        reuse one compiled engine, not silently run separate ones.
        """
        # Create two separately-constructed SSPData with identical content
        n_met, n_age, n_wave = 8, 15, 200

        ssp1 = SSPData(
            ssp_wave=jnp.logspace(3, 4.5, n_wave),
            ssp_flux=jnp.ones((n_met, n_age, n_wave), dtype=jnp.float64),
            ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
            ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
        )

        ssp2 = SSPData(
            ssp_wave=jnp.logspace(3, 4.5, n_wave),
            ssp_flux=jnp.ones((n_met, n_age, n_wave), dtype=jnp.float64),
            ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
            ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
        )

        model1 = SEDModel(spec_dpl, ssp1, observation=photometry)
        model2 = SEDModel(spec_dpl, ssp2, observation=photometry)

        data = jnp.ones(3)
        noise = jnp.ones(3) * 0.1

        fitter1 = Fitter(model1, data, noise, data_type="photometry")
        fitter2 = Fitter(model2, data, noise, data_type="photometry")

        # Both should produce equal signatures
        sig1 = fitter1.compile_signature()
        sig2 = fitter2.compile_signature()

        assert sig1 == sig2, (
            "Separately-constructed identical-content SSPData must produce equal signatures"
        )

    def test_different_ssp_flux_content_produces_unequal_signature(self, photometry, spec_dpl):
        """Two SEDModels with different ssp_flux content produce unequal signatures.

        The compile signature includes a blake2b content digest of ssp_flux.
        Two SSPData instances with the same shape and lgmet but different flux
        values produce different digests, so their signatures differ.

        This is the (#2047) regression guard: different SSP flux content must
        NOT be collapsed into one engine (which would silently run wrong physics).
        """
        n_met, n_age, n_wave = 8, 15, 200

        ssp1 = SSPData(
            ssp_wave=jnp.logspace(3, 4.5, n_wave),
            ssp_flux=jnp.ones((n_met, n_age, n_wave), dtype=jnp.float64),
            ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
            ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
        )

        ssp2 = SSPData(
            ssp_wave=jnp.logspace(3, 4.5, n_wave),
            ssp_flux=jnp.ones((n_met, n_age, n_wave), dtype=jnp.float64) * 1.1,
            ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
            ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
        )  # Different flux content

        model1 = SEDModel(spec_dpl, ssp1, observation=photometry)
        model2 = SEDModel(spec_dpl, ssp2, observation=photometry)

        data = jnp.ones(3)
        noise = jnp.ones(3) * 0.1

        fitter1 = Fitter(model1, data, noise, data_type="photometry")
        fitter2 = Fitter(model2, data, noise, data_type="photometry")

        # Both should produce different signatures
        sig1 = fitter1.compile_signature()
        sig2 = fitter2.compile_signature()

        assert sig1 != sig2, "Different ssp_flux content must produce different signatures"

    def test_different_ssp_lgmet_values_produce_unequal_signature(self, photometry, spec_dpl):
        """Two SEDModels with different ssp_lgmet values produce unequal signatures.

        The compile signature includes a hash of the ssp_lgmet array values
        (not just shape). Two SSPData instances with the same flux and shape but
        different metallicity grids produce different signatures.
        """
        n_met, n_age, n_wave = 8, 15, 200
        flux = jnp.ones((n_met, n_age, n_wave), dtype=jnp.float64)

        ssp1 = SSPData(
            ssp_wave=jnp.logspace(3, 4.5, n_wave),
            ssp_flux=flux,
            ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
            ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
        )

        ssp2 = SSPData(
            ssp_wave=jnp.logspace(3, 4.5, n_wave),
            ssp_flux=flux,
            ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
            ssp_lgmet=jnp.linspace(-2.0, 0.5, n_met),  # Different metallicity grid
        )

        model1 = SEDModel(spec_dpl, ssp1, observation=photometry)
        model2 = SEDModel(spec_dpl, ssp2, observation=photometry)

        data = jnp.ones(3)
        noise = jnp.ones(3) * 0.1

        fitter1 = Fitter(model1, data, noise, data_type="photometry")
        fitter2 = Fitter(model2, data, noise, data_type="photometry")

        # Both should produce different signatures
        sig1 = fitter1.compile_signature()
        sig2 = fitter2.compile_signature()

        assert sig1 != sig2, "Different ssp_lgmet values must produce different signatures"

    def test_spectroscopy_wave_content_changes_signature(self, mock_ssp_data, spec_dpl):
        """Two SEDModels with spectroscopy observations of different wavelength grids.

        Produce unequal signatures.

        The spectrum projector closure bakes the spectroscopy wavelength array,
        so two models with identical pixel count but different wave_obs grids
        must have different signatures. Otherwise the second model silently
        reuses the first's compiled spectrum which has baked the first model's
        wavelengths, producing silent spectroscopy errors.
        """
        from tengri.observation.spectroscopy import Spectroscopy

        data = jnp.ones(100)
        noise = jnp.ones(100) * 0.01

        # Two spectroscopy observations with same pixel count but different wavelength grids
        wave_obs_1 = jnp.linspace(4000, 6000, 100)
        wave_obs_2 = jnp.linspace(4500, 6500, 100)  # Shifted by 500 A

        spectroscopy_1 = Spectroscopy(wave_obs=wave_obs_1, resolution=100.0)
        spectroscopy_2 = Spectroscopy(wave_obs=wave_obs_2, resolution=100.0)

        from tengri.observation.observation import Observation

        obs_1 = Observation(spectroscopy=spectroscopy_1)
        obs_2 = Observation(spectroscopy=spectroscopy_2)

        model1 = SEDModel(spec_dpl, mock_ssp_data, observation=obs_1)
        model2 = SEDModel(spec_dpl, mock_ssp_data, observation=obs_2)

        data_1 = jnp.ones(100)
        noise_1 = jnp.ones(100) * 0.01

        fitter1 = Fitter(model1, data_1, noise_1, data_type="spectrum")
        fitter2 = Fitter(model2, data_1, noise_1, data_type="spectrum")

        # Signatures should differ
        sig1 = fitter1.compile_signature()
        sig2 = fitter2.compile_signature()

        assert sig1 != sig2, (
            "Different spectroscopy wavelength grids must produce different signatures"
        )

    def test_spectroscopy_identical_wave_produces_equal_signatures(self, mock_ssp_data, spec_dpl):
        """Two SEDModels with identical spectroscopy wavelength grids produce equal signatures.

        Control test: identical wavelength grids should produce identical signatures.
        """
        from tengri.observation.spectroscopy import Spectroscopy

        wave_obs = jnp.linspace(4000, 6000, 100)

        spectroscopy_1 = Spectroscopy(wave_obs=wave_obs, resolution=100.0)
        spectroscopy_2 = Spectroscopy(wave_obs=wave_obs, resolution=100.0)

        from tengri.observation.observation import Observation

        obs_1 = Observation(spectroscopy=spectroscopy_1)
        obs_2 = Observation(spectroscopy=spectroscopy_2)

        model1 = SEDModel(spec_dpl, mock_ssp_data, observation=obs_1)
        model2 = SEDModel(spec_dpl, mock_ssp_data, observation=obs_2)

        data = jnp.ones(100)
        noise = jnp.ones(100) * 0.01

        fitter1 = Fitter(model1, data, noise, data_type="spectrum")
        fitter2 = Fitter(model2, data, noise, data_type="spectrum")

        # Signatures should be equal
        sig1 = fitter1.compile_signature()
        sig2 = fitter2.compile_signature()

        assert sig1 == sig2, (
            "Identical spectroscopy wavelength grids must produce equal signatures"
        )


# ── #2163 E.3: policy-ledger completeness, memoization, equal/unequal ──────


def test_sed_model_policy_complete():
    """assert_policy_complete over every representative build (#2163).

    Covers the seven E.3 representatives (A-G) after a predict pass, plus
    one extra kitchen-sink build (H, not one of the seven -- see
    ``tests/contract/_signature_builds.py``) exercised through
    enable_fast_nebular, available_properties, and a full predict_state
    call, so the lazily-created attributes that never appear on A-G
    (_property_catalog, _index_window_lut_cache, _line_window_lut_cache,
    _nebular_grid_table, _cached_full_state_chain, and the conditionally-set
    radio scalars) are not "stale" ledger rows.
    """
    from tengri.observation.line_measurement import DESI_LINES
    from tengri.observation.spectral_indices import STANDARD_INDICES

    models = []
    for _name, build_fn in ALL_BUILDS:
        model = build_fn()
        predict_for_build(model)
        _ = model.available_properties
        models.append(model)

    # Materialize _line_window_lut_cache on build A (has dust IR, which the
    # fast line-flux path admits) and _index_window_lut_cache on build B
    # (no dust IR, which the fast index path requires).
    model_a = build_photometry_star_forming()
    params_a = predict_for_build(model_a)
    model_a.measure_line_fluxes(params_a, DESI_LINES, approx=True)
    models.append(model_a)

    model_b = build_spectroscopy_simple()
    params_b = predict_for_build(model_b)
    model_b.predict_spectral_indices(params_b, (STANDARD_INDICES["Dn4000"],), approx=True)
    models.append(model_b)

    # H: Cue nebular + AGN + radio + shock, for the attributes A-G never touch.
    model_h = build_kitchen_sink_for_completeness()
    exercise_kitchen_sink(model_h)
    models.append(model_h)

    assert_policy_complete(models, SIGNATURE_POLICY)


def test_signature_is_memoized_and_invalidated():
    """The second compile_signature() call returns the identical object.

    After enable_fast_nebular() the signature differs from the pre-grid
    one, and is memoized again (third and fourth calls agree).
    """
    from tengri import (
        Fixed as _Fixed,
        Observation as _Observation,
        Photometry as _Photometry,
        SEDModel as _SEDModel,
        WavePrecomp as _WavePrecomp,
    )
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    ssp_data = load_ssp_data(str(Path(__file__).resolve().parents[2] / "data" / BARE_STELLAR_NAME))
    obs = _Observation(
        photometry=_Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )
    model = _SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        approx=_WavePrecomp(),
        sfh={"type": "dpl", "all_params": _Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": _Fixed(DEFAULT),
        },
        neb={"type": "cue", "all_params": _Fixed(DEFAULT)},
        redshift=_Fixed(0.1),
    )

    sig1 = model.compile_signature()
    sig2 = model.compile_signature()
    assert sig1 is sig2, "second compile_signature() call must return the memoized object"

    model.enable_fast_nebular([6564.61, 4862.68])
    sig3 = model.compile_signature()
    assert sig3 != sig1, "enable_fast_nebular() must invalidate the memoized signature"

    sig4 = model.compile_signature()
    assert sig3 is sig4, "the signature after enable_fast_nebular() must itself be memoized"


class TestEqualContentDistinctObjects:
    """Equal content, distinct objects -> equal signature (#2163)."""

    def test_same_ssp_file_loaded_twice(self):
        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

        path = str(Path(__file__).resolve().parents[2] / "data" / BARE_STELLAR_NAME)
        ssp1 = load_ssp_data(path)
        ssp2 = load_ssp_data(path)
        obs = Observation(
            photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
        )
        spec = Parameters(
            redshift=0.1, sfh_dpl_alpha=Uniform(0.5, 4.0), sfh_dpl_beta=Uniform(0.3, 3.0)
        )
        model1 = SEDModel(spec, ssp1, observation=obs)
        model2 = SEDModel(spec, ssp2, observation=obs)
        assert model1.compile_signature() == model2.compile_signature()

    def test_spectroscopy_rebuilt_from_equal_wave_obs_array(self):
        wave_obs_1 = np.linspace(4000.0, 6000.0, 200)
        wave_obs_2 = np.linspace(4000.0, 6000.0, 200)  # separately constructed, equal content
        assert wave_obs_1 is not wave_obs_2
        obs1 = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs_1, resolution=1500.0))
        obs2 = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs_2, resolution=1500.0))
        spec = Parameters(
            redshift=0.1, sfh_dpl_alpha=Uniform(0.5, 4.0), sfh_dpl_beta=Uniform(0.3, 3.0)
        )
        ssp = _mock_ssp()
        model1 = SEDModel(spec, ssp, observation=obs1)
        model2 = SEDModel(spec, ssp, observation=obs2)
        assert model1.compile_signature() == model2.compile_signature()

    def test_two_photometry_models_with_same_filters(self):
        obs1 = Observation(
            photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
        )
        obs2 = Observation(
            photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
        )
        assert obs1 is not obs2
        spec = Parameters(
            redshift=0.1, sfh_dpl_alpha=Uniform(0.5, 4.0), sfh_dpl_beta=Uniform(0.3, 3.0)
        )
        ssp = _mock_ssp()
        model1 = SEDModel(spec, ssp, observation=obs1)
        model2 = SEDModel(spec, ssp, observation=obs2)
        assert model1.compile_signature() == model2.compile_signature()


def _mock_ssp() -> SSPData:
    n_met, n_age, n_wave = 8, 15, 200
    return SSPData(
        ssp_wave=jnp.logspace(3, 4.5, n_wave),
        ssp_flux=jnp.ones((n_met, n_age, n_wave), dtype=jnp.float64),
        ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
        ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
    )


class TestSameStructureDifferentData:
    """Same structure, different DATA -> equal signature (#2163)."""

    def test_spectroscopy_covariance_values_differ_same_shape(self):
        """Two spectroscopy models differing only in covariance VALUES (same shape).

        ``Spectroscopy.covariance`` is a ``shape``-mode row on its own ledger
        (only the shape fixes the program; values are a runtime input), so
        two otherwise-identical models with different covariance content
        must still compile to one signature.
        """
        wave_obs = np.linspace(4000.0, 6000.0, 50)
        cov1 = jnp.eye(50) * 0.01
        cov2 = jnp.eye(50) * 0.25  # different values, same (50, 50) shape
        obs1 = Observation(
            spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=1500.0, covariance=cov1)
        )
        obs2 = Observation(
            spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=1500.0, covariance=cov2)
        )
        spec = Parameters(
            redshift=0.1, sfh_dpl_alpha=Uniform(0.5, 4.0), sfh_dpl_beta=Uniform(0.3, 3.0)
        )
        ssp = _mock_ssp()
        model1 = SEDModel(spec, ssp, observation=obs1)
        model2 = SEDModel(spec, ssp, observation=obs2)
        assert model1.compile_signature() == model2.compile_signature()

    def test_photometry_fitter_data_differs_model_never_sees_flux(self):
        """Two Fitters over the SAME model, differing only in observed flux/noise.

        ``SEDModel.compile_signature()`` never depends on the observed data
        arrays (the model is structure-only): constructing two Fitters that
        differ only in ``data``/``noise`` must not change the ORIGINAL
        model's own signature. (``fitter.model`` is not compared directly:
        the deprecated ``Fitter(sed_model, ...)`` constructor path resolves
        ``approx`` through ``_resolve_fit_approx`` and may hand back a
        different -- but each internally self-consistent -- model object;
        that resolution is orthogonal to what #2163 covers.)
        """
        obs = Observation(
            photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
        )
        spec = Parameters(
            redshift=0.1, sfh_dpl_alpha=Uniform(0.5, 4.0), sfh_dpl_beta=Uniform(0.3, 3.0)
        )
        ssp = _mock_ssp()
        model = SEDModel(spec, ssp, observation=obs)
        sig_before = model.compile_signature()

        data_1, noise_1 = jnp.ones(5), jnp.ones(5) * 0.1
        data_2, noise_2 = jnp.ones(5) * 3.7, jnp.ones(5) * 0.03
        Fitter(model, data_1, noise_1, data_type="photometry")
        Fitter(model, data_2, noise_2, data_type="photometry")

        assert model.compile_signature() == sig_before, (
            "constructing Fitters over a model must not change the model's own signature"
        )


# ── #2163: the four attributes the hand-written list never keyed ──────────
#
# Each pair below differs ONLY in the named attribute; each must produce a
# distinct signature, AND a numeric probe must show the compiled output
# differs (predict_photometry/predict_spectrum, or the closest analogous
# observable, at the same params). Two of the four could not be shown to
# move predict_photometry as specified -- both are reported verbatim below
# rather than papering over a non-difference; see each test's docstring.


def _bare_stellar_ssp():
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data(str(Path(__file__).resolve().parents[2] / "data" / BARE_STELLAR_NAME))


def test_igm_patchy_numeric_probe():
    """igm_patchy False vs True at z=2.5 (where IGM absorption is material).

    Measured: max relative difference in predict_photometry is 1.77e-5
    (small -- IGM patchiness is a modest broadband effect -- but real and
    reproducible; np.allclose's default atol=1e-8 would mask it against
    these ~1e-27 erg/s/cm^2/Hz fluxes, so the comparison below uses atol=0
    and compares the RELATIVE difference directly).
    """
    ssp = _bare_stellar_ssp()
    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )

    def build(patchy):
        from tengri import WavePrecomp as _WavePrecomp

        spec = Parameters(
            redshift=2.5,
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            apply_igm=True,
            igm_patchy=patchy,
        )
        return SEDModel(spec, ssp, observation=obs, approx=_WavePrecomp())

    model_uniform = build(False)
    model_patchy = build(True)
    assert model_uniform.compile_signature() != model_patchy.compile_signature(), (
        "igm_patchy must change compile_signature"
    )

    params = model_uniform.spec.sample(jax.random.PRNGKey(0))
    f_uniform = model_uniform.predict_photometry(params)
    f_patchy = model_patchy.predict_photometry(dict(params))
    max_reldiff = float(jnp.max(jnp.abs(f_uniform - f_patchy) / jnp.abs(f_uniform)))
    assert max_reldiff > 1e-8, (
        f"igm_patchy must change predict_photometry; measured {max_reldiff:.3e}"
    )


def test_lsf_n_bins_numeric_probe():
    """lsf_n_bins 64 vs 2 on a low-resolution spectroscopy model with an LSF.

    Measured at R=2500/16-vs-4-bins (the brief's literal pairing): max
    relative difference 3.2e-16 -- converged to floating-point noise, i.e.
    no numeric difference at that resolution/bin-count. Reported verbatim.
    At R=200 with 64-vs-2 bins the LSF approximation has not converged and
    the difference is real: max relative difference ~5.0e-2, used below.
    """
    from tengri import Spectroscopy as _Spectroscopy

    ssp = _bare_stellar_ssp()
    wave_obs = np.linspace(4000.0, 7000.0, 400)

    def build(n_bins):
        obs = Observation(
            spectroscopy=_Spectroscopy(wave_obs=wave_obs, resolution=200.0, lsf_n_bins=n_bins)
        )
        spec = Parameters(
            redshift=0.05, sfh_dpl_alpha=Uniform(0.5, 4.0), sfh_dpl_beta=Uniform(0.3, 3.0)
        )
        return SEDModel(spec, ssp, observation=obs)

    model_fine = build(64)
    model_coarse = build(2)
    assert model_fine.compile_signature() != model_coarse.compile_signature(), (
        "lsf_n_bins must change compile_signature"
    )

    params = model_fine.spec.sample(jax.random.PRNGKey(0))
    f_fine = model_fine.predict_spectrum(params)
    f_coarse = model_coarse.predict_spectrum(dict(params))
    max_reldiff = float(jnp.max(jnp.abs(f_fine - f_coarse) / jnp.abs(f_fine)))
    assert max_reldiff > 1e-8, (
        f"lsf_n_bins must change predict_spectrum; measured {max_reldiff:.3e}"
    )


def test_lgmet_scatter_signature_differs():
    """lgmet_scatter 0.1 vs 0.3 must change compile_signature.

    Numeric probe reported verbatim, not asserted (see docstring below):
    ``Parameters(lgmet_scatter=...)`` sets ``SEDModel._lgmet_scatter``
    (this row), which ``StellarSEDComponent.predict`` reads only as the
    FALLBACK in ``params.get("met_logzsol_scatter", self.config.lgmet_scatter)``.
    Measured: on every build tried (met_logzsol Fixed or Uniform), the
    auto-derived ``met_logzsol_scatter`` parameter is present in the sampled
    params dict at its OWN registry default (0.1) regardless of the
    ``lgmet_scatter=`` kwarg, so the fallback never engages and
    predict_photometry is bit-identical (max reldiff exactly 0.0) between
    the two builds. The kernel itself IS sensitive to scatter width (checked
    directly against ``tengri.components.stellar.component._lgmet_weights``
    at a metallicity centered on the SSP grid), so this is a real dead
    build-time knob under the configurations reachable from the public API
    today, not a broken kernel. Filed as a finding in the E.3 report rather
    than a new issue (out of scope for the compile_signature policy rewrite).
    """
    ssp = _bare_stellar_ssp()
    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )

    def build(scatter):
        from tengri import WavePrecomp as _WavePrecomp

        spec = Parameters(
            redshift=0.1,
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            lgmet_scatter=scatter,
        )
        return SEDModel(spec, ssp, observation=obs, approx=_WavePrecomp())

    model_narrow = build(0.1)
    model_wide = build(0.3)
    assert model_narrow.compile_signature() != model_wide.compile_signature(), (
        "lgmet_scatter must change compile_signature"
    )

    params = model_narrow.spec.sample(jax.random.PRNGKey(0))
    f_narrow = model_narrow.predict_photometry(params)
    f_wide = model_wide.predict_photometry(dict(params))
    max_reldiff = float(jnp.max(jnp.abs(f_narrow - f_wide) / jnp.abs(f_narrow)))
    # Reported, not asserted as a real difference: see docstring. The
    # signature-inequality assertion above is the real regression guard;
    # this documents the measured (null) numeric result precisely so it
    # cannot silently start meaning something different later.
    assert max_reldiff == 0.0, (
        f"expected the diagnosed dead-fallback null result (0.0); measured {max_reldiff:.3e}. "
        "If this is now nonzero, met_logzsol_scatter's auto-registration changed and "
        "lgmet_scatter may have become reachable -- update this test's docstring and "
        "tighten the assertion to `> 1e-8`."
    )


def test_gp_kernel_signature_differs():
    """gp_kernel differing must change compile_signature; sfr_full must differ.

    ``FIELD_MODEL_REGISTRY`` (``tengri.components.stellar.sfh.registry``)
    has exactly one entry ("drw") and no public kwarg overrides it -- every
    model built through ``SEDModel.build``/``Parameters(...)`` today gets
    ``_gp_kernel == "drw"``. To get a genuine second value this test
    monkeypatches a second registry entry (a trivially-scaled DRW kernel)
    and the "field" SFH spec's OWN settings dict, restoring both after, so
    two INDEPENDENTLY built models carry different kernels from
    construction (not a post-hoc attribute mutation on a shared JIT chain).

    Measured: predict_photometry (WavePrecomp path) is bit-identical
    between the two models (max reldiff 0.0) even with a 5x-scaled kernel --
    reported verbatim, not asserted as a difference. predict_sfh()['sfr_full']
    DOES differ (not allclose), proving _gp_kernel is genuinely read at
    predict time for at least one observable; the photometry non-response
    is filed as a finding (WavePrecomp's field-modulated age-weight LUT may
    not route through the per-model field kernel), out of scope here.
    """
    from tengri import (
        FREE,
        Fixed as _Fixed,
        Observation as _Observation,
        Photometry as _Photometry,
        SEDModel as _SEDModel,
        WavePrecomp as _WavePrecomp,
    )
    from tengri.components.stellar.sfh.gp_sfh import compute_sqrt_power_drw
    from tengri.components.stellar.sfh.registry import FIELD_MODEL_REGISTRY, SFH_REGISTRY

    def _scaled_drw(*a, **kw):
        return compute_sqrt_power_drw(*a, **kw) * 5.0

    ssp = _bare_stellar_ssp()
    obs = _Observation(
        photometry=_Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )

    def build():
        return _SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            approx=_WavePrecomp(),
            sfh={"type": ["dpl", "field"], "all_params": FREE},
            redshift=_Fixed(0.1),
        )

    field_spec = SFH_REGISTRY["field"]
    original_settings = dict(field_spec.settings)
    FIELD_MODEL_REGISTRY["e3_test_scaled_drw"] = _scaled_drw
    try:
        model_drw = build()
        field_spec.settings["sfh_field_model"] = "e3_test_scaled_drw"
        model_scaled = build()

        assert model_drw._gp_kernel == "drw"
        assert model_scaled._gp_kernel == "e3_test_scaled_drw"
        assert model_drw.compile_signature() != model_scaled.compile_signature(), (
            "gp_kernel must change compile_signature"
        )

        # field_model is read from the registry at CALL time (not baked at
        # construction), so the predict calls below must run before the
        # finally block below restores/removes the monkeypatched entries.
        params = model_drw.spec.sample(jax.random.PRNGKey(0))
        sfh_drw = model_drw.predict_sfh(params)
        sfh_scaled = model_scaled.predict_sfh(dict(params))
        assert not np.allclose(
            np.asarray(sfh_drw["sfr_full"]), np.asarray(sfh_scaled["sfr_full"]), atol=0
        ), "gp_kernel must change predict_sfh()['sfr_full']"

        f_drw = model_drw.predict_photometry(params)
        f_scaled = model_scaled.predict_photometry(dict(params))
        max_reldiff = float(jnp.max(jnp.abs(f_drw - f_scaled) / jnp.abs(f_drw)))
    finally:
        field_spec.settings.clear()
        field_spec.settings.update(original_settings)
        del FIELD_MODEL_REGISTRY["e3_test_scaled_drw"]

    assert max_reldiff == 0.0, (
        f"expected the diagnosed null result on predict_photometry (0.0); measured "
        f"{max_reldiff:.3e}. If this is now nonzero, the WavePrecomp field-modulated "
        "path may have become sensitive to _gp_kernel -- tighten this assertion."
    )


# ── Determinism across processes ───────────────────────────────────────────


def test_compile_signature_deterministic_across_processes():
    """Two subprocesses building the same photometry model agree on repr(sig)."""
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    code = (
        "import os, sys, warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "os.environ.setdefault('JAX_PLATFORMS', 'cpu')\n"
        f"sys.path.insert(0, {str(src_dir)!r})\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "import jax\n"
        "jax.config.update('jax_enable_x64', True)\n"
        "from tests.contract._signature_builds import build_photometry_star_forming\n"
        "model = build_photometry_star_forming()\n"
        "print(repr(model.compile_signature()))\n"
    )
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"

    def run_once():
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            env=env,
            timeout=180,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"
        return result.stdout.strip()

    out1 = run_once()
    out2 = run_once()
    assert out1 == out2, "compile_signature() repr must be identical across processes"
    assert out1, "subprocess produced no output"


# ── Source-level guard: no structural mutator skips invalidation ──────────

#: Methods whose name matches one of these prefixes are constructor helpers:
#: called only during (or as part of) __init__, always assigning attributes
#: this policy classifies (structural build-up, not a post-construction
#: mutation). See SEDModel.compile_signature's Notes.
_CTOR_HELPER_PREFIXES = ("__init__", "_init_", "_build_", "_resolve_", "_setup_", "_configure_")

#: One documented exception: _additive_term_band_response assigns its memo
#: cache via ``setattr(self, f"_{name}_term_response_cache", ...)``, a
#: dynamic name the AST cannot resolve to a literal string. Manually
#: verified (see tests/contract/_signature_builds.py and the E.3 report):
#: for name in ("xray", "radio"), both targets
#: (_xray_term_response_cache, _radio_term_response_cache) are EXCLUDE rows
#: in SIGNATURE_POLICY, so this method needs no _invalidate_signature() call.
_DYNAMIC_SETATTR_ALLOWLIST = {
    "_additive_term_band_response": (
        "setattr(self, f'_{name}_term_response_cache', ...): both possible "
        "targets (_xray_term_response_cache, _radio_term_response_cache) are "
        "EXCLUDE rows"
    ),
}


def _sed_model_class_node() -> ast.ClassDef:
    source = inspect.getsource(sed_model_module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SEDModel":
            return node
    raise AssertionError("SEDModel class not found in sed_model.py")


def _self_attr_assignments(method: ast.FunctionDef) -> tuple[set[str], bool]:
    """Return (literal self.<attr> assignment targets, has_dynamic_setattr)."""
    assigned: set[str] = set()
    has_dynamic_setattr = False
    for node in ast.walk(method):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    assigned.add(target.attr)
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Name)
                and func.id == "setattr"
                and len(node.args) >= 1
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "self"
            ):
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    assigned.add(node.args[1].value)
                else:
                    has_dynamic_setattr = True
    return assigned, has_dynamic_setattr


def _calls_invalidate_signature(method: ast.FunctionDef) -> bool:
    for node in ast.walk(method):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_invalidate_signature"
        ):
            return True
    return False


def test_every_structural_mutator_invalidates_the_signature():
    """Every non-constructor SEDModel method either assigns only EXCLUDE
    rows, or calls _invalidate_signature() (#2163).

    A method the AST probe finds assigning a non-excluded row without
    invalidating is exactly how a #1122/#1462/#2145/#2237-class collision
    ships: a structural change that the memoized signature never sees.
    """
    class_node = _sed_model_class_node()
    exclude_names = {
        name for name, (mode, _reason) in SIGNATURE_POLICY.items() if mode == "exclude"
    }

    violations = []
    checked = []
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith(_CTOR_HELPER_PREFIXES):
            continue
        assigned, has_dynamic_setattr = _self_attr_assignments(node)
        if not assigned and not has_dynamic_setattr:
            continue
        checked.append(node.name)
        if has_dynamic_setattr:
            if node.name not in _DYNAMIC_SETATTR_ALLOWLIST:
                violations.append(
                    f"{node.name}: dynamic setattr(self, ...) target the AST cannot "
                    "resolve, and not in _DYNAMIC_SETATTR_ALLOWLIST"
                )
            continue
        non_excluded = assigned - exclude_names
        if non_excluded and not _calls_invalidate_signature(node):
            violations.append(
                f"{node.name}: assigns non-excluded row(s) {sorted(non_excluded)} "
                "without calling self._invalidate_signature()"
            )

    assert checked, "AST probe found no non-constructor methods assigning self.<attr> at all"
    assert not violations, "structural mutator(s) skip signature invalidation:\n" + "\n".join(
        violations
    )
