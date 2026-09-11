# SPDX-License-Identifier: BSD-3-Clause
"""Tests for nested cache_key() implementations on observation and parameter classes.

Validates that each class correctly delegates cache key computation via
a written policy ledger, covering content/shape/exclude rows, mutation
detection, and determinism across processes.
"""

import os
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tengri._cache_keys import assert_policy_complete
from tengri.components.stellar.sps.dsps_wrapper import _SSP_CACHE_KEY_POLICY, SSPData
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.line_flux_data import _LINE_FLUX_DATA_CACHE_KEY_POLICY, LineFluxData
from tengri.observation.line_list import (
    _DOUBLET_CONSTRAINT_CACHE_KEY_POLICY,
    _LINE_LIST_CACHE_KEY_POLICY,
    DoubletConstraint,
    LineList,
)
from tengri.observation.line_ratio_data import _LINE_RATIO_DATA_CACHE_KEY_POLICY, LineRatioData
from tengri.observation.noise_model import _NOISE_MODEL_CACHE_KEY_POLICY, NoiseModel
from tengri.observation.observation import _OBSERVATION_CACHE_KEY_POLICY, Observation
from tengri.observation.photometry import _FILTER_CURVE_CACHE_KEY_POLICY, FilterCurve
from tengri.observation.photometry_config import _PHOTOMETRY_CACHE_KEY_POLICY, Photometry
from tengri.observation.spectral_indices import (
    _SPECTRAL_INDEX_DATA_CACHE_KEY_POLICY,
    SpectralIndexData,
)
from tengri.observation.spectroscopy import _SPECTROSCOPY_CACHE_KEY_POLICY, Spectroscopy
from tengri.parameters import FREE, Fixed, Uniform
from tengri.parameters.groups import parse_groups
from tengri.parameters.parameters import _PARAMETERS_CACHE_KEY_POLICY, Parameters

pytestmark = pytest.mark.contract


# ── Policy Completeness ────────────────────────────────────────────


class TestFilterCurvePolicy:
    """FilterCurve cache_key policy coverage."""

    def test_policy_complete(self):
        """All FilterCurve attributes are classified."""
        fc = FilterCurve(wave=jnp.array([4000, 5000]), trans=jnp.array([0, 1]), name="test")
        assert_policy_complete([fc], _FILTER_CURVE_CACHE_KEY_POLICY)

    def test_cache_key_generated(self):
        """cache_key() returns a tuple."""
        fc = FilterCurve(wave=jnp.array([4000, 5000]), trans=jnp.array([0, 1]), name="test")
        key = fc.cache_key()
        assert isinstance(key, tuple)


class TestPhotometryPolicy:
    """Photometry cache_key policy coverage."""

    def test_policy_complete(self):
        """All Photometry attributes are classified."""
        phot = Photometry.from_names(["sdss_g", "sdss_r"])
        assert_policy_complete([phot], _PHOTOMETRY_CACHE_KEY_POLICY)

    def test_cache_key_generated(self):
        """cache_key() returns a tuple."""
        phot = Photometry.from_names(["sdss_g", "sdss_r"])
        key = phot.cache_key()
        assert isinstance(key, tuple)


class TestSpectroscopyPolicy:
    """Spectroscopy cache_key policy coverage."""

    def test_policy_complete(self):
        """All Spectroscopy attributes are classified."""
        spec = Spectroscopy(wave_obs=jnp.linspace(4000, 9000, 500))
        assert_policy_complete([spec], _SPECTROSCOPY_CACHE_KEY_POLICY)

    def test_cache_key_generated(self):
        """cache_key() returns a tuple."""
        spec = Spectroscopy(
            wave_obs=jnp.linspace(4000, 9000, 500),
            covariance=jnp.eye(500) * 0.1,
        )
        key = spec.cache_key()
        assert isinstance(key, tuple)

    def test_covariance_shape_insensitive_to_values(self):
        """Covariance VALUES do not change cache key, only shape does.

        Per-galaxy data: the shape of the covariance matrix (which fixes
        program structure) affects the key, but the VALUES are runtime input
        and must not.
        """
        wave_obs = jnp.linspace(4000, 9000, 100)
        cov_shape = (100, 100)

        # Same wave_obs, same-shaped covariance, but different values
        spec1 = Spectroscopy(
            wave_obs=wave_obs,
            covariance=jnp.eye(100) * 0.1,
        )
        spec2 = Spectroscopy(
            wave_obs=wave_obs,
            covariance=jnp.eye(100) * 0.5,  # Different values, same shape
        )

        # Cache keys must be EQUAL (same shape → same structure)
        assert spec1.cache_key() == spec2.cache_key()


class TestObservationPolicy:
    """Observation cache_key policy coverage."""

    def test_policy_complete(self):
        """All Observation attributes are classified."""
        phot = Photometry.from_names(["sdss_g", "sdss_r"])
        obs = Observation(photometry=phot)
        assert_policy_complete([obs], _OBSERVATION_CACHE_KEY_POLICY)

    def test_cache_key_generated(self):
        """cache_key() returns a tuple."""
        phot = Photometry.from_names(["sdss_g", "sdss_r"])
        obs = Observation(photometry=phot)
        key = obs.cache_key()
        assert isinstance(key, tuple)


class TestLineFluxDataPolicy:
    """LineFluxData cache_key policy coverage."""

    def test_policy_complete(self):
        """All LineFluxData attributes are classified."""
        lfd = LineFluxData(
            names=("Halpha", "Hbeta"),
            fluxes=jnp.array([1.2e-16, 3.5e-17]),
            errors=jnp.array([0.1e-16, 0.5e-17]),
            wavelengths=jnp.array([6564.61, 4862.68]),
        )
        assert_policy_complete([lfd], _LINE_FLUX_DATA_CACHE_KEY_POLICY)

    def test_cache_key_generated(self):
        """cache_key() returns a tuple."""
        lfd = LineFluxData(
            names=("Halpha", "Hbeta"),
            fluxes=jnp.array([1.2e-16, 3.5e-17]),
            errors=jnp.array([0.1e-16, 0.5e-17]),
            wavelengths=jnp.array([6564.61, 4862.68]),
        )
        key = lfd.cache_key()
        assert isinstance(key, tuple)


class TestLineRatioDataPolicy:
    """LineRatioData cache_key policy coverage."""

    def test_policy_complete(self):
        """All LineRatioData attributes are classified."""
        lrd = LineRatioData(
            numerators=("Halpha",),
            denominators=("Hbeta",),
            ratios=jnp.array([4.2]),
            errors=jnp.array([0.3]),
            numerator_waves=jnp.array([6564.61]),
            denominator_waves=jnp.array([4862.68]),
        )
        assert_policy_complete([lrd], _LINE_RATIO_DATA_CACHE_KEY_POLICY)

    def test_cache_key_generated(self):
        """cache_key() returns a tuple."""
        lrd = LineRatioData(
            numerators=("Halpha",),
            denominators=("Hbeta",),
            ratios=jnp.array([4.2]),
            errors=jnp.array([0.3]),
            numerator_waves=jnp.array([6564.61]),
            denominator_waves=jnp.array([4862.68]),
        )
        key = lrd.cache_key()
        assert isinstance(key, tuple)


class TestSpectralIndexDataPolicy:
    """SpectralIndexData cache_key policy coverage."""

    def test_policy_complete(self):
        """All SpectralIndexData attributes are classified."""
        sid = SpectralIndexData.from_names(["Dn4000"], [1.35], [0.05])
        assert_policy_complete([sid], _SPECTRAL_INDEX_DATA_CACHE_KEY_POLICY)

    def test_cache_key_generated(self):
        """cache_key() returns a tuple."""
        sid = SpectralIndexData.from_names(["Dn4000"], [1.35], [0.05])
        key = sid.cache_key()
        assert isinstance(key, tuple)


class TestDoubletConstraintPolicy:
    """DoubletConstraint cache_key policy coverage."""

    def test_policy_complete(self):
        """All DoubletConstraint attributes are classified."""
        dc = DoubletConstraint(primary_idx=0, secondary_idx=1, ratio=2.98)
        assert_policy_complete([dc], _DOUBLET_CONSTRAINT_CACHE_KEY_POLICY)

    def test_cache_key_generated(self):
        """cache_key() returns a tuple."""
        dc = DoubletConstraint(primary_idx=0, secondary_idx=1, ratio=2.98)
        key = dc.cache_key()
        assert isinstance(key, tuple)


class TestLineListPolicy:
    """LineList cache_key policy coverage."""

    def test_policy_complete(self):
        """All LineList attributes are classified."""
        cat = LineList.default_13()
        assert_policy_complete([cat], _LINE_LIST_CACHE_KEY_POLICY)

    def test_cache_key_generated(self):
        """cache_key() returns a tuple."""
        cat = LineList.default_13()
        key = cat.cache_key()
        assert isinstance(key, tuple)


class TestNoiseModelPolicy:
    """NoiseModel cache_key policy coverage."""

    def test_policy_complete(self):
        """All NoiseModel attributes are classified."""
        nm = NoiseModel()
        assert_policy_complete([nm], _NOISE_MODEL_CACHE_KEY_POLICY)

    def test_cache_key_generated(self):
        """cache_key() returns a tuple."""
        nm = NoiseModel()
        key = nm.cache_key()
        assert isinstance(key, tuple)


class TestSSPDataPolicy:
    """SSPData cache_key policy coverage."""

    def test_policy_complete(self):
        """All SSPData fields are classified in the policy."""
        # SSPData is a NamedTuple, so vars() doesn't work on instances.
        # Check that the policy covers all SSPData fields instead.
        policy_fields = set(_SSP_CACHE_KEY_POLICY.keys())
        ssp_fields = set(SSPData._fields)
        assert policy_fields == ssp_fields, f"Policy mismatch: {policy_fields ^ ssp_fields}"

    def test_cache_key_generated(self):
        """cache_key() returns a tuple."""
        ssp = SSPData(
            ssp_wave=jnp.array([1000.0, 2000.0]),
            ssp_flux=jnp.zeros((2, 3, 2)),
            ssp_lg_age_gyr=jnp.array([6.0, 10.0, 10.1]),
            ssp_lgmet=jnp.array([-2.0, 0.0]),
            ssp_mass_remaining=jnp.ones((2, 3)),
            ssp_alpha_fe=None,
            imf="chabrier",
            source="miles",
            nebular="bare",
        )
        key = ssp.cache_key()
        assert isinstance(key, tuple)


class TestParametersPolicy:
    """Parameters cache_key policy coverage."""

    def test_policy_complete_over_flat_and_grammar_specs(self):
        """The policy classifies every attribute across both spec kinds.

        ``assert_policy_complete`` takes the union of ``vars()`` over the
        objects it is given (see its docstring in ``_cache_keys.py``), so
        passing a flat spec and a grammar-built spec together in one call
        checks the full attribute surface, including ``_group_provenance``
        (grammar-only). Filtering that attribute out of the ledger before
        calling the helper -- as the two tests this replaces did for the
        flat case -- would hide a stale ``_group_provenance`` policy row
        instead of catching it.
        """
        flat = Parameters()
        grammar = parse_groups(
            sfh={"type": "dpl", "all_params": FREE},
            redshift=Fixed(0.1),
        )
        assert_policy_complete([flat, grammar], _PARAMETERS_CACHE_KEY_POLICY)

    def test_cache_key_generated(self):
        """cache_key() returns a tuple."""
        params = Parameters()
        key = params.cache_key()
        assert isinstance(key, tuple)


# ── Equal Content, Distinct Objects → Equal Key ────────────────────


class TestKeyEquivalence:
    """Verify same content produces same cache key."""

    def test_photometry_same_filters(self):
        """Two Photometry objects with same filters have equal keys."""
        phot1 = Photometry.from_names(["sdss_g", "sdss_r"])
        phot2 = Photometry.from_names(["sdss_g", "sdss_r"])
        assert phot1.cache_key() == phot2.cache_key()

    def test_spectroscopy_same_wave(self):
        """Two Spectroscopy objects with same wave_obs have equal keys."""
        wave = jnp.linspace(4000, 9000, 500)
        cov = jnp.eye(500) * 0.1
        spec1 = Spectroscopy(wave_obs=wave, resolution=1000.0, covariance=cov)
        spec2 = Spectroscopy(wave_obs=wave, resolution=1000.0, covariance=cov)
        assert spec1.cache_key() == spec2.cache_key()

    def test_ssp_same_arrays(self):
        """Two SSPData with identical arrays have equal keys."""
        wave = jnp.array([1000.0, 2000.0])
        flux = jnp.zeros((2, 3, 2))
        age = jnp.array([6.0, 10.0, 10.1])
        met = jnp.array([-2.0, 0.0])
        mass = jnp.ones((2, 3))

        ssp1 = SSPData(
            ssp_wave=wave,
            ssp_flux=flux,
            ssp_lg_age_gyr=age,
            ssp_lgmet=met,
            ssp_mass_remaining=mass,
            ssp_alpha_fe=None,
            imf="chabrier",
            source="miles",
            nebular="bare",
        )
        ssp2 = SSPData(
            ssp_wave=wave,
            ssp_flux=flux,
            ssp_lg_age_gyr=age,
            ssp_lgmet=met,
            ssp_mass_remaining=mass,
            ssp_alpha_fe=None,
            imf="chabrier",
            source="miles",
            nebular="bare",
        )
        assert ssp1.cache_key() == ssp2.cache_key()


# ── Content Rows Move the Key ───────────────────────────────────────


class TestKeyContentSensitivity:
    """Verify content rows change the cache key."""

    def test_filter_transmission_values_change_key(self):
        """Different filter transmissions produce different keys."""
        fc1 = FilterCurve(
            wave=jnp.array([4000, 5000, 6000]), trans=jnp.array([0, 1, 0]), name="test"
        )
        fc2 = FilterCurve(
            wave=jnp.array([4000, 5000, 6000]), trans=jnp.array([0, 0.5, 0]), name="test"
        )
        assert fc1.cache_key() != fc2.cache_key()

    def test_line_names_change_key(self):
        """Different line names produce different keys."""
        lfd1 = LineFluxData(
            names=("Halpha", "Hbeta"),
            fluxes=jnp.array([1.0, 2.0]),
            errors=jnp.array([0.1, 0.2]),
            wavelengths=jnp.array([6564.61, 4862.68]),
        )
        lfd2 = LineFluxData(
            names=("Halpha", "Hgamma"),
            fluxes=jnp.array([1.0, 2.0]),
            errors=jnp.array([0.1, 0.2]),
            wavelengths=jnp.array([6564.61, 4340.47]),
        )
        assert lfd1.cache_key() != lfd2.cache_key()

    def test_spectroscopy_wave_obs_values_change_key(self):
        """Different wave_obs values produce different keys (content row)."""
        cov = jnp.eye(100) * 0.1
        spec1 = Spectroscopy(wave_obs=jnp.linspace(4000, 9000, 100), covariance=cov)
        spec2 = Spectroscopy(
            wave_obs=jnp.linspace(4000, 9000, 100) + 1.0,  # Different values
            covariance=cov,
        )
        assert spec1.cache_key() != spec2.cache_key()

    def test_spectroscopy_resolution_changes_key(self):
        """Different resolution values produce different keys (content row)."""
        wave = jnp.linspace(4000, 9000, 100)
        spec1 = Spectroscopy(wave_obs=wave, resolution=1000.0, covariance=jnp.eye(100) * 0.1)
        spec2 = Spectroscopy(
            wave_obs=wave,
            resolution=2000.0,  # Different resolution
            covariance=jnp.eye(100) * 0.1,
        )
        assert spec1.cache_key() != spec2.cache_key()

    def test_ssp_lgmet_values_change_key(self):
        """Different ssp_lgmet values produce different keys (content row)."""
        wave = jnp.array([1000.0, 2000.0])
        flux = jnp.zeros((2, 3, 2))
        age = jnp.array([6.0, 10.0, 10.1])
        mass = jnp.ones((2, 3))

        ssp1 = SSPData(
            ssp_wave=wave,
            ssp_flux=flux,
            ssp_lg_age_gyr=age,
            ssp_lgmet=jnp.array([-2.0, 0.0]),
            ssp_mass_remaining=mass,
            ssp_alpha_fe=None,
            imf="chabrier",
            source="miles",
            nebular="bare",
        )
        ssp2 = SSPData(
            ssp_wave=wave,
            ssp_flux=flux,
            ssp_lg_age_gyr=age,
            ssp_lgmet=jnp.array([-2.0, 0.1]),  # Different metallicity grid
            ssp_mass_remaining=mass,
            ssp_alpha_fe=None,
            imf="chabrier",
            source="miles",
            nebular="bare",
        )
        assert ssp1.cache_key() != ssp2.cache_key()

    def test_parameters_dust_law_changes_key(self):
        """Different dust_law values produce different keys (content row)."""
        params1 = parse_groups(
            dust_attenuation={"type": "two_component", "law": "calzetti"},
            redshift=Fixed(0.1),
        )
        params2 = parse_groups(
            dust_attenuation={"type": "two_component", "law": "smc"},
            redshift=Fixed(0.1),
        )
        assert params1.cache_key() != params2.cache_key()

    def test_parameters_mirror_changes_key(self):
        """Different mirror values produce different keys (content row)."""
        params1 = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "calzetti",
            },
            redshift=Fixed(0.1),
        )
        params2 = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "smc",
            },
            redshift=Fixed(0.1),
        )
        assert params1.cache_key() != params2.cache_key()


# ── Shape Rows Do Not Move the Key ──────────────────────────────────


class TestKeyShapeInsensitivity:
    """Verify shape-only rows don't change cache keys."""

    def test_flux_values_dont_change_key(self):
        """Different flux values (same shape) produce same key."""
        lfd1 = LineFluxData(
            names=("Halpha", "Hbeta"),
            fluxes=jnp.array([1.0, 2.0]),
            errors=jnp.array([0.1, 0.2]),
            wavelengths=jnp.array([6564.61, 4862.68]),
        )
        lfd2 = LineFluxData(
            names=("Halpha", "Hbeta"),
            fluxes=jnp.array([2.0, 3.0]),  # Different values, same shape
            errors=jnp.array([0.1, 0.2]),
            wavelengths=jnp.array([6564.61, 4862.68]),
        )
        assert lfd1.cache_key() == lfd2.cache_key()


# ── Fixed Parameters Do Not Move the Key ───────────────────────────────


class TestKeyFixedParametersInsensitivity:
    """Verify fixed parameter values don't change cache keys (are not structural)."""

    def test_fixed_parameter_value_change_equal(self):
        """Different fixed parameter values produce equal keys (value is not structural)."""
        params1 = parse_groups(
            redshift=Fixed(0.1),
        )
        params2 = parse_groups(
            redshift=Fixed(0.5),  # Different fixed value
        )
        assert params1.cache_key() == params2.cache_key()

    def test_free_parameter_prior_bounds_equal(self):
        """Different bounds on a free parameter produce equal keys (bounds are not structural)."""
        params1 = parse_groups(
            sfh={"type": "dpl", "all_params": FREE, "alpha": Uniform(0.5, 2.0)},
            redshift=Fixed(0.1),
        )
        params2 = parse_groups(
            sfh={
                "type": "dpl",
                "all_params": FREE,
                "alpha": Uniform(0.7, 1.8),
            },  # Different bounds
            redshift=Fixed(0.1),
        )
        assert params1.cache_key() == params2.cache_key()


# ── Additional Key Semantics ────────────────────────────────────────


class TestKeySemantics:
    """Semantic checks not covered by the policy-completeness, equivalence,
    or content/shape-sensitivity buckets above."""

    def test_spectroscopy_resolution_matrix_values_change_key(self):
        """Two resolution matrices with the same band shape but different
        values must not share a Spectroscopy cache key (#1163): the spectrum
        projector closes over the matrix content, not merely its shape.
        """
        wave = np.geomspace(4000.0, 7000.0, 100)
        rm1 = gaussian_resolution_bands(jnp.asarray(wave), 2000.0, n_diag=11)
        rm2 = gaussian_resolution_bands(jnp.asarray(wave), 800.0, n_diag=11)
        assert rm1.offsets.shape == rm2.offsets.shape
        assert rm1.data.shape == rm2.data.shape
        assert not jnp.array_equal(rm1.data, rm2.data)

        spec1 = Spectroscopy(wave_obs=jnp.asarray(wave), resolution_matrix=rm1)
        spec2 = Spectroscopy(wave_obs=jnp.asarray(wave), resolution_matrix=rm2)
        assert spec1.cache_key() != spec2.cache_key()

    def test_spectroscopy_resolution_matrix_equal_content_equal_key(self):
        """Two distinct BandedMatrix objects built from equal content give
        equal Spectroscopy cache keys."""
        wave = np.geomspace(4000.0, 7000.0, 100)
        rm_a = gaussian_resolution_bands(jnp.asarray(wave), 2000.0, n_diag=11)
        rm_b = gaussian_resolution_bands(jnp.asarray(wave), 2000.0, n_diag=11)
        assert rm_a is not rm_b

        spec_a = Spectroscopy(wave_obs=jnp.asarray(wave), resolution_matrix=rm_a)
        spec_b = Spectroscopy(wave_obs=jnp.asarray(wave), resolution_matrix=rm_b)
        assert spec_a.cache_key() == spec_b.cache_key()

    def test_parameters_free_vs_fixed_dust_tau_diff_moves_key(self):
        """Fixing one more parameter moves the key (the fixed/free name sets
        are in the cache_key() tail)."""
        params_fixed = Parameters(dust_tau_diff=Fixed(0.3))
        params_free = Parameters(dust_tau_diff=Uniform(0.0, 1.0))
        assert params_fixed.cache_key() != params_free.cache_key()

    def test_parameters_dust_slope_provenance_moves_key(self):
        """Naming a dust shape parameter explicitly moves the key through
        provenance, even when pinned at the registry's own default (#2231):
        the fixed value is identical in both cases, so the difference is in
        ``_user_provided`` / ``_flat_provenance``, not in the value."""
        params_named = Parameters(dust_slope=Fixed(-0.7))
        params_default = Parameters()
        assert params_named.fixed_value("dust_slope") == params_default.fixed_value("dust_slope")
        assert params_named.cache_key() != params_default.cache_key()


# ── Determinism Across Processes ────────────────────────────────────


class TestDeterminismAcrossProcesses:
    """Verify cache keys are stable across processes."""

    def test_photometry_deterministic(self):
        """Photometry.cache_key() is deterministic across processes.

        This is exactly what the process-salted builtin ``hash()`` that
        ``derive_key`` replaces would fail: ``PYTHONHASHSEED`` randomizes
        ``hash()`` per process, so two independent interpreters computing
        the "same" key via the builtin would disagree. ``derive_key`` uses
        ``stable_digest`` (BLAKE2b) instead, which must agree across
        processes -- so this spawns two separate subprocesses (not two
        calls inside this process) and compares their stdout.
        """
        src_dir = str(Path(__file__).resolve().parents[2] / "src")
        code = (
            "from tengri.observation.photometry_config import Photometry\n"
            "phot = Photometry.from_names(['sdss_g', 'sdss_r'])\n"
            "print(repr(phot.cache_key()))\n"
        )
        # Inherit the environment (PATH, HOME for the JAX cache dir) and pin the two
        # knobs that decide what the child imports and where it runs.
        env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": src_dir}
        result1 = subprocess.check_output([sys.executable, "-c", code], env=env).decode().strip()
        result2 = subprocess.check_output([sys.executable, "-c", code], env=env).decode().strip()
        assert result1 == result2

    def test_no_builtin_hash_used(self):
        """cache_key() methods do not call builtin hash()."""
        # Scan the source files for 'hash(' calls in cache_key implementations
        import inspect

        import tengri.observation.photometry as phot_mod

        source = inspect.getsource(phot_mod.FilterCurve.cache_key)
        assert "hash(" not in source, (
            "FilterCurve.cache_key uses hash() - use stable_digest instead"
        )
