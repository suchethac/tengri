# SPDX-License-Identifier: BSD-3-Clause
"""Invariant tests for Parameters (Parameters) construction and contracts.

Bug classes covered:
- Parameter name consistency: wrong keys returned by sample(), fixed params
  leaking into free_params, short-name expansion producing wrong full names.
- Ordering stability: free_params must be sorted for stable cache keys and
  reproducible inference.
- Immutability: merge_observation_params() must not mutate the original spec.

No SSP data required. All tests use pure Parameters construction.

Key invariant (corrected from naive expectation):
    sample() returns ALL params (fixed + free + sfh_field_xi if stochastic),
    NOT just free params. free_params is the list of non-fixed params only.
"""

from __future__ import annotations

import jax
import pytest

pytestmark = pytest.mark.contract

from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform
from tengri.parameters.translate import (
    _SFH_SHORT_NAMES,
    _UNIVERSAL_SHORT_NAMES,
    resolve_short_names,
)

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def dpl_spec() -> Parameters:
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 5.0),
        sfh_dpl_log_total_mass=Uniform(-1.0, 3.0),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_bc=Fixed(0.1),
        dust_tau_diff=Fixed(0.3),
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="module")
def db_spec() -> Parameters:
    return Parameters(
        mean_sfh_type="dense_basis",
        sfh_db_log_total_mass=Uniform(8.0, 12.0),
        sfh_db_log_sfr_inst=Uniform(-3.0, 3.0),
        sfh_db_tx_frac_0=Uniform(0.05, 0.95),
        sfh_db_tx_frac_1=Uniform(0.05, 0.95),
        sfh_db_tx_frac_2=Uniform(0.05, 0.95),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="module")
def stochastic_spec() -> Parameters:
    return Parameters(
        mean_sfh_type=["dpl", "field"],
        n_grid=16,
        sfh_dpl_alpha=Uniform(0.5, 5.0),
        sfh_dpl_log_total_mass=Uniform(-1.0, 3.0),
        sfh_field_psd_sigma=Uniform(0.1, 5.0),
        sfh_field_psd_tau_myr=Uniform(10.0, 500.0),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
    )


# ── sample() returns ALL params (fixed + free) ────────────────────


class TestSampleKeys:
    """sample() must return exactly all_params (fixed + free), plus sfh_field_xi if stochastic."""

    def test_sample_keys_equal_all_params(self, dpl_spec: Parameters) -> None:
        key = jax.random.PRNGKey(0)
        sample = dpl_spec.sample(key)
        assert set(sample.keys()) == set(dpl_spec.all_params), (
            f"sample() keys differ from all_params.\n"
            f"  In sample but not all_params: {set(sample.keys()) - set(dpl_spec.all_params)}\n"
            f"  In all_params but not sample: {set(dpl_spec.all_params) - set(sample.keys())}"
        )

    def test_sample_includes_fixed_params(self, dpl_spec: Parameters) -> None:
        """Fixed params must appear in sample() output at their fixed value."""
        key = jax.random.PRNGKey(1)
        sample = dpl_spec.sample(key)
        for name in dpl_spec.fixed_params:
            assert name in sample, f"Fixed param {name!r} missing from sample()"

    def test_sample_fixed_params_at_fixed_value(self, dpl_spec: Parameters) -> None:
        """Fixed params in sample() must equal the value they were Fixed to."""
        key = jax.random.PRNGKey(2)
        sample = dpl_spec.sample(key)
        fixed_vals = dpl_spec.get_fixed_values()
        for name, expected in fixed_vals.items():
            actual = float(sample[name])
            assert abs(actual - expected) < 1e-7, (
                f"Fixed param {name!r}: sample()={actual}, expected={expected}"
            )

    def test_sample_keys_stochastic_has_xi(self, stochastic_spec: Parameters) -> None:
        """Stochastic spec must also produce sfh_field_xi in sample()."""
        key = jax.random.PRNGKey(3)
        sample = stochastic_spec.sample(key)
        assert "sfh_field_xi" in sample, "sfh_field_xi missing from stochastic spec sample()"

    def test_sample_xi_shape_matches_n_grid(self, stochastic_spec: Parameters) -> None:
        key = jax.random.PRNGKey(4)
        sample = stochastic_spec.sample(key)
        xi = sample["sfh_field_xi"]
        assert xi.shape == (16,), f"sfh_field_xi shape {xi.shape} does not match n_grid=16"

    def test_sample_non_stochastic_has_no_xi(self, dpl_spec: Parameters) -> None:
        key = jax.random.PRNGKey(5)
        sample = dpl_spec.sample(key)
        assert "sfh_field_xi" not in sample, (
            "Non-stochastic spec sample() unexpectedly contains sfh_field_xi"
        )


# ── free_params and fixed_params are disjoint and partition all_params


class TestParamPartitioning:
    """free_params and fixed_params must partition all_params exactly."""

    @pytest.mark.parametrize("spec_name", ["dpl_spec", "db_spec", "stochastic_spec"])
    def test_free_and_fixed_are_disjoint(self, request, spec_name: str) -> None:
        spec: Parameters = request.getfixturevalue(spec_name)
        overlap = set(spec.free_params) & set(spec.fixed_params)
        assert overlap == set(), f"Params appear in both free and fixed: {overlap}"

    @pytest.mark.parametrize("spec_name", ["dpl_spec", "db_spec", "stochastic_spec"])
    def test_free_and_fixed_cover_all_params(self, request, spec_name: str) -> None:
        spec: Parameters = request.getfixturevalue(spec_name)
        reconstructed = set(spec.free_params) | set(spec.fixed_params)
        assert reconstructed == set(spec.all_params), (
            f"free_params ∪ fixed_params ≠ all_params.\n"
            f"  Missing: {set(spec.all_params) - reconstructed}\n"
            f"  Extra: {reconstructed - set(spec.all_params)}"
        )

    @pytest.mark.parametrize("spec_name", ["dpl_spec", "db_spec", "stochastic_spec"])
    def test_free_params_sorted(self, request, spec_name: str) -> None:
        spec: Parameters = request.getfixturevalue(spec_name)
        assert spec.free_params == sorted(spec.free_params), (
            f"free_params is not sorted: {spec.free_params}"
        )

    @pytest.mark.parametrize("spec_name", ["dpl_spec", "db_spec", "stochastic_spec"])
    def test_no_duplicate_free_params(self, request, spec_name: str) -> None:
        spec: Parameters = request.getfixturevalue(spec_name)
        assert len(spec.free_params) == len(set(spec.free_params)), (
            f"free_params contains duplicates: {spec.free_params}"
        )

    def test_fixed_params_not_in_free_params(self, dpl_spec: Parameters) -> None:
        for name in dpl_spec.fixed_params:
            assert name not in dpl_spec.free_params, (
                f"Fixed param {name!r} unexpectedly found in free_params"
            )


# ── merge_observation_params() immutability ───────────────────────


class TestMergeObservationParams:
    """merge_observation_params() must return a new spec without mutating the original."""

    def test_merge_does_not_mutate_original(self, dpl_spec: Parameters) -> None:
        original_free = list(dpl_spec.free_params)
        original_all = list(dpl_spec.all_params)
        _ = dpl_spec.merge_observation_params(
            eline_ha_amp=Uniform(0.0, 1.0),
            eline_hb_amp=Uniform(0.0, 0.5),
        )
        assert list(dpl_spec.free_params) == original_free, (
            "merge_observation_params() mutated original free_params"
        )
        assert list(dpl_spec.all_params) == original_all, (
            "merge_observation_params() mutated original all_params"
        )

    def test_merge_adds_to_free_params(self, dpl_spec: Parameters) -> None:
        extra = {"eline_ha_amp": Uniform(0.0, 1.0), "eline_hb_amp": Uniform(0.0, 0.5)}
        merged = dpl_spec.merge_observation_params(**extra)
        for name in extra:
            assert name in merged.free_params, (
                f"Merged spec does not include new param {name!r} in free_params"
            )

    def test_merge_original_params_still_present(self, dpl_spec: Parameters) -> None:
        merged = dpl_spec.merge_observation_params(eline_ha_amp=Uniform(0.0, 1.0))
        for name in dpl_spec.all_params:
            assert name in merged.all_params, f"Original param {name!r} lost after merge"


# ── resolve_short_names — SFH-specific expansion ──────────────────


class TestResolveShortNamesDPL:
    def test_alpha_expands_to_sfh_dpl_alpha(self) -> None:
        result = resolve_short_names("dpl", {"alpha": Uniform(0.5, 5.0)})
        assert "sfh_dpl_alpha" in result
        assert "alpha" not in result

    def test_log_total_mass_expands_for_dpl(self) -> None:
        result = resolve_short_names("dpl", {"log_total_mass": Uniform(-1.0, 3.0)})
        assert "sfh_dpl_log_total_mass" in result
        assert "log_total_mass" not in result

    def test_all_dpl_short_names_expand(self) -> None:
        short_map = _SFH_SHORT_NAMES["dpl"]
        priors = {k: Uniform(0.0, 1.0) for k in short_map}
        result = resolve_short_names("dpl", priors)
        for short, full in short_map.items():
            assert full in result, f"Short name {short!r} → {full!r} not in result"
            assert short not in result, f"Short name {short!r} still in result (not expanded)"


class TestResolveShortNamesDenseBasis:
    def test_log_total_mass_expands(self) -> None:
        result = resolve_short_names("dense_basis", {"log_total_mass": Uniform(8.0, 12.0)})
        assert "sfh_db_log_total_mass" in result
        assert "log_total_mass" not in result

    def test_db_alias_works_same_as_dense_basis(self) -> None:
        priors = {"tx_frac_0": Uniform(0.05, 0.95)}
        result_full = resolve_short_names("dense_basis", priors)
        result_short = resolve_short_names("db", priors)
        assert result_full == result_short


class TestResolveShortNamesField:
    def test_psd_sigma_expands_for_field(self) -> None:
        result = resolve_short_names("field", {"psd_sigma": Uniform(0.1, 5.0)})
        assert "sfh_field_psd_sigma" in result
        assert "psd_sigma" not in result

    def test_field_expands_in_combined_sfh_list(self) -> None:
        result = resolve_short_names(["dpl", "field"], {"psd_tau_myr": Uniform(10.0, 500.0)})
        assert "sfh_field_psd_tau_myr" in result


class TestResolveShortNamesUniversal:
    @pytest.mark.parametrize("short,full", list(_UNIVERSAL_SHORT_NAMES.items()))
    def test_universal_short_name_expands(self, short: str, full: str) -> None:
        """Every universal short name must expand correctly for any SFH type."""
        result = resolve_short_names("dpl", {short: Uniform(0.0, 1.0)})
        assert full in result, f"Universal short {short!r} → {full!r} not in result"
        assert short not in result, f"Universal short {short!r} not removed from result"

    @pytest.mark.parametrize("sfh_type", ["dpl", "dense_basis", "tsnorm", "delayed"])
    def test_logzsol_expands_for_all_sfh_types(self, sfh_type: str) -> None:
        result = resolve_short_names(sfh_type, {"logzsol": Uniform(-2.0, 0.5)})
        assert "met_logzsol" in result
        assert "logzsol" not in result


class TestResolveShortNamesFullNamesPassThrough:
    def test_full_name_not_mangled(self) -> None:
        """Full parameter names must pass through resolve_short_names unchanged."""
        result = resolve_short_names("dpl", {"sfh_dpl_alpha": Uniform(0.5, 5.0)})
        assert "sfh_dpl_alpha" in result

    def test_unknown_key_passes_through(self) -> None:
        """Unknown keys that are not short or full names pass through unchanged."""
        result = resolve_short_names("dpl", {"custom_param": Uniform(0.0, 1.0)})
        assert "custom_param" in result
