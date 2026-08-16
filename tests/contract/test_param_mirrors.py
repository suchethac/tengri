# SPDX-License-Identifier: BSD-3-Clause
"""Tests for parameter mirroring / tying in Parameters.

Verifies:
1. String-valued kwargs are detected as mirrors
2. Mirrored params become Fixed(0.0) internally
3. resolve_mirrors() copies source → target
4. Chained mirrors are rejected
5. sample() returns mirrored values
6. summary() displays mirrors correctly
7. Mirror targets excluded from free_params
"""

import jax
import pytest

pytestmark = pytest.mark.contract

from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform


@pytest.fixture
def mirrored_spec():
    return Parameters(
        mean_sfh_type="tsnorm",
        dust_delta=Uniform(-1.0, 0.5),
        dust_slope="dust_delta",
    )


class TestMirrorDetection:
    def test_string_kwarg_creates_mirror(self, mirrored_spec):
        assert mirrored_spec.mirrors == {"dust_slope": "dust_delta"}

    def test_mirrored_param_becomes_fixed(self, mirrored_spec):
        dist = mirrored_spec.get_distribution("dust_slope")
        assert isinstance(dist, Fixed)

    def test_mirrored_param_not_in_free_params(self, mirrored_spec):
        assert "dust_slope" not in mirrored_spec.free_params
        assert "dust_delta" in mirrored_spec.free_params

    def test_no_mirrors_by_default(self):
        spec = Parameters(mean_sfh_type="tsnorm")
        assert spec.mirrors == {}

    def test_multiple_mirrors(self):
        spec = Parameters(
            mean_sfh_type="tsnorm",
            met_logzsol=Uniform(-2.0, 0.2),
            dust_delta=Uniform(-1.0, 0.5),
            dust_slope="dust_delta",
            dust_tau_diff="dust_tau_bc",
        )
        assert len(spec.mirrors) == 2
        assert spec.mirrors["dust_slope"] == "dust_delta"
        assert spec.mirrors["dust_tau_diff"] == "dust_tau_bc"


class TestChainValidation:
    def test_chained_mirrors_raise(self):
        with pytest.raises(ValueError, match="Chained mirror"):
            Parameters(
                mean_sfh_type="tsnorm",
                dust_delta=Uniform(-1.0, 0.5),
                dust_slope="dust_delta",
                dust_bump_strength="dust_slope",
            )


class TestResolveMethod:
    def test_resolve_copies_source_to_target(self, mirrored_spec):
        params = {"dust_delta": -0.3, "dust_slope": 0.0}
        resolved = mirrored_spec.resolve_mirrors(params)
        assert resolved["dust_slope"] == -0.3
        assert resolved["dust_delta"] == -0.3

    def test_resolve_returns_new_dict(self, mirrored_spec):
        params = {"dust_delta": -0.3, "dust_slope": 0.0}
        resolved = mirrored_spec.resolve_mirrors(params)
        assert resolved is not params

    def test_resolve_noop_without_mirrors(self):
        spec = Parameters(mean_sfh_type="tsnorm")
        params = {"dust_delta": -0.3}
        resolved = spec.resolve_mirrors(params)
        assert resolved is params

    def test_resolve_preserves_other_params(self, mirrored_spec):
        params = {
            "dust_delta": -0.3,
            "dust_slope": 0.0,
            "redshift": 1.0,
        }
        resolved = mirrored_spec.resolve_mirrors(params)
        assert resolved["redshift"] == 1.0


class TestSampleIntegration:
    def test_sample_resolves_mirrors(self, mirrored_spec):
        key = jax.random.PRNGKey(42)
        sample = mirrored_spec.sample(key)
        assert sample["dust_slope"] == sample["dust_delta"]

    def test_sample_mirror_varies_with_source(self, mirrored_spec):
        s1 = mirrored_spec.sample(jax.random.PRNGKey(0))
        s2 = mirrored_spec.sample(jax.random.PRNGKey(99))
        assert s1["dust_delta"] != s2["dust_delta"]
        assert s1["dust_slope"] == s1["dust_delta"]
        assert s2["dust_slope"] == s2["dust_delta"]


class TestSummaryDisplay:
    def test_summary_shows_mirror_section(self, mirrored_spec):
        text = mirrored_spec.summary_str()
        assert "Mirror(dust_delta)" in text
        assert "dust_slope" in text

    def test_summary_shows_mirrored_count(self, mirrored_spec):
        text = mirrored_spec.summary_str()
        assert "1 mirrored" in text

    def test_summary_mirror_not_in_fixed(self):
        spec = Parameters(
            mean_sfh_type="tsnorm",
            dust_delta=Uniform(-1.0, 0.5),
            dust_slope="dust_delta",
            redshift=Fixed(1.0),
        )
        text = spec.summary_str()
        lines = text.split("\n")
        fixed_lines = [l for l in lines if "Fixed" in l]
        mirror_lines = [l for l in lines if "Mirror" in l]
        fixed_names = [l.split()[0].strip() for l in fixed_lines]
        assert "dust_slope" not in fixed_names
        assert any("dust_slope" in l for l in mirror_lines)
