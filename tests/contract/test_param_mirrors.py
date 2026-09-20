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
8. A mirrored spec's sample() reaches every predict surface (#2296): a
   mirror target is internally Fixed(0.0) (item 2 above), so it is a member
   of ``spec.fixed_params`` -- ``refuse_fixed_overrides`` must exempt
   ``spec.mirrors`` targets specifically, or every mirrored model's own
   ``sample()`` output would be refused by the very refusal #2296 added.
"""

import jax
import pytest

pytestmark = pytest.mark.contract

from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform


@pytest.fixture
def mirrored_spec():
    # power_law (bc) reads dust_slope and kriek_conroy (diff) reads
    # dust_delta, so both names are live and flat accepts them. The mirror
    # mechanism, not physics, is the subject here.
    return Parameters(
        mean_sfh_type="tsnorm",
        dust_law_bc="power_law",
        dust_law_diff="kriek_conroy",
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
            dust_law_bc="power_law",
            dust_law_diff="kriek_conroy",
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
                dust_law_bc="power_law",
                dust_law_diff="kriek_conroy",
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
            dust_law_bc="power_law",
            dust_law_diff="kriek_conroy",
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


class TestMirroredSampleReachesEveryPredictSurface:
    """A mirrored spec's own sample() must not be refused by #2296.

    ``dust_slope`` (the target) is internally ``Fixed(0.0)`` -- a
    placeholder, per ``TestMirrorDetection.test_mirrored_param_becomes_fixed``
    -- so it is a member of ``spec.fixed_params``. #2296's refusal is a
    presence check on ``spec.fixed_params``, so without an explicit
    exemption for mirror targets, ``sample()``'s own output (which legally
    carries the target at its tied value, per ``TestSampleIntegration``
    above) would trip the very guard #2296 added at every predict surface --
    a spec cannot fit its own reflection.
    """

    @pytest.fixture(scope="class")
    def mirrored_model(self, synthetic_ssp_wide, synthetic_tophat_obs):
        from tengri.forward.sed_model import SEDModel

        spec = Parameters(
            mean_sfh_type="tsnorm",
            dust_law_bc="power_law",
            dust_law_diff="kriek_conroy",
            dust_delta=Uniform(-1.0, 0.5),
            dust_slope="dust_delta",
            redshift=Fixed(0.1),
        )
        return SEDModel(spec, synthetic_ssp_wide, observation=synthetic_tophat_obs)

    def test_sample_output_ties_the_mirror(self, mirrored_model):
        p = dict(mirrored_model.spec.sample(jax.random.PRNGKey(0)))
        assert p["dust_slope"] == p["dust_delta"]

    def test_predict_photometry_accepts_the_mirrored_sample(self, mirrored_model):
        p = dict(mirrored_model.spec.sample(jax.random.PRNGKey(1)))
        flux = mirrored_model.predict_photometry(p)
        assert flux.shape[0] > 0

    def test_predict_accepts_the_mirrored_sample(self, mirrored_model):
        p = dict(mirrored_model.spec.sample(jax.random.PRNGKey(2)))
        pred = mirrored_model.predict(p)
        assert pred.photometry().shape[0] > 0

    def test_predict_properties_accepts_the_mirrored_sample(self, mirrored_model):
        p = dict(mirrored_model.spec.sample(jax.random.PRNGKey(3)))
        props = mirrored_model.predict_properties(p, names=("stellar_mass",))
        assert "stellar_mass" in props

    def test_merge_fixed_params_resolves_the_mirror_from_a_free_only_dict(self, mirrored_model):
        """A dict that omits the mirror target entirely still resolves correctly.

        ``merge_fixed_params`` (used by, e.g., direct component-level calls
        that bypass ``predict_state``'s own merge boundary) must not leave
        the mirror target at its internal Fixed(0.0) placeholder.
        """
        from tengri.parameters.resolve import merge_fixed_params

        spec = mirrored_model.spec
        free_only = {"dust_delta": 0.37}
        merged = merge_fixed_params(spec, free_only)
        assert merged["dust_slope"] == free_only["dust_delta"]
