# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for tengri.recipes.

Three layers, strongest first:

1. **Frozen free-parameter lists** — the exact ``spec.free_params`` each curated
   recipe exposes after ``parse_groups``. Any recipe edit that adds, drops, or
   renames a free parameter fails here and must update the frozen list
   deliberately.
2. **Structure** — one test per recipe asserting the dict grammar it returns
   (component types, fixed values, IGM/approx switches). These document recipe
   intent in a single place per recipe.
3. **Build + predict** — recipes actually build a ``SEDModel`` against the
   session SSP fixtures and produce finite, non-negative photometry from a
   prior draw. This is the layer that catches silently-broken recipes (the
   ``radio=True`` silent-drop class, NaN-producing defaults).
"""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest

from tengri import SEDModel, parse_groups, recipes

pytestmark = pytest.mark.contract

ALL_RECIPES = (
    "star_forming_photometry",
    "quiescent_z0",
    "agn_panchromatic",
    "composable_agn",
    "stochastic_sfh_jwst",
    "mock_recovery_minimal",
    "unified_agn",
    "high_z",
    "photoz",
    "dust_demo",
)

#: Recipes exercised end-to-end, split by the SSP their docstrings require.
#: Named here rather than inline in ``parametrize`` so the coverage test below
#: can read them, instead of a second hand-written copy drifting from the first.
BUILD_AND_PREDICT_WNE = {"mock_recovery_minimal", "photoz", "high_z", "dust_demo"}
BUILD_AND_PREDICT_CUE = {"star_forming_photometry", "quiescent_z0", "stochastic_sfh_jwst"}

#: Recipes deliberately not built here, with the reason on the record. An
#: exemption stated in one place can be reviewed; an omission from two
#: hand-written lists cannot be told from an oversight.
BUILD_EXEMPT = {
    "agn_panchromatic": "heavy AGN template libraries; covered by the structure "
    "and frozen-free-param layers",
    "composable_agn": "heavy AGN template libraries; covered by the structure "
    "and frozen-free-param layers",
    "unified_agn": "grid-gated on data/synthesizer_grids/, absent in CI",
}


def _recipe_functions() -> set[str]:
    """Every public recipe factory this module *owns*.

    Read off the live module, so the tests below cannot be satisfied by a
    stale pinned list agreeing with itself.

    Owned means defined here **or** exported here, and neither test alone is
    enough. ``__module__`` alone would miss a recipe moved into a submodule and
    re-exported; ``__all__`` alone would miss one defined here and forgotten
    from ``__all__``, which is precisely the oversight this census exists to
    catch. Their union admits both and still excludes a helper merely imported
    into the namespace.

    The predicate used to be ``__module__.startswith("tengri")``, which was
    looser than this docstring's own word "defined": #1690 imported
    ``tengri._completion.curated_dir`` into ``tengri.recipes`` and the census
    reported it as an unpinned recipe. The ``__dir__`` override installed on
    the same line hides it from ``dir()``, but this scan reads ``vars()``.
    """
    import inspect

    exported = set(getattr(recipes, "__all__", ()) or ())
    return {
        name
        for name, obj in vars(recipes).items()
        if not name.startswith("_")
        and inspect.isfunction(obj)
        and (getattr(obj, "__module__", None) == recipes.__name__ or name in exported)
    }


# Frozen contract: the exact free-parameter set of every recipe (sorted).
# unified_agn is excluded — it is grid-gated on data/synthesizer_grids/.
# Regenerate a line ONLY for a deliberate recipe change:
#   sorted(parse_groups(**recipes.<name>()).free_params)
RECIPE_FREE_PARAMS = {
    "star_forming_photometry": [
        "dust_tau_bc",
        "dust_tau_diff",
        "met_logzsol",
        "redshift",
        "sfh_dpl_age_gyr",
        "sfh_dpl_alpha",
        "sfh_dpl_beta",
        "sfh_dpl_log_total_mass",
        "sfh_dpl_tau_gyr",
    ],
    "quiescent_z0": [
        "dust_tau_bc",
        "dust_tau_diff",
        "met_logzsol",
        "sfh_dexp_log_total_mass",
        "sfh_dexp_tau_gyr",
    ],
    "agn_panchromatic": [
        "agn_a_spin",
        "agn_cos_inc",
        "agn_ir_frac",
        "agn_log_lbol",
        "agn_log_mbh",
        "agn_lum_ratio",
        "agn_nlr_cf",
        "agn_nlr_line_efficiency",
        "agn_oa_skirtor",
        "agn_p_skirtor",
        "agn_polar_T",
        "agn_polar_beta",
        "agn_polar_ebv",
        "agn_q_skirtor",
        "agn_tau_skirtor",
        "agn_torus_frac",
        "dust_tau_bc",
        "dust_tau_diff",
        "met_logzsol",
        "redshift",
        "sfh_dpl_age_gyr",
        "sfh_dpl_alpha",
        "sfh_dpl_beta",
        "sfh_dpl_log_total_mass",
        "sfh_dpl_tau_gyr",
    ],
    "composable_agn": [
        "agn_a_spin",
        "agn_blr_cf",
        "agn_blr_line_efficiency",
        "agn_cos_inc",
        "agn_fe2_strength",
        "agn_ir_frac",
        "agn_log_lbol",
        "agn_log_mbh",
        "agn_lum_ratio",
        "agn_nlr_cf",
        "agn_nlr_line_efficiency",
        "agn_oa_skirtor",
        "agn_p_skirtor",
        "agn_polar_T",
        "agn_polar_beta",
        "agn_polar_ebv",
        "agn_polar_oa",
        "agn_q_skirtor",
        "agn_tau_skirtor",
        "agn_torus_frac",
        "dust_tau_bc",
        "dust_tau_diff",
        "met_logzsol",
        "redshift",
        "sfh_dpl_age_gyr",
        "sfh_dpl_alpha",
        "sfh_dpl_beta",
        "sfh_dpl_log_total_mass",
        "sfh_dpl_tau_gyr",
    ],
    "stochastic_sfh_jwst": [
        "dust_tau_bc",
        "dust_tau_diff",
        "met_logzsol",
        "redshift",
        "sfh_dpl_age_gyr",
        "sfh_dpl_alpha",
        "sfh_dpl_beta",
        "sfh_dpl_log_total_mass",
        "sfh_dpl_tau_gyr",
        "sfh_field_psd_sigma",
        "sfh_field_psd_tau_myr",
    ],
    "mock_recovery_minimal": [
        "dust_tau_bc",
        "met_logzsol",
        "sfh_tsnorm_log_total_mass",
        "sfh_tsnorm_peak_lbt_gyr",
        "sfh_tsnorm_skew",
        "sfh_tsnorm_trunc",
        "sfh_tsnorm_width_gyr",
    ],
    "high_z": [
        "dust_tau_bc",
        "dust_tau_diff",
        "met_logzsol",
        "redshift",
        "sfh_tsnorm_log_total_mass",
        "sfh_tsnorm_peak_lbt_gyr",
        "sfh_tsnorm_skew",
        "sfh_tsnorm_trunc",
        "sfh_tsnorm_width_gyr",
    ],
    "photoz": [
        "dust_tau_bc",
        "dust_tau_diff",
        "met_logzsol",
        "redshift",
        "sfh_dpl_alpha",
        "sfh_dpl_beta",
        "sfh_dpl_log_total_mass",
        "sfh_dpl_tau_gyr",
    ],
    "dust_demo": [],
}


def _skip_if_no_synthesizer_grids():
    if not Path("data/synthesizer_grids").exists():
        pytest.skip("Synthesizer AGN grids not available at data/synthesizer_grids/")


class TestRecipesSurface:
    """Export surface + the frozen free-parameter contract."""

    def test_all_recipes_listed_in_module(self):
        """Every curated recipe is exported from tengri.recipes."""
        actual = {name for name in dir(recipes) if not name.startswith("_")}
        missing = set(ALL_RECIPES) - actual
        assert not missing, f"Missing recipes: {missing}"

    def test_no_recipe_exists_that_is_not_pinned(self):
        """The converse, which nothing checked.

        The assertion above is one-directional: it catches a pinned recipe that
        was deleted, never a recipe that was added and never pinned. Every other
        test in this file parametrizes over a pinned list, so an unpinned recipe
        would ship with no free-param contract, no structure test and no
        build-and-predict — silently, and green. Same asymmetry #1606 found
        between the two API-coverage directions.
        """
        actual = _recipe_functions()
        unpinned = actual - set(ALL_RECIPES)
        assert not unpinned, (
            f"recipes exist but are not in ALL_RECIPES: {sorted(unpinned)}. "
            f"Add them there and to RECIPE_FREE_PARAMS, then either give them a "
            f"build-and-predict case or list them in BUILD_EXEMPT with a reason."
        )

    def test_the_census_ignores_helpers_merely_imported_into_the_module(self):
        """A helper in the namespace is not a recipe.

        The predicate was ``__module__.startswith("tengri")``, looser than the
        word "defined" in its own docstring. #1690 imported
        ``tengri._completion.curated_dir`` into ``tengri.recipes`` and this
        census reported it as an unpinned recipe — turning a guard against
        untested recipes into a guard against importing anything.

        Narrowing it must not narrow what it catches, so both directions are
        pinned here and in
        :meth:`test_the_census_still_catches_a_recipe_defined_elsewhere`.
        """
        from tengri._completion import curated_dir

        assert curated_dir.__module__ != recipes.__name__
        assert "curated_dir" not in getattr(recipes, "__all__", ())
        assert "curated_dir" not in _recipe_functions()

    def test_the_census_still_catches_a_recipe_defined_elsewhere(self, monkeypatch):
        """Owned means defined here **or** exported here.

        ``__module__`` alone would miss a recipe moved to a submodule and
        re-exported — a refactor that must not silently drop it from the
        census.
        """

        def submodule_recipe():  # pragma: no cover - never called
            return {}

        submodule_recipe.__module__ = "tengri.recipes.agn"
        monkeypatch.setattr(recipes, "submodule_recipe", submodule_recipe, raising=False)
        monkeypatch.setattr(
            recipes, "__all__", [*recipes.__all__, "submodule_recipe"], raising=False
        )
        assert "submodule_recipe" in _recipe_functions()

    def test_every_recipe_is_build_tested_or_exempt_on_the_record(self):
        """A recipe must build, or say in one place why it is not built here.

        ``TestRecipesBuildAndPredict`` parametrizes two hand-written lists. The
        three AGN recipes are absent from both on purpose — their template
        libraries are heavy and they are covered by the structure and
        frozen-free-param layers instead — but that reasoning lived in a
        docstring, where a fourth omission would have looked identical to it.
        """
        covered = BUILD_AND_PREDICT_WNE | BUILD_AND_PREDICT_CUE
        accounted = covered | set(BUILD_EXEMPT)
        unaccounted = _recipe_functions() - accounted
        assert not unaccounted, (
            f"recipes with neither a build-and-predict case nor an exemption: "
            f"{sorted(unaccounted)}"
        )
        assert not (covered & set(BUILD_EXEMPT)), (
            "a recipe is both build-tested and exempt; the exemption is stale"
        )

    @pytest.mark.parametrize("name", sorted(RECIPE_FREE_PARAMS))
    def test_recipe_free_params_frozen(self, name):
        """Surface protected: the exact free-parameter set of each recipe.

        Subsumes the old returns-dict / builds-parameters / has-free-X /
        param-count-range tests — an exact frozen list is strictly stronger
        than membership or range checks.
        """
        spec = parse_groups(**getattr(recipes, name)())
        assert sorted(spec.free_params) == RECIPE_FREE_PARAMS[name]


class TestRecipeStructure:
    """One test per recipe pinning the dict grammar it returns."""

    def test_star_forming_photometry_structure(self):
        """DPL SFH + two-component Calzetti dust + Dale2014 emission + fixed Cue."""
        r = recipes.star_forming_photometry()
        assert r["sfh"]["type"] == "dpl"
        assert r["dust_attenuation"]["type"] == "two_component"
        assert r["dust_attenuation"]["law"] == "calzetti"
        assert r["dust_emission"]["type"] == "dale2014"
        assert r["neb"]["type"] == "cue"
        assert r.get("apply_igm", True) is True

    def test_quiescent_z0_structure(self):
        """Delayed-exponential SFH, redshift pinned to z=0.05."""
        r = recipes.quiescent_z0()
        assert r["sfh"]["type"] == "dexp"
        spec = parse_groups(**r)
        assert "redshift" in spec.fixed_params
        assert spec.get_distribution("redshift") == recipes.Fixed(0.05)

    def test_agn_panchromatic_structure(self):
        """DPL SFH plus AGN disc + torus sub-blocks."""
        r = recipes.agn_panchromatic()
        assert r["sfh"]["type"] == "dpl"
        assert "disc" in r["agn"]
        assert "torus" in r["agn"]

    def test_agn_panchromatic_includes_radio_xray(self):
        """Recipe declares radio and xray via the dict grammar AND the built
        spec actually carries their parameters.

        Regression: the recipe previously used the bool form (``radio=True``),
        which the group grammar silently skipped — so the panchromatic recipe
        shipped with no radio / X-ray at all while this test (which only
        checked ``recipe_dict['radio'] is True``) stayed green. Assert the
        real thing: radio / X-ray params exist after ``parse_groups``.
        """
        recipe_dict = recipes.agn_panchromatic()
        # #1980: the recipe declares radio composably (condon92's resolution).
        # Structural rather than whole-dict: the bug this guards is the bool
        # form (``radio=True``), which raises TypeError on these subscripts,
        # while whole-dict equality would also reject an additive key such as
        # a stated ``all_params`` disposition.
        assert recipe_dict["radio"]["sf"]["type"] == "bell2003"
        assert recipe_dict["radio"]["agn"]["type"] == "powerlaw"
        assert recipe_dict["xray"]["type"] == "simple"
        spec = parse_groups(**recipe_dict)
        allp = set(spec.free_params) | set(spec.get_fixed_values())
        assert any("radio" in k for k in allp), "radio params absent from built spec"
        assert any("xray" in k for k in allp), "xray params absent from built spec"

    def test_stochastic_sfh_jwst_structure(self):
        """DPL + field SFH composition with IGM on (high-z target)."""
        r = recipes.stochastic_sfh_jwst()
        assert r["sfh"]["type"] == ["dpl", "field"]
        assert r.get("apply_igm", True) is True

    def test_mock_recovery_minimal_structure(self):
        """Top-hat (tsnorm) SFH, Calzetti dust, nebular off, z pinned to 0.05."""
        r = recipes.mock_recovery_minimal()
        assert r["sfh"]["type"] == "tsnorm"
        assert r["dust_attenuation"]["law"] == "calzetti"
        assert r["neb"]["type"] == "none"
        spec = parse_groups(**r)
        assert "redshift" in spec.fixed_params
        assert spec.get_distribution("redshift") == recipes.Fixed(0.05)

    def test_unified_agn_structure(self):
        """Kubota & Done disc, simple torus, fixed delayed SFH, zero dust, z=0."""
        r = recipes.unified_agn()
        assert r["agn"]["disc"]["type"] == "kubota_done"
        assert r["agn"]["torus"]["type"] == "simple"
        assert r["sfh"]["type"] == "delayed"
        assert r["sfh"]["all_params"] == recipes.FIXED
        assert r["dust_attenuation"]["tau_bc"] == 0.0
        assert r["dust_attenuation"]["tau_diff"] == 0.0
        assert r["redshift"] == recipes.Fixed(0.0)

    def test_unified_agn_synthesizer_line_regions(self):
        """NLR and BLR come from Synthesizer spectra grids (grid-gated)."""
        _skip_if_no_synthesizer_grids()
        r = recipes.unified_agn()
        assert r["agn"]["nlr"]["type"] == "synthesizer_spectra"
        assert r["agn"]["blr"]["type"] == "synthesizer_spectra"

    def test_composable_agn_structure(self):
        """All six AGN slots on committed data, CIGALE-joint normalization."""
        r = recipes.composable_agn()
        agn = r["agn"]
        assert agn["disc"]["type"] == "multicolor"
        assert agn["nlr"]["type"] == "analytic"
        assert agn["blr"]["type"] == "analytic"
        assert agn["feii"]["type"] == "boroson_green"
        assert agn["torus"]["type"] == "skirtor"
        assert agn["atten"]["type"] == "polar_dust"
        assert agn["norm"] == "cigale_joint"
        fracagn = agn["ir_frac"]
        assert hasattr(fracagn, "lo") and fracagn.lo > 0 and fracagn.hi < 1.0
        assert r["sfh"]["type"] == "dpl"

    def test_composable_agn_includes_radio_xray(self):
        """Recipe declares radio/xray via the dict grammar and the built spec
        carries their params (see agn_panchromatic counterpart for context)."""
        recipe_dict = recipes.composable_agn()
        # #1980: the recipe declares radio composably (condon92's resolution).
        # Structural rather than whole-dict: the bug this guards is the bool
        # form (``radio=True``), which raises TypeError on these subscripts,
        # while whole-dict equality would also reject an additive key such as
        # a stated ``all_params`` disposition.
        assert recipe_dict["radio"]["sf"]["type"] == "bell2003"
        assert recipe_dict["radio"]["agn"]["type"] == "powerlaw"
        assert recipe_dict["xray"]["type"] == "simple"
        spec = parse_groups(**recipe_dict)
        allp = set(spec.free_params) | set(spec.get_fixed_values())
        assert any("radio" in k for k in allp), "radio params absent from built spec"
        assert any("xray" in k for k in allp), "xray params absent from built spec"

    def test_high_z_structure(self):
        """tsnorm SFH, IGM mandatory, no IR emission block, exact wave path.

        No WavePrecomp: the dust Taylor projection biases rest-UV bands,
        which is exactly the regime this recipe samples (#617/#731).
        """
        r = recipes.high_z()
        assert r["sfh"]["type"] == "tsnorm"
        assert r["apply_igm"] is True
        assert "dust_emission" not in r
        assert "approx" not in r

    def test_photoz_structure(self):
        """DPL SFH, nebular off, IGM on, exact wave path (#617/#731)."""
        r = recipes.photoz()
        assert r["sfh"]["type"] == "dpl"
        assert r["neb"]["type"] == "none"
        assert r["apply_igm"] is True
        assert "approx" not in r


class TestGateGroupBoolRejected:
    """Additive gate groups (radio / xray / shock) are declared like every
    other component — a dict selecting the model. The bool form must raise
    (it used to be silently skipped, absenting the component)."""

    @pytest.mark.parametrize("group", ["radio", "xray", "shock"])
    def test_bool_gate_group_raises_actionable_error(self, group):
        with pytest.raises(ValueError, match=r"type"):
            parse_groups(**{group: True, "sfh": {"type": "dpl"}})

    @pytest.mark.parametrize(
        "group,decl,extra",
        [
            ("radio", {"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}}, {}),
            ("xray", {"type": "simple"}, {}),
            ("shock", {"type": "mappings"}, {"neb": {"type": "cue"}}),
        ],
    )
    def test_dict_gate_group_activates_params(self, group, decl, extra):
        """The dict form activates the component — its params appear in the
        built spec (guards the silent-drop regression from the positive side)."""
        spec = parse_groups(sfh={"type": "dpl"}, **{group: decl}, **extra)
        allp = set(spec.free_params) | set(spec.get_fixed_values())
        assert any(group in k for k in allp), (
            f"{group}={decl} produced no {group} params — silently absent"
        )


# Recipes with a free redshift get it pinned after the prior draw so the
# synthetic top-hat bands (3500–9000 Å) stay redward of the observed Lyman
# break — otherwise a high-z draw legitimately zeroes every band.
_REDSHIFT_PIN = {
    "star_forming_photometry": 0.5,
    "stochastic_sfh_jwst": 4.0,
    "high_z": 4.0,
    "photoz": 1.0,
}


def _build_and_predict(name, ssp_data, obs):
    model = SEDModel.build(ssp_data=ssp_data, observation=obs, **getattr(recipes, name)())
    params = model.spec.sample(jax.random.PRNGKey(0))
    if name in _REDSHIFT_PIN:
        params = {**params, "redshift": jnp.asarray(_REDSHIFT_PIN[name])}
    flux = model.predict_photometry(params)
    n_bands = len(obs.photometry.filters)
    assert flux.shape == (n_bands,)
    chex.assert_tree_all_finite(flux)
    assert jnp.all(flux >= 0.0), f"{name}: negative photometry {flux}"
    assert jnp.max(flux) > 0.0, f"{name}: all-zero photometry"


class TestRecipesBuildAndPredict:
    """Recipes build a real SEDModel and predict physical photometry.

    Surface protected: the end-to-end ``SEDModel.build(**recipe)`` path —
    component resolution, prior sampling, and a forward prediction. Catches
    recipes that parse but cannot build, and NaN-producing defaults.
    """

    @pytest.mark.parametrize("name", sorted(BUILD_AND_PREDICT_WNE))
    def test_wne_compatible_recipes_build_and_predict(
        self, name, ssp_data_wne, synthetic_tophat_obs
    ):
        """Recipes whose SSP requirement is wNE or 'any' (per their docstrings)."""
        _build_and_predict(name, ssp_data_wne, synthetic_tophat_obs)

    @pytest.mark.parametrize("name", sorted(BUILD_AND_PREDICT_CUE))
    def test_cue_recipes_build_and_predict(self, name, ssp_data_bc03, synthetic_tophat_obs):
        """Cue-backed recipes need a bare-stellar SSP (skips when bc03 grid absent).

        The AGN recipes (agn_panchromatic, composable_agn, unified_agn) are
        covered by the frozen-free-params and structure layers; their heavy
        template loads live in the dedicated AGN component tests.
        """
        _build_and_predict(name, ssp_data_bc03, synthetic_tophat_obs)
