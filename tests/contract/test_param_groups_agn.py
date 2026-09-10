# SPDX-License-Identifier: BSD-3-Clause
"""Tests for AGN composable grammar in parse_groups().

Tests the nested-dict AGN specification via parse_groups(), including:
- Automatic agn_model='composable' activation
- Per-block type selection (disc, torus, lines, feii, atten)
- Shared agn-* parameters
- Sub-block parameter routing and short-name resolution
- Wildcard semantics at agn and sub-block levels
- Provenance tagging
"""

import warnings

import pytest

pytestmark = pytest.mark.contract
from tengri.components.agn.blocks import RecipeWarning
from tengri.parameters import DEFAULT, FREE, Fixed, Uniform
from tengri.parameters.groups import parse_groups
from tengri.parameters.parameters import Parameters


class TestAGNBasics:
    """Test basic AGN composable grammar activation and structure."""

    def test_agn_sets_composable_model(self):
        """Presence of agn group auto-activates agn_model='composable'."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={"disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)}},
            redshift=Fixed(0.1),
        )
        assert isinstance(params, Parameters)
        assert params.agn_model == "composable"

    def test_agn_omitted_block_defaults_to_none(self):
        """Omitted blocks default to 'none'."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={"disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)}},
            redshift=Fixed(0.1),
        )
        assert params.agn_disc_block == "powerlaw"
        assert params.agn_torus_block == "none"
        assert params.agn_nlr_block == "none"
        assert params.agn_blr_block == "none"
        assert params.agn_feii_block == "none"
        assert params.agn_attenuation_block == "none"

    def test_agn_all_blocks_specified(self):
        """User can specify all 6 blocks."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
                "torus": {"type": "simple", "all_params": Fixed(DEFAULT)},
                "nlr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
                "blr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
                "feii": {"type": "none", "all_params": Fixed(DEFAULT)},
                "atten": {"type": "none", "all_params": Fixed(DEFAULT)},
            },
            redshift=Fixed(0.1),
        )
        assert params.agn_disc_block == "powerlaw"
        assert params.agn_torus_block == "simple"
        assert params.agn_nlr_block == "analytic"
        assert params.agn_blr_block == "analytic"
        assert params.agn_feii_block == "none"
        assert params.agn_attenuation_block == "none"


class TestAGNParameterRouting:
    """Test parameter extraction and routing to correct sub-block."""

    def test_shared_agn_params_routed_correctly(self):
        """Shared agn_* params (frac, log_lbol) recognized at agn-level."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
                "all_params": FREE,  # Free all shared agn params
                "log_lbol": Uniform(9.42, 13.42),  # Override with explicit prior
            },
            redshift=Fixed(0.1),
        )
        # agn_log_lbol should have the user-provided Uniform prior
        dist = params.get_distribution("agn_log_lbol")
        assert dist.bounds == (9.42, 13.42)

    def test_sub_block_param_routing_torus_skirtor(self):
        """agn.torus 'tau_skirtor' override routes to agn_tau_skirtor."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
                "torus": {"type": "skirtor", "all_params": FREE, "tau_skirtor": Uniform(3, 11)},
            },
            redshift=Fixed(0.1),
        )
        # The agn_tau_skirtor param should be free with the user's Uniform prior
        assert "agn_tau_skirtor" in params.free_params
        dist = params.get_distribution("agn_tau_skirtor")
        assert dist.bounds == (3.0, 11.0)

    def test_sub_block_param_routing_atten_polar(self):
        """agn.atten 'polar_ebv' routes to agn_polar_ebv."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "atten": {
                    "type": "polar_dust",
                    "all_params": Fixed(DEFAULT),
                    "polar_ebv": Fixed(0.3),
                },
            },
            redshift=Fixed(0.1),
        )
        assert "agn_polar_ebv" in params.fixed_params
        assert params.get_distribution("agn_polar_ebv").value == 0.3

    def test_construction_warns_on_polar_dust_zero_ebv(self):
        """polar_dust attenuation with a Fixed E(B-V)=0 is a silent no-op; the
        RecipeWarning must fire at construction, not only when
        validate_block_recipe is called by hand (#890)."""
        with pytest.warns(RecipeWarning, match="agn_polar_ebv=0"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={
                    "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                    "atten": {
                        "type": "polar_dust",
                        "all_params": Fixed(DEFAULT),
                        "polar_ebv": Fixed(0.0),
                    },
                },
                redshift=Fixed(0.1),
            )

    def test_construction_silent_when_ebv_free(self):
        """A free (fitted) E(B-V) is intentional — no polar no-op warning (#890)."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RecipeWarning)
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={
                    "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                    "atten": {
                        "type": "polar_dust",
                        "all_params": Fixed(DEFAULT),
                        "polar_ebv": Uniform(0.0, 1.0),
                    },
                },
                redshift=Fixed(0.1),
            )

    def test_construction_silent_when_ebv_fixed_nonzero(self):
        """A Fixed nonzero E(B-V) applies extinction — no no-op warning (#890)."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RecipeWarning)
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={
                    "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                    "atten": {
                        "type": "polar_dust",
                        "all_params": Fixed(DEFAULT),
                        "polar_ebv": Fixed(0.3),
                    },
                },
                redshift=Fixed(0.1),
            )

    def test_agn_wildcard_at_agn_level_frees_shared(self):
        """agn={'all_params': FREE, 'disc': {...}} frees exactly the TRULY
        shared params (agn_log_lbol/agn_lum_ratio) -- the ones NOT owned by
        any specific composable sub-block in ``_AGN_PARTITION``.

        Before Task 16 (item 1), ``agn_a_spin``/``agn_log_mbh`` were
        misclassified as shared "agn" too, so the top-level wildcard froze
        them even when the disc sub-block explicitly stated its OWN
        disposition as ``Fixed(DEFAULT)`` (as here) -- the disc's explicit
        "keep me fixed" was silently overridden for its own physics
        parameters. Now that they are correctly ``agn.disc``-owned, the
        disc sub-block's OWN explicit ``Fixed(DEFAULT)`` governs them, and
        the top-level wildcard (whose scope is still
        ``agn_active_param_set``, unioned across active blocks) can only
        reach the names that stay outside every sub-block's ownership.
        """
        from tengri.components.agn.blocks._consumes import agn_active_param_set

        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "all_params": FREE,
                "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
            },
            redshift=Fixed(0.1),
        )
        # agn_active_param_set (unchanged) still reports agn_a_spin/
        # agn_log_mbh as consumed by multicolor -- that table describes
        # what the ACTIVE BLOCK reads, not who is allowed to free it via
        # which wildcard. The top-level wildcard's ACTUAL reach is now
        # narrower: it is scoped to agn_active_param_set for names owned by
        # the shared "agn" group, and agn_a_spin/agn_log_mbh moved out of
        # that group (see _AGN_PARTITION).
        consumed = agn_active_param_set(
            {
                "agn_model": "composable",
                "agn_disc_block": "multicolor",
                "agn_torus_block": "none",
                "agn_nlr_block": "none",
                "agn_blr_block": "none",
                "agn_feii_block": "none",
                "agn_attenuation_block": "none",
            }
        )
        assert {"agn_a_spin", "agn_log_mbh"} <= consumed
        free_agn = {p for p in params.free_params if p.startswith("agn_")}
        # Pinned so a change to either source is visible here, not only
        # through the indirection above.
        assert free_agn == {"agn_lum_ratio", "agn_log_lbol"}
        assert not ({"agn_a_spin", "agn_log_mbh"} & free_agn), (
            "disc's explicit Fixed(DEFAULT) should govern its own physics "
            "parameters, not be silently overridden by the top-level wildcard"
        )
        for name in free_agn:
            assert not params.get_distribution(name).is_fixed

    def test_wildcard_at_sub_block_level_frees_block_params(self):
        """``torus={'type': 'skirtor', 'all_params': FREE}`` frees exactly
        SKIRTOR's own declared parameters -- :func:`_agn_subblock_declared_params`
        (signature introspection on the composable torus block), narrower
        than (and independent of) the top-level ``agn`` scope's
        all-active-blocks union. Was ``assert isinstance(params, Parameters)``
        (task-12 audit D2: this vacuous shape passed identically whether the
        wildcard froze 0 of SKIRTOR's params, all of them, or a foreign
        torus's params entirely -- exactly the defect this rewrite exists to
        catch)."""
        from tengri.parameters.groups import _agn_subblock_declared_params

        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                "torus": {"type": "skirtor", "all_params": FREE},
            },
            redshift=Fixed(0.1),
        )
        expected = _agn_subblock_declared_params("torus", "skirtor")
        assert expected == {
            "agn_oa_skirtor",
            "agn_p_skirtor",
            "agn_q_skirtor",
            "agn_radius_ratio",
            "agn_tau_skirtor",
            "agn_torus_frac",
        }
        free_agn = {p for p in params.free_params if p.startswith("agn_")}
        # disc=multicolor is Fixed(DEFAULT), so torus's own wildcard is the
        # ONLY source of free AGN params here: exact equality, not a subset.
        assert free_agn == expected
        for name in expected:
            assert not params.get_distribution(name).is_fixed

    def test_per_param_override_beats_wildcard(self):
        """An explicit per-parameter prior inside a wildcarded sub-block wins,
        and does not disturb the wildcard's freedom over the block's OTHER
        declared parameters. Was ``assert isinstance(params, Parameters)``
        (task-12 audit: vacuous -- the docstring claimed a precedence rule
        this body never exercised, because ``agn_log_lbol`` is a SHARED
        param and ``powerlaw`` disc owns nothing of its own to override)."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                "torus": {
                    "type": "skirtor",
                    "all_params": FREE,
                    "tau_skirtor": Fixed(7.0),  # per-param override
                },
            },
            redshift=Fixed(0.1),
        )
        tau_dist = params.get_distribution("agn_tau_skirtor")
        assert tau_dist.is_fixed
        assert float(tau_dist.default) == 7.0
        assert "agn_tau_skirtor" not in params.free_params
        # The wildcard still frees every OTHER declared torus param: the
        # override did not collapse the whole sub-block to Fixed.
        for name in ("agn_torus_frac", "agn_p_skirtor", "agn_q_skirtor", "agn_oa_skirtor"):
            assert name in params.free_params, f"{name} should stay free under the wildcard"


class TestAGNValidation:
    """Test error handling for invalid AGN specifications."""

    def test_agn_unknown_disc_type_raises(self):
        """Unknown disc block type raises ValueError."""
        with pytest.raises(ValueError, match=r"Unknown.*disc.*block.*type"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={"disc": {"type": "banana_disc"}},
                redshift=Fixed(0.1),
            )

    def test_agn_unknown_torus_type_raises(self):
        """Unknown torus block type raises ValueError."""
        with pytest.raises(ValueError, match=r"Unknown.*torus.*block.*type"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={"torus": {"type": "donut_model"}},
                redshift=Fixed(0.1),
            )

    def test_agn_unknown_lines_type_raises(self):
        """Unknown lines block type raises ValueError."""
        with pytest.raises(ValueError, match=r"Unknown.*lines.*block.*type"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={"lines": {"type": "squiggly"}},
                redshift=Fixed(0.1),
            )

    def test_agn_unknown_feii_type_raises(self):
        """Unknown feii block type raises ValueError."""
        with pytest.raises(ValueError, match=r"Unknown.*feii.*block.*type"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={"feii": {"type": "iron_oxide"}},
                redshift=Fixed(0.1),
            )

    def test_agn_unknown_atten_type_raises(self):
        """Unknown attenuation block type raises ValueError."""
        with pytest.raises(ValueError, match=r"Unknown.*atten.*block.*type"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={"atten": {"type": "cloud_of_dust"}},
                redshift=Fixed(0.1),
            )


class TestAGNProvenance:
    """Test provenance tagging for AGN parameters."""

    def test_agn_provenance_user_fixed(self):
        """User-fixed agn param tagged 'user_fixed'."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "all_params": Fixed(DEFAULT),  # Fix shared params
                "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
                "log_lbol": Fixed(11.42),  # Override shared param at agn level
            },
            redshift=Fixed(0.1),
        )
        prov = params._group_provenance
        assert prov["agn_log_lbol"] == "user_fixed"

    def test_agn_provenance_wildcard_free(self):
        """Agn params from wildcard FREE tagged 'wildcard_free'.

        The wildcard sits on the shared top-level ``agn`` dict, not nested in
        ``disc`` (#2187): every ``multicolor`` disc parameter is a *shared*
        AGN parameter -- partitioned under ``"agn"``, never ``"agn.disc"`` --
        so a wildcard restated on ``disc`` itself covers zero parameters and
        raises. The top-level wildcard is what actually reaches them.
        """
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "disc": {"type": "multicolor"},
                "all_params": FREE,
            },
            redshift=Fixed(0.1),
        )
        prov = params._group_provenance
        # At least some disc-related (shared) params carry the wildcard_free tag.
        assert isinstance(params, Parameters)
        assert prov["agn_log_lbol"] == "wildcard_free"


class TestAGNValidBlockTypes:
    """Test all canonical valid block types."""

    @pytest.mark.parametrize(
        "block_type",
        ["none", "powerlaw", "multicolor", "kubota_done", "adaf", "qsogen", "grahsp_sbpl"],
    )
    def test_valid_disc_types(self, block_type):
        """All known disc block types accepted."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={"disc": {"type": block_type, "all_params": Fixed(DEFAULT)}},
            redshift=Fixed(0.1),
        )
        assert params.agn_disc_block == block_type

    @pytest.mark.parametrize(
        "block_type",
        [
            "none",
            "simple",
            "two_temperature",
            "nenkova",
            "skirtor",
            "silva04",
            "cat3d_wind",
            "qsogen",
            "grahsp",
        ],
    )
    def test_valid_torus_types(self, block_type):
        """All known torus block types accepted."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={"torus": {"type": block_type, "all_params": Fixed(DEFAULT)}},
            redshift=Fixed(0.1),
        )
        assert params.agn_torus_block == block_type

    @pytest.mark.parametrize(
        "block_type",
        ["none", "analytic", "synthesizer", "synthesizer_spectra", "grahsp"],
    )
    def test_valid_nlr_types(self, block_type):
        """All known NLR block types accepted."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={"nlr": {"type": block_type, "all_params": Fixed(DEFAULT)}},
            redshift=Fixed(0.1),
        )
        assert params.agn_nlr_block == block_type

    @pytest.mark.parametrize(
        "block_type",
        ["none", "analytic", "synthesizer", "synthesizer_spectra", "grahsp", "qsogen"],
    )
    def test_valid_blr_types(self, block_type):
        """All known BLR block types accepted."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={"blr": {"type": block_type, "all_params": Fixed(DEFAULT)}},
            redshift=Fixed(0.1),
        )
        assert params.agn_blr_block == block_type

    @pytest.mark.parametrize(
        "block_type",
        ["none", "grahsp", "qsogen_balmer"],
    )
    def test_valid_feii_types(self, block_type):
        """All known feii block types accepted."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={"feii": {"type": block_type, "all_params": Fixed(DEFAULT)}},
            redshift=Fixed(0.1),
        )
        assert params.agn_feii_block == block_type

    @pytest.mark.parametrize(
        "block_type",
        ["none", "polar_dust", "grahsp_biatten", "qsogen_smc", "qsogen"],
    )
    def test_valid_atten_types(self, block_type):
        """All known attenuation block types accepted via type key."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={"atten": {"type": block_type, "all_params": Fixed(DEFAULT)}},
            redshift=Fixed(0.1),
        )
        assert params.agn_attenuation_block == block_type

    def test_valid_atten_smc_prevot_via_law_key(self):
        """smc_prevot is now selected via law='prevot_smc', not type key."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={"atten": {"law": "prevot_smc", "all_params": Fixed(DEFAULT)}},
            redshift=Fixed(0.1),
        )
        assert params.agn_attenuation_block == "smc_prevot"

    @pytest.mark.parametrize("bad_type", ["smc_prevot", "prevot_smc"])
    def test_old_smc_prevot_type_key_raises(self, bad_type):
        """Both retired type= spellings reach the working form in ONE
        message (Task 16, item 9, F8): type='smc_prevot' (the registry key)
        and type='prevot_smc' (the reversed spelling, which used to fall
        through to a generic "Unknown type" error whose difflib suggestion
        was 'smc_prevot' -- itself ALSO refused by this same check, a
        second hop to the same destination)."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={"atten": {"type": bad_type}},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert bad_type in error_msg
        assert "law" in error_msg
        assert "prevot_smc" in error_msg


class TestAGNEbvMigration:
    """D1 (task-12 public-API audit): the migration message this repo's own
    error raises for the retired ``type='smc_prevot'`` spelling must recommend
    a spelling that frees the parameter the user asked for.

    Two ``agn_*`` names end in ``ebv`` and each keeps its own prefix-stripped
    short name: ``agn_attenuation_ebv`` -> ``'attenuation_ebv'`` (the atten
    block's own E(B-V)) and ``agn_ebv`` -> ``'ebv'`` (the unrelated,
    pre-existing ``qsogen_smc`` reddening knob). D1's symptom was the message
    advertising ``'ebv'``: following it verbatim froze
    ``agn_attenuation_ebv`` at ``Fixed(0.0)`` and freed ``agn_ebv`` instead.

    R30 fixes that at the message rather than by aliasing ``'ebv'`` onto the
    other name -- one short spelling per parameter, no dual spellings, and the
    two names stay distinguishable. So the contract is: the message's own
    recommended key, taken verbatim, frees the live ``agn_attenuation_ebv``.
    """

    def _recommended_atten_key(self) -> str:
        """The short key the retired-spelling migration message advertises.

        Read out of the message rather than restated, so a message that drifts
        back to the D1 spelling fails here instead of silently teaching it.
        """
        import re

        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={
                    "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                    "atten": {"type": "smc_prevot"},
                    "all_params": Fixed(DEFAULT),
                },
                redshift=Fixed(0.1),
            )
        pattern = r"agn=\{'atten': \{'law': 'prevot_smc', '([a-z_]+)'"
        recipe = re.search(pattern, str(exc_info.value))
        assert recipe is not None, str(exc_info.value)
        return recipe.group(1)

    def test_migration_message_recommends_the_working_short_key(self):
        """The advertised key frees ``agn_attenuation_ebv``, not ``agn_ebv``."""
        key = self._recommended_atten_key()
        assert key == "attenuation_ebv", (
            f"the migration message advertises {key!r}; 'ebv' is agn_ebv's own "
            f"short name and freeing it is exactly the D1 defect"
        )
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                "atten": {"law": "prevot_smc", key: Uniform(0.0, 1.0)},
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.1),
        )
        free_agn = {p for p in params.free_params if p.startswith("agn_")}
        assert free_agn == {"agn_attenuation_ebv"}, (
            f"expected exactly {{'agn_attenuation_ebv'}}, got {sorted(free_agn)} "
            f"-- {key!r} resolved to the wrong parameter"
        )
        dist = params.get_distribution("agn_attenuation_ebv")
        assert dist.bounds == (0.0, 1.0)
        # agn_ebv (the unrelated qsogen_smc knob) must stay at its own
        # registry default, untouched by the atten-level key.
        agn_ebv_dist = params.get_distribution("agn_ebv")
        assert agn_ebv_dist.is_fixed

    def test_short_keys_stay_distinct_between_the_two_ebv_parameters(self):
        """Each name keeps its own prefix-stripped short spelling.

        ``'ebv'`` under ``atten`` is ``agn_ebv``'s short name and resolves
        there; it is not a second spelling of ``agn_attenuation_ebv``.
        """
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                "atten": {"law": "prevot_smc", "ebv": Uniform(0.0, 1.0)},
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.1),
        )
        free_agn = {p for p in params.free_params if p.startswith("agn_")}
        assert free_agn == {"agn_ebv"}, sorted(free_agn)
        assert params.get_distribution("agn_attenuation_ebv").is_fixed

    def test_prevot_smc_ebv_is_live_not_dead(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """The freed parameter must be the LIVE one: jax.grad != 0 on a band
        flux (D1's original symptom was a dead free parameter)."""
        import jax
        import jax.numpy as jnp

        from tengri import SEDModel

        model = SEDModel.build(
            synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={
                "type": "const",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": 10.0,
                "start_gyr": 1.0,
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            agn={
                "type": "composable",
                "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                "atten": {"law": "prevot_smc", "attenuation_ebv": Uniform(0.0, 1.0)},
                "all_params": Fixed(DEFAULT),
                "agn_log_lbol": Fixed(12.0),
                "norm": "independent",
            },
            redshift=Fixed(1.0),
        )
        p = dict(model.spec.sample(jax.random.PRNGKey(0)))
        v0 = jnp.asarray(p["agn_attenuation_ebv"])

        def obj(v):
            pd = {**p, "agn_attenuation_ebv": v}
            return jnp.log(jnp.sum(model.predict_photometry(pd)) + 1e-300)

        grad = float(jax.grad(obj)(v0))
        assert grad != 0.0, "agn_attenuation_ebv is dead -- the D1 fix regressed"


class TestAGNComplexScenarios:
    """Test realistic, multi-block AGN configurations."""

    def test_grahsp_full_recipe(self):
        """GRAHSP-pure recipe: every block uses GRAHSP impl."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "disc": {"type": "grahsp_sbpl", "all_params": Fixed(DEFAULT)},
                "torus": {"type": "grahsp", "all_params": Fixed(DEFAULT)},
                "nlr": {"type": "grahsp", "all_params": Fixed(DEFAULT)},
                "blr": {"type": "grahsp", "all_params": Fixed(DEFAULT)},
                "feii": {"type": "grahsp", "all_params": Fixed(DEFAULT)},
                "atten": {"type": "grahsp_biatten", "all_params": Fixed(DEFAULT)},
            },
            redshift=Fixed(0.1),
        )
        assert params.agn_model == "composable"
        assert params.agn_disc_block == "grahsp_sbpl"
        assert params.agn_torus_block == "grahsp"
        assert params.agn_nlr_block == "grahsp"
        assert params.agn_blr_block == "grahsp"
        assert params.agn_feii_block == "grahsp"
        assert params.agn_attenuation_block == "grahsp_biatten"

    def test_mixed_block_recipe(self):
        """Mix blocks: GRAHSP BBB + simple two-temperature torus + Prevot SMC."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "disc": {"type": "grahsp_sbpl", "all_params": Fixed(DEFAULT)},
                "nlr": {"type": "none", "all_params": Fixed(DEFAULT)},
                "blr": {"type": "none", "all_params": Fixed(DEFAULT)},
                "feii": {"type": "none", "all_params": Fixed(DEFAULT)},
                "torus": {"type": "two_temperature", "all_params": Fixed(DEFAULT)},
                "atten": {"law": "prevot_smc", "all_params": Fixed(DEFAULT)},
            },
            redshift=Fixed(0.1),
        )
        assert params.agn_disc_block == "grahsp_sbpl"
        assert params.agn_nlr_block == "none"
        assert params.agn_blr_block == "none"
        assert params.agn_feii_block == "none"
        assert params.agn_torus_block == "two_temperature"
        assert params.agn_attenuation_block == "smc_prevot"

    def test_minimal_agn_no_blocks(self):
        """Empty agn dict: all blocks default to 'none'."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={},
            redshift=Fixed(0.1),
        )
        assert params.agn_model == "composable"
        assert params.agn_disc_block == "none"
        assert params.agn_torus_block == "none"
        assert params.agn_nlr_block == "none"
        assert params.agn_blr_block == "none"
        assert params.agn_feii_block == "none"
        assert params.agn_attenuation_block == "none"


class TestAGNCrossLevelPlacement:
    """Two-level AGN grammar: top-level vs sub-block placement of a param.

    Regression: until 2026-05-23, the nested-dict resolver only looked
    at the canonical location for each AGN parameter (top level for
    shared params, the matching sub-block for sub-block params). Users
    who naturally placed ``agn_log_lbol`` inside ``disc`` (or
    ``tau_skirtor`` at the top level) silently fell back to the registry
    default. These tests pin the friendlier "accept either location"
    contract with conflict detection.
    """

    def test_shared_param_inside_sub_block_is_honored(self):
        """``agn_log_lbol`` (shared) supplied inside ``disc`` must apply."""
        supplied = Uniform(9.42, 13.42)
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "disc": {"type": "qsogen", "all_params": Fixed(DEFAULT), "agn_log_lbol": supplied},
                "torus": {"type": "none"},
                "lines": {"type": "none"},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
            },
            redshift=Fixed(0.1),
        )
        dist = params.get_distribution("agn_log_lbol")
        assert not dist.is_fixed
        # Compare against what was supplied: the claim is that the sub-block
        # prior survives, not that it equals any particular pair of numbers.
        assert dist.bounds == (supplied.lo, supplied.hi)
        assert params._group_provenance.get("agn_log_lbol") == "user_prior"

    def test_sub_block_param_at_top_level_now_raises(self):
        """Sub-block params written at top level must be nested (PR6)."""
        # PR6: AGN adopts dust.emission strictness — sub-block params
        # must be written under their owning sub-block, not at the agn level.
        with pytest.raises(ValueError, match="is a 'agn\\.torus' parameter"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={
                    "all_params": Fixed(DEFAULT),
                    "tau_skirtor": 7.5,  # Wrong level! Should be in torus={...}
                    "disc": {"type": "qsogen", "all_params": Fixed(DEFAULT)},
                    "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
                    "lines": {"type": "none"},
                    "feii": {"type": "none"},
                    "atten": {"type": "none"},
                },
                redshift=Fixed(0.1),
            )

    def test_param_in_two_locations_raises(self):
        """Same param at top level and inside a sub-block ⇒ ValueError."""
        with pytest.raises(ValueError, match="set in multiple locations"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={
                    "agn_log_lbol": Uniform(9.42, 13.42),
                    "disc": {
                        "type": "qsogen",
                        "all_params": Fixed(DEFAULT),
                        "agn_log_lbol": Uniform(10.42, 12.42),
                    },
                    "torus": {"type": "none"},
                    "lines": {"type": "none"},
                    "feii": {"type": "none"},
                    "atten": {"type": "none"},
                },
                redshift=Fixed(0.1),
            )

    def test_shared_param_in_sub_block_short_name_works(self):
        """Short-name form (``log_lbol``) inside a sub-block must also work,
        not only the full-prefix form (``agn_log_lbol``).
        """
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "disc": {
                    "type": "qsogen",
                    "all_params": Fixed(DEFAULT),
                    "log_lbol": Uniform(9.42, 13.42),
                },
                "torus": {"type": "none"},
                "lines": {"type": "none"},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
            },
            redshift=Fixed(0.1),
        )
        dist = params.get_distribution("agn_log_lbol")
        assert not dist.is_fixed
        assert dist.bounds == (9.42, 13.42)


class TestUniversalKeyValidator:
    """Every group dict now rejects unknown keys with a "Did you mean ...?"
    hint. Before this validator existed (2026-05-23), typos in any group
    (and parameters placed in the wrong group) silently fell back to the
    registry default — the dominant "AI slop" failure mode of the nested
    grammar.
    """

    @pytest.mark.parametrize(
        ("group_name", "group_dict"),
        [
            ("sfh", {"type": "dpl", "all_params": Fixed(DEFAULT), "pretend_param": 5}),
            (
                "dust_attenuation",
                {
                    "law": "power_law",
                    "type": "two_component",
                    "all_params": Fixed(DEFAULT),
                    "completely_fake_key": 99,
                },
            ),
            ("neb", {"type": "none", "phantom_neb_key": 3}),
            ("igm", {"type": "madau", "typo_igm_key": 1}),
            (
                "radio",
                {"sf": {"type": "none"}, "synth_radio_key": 1},
            ),  # composable, not legacy
            ("xray", {"type": "none", "typo_xray_key": 1}),
        ],
    )
    def test_unknown_key_in_top_level_group_raises(self, group_name, group_dict):
        with pytest.raises(ValueError, match=r"Unknown key '[^']+' in group"):
            parse_groups(**{group_name: group_dict, "redshift": Fixed(0.1)})

    def test_unknown_key_in_dust_emission_subblock_raises(self):
        with pytest.raises(ValueError, match=r"Unknown key '[^']+' in group 'dust_emission'"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "all_params": Fixed(DEFAULT),
                },
                dust_emission={
                    "type": "draine_li2007",
                    "all_params": Fixed(DEFAULT),
                    "phantom_emission_key": 77,
                },
                redshift=Fixed(0.1),
            )

    def test_unknown_key_in_agn_subblock_raises(self):
        with pytest.raises(ValueError, match=r"Unknown key '[^']+' in group 'agn.disc'"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={
                    "disc": {
                        "type": "qsogen",
                        "all_params": Fixed(DEFAULT),
                        "totally_made_up_key": 99,
                    },
                    "torus": {"type": "none"},
                    "lines": {"type": "none"},
                    "feii": {"type": "none"},
                    "atten": {"type": "none"},
                },
                redshift=Fixed(0.1),
            )

    def test_did_you_mean_suggestion_in_message(self):
        """The error should include difflib suggestions when a typo is close
        to a real parameter name."""
        with pytest.raises(ValueError, match=r"Did you mean:.*tau_skirtor"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={
                    "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT), "tau_skirto": 5.0},
                    "disc": {"type": "qsogen", "all_params": Fixed(DEFAULT)},
                    "lines": {"type": "none"},
                    "feii": {"type": "none"},
                    "atten": {"type": "none"},
                },
                redshift=Fixed(0.1),
            )


class TestComposableAGNRuntimeWiring:
    """Regression: composable AGN block selectors must reach the runtime
    forward model, not just the spec. Issue #258.

    Before this PR, ``AGNSEDComponent.apply`` called
    ``composable_agn_l_nu(wave, agn_log_lbol=..., agn_lum_ratio=..., ...)``
    without the five block selectors. Every block defaulted to
    ``"none"`` and the composable AGN SED came out identically zero
    regardless of the user's spec. The bug was downstream of
    parameter resolution — the spec showed the right block selectors,
    but the runtime never read them.

    These tests verify (a) the AGNSEDComponentConfig now carries the
    selectors and (b) the composable AGN actually produces non-zero
    SED through the standard ``SEDModel.build → predict_rest_sed`` path.
    Needs SSP data; skips when unavailable.
    """

    def _build_composable_model(self, ssp):
        import tengri

        # #613: synthetic SSP — the composable-AGN nonzero-SED check is driven by
        # the AGN bolometric luminosity (log_lbol), independent of the SSP.
        return tengri.SEDModel.build(
            ssp,
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            agn={
                "type": "composable",
                "all_params": Fixed(DEFAULT),
                "frac": 1.0,
                "log_lbol": 12.5,
                "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
                "lines": {"type": "nlr", "all_params": Fixed(DEFAULT)},
            },
            redshift=Fixed(0.05),
        )

    def test_composable_agn_emits_nonzero_sed(self, synthetic_ssp_wide):
        """``predict_rest_sed`` on a composable-AGN model must produce a
        non-zero SED dominated by the AGN (BBB peak in the UV)."""
        import jax
        import numpy as np

        model = self._build_composable_model(synthetic_ssp_wide)
        p = dict(model.spec.sample(jax.random.PRNGKey(0)))
        result = model.predict_rest_sed(p)
        sed = np.asarray(result.sed)
        wave = np.asarray(result.wavelength)
        peak_nu_L = (2.998e18 / wave * sed).max()
        # AGN BBB should dominate at ~1e46 erg/s. The stellar host-Hα floor
        # is ~1e34 — we set a safety margin of 1e40 to detect the AGN even
        # if the disc model implementation drifts slightly.
        assert peak_nu_L > 1e40, (
            f"composable AGN appears suppressed: peak ν·L = {peak_nu_L:.3e}; "
            "block selectors may not be reaching the runtime."
        )

    def test_adaf_plasma_params_not_a_noop(self, synthetic_ssp_wide):
        """agn_adaf_delta / agn_adaf_alpha measurably change the *AGN* SED through
        the public grammar — proof at the SEDModel.build layer that the plasma
        params route through spec -> runner -> block. Before #898 they were
        undeclared silent no-ops (agn_adaf_beta/delta were never in _params.py or
        _consumes). The AGN component (not the total SED) is the probe: the ADAF
        synchrotron/Compton action peaks in the radio-mm and X-ray, so on an
        optical-NIR grid it is dwarfed by the stellar host in the total SED.

        Parameters are chosen in the low-mdot regime (log_lbol=9, log_mbh=9 ->
        mdot ~ 1e-3, alpha_c>1) where delta is physically active: at high mdot
        (alpha_c<1) Mahadevan's Eq. 43 for T_e has no delta dependence, so delta
        is *correctly* inert there — a regime-dependence, not a no-op."""
        import jax
        import numpy as np

        from tengri import SEDModel

        def _build(delta, alpha):
            return SEDModel.build(
                synthetic_ssp_wide,
                sfh={
                    "type": "delayed",
                    "tau_gyr": Fixed(1.0),
                    "age_gyr": Fixed(5.0),
                    "log_total_mass": Fixed(0.0),
                    "all_params": Fixed(DEFAULT),
                },
                dust_attenuation={
                    "law": "power_law",
                    "type": "two_component",
                    "tau_bc": Fixed(0.0),
                    "tau_diff": Fixed(0.0),
                    "all_params": Fixed(DEFAULT),
                },
                agn={
                    "type": "composable",
                    "all_params": Fixed(DEFAULT),
                    "frac": 1.0,
                    "log_lbol": 9.0,
                    "disc": {
                        "type": "adaf",
                        "all_params": Fixed(DEFAULT),
                        # Task 16 (item 1): agn_log_mbh is an 'agn.disc'-owned
                        # parameter (disc physics), not the shared 'agn' group
                        # -- nest it here (placing it flat at the top level
                        # now raises, D1-guard-style: "nest it").
                        "log_mbh": 9.0,
                        "adaf_delta": Fixed(delta),
                        "adaf_alpha": Fixed(alpha),
                    },
                },
                redshift=Fixed(0.05),
            )

        # Grammar routing: the short-form keys reach the canonical param names.
        spec = _build(0.4, 0.2).spec
        assert spec.get_distribution("agn_adaf_delta").value == 0.4
        assert spec.get_distribution("agn_adaf_alpha").value == 0.2

        def _sed_agn(model):
            p = dict(model.spec.sample(jax.random.PRNGKey(0)))
            return np.asarray(model.predict_state(p).derived["sed_agn"])

        base = _sed_agn(_build(0.05, 0.3))

        def _max_rel(other):
            return float(np.max(np.abs(other - base) / (np.abs(base) + base.max() * 1e-12)))

        assert _max_rel(_sed_agn(_build(0.4, 0.3))) > 1e-3, (
            "agn_adaf_delta is a silent no-op through SEDModel.build"
        )
        assert _max_rel(_sed_agn(_build(0.05, 0.1))) > 1e-3, (
            "agn_adaf_alpha is a silent no-op through SEDModel.build"
        )

    def test_composable_agn_wildcard_fixed_emits_nonzero_sed(self, synthetic_ssp_wide):
        """Wildcard ``'all_params': Fixed(DEFAULT)`` with no explicit ``frac``
        must still produce a non-zero AGN SED (regression for #417).

        Before the fix, ``agn_lum_ratio`` defaulted to ``Fixed(0.0)`` in the
        param registry, so a wildcard-fixed AGN group collapsed
        ``composable_agn_l_nu = agn_lum_ratio * compose_l_nu(...) = 0`` and
        the AGN contribution was silently identically zero — even though
        ``L_agn_bol`` was published correctly.
        """
        import numpy as np

        import tengri

        # #613: synthetic SSP (AGN-luminosity-driven check, SSP-independent).
        model = tengri.SEDModel.build(
            synthetic_ssp_wide,
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            agn={
                "type": "composable",
                "all_params": Fixed(DEFAULT),  # NB: no explicit ``frac`` override
                "log_lbol": 13.0,
                "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
            },
            redshift=Fixed(0.0),
        )
        state = model.predict_state({})
        sed_agn_max = float(np.asarray(state.derived["sed_agn"]).max())
        assert sed_agn_max > 0.0, (
            f"composable AGN with 'all_params': Fixed(DEFAULT) produced zero SED "
            f"(sed_agn max = {sed_agn_max}); the registry default for "
            "agn_lum_ratio may have regressed back to 0."
        )

    def test_agn_norm_conserving_reachable_and_not_a_noop(self, synthetic_ssp_wide):
        """``agn_norm='conserving'`` must be selectable through the *public*
        grammar (not just the low-level ``composable()`` call) and must
        actually change ``predict()`` versus ``'independent'``.

        Guards the two failure modes this repo's multi-layer ``agn_norm``
        wiring keeps hitting: (a) a policy accepted by the runner but rejected
        by the grammar validator (unreachable from ``SEDModel.build``), and
        (b) a policy that is wired but silently a no-op.
        """
        import numpy as np

        import tengri

        def _build(norm):
            # disc=multicolor + torus=silva04: under 'conserving' the disc is
            # debited by (1 - agn_torus_frac); under 'independent' it is not, so
            # the two policies must give measurably different AGN SEDs.
            return tengri.SEDModel.build(
                synthetic_ssp_wide,
                sfh={
                    "type": "delayed",
                    "tau_gyr": Fixed(1.0),
                    "age_gyr": Fixed(5.0),
                    "log_total_mass": Fixed(0.0),
                    "all_params": Fixed(DEFAULT),
                },
                dust_attenuation={
                    "law": "power_law",
                    "type": "two_component",
                    "tau_bc": Fixed(0.0),
                    "tau_diff": Fixed(0.0),
                    "all_params": Fixed(DEFAULT),
                },
                agn={
                    "type": "composable",
                    "all_params": Fixed(DEFAULT),
                    "log_lbol": 13.0,
                    "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                    "torus": {"type": "silva04", "all_params": Fixed(DEFAULT)},
                    "norm": norm,
                },
                redshift=Fixed(0.0),
            )

        # (a) reachability: 'conserving' constructs through the grammar.
        m_cons = _build("conserving")
        m_indep = _build("independent")
        sed_cons = np.asarray(m_cons.predict_state({}).derived["sed_agn"])
        sed_indep = np.asarray(m_indep.predict_state({}).derived["sed_agn"])

        # (b) not a no-op: the conserving disc debit measurably changes the AGN
        # SED, and makes the AGN dimmer than the non-conserving 'independent' sum.
        assert not np.allclose(sed_cons, sed_indep), (
            "agn_norm='conserving' gave the same SED as 'independent' — the "
            "policy is a silent no-op (disc not debited)."
        )
        assert sed_cons.max() < sed_indep.max()

        # (c) end-to-end: the policy must propagate all the way to the total
        # predicted rest-frame SED (the spectrum photometry integrates), not
        # just the isolated AGN component.
        total_cons = np.asarray(m_cons.predict_rest_sed({}).sed)
        total_indep = np.asarray(m_indep.predict_rest_sed({}).sed)
        assert not np.allclose(total_cons, total_indep), (
            "agn_norm='conserving' did not change the total predicted SED — "
            "the policy is not propagating end-to-end."
        )

    def test_top_level_agn_type_selects_monolithic_model(self):
        """``agn={'type':'richards2006', ...}`` must produce a non-zero
        AGN SED (regression for #417).

        Before the fix, the top-level ``'type'`` key was silently dropped
        by ``_translate_agn`` and the model collapsed to
        ``composable``-with-all-none-blocks, which emits identically zero.
        """
        import pathlib

        ssp_path = (
            pathlib.Path(__file__).parents[2]
            / "data"
            / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
        )
        if not ssp_path.exists():
            pytest.skip(f"SSP file not available at {ssp_path}")

        import numpy as np

        import tengri

        model = tengri.SEDModel.build(
            tengri.load_ssp(),
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            agn={
                "type": "richards2006",
                "agn_log_lbol": Fixed(13.0),
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.0),
        )
        assert model._agn_model == "richards2006", (
            f"top-level type='richards2006' did not propagate to model "
            f"(_agn_model={model._agn_model!r}); _translate_agn may have "
            "regressed."
        )
        state = model.predict_state({})
        sed_agn_max = float(np.asarray(state.derived["sed_agn"]).max())
        assert sed_agn_max > 0.0, (
            f"richards2006 AGN produced zero SED (sed_agn max = {sed_agn_max})."
        )

    def test_mixing_top_type_with_sub_blocks_raises(self):
        """``agn={'type': 'richards2006', 'disc': {...}}`` must raise."""
        with pytest.raises(ValueError, match="monolithic"):
            parse_groups(
                sfh={"type": "const"},
                agn={
                    "type": "richards2006",
                    "disc": {"type": "multicolor"},
                },
                redshift=Fixed(0.1),
            )


class TestAGNLinesDeprecation:
    """The retired ``lines`` slot maps to ``nlr``/``blr`` via a deprecated alias.

    PR-A back-compat contract: external code using the old single ``lines`` slot
    keeps working, expands to the independent ``nlr``/``blr`` selectors, and emits
    a ``DeprecationWarning`` naming the mapping. Specifying both surfaces is an error.
    """

    def test_nested_lines_alias_warns_and_maps(self):
        """``agn={'lines': {'type': 'nlr_blr'}}`` -> nlr='analytic', blr='analytic'."""
        with pytest.warns(DeprecationWarning, match="lines"):
            params = parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={
                    "disc": {"type": "multicolor"},
                    "lines": {"type": "nlr_blr"},
                    "all_params": Fixed(DEFAULT),
                },
                redshift=Fixed(0.1),
            )
        assert params.agn_nlr_block == "analytic"
        assert params.agn_blr_block == "analytic"

    def test_nested_lines_single_region_alias(self):
        """A single-region legacy name maps to one slot only (other stays 'none')."""
        with pytest.warns(DeprecationWarning):
            params = parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={
                    "disc": {"type": "multicolor"},
                    "lines": {"type": "blr"},
                    "all_params": Fixed(DEFAULT),
                },
                redshift=Fixed(0.1),
            )
        assert params.agn_nlr_block == "none"
        assert params.agn_blr_block == "analytic"

    def test_flat_lines_kwarg_alias_warns_and_maps(self):
        """Flat ``Parameters(agn_lines_block='nlr_blr_synthesizer_spectra')`` expands."""
        with pytest.warns(DeprecationWarning, match="lines"):
            p = Parameters(
                agn_model="composable",
                agn_lines_block="nlr_blr_synthesizer_spectra",
            )
        assert p.agn_nlr_block == "synthesizer_spectra"
        assert p.agn_blr_block == "synthesizer_spectra"

    def test_specifying_both_lines_and_nlr_blr_raises(self):
        """Mixing the deprecated ``lines`` with new ``nlr``/``blr`` is an error."""
        with pytest.raises(ValueError, match="both"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={
                    "disc": {"type": "multicolor"},
                    "lines": {"type": "nlr_blr"},
                    "nlr": {"type": "analytic"},
                    "all_params": Fixed(DEFAULT),
                },
                redshift=Fixed(0.1),
            )


@pytest.mark.contract
class TestAGNSubblockStrictness:
    """Sub-block-owned parameters must be nested, not written flat at agn level.

    PR6: AGN adopts the dust.emission policy — rejecting sub-block params
    written at the wrong level with a clear error naming the correct nesting.
    """

    def test_subblock_owned_param_at_agn_level_raises_with_guidance(self):
        """A torus parameter written flat at agn level raises with nesting hint."""
        # agn_tau_skirtor is owned by the torus block, not shared
        with pytest.raises(ValueError, match="is a 'agn\\.torus' parameter"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={
                    "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
                    "tau_skirtor": Uniform(3, 11),  # Wrong level!
                },
                redshift=Fixed(0.1),
            )

    def test_subblock_owned_param_error_names_correct_nesting(self):
        """Error message for misplaced sub-block param shows correct nesting."""
        with pytest.raises(ValueError) as excinfo:
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={
                    "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
                    "tau_skirtor": Uniform(3, 11),
                },
                redshift=Fixed(0.1),
            )
        message = str(excinfo.value)
        # The error should name the correct nesting path
        assert "agn=" in message
        assert "torus" in message
        assert "tau_skirtor" in message

    def test_subblock_owned_param_in_correct_nest_works(self):
        """Same parameter nested correctly under torus dict works fine."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
                "torus": {
                    "type": "skirtor",
                    "all_params": Fixed(DEFAULT),
                    "tau_skirtor": Uniform(3, 11),
                },
            },
            redshift=Fixed(0.1),
        )
        dist = params.get_distribution("agn_tau_skirtor")
        assert dist.bounds == (3, 11)

    def test_nlr_owned_param_at_agn_level_raises(self):
        """A NLR parameter at agn level raises."""
        with pytest.raises(ValueError, match="is a 'agn\\.nlr' parameter"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={
                    "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
                    "nlr_cf": Uniform(0.01, 0.5),  # nlr param at wrong level
                },
                redshift=Fixed(0.1),
            )

    def test_shared_agn_params_still_work_at_agn_level(self):
        """Genuinely shared params (agn_log_lbol, agn_lum_ratio) work at agn level."""
        # These are shared across all blocks, not owned by any sub-block
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
                "log_lbol": Uniform(9.42, 13.42),  # Shared param at agn level is OK
            },
            redshift=Fixed(0.1),
        )
        dist = params.get_distribution("agn_log_lbol")
        assert dist.bounds == (9.42, 13.42)

    def test_multiple_subblock_params_at_agn_level_all_rejected(self):
        """Multiple misplaced sub-block params all raise (first one caught)."""
        with pytest.raises(ValueError, match="is a 'agn"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={
                    "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
                    "tau_skirtor": Uniform(3, 11),  # torus param
                    "nlr_cf": Uniform(0.01, 0.5),  # nlr param
                },
                redshift=Fixed(0.1),
            )


class TestAGNRoundTrip:
    """D9 (task-12 public-API audit): ``SEDModel.to_dict()`` (thin alias of
    ``model.spec.to_groups()``) round-trips through ``SEDModel.build`` bit-
    exactly, on the composable AGN capstone config this file's other classes
    exercise piecewise."""

    def test_to_dict_and_to_groups_roundtrip_bit_exact(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """``SEDModel.build(**m.spec.to_groups())`` and
        ``SEDModel.build(**m.to_dict())`` both reproduce
        ``predict_photometry`` bit-exactly on a composable AGN model
        spanning disc + torus (wildcarded) + nlr + atten (short-form
        'attenuation_ebv')."""
        import jax
        import numpy as np

        from tengri import SEDModel

        def _build():
            return SEDModel.build(
                synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                sfh={
                    "type": "const",
                    "all_params": Fixed(DEFAULT),
                    "log_total_mass": 10.0,
                    "start_gyr": 1.0,
                },
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "all_params": Fixed(DEFAULT),
                },
                agn={
                    "type": "composable",
                    "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                    "torus": {"type": "skirtor", "all_params": FREE},
                    "nlr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
                    "atten": {"law": "prevot_smc", "attenuation_ebv": Uniform(0.0, 1.0)},
                    "all_params": Fixed(DEFAULT),
                    "agn_log_lbol": Fixed(12.0),
                    "norm": "independent",
                },
                redshift=Fixed(1.0),
            )

        model = _build()
        p = dict(model.spec.sample(jax.random.PRNGKey(0)))
        reference = np.asarray(model.predict_photometry(p))

        via_to_groups = SEDModel.build(
            synthetic_ssp_wide, observation=synthetic_tophat_obs, **model.spec.to_groups()
        )
        via_to_dict = SEDModel.build(
            synthetic_ssp_wide, observation=synthetic_tophat_obs, **model.to_dict()
        )

        assert model.spec.to_groups() == model.to_dict(), (
            "to_dict() must be exactly spec.to_groups(), not merely equivalent"
        )
        assert np.array_equal(reference, np.asarray(via_to_groups.predict_photometry(p)))
        assert np.array_equal(reference, np.asarray(via_to_dict.predict_photometry(p)))


class TestCrossCategoryCompanionDoesNotDisturbOtherWildcards:
    """R36's cross-category claim must not reach a shared name, and must book
    its outcome against the wildcard that actually did the freeing.

    The claim exists so a block can free a parameter another category owns when
    that category's selected block does not read it (``schartmann2005_skirtor_atten``
    reading SKIRTOR geometry with no SKIRTOR torus). Two ways it over-reached:

    * the claims map admitted any companion whose owner differed from the
      block's own group, and a SHARED name's owner is ``"agn"``, which differs
      from every ``"agn.<category>"`` -- so ``agn_cos_inc`` and
      ``agn_polar_law`` were claimed by ``atten``;
    * the outcome was re-booked to the claiming category whenever a wildcard
      freed the name, without asking which wildcard fired.

    Together those made an ordinary build raise, and made two advisories lie.
    """

    def test_top_level_wildcard_with_a_pinned_polar_dust_atten_builds(self):
        """The hard regression: this raised ``ParameterError`` mid-round.

        ``atten='polar_dust'`` with an explicit ``Fixed(DEFAULT)`` wildcard --
        the exact spelling ``DefaultFixedParametersWarning`` tells users to
        write -- plus a top-level ``all_params: FREE``. The shared
        ``agn_cos_inc``/``agn_polar_law``, freed by the TOP-LEVEL wildcard,
        were booked under ``agn.atten``; that group was then rebuilt from its
        declared set (the four polar names, none of them freed) and reported
        as ``'all_params: FREE' freed 0 of 4 parameters in group 'agn.atten'``.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RecipeWarning)
            params = parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={
                    "type": "composable",
                    "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                    "nlr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
                    "blr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
                    "atten": {"type": "polar_dust", "all_params": Fixed(DEFAULT)},
                    "all_params": FREE,
                    "norm": "independent",
                },
                redshift=Fixed(0.1),
            )
        free_agn = {p for p in params.free_params if p.startswith("agn_")}
        # The shared knobs the top-level wildcard reaches, and nothing else.
        assert free_agn == {"agn_cos_inc", "agn_log_lbol", "agn_lum_ratio"}, sorted(free_agn)
        # The atten sub-block said Fixed(DEFAULT), so its own names stay pinned.
        for name in ("agn_polar_T", "agn_polar_beta", "agn_polar_ebv", "agn_polar_oa"):
            assert params.get_distribution(name).is_fixed, name

    def test_a_feii_wildcard_does_not_report_a_blr_group_it_never_wrote(self):
        """The false advisories: a group the user never wildcarded was
        reported, and a freed parameter was reported pinned.

        ``agn_fe2_strength`` is owned by ``feii`` and read by the analytic BLR
        block, so the feii wildcard frees it (R33). Booking it under
        ``agn.blr`` -- a group with no ``'*'`` at all here -- produced a
        ``WildcardPartialFreeWarning`` for that phantom group AND made the feii
        group report the parameter pinned while ``spec.free_params`` contained
        it.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            params = parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                agn={
                    "type": "composable",
                    "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                    "blr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
                    "feii": {"type": "grahsp", "all_params": FREE},
                    "all_params": Fixed(DEFAULT),
                    "norm": "independent",
                },
                redshift=Fixed(0.1),
            )
        assert "agn_fe2_strength" in params.free_params

        wildcard_messages = [str(w.message) for w in caught if "Wildcard" in w.category.__name__]
        assert not [m for m in wildcard_messages if "'agn.blr'" in m], wildcard_messages
        assert not [m for m in wildcard_messages if "agn_fe2_strength" in m and "pinned" in m], (
            wildcard_messages
        )

    def test_no_name_is_declared_by_two_active_blocks(self):
        """The invariant the claim must preserve, over every selection pair.

        A name two active blocks both claim could be freed by either wildcard,
        which makes "which disposition wins" depend on resolution order rather
        than on what the user wrote. Swept over every ordered pair of
        categories and their registered types.
        """
        from tengri.components.agn.blocks._protocol import AGN_BLOCKS
        from tengri.parameters.groups import _AGN_CONSUMES_CATEGORY, _agn_subblock_declared_params

        categories = list(_AGN_CONSUMES_CATEGORY)
        types = {
            category: [
                name for name in AGN_BLOCKS[_AGN_CONSUMES_CATEGORY[category]] if name != "none"
            ]
            for category in categories
        }
        overlaps = []
        for first in categories:
            for second in categories:
                if first >= second:
                    continue
                for first_type in types[first]:
                    for second_type in types[second]:
                        selection = {first: first_type, second: second_type}
                        both = (
                            _agn_subblock_declared_params(first, first_type, selection=selection)
                            or frozenset()
                        ) & (
                            _agn_subblock_declared_params(second, second_type, selection=selection)
                            or frozenset()
                        )
                        if both:
                            overlaps.append(
                                f"{first}/{first_type} and {second}/{second_type} "
                                f"both declare {sorted(both)}"
                            )
        assert not overlaps, "\n".join(sorted(set(overlaps)))


class TestAgnLogLbolFracAgnConflict:
    """A user-provided ``agn_log_lbol`` that cannot act must raise (R55).

    Under ``agn_norm='cigale_joint'`` with a SKIRTOR torus and an active
    ``agn_ir_frac``, the AGN power is derived from the dust-absorbed stellar
    luminosity, ``L_absorbed x f/(1-f)``, and a user-supplied
    ``agn_log_lbol`` is computed over and discarded.

    Measured on this branch (composable disc=schartmann2005, ``sfh=delayed``,
    ``dust_emission=dale2014_cigale``), sweeping ``agn_log_lbol`` across the
    5%/95% quantiles of its declared ``Uniform(8, 14)`` prior -- 8.3 to 13.7,
    a 5.4-dex range -- and reading ``max|d sed_agn| / max sed_agn``:

    ==========================================  ==========  ======
    configuration (``agn_ir_frac=0.3``)         rel change  verdict
    ==========================================  ==========  ======
    cigale_joint + skirtor, no line block         6.44e-15  inert
    cigale_joint + skirtor, ``nlr='analytic'``    2.51e+05  live
    cigale_joint + skirtor, ``blr='analytic'``    2.51e+05  live
    cigale_joint + ``torus='fritz'``              2.51e+05  live
    cigale_joint + no torus                       2.51e+05  live
    ``norm='independent'`` + skirtor              2.51e+05  live
    cigale_joint + skirtor, ``agn_ir_frac=0.0``   2.51e+05  live
    ==========================================  ==========  ======

    The ``norm='independent'`` row stays live and R55 still does not refuse
    it -- but that build is refused one guard over, by R65
    (``_validate_independent_norm_without_fracagn``), because disc and torus
    then sit on unrelated luminosity scales. R55's message no longer offers it
    as a way out.

    Exactly one row is inert, so the refusal must be that narrow. It is, and
    not by restating those five carve-outs as a condition list that can go
    stale: #2069's guard *measures* the SED at the prior bounds and refuses
    only a flat direction. R55's addition is that a **user-provided Fixed**
    value gets the same measurement a FREE one already got -- the value being
    silently discarded is the same defect whether or not a sampler is moving
    it (explicit-over-silent). The registry default is never refused: every
    ``'all_params': Fixed(DEFAULT)`` AGN build carries one.

    Sibling: ``_validate_torus_frac_fracagn_conflict`` (#2189, R15).
    """

    def _build(self, ssp, *, log_lbol=None, ir_frac=None, norm="cigale_joint", torus="skirtor"):
        import tengri

        agn = {
            "type": "composable",
            "norm": norm,
            "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
            "all_params": Fixed(DEFAULT),
        }
        if torus is not None:
            agn["torus"] = {"type": torus, "all_params": Fixed(DEFAULT)}
        if log_lbol is not None:
            agn["agn_log_lbol"] = log_lbol
        if ir_frac is not None:
            agn["agn_ir_frac"] = ir_frac
        return tengri.SEDModel.build(
            ssp,
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "law": "calzetti",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.5),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
            agn=agn,
            redshift=Fixed(0.05),
        )

    # ── the refusal ────────────────────────────────────────────────

    def test_user_fixed_log_lbol_with_active_ir_frac_raises(self, synthetic_ssp_wide):
        """R55: a pinned value that is discarded is refused, not accepted."""
        from tengri.config.exceptions import ConfigError

        with pytest.raises(ConfigError, match=r"agn_log_lbol"):
            self._build(synthetic_ssp_wide, log_lbol=Fixed(12.0), ir_frac=Fixed(0.3))

    def test_refusal_names_the_coupling_and_the_way_out(self, synthetic_ssp_wide):
        """The message says where the AGN power comes from and what to change.

        And it must not advise a configuration this same guard refuses --
        #1364's rule. Before R55 the message read "Fix agn_log_lbol (any
        value; it cancels)", which is exactly the build this test performs.
        """
        from tengri.config.exceptions import ConfigError

        with pytest.raises(ConfigError) as exc:
            self._build(synthetic_ssp_wide, log_lbol=Fixed(12.0), ir_frac=Fixed(0.3))
        msg = str(exc.value)
        assert "agn_ir_frac" in msg
        assert "identical" in msg, "the refusal must report its measurement"
        assert "Fix agn_log_lbol" not in msg, (
            "the advice must not name the configuration the guard refuses (#1364)"
        )

    def test_free_log_lbol_with_active_ir_frac_still_raises(self, synthetic_ssp_wide):
        """#2069's original case keeps raising: a flat FREE direction."""
        from tengri.config.exceptions import ConfigError

        with pytest.raises(ConfigError, match=r"agn_log_lbol"):
            self._build(synthetic_ssp_wide, log_lbol=FREE, ir_frac=Fixed(0.3))

    def test_free_ir_frac_with_user_fixed_log_lbol_raises(self, synthetic_ssp_wide):
        """A FREE fracAGN is active by construction; the pin is still dead."""
        from tengri.config.exceptions import ConfigError

        with pytest.raises(ConfigError, match=r"agn_log_lbol"):
            self._build(synthetic_ssp_wide, log_lbol=Fixed(12.0), ir_frac=FREE)

    # ── every configuration measured LIVE must keep building ───────

    def test_ir_frac_alone_builds(self, synthetic_ssp_wide):
        """fracAGN on its own is the CIGALE-style configuration: legal."""
        assert self._build(synthetic_ssp_wide, ir_frac=Fixed(0.3)).spec.agn_model == "composable"

    def test_log_lbol_alone_builds(self, synthetic_ssp_wide):
        """A direct AGN luminosity with no fracAGN coupling: legal, and live."""
        assert self._build(synthetic_ssp_wide, log_lbol=Fixed(12.0)).spec.agn_model == "composable"

    def test_ir_frac_explicitly_zero_does_not_raise(self, synthetic_ssp_wide):
        """``agn_ir_frac=0.0`` states "no coupling", so nothing is discarded.

        Measured live: 2.51e5 relative across the prior.
        """
        model = self._build(synthetic_ssp_wide, log_lbol=Fixed(12.0), ir_frac=Fixed(0.0))
        assert model.spec.agn_model == "composable"

    def test_independent_norm_with_active_ir_frac_now_raises(self, synthetic_ssp_wide):
        """R65 refuses this build; R55's own measurement is unchanged.

        ``norm='independent'`` does put the disc on ``agn_log_lbol`` itself --
        measured live 2.51e5 relative across the prior with
        ``agn_ir_frac=0.3`` active, the table row above, and that is why R55
        does not refuse it. It is now refused one guard over, by R65
        (``_validate_independent_norm_without_fracagn``), for a different
        reason: the disc is then on ``agn_log_lbol`` while the torus follows
        ``L_absorbed x f/(1 - f)``, so ``int(disc)/int(torus)`` reports the
        stellar mass (measured 5.00e+10 at ``log M* = 0`` down to 5.00e-02 at
        ``log M* = 12``). R65 fires first, so the two refusals are told apart
        by which message comes back -- this asserts R65's, and
        ``tests/contract/test_agn_norm_ir_frac_coherence.py`` holds the rest.
        R55's own remedy no longer names this configuration.
        """
        from tengri.config.exceptions import ConfigError

        with pytest.raises(ConfigError, match=r"agn_norm='independent'") as exc:
            self._build(
                synthetic_ssp_wide, log_lbol=Fixed(12.0), ir_frac=Fixed(0.3), norm="independent"
            )
        assert "cigale_joint" in str(exc.value)

    def test_independent_norm_without_ir_frac_still_builds(self, synthetic_ssp_wide):
        """The half of the row R65 leaves alone: no coupling, no refusal.

        Keeps the ``norm='independent'`` arm of R55's table live after R65
        narrowed it, so the guard above is not the only thing this class says
        about that policy.
        """
        model = self._build(synthetic_ssp_wide, log_lbol=Fixed(12.0), norm="independent")
        assert model.spec.agn_model == "composable"

    def test_non_skirtor_torus_does_not_raise(self, synthetic_ssp_wide):
        """The joint coupling is a SKIRTOR mechanism; ``fritz`` is untouched.

        Measured live: 2.51e5 relative across the prior.
        """
        model = self._build(
            synthetic_ssp_wide, log_lbol=Fixed(12.0), ir_frac=Fixed(0.3), torus="fritz"
        )
        assert model.spec.agn_model == "composable"

    def test_active_nlr_block_does_not_raise(self, synthetic_ssp_wide):
        """An NLR block reads the disc luminosity, so the direction is live.

        Measured live: 2.51e5 relative across the prior. This is the carve-out
        a static condition list would have to remember and the measurement
        gets for free.
        """
        import tengri

        model = tengri.SEDModel.build(
            synthetic_ssp_wide,
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "law": "calzetti",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.5),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
            agn={
                "type": "composable",
                "norm": "cigale_joint",
                "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
                "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
                "nlr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
                "agn_log_lbol": Fixed(12.0),
                "agn_ir_frac": Fixed(0.3),
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.05),
        )
        assert model.spec.agn_model == "composable"

    # ── the guard gates on user-provided, never on the default ─────

    def test_default_log_lbol_beside_active_ir_frac_builds(self, synthetic_ssp_wide):
        """Every ``'all_params': Fixed(DEFAULT)`` AGN build carries the default.

        Refusing those would refuse the recipes, so the registry default is
        not a conflict even though it is just as inert.
        """
        model = self._build(synthetic_ssp_wide, ir_frac=Fixed(0.3))
        assert "agn_log_lbol" in model.spec.get_fixed_values()
