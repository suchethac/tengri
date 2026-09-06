# SPDX-License-Identifier: BSD-3-Clause
"""Block-scoped AGN wildcard: CONSUMES table integrity and no-op-free recipes.

Background. AGN parameters carry *free* Uniform/LogUniform registry defaults so
the FREE grammar can expand them (before, every agn_* param had a Fixed default,
so ``agn={'all_params': FREE}`` and ``recipes.agn_panchromatic()`` silently produced zero
free AGN parameters). Because the composable AGN component declares the full
~50-param superset, a group wildcard is *scoped* to the parameters the active
disc/torus/lines/feii/atten blocks actually consume — otherwise it would free
dozens of no-op nuisance dimensions. These tests pin that contract:

1. the CONSUMES tables reference only real declared params (no typos);
2. :func:`agn_active_param_set` scopes the wildcard correctly and falls back to
   the full superset for unknown / grid-gated blocks (never under-frees);
3. a composable ``'all_params': FREE`` frees exactly the active set (spec-level, synthetic
   SSP — CI-runnable, no grids);
4. every free AGN parameter in ``recipes.agn_panchromatic()`` actually moves
   ``predict()`` — the "no no-op free parameters" guarantee (gate-2; SSP-gated).
"""

from __future__ import annotations

import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

from tengri import DEFAULT, FREE, Fixed, SEDModel
from tengri.components.agn._params import PARAMS as _AGN_PARAMS
from tengri.components.agn.blocks._consumes import (
    AGN_BLOCK_CONSUMES,
    AGN_MODEL_CONSUMES,
    AGN_SHARED_PARAMS,
    ALL_AGN_PARAMS,
    agn_active_param_set,
)

_DECLARED = {pd.name for pd in _AGN_PARAMS}


def test_consumes_tables_reference_only_declared_params():
    """Every name in the CONSUMES tables is a real declared agn_* parameter."""
    assert AGN_SHARED_PARAMS <= _DECLARED
    assert ALL_AGN_PARAMS == _DECLARED
    for key, params in AGN_BLOCK_CONSUMES.items():
        assert params <= _DECLARED, f"{key} lists undeclared params: {params - _DECLARED}"
    for model, params in AGN_MODEL_CONSUMES.items():
        assert params <= _DECLARED, f"{model} lists undeclared params: {params - _DECLARED}"


def test_skirtor_torus_no_longer_consumes_polar_dust_knobs():
    """SKIRTOR torus must NOT credit any polar-dust knob (R22 regression guard).

    Superseded test (was ``test_skirtor_torus_consumes_all_polar_dust_knobs``):
    ``skirtor_torus_block`` used to bundle its OWN Casey-2012 polar-dust
    graybody (reading ``agn_polar_ebv``/``agn_polar_T``/``agn_polar_beta``,
    active by default at ``agn_polar_ebv = 0.03``) — a SECOND, independent
    polar-dust mechanism alongside the composable runner's Stage-1.5 disc
    reddening and the standalone ``polar_dust`` attenuation block, so a
    ``torus="skirtor"`` + ``atten="polar_dust"`` recipe screened the disc
    TWICE (task13 fix-round-1, ruling R22). There is now exactly ONE
    polar-dust mechanism — the standalone ``polar_dust`` attenuation block
    (``("attenuation", "polar_dust")`` below) — and this torus emits only the
    thermal SKIRTOR template. This guards against the bundled term coming
    back, without needing the gitignored SKIRTOR grid (pure CONSUMES-table
    membership).
    """
    skirtor = AGN_BLOCK_CONSUMES[("torus", "skirtor")]
    assert not ({"agn_polar_ebv", "agn_polar_T", "agn_polar_beta"} & skirtor)


def test_skirtor_torus_consumes_radius_ratio():
    """Task 16 (item 3, F4): agn_radius_ratio (the SKIRTOR grid's third axis)
    was missing from AGN_BLOCK_CONSUMES[('torus', 'skirtor')] even though
    skirtor_torus_block's own signature has always read it -- so
    agn={'all_params': FREE} + torus='skirtor' never froze it via the
    top-level wildcard (only the sub-block's own explicit short-key path
    reached it). See test_skirtor_torus_wiring.py's own liveness guard.
    """
    assert "agn_radius_ratio" in AGN_BLOCK_CONSUMES[("torus", "skirtor")]


def test_qsogen_and_smc_prevot_atten_consume_attenuation_ebv():
    """Task 16 (item 3): ('attenuation', 'qsogen') was missing entirely from
    AGN_BLOCK_CONSUMES (the top-level wildcard silently fell back to the
    full superset whenever Temple+2021's own quasar extinction curve was
    selected); ('attenuation', 'smc_prevot') was present but wrongly empty
    (smc_prevot_block's signature reads agn_attenuation_ebv, same as
    qsogen's). Both attenuation blocks delegate to the same E(B-V) knob.
    """
    assert AGN_BLOCK_CONSUMES[("attenuation", "qsogen")] == frozenset({"agn_attenuation_ebv"})
    assert AGN_BLOCK_CONSUMES[("attenuation", "smc_prevot")] == frozenset({"agn_attenuation_ebv"})


def test_slone_netzer_disc_registered():
    """Task 16 (item 3): ('disc', 'slone_netzer') was omitted on a stale
    "grid absent from CI" rationale. Both of the block's own axis parameters
    are live and both are listed.

    ``agn_log_ledd`` was briefly recorded as dead here. That reading was taken
    at the shared declared default -1.0, which lies above the SN12 axis
    ``[-4, -1.9586]`` and is clipped onto the edge node, where ``jnp.clip``
    makes the gradient exactly zero by construction -- the dead baseline
    ``tests/regression/agn/test_issue_1586_grid_support.py`` exists to describe,
    not a #846 degeneracy. Measured inside the axis the gradient runs 5.9e-2 to
    8.8e-1; that file pins the measurement.
    """
    assert AGN_BLOCK_CONSUMES[("disc", "slone_netzer")] == frozenset(
        {"agn_log_mbh", "agn_log_ledd"}
    )


def test_every_declared_and_consumed_name_has_one_partition_owner():
    """R34: partition completeness is a contract, not a convenience.

    ``_AGN_PARTITION`` decides which wildcard can free a name. A name absent
    from it silently defaults to the shared ``"agn"`` group, so the model is
    only as complete as the hand-maintained table -- and every instrument built
    on top inherits the hole. The measured-liveness module derives its
    ``owned`` set from the same table, so a missing entry is invisible to the
    measurement too: the blind spot moves rather than closing.

    Measured before this contract existed: 31 of 94 declared ``agn_*``
    parameters had no entry, among them every ``agn_nlr_*`` grid knob,
    ``agn_blr_logU``/``logZ``/``logn``, ``agn_adaf_alpha``/``beta``/``delta``,
    ``agn_astar``, ``agn_log_mdot`` and ``agn_cigale_disk_delta``; eight of
    them were measured live on ``predict_photometry`` while no wildcard could
    free them.

    The expected set is derived from the registries -- the declared parameter
    table and every name any block records as consumed -- never from
    ``_AGN_PARTITION`` itself, which is the thing under test.
    """
    from tengri.components.agn._params import PARAMS
    from tengri.parameters.groups import _AGN_PARTITION

    universe = {pd.name for pd in PARAMS}
    for consumed in AGN_BLOCK_CONSUMES.values():
        universe |= set(consumed)

    missing = sorted(name for name in universe if name not in _AGN_PARTITION)
    assert not missing, (
        f"{len(missing)} declared/consumed AGN names have no explicit "
        f"_AGN_PARTITION owner and silently fall through to the shared 'agn' "
        f"group: {missing}"
    )

    # The reverse direction: an agn_* entry naming a parameter that no longer
    # exists is a stale owner nothing can reach.
    stale = sorted(
        name for name in _AGN_PARTITION if name.startswith("agn_") and name not in universe
    )
    assert not stale, f"_AGN_PARTITION owns names that are not declared: {stale}"


def test_active_set_scopes_to_active_blocks():
    """agn_active_param_set unions shared + active-block consumed params."""
    cfg = {
        "agn_model": "composable",
        "agn_disc_block": "multicolor",
        "agn_torus_block": "skirtor",
        "agn_nlr_block": "analytic",
        "agn_blr_block": "none",
        "agn_feii_block": "none",
        "agn_attenuation_block": "none",
    }
    active = agn_active_param_set(cfg)
    # Shared knobs + the consumed params of each active block, nothing else.
    expected = (
        AGN_SHARED_PARAMS
        | AGN_BLOCK_CONSUMES[("disc", "multicolor")]
        | AGN_BLOCK_CONSUMES[("torus", "skirtor")]
        | AGN_BLOCK_CONSUMES[("nlr", "analytic")]
    )
    assert active == expected
    # Params owned by *inactive* blocks must not be active.
    assert "agn_tau" not in active  # Nenkova torus
    assert "agn_grahsp_l5100" not in active  # GRAHSP disc
    assert "agn_T_hot" not in active  # two-temperature torus
    # ... and the superset is much larger, so scoping is doing real work.
    assert len(active) < len(ALL_AGN_PARAMS)


def test_active_set_empty_without_agn():
    assert agn_active_param_set({}) == frozenset()
    assert agn_active_param_set({"agn_model": None}) == frozenset()


def test_unknown_block_falls_back_to_full_superset():
    """An unknown/grid-gated block over-frees (safe) rather than under-frees.

    Uses a name that is not, and can never accidentally become, a real
    registered block (``cat3d_wind`` used to sit here on the theory its grid
    was absent from CI; fix round 1 found the grid IS tracked and added the
    real entry below, so this regression test now needs a genuinely
    unregistered name rather than one that could quietly stop testing the
    fallback the day someone registers it).
    """
    cfg = {"agn_model": "composable", "agn_torus_block": "definitely_unregistered_torus_type"}
    assert agn_active_param_set(cfg) == ALL_AGN_PARAMS
    # Unknown monolithic model likewise.
    assert agn_active_param_set({"agn_model": "grahsp"}) == ALL_AGN_PARAMS


def test_cat3d_wind_family_top_level_wildcard_frees_exact_consumed_set():
    """Fix round 1 (CRITICAL): ``cat3d_wind`` and ``cat3d_wind_lowfwd`` were
    both absent from ``AGN_BLOCK_CONSUMES``, so ``agn={'all_params': FREE}``
    with either torus fell back to the full ~96-name superset (89 of them
    foreign to the block). Assert the exact freed set, not a count -- a
    superset regression must fail here even if some other unrelated entry
    changes the total count.
    """
    for torus_type, axis_params in (
        ("cat3d_wind", {"agn_a_cat3d", "agn_fwd_cat3d"}),
        ("cat3d_wind_lowfwd", {"agn_a_cat3d_lowfwd", "agn_fwd_cat3d_lowfwd"}),
    ):
        cfg = {"agn_model": "composable", "agn_torus_block": torus_type}
        active = agn_active_param_set(cfg)
        assert active != ALL_AGN_PARAMS, f"{torus_type}: still falling back to the full superset"
        expected = (
            AGN_SHARED_PARAMS | {"agn_cos_inc", "agn_theta_torus", "agn_torus_frac"} | axis_params
        )
        assert active == expected, (
            f"{torus_type}: freed {sorted(active)}, expected {sorted(expected)}"
        )


def test_grahsp_composable_blocks_scope_not_superset():
    """All six GRAHSP composable blocks are mapped, so a full grahsp config
    scopes to its consumed params instead of falling back to the full superset.

    ``('nlr', 'grahsp')`` and ``('blr', 'grahsp')`` and ``('feii', 'grahsp')`` were
    missing from the CONSUMES map; an unmapped block makes ``agn_active_param_set``
    over-free to ``ALL_AGN_PARAMS``, i.e. ~39 no-op nuisance dimensions under a
    top-level ``agn={'all_params': FREE}`` (the disc/torus/attenuation grahsp blocks *were*
    mapped, so the omission was an inconsistency, not a deliberate grid gate).
    """
    for key in (
        ("disc", "grahsp_sbpl"),
        ("torus", "grahsp"),
        ("nlr", "grahsp"),
        ("blr", "grahsp"),
        ("feii", "grahsp"),
        ("attenuation", "grahsp_biatten"),
    ):
        assert key in AGN_BLOCK_CONSUMES, f"{key} unmapped — would over-free to superset"
    cfg = {
        "agn_model": "composable",
        "agn_disc_block": "grahsp_sbpl",
        "agn_torus_block": "grahsp",
        "agn_nlr_block": "grahsp",
        "agn_blr_block": "grahsp",
        "agn_feii_block": "grahsp",
        "agn_attenuation_block": "grahsp_biatten",
    }
    active = agn_active_param_set(cfg)
    assert active != ALL_AGN_PARAMS, "grahsp config over-frees to the full superset"
    # Every active param is a grahsp knob or a shared knob — no foreign no-ops.
    assert all(p.startswith("agn_grahsp_") or p in AGN_SHARED_PARAMS for p in active)


def test_combined_nlr_blr_lines_blocks_registered_and_mapped():
    """Independent NLR+BLR blocks are registered; scoped config includes both.

    The ``lines`` slot (single region) has been split into independent ``nlr``
    and ``blr`` slots. A unified AGN (disc + torus + NLR + BLR) is now expressed
    by setting both ``nlr_block`` and ``blr_block`` selectors. Independent blocks
    register in the CONSUMES table, and a config using both scopes to their
    union (no superset fallback).
    """
    from tengri.components.agn.blocks._protocol import AGN_BLOCKS

    assert "analytic" in AGN_BLOCKS["nlr"]
    assert "synthesizer" in AGN_BLOCKS["nlr"]
    assert "synthesizer_spectra" in AGN_BLOCKS["nlr"]
    assert "grahsp" in AGN_BLOCKS["nlr"]
    assert "analytic" in AGN_BLOCKS["blr"]
    assert "synthesizer" in AGN_BLOCKS["blr"]
    assert "synthesizer_spectra" in AGN_BLOCKS["blr"]
    assert "grahsp" in AGN_BLOCKS["blr"]
    assert "qsogen" in AGN_BLOCKS["blr"]


# The three variants whose NLR and BLR blocks are genuinely independent, i.e.
# the two regions declare disjoint parameters and enabling both costs the union.
# ``grahsp`` is deliberately absent — see the negative control below.
_INDEPENDENT_NLR_BLR_VARIANTS = ["analytic", "synthesizer", "synthesizer_spectra"]


@pytest.mark.parametrize("variant", _INDEPENDENT_NLR_BLR_VARIANTS)
def test_independent_nlr_and_blr_consume_disjoint_params(variant):
    """Enabling both regions must cost exactly the sum of their parameters.

    This is what "the CONSUMES are unionable" has to mean to be worth
    asserting: the two blocks are independent, so the union loses nothing.
    If they ever came to share a knob, one region would silently move the
    other's parameter and ``agn_active_param_set`` would under-count the
    free parameters for a unified AGN.

    The assertion this replaced was ``assert (nlr | blr) == (nlr | blr)`` for
    each of the three variants — the same expression on both sides, so it
    held for any contents whatsoever and could only fail if ``|`` raised.
    """
    nlr = AGN_BLOCK_CONSUMES[("nlr", variant)]
    blr = AGN_BLOCK_CONSUMES[("blr", variant)]
    overlap = set(nlr) & set(blr)
    assert not overlap, f"{variant}: nlr and blr both consume {sorted(overlap)}"
    assert len(set(nlr) | set(blr)) == len(set(nlr)) + len(set(blr))


def test_the_disjointness_check_is_not_vacuous():
    """``grahsp`` shares its NLR/BLR knobs — the counterexample that proves teeth.

    Both regions are driven by one line amplitude and one line width, so the
    union is *smaller* than the sum. Kept as a live negative control: if this
    ever became disjoint too, the test above would be passing on a property no
    registered variant can violate, and would need a new counterexample.
    """
    nlr = set(AGN_BLOCK_CONSUMES[("nlr", "grahsp")])
    blr = set(AGN_BLOCK_CONSUMES[("blr", "grahsp")])
    assert nlr & blr, "no registered variant shares nlr/blr params any more"
    assert len(nlr | blr) < len(nlr) + len(blr)
    cfg = {
        "agn_model": "composable",
        "agn_disc_block": "kubota_done",
        "agn_torus_block": "simple",
        "agn_nlr_block": "analytic",
        "agn_blr_block": "analytic",
        "agn_feii_block": "none",
        "agn_attenuation_block": "none",
    }
    active = agn_active_param_set(cfg)
    assert active != ALL_AGN_PARAMS
    assert {"agn_nlr_cf", "agn_blr_cf"} <= active


def test_unified_agn_nlr_blr_additive(synthetic_ssp_wide):
    """A unified disc+torus+NLR+BLR builds in one call and the lines add linearly.

    Reproduces the Synthesizer UnifiedAGN decomposition through the grammar
    (analytic line path, no grids needed): independent ``nlr`` and ``blr`` blocks
    sum onto the disc+torus continuum additively.
    """

    def sed(nlr_block, blr_block):
        m = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": Fixed(DEFAULT),
            },
            agn={
                "type": "composable",
                "disc": {"type": "kubota_done"},
                "torus": {"type": "simple"},
                "nlr": {"type": nlr_block},
                "blr": {"type": blr_block},
                "agn_log_lbol": Fixed(12.0),
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.05),
        )
        return np.asarray(m.predict_state({}).derived["sed_agn"])

    base = sed("none", "none")
    nlr = sed("analytic", "none") - base
    blr = sed("none", "analytic") - base
    combined = sed("analytic", "analytic") - base
    denom = np.max(np.abs(combined)) + 1e-300
    assert np.max(np.abs(combined - (nlr + blr))) / denom < 1e-6


def test_unified_agn_type1_type2_masking(synthetic_ssp_wide):
    """Composable unified AGN: disc+BLR obscured edge-on, NLR stays isotropic.

    The gray Type-1/2 visibility mask (runner Stage 4.5) obscures the anisotropic
    central engine (disc + BLR) as the sightline grazes the torus, while the
    spatially-extended NLR — illuminated by the intrinsic bolometric — is
    inclination-independent. This is the physics-correct behavior, reproduced
    through the composable grammar with independent nlr and blr slots.
    """

    def sed(nlr_block, blr_block, cos_inc):
        m = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            agn={
                "type": "composable",
                "disc": {"type": "multicolor"},
                # agn_theta_torus is the gray mask's own opening angle, read at
                # the runner's torus stage; R34 gave it its torus owner, so the
                # composable spelling nests it (agn_cos_inc stays shared -- the
                # sightline is read by disc, torus and atten alike).
                "torus": {"type": "simple", "agn_theta_torus": Fixed(45.0)},
                "nlr": {"type": nlr_block},
                "blr": {"type": blr_block},
                "agn_log_lbol": Fixed(12.0),
                "agn_cos_inc": Fixed(cos_inc),
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.05),
        )
        return np.asarray(m.predict_state({}).derived["sed_agn"])

    base_f, base_e = sed("none", "none", 0.99), sed("none", "none", 0.05)
    nlr_f, nlr_e = sed("analytic", "none", 0.99), sed("analytic", "none", 0.05)
    blr_f, blr_e = sed("none", "analytic", 0.99), sed("none", "analytic", 0.05)

    # NLR contribution is isotropic (edge-on == face-on).
    nlr_ratio = (nlr_e - base_e).sum() / (nlr_f - base_f).sum()
    assert abs(nlr_ratio - 1.0) < 1e-3
    # BLR contribution is obscured edge-on.
    blr_ratio = (blr_e - base_e).sum() / (blr_f - base_f).sum()
    assert blr_ratio < 0.05
    # The disc continuum is likewise obscured edge-on (sum over the SED drops).
    assert base_e.sum() < base_f.sum()


def test_unified_agn_recipe_structure():
    """``recipes.unified_agn()`` is the composable disc+torus+NLR+BLR unified model."""
    import tengri

    agn = tengri.recipes.unified_agn()["agn"]
    assert agn["type"] == "composable"
    # Faithful Synthesizer UnifiedAGN reproduction (grid-backed line regions).
    assert agn["disc"]["type"] == "kubota_done"
    assert agn["torus"]["type"] == "simple"
    assert agn["nlr"]["type"] == "synthesizer_spectra"  # independent NLR block
    assert agn["blr"]["type"] == "synthesizer_spectra"  # independent BLR block
    # Parametric luminosity mode: the two scaling knobs are pinned fixed.
    assert isinstance(agn["lum_ratio"], Fixed)
    assert isinstance(agn["ir_frac"], Fixed)
    # Mixed group (explicit lum_ratio/ir_frac entries alongside the wildcard)
    # spells the wildcard "other_params", not "all_params".
    assert agn["other_params"] is FREE


def test_composable_wildcard_frees_only_active_params(synthetic_ssp_wide):
    """Spec-level: ``agn={'all_params': FREE}`` frees exactly the active set (no grids)."""
    cfg = {
        "agn_model": "composable",
        "agn_disc_block": "multicolor",
        "agn_torus_block": "two_temperature",
        "agn_nlr_block": "analytic",
        "agn_blr_block": "none",
        "agn_feii_block": "none",
        "agn_attenuation_block": "none",
    }
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
        },
        agn={
            "type": "composable",
            "disc": {"type": "multicolor"},
            "torus": {"type": "two_temperature"},
            "nlr": {"type": "analytic"},
            "blr": {"type": "none"},
            "all_params": FREE,
        },
        redshift=Fixed(0.05),
    )
    free_agn = {p for p in model.spec.free_params if p.startswith("agn")}
    # agn_ebv_disc (Task 16, item 2): a category-wide companion read
    # (compose_l_nu reddens EVERY disc block's own continuum with this
    # Prevot-SMC screen at the runner stage, blocks/runner.py) that no
    # individual disc TYPE's own signature names, so it is invisible to
    # agn_active_param_set's AGN_BLOCK_CONSUMES-only union -- but IS
    # reachable here because the disc sub-dict states no 'all_params' of
    # its own, so it inherits the top-level wildcard, and its OWN
    # sub-block scope (_agn_subblock_declared_params, groups.py's
    # _AGN_CATEGORY_WIDE_COMPANION_PARAMS) correctly includes it. Not a
    # gap in agn_active_param_set's OWN contract (it unions per-block
    # CONSUMES entries for params partitioned to the top-level "agn"
    # group; agn_ebv_disc is partitioned to "agn.disc" instead) -- see
    # test_agn_wildcard_measured_liveness.py for the measured liveness
    # proof.
    assert free_agn == agn_active_param_set(cfg) | {"agn_ebv_disc"}


def test_all_fixed_wildcard_frees_nothing_and_keeps_old_defaults(synthetic_ssp_wide):
    """Back-compat: ``'all_params': Fixed(DEFAULT)`` yields no free AGN params.

    Values stay at their historic defaults.
    """
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
        },
        agn={
            "type": "composable",
            "disc": {"type": "multicolor"},
            "torus": {"type": "two_temperature"},
            "lines": {"type": "nlr"},
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.05),
    )
    assert not any(p.startswith("agn") for p in model.spec.free_params)
    # Historic fixed defaults preserved.
    for name, expected in (("agn_log_lbol", 10.0), ("agn_log_mbh", 7.0)):
        d = model.spec.get_distribution(name)
        assert d.is_fixed and abs(float(d.default) - expected) < 1e-9


# ── gate-2: every free AGN param in the flagship recipe moves predict() ──────

_SSP = Path(__file__).resolve().parents[1].parent / "data"

# Actually bare-stellar now (#1579). The name said ``_BARE`` and the path was
# the wNE grid -- nebular baked into the templates -- while the recipe under
# test, ``agn_panchromatic()``, selects Cue and documents "SSP requirement:
# bare-stellar". The pairing raises ``CueWNESSPError`` in production; it ran
# here only because conftest set TENGRI_ALLOW_WNE_CUE=1 suite-wide for the
# synthetic fixtures, which also disabled the metadata check.
_BARE = _SSP / "fsps_prsc_miles_chabrier.h5"


@pytest.mark.skipif(not _BARE.exists(), reason="SSP grid not present (CI has no data/ssp_*.h5)")
def test_agn_panchromatic_free_params_all_move_predict(real_ssp_only):
    """No-op guard: each free AGN param in agn_panchromatic changes predict().

    Requires the real SSP grid: the synthetic #613 fixture is a smooth,
    featureless continuum on which the AGN contribution is swamped, so the
    per-param ``predict`` deltas fall below the no-op threshold. Skips on CI
    (synthetic-only) via ``real_ssp_only``.
    """
    from tengri import load_ssp_data, recipes
    from tengri.observation import Observation, Photometry
    from tengri.observation.photometry import FilterCurve

    ssp = load_ssp_data(str(_BARE))

    def _tophat(c, frac=0.16, n=40):
        w = jnp.linspace(c * (1 - frac), c * (1 + frac), n)
        return FilterCurve(
            wave=w, trans=jnp.sin(jnp.linspace(0, jnp.pi, n)) * 0.6, name=f"b{int(c)}"
        )

    obs = Observation(
        photometry=Photometry(
            filters=tuple(_tophat(c) for c in (1000.0, 3000.0, 6000.0, 2e4, 1e5, 1e7, 1e9))
        )
    )

    spec = SEDModel.build(ssp_data=ssp, **recipes.agn_panchromatic()).spec
    free_agn = sorted(p for p in spec.free_params if p.startswith("agn"))
    assert free_agn, "recipe must free at least some AGN params"

    bounds = {pd.name: pd.prior.bounds for pd in _AGN_PARAMS}

    def predict(name, value):
        agn = {
            "type": "composable",
            "disc": {"type": "multicolor"},
            "torus": {"type": "skirtor"},
            "nlr": {"type": "analytic"},
            "blr": {"type": "none"},
            "feii": {"type": "none"},
            # polar_dust, not none: the recipe frees agn_polar_ebv, and the
            # polar params only act with the polar-dust atten stage on —
            # atten 'none' would make them no-ops BY CONSTRUCTION here and
            # invalidate the no-op guard for exactly those params.
            "atten": {"type": "polar_dust"},
            "agn_log_lbol": Fixed(12.0),
            "all_params": Fixed(DEFAULT),
        }
        if name is not None:
            # #1980: sub-block-owned params must nest under their PARTITION
            # owner — the block the flat-placement error names. The consumes
            # map is NOT the placement authority: params consumed by several
            # blocks (cos_inc, polar_ebv, polar_beta) are grammar-ACCEPTED
            # under any consuming block, but only the partition owner's copy
            # reaches the spec — the others are silently dropped (measured:
            # torus-nested polar_ebv left the spec at its 0.03 default; that
            # pre-existing silent drop is its own issue). So ask the grammar:
            # try flat, and nest wherever its refusal points.
            if name in AGN_SHARED_PARAMS:
                agn[name] = Fixed(value)
            else:
                from tengri.parameters.groups import parse_groups

                short = name.removeprefix("agn_")
                try:
                    parse_groups(
                        agn={**agn, name: Fixed(value)},
                        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
                        redshift=Fixed(0.05),
                    )
                except ValueError as exc:
                    owner = re.search(r"'agn\.(\w+)' parameter", str(exc))
                    assert owner, f"unexpected flat-placement refusal for {name}: {exc}"
                    sub = owner.group(1)
                    agn[sub] = {**agn[sub], short: Fixed(value)}
                else:
                    # The grammar accepts it flat (a shared-style param).
                    agn[name] = Fixed(value)
        m = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_diff": Fixed(0.3),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014_cigale"},
            agn=agn,
            radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}},
            xray={"type": "simple"},
            redshift=Fixed(0.05),
        )
        return np.asarray(m.predict_photometry({}))

    f0 = predict(None, None)
    norm = max(np.max(np.abs(f0)), 1e-300)
    no_ops = []
    for name in free_agn:
        lo, hi = bounds[name]
        a, b = lo + 0.3 * (hi - lo), lo + 0.7 * (hi - lo)
        rel = np.max(np.abs(predict(name, b) - predict(name, a))) / norm
        if rel <= 1e-6:
            no_ops.append(name)
    assert not no_ops, f"recipe frees no-op AGN params (no effect on predict): {no_ops}"


def test_every_consumed_name_is_reachable_from_some_wildcard():
    """Every name a block reads can be freed by SOME wildcard on a build that
    selects that block (R36).

    Derived from ``AGN_BLOCK_CONSUMES`` -- the record of what each block
    actually reads -- and checked against the partition, so it is the reads
    that drive the expectation and never the ownership table it tests. A read
    the grammar offers no way to fit is the defect this catches.

    It caught one: ``('disc', 'schartmann2005_skirtor_atten')`` applies
    SKIRTOR's own geometry to its disc continuum, so its CONSUMES entry names
    ``agn_oa_skirtor``/``p``/``q``/``tau_skirtor`` -- all owned by ``agn.torus``
    and all measured live (grads 6.9e-20 to 1.9e-19 with no torus selected).
    With torus absent, no wildcard reached any of them: the disc's own frees
    only what it owns, and the shared agn-level one cannot reach a
    sub-block-owned name. Fixed by a cross-category companion, so the reading
    block's wildcard claims such a name exactly while the owning category's
    selected block does not read it itself.
    """
    from tengri.parameters.groups import (
        _AGN_CONSUMES_CATEGORY,
        _agn_param_group,
        _agn_subblock_declared_params,
    )

    grammar_of = {v: k for k, v in _AGN_CONSUMES_CATEGORY.items()}
    unreachable = []
    for (consumes_cat, block_type), consumed in AGN_BLOCK_CONSUMES.items():
        grammar_cat = grammar_of.get(consumes_cat, consumes_cat)
        # The build under test selects this block and nothing else, which is
        # the configuration in which its own wildcard has to suffice.
        own = _agn_subblock_declared_params(
            grammar_cat, block_type, selection={grammar_cat: block_type}
        )
        for name in sorted(consumed):
            if name in own or _agn_param_group(name) == "agn":
                continue
            unreachable.append(
                f"({consumes_cat!r}, {block_type!r}) reads {name!r}, owned by "
                f"{_agn_param_group(name)!r}: neither this block's wildcard nor "
                f"the shared agn-level one frees it"
            )
    assert not unreachable, "\n".join(unreachable)
