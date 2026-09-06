# SPDX-License-Identifier: BSD-3-Clause
"""Which ``agn_*`` parameters each AGN block / monolithic model consumes.

The composable AGN component declares the *full* ``agn_*`` superset so any
registered block can run without missing keys (see
:meth:`AGNSEDComponent.declared_parameters`). That superset is ~50 parameters,
most of which belong to inactive blocks for any given configuration. A naive
group wildcard ``agn={'*': FREE}`` would therefore free dozens of parameters
that do not affect ``predict()`` for the selected blocks: unconstrained no-op
nuisance dimensions in a fit.

This module records, per block and per monolithic model, the parameters that
*actually* move the AGN SED when that block/model is active.
:func:`agn_active_param_set` unions the consumed sets of the active block
selection (plus the always-active normalization knobs in
:data:`AGN_SHARED_PARAMS`); the builder grammar uses it to scope a group
wildcard to exactly those parameters.

Provenance
----------
The sets were determined empirically: each block was activated in isolation
(downstream blocks driven by a multicolor disc so ``l5100_disc`` is non-zero)
and every declared ``agn_*`` parameter was perturbed across its prior; a
parameter is "consumed" if it changed ``sed_agn`` by more than a relative
``1e-6``. Driver-disc parameters (``agn_log_mbh``/``agn_log_ledd``/
``agn_a_spin``) were subtracted from the non-disc sets so each block lists only
its *own* parameters; the union over an active config (which always includes a
disc block) reconstructs the full live set. The contract test
``tests/contract/test_agn_block_consumes.py`` re-runs the scan and asserts the
empirically live set is a subset of the scoped active set, so this table cannot
silently drift out of date.

Blocks/models genuinely absent from this table fall back to the full
superset via :func:`agn_active_param_set`, so the wildcard over-frees (never
under-frees) for an unregistered name -- a safe, documented degradation
rather than a silent exclusion. Fix round 2: this sentence previously listed
``cat3d_wind`` and "the GRAHSP line/feii blocks" here too, but both are
registered below (``cat3d_wind``/``cat3d_wind_lowfwd`` since fix round 1 --
their grids are tracked in CI, so the original "grid absent from CI" rationale
no longer held; the GRAHSP ``nlr``/``blr``/``feii`` entries were registered
independently of this task). Task 16 similarly registered ``("disc",
"slone_netzer")``, previously omitted on the same "grid absent from CI"
rationale -- also stale (measured, this checkout).
"""

from __future__ import annotations

from tengri.components.agn._params import PARAMS

#: All declared AGN parameter names (the full superset).
ALL_AGN_PARAMS: frozenset[str] = frozenset(pd.name for pd in PARAMS)

#: Normalization knobs active whenever any AGN model/block is on.
AGN_SHARED_PARAMS: frozenset[str] = frozenset({"agn_lum_ratio", "agn_log_lbol"})

#: Composable block -> the agn_* params that block itself consumes.
AGN_BLOCK_CONSUMES: dict[tuple[str, str], frozenset[str]] = {
    ("disc", "adaf"): frozenset(
        {"agn_log_mbh", "agn_adaf_alpha", "agn_adaf_beta", "agn_adaf_delta"}
    ),
    ("disc", "adaf_lopez2024"): frozenset({"agn_cigale_disk_delta"}),
    ("disc", "grahsp_sbpl"): frozenset(
        {
            "agn_grahsp_cutoff_nm",
            "agn_grahsp_l5100",
            "agn_grahsp_plbendloc_nm",
            "agn_grahsp_plbendwidth",
            "agn_grahsp_plslope",
            "agn_grahsp_uvslope",
        }
    ),
    ("disc", "kubota_done"): frozenset(
        {
            "agn_a_spin",
            "agn_cos_inc",
            "agn_f_hard",
            "agn_gamma_hard",
            "agn_gamma_warm",
            "agn_kt_hot",
            "agn_kt_warm",
            # agn_log_ledd is deliberately not consumed here (#846): for this
            # disc block the Eddington ratio is derived from agn_log_lbol, so
            # agn_log_ledd does not move the SED. The parameter itself is
            # still declared (and consumed by the kd18/unified disc paths).
            "agn_log_mbh",
            "agn_r_warm_ratio",
        }
    ),
    ("disc", "multicolor"): frozenset({"agn_a_spin", "agn_log_mbh"}),
    ("disc", "powerlaw"): frozenset({"agn_alpha"}),
    ("disc", "qsogen"): frozenset(),
    ("disc", "relagn"): frozenset({"agn_log_mbh", "agn_log_mdot", "agn_astar", "agn_cos_inc"}),
    ("disc", "richards2006"): frozenset(),
    ("disc", "schartmann2005"): frozenset({"agn_cigale_disk_delta"}),
    ("disc", "schartmann2005_skirtor_atten"): frozenset(
        {
            "agn_cigale_disk_delta",
            "agn_cos_inc",
            "agn_oa_skirtor",
            "agn_p_skirtor",
            "agn_q_skirtor",
            "agn_tau_skirtor",
        }
    ),
    ("disc", "skirtor"): frozenset({"agn_cigale_disk_delta"}),
    # Task 16 (item 3): previously omitted with "requires a data grid absent
    # from CI" -- that no longer holds (the grid this checkout ships builds
    # and differentiates it fine); measured like every other disc entry:
    # agn_log_ledd is dead (jax.grad exactly 0.0 at every sampled point,
    # same #846-shaped degeneracy as kubota_done/multicolor above -- the
    # Eddington ratio does not independently move this disc's SED shape),
    # agn_log_mbh is live (tiny but consistently nonzero, ~1e-15).
    ("disc", "slone_netzer"): frozenset({"agn_log_mbh"}),
    ("torus", "grahsp"): frozenset(
        {
            "agn_grahsp_cool_lam_um",
            "agn_grahsp_cool_width",
            "agn_grahsp_fcov",
            "agn_grahsp_hot_fcov",
            "agn_grahsp_hot_lam_um",
            "agn_grahsp_hot_width",
            "agn_grahsp_si",
        }
    ),
    # The gray Type-1/2 visibility mask (runner Stage 4.5) applies to the
    # physical-decomposition tori, so agn_cos_inc + agn_theta_torus move predict
    # for them (empirically ~7e-3 and ~8e-3 on a multicolor+<torus>+nlr_blr
    # config) and must be freeable under agn={'*': FREE}.
    ("torus", "nenkova"): frozenset(
        {"agn_ir_frac", "agn_tau", "agn_torus_frac", "agn_cos_inc", "agn_theta_torus"}
    ),
    # Task 16 (item 3): agn_torus_frac was missing -- nenkova_agnfitter_torus_
    # block's own signature has always read it (its own covering fraction,
    # like every other composable torus block); this entry previously listed
    # only the generic gray Type-1/2 mask reads (agn_cos_inc/agn_theta_torus).
    ("torus", "nenkova_agnfitter"): frozenset(
        {"agn_cos_inc", "agn_theta_torus", "agn_torus_frac"}
    ),
    # CAT3D-Wind (Hönig & Kishimoto 2017): neither in TORUS_SCREEN_PARAMS
    # (torus_screen.py) nor _SELF_CONTAINED_TORI (runner.py), so both receive
    # the same generic gray mask as nenkova/silva04/two_temperature above
    # (empirically confirmed: perturbing agn_theta_torus across its prior
    # moves sed_agn by ~1.5e-3 for both, isolated with a multicolor disc).
    # Fix round 1 (CRITICAL): both entries were previously absent -- the
    # historical "grid absent from CI" rationale for cat3d_wind no longer
    # holds (data/cat3d_wind_torus_grid.h5 is tracked and negated in
    # .gitignore) -- so agn={'all_params': FREE} with either torus silently
    # fell back to the full ~96-name superset (89 of them foreign to the
    # block). See test_cat3d_wind_family_top_level_wildcard_frees_exact_consumed_set.
    ("torus", "cat3d_wind"): frozenset(
        {"agn_cos_inc", "agn_theta_torus", "agn_a_cat3d", "agn_fwd_cat3d", "agn_torus_frac"}
    ),
    ("torus", "cat3d_wind_lowfwd"): frozenset(
        {
            "agn_cos_inc",
            "agn_theta_torus",
            "agn_a_cat3d_lowfwd",
            "agn_fwd_cat3d_lowfwd",
            "agn_torus_frac",
        }
    ),
    ("torus", "nenkova_agnfitter_2p"): frozenset(
        {"agn_cos_inc", "agn_theta_torus", "agn_oa_nenkova", "agn_torus_frac"}
    ),
    ("torus", "nenkova_agnfitter_3p"): frozenset(
        {
            "agn_cos_inc",
            "agn_theta_torus",
            "agn_oa_nenkova",
            "agn_tv_nenkova",
            "agn_torus_frac",
        }
    ),
    ("torus", "qsogen"): frozenset(),
    # Task 16 (item 3): silva04's own axis (agn_log_nh_silva) and covering
    # fraction (agn_torus_frac) were missing -- this entry previously listed
    # ONLY the generic gray Type-1/2 mask params (agn_cos_inc/agn_theta_torus,
    # a runner Stage-4.5 read every physical-decomposition torus shares), so
    # the top-level agn={'all_params': FREE} wildcard never reached silva04's
    # own two declared parameters at all. Invisible until
    # _agn_subblock_declared_params started sourcing from this table (Task 16,
    # item 1): it used to fall back to raw signature introspection, which
    # DID find them.
    ("torus", "silva04"): frozenset(
        {"agn_cos_inc", "agn_theta_torus", "agn_log_nh_silva", "agn_torus_frac"}
    ),
    ("torus", "simple"): frozenset(
        {"agn_T_torus", "agn_ir_frac", "agn_torus_frac", "agn_cos_inc", "agn_theta_torus"}
    ),
    ("torus", "fritz"): frozenset(
        {
            "agn_fritz_r_ratio",
            "agn_fritz_tau",
            "agn_fritz_beta",
            "agn_fritz_gamma",
            "agn_fritz_oa",
            "agn_fritz_psy",
            "agn_torus_frac",
        }
    ),
    ("torus", "skirtor"): frozenset(
        {
            "agn_cos_inc",
            "agn_ir_frac",
            "agn_oa_skirtor",
            "agn_p_skirtor",
            # R22 (task13 fix-round-1): the polar_dust knobs used to be listed
            # here too -- skirtor_torus_block bundled its OWN Casey-2012 polar
            # graybody (active by default at agn_polar_ebv=0.03), a SECOND
            # polar-dust mechanism alongside the standalone ``polar_dust``
            # attenuation block, so torus=skirtor + atten=polar_dust screened
            # the disc twice. There is now exactly ONE mechanism (the
            # standalone atten block, see ("attenuation", "polar_dust")
            # below); this torus emits only the thermal SKIRTOR template and
            # reads no agn_polar_* name.
            "agn_q_skirtor",
            "agn_tau_skirtor",
            "agn_torus_frac",
            # Task 14/16 (F4): agn_radius_ratio (the SKIRTOR grid's third
            # axis) was missing here -- skirtor_torus_block's own signature
            # has always read it (unlike SKIRTORTorus, the class, whose
            # equivalent bug -- hardcoded 20.0, never passed through -- Task
            # 14 fixed); measured live via
            # test_skirtor_torus_wiring.py::test_skirtor_radius_ratio_live_on_composable_path.
            "agn_radius_ratio",
        }
    ),
    ("torus", "skirtor_agnfitter"): frozenset(
        {
            "agn_oa_skirtor",
            "agn_incl_skirtor",
            "agn_tv_skirtor",
            "agn_torus_frac",
        }
    ),
    ("torus", "skirtor_agnfitter_1p"): frozenset({"agn_incl_skirtor", "agn_torus_frac"}),
    ("torus", "skirtor_agnfitter_2p"): frozenset(
        {"agn_oa_skirtor", "agn_incl_skirtor", "agn_torus_frac"}
    ),
    ("torus", "two_temperature"): frozenset(
        {
            "agn_T_hot",
            "agn_T_warm",
            "agn_ir_frac",
            "agn_frac_hot",
            "agn_torus_frac",
            "agn_cos_inc",
            "agn_theta_torus",
        }
    ),
    ("nlr", "analytic"): frozenset({"agn_nlr_cf", "agn_nlr_line_efficiency"}),
    ("nlr", "synthesizer"): frozenset({"agn_nlr_cf"}),
    ("nlr", "synthesizer_spectra"): frozenset({"agn_nlr_cf"}),
    ("nlr", "grahsp"): frozenset({"agn_grahsp_a_lines", "agn_grahsp_linewidth_kms"}),
    ("blr", "analytic"): frozenset(
        {
            "agn_blr_cf",
            "agn_blr_line_efficiency",
            "agn_fe2_strength",
        }
    ),
    ("blr", "synthesizer"): frozenset({"agn_blr_cf"}),
    ("blr", "synthesizer_spectra"): frozenset({"agn_blr_cf"}),
    ("blr", "grahsp"): frozenset({"agn_grahsp_a_lines", "agn_grahsp_linewidth_kms"}),
    ("blr", "qsogen"): frozenset(),
    ("feii", "boroson_green"): frozenset(
        {
            "agn_fe2_strength",
            "agn_blr_cf",
            "agn_blr_line_efficiency",
        }
    ),
    ("feii", "grahsp"): frozenset({"agn_grahsp_a_feii", "agn_grahsp_a_lines"}),
    # QSOgen Balmer continuum (Temple+2021), registered in #1488 but never
    # added here, so the block's own enabling knob was invisible to the
    # top-level ``agn={'all_params': FREE}`` wildcard scope: selectable,
    # buildable, and (until a caller named ``agn_bcnorm`` explicitly)
    # permanently pinned at its Fixed(DEFAULT)=0.0 (issue #2175).
    ("feii", "qsogen_balmer"): frozenset({"agn_bcnorm"}),
    ("attenuation", "grahsp_biatten"): frozenset({"agn_grahsp_ebv", "agn_grahsp_ebv_agn"}),
    ("attenuation", "polar_dust"): frozenset(
        {
            "agn_cos_inc",
            "agn_polar_beta",
            "agn_polar_ebv",
            "agn_polar_oa",
            # Re-emission temperature (task13): the standalone polar_dust
            # attenuation block's graybody term reads agn_polar_T via
            # polar_dust_reemission_lnu (blocks/atten.py), called
            # unconditionally by the runner whenever this block is
            # selected. Previously omitted because a name mismatch
            # (agn_polar_temperature vs the declared agn_polar_T) made the
            # parameter empirically dead.
            "agn_polar_T",
        }
    ),
    ("attenuation", "qsogen_smc"): frozenset(),
    ("attenuation", "smc_prevot"): frozenset({"agn_attenuation_ebv"}),
    # Task 16 (item 3): previously missing entirely -- the top-level wildcard
    # silently fell back to the full ~50-name superset whenever atten='qsogen'
    # (Temple+2021's own quasar extinction curve, alternates.py
    # qsogen_quasar_ext_block) was selected. Its signature reads
    # agn_attenuation_ebv (like smc_prevot above), nothing else.
    ("attenuation", "qsogen"): frozenset({"agn_attenuation_ebv"}),
}

#: Monolithic (non-composable) AGN model -> the agn_* params it consumes.
AGN_MODEL_CONSUMES: dict[str, frozenset[str]] = {
    "adaf": frozenset({"agn_ir_frac", "agn_log_ledd", "agn_log_mbh", "agn_torus_frac"}),
    "kubota_done": frozenset(
        {
            "agn_a_spin",
            "agn_ir_frac",
            "agn_log_ledd",
            "agn_log_mbh",
            "agn_torus_frac",
        }
    ),
    "kubota_done_full": frozenset(
        {
            "agn_a_spin",
            "agn_cos_inc",
            "agn_f_hard",
            "agn_ir_frac",
            "agn_gamma_hard",
            "agn_gamma_warm",
            "agn_kt_hot",
            "agn_kt_warm",
            "agn_log_ledd",
            "agn_log_mbh",
            "agn_r_warm_ratio",
            "agn_torus_frac",
        }
    ),
    "multicolor_agn": frozenset(
        {
            "agn_a_spin",
            "agn_ir_frac",
            "agn_log_ledd",
            "agn_log_mbh",
            "agn_torus_frac",
        }
    ),
    "qsogen": frozenset(),
    "relagn": frozenset({"agn_cos_inc", "agn_ir_frac", "agn_log_mbh", "agn_torus_frac"}),
    "richards2006": frozenset(),
    "silva04": frozenset({"agn_ir_frac", "agn_torus_frac"}),
    "skirtor": frozenset(
        {
            "agn_cos_inc",
            "agn_ir_frac",
            "agn_oa_skirtor",
            "agn_p_skirtor",
            "agn_q_skirtor",
            "agn_tau_skirtor",
            "agn_torus_frac",
        }
    ),
    "unified_nlr_blr": frozenset(
        {
            "agn_a_spin",
            "agn_blr_cf",
            "agn_cos_inc",
            "agn_fe2_strength",
            "agn_ir_frac",
            "agn_log_ledd",
            "agn_log_mbh",
            "agn_nlr_cf",
            "agn_polar_ebv",
            "agn_torus_frac",
        }
    ),
}

_BLOCK_SELECTOR_KWARGS: tuple[tuple[str, str], ...] = (
    ("disc", "agn_disc_block"),
    ("nlr", "agn_nlr_block"),
    ("blr", "agn_blr_block"),
    ("feii", "agn_feii_block"),
    ("torus", "agn_torus_block"),
    ("attenuation", "agn_attenuation_block"),
)

__all__ = [
    "AGN_BLOCK_CONSUMES",
    "AGN_MODEL_CONSUMES",
    "AGN_SHARED_PARAMS",
    "ALL_AGN_PARAMS",
    "agn_active_param_set",
]


def agn_active_param_set(structural_kwargs: dict) -> frozenset[str]:
    """Return the ``agn_*`` params consumed by the active AGN configuration.

    Parameters
    ----------
    structural_kwargs : dict
        Resolved structural kwargs (post ``_translate_structural``), carrying
        ``agn_model`` and, for the composable model, the per-block selectors
        ``agn_{disc,torus,lines,feii,attenuation}_block``.

    Returns
    -------
    frozenset of str
        Parameter names a group-level wildcard should free. Empty if no AGN
        model is active. Falls back to the full superset
        (:data:`ALL_AGN_PARAMS`) for any model/block missing from the tables
        (grid-gated blocks), so the wildcard over-frees rather than silently
        excluding a consumed parameter.

    Notes
    -----
    **JIT-compatible**: no, pure-Python builder-time helper.
    """
    model = structural_kwargs.get("agn_model")
    if not model:
        return frozenset()

    if model != "composable":
        consumed = AGN_MODEL_CONSUMES.get(model)
        if consumed is None:
            return ALL_AGN_PARAMS  # unknown monolithic model: safe over-free
        return AGN_SHARED_PARAMS | consumed

    active: set[str] = set(AGN_SHARED_PARAMS)
    for category, kwarg in _BLOCK_SELECTOR_KWARGS:
        block_type = structural_kwargs.get(kwarg, "none")
        if not block_type or block_type == "none":
            continue
        consumed = AGN_BLOCK_CONSUMES.get((category, block_type))
        if consumed is None:
            return ALL_AGN_PARAMS  # unknown/grid-gated block: safe over-free
        active |= consumed
    return frozenset(active)
