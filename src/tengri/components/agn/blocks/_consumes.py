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
    # R34: agn_T_max (the block's own UV cutoff temperature) was missing --
    # measured live on predict_photometry once the partition gave it a disc
    # owner, so the disc wildcard would otherwise free the slope and leave the
    # cutoff pinned.
    ("disc", "powerlaw"): frozenset({"agn_alpha", "agn_T_max"}),
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
    # and differentiates it fine). Both axis parameters are live and both are
    # listed. agn_log_ledd was briefly recorded here as dead; that reading came
    # from a baseline outside the block's own grid. The shared declared default
    # -1.0 sits above the SN12 axis [-4, -1.9586] and is clipped onto the edge
    # node, where jnp.clip makes the gradient exactly zero by construction
    # (#1586's whole subject). Re-measured at points inside the axis, jax.grad
    # of log(sum L_lambda) is 5.9e-2 at -2.0, 8.8e-1 at -2.5 and 1.2e-1 at
    # -3.5, with a 754% relative SED change between -3.5 and -2.0 -- so this is
    # nothing like the #846 degeneracy of kubota_done/multicolor above, where
    # the Eddington ratio is genuinely derived from agn_log_lbol. Pinned by
    # tests/regression/agn/test_issue_1586_grid_support.py, which measures the
    # gradient rather than restating this table.
    ("disc", "slone_netzer"): frozenset({"agn_log_mbh", "agn_log_ledd"}),
    # The two AGNfitter-rX Kubota & Done grids. Registered here rather than
    # left to the signature fallback (R37 round 2): measured on the scoping
    # module's own fixtures, worst-of-five-seeds |grad| on
    # predict_photometry, every name in each signature is live --
    # agn_log_mbh 1.6e-15, agn_log_ledd 1.9e-15 for both, and
    # agn_gamma_warm 1.8e-16 for the warm-index variant, which is the axis
    # that distinguishes it. agn_log_lbol is omitted from both, as from every
    # other entry: it is in AGN_SHARED_PARAMS and always active.
    ("disc", "kd18_agnfitter"): frozenset({"agn_log_mbh", "agn_log_ledd"}),
    ("disc", "kd18_agnfitter_warmindex"): frozenset(
        {"agn_log_mbh", "agn_log_ledd", "agn_gamma_warm"}
    ),
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
    # R34: agn_theta_torus (the runner's gray Type-1/2 visibility mask, the
    # same read the nenkova/silva04/cat3d entries already record) was missing
    # from all three AGNfitter-rX SKIRTOR variants -- measured live on
    # predict_photometry for each once the partition gave it a torus owner.
    ("torus", "skirtor_agnfitter"): frozenset(
        {
            "agn_oa_skirtor",
            "agn_incl_skirtor",
            "agn_tv_skirtor",
            "agn_torus_frac",
            "agn_theta_torus",
        }
    ),
    ("torus", "skirtor_agnfitter_1p"): frozenset(
        {"agn_incl_skirtor", "agn_torus_frac", "agn_theta_torus"}
    ),
    ("torus", "skirtor_agnfitter_2p"): frozenset(
        {"agn_oa_skirtor", "agn_incl_skirtor", "agn_torus_frac", "agn_theta_torus"}
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
    # agn_nlr_fwhm_kms is deliberately absent, and the two measurements that
    # disagree about it are both recorded here. This block does pass the line
    # width through, and a jax.grad on predict_photometry is not exactly zero
    # -- but the quantity this table records is "moves the SED by more than a
    # relative 1e-6" (see Provenance above), and a line width at fixed line
    # luminosity redistributes flux inside a line that a broadband filter
    # integrates over: measured <= 1e-6 relative on the agn_panchromatic
    # recipe's own filters, which is why
    # test_agn_panchromatic_free_params_all_move_predict calls it a no-op.
    # Listing it would make a recipe free a dimension no fit can constrain.
    ("nlr", "analytic"): frozenset({"agn_nlr_cf", "agn_nlr_line_efficiency"}),
    ("nlr", "synthesizer"): frozenset({"agn_nlr_cf"}),
    ("nlr", "synthesizer_spectra"): frozenset({"agn_nlr_cf"}),
    ("nlr", "grahsp"): frozenset({"agn_grahsp_a_lines", "agn_grahsp_linewidth_kms"}),
    # Cue's five photoionization axes plus the covering fraction, all measured
    # live (agn_nlr_cf 8.5e-15, agn_nlr_alpha_pl 2.9e-15, agn_nlr_logZ
    # 1.2e-15, agn_nlr_logn 3.3e-16, agn_nlr_logU 1.1e-16).
    # agn_nlr_fwhm_kms is excluded for the same reason as ('nlr', 'analytic')
    # below: a line width at fixed line luminosity redistributes flux inside a
    # line a broadband filter integrates over, so it falls under this table's
    # own "> 1e-6 relative" criterion (its grad here, 1.5e-18, is the smallest
    # of the seven by three orders of magnitude).
    ("nlr", "cue"): frozenset(
        {
            "agn_nlr_cf",
            "agn_nlr_alpha_pl",
            "agn_nlr_logU",
            "agn_nlr_logZ",
            "agn_nlr_logn",
        }
    ),
    # R34 partitioned the six Feltre grid axes to "agn.nlr", which put them in
    # this block's own wildcard scope for the first time. Without an entry here
    # the scope came from raw signature introspection, which also swept in
    # agn_nlr_xi_d -- measured exactly dead (grad 0.0 at every one of five
    # sampled points, 0.0 relative change across the declared [0.1, 0.5]): the
    # shipped Feltre grid carries a single dust-to-metal node, so interpolation
    # along that axis is constant and the gradient is zero by construction, not
    # by underflow. Listing the six the block does read keeps the wildcard off
    # a dimension a fit cannot move.
    ("nlr", "feltre"): frozenset(
        {
            "agn_nlr_cf",
            "agn_nlr_fwhm_kms",
            "agn_nlr_alpha_pl",
            "agn_nlr_logU",
            "agn_nlr_logn",
            "agn_nlr_logZ",
        }
    ),
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
    # R34: this block's own reddening knob, agn_ebv, was recorded as reading
    # nothing at all -- measured live on predict_photometry once the partition
    # gave agn_ebv its atten owner. It is NOT agn_attenuation_ebv, the separate
    # E(B-V) the smc_prevot/qsogen blocks apply (R30).
    ("attenuation", "qsogen_smc"): frozenset({"agn_ebv"}),
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
    "monolithic_agn_declared_params",
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


def monolithic_agn_declared_params(model: str) -> frozenset[str]:
    """The ``agn_*`` names a non-composable AGN model declares.

    A monolithic model has no sub-block to nest a parameter under: its declared
    parameters ARE the ``agn`` top level, so this is the set the key validator
    accepts there (R27). Derived, never hand-listed:

    * a **preset** name routes through the composable runner with fixed block
      selectors, so it declares :data:`AGN_SHARED_PARAMS`, whatever its selected
      blocks declare in :data:`AGN_BLOCK_CONSUMES`, and its own
      :data:`AGN_MODEL_CONSUMES` entry when it has one;
    * a **self-contained** name resolves to its own forward function, so it
      declares the ``agn_*`` parameters in that function's signature plus the
      shared normalization knobs.

    Parameters
    ----------
    model : str
        A non-composable AGN model name (see
        :func:`~tengri.components.agn.unified.monolithic_agn_model_names`).

    Returns
    -------
    frozenset of str
        Fully prefixed ``agn_*`` names. Empty for an unrecognized model.

    Notes
    -----
    **JIT-compatible**: no, pure-Python builder-time helper.

    A ``**kwargs`` catch-all in a self-contained forward function is NOT a
    declaration: ``skirtor_stalevski`` swallows ``agn_torus_frac`` that way and
    pins ``frac_agn=1.0`` internally, so the parameter is exactly inert there
    (measured: 0.0 relative SED change from 0.05 to 0.95, against 19.7 for
    ``agn_oa_skirtor`` on the same call). Accepting a swallowed name would free
    a dimension the model cannot see.
    """
    import inspect

    from tengri.components.agn.unified import (
        _AGN_PRESETS,
        _SELF_CONTAINED_AGN_MODELS,
        _resolve_monolithic_model,
    )

    if model in _SELF_CONTAINED_AGN_MODELS:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            fn = _resolve_monolithic_model(model)
        if fn is None:  # pragma: no cover - table and resolver share one source
            return frozenset()
        # Intersected with ALL_AGN_PARAMS so a structural selector that happens
        # to carry the prefix (grahsp's ``agn_type``, which chooses a variant
        # rather than being fitted) is not offered as a fittable parameter.
        signature_names = {
            name for name in inspect.signature(fn).parameters if name.startswith("agn_")
        } & ALL_AGN_PARAMS
        return frozenset(AGN_SHARED_PARAMS | signature_names)

    preset = _AGN_PRESETS.get(model)
    if preset is None:
        return frozenset()

    declared = set(AGN_SHARED_PARAMS) | set(AGN_MODEL_CONSUMES.get(model, ()))
    for category, kwarg in _BLOCK_SELECTOR_KWARGS:
        block_type = preset.get(kwarg)
        if not block_type or block_type == "none":
            continue
        declared |= set(AGN_BLOCK_CONSUMES.get((category, block_type), ()))
    return frozenset(declared)
