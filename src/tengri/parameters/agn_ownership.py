# SPDX-License-Identifier: BSD-3-Clause
"""Which AGN sub-block owns each ``agn_*`` parameter, and what each block reads.

The composable AGN component declares one flat ``agn_*`` superset, but the
grammar presents six sub-blocks, and a wildcard written in one of them must
free that block's own physics and nothing else. Two questions decide every
such answer and both live here:

* **Who owns a name?** :data:`_AGN_PARTITION` maps each declared parameter to
  ``"agn"`` (genuinely shared across categories at runtime) or
  ``"agn.<category>"``. Completeness is a contract, not a convenience -- an
  unlisted name silently defaults to shared, and every instrument built on the
  table inherits the hole, so
  ``tests/contract/test_agn_block_consumes.py`` asserts every declared and
  every consumed name has an explicit entry (R34).
* **What does a block read?** :func:`_agn_subblock_declared_params` answers
  from ``AGN_BLOCK_CONSUMES`` where an entry exists and from the block
  function's own signature otherwise, then filters to the names this block
  owns. Reads that span two functions, or that a companion step performs on
  the block's behalf, come from :func:`_agn_subblock_companion_params`.

Extracted from ``parameters/groups.py`` (R35): one coherent unit with a single
import edge into ``components.agn.blocks``, in a file that was seven times the
repository's size limit. ``groups.py`` re-exports every name it used, so the
import paths callers and tests already use are unchanged.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping

from tengri.config.exceptions import ConfigError
from tengri.parameters.priors import Distribution, Fixed, _is_default_fixed

__all__ = [
    "agn_cross_category_claims",
    "check_agn_torus_registry_agreement",
    "is_cross_category_companion",
]

#: Partition table: agn_* param name -> group path (for sub-block routing).
#: Maps full agn_* param names to their owning group (agn, agn.disc, agn.torus, etc.)
_AGN_PARTITION = {
    # Truly shared params (no sub-block prefix; read across every category,
    # or governing the top-level AGN normalization itself, not one disc/torus
    # physics choice): agn_log_lbol (bolometric normalization every block
    # scales against), agn_cos_inc (viewing-angle masking knob every
    # physical-decomposition disc/torus reads), agn_lum_ratio (cross-block
    # luminosity-ratio normalization used by disc, torus and the qsogen/
    # richards2006/GRAHSP recipes alike).
    "agn_lum_ratio": "agn",
    "agn_log_lbol": "agn",
    "agn_cos_inc": "agn",
    # Disc physics (Task 16, item 1 addendum -- Task 5 review): these eleven
    # names were misclassified "agn" (shared) even though every one is read
    # ONLY by disc block functions (adaf/disc_precompute/disc.py, kd18_*,
    # slone_netzer.py -- grep-verified, no torus/nlr/blr/feii/atten file
    # references any of them). Misclassifying them "agn" made
    # disc={'type': T, 'all_params': FREE} (the agn.disc sub-block's OWN
    # wildcard) a structural no-op for 13 of the 14 registered disc types
    # (measured; test_kd18_agnfitter_public_grammar.py pinned the CURRENT,
    # now-fixed, no-op behavior): _agn_subblock_declared_params filters a
    # disc block's own signature down to names owned by "agn.disc", and
    # _AGN_PARTITION had no "agn.disc" entries at all.
    "agn_alpha": "agn.disc",
    "agn_log_mbh": "agn.disc",
    "agn_log_ledd": "agn.disc",
    "agn_a_spin": "agn.disc",
    "agn_f_hard": "agn.disc",
    "agn_gamma_warm": "agn.disc",
    "agn_kt_warm": "agn.disc",
    "agn_gamma_hard": "agn.disc",
    "agn_kt_hot": "agn.disc",
    "agn_r_warm_ratio": "agn.disc",
    # Disc dust obscuration (Prevot SMC; AGNfitter EBVbbb). redden_disc
    # applies it at the runner's disc stage for every disc type -- a
    # runner-level "disc category" read, not one specific disc type's own
    # parameter, but disc-owned nonetheless (declared-reads model, item 1):
    # the agn.disc wildcard should free it, same as any other disc-stage
    # read.
    "agn_ebv_disc": "agn.disc",
    # Torus
    "agn_T_torus": "agn.torus",
    "agn_T_hot": "agn.torus",
    "agn_T_warm": "agn.torus",
    "agn_frac_hot": "agn.torus",
    "agn_tau_torus": "agn.torus",
    "agn_tau": "agn.torus",  # Nenkova+2008 CLUMPY equatorial optical depth
    "agn_tau_skirtor": "agn.torus",
    "agn_p_skirtor": "agn.torus",
    "agn_q_skirtor": "agn.torus",
    "agn_oa_skirtor": "agn.torus",
    "agn_radius_ratio": "agn.torus",
    # SKIRTOR_mean_3p (AGNfitter-rX averaged) torus
    "agn_incl_skirtor": "agn.torus",
    "agn_tv_skirtor": "agn.torus",
    # CAT3D-Wind clumpy torus (Hönig & Kishimoto 2017)
    "agn_a_cat3d": "agn.torus",
    "agn_fwd_cat3d": "agn.torus",
    # CAT3D-Wind low-fwd sub-library (rows 0-209; disjoint a/fwd extents from
    # the pair above)
    "agn_a_cat3d_lowfwd": "agn.torus",
    "agn_fwd_cat3d_lowfwd": "agn.torus",
    # NK0_mean_2p / NK0_mean_3p (AGNfitter-rX) torus axes not shared with the
    # NK0_mean_1p / SKIRTOR_mean_3p declarations above
    "agn_oa_nenkova": "agn.torus",
    "agn_tv_nenkova": "agn.torus",
    # Silva+04 obscured-torus column density
    "agn_log_nh_silva": "agn.torus",
    "agn_torus_frac": "agn.torus",
    # Fritz et al. (2006) smooth-dust torus
    "agn_fritz_r_ratio": "agn.torus",
    "agn_fritz_tau": "agn.torus",
    "agn_fritz_beta": "agn.torus",
    "agn_fritz_gamma": "agn.torus",
    "agn_fritz_oa": "agn.torus",
    "agn_fritz_psy": "agn.torus",
    # Narrow-line region
    "agn_nlr_cf": "agn.nlr",
    # R50 (#2214): "agn_alpha_ion" sat here, a second name for the Feltre NLR
    # ionizing power-law slope this block reads as agn_nlr_alpha_pl (entered
    # below, with the other five axes). Identical prior and default, read by
    # nothing on any live path -- the same disease R41 fixed for the
    # dust-to-metal axis one ruling earlier. The axis has one name now.
    # R41 (#2214): "neb_xid" sat here, a second name for the Feltre NLR
    # dust-to-metal grid axis this block reads as agn_nlr_xi_d (entered below,
    # with the other five axes). The partition is consulted only for
    # agn_-prefixed names, so the nebular-prefixed entry could never fire --
    # the dead entry #2214 was filed against. The axis has one name now.
    # Broad-line region
    "agn_blr_cf": "agn.blr",
    # FeII
    "agn_fe2_strength": "agn.feii",
    # Task 16 (item 7, F5): agn_bcnorm (qsogen_balmer's Balmer-continuum
    # strength knob, #2175) had no partition entry at all, so it fell
    # through to the shared "agn" group -- the feii sub-block's own wildcard
    # could never reach it (only the top-level agn wildcard could, via
    # AGN_BLOCK_CONSUMES[("feii", "qsogen_balmer")]).
    "agn_bcnorm": "agn.feii",
    # Attenuation
    "agn_polar_ebv": "agn.atten",
    "agn_polar_oa": "agn.atten",
    "agn_polar_T": "agn.atten",
    "agn_polar_beta": "agn.atten",
    "agn_attenuation_ebv": "agn.atten",  # smc_prevot block E(B-V)
    # GRAHSP torus (Buchner+2024): all seven are torus-only per
    # AGN_BLOCK_CONSUMES[("torus", "grahsp")]. Without an explicit entry
    # here, every "agn_grahsp_*" name falls through the catch-all below to
    # "agn.disc", so a torus wildcard could never reach them (task-12 audit,
    # same mechanism family as D2/D3: the catch-all was written for the
    # GENUINE grahsp disc params and over-reached to every OTHER grahsp
    # category, which happens to share the same substring).
    "agn_grahsp_cool_lam_um": "agn.torus",
    "agn_grahsp_cool_width": "agn.torus",
    "agn_grahsp_fcov": "agn.torus",
    "agn_grahsp_hot_fcov": "agn.torus",
    "agn_grahsp_hot_lam_um": "agn.torus",
    "agn_grahsp_hot_width": "agn.torus",
    "agn_grahsp_si": "agn.torus",
    # GRAHSP FeII-only knob.
    "agn_grahsp_a_feii": "agn.feii",
    # GRAHSP line-strength/-width pair: consumed by NLR, BLR *and* FeII
    # simultaneously (AGN_BLOCK_CONSUMES lists both under all three
    # categories). One sub-block owner cannot serve three readers, because
    # _agn_subblock_declared_params filters a block's consumed set down to
    # names the partition assigns to THAT block: under "agn.nlr" (their
    # previous home) blr='grahsp' read two parameters and its own wildcard
    # freed nothing, measured live at 1.20e-16 and -1.14e-22. Genuinely
    # shared at runtime, so shared is the assignment that lets every reader's
    # build reach them, through the agn-level wildcard (R34).
    "agn_grahsp_a_lines": "agn",
    "agn_grahsp_linewidth_kms": "agn",
    # GRAHSP bi-attenuation (attenuation-only).
    "agn_grahsp_ebv": "agn.atten",
    "agn_grahsp_ebv_agn": "agn.atten",
    # ── R34: the 31 names that used to have no entry at all ──────────────
    # An unlisted name fell through to the shared "agn" group silently, so
    # the table was only as complete as its last editor and every instrument
    # built on it inherited the hole. Each name below is assigned by reading
    # the code that consumes it; a name genuinely read across categories at
    # runtime keeps the shared "agn" owner and says why.
    #
    # Disc physics, each read only inside a disc block function.
    "agn_T_max": "agn.disc",  # powerlaw_disc_block's UV cutoff temperature
    "agn_adaf_alpha": "agn.disc",
    "agn_adaf_beta": "agn.disc",
    "agn_adaf_delta": "agn.disc",
    "agn_astar": "agn.disc",  # relagn black-hole spin
    "agn_log_mdot": "agn.disc",  # relagn accretion rate
    "agn_cigale_disk_delta": "agn.disc",  # skirtor/schartmann2005 disc slope
    "agn_grahsp_cutoff_nm": "agn.disc",
    "agn_grahsp_log_l5100": "agn.disc",
    "agn_grahsp_plbendloc_nm": "agn.disc",
    "agn_grahsp_plbendwidth": "agn.disc",
    "agn_grahsp_plslope": "agn.disc",
    "agn_grahsp_uvslope": "agn.disc",
    # Torus. agn_theta_torus is the gray Type-1/2 visibility mask's opening
    # angle, read at the runner's torus stage for every physical-decomposition
    # torus; agn_delta backs SKIRTORTorus's own ``delta`` class attribute.
    "agn_theta_torus": "agn.torus",
    "agn_delta": "agn.torus",
    # Narrow-line region: the analytic block's efficiency plus the six
    # photoionization-grid axes, all read in blocks/nlr.py.
    "agn_nlr_line_efficiency": "agn.nlr",
    "agn_nlr_alpha_pl": "agn.nlr",
    "agn_nlr_fwhm_kms": "agn.nlr",
    "agn_nlr_logU": "agn.nlr",
    "agn_nlr_logZ": "agn.nlr",
    "agn_nlr_logn": "agn.nlr",
    "agn_nlr_xi_d": "agn.nlr",
    # Broad-line region: read in blocks/blr.py. agn_blr_line_efficiency is
    # also read by the feii stage, the same shape as agn_blr_cf above, which
    # has been blr-owned since it was partitioned; kept consistent with it.
    "agn_blr_line_efficiency": "agn.blr",
    "agn_blr_logU": "agn.blr",
    "agn_blr_logZ": "agn.blr",
    "agn_blr_logn": "agn.blr",
    # Attenuation: the qsogen_smc block's own reddening knob, the one whose
    # short name 'ebv' is NOT agn_attenuation_ebv's (R30).
    "agn_ebv": "agn.atten",
    # Genuinely shared, and each says why.
    #
    # agn_ir_frac (the fracAGN spelling) is read by compose_l_nu at the
    # runner's cross-block normalization stage, where it sets the disc/torus
    # energy split -- not inside any one torus block, which is why writing it
    # at the agn top level is the spelling every caller and the #2189 guard
    # already use.
    "agn_ir_frac": "agn",
    # The three self-contained-GRAHSP knobs: no composable block reads them
    # (grep-verified across blocks/), only the monolithic grahsp forward
    # function, where every parameter is written flat at the agn level (R27).
    # A sub-block owner would name a sub-block that never runs for them.
    "agn_grahsp_a_bc": "agn",
    "agn_grahsp_tor_temp": "agn",
    "agn_grahsp_tor_cutoff_um": "agn",
}


def _agn_param_group(name: str) -> str:
    """Owning group for one ``agn_*`` full parameter name.

    Single source :func:`_partition_by_group` (the general per-build
    partition table) and :func:`_agn_subblock_declared_params` (composable
    sub-block wildcard scoping) both read, so the two cannot independently
    drift on the ``agn_grahsp_*`` catch-all the way they briefly did: an
    earlier version of ``_agn_subblock_declared_params`` read
    ``_AGN_PARTITION`` directly (skipping this fallback), which silently
    emptied ``grahsp_sbpl``'s and every other un-listed grahsp disc
    parameter's declared set.

    Parameters
    ----------
    name : str
        Full ``agn_*`` parameter name.

    Returns
    -------
    str
        ``"agn"`` (shared) or ``"agn.<subblock>"``.
    """
    # No substring catch-all: the six GRAHSP disc parameters this used to
    # cover (agn_grahsp_log_l5100, _uvslope, _plslope, _plbendloc_nm,
    # _plbendwidth, _cutoff_nm) now have explicit entries, and R34's census
    # keeps it that way. The rule was `"grahsp" in name -> "agn.disc"`, which
    # OVERRODE an explicit shared entry rather than filling a gap: with
    # agn_grahsp_a_lines correctly marked shared (three categories read it),
    # the substring test still routed it to agn.disc, so a build with
    # blr='grahsp' + agn={'all_params': FREE} froze it at its default while
    # the parameter was measurably live.
    return _AGN_PARTITION.get(name, "agn")


#: Grammar sub-block key -> the composable-block registry's category label.
#: Five of six match; ``AGN_BLOCKS``/``AGN_BLOCK_CONSUMES`` spell the sixth
#: ``"attenuation"``, the grammar's terser ``"atten"``.
_AGN_CONSUMES_CATEGORY: dict[str, str] = {
    "disc": "disc",
    "torus": "torus",
    "nlr": "nlr",
    "blr": "blr",
    "feii": "feii",
    "atten": "attenuation",
}


def _agn_subblock_declared_params(
    category: str,
    block_type: str | None,
    *,
    selection: Mapping[str, str] | None = None,
) -> frozenset[str] | None:
    """Declared parameters ONE AGN sub-block's OWN wildcard may free.

    Ground truth (Task 16, item 1: the declared-reads model) is
    ``tengri.components.agn.blocks._consumes.AGN_BLOCK_CONSUMES`` -- the
    SAME empirically-measured, per-(category, type) table the TOP-LEVEL
    ``agn`` group's scope already sources (unioned across every active
    block via :func:`_agn_active_param_set`) -- when it has an entry for
    this ``(category, block_type)``. Falls back to raw
    ``inspect.signature`` introspection on ``AGN_BLOCKS[category][block_type]``
    (mirroring :func:`_law_shape_params`'s dust-law introspection) only for a
    type genuinely absent from that table (grid-gated or otherwise
    unregistered), matching the top-level scope's own safe "over-free"
    fallback.

    One source, not two: the earlier version of this function read only the
    raw signature, independent of ``AGN_BLOCK_CONSUMES``. A block's function
    signature commonly accepts a parameter it does not, for that specific
    registered type, actually use (Task 16, item 1 addendum: ``kubota_done``'s
    own signature names ``agn_log_ledd`` -- the Eddington ratio is derived
    from ``agn_log_lbol`` instead, #846 -- and ``multicolor``/``slone_netzer``
    the same; measured dead, ``jax.grad`` exactly 0.0 at every sampled point).
    ``AGN_BLOCK_CONSUMES`` already excludes these EMPIRICALLY (the top-level
    wildcard has relied on that since task-12); reading raw signatures here
    instead let the sub-block wildcard free them anyway, an inert dimension
    the resolver could not see was dead. Sourcing from the same table makes
    the two mechanisms agree by construction
    (``tests/contract/test_agn_block_consumes.py``).

    Every name this returns is additionally filtered to what this sub-block
    OWNS in the partition table (:data:`_AGN_PARTITION`): a shared masking
    knob (``agn_cos_inc``, ``agn_theta_torus``) that ``AGN_BLOCK_CONSUMES``
    lists as consumed (because the runner reads it for this category, not
    because the block itself owns it) is filtered out here -- a param
    partitioned outside ``agn.<category>`` is reachable only through ITS
    owning group's wildcard or an explicit name (see
    :func:`_build_agn_search_view`), never through this sub-block's own
    ``'*'`` -- crediting it here would claim a freedom the resolver cannot
    actually deliver. When every parameter a type reads is such a shared
    knob (``nenkova_agnfitter``'s sole axis is the shared ``agn_cos_inc``;
    GRAHSP's NLR/BLR line-strength pair is consumed by three categories at
    once), the filtered answer is a non-``None`` empty set: NOT "unknown,
    leave unnarrowed" but "covers zero parameters here", which
    :func:`_check_wildcard_freed_something` turns into a loud
    :class:`~tengri.config.exceptions.ParameterError` instead of silently
    freeing nothing (D3). That signal was a ``WildcardNoOpWarning`` until
    #2187 escalated covered-zero to a raise, on the grounds that a warning
    here is exactly as swallowable as the silence it replaced.

    Parameters
    ----------
    category : str
        Grammar sub-block key: ``"disc"``, ``"torus"``, ``"nlr"``, ``"blr"``,
        ``"feii"``, or ``"atten"``.
    block_type : str or None
        The selected type/law for this sub-block, or ``"none"``/``None``.

    Returns
    -------
    frozenset of str or None
        This sub-block's own declared, freeable parameters for
        ``block_type`` (possibly empty -- see above), or ``None`` when
        ``block_type`` is not a registered composable block at all (should
        not happen for a grammar-validated type; the safe "leave unnarrowed"
        answer if it ever does).
    """
    if not block_type or block_type == "none":
        return frozenset()

    from tengri.components.agn.blocks._consumes import AGN_BLOCK_CONSUMES
    from tengri.components.agn.blocks._protocol import AGN_BLOCKS

    consumes_category = _AGN_CONSUMES_CATEGORY.get(category, category)
    fn = AGN_BLOCKS.get(consumes_category, {}).get(block_type)
    if fn is None:
        return None

    companions = _agn_subblock_companion_params(category, block_type, selection=selection)
    consumed = AGN_BLOCK_CONSUMES.get((consumes_category, block_type))
    if consumed is not None:
        read = frozenset(consumed) | companions
    else:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):  # pragma: no cover - no unsigned callables registered
            return None

        read = (
            frozenset(
                p.name
                for p in sig.parameters.values()
                if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL) and p.name.startswith("agn_")
            )
            | companions
        )
    owning_group = f"agn.{category}"
    # The filter stops a block claiming a name it merely happens to mention --
    # a shared knob, or one another sub-block owns and reads itself. A
    # CROSS-CATEGORY companion is the opposite statement: this block reads the
    # name and the owning category's selected block does not, so it survives
    # the filter deliberately (R36). Without the exception the disc that
    # applies SKIRTOR geometry could never free the geometry it reads.
    cross_category = frozenset(
        name for name in companions if is_cross_category_companion(name, owning_group)
    )
    return (
        frozenset(name for name in read if _agn_param_group(name) == owning_group) | cross_category
    )


#: (grammar category, block_type) whose physics spans TWO functions: the
#: registered dispatch callable (``AGN_BLOCKS[category][block_type]``,
#: introspected above) plus a companion helper the runner calls SEPARATELY
#: for a later pipeline stage, invisible to that introspection. Currently
#: just the standalone polar-dust attenuation block: the registered
#: ``polar_dust_attenuation_block`` (Stage 5, the LOS reddening factor)
#: declares ``agn_polar_ebv``/``agn_cos_inc``/``agn_polar_oa``/
#: ``agn_polar_law``, while ``agn_polar_T``/``agn_polar_beta`` are read only
#: by ``polar_dust_reemission_lnu`` (Stage 6, the re-emission graybody),
#: which ``compose_l_nu`` calls directly rather than through the block
#: registry (task13 fix-round-1). Both functions' ``agn_*`` names belong to
#: this sub-block's own wildcard scope.
_AGN_SUBBLOCK_COMPANION_KEY: tuple[str, str] = ("atten", "polar_dust")

#: category -> agn_* names read by a companion step that applies to EVERY
#: type registered in that category (not one specific type, unlike
#: :data:`_AGN_SUBBLOCK_COMPANION_KEY` above). Found via Task 16, item 2's
#: measured (not scoping-derived) liveness sweep
#: (``tests/contract/test_agn_wildcard_measured_liveness.py``), which
#: caught both entries as live-but-excluded for every affected type:
_AGN_CATEGORY_WIDE_COMPANION_PARAMS: dict[str, frozenset[str]] = {
    # agn_ebv_disc (#916): compose_l_nu reddens EVERY disc block's own
    # continuum with this Prevot-SMC screen at the runner stage
    # (blocks/runner.py, ``redden_disc(wave, L_lambda_disc,
    # params.get("agn_ebv_disc", 0.0))``) -- no individual disc TYPE's own
    # function signature names it, so it is invisible to both raw
    # introspection and every per-type AGN_BLOCK_CONSUMES entry alike.
    "disc": frozenset({"agn_ebv_disc"}),
    # The feii entry that used to sit here (agn_fe2_strength) is gone: the
    # read it described belongs to the BLR analytic block, not to the feii
    # category, so unconditionally it freed a parameter the selected
    # configuration ignores -- measured DEAD for feii='grahsp' and
    # feii='qsogen_balmer' under blr='none', LIVE for both under
    # blr='analytic'. It is now a companion conditioned on the selected BLR
    # block, in _agn_subblock_companion_params (R33).
}


def is_cross_category_companion(name: str, owning_group: str) -> bool:
    """Whether ``name`` is a companion of ANOTHER sub-block, for ``owning_group``.

    The one statement of R36's rule. Three layers act on it -- the wildcard
    scope, the resolver's claims map, and the outcome bookkeeping -- and when
    each spelled it for itself the restriction landed in one of them: a SHARED
    name's owner is ``"agn"``, which differs from every ``"agn.<category>"``,
    so ``agn_cos_inc`` and ``agn_polar_law`` were claimed by whichever block
    read them, and an ordinary
    ``agn={'all_params': FREE, 'atten': {'type': 'polar_dust',
    'all_params': Fixed(DEFAULT)}}`` raised ``ParameterError``.

    A shared name is never cross-category: every wildcard can already reach it
    through the agn top level, so claiming it takes it from the group that owns
    it rather than giving it to one that could not have it.

    Parameters
    ----------
    name : str
        Full ``agn_*`` parameter name.
    owning_group : str
        The ``"agn.<category>"`` group of the block doing the reading.

    Returns
    -------
    bool
        True only for a name a DIFFERENT sub-block owns.

    Notes
    -----
    **JIT-compatible**: no, pure-Python builder-time helper.
    """
    owner = _agn_param_group(name)
    return owner.startswith("agn.") and owner != owning_group


def _agn_own_side_companion_params(
    category: str, block_type: str, selection: Mapping[str, str]
) -> set[str]:
    """Companions a block claims for itself, before any cross-category claim.

    The category-wide table, the R33 conditional feii read, and the
    type-specific companion FUNCTION. Split out from the cross-category shape
    so the latter can ask "does the owner already claim this?" without
    recursing back into itself.
    """
    from tengri.components.agn.blocks._consumes import AGN_BLOCK_CONSUMES

    out = set(_AGN_CATEGORY_WIDE_COMPANION_PARAMS.get(category, frozenset()))

    # Conditional companion (R33): agn_fe2_strength is read by the BLR analytic
    # block, so the feii sub-block's wildcard may claim it only when such a BLR
    # block is actually selected. As an unconditional category-wide entry it
    # freed a measured-dead parameter under blr='none' (grahsp and
    # qsogen_balmer both DEAD there, LIVE with blr='analytic'); the condition is
    # read off the selected BLR block's own CONSUMES entry, so the two move
    # together. A feii type that declares the name itself (boroson_green) gets
    # it from its own entry regardless.
    blr_type = selection.get("blr")
    if category == "feii" and blr_type:
        out |= {"agn_fe2_strength"} & set(AGN_BLOCK_CONSUMES.get(("blr", blr_type), ()))

    if (category, block_type) == _AGN_SUBBLOCK_COMPANION_KEY:
        from tengri.components.agn.blocks.atten import polar_dust_reemission_lnu

        sig = inspect.signature(polar_dust_reemission_lnu)
        out.update(
            p.name
            for p in sig.parameters.values()
            if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL) and p.name.startswith("agn_")
        )
    return out


def agn_cross_category_claims(selection: Mapping[str, str] | None) -> dict[str, str]:
    """Which sub-block's wildcard governs each cross-category companion name.

    A block that reads a name another category owns has that name in its
    wildcard SCOPE (see :func:`_agn_subblock_companion_params`), but the
    resolver looks for a parameter in its OWNING sub-block dict, so the
    reading block's ``'*'`` would never be consulted. This is the map that
    tells the resolver otherwise: name -> the grammar category whose wildcard
    also governs it on this build.

    Parameters
    ----------
    selection : mapping or None
        Grammar category -> selected block type for the whole build.

    Returns
    -------
    dict
        Name -> claiming category. Empty when nothing is claimed. A name
        claimed by two categories at once keeps the first in the canonical
        pipeline order, which is deterministic rather than
        dict-iteration-dependent.

    Notes
    -----
    **JIT-compatible**: no, pure-Python builder-time helper.
    """
    claims: dict[str, str] = {}
    if not selection:
        return claims
    for category in _AGN_CONSUMES_CATEGORY:
        block_type = selection.get(category)
        if not block_type or block_type == "none":
            continue
        owning_group = f"agn.{category}"
        for name in _agn_subblock_companion_params(category, block_type, selection=selection):
            if is_cross_category_companion(name, owning_group):
                claims.setdefault(name, category)
    return claims


def _agn_subblock_companion_params(
    category: str, block_type: str, *, selection: Mapping[str, str] | None = None
) -> frozenset[str]:
    """``agn_*`` names read by a sub-block's companion helper(s), if any.

    Three shapes of companion, none of them visible to a plain
    ``inspect.signature(AGN_BLOCKS[category][block_type])`` (and, for a type
    with its own ``AGN_BLOCK_CONSUMES`` entry, invisible to that entry too):
    a type-specific companion FUNCTION (:data:`_AGN_SUBBLOCK_COMPANION_KEY`),
    a category-wide companion READ that applies for every type in a category
    (:data:`_AGN_CATEGORY_WIDE_COMPANION_PARAMS`), and a CROSS-CATEGORY read,
    where a block reads a name another category owns. The last two are
    conditioned on what the build actually selects, which is why ``selection``
    is passed: freeing a name whose owner is not active is the same defect as
    not freeing one whose owner is.

    Parameters
    ----------
    category : str
        Grammar sub-block key of the block whose wildcard is being scoped.
    block_type : str
        The selected type for that sub-block.
    selection : mapping, optional
        Grammar category -> selected block type for the whole build. Absent
        (the default) means "nothing else is selected", which is the right
        reading for a caller describing a block in isolation
        (``describe_agn_block``): a cross-category name is claimed, because on
        such a build no other block reads it, while the feii companion is not,
        because it exists only when a BLR block that reads it is present. The
        two differ because one asks "is anyone else taking this?" and the
        other "is anyone reading it at all?".

    Returns
    -------
    frozenset of str
        Companion names this block's own wildcard may free.
    """
    from tengri.components.agn.blocks._consumes import AGN_BLOCK_CONSUMES

    selection = dict(selection or {})
    out = _agn_own_side_companion_params(category, block_type, selection)

    # Cross-category companion (R36): a block can read a name another category
    # owns. `('disc', 'schartmann2005_skirtor_atten')` applies SKIRTOR's own
    # geometry to its disc continuum, so it reads agn_oa_skirtor / agn_p_skirtor
    # / agn_q_skirtor / agn_tau_skirtor -- all owned by `agn.torus`. With no
    # torus selected, nothing could free them: the disc's own wildcard frees
    # only what it owns, and the shared agn-level one cannot reach a
    # sub-block-owned name. Measured on that build, all four are live (grads
    # 6.9e-20, 1.2e-19, -4.3e-19, 1.9e-19).
    #
    # The reading block claims such a name exactly while the OWNING category's
    # selected block does not read it itself -- so with `torus='skirtor'` the
    # torus keeps sole ownership and no name is freeable twice, while with the
    # torus absent or on a torus that ignores the geometry the disc's wildcard
    # reaches what the disc reads.
    owning_group = f"agn.{category}"
    consumes_cat = _AGN_CONSUMES_CATEGORY.get(category, category)
    for name in AGN_BLOCK_CONSUMES.get((consumes_cat, block_type), frozenset()):
        if not is_cross_category_companion(name, owning_group):
            continue
        owner_category = _agn_param_group(name)[len("agn.") :]
        owner_type = selection.get(owner_category)
        if not owner_type or owner_type == "none":
            out.add(name)
            continue
        owner_consumes_cat = _AGN_CONSUMES_CATEGORY.get(owner_category, owner_category)
        # The owner keeps the name when it reads it itself, and equally when it
        # claims it as its OWN-side companion: agn_fe2_strength is owned by
        # feii and read by the analytic BLR block, and R33 gives it to the feii
        # wildcard on exactly that build. Without this second half both blocks
        # declared it, so which disposition won depended on resolution order.
        if name in AGN_BLOCK_CONSUMES.get((owner_consumes_cat, owner_type), frozenset()):
            continue
        if name in _agn_own_side_companion_params(owner_category, owner_type, selection):
            continue
        out.add(name)

    return frozenset(out)


#: (torus type name) -> (class-only names allowed, block-only names allowed),
#: each a documented, physics-grounded deviation. A torus type string present
#: in both registries but absent here must match EXACTLY.
_TORUS_REGISTRY_ALLOWED_DEVIATIONS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    # SKIRTORTorus (the monolithic class) additionally bundles its own disc
    # shape (agn_delta, R12c: the composable design puts the disc in a
    # separate disc block instead) and its own polar-dust reemission trio
    # (agn_polar_ebv/T/beta -- R22, task13 fix-round-1, retired the
    # composable torus block's OWN bundled polar dust; on the composable
    # path polar dust is owned exclusively by the standalone ``polar_dust``
    # attenuation block now). See test_skirtor_torus_wiring.py's own
    # class/block reconciliation test, which pins the identical exclusion.
    "skirtor": (
        frozenset({"agn_delta", "agn_polar_ebv", "agn_polar_T", "agn_polar_beta"}),
        frozenset(),
    ),
}


def check_agn_torus_registry_agreement() -> None:
    """RULING R13: every torus type registered in BOTH the monolithic and
    composable AGN registries must declare the SAME ``agn_*`` parameter
    names.

    Two independent registries can dispatch the identical torus physics: the
    monolithic ``component_factory._REGISTRY`` (a bare ``SEDModelComponent``
    subclass, e.g. ``agn=SKIRTORTorus(...)``) and the composable
    ``AGN_BLOCKS['torus']`` (``agn={'torus': {'type': ...}}``). Before R17
    (Task 16) they silently disagreed for ``'skirtor'``: the class declared
    ``agn_band_frac`` for its own covering fraction while the composable
    block (and six OTHER composable torus blocks) declared the identical
    quantity ``agn_torus_frac`` -- two spellings for one physical parameter,
    reachable through two different names depending which path a caller
    used. This function is the live guard against that class of drift
    recurring, over EVERY currently-shared torus type string, not just
    ``'skirtor'``.

    Scoped to the torus category deliberately: ``AGN_BLOCKS['disc']`` ALSO
    has an entry named ``'skirtor'`` (the SKIRTOR-inspired disc continuum
    shape), but ``component_factory._REGISTRY['skirtor']`` is
    ``SKIRTORTorus`` -- a TORUS class. Comparing it against the DISC block
    would compare unrelated physics that happen to share an English name,
    not the same model reached two ways -- ``AGN_MODEL_CONSUMES``'s own
    ``'skirtor'`` entry (the monolithic dispatch's own consumed-param table)
    confirms the class is torus-flavored, matching
    ``AGN_BLOCK_CONSUMES[('torus', 'skirtor')]`` closely.

    Raises
    ------
    ConfigError
        Naming both sides' full parameter-name sets, for every type string
        that disagrees beyond :data:`_TORUS_REGISTRY_ALLOWED_DEVIATIONS`.

    Notes
    -----
    **JIT-compatible**: no; build-time/CI introspection over the registries,
    not called on the model-build hot path. Exercised by
    ``tests/contract/test_agn_registry_param_agreement.py`` (a monkeypatched
    synthetic disagreement, and a live check over every currently-shared
    string).
    """
    from tengri.components.agn.blocks._protocol import AGN_BLOCKS
    from tengri.forward.component_factory import _REGISTRY

    torus_blocks = AGN_BLOCKS.get("torus", {})
    mismatches = []
    for name in sorted(set(_REGISTRY) & set(torus_blocks)):
        cls = _REGISTRY[name]
        fn = torus_blocks[name]
        try:
            instance = cls()
        except TypeError:
            continue  # not a bare-constructible SEDModelComponent; not this check's concern
        class_names = {d.name for d in instance.declared_parameters()}
        block_names = frozenset(
            p.name
            for p in inspect.signature(fn).parameters.values()
            if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL) and p.name.startswith("agn_")
        )
        allowed_class_only, allowed_block_only = _TORUS_REGISTRY_ALLOWED_DEVIATIONS.get(
            name, (frozenset(), frozenset())
        )
        class_only = class_names - block_names - allowed_class_only
        block_only = block_names - class_names - allowed_block_only
        if class_only or block_only:
            mismatches.append((name, class_names, block_names, class_only, block_only))

    if mismatches:
        name, class_names, block_names, class_only, block_only = mismatches[0]
        raise ConfigError(
            f"AGN torus registry disagreement for {name!r}: the monolithic "
            f"component_factory._REGISTRY class and the composable "
            f"AGN_BLOCKS['torus'] block declare different agn_* parameter "
            f"names.\n"
            f"  class names: {sorted(class_names)}\n"
            f"  block names: {sorted(block_names)}\n"
            f"  class-only (unexpected): {sorted(class_only)}\n"
            f"  block-only (unexpected): {sorted(block_only)}\n"
            f"Reconcile the two registrations to the same names, or document "
            f"the deviation in _TORUS_REGISTRY_ALLOWED_DEVIATIONS with the "
            f"physics reason (RULING R13)."
        )


#: Every spelling that resolves to ``agn_ir_frac`` (fracAGN) in the raw,
#: pre-resolution ``agn={...}`` dict: the canonical full name, its bare short
#: form, and both spellings of the pre-#1296 legacy name.
_AGN_IR_FRAC_SPELLINGS = frozenset({"agn_ir_frac", "ir_frac", "agn_fracAGN", "fracAGN"})

#: The pre-#1296 legacy half of the set above. ``_build_agn_search_view``
#: resolves the canonical name's own two spellings; these two it does not, so
#: they are scanned separately across the same locations.
_AGN_IR_FRAC_LEGACY_SPELLINGS: tuple[str, ...] = ("agn_fracAGN", "fracAGN")


def _fracagn_value_is_active(value: object) -> bool:
    """Whether one written fracAGN value can be nonzero.

    A free prior almost surely samples positive, so any distribution counts.
    ``Fixed(DEFAULT)`` (fracAGN's registry default is 0.0) and an explicit
    ``Fixed(0.0)`` do not: explicit-but-inert is not the #2189 conflict.
    Anything that is not a number -- a string, a multi-element array -- is not
    a value this parameter can take, so it is not active either.
    """
    if isinstance(value, Fixed):
        if _is_default_fixed(value):
            return False
        value = value.value
    if isinstance(value, Distribution):
        return True
    try:
        return float(value) > 0.0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
