# SPDX-License-Identifier: BSD-3-Clause
"""Composable AGN runner — picks one block per pipeline stage and runs them.

Canonical execution order (paper §2.1.6 / upstream GRAHSP module ordering)::

    [disc] → [nlr] → [blr] → [feii] → [torus] → [attenuation]

Each stage is owned by a registered block from
:mod:`tengri.components.agn.blocks._protocol`.

The runner is registered as ``AGN_MODELS["composable"]`` so users can opt in
via the standard ``Parameters(agn_model="composable", ...)`` flow without
replacing the existing monolithic models.

Example
-------
GRAHSP-pure recipe (every stage uses the GRAHSP impl)::

    from tengri import Parameters

    p = Parameters(
        agn_model="composable",
        agn_disc_block="grahsp_sbpl",
        agn_nlr_block="grahsp",
        agn_blr_block="grahsp",
        agn_feii_block="grahsp",
        agn_torus_block="grahsp",
        agn_attenuation_block="grahsp_biatten",
        # ... GRAHSP params ...
    )

Cross-model mix (GRAHSP BBB + simple two-temperature torus + Prevot SMC
attenuation)::

    p = Parameters(
        agn_model="composable",
        agn_disc_block="grahsp_sbpl",
        agn_nlr_block="none",
        agn_blr_block="none",
        agn_feii_block="none",
        agn_torus_block="two_temperature",
        agn_attenuation_block="smc_prevot",
        agn_grahsp_l5100=...,
        agn_T_hot=...,  # belongs to two_temperature torus block
        agn_attenuation_ebv=...,
    )
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import (
    AGN_BLOCKS,
    resolve_agn_block,
)
from tengri.components.agn.blocks.atten import polar_dust_reemission_lnu
from tengri.components.agn.blocks.masking import (
    sigmoid_visibility_mask,
    split_lines_result,
)
from tengri.components.agn.blocks.torus_screen import (
    TORUS_SCREEN_PARAMS,
    torus_screen_transmission,
)
from tengri.components.agn.polar_dust import _RV_SMC, smc_extinction_curve
from tengri.components.agn.reddening import redden_disc
from tengri.components.agn.skirtor import skirtor_disc_dust_ratio
from tengri.utils.physics_constants import L_SUN

#: Torus selectors that do NOT receive the gray Type-1/2 visibility mask:
#: ``none`` (no torus) and the self-contained empirical quasar templates
#: (``qsogen``, ``grahsp``), which already encode an inclination-averaged SED —
#: masking them would be double-counting. The dusty-screen tori (skirtor/fritz)
#: are handled by their own wavelength-dependent screen above.
_SELF_CONTAINED_TORI: frozenset[str] = frozenset({"none", "qsogen", "grahsp"})

__all__ = [
    "BLOCK_SELECTOR_KEYS",
    "C_AA_PER_S",
    "DEFAULT_BLOCK_SELECTORS",
    "RecipeWarning",
    "composable_agn_l_nu",
    "compose_l_nu",
    "validate_block_recipe",
]


class RecipeWarning(UserWarning):
    """Emitted by :func:`validate_block_recipe` for suspicious block combos.

    Subclassing :class:`UserWarning` lets callers ``warnings.simplefilter
    ("error", RecipeWarning)`` to turn recipe issues into hard errors during
    development without affecting other warnings.
    """


# Suspicious-combination rule table. Each entry is a structured rule the
# validator runs at composition time. Adding a new rule = one line here.
#
# Disc impls known to produce a sensible UV/optical continuum at 5100Å
# (i.e. compatible with GRAHSP-style downstream blocks that normalize to
# λL_λ(5100Å)). Block impls outside this set may emit zero or NaN at
# 5100Å, breaking the downstream normalization silently.
_DISCS_WITH_5100A_CONTINUUM: frozenset[str] = frozenset(
    {
        "grahsp_sbpl",
        "powerlaw",
        "multicolor",
        "kubota_done_3zone",
    }
)

# Downstream blocks that *require* a sensible disc 5100Å luminosity.
# Includes BLR/NLR (their λL_λ → L_disc_bol conversion uses the Krawczyk+2013
# bolometric correction, which assumes a UV/optical continuum at 5100Å).
_DOWNSTREAM_NEEDS_L5100: dict[str, frozenset[str]] = {
    "nlr": frozenset({"analytic", "grahsp"}),
    "blr": frozenset({"analytic", "grahsp"}),
    "feii": frozenset({"grahsp", "boroson_green"}),
    "torus": frozenset({"grahsp"}),
}

# Disc impls covered by the multicolor / Kubota-Done set are added to the
# 5100Å-OK list. ADAF deliberately is NOT (its inner flow is X-ray dominated;
# any 5100Å contribution is from the truncated outer disc only).
_DISCS_WITH_5100A_CONTINUUM = _DISCS_WITH_5100A_CONTINUUM | frozenset(
    {
        "multicolor",
        "kubota_done",
        "qsogen",
        "relagn",
        "richards2006",
    }
)

#: Speed of light in Å × Hz, used for L_λ → L_ν conversion.
from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL, DEFAULT_AGN_LUM_RATIO
from tengri.utils.physics_constants import C_AA as C_AA_PER_S

#: Selector keys recognized by the runner. Match the canonical pipeline order.
BLOCK_SELECTOR_KEYS: tuple[str, ...] = (
    "agn_disc_block",
    "agn_nlr_block",
    "agn_blr_block",
    "agn_feii_block",
    "agn_torus_block",
    "agn_attenuation_block",
)

#: Default to a no-op pipeline so a bare ``agn_model="composable"`` doesn't
#: silently emit garbage — users must opt in to each block by name.
DEFAULT_BLOCK_SELECTORS: dict[str, str] = {
    "agn_disc_block": "none",
    "agn_nlr_block": "none",
    "agn_blr_block": "none",
    "agn_feii_block": "none",
    "agn_torus_block": "none",
    "agn_attenuation_block": "none",
}


def validate_block_recipe(
    *,
    agn_disc_block: str,
    agn_nlr_block: str,
    agn_blr_block: str,
    agn_feii_block: str,
    agn_torus_block: str,
    agn_attenuation_block: str,
    params: dict | None = None,
) -> list[str]:
    r"""Check a block recipe for suspicious / unphysical combinations.

    Emits a :class:`RecipeWarning` per issue found and returns the list of
    issue strings (so tests can introspect them deterministically).

    Validation runs at composition time (Python-side, not under JIT), so
    cost is paid once per recipe construction — there is no inner-loop
    overhead.

    Rules implemented
    -----------------
    1. **Unknown block name** — the selector points at a block that is not
       registered; raise ``ValueError`` rather than warn (typo == hard
       error so users notice immediately).
    2. **All-none recipe** — every selector is ``"none"``; output will be
       identically zero. Almost certainly a misuse.
    3. **No disc, active downstream** — disc is ``"none"`` but nlr / blr /
       feii / torus are not. The downstream blocks scale by the disc's
       :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` (zero), so they emit
       zero too. Either the user forgot to pick a disc impl, or the recipe
       is genuinely degenerate.
    4. **GRAHSP downstream + non-5100Å disc** — GRAHSP nlr / blr / feii / torus
       expect the disc to deliver a meaningful UV/optical continuum at
       5100Å. Pairing them with an exotic disc (e.g. pure ADAF) likely
       produces an unintended SED.
    5. **GRAHSP biatten with no GRAHSP body** — the SMC-Prevot curve is
       generic, so this is technically valid; warn that the user might
       prefer the more clearly named ``"smc_prevot"`` block (when wrapped
       in a future PR).
    6. **NLR / BLR without UV/optical disc** — these lines blocks convert
       :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` to a bolometric disc
       luminosity via the Krawczyk+ 2013 correction. A non-5100Å disc
       triggers the same warning as rule 4.
    7. **Polar-dust block with E(B-V)=0** — the ``polar_dust`` attenuation
       block is a no-op when ``agn_polar_ebv = 0``; warn to surface unset
       params before the user wonders why the SED is unattenuated.

    Parameters
    ----------
    agn_disc_block, agn_nlr_block, agn_blr_block, agn_feii_block, \
agn_torus_block, agn_attenuation_block : str
        Selectors for each pipeline stage.
    params : dict, optional
        Free parameter dict; reserved for future per-impl param-presence
        checks. Currently unused but accepted for forward compatibility.

    Returns
    -------
    issues : list[str]
        Human-readable strings describing each warning emitted (empty if
        the recipe is clean).

    Raises
    ------
    ValueError
        If any selector points at a name not in :data:`AGN_BLOCKS`.
    """
    selectors = {
        "disc": agn_disc_block,
        "nlr": agn_nlr_block,
        "blr": agn_blr_block,
        "feii": agn_feii_block,
        "torus": agn_torus_block,
        "attenuation": agn_attenuation_block,
    }

    # Rule 1: hard error on unknown block names.
    for category, name in selectors.items():
        if name not in AGN_BLOCKS[category]:
            available = sorted(AGN_BLOCKS[category])
            raise ValueError(f"Unknown {category} block {name!r}. Available: {available}.")

    issues: list[str] = []

    def _emit(msg: str) -> None:
        issues.append(msg)
        warnings.warn(msg, RecipeWarning, stacklevel=3)

    # Rule 2: all-none.
    if all(name == "none" for name in selectors.values()):
        _emit(
            "Composable AGN: every block selector is 'none' — the AGN SED "
            "will be identically zero. Pick at least a disc block to "
            "produce non-trivial output."
        )

    # Rule 3: no disc, active downstream.
    downstream_active = any(selectors[cat] != "none" for cat in ("nlr", "blr", "feii", "torus"))
    if selectors["disc"] == "none" and downstream_active:
        active = [
            f"{cat}={selectors[cat]!r}"
            for cat in ("nlr", "blr", "feii", "torus")
            if selectors[cat] != "none"
        ]
        _emit(
            f"Composable AGN: agn_disc_block='none' but downstream "
            f"blocks are active ({', '.join(active)}). These blocks "
            f"normalize to lambda*L_lambda(5100A) of the disc, which "
            f"is zero — the active blocks will emit zero. Pick a disc "
            f"impl (e.g. 'grahsp_sbpl' or 'powerlaw')."
        )

    # Rule 4: downstream blocks that need an UV/optical disc.
    for category, requiring_blocks in _DOWNSTREAM_NEEDS_L5100.items():
        if (
            selectors[category] in requiring_blocks
            and selectors["disc"] not in _DISCS_WITH_5100A_CONTINUUM
            and selectors["disc"] != "none"  # rule 3 already covers this
        ):
            _emit(
                f"Composable AGN: {category}_block={selectors[category]!r} "
                f"normalizes to the disc's lambda*L_lambda(5100A), but "
                f"disc_block={selectors['disc']!r} is not in the set of "
                f"impls known to produce a meaningful UV/optical continuum "
                f"at 5100A: {sorted(_DISCS_WITH_5100A_CONTINUUM)}. "
                f"Verify your disc impl emits sensible flux at 5100A."
            )

    # Rule 7: polar dust selected but E(B-V) is 0 (no-op). The value is only
    # inspectable when concrete (build-time / a Fixed param); a traced (fitted)
    # agn_polar_ebv raises on float() — ConcretizationTypeError is a TypeError
    # subclass — and needs no no-op warning, since the user is explicitly
    # fitting it. The concreteness guard keeps this public validator safe for
    # any caller even though the forward pass no longer invokes it.
    if selectors["attenuation"] == "polar_dust" and params is not None:
        try:
            ebv_is_zero = float(params.get("agn_polar_ebv", 0.0)) == 0.0
        except (TypeError, ValueError):
            ebv_is_zero = False
        if ebv_is_zero:
            _emit(
                "Composable AGN: agn_attenuation_block='polar_dust' but "
                "agn_polar_ebv=0 (no extinction applied). Either set "
                "agn_polar_ebv > 0 or pick agn_attenuation_block='none'."
            )

    # (Rule 8, the adaf-deprecation steer, was removed once the faithful
    # Mahadevan 1997 ADAF rewrite landed in #898 — the block is now production.)

    return issues


def compose_l_nu(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_disc_block: str,
    agn_nlr_block: str,
    agn_blr_block: str,
    agn_feii_block: str,
    agn_torus_block: str,
    agn_attenuation_block: str,
    template_state: dict | None = None,
    return_l2500: bool = False,
    **params,
) -> Array | tuple[Array, float]:
    r"""Compose AGN-side :math:`L_\nu` from per-stage block implementations.

    Pipeline (paper §2.1.6 / upstream module order)::

        L_λ_total = L_disc + L_nlr + L_blr + L_feii + L_torus
        L_λ_atten = L_λ_total × attenuation_factor
        L_ν       = L_λ_atten × λ²/c

    The disc stage runs first so its 5100Å luminosity can scale the NLR /
    BLR / FeII / torus normalizations (matching upstream GRAHSP convention).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    agn_disc_block, agn_nlr_block, agn_blr_block, agn_feii_block, \
agn_torus_block, agn_attenuation_block : str
        Names of the registered block implementations to use. **Static**
        under JIT (Python strings; the runner resolves them at trace time).
    template_state : dict, optional
        Pre-loaded template bundles keyed by family name (e.g.
        ``{"grahsp": GRAHSPTemplates}``). When supplied, each block reads
        templates from this dict instead of calling its own
        ``load_*_templates()`` helper at trace time — keeps HDF5 / file
        I/O out of the JIT trace boundary. ``None`` (default) falls back
        to the in-block lru_cache load.
    return_l2500 : bool, optional
        When True, return ``(L_nu, L_2500_intrinsic, L_4400_intrinsic)``
        tuple. When False (default), return only ``L_nu`` for backward
        compatibility with existing single-return callers. Default: False.
    **params
        Per-impl free parameters. Each block consumes the keys it
        recognizes and ignores the rest.

    Returns
    -------
    L_nu : ndarray, shape (n_wave,)
        Total AGN-side :math:`L_\nu` [erg/s/Hz].
    L_2500_intrinsic : float, optional
        When ``return_l2500=True``, the un-reddened intrinsic disc
        monochromatic luminosity at 2500 Å [erg/s/Hz], capturing the
        disc shape at the ``agn_log_lbol`` normalization. Returned as
        second element of tuple. Otherwise not returned.
    L_4400_intrinsic : float, optional
        When ``return_l2500=True``, the un-reddened intrinsic disc
        monochromatic luminosity at 4400 Å [erg/s/Hz], capturing the
        disc shape at the ``agn_log_lbol`` normalization. Returned as
        third element of tuple. Otherwise not returned.

    Notes
    -----
    JIT-compatible (selectors are static). The order of operations matches
    the upstream GRAHSP ``ActivateGTorus``/``ActivateNarrowLines``/
    ``ActivateBroadLines``/``ActivatePL``/``BiAttenuationLaw`` chain so
    an all-grahsp selection is numerically equivalent to
    :func:`tengri.components.agn.grahsp.compute_grahsp_sed`.

    **Disc shape propagation**: ``L_2500_intrinsic`` differs between disc
    implementations (e.g. ``multicolor`` vs ``richards2006``) at the same
    ``agn_log_lbol``, enabling disc-shape-dependent downstream physics
    (e.g. α_ox in X-ray corona).
    """
    wave = jnp.asarray(wavelength)

    # Pre-loaded template libraries are forwarded to each stage under a stable
    # kwarg name blocks recognize (``templates``). The lookup is PER STAGE:
    # every block family has its own library, so handing the same bundle to
    # all six stages (as this did until the threading fix) can only ever feed
    # one family and silently leaves the rest to load their own grid at trace
    # time — which bakes it into the graph as constants.
    #
    # Keys are ``"<category>/<name>"``, matching ``collect_block_templates``.
    # ``"grahsp"`` is still honoured so callers holding the old flat bundle
    # keep working. When a stage has no entry, the block falls back to its own
    # cached load.
    _legacy_grahsp = template_state.get("grahsp") if template_state is not None else None

    def _templates_for(category: str, name: str):
        """Resolve the pre-loaded library for one stage, if any."""
        if template_state is None:
            return _legacy_grahsp
        found = template_state.get(f"{category}/{name}")
        return _legacy_grahsp if found is None else found

    disc_templates = _templates_for("disc", agn_disc_block)

    # Stage 1: disc continuum (L_lambda [erg/s/Å]).
    disc_fn = resolve_agn_block("disc", agn_disc_block)
    L_lambda_disc = disc_fn(
        wave,
        agn_log_lbol=agn_log_lbol,
        templates=disc_templates,
        **params,
    )
    # Disc dust obscuration (agn_ebv_disc, Prévot SMC). Applied on the composable
    # path so every disc block respects it — previously only the monolithic
    # forward models reddened, so composable-routed presets (adaf, kubota_done_full)
    # silently ignored agn_ebv_disc (#916). No-op at the default agn_ebv_disc=0.
    # The intrinsic L_2500/L_4400 below recompute from the un-reddened disc block,
    # so the X-ray/radio anchors stay obscuration-independent.
    L_lambda_disc = redden_disc(wave, L_lambda_disc, jnp.asarray(params.get("agn_ebv_disc", 0.0)))

    # Capture L_2500_intrinsic and L_4400_intrinsic: the un-reddened,
    # agn_log_lbol-normalized disc monochromatic luminosities [erg/s/Hz] that
    # drive X-ray alpha_ox and radio loudness. These follow CIGALE's
    # ``intrin_Lnu_2500A_30deg`` convention — they are evaluated at a FIXED 30 deg
    # reference inclination, NOT the (free) viewing angle ``agn_cos_inc``. The
    # disc viewing inclination stays free and shapes the observed SED
    # (foreshortening, Type-1/2 mask), but the *intrinsic* accretion luminosity
    # that anchors alpha_ox / radio-loudness must be inclination-INDEPENDENT, or a
    # fit would let the viewing angle spuriously drive the X-ray/radio normalization.
    # Re-evaluate the disc block at cos(30 deg) (block-agnostic: each disc models
    # its own inclination law). Cheap relative to the full pipeline.
    #
    # Convention note (composable-AGN physics audit): each disc block applies
    # its OWN inclination law here, NOT CIGALE's SKIRTOR-template anisotropy
    # factor eta(i) = cos i (1 + 2 cos i)/3 (skirtor2016.py:405-406). CIGALE's
    # eta(30 deg) = 0.789 is specific to the SKIRTOR intrinsic-disc TEMPLATE
    # (``AGN1.disk``); it is NOT a universal correction for the analytic /
    # physical disc models used here (multicolor, kubota_done, richards2006,
    # ...), which already carry their own foreshortening. Comparing this
    # L_2500_intrinsic to CIGALE's ``intrin_Lnu_2500A_30deg`` therefore shows an
    # ~eta(30 deg) (~27%) offset for a non-SKIRTOR disc — that is a convention
    # difference between disc models, not a bug. Do NOT blindly multiply by
    # eta(30 deg) here (it would double-count inclination for discs that model
    # their own, and be wrong for isotropic ones).
    _COS_30DEG = 0.86602540378443864
    L_lambda_disc_30deg = disc_fn(
        wave,
        agn_log_lbol=agn_log_lbol,
        templates=disc_templates,
        **{**params, "agn_cos_inc": _COS_30DEG},
    )
    L_2500_intrinsic = jnp.interp(2500.0, wave, L_lambda_disc_30deg) * (2500.0**2 / C_AA_PER_S)
    L_4400_intrinsic = jnp.interp(4400.0, wave, L_lambda_disc_30deg) * (4400.0**2 / C_AA_PER_S)

    # Disc extinction (CIGALE skirtor2016.py:341-348). For Type-1 viewing
    # (i <= 90 - oa, i.e. cos_inc >= sin(oa)) the line-of-sight disc is
    # reddened: ``disk *= ext_fac``. The energy the reddening removes is
    # routed to the polar graybody (the SKIRTOR torus block normalizes
    # disc+torus+polar jointly to agn_power — see below). Gated on
    # ``agn_polar_ebv > 0`` → models without polar dust are untouched.
    _polar_ebv = jnp.asarray(params.get("agn_polar_ebv", 0.0))
    _cos_inc = jnp.asarray(params.get("agn_cos_inc", 0.86602540378443864))
    _oa_deg = jnp.asarray(params.get("agn_oa_skirtor", params.get("agn_polar_oa", 40.0)))
    _is_type1 = _cos_inc >= jnp.sin(jnp.deg2rad(_oa_deg))
    _disc_ext = jnp.exp(-0.921 * _polar_ebv * _RV_SMC * smc_extinction_curve(wave))
    _disc_ext = jnp.where(_is_type1 & (_polar_ebv > 0.0), _disc_ext, 1.0)

    # CIGALE single-reference disc normalization (#556). In fracAGN-coupled
    # mode with the SKIRTOR torus, CIGALE ties the disc to the SAME
    # ``agn_power`` as the dust via the fixed template ratio
    # ``R = lumin_disk/lumin_dust`` (skirtor2016.py ``norm = 1/∫dust``), so
    # disc, torus and polar all scale together and the disc-reddening
    # absorbed power is conserved into the polar graybody. ``R`` carries the
    # anisotropy factor ``η(i) = cos(i)(1+2cos(i))/3``. We capture R from the
    # *un-reddened* disc shape here (CIGALE normalizes before reddening) and
    # apply the disc renormalization after the torus block fixes agn_power.
    # Static dispatch on the torus name (JIT-safe); the fracAGN>0 gate is
    # applied branchlessly after the torus block (below).
    _agn_fracAGN = jnp.asarray(params.get("agn_ir_frac", 0.0))
    _disc_R = None
    _disc_incl = None
    # ``agn_norm`` policy: "cigale_joint" (current default) ties disc/torus/
    # polar to the single agn_power reference (only meaningful for the SKIRTOR
    # torus, whose template ratios define R, #556); "conserving" debits the disc
    # by the reprocessed fraction so disc(1-f)+torus(f) conserves L_bol for ALL
    # tori (the energy-ledger debit below) — opt-in for now; it becomes the
    # default once the CIGALE reproduction + recipes pin cigale_joint explicitly
    # (Phase 2), so flipping it here would silently change the CIGALE §9 parity;
    # "independent" keeps the legacy per-component scaling. Static string
    # (JIT-safe Python branch).
    _agn_norm = params.get("agn_norm", "cigale_joint")
    # agn_torus_frac clipped to [0, 1] once and reused by both the conserving
    # disc debit and the cigale_joint SKIRTOR R-tie fallback, so the default
    # (0.5) can never drift between the sites that debit the disc.
    _torus_frac = jnp.clip(jnp.asarray(params.get("agn_torus_frac", 0.5)), 0.0, 1.0)
    if _agn_norm == "cigale_joint" and agn_torus_block == "skirtor":
        _disc_R, _disc_incl, _disc_R_faceon = skirtor_disc_dust_ratio(
            wave,
            L_lambda_disc,
            _disc_ext,
            agn_tau_skirtor=params.get("agn_tau_skirtor", 7.0),
            agn_p_skirtor=params.get("agn_p_skirtor", 1.0),
            agn_q_skirtor=params.get("agn_q_skirtor", 1.0),
            agn_oa_skirtor=params.get("agn_oa_skirtor", 40.0),
            agn_cos_inc=_cos_inc,
        )
        # #556 mechanism 3 — tie the polar ``l_ext`` to the SAME agn_power as
        # the disc. The SKIRTOR torus block estimates the absorbed disc power
        # from a FACE-ON disc proxy; in fracAGN mode it must use the
        # agn_power-tied face-on disc ``agn_power·R/η`` (not the legacy
        # ``18/7·10^agn_log_lbol`` which assumes agn_log_lbol = intrinsic 4π
        # power, over-estimating l_ext ~2× → FIR +15%). ``agn_power`` here is
        # the L_absorbed-coupled value the torus block normalizes to
        # (``agn_torus_frac × L_bol``), available *before* the torus runs so
        # there is no circular dependency on ∫torus. Branchless: fall back to
        # the legacy proxy where ``agn_ir_frac == 0``.
        _Lbol = 10.0**agn_log_lbol * L_SUN
        _agn_power_pre = _torus_frac * _Lbol
        # Face-on UN-reddened disc ∫AGN1.disk = agn_power × R_faceon (the
        # ratio the CIGALE l_ext proxy needs), NOT agn_power × R (which is the
        # reddened, inclination-weighted *observed* disc).
        _faceon_frac = _agn_power_pre * _disc_R_faceon
        _faceon_proxy = _Lbol * (18.0 / 7.0)
        _faceon = jnp.where(_agn_fracAGN > 0.0, _faceon_frac, _faceon_proxy)
        params = {
            **params,
            "agn_disc_faceon_lbol": jnp.log10(jnp.maximum(_faceon / L_SUN, 1e-30)),
        }

    L_lambda_disc = L_lambda_disc * _disc_ext

    # Compute lambda*L_lambda(5100Å) for downstream block (line/FeII/torus)
    # normalizations. Convention: this is the LOS-reddened disc — taken *after*
    # the polar/LOS disc extinction (``_disc_ext`` above) but *before* the
    # conserving debit below. With agn_polar_ebv=0 (the common case) it equals
    # the intrinsic disc; with Type-1 polar reddening it carries the extinction.
    # GRAHSP l5100 parity (Phase 2) should confirm and pin the intended anchor.
    l5100_disc = jnp.interp(5100.0, wave, L_lambda_disc) * 5100.0

    # ── Energy ledger (energy-conserving policies) ───────────────────────
    # The disc carries the intrinsic L_bol; the torus reprocesses a fraction of
    # it. Debit the observed disc by (1 - agn_torus_frac) so that
    # disc(1-f) + torus(f) conserves L_bol for every torus — reproducing the
    # monolithic models (e.g. silva04_agn passes agn_lum_ratio=1-agn_torus_frac to
    # the disc). The torus block already normalizes its output to
    # agn_torus_frac * L_bol, so only the disc side changes.
    #
    # CONSERVATION DOMAIN: exact only when agn_polar_ebv=0. With Type-1 polar
    # reddening, ``_disc_ext`` (above) removes disc UV that nothing re-credits
    # under this policy (the polar-graybody re-credit currently lives in the
    # cigale_joint branch). So "conserving" guarantees Sigma=L_bol iff
    # agn_polar_ebv=0; the reddening unification (Phase 3) wires the re-credit
    # here so the guarantee becomes unconditional.
    #
    # Self-contained tori (``none``, ``qsogen``, ``grahsp``) bundle disc+torus
    # in one self-normalized template and bypass the ledger — no debit. This
    # also covers the disc-only (``torus="none"``) case: with no reprocessor,
    # the disc keeps its full L_bol.
    #
    # Which policies debit: "conserving" always; "cigale_joint" too EXCEPT for
    # the SKIRTOR torus, which instead uses the agn_power×R template tie (Stage
    # 4 below) — the CIGALE-faithful path. So cigale_joint is energy-conserving
    # for *every* torus (R-tie for skirtor, agn_torus_frac split otherwise),
    # never the silent additive leak it used to be for non-skirtor tori.
    # "independent" never debits (each component on its own luminosity scale).
    # Static Python branch on the policy string + torus name (JIT-safe).
    _conserve_via_debit = agn_torus_block not in _SELF_CONTAINED_TORI and (
        _agn_norm == "conserving" or (_agn_norm == "cigale_joint" and agn_torus_block != "skirtor")
    )
    # Intrinsic (pre-reprocessor) disc shape, captured before any debit so the
    # line-energy debit below (#929) can subtract exactly the integrated line
    # energy additively with the torus debit.
    _disc_intrinsic = L_lambda_disc
    if _conserve_via_debit:
        L_lambda_disc = L_lambda_disc * (1.0 - _torus_frac)

    # Stage 2a: narrow-line region.
    nlr_fn = resolve_agn_block("nlr", agn_nlr_block)
    nlr_aniso, nlr_iso = split_lines_result(
        nlr_fn(
            wave,
            agn_log_lbol=agn_log_lbol,
            l5100_disc=l5100_disc,
            templates=_templates_for("nlr", agn_nlr_block),
            **params,
        )
    )
    # Stage 2b: broad-line region.
    blr_fn = resolve_agn_block("blr", agn_blr_block)
    blr_aniso, blr_iso = split_lines_result(
        blr_fn(
            wave,
            agn_log_lbol=agn_log_lbol,
            l5100_disc=l5100_disc,
            templates=_templates_for("blr", agn_blr_block),
            **params,
        )
    )
    L_lambda_lines_aniso = nlr_aniso + blr_aniso
    L_lambda_lines_iso = nlr_iso + blr_iso

    # Stage 3: FeII forest.
    feii_fn = resolve_agn_block("feii", agn_feii_block)
    L_lambda_feii = feii_fn(
        wave,
        agn_log_lbol=agn_log_lbol,
        l5100_disc=l5100_disc,
        templates=_templates_for("feii", agn_feii_block),
        **params,
    )

    # Line-energy debit (#929, the Sigma-f ledger). The NLR/BLR/FeII lines are
    # reprocessed disc photons, so under the *conserving* ledger they must be
    # debited from the disc, not stacked on a full-luminosity disc (which
    # inflates the total above L_bol). Subtract exactly the integrated line
    # energy, shaped as the intrinsic disc — additive with the torus debit
    # (disc -> 1 - f_torus - f_lines), matching Synthesizer's covering-fraction
    # dimming. Scoped to "conserving": cigale_joint follows CIGALE (nebular added
    # separately, allocation-conserving) and independent keeps each component on
    # its own luminosity scale. Excludes only the self-normalized bundled
    # templates (grahsp/qsogen carry disc+torus+lines in one template) — NOT
    # ``torus="none"``, whose disc and lines are still real ledger emission.
    # E_disc guards a zero/near-zero disc (e.g. agn_disc_block="none") so the
    # ratio never blows up.
    if _agn_norm == "conserving" and agn_torus_block not in ("grahsp", "qsogen"):
        _e_lines = jnp.trapezoid(L_lambda_lines_aniso + L_lambda_lines_iso + L_lambda_feii, wave)
        _e_disc = jnp.maximum(jnp.trapezoid(_disc_intrinsic, wave), 1e-30)
        L_lambda_disc = L_lambda_disc - (_e_lines / _e_disc) * _disc_intrinsic

    # Stage 4: IR torus.
    torus_fn = resolve_agn_block("torus", agn_torus_block)
    L_lambda_torus = torus_fn(
        wave,
        agn_log_lbol=agn_log_lbol,
        l5100_disc=l5100_disc,
        templates=_templates_for("torus", agn_torus_block),
        **params,
    )

    # CIGALE single-reference disc normalization (#556), part 2. The SKIRTOR
    # torus block fixes ``agn_power = ∫L_lambda_torus`` (disc+torus+polar share
    # this budget). Two regimes, selected branchlessly by the *traced*
    # ``agn_ir_frac`` (so this cannot join the static _conserve_via_debit gate):
    #   * fracAGN > 0 (CIGALE-coupled): tie the disc to ``agn_power × R`` so
    #     disc/torus/polar share one reference — *allocation*-conserving (the
    #     components can't drift apart), CIGALE-faithful, inclination-correct via
    #     the η(i) baked into R. This is NOT *ledger* conservation: ∫total scales
    #     with ``agn_power = agn_torus_frac·L_bol``, so agn_torus_frac→0 drives
    #     the whole AGN to zero — outside CIGALE's reachable domain, but a free
    #     agn_torus_frac sampler can reach that degenerate zero-AGN plateau.
    #   * fracAGN = 0 (default): no CIGALE coupling, so debit the disc by
    #     (1 − agn_torus_frac) exactly like the ``conserving`` policy — *ledger*
    #     conservation (∫total = L_bol). This closes the leak that used to hit
    #     the DEFAULT skirtor config, where neither the R-tie nor the
    #     _conserve_via_debit gate (which excludes skirtor) fired.
    if _disc_R is not None:
        _agn_power = jnp.trapezoid(L_lambda_torus, wave)
        # Apply the wavelength-dependent ``disk(i)/disk(0)`` inclination
        # attenuation to the disc *shape* (CIGALE ``SKIRTOR.disk(i)/AGN1.disk(0)``)
        # so the disc spectrum is inclination-correct, then renormalize the
        # reweighted shape to the agn_power-tied bolometric ``agn_power × R``.
        _disc_reweighted = L_lambda_disc * _disc_incl
        _disc_int = jnp.maximum(jnp.trapezoid(_disc_reweighted, wave), 1e-30)
        _disc_scaled = _disc_reweighted * (_agn_power * _disc_R) / _disc_int
        _disc_debited = L_lambda_disc * (1.0 - _torus_frac)
        L_lambda_disc = jnp.where(_agn_fracAGN > 0.0, _disc_scaled, _disc_debited)

    # Stage 4.5: Type-1/2 obscuration of the *anisotropic* central engine (disc +
    # broad lines + FeII). The isotropic NLR is added back afterwards, so it stays
    # visible at all inclinations. Each torus carries ONE obscuration model (no
    # double-counting): dusty-screen tori (fritz/skirtor, #294) apply a
    # wavelength-dependent screen; every other non-"none" torus applies the gray
    # geometric visibility mask — the same one the monolithic ``unified_nlr_blr``
    # uses — so a composable disc+torus+NLR+BLR reproduces its Type-1/2 geometry.
    # Defaults (i=30, theta_torus=30 -> inc_crit=60 > i) give mask ~ 1, so
    # default-inclination models are unchanged. Static dispatch on the torus name
    # is JIT-safe.
    L_lambda_central = L_lambda_disc + L_lambda_lines_aniso + L_lambda_feii
    if agn_torus_block in TORUS_SCREEN_PARAMS:
        _oa_key, _tau_key = TORUS_SCREEN_PARAMS[agn_torus_block]
        screen = torus_screen_transmission(
            wave,
            cos_inc=params.get("agn_cos_inc", 0.86602540378443864),
            oa_deg=params.get(_oa_key, 40.0),
            tau_v=params.get(_tau_key, 7.0),
        )
        L_lambda_central = L_lambda_central * screen
    elif agn_torus_block not in _SELF_CONTAINED_TORI:
        vis = sigmoid_visibility_mask(
            params.get("agn_cos_inc", 0.86602540378443864),
            params.get("agn_theta_torus", 30.0),
        )
        L_lambda_central = L_lambda_central * vis
    # Isotropic NLR: visible at every inclination, so added after the mask.
    L_lambda_central = L_lambda_central + L_lambda_lines_iso

    # Stage 5: attenuation factor (multiplicative; host/foreground screen).
    atten_fn = resolve_agn_block("attenuation", agn_attenuation_block)
    factor = atten_fn(wave, **params)

    L_lambda_total = (L_lambda_central + L_lambda_torus) * factor

    # Convert to L_nu [erg/s/Hz] using L_nu = L_lambda * lambda^2 / c.
    L_nu_atten = L_lambda_total * wave**2 / C_AA_PER_S

    # Stage 6 (conditional): polar-dust reemission.
    # When polar_dust attenuation is selected, the absorbed photons are re-emitted
    # as a geometry-independent FIR graybody. Compute and add this to the SED.
    # Static dispatch on agn_attenuation_block (a Python string) is JIT-safe.
    if agn_attenuation_block == "polar_dust":
        # Compute reemission in L_nu from the pre-attenuation SED (torus-screened
        # central engine + torus IR).
        L_lambda_pre_atten = L_lambda_central + L_lambda_torus
        L_nu_reemit = polar_dust_reemission_lnu(wave, L_lambda_pre_atten, **params)
        L_nu_result = L_nu_atten + L_nu_reemit
    else:
        L_nu_result = L_nu_atten

    # Return with optional L_2500_intrinsic and L_4400_intrinsic tuple.
    if return_l2500:
        return (L_nu_result, L_2500_intrinsic, L_4400_intrinsic)
    else:
        return L_nu_result


def composable_agn_l_nu(
    wavelength: Array,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_disc_block: str = "none",
    agn_nlr_block: str = "none",
    agn_blr_block: str = "none",
    agn_feii_block: str = "none",
    agn_torus_block: str = "none",
    agn_attenuation_block: str = "none",
    template_state: dict | None = None,
    return_l2500: bool = False,
    **params,
) -> Array | tuple[Array, float]:
    r"""AGN_MODELS["composable"] entry point — :data:`L_ν` in erg/s/Hz.

    Thin wrapper around :func:`compose_l_nu` matching the AGN_MODELS
    registry signature::

        fn(wavelength, agn_log_lbol, agn_lum_ratio, **kwargs) -> L_nu

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float, optional
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`. Defaults to the declared
        ``agn_log_lbol`` default.
    agn_lum_ratio : float, optional
        Overall AGN fraction scaling [dimensionless]. Default ``1.0``.
    agn_disc_block, agn_nlr_block, agn_blr_block, agn_feii_block, \
agn_torus_block, agn_attenuation_block : str, optional
        Per-stage block selectors. Default ``"none"`` for every stage
        (a no-op pipeline; users **must** opt in by name).
    return_l2500 : bool, optional
        When True, return ``(L_nu, L_2500_intrinsic, L_4400_intrinsic)``
        tuple. When False (default), return only ``L_nu`` for backward
        compatibility. Default: False.
    **params
        Per-impl free parameters forwarded to every block.

    Returns
    -------
    L_nu : ndarray, shape (n_wave,)
        Total AGN :math:`L_\nu` [erg/s/Hz], scaled by ``agn_lum_ratio``.
    L_2500_intrinsic : float, optional
        When ``return_l2500=True``, the un-reddened intrinsic disc
        monochromatic luminosity at 2500 Å [erg/s/Hz]. NOT scaled by
        ``agn_lum_ratio`` (maintains the unscaled-intrinsic convention of
        ``L_agn_bol``). Returned as second element of tuple when enabled.
    L_4400_intrinsic : float, optional
        When ``return_l2500=True``, the un-reddened intrinsic disc
        monochromatic luminosity at 4400 Å [erg/s/Hz]. NOT scaled by
        ``agn_lum_ratio`` (maintains the unscaled-intrinsic convention of
        ``L_agn_bol``). Returned as third element of tuple when enabled.

    Notes
    -----
    JIT-compatible (block selectors are static). Validation runs once at
    Python entry, so the JIT cache picks up changes only on selector
    changes (which trigger a recompile anyway).

    The returned ``L_2500_intrinsic`` and ``L_4400_intrinsic`` (when
    ``return_l2500=True``) are NOT scaled by ``agn_lum_ratio``, matching the
    normalization convention of ``L_agn_bol``. This allows downstream
    components (e.g. X-ray, radio) to scale the monochromatic luminosities
    independently.
    """
    # Recipe validation runs at *construction* time (Parameters.__init__ and
    # Recipe.__post_init__), where selectors and params are concrete Python
    # values. It must NOT run here: composable_agn_l_nu is called inside the
    # jitted forward pass, so any param-value inspection (Rule 7 reads
    # agn_polar_ebv) would hit a JAX tracer and raise ConcretizationTypeError.
    # See validate_block_recipe's docstring ("runs at composition time, not
    # under JIT").
    result = compose_l_nu(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_disc_block=agn_disc_block,
        agn_nlr_block=agn_nlr_block,
        agn_blr_block=agn_blr_block,
        agn_feii_block=agn_feii_block,
        agn_torus_block=agn_torus_block,
        agn_attenuation_block=agn_attenuation_block,
        template_state=template_state,
        return_l2500=return_l2500,
        **params,
    )
    if return_l2500:
        L_nu, L_2500_intrinsic, L_4400_intrinsic = result
        return (agn_lum_ratio * L_nu, L_2500_intrinsic, L_4400_intrinsic)
    else:
        return agn_lum_ratio * result
