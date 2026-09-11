# SPDX-License-Identifier: BSD-3-Clause
"""Composable AGN runner: picks one block per pipeline stage and runs them.

Canonical execution order (paper §2.1.6 / upstream GRAHSP module ordering)::

    [disc] → [nlr] → [blr] → [feii] → [torus] → [attenuation]

Each stage is owned by a registered block from
``tengri.components.agn.blocks._protocol``.

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

from tengri.components.agn.blocks._grid_support import (
    block_grid_support,
    describe_clipping,
)
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
from tengri.components.agn.reddening import redden_disc
from tengri.components.agn.skirtor import SKIRTORBundle, skirtor_disc_dust_ratio
from tengri.config.exceptions import AdvisoryWarning

#: Torus selectors that do NOT receive the gray Type-1/2 visibility mask:
#: ``none`` (no torus) and the self-contained empirical quasar templates
#: (``qsogen``, ``grahsp``), which already encode an inclination-averaged SED;
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


class RecipeWarning(AdvisoryWarning):
    """Emitted by :func:`validate_block_recipe` for suspicious block combos.

    Callers can still ``warnings.simplefilter("error", RecipeWarning)`` to turn
    recipe issues into hard errors during development without affecting other
    warnings: :class:`~tengri.config.exceptions.AdvisoryWarning` derives from
    :class:`UserWarning`, so existing filters keep matching.

    It subclasses ``AdvisoryWarning`` because it is exactly that: a statement
    about a model the caller is *building*. The paths that construct a
    throwaway ``Parameters``: recipe introspection, and the structural spec
    that only enumerates declared names: silence that category wholesale, so
    these no longer fire on ``import tengri`` or describe a pre-narrowing range
    that has already been superseded (#1586).
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
    # ``nlr='analytic'`` is deliberately absent. ``nlr_analytic_block`` is
    # illuminated by the intrinsic bolometric ``10**agn_log_lbol`` and its body
    # opens with ``del l5100_disc``, so it is not disc-anchored and Rule 4
    # naming it was a false advisory -- R48's own rule, that a block which does
    # not normalize off the disc's 5100 A luminosity must not be listed here.
    # Measured with every other slot off, marginal ``sum|sed_agn|`` over
    # 500 A - 1 mm at ``agn_log_lbol=12``:
    #
    #   block               disc='none'     disc='multicolor'
    #   nlr='analytic'      2.022386e+31    2.022386e+31   <- bit-identical
    #   nlr='grahsp'        0.000000e+00    1.642265e+31
    #   blr='analytic'      0.000000e+00    4.686986e+30
    #   blr='grahsp'        0.000000e+00    7.246841e+31
    #   feii='grahsp'       0.000000e+00    5.363636e+30
    #   torus='grahsp'      0.000000e+00    2.493180e+34
    #
    # Every listed entry goes to exactly zero without a disc; ``analytic`` is
    # the one that does not move at all.
    "nlr": frozenset({"grahsp"}),
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

# Grid-tabulated discs: node-exact template libraries whose crossval tests
# (test_slone_netzer_vs_agnfitter.py / test_kd18_grid_vs_agnfitter.py) confirm
# the standard UV/optical accretion-disc peak (< 1 um in L_nu), so 5100A is a
# meaningful continuum for all three, not an edge case like ADAF's inner flow.
_DISCS_WITH_5100A_CONTINUUM = _DISCS_WITH_5100A_CONTINUUM | frozenset(
    {
        "slone_netzer",
        "kd18_agnfitter",
        "kd18_agnfitter_warmindex",
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
#: silently emit garbage: users must opt in to each block by name.
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
    agn_norm: str | None = None,
    params: dict | None = None,
    param_support: dict[str, tuple[float, float]] | None = None,
) -> list[str]:
    r"""Check a block recipe for suspicious / unphysical combinations.

    Emits a :class:`RecipeWarning` per issue found and returns the list of
    issue strings (so tests can introspect them deterministically).

    Validation runs at composition time (Python-side, not under JIT), so
    cost is paid once per recipe construction: there is no inner-loop
    overhead.

    Rules implemented
    -----------------
    1. **Unknown block name**, the selector points at a block that is not
       registered; raise ``ValueError`` rather than warn (typo == hard
       error so users notice immediately).
    2. **All-none recipe**: every selector is ``"none"``; output will be
       identically zero. Almost certainly a misuse.
    3. **No disc, active anchored downstream**: disc is ``"none"`` but an
       nlr / blr / feii / torus block that actually *reads* the disc's
       :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` for its normalization
       (:data:`_DOWNSTREAM_NEEDS_L5100`) is selected. That block scales by
       the disc's 5100Å luminosity (zero), so it emits zero too. Either the
       user forgot to pick a disc impl, or the recipe is genuinely
       degenerate. **Not every torus/nlr/blr/feii block anchors this way**:
       most torus impls (``cat3d_wind``, ``skirtor``, ``simple``,
       ``qsogen``, ...) normalize off ``agn_log_lbol``/``agn_torus_frac``
       directly and emit non-zero flux with ``agn_disc_block='none'``
       (measured: ``cat3d_wind`` under both ``agn_norm='independent'`` and
       ``agn_norm='cigale_joint'`` sums ``sed_agn_torus`` to 3.849e34, never
       zero) -- naming them here would be a false advisory (R48). The one
       torus exception is ``"grahsp"``, whose block body reads
       ``l5100_disc`` directly (measured: it correctly goes to zero with
       ``agn_disc_block='none'``). ``agn_norm`` is accepted so this check
       can become policy-aware if a future norm policy changes which blocks
       anchor to the disc; measured at this HEAD, no registered policy does
       -- every block's anchoring is intrinsic to its own implementation,
       not to ``agn_norm`` (``cigale_joint``'s disc/torus tie only changes
       how the *disc* is normalized when ``agn_torus_block='skirtor'``, not
       whether a downstream block reads ``l5100_disc``).
    4. **GRAHSP downstream + non-5100Å disc**; GRAHSP nlr / blr / feii / torus
       expect the disc to deliver a meaningful UV/optical continuum at
       5100Å. Pairing them with an exotic disc (e.g. pure ADAF) likely
       produces an unintended SED.
    5. **GRAHSP biatten with no GRAHSP body**, the SMC-Prevot curve is
       generic, so this is technically valid; warn that the user might
       prefer the more clearly named ``"smc_prevot"`` block (when wrapped
       in a future PR).
    6. **NLR / BLR without UV/optical disc**: these lines blocks convert
       :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` to a bolometric disc
       luminosity via the Krawczyk+ 2013 correction. A non-5100Å disc
       triggers the same warning as rule 4.
    7. **Polar-dust block with E(B-V)=0**, the ``polar_dust`` attenuation
       block is a no-op when ``agn_polar_ebv = 0``; warn to surface unset
       params before the user wonders why the SED is unattenuated.
    9. **Support wider than the block's template grid**, a template-backed
       block interpolates over fixed axes and *clips* outside them, so the
       excess is bit-identical to the edge node and its gradient is exactly
       zero. A prior wider than the grid therefore contains parameter space
       a fit can never move through, silently. Warn with the live fraction
       and the grid extent. See
       ``tengri.components.agn.blocks._grid_support`` for why this cannot
       be expressed on the parameter declaration itself (#1586).

    Parameters
    ----------
    agn_disc_block, agn_nlr_block, agn_blr_block, agn_feii_block, \
agn_torus_block, agn_attenuation_block : str
        Selectors for each pipeline stage.
    agn_norm : str, optional
        Cross-block normalization policy (``"independent"`` or
        ``"cigale_joint"``, see :attr:`AGNSEDComponentConfig.agn_norm`).
        Consumed by Rule 3 so the disc-anchoring check can become
        policy-aware if a future policy changes which blocks read
        ``l5100_disc``; measured at this HEAD, none does (see Rule 3).
        ``None`` (default) is treated the same as either policy.
    params : dict, optional
        Concrete parameter values, used by Rule 7 to surface a no-op
        ``agn_polar_ebv``. Values may legitimately be absent or traced.
    param_support : dict[str, tuple[float, float]], optional
        ``{param_name: (lo, hi)}``, the range each parameter can actually
        take, i.e. a prior's bounds or ``(v, v)`` for a fixed value. Consumed
        by Rule 9; when omitted, that rule is skipped.

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
            "Composable AGN: every block selector is 'none', the AGN SED "
            "will be identically zero. Pick at least a disc block to "
            "produce non-trivial output."
        )

    # Rule 3: no disc, active ANCHORED downstream. Only the (category, name)
    # pairs in _DOWNSTREAM_NEEDS_L5100 actually read l5100_disc for their
    # normalization (measured per-block, see the module docstring's Rule 3
    # entry): most torus/nlr/blr/feii impls normalize off agn_log_lbol
    # directly and emit non-zero flux even with disc='none', so naming them
    # here would be a false advisory (R48 -- the prior version named every
    # active torus block regardless). ``agn_norm`` is accepted for a future
    # policy-aware check but does not change any block's anchoring today.
    del agn_norm  # unused: measured, no registered agn_norm policy changes anchoring
    anchored_active = [
        f"{cat}={selectors[cat]!r}"
        for cat in ("nlr", "blr", "feii", "torus")
        if selectors[cat] != "none"
        and selectors[cat] in _DOWNSTREAM_NEEDS_L5100.get(cat, frozenset())
    ]
    if selectors["disc"] == "none" and anchored_active:
        _emit(
            f"Composable AGN: agn_disc_block='none' but downstream "
            f"blocks are active ({', '.join(anchored_active)}). These blocks "
            f"normalize to lambda*L_lambda(5100A) of the disc, which "
            f"is zero, the active blocks will emit zero. Pick a disc "
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
    # agn_polar_ebv raises on float(): ConcretizationTypeError is a TypeError
    # subclass, and needs no no-op warning, since the user is explicitly
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
    # Mahadevan 1997 ADAF rewrite landed in #898, the block is now production.)

    # Rule 9: a template-backed block's grid axes are a SECOND support that no
    # parameter declaration records. Outside them jnp.clip is flat, so the SED
    # is bit-identical and the gradient is exactly 0.0, a fit gets no signal
    # and cannot move the parameter, with nothing raised or warned (#1586).
    # Checked per (block, param) because the same parameters are shared with
    # grid-free analytic discs that legitimately want the wider support.
    if param_support:
        for category, name in selectors.items():
            for pname, (g_lo, g_hi) in block_grid_support(category, name).items():
                active = param_support.get(pname)
                if active is None:
                    continue
                # One definition of the overhang wording, shared with every
                # other template-backed component (tengri.components.grid_support).
                detail = describe_clipping(active, (g_lo, g_hi))
                if detail is None:
                    continue  # no reachable value can be clipped
                extent = f"[{g_lo:g}, {g_hi:g}]"
                _emit(
                    f"Composable AGN: {pname} with the {name!r} {category} "
                    f"block: {detail}. The SED there is bit-identical to the "
                    "edge node and the gradient is exactly zero, so a fit "
                    f"cannot move it. Narrow {pname} to {extent}, or select a "
                    f"{category} block with no template grid."
                )

    return issues


def _agn_sed_components(
    *,
    L_lambda_disc: Array,
    L_lambda_torus: Array,
    L_lambda_lines_aniso: Array,
    L_lambda_feii: Array,
    L_lambda_lines_iso: Array,
    central_mask: Array | float,
    atten_factor: Array,
    torus_factor: Array | float,
    l_nu_conv: Array,
    L_nu_polar: Array,
) -> dict[str, Array]:
    r"""Per-sub-block rest-frame :math:`L_\nu` SEDs (NAMING_CONTRACT §4b.5).

    Pure decomposition of the SAME additive pieces :func:`compose_l_nu`
    folds into its un-decomposed ``L_nu_result``: distributes
    ``central_mask`` and ``atten_factor`` over the ``(disc + aniso-lines)``
    sum individually instead of multiplying the combined bundle, so the
    four returned arrays sum EXACTLY (to floating-point reassociation) back
    to that same total (guarded by the 1e-12-relative sum contract test).
    Extracted as its own function purely for readability -- no behavior
    change from inlining it at the call site.

    Parameters
    ----------
    L_lambda_disc, L_lambda_torus : array_like, shape (n_wave,)
        Post-Stage-4 disc and torus :math:`L_\lambda` [erg/s/Å].
    L_lambda_lines_aniso, L_lambda_feii : array_like, shape (n_wave,)
        Anisotropic (Stage-4.5-masked) NLR+BLR and FeII :math:`L_\lambda`
        [erg/s/Å].
    L_lambda_lines_iso : array_like, shape (n_wave,)
        Isotropic (unmasked) NLR :math:`L_\lambda` [erg/s/Å].
    central_mask : array_like or float
        Stage-4.5 Type-1/2 obscuration factor applied to disc + aniso-lines.
    atten_factor : array_like, shape (n_wave,)
        Stage-5 attenuation-block multiplicative factor, applied to the
        central engine (disc + lines).
    torus_factor : array_like or float
        What multiplies the torus. Equal to ``atten_factor`` for a genuine
        foreground screen; for the ``polar_dust`` block it is instead the
        scalar budget share ``1 - share`` under the joint/conserving
        policies, or ``1.0`` under ``'independent'`` -- the polar screen
        reddens the disc, not the torus IR (R61), and the torus is rescaled
        only because the two share one dust budget (R59).
    l_nu_conv : array_like, shape (n_wave,)
        :math:`\lambda^2/c` conversion factor, L_lambda -> L_nu [Hz/Å].
    L_nu_polar : array_like, shape (n_wave,)
        Stage-6 polar-dust re-emission :math:`L_\nu` [erg/s/Hz] (zeros
        when inactive).

    Returns
    -------
    dict
        ``{"disc", "torus", "lines", "polar"}`` -> ndarray, shape
        ``(n_wave,)``, each in :math:`L_\nu` [erg/s/Hz]. ``"lines"`` is
        NLR + BLR + FeII combined.

    Notes
    -----
    **JIT-compatible**: yes, pure JAX arithmetic.
    """
    L_lambda_lines_total = (
        L_lambda_lines_aniso + L_lambda_feii
    ) * central_mask + L_lambda_lines_iso
    return {
        "disc": L_lambda_disc * central_mask * atten_factor * l_nu_conv,
        "torus": L_lambda_torus * torus_factor * l_nu_conv,
        "lines": L_lambda_lines_total * atten_factor * l_nu_conv,
        "polar": L_nu_polar,
    }


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
    return_components: bool = False,
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
        ``load_*_templates()`` helper at trace time: keeps HDF5 / file
        I/O out of the JIT trace boundary. ``None`` (default) falls back
        to the in-block lru_cache load.
    return_l2500 : bool, optional
        When True, return ``(L_nu, L_2500_intrinsic, L_4400_intrinsic)``
        tuple. When False (default), return only ``L_nu`` for backward
        compatibility with existing single-return callers. Default: False.
    return_components : bool, optional
        When True, additionally return a ``components`` dict with keys
        ``"disc"``, ``"torus"``, ``"lines"`` (nlr + blr + feii), ``"polar"``
        -- the public per-sub-block rest-frame SEDs (NAMING_CONTRACT §4b.5,
        published as ``sed_agn_disc``/``sed_agn_torus``/``sed_agn_lines``/
        ``sed_agn_polar`` by :class:`~tengri.components.agn.component.AGNSEDComponent`).
        The four arrays sum to ``L_nu`` (to floating-point reassociation).
        Combines orthogonally with ``return_l2500``: the ``components`` dict
        is always the LAST element of the returned tuple. Default: False.
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
    components : dict, optional
        When ``return_components=True``, ``{"disc", "torus", "lines",
        "polar"}`` -> ndarray, shape ``(n_wave,)``, each in :math:`L_\nu`
        [erg/s/Hz]. Returned as the last element of the tuple. Otherwise
        not returned.

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
    # time, which bakes it into the graph as constants.
    #
    # Keys are ``"<category>/<name>"``, matching ``collect_block_templates``.
    # ``"grahsp"`` is still honored so callers holding the old flat bundle
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
    # path so every disc block respects it: previously only the monolithic
    # forward models reddened, so composable-routed presets (adaf, kubota_done_full)
    # silently ignored agn_ebv_disc (#916). No-op at the default agn_ebv_disc=0.
    # The intrinsic L_2500/L_4400 below recompute from the un-reddened disc block,
    # so the X-ray/radio anchors stay obscuration-independent.
    L_lambda_disc = redden_disc(wave, L_lambda_disc, jnp.asarray(params.get("agn_ebv_disc", 0.0)))

    # Capture L_2500_intrinsic and L_4400_intrinsic: the un-reddened,
    # agn_log_lbol-normalized disc monochromatic luminosities [erg/s/Hz] that
    # drive X-ray alpha_ox and radio loudness. These follow CIGALE's
    # ``intrin_Lnu_2500A_30deg`` convention: they are evaluated at a fixed 30 deg
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
    # ~eta(30 deg) (~27%) offset for a non-SKIRTOR disc; that is a convention
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

    # R22 (ONE polar-dust mechanism): before this fix, polar-dust LOS
    # reddening of the disc applied HERE unconditionally whenever
    # ``agn_polar_ebv > 0`` -- regardless of the selected
    # ``agn_attenuation_block`` (even ``"none"``) and regardless of
    # ``agn_norm`` -- with no re-emission credit, while a SEPARATE bundled
    # graybody term inside the SKIRTOR torus block AND the standalone
    # ``polar_dust`` attenuation block (Stage 5/6 below) could also apply.
    # Measured: at ``atten="none"``, ``norm="independent"``, this stage alone
    # removed 76% of the AGN SED when ``agn_polar_ebv`` went 0 -> 0.3, with
    # torus="skirtor" + atten="polar_dust" double-screening the disc on top.
    # Now there is exactly ONE mechanism: the ``polar_dust`` attenuation
    # block owns LOS reddening (Type-1 sightlines only, via the smooth
    # Type-1/2 mask) AND isotropic re-emission end to end (Stage 5/6 below).
    # Polar dust therefore applies ONLY when ``agn_attenuation_block ==
    # "polar_dust"`` is explicitly selected -- a deliberate behavior change:
    # the ``agn_polar_ebv`` default (0.03) no longer silently reddens every
    # AGN. See task13 fix-round-1 report for the before/after measurements.

    # CIGALE single-reference disc normalization (#556). In fracAGN-coupled
    # mode with the SKIRTOR torus, CIGALE ties the disc to the SAME
    # ``agn_power`` as the dust via the fixed template ratio
    # ``R = lumin_disk/lumin_dust`` (skirtor2016.py ``norm = 1/∫dust``), so
    # disc and torus scale together. ``R`` carries the anisotropy factor
    # ``η(i) = cos(i)(1+2cos(i))/3``. Captured from the UN-reddened disc
    # shape: R22 removed the ONLY LOS-reddening path that used to reach this
    # far (the polar screen now lives exclusively in the standalone
    # ``polar_dust`` attenuation block, downstream of this R-tie), so the
    # reddening factor ``skirtor_disc_dust_ratio`` takes is always the
    # identity now.
    _agn_fracAGN = jnp.asarray(params.get("agn_ir_frac", 0.0))
    _cos_inc = jnp.asarray(params.get("agn_cos_inc", 0.86602540378443864))
    _disc_R = None
    _disc_incl = None
    _disc_R_faceon = None
    _disc_shape_faceon = None
    _disc_wave_native = None
    # ``agn_norm`` policy: "cigale_joint" (current default) ties disc/torus to
    # the single agn_power reference (only meaningful for the SKIRTOR torus,
    # whose template ratios define R, #556); "conserving" debits the disc
    # by the reprocessed fraction so disc(1-f)+torus(f) conserves L_bol for ALL
    # tori (the energy-ledger debit below): opt-in for now; it becomes the
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
        _skirtor_bundle = _templates_for("torus", agn_torus_block)
        _disc_tie = skirtor_disc_dust_ratio(
            wave,
            L_lambda_disc,
            jnp.ones_like(wave),
            _template=(
                _skirtor_bundle.disc_dust if isinstance(_skirtor_bundle, SKIRTORBundle) else None
            ),
            agn_tau_skirtor=params.get("agn_tau_skirtor", 7.0),
            agn_p_skirtor=params.get("agn_p_skirtor", 1.0),
            agn_q_skirtor=params.get("agn_q_skirtor", 1.0),
            agn_oa_skirtor=params.get("agn_oa_skirtor", 40.0),
            # R70: the grid's third geometry axis. Left out, the tie
            # interpolated the SKIRTOR templates at the signature default
            # R = 20 while the torus block honored the model's value, so
            # int(polar)/int(torus) came out bit-identical (2.605153276 at
            # i=80) across R = 10, 20 and 30. Since R64 put norm(0)/norm(i)
            # into R_faceon that also picked the wrong normalization: the
            # factor is R-dependent (2.896205 / 3.172626 / 3.338009 at i=80
            # for R = 10 / 20 / 30, a 15% spread).
            agn_radius_ratio=params.get("agn_radius_ratio", 20.0),
            agn_cos_inc=_cos_inc,
        )
        _disc_R = _disc_tie.R
        _disc_incl = _disc_tie.incl_ratio
        _disc_R_faceon = _disc_tie.R_faceon
        # The face-on reference travels with the grid it was normalized on --
        # see the Stage-6 polar block and :class:`SkirtorDiscTie`.
        _disc_shape_faceon = _disc_tie.faceon_shape_native
        _disc_wave_native = _disc_tie.wave_native

    # Compute lambda*L_lambda(5100Å) for downstream block (line/FeII/torus)
    # normalizations. Convention: this is the intrinsic (un-reddened) disc,
    # before the conserving debit below -- R22 removed the only LOS-reddening
    # path that used to reach this point, so l5100_disc no longer carries any
    # polar-dust extinction (LOS reddening now lives exclusively downstream,
    # in the standalone ``polar_dust`` attenuation block).
    l5100_disc = jnp.interp(5100.0, wave, L_lambda_disc) * 5100.0

    # ── Energy ledger (energy-conserving policies) ───────────────────────
    # The disc carries the intrinsic L_bol; the torus reprocesses a fraction of
    # it. Debit the observed disc by (1 - agn_torus_frac) so that
    # disc(1-f) + torus(f) conserves L_bol for every torus: reproducing the
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
    # in one self-normalized template and bypass the ledger: no debit. This
    # also covers the disc-only (``torus="none"``) case: with no reprocessor,
    # the disc keeps its full L_bol.
    #
    # Which policies debit: "conserving" always; "cigale_joint" too EXCEPT for
    # the SKIRTOR torus, which instead uses the agn_power×R template tie (Stage
    # 4 below), the CIGALE-faithful path. So cigale_joint is energy-conserving
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
    # energy, shaped as the intrinsic disc: additive with the torus debit
    # (disc -> 1 - f_torus - f_lines), matching Synthesizer's covering-fraction
    # dimming. Scoped to "conserving": cigale_joint follows CIGALE (nebular added
    # separately, allocation-conserving) and independent keeps each component on
    # its own luminosity scale. Excludes only the self-normalized bundled
    # templates (grahsp/qsogen carry disc+torus+lines in one template); NOT
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
    #     disc/torus/polar share one reference: *allocation*-conserving (the
    #     components can't drift apart), CIGALE-faithful, inclination-correct via
    #     the η(i) baked into R. This is NOT *ledger* conservation: ∫total scales
    #     with ``agn_power = agn_torus_frac·L_bol``, so agn_torus_frac→0 drives
    #     the whole AGN to zero: outside CIGALE's reachable domain, but a free
    #     agn_torus_frac sampler can reach that degenerate zero-AGN plateau.
    #   * fracAGN = 0 (default): no CIGALE coupling, so debit the disc by
    #     (1 − agn_torus_frac) exactly like the ``conserving`` policy: *ledger*
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
    # geometric visibility mask, the same one the monolithic ``unified_nlr_blr``
    # uses: so a composable disc+torus+NLR+BLR reproduces its Type-1/2 geometry.
    # Defaults (i=30, theta_torus=30 -> inc_crit=60 > i) give mask ~ 1, so
    # default-inclination models are unchanged. Static dispatch on the torus name
    # is JIT-safe.
    # ``_central_mask`` is factored out of the (disc + aniso-lines) sum
    # instead of multiplying ``L_lambda_central`` in place, so the
    # per-sub-block decomposition below (``sed_agn_disc`` / ``sed_agn_torus``
    # / ``sed_agn_lines`` / ``sed_agn_polar``, NAMING_CONTRACT §4b.5) can
    # apply the IDENTICAL mask to the disc and aniso-lines terms
    # individually: multiplication distributes over the sum, so the two
    # formulations agree to floating-point reassociation.
    _central_mask = 1.0
    if agn_torus_block in TORUS_SCREEN_PARAMS:
        _oa_key, _tau_key = TORUS_SCREEN_PARAMS[agn_torus_block]
        _central_mask = torus_screen_transmission(
            wave,
            cos_inc=params.get("agn_cos_inc", 0.86602540378443864),
            oa_deg=params.get(_oa_key, 40.0),
            tau_v=params.get(_tau_key, 7.0),
        )
    elif agn_torus_block not in _SELF_CONTAINED_TORI:
        _central_mask = sigmoid_visibility_mask(
            params.get("agn_cos_inc", 0.86602540378443864),
            params.get("agn_theta_torus", 30.0),
        )
    L_lambda_central = (L_lambda_disc + L_lambda_lines_aniso + L_lambda_feii) * _central_mask
    # Isotropic NLR: visible at every inclination, so added after the mask.
    L_lambda_central = L_lambda_central + L_lambda_lines_iso

    # Stage 5: attenuation factor (multiplicative; host/foreground screen).
    atten_fn = resolve_agn_block("attenuation", agn_attenuation_block)
    factor = atten_fn(wave, **params)

    # Convert to L_nu [erg/s/Hz] using L_nu = L_lambda * lambda^2 / c.
    _conv = wave**2 / C_AA_PER_S

    # Stage 6 (conditional): polar-dust reemission (CIGALE skirtor2016 polar
    # dust convention; Yang et al. 2020, MNRAS, 491, 740, section 2.2.2).
    # Static dispatch on agn_attenuation_block (a Python string) is JIT-safe.
    if agn_attenuation_block == "polar_dust":
        # R61: the polar screen reddens the DISC, never the torus IR. The polar
        # dust sits in the cone between the observer and the central engine;
        # the torus IR neither passes through it nor feeds its re-emission
        # budget, which is disc light. CIGALE reddens only ``disk``
        # (skirtor2016.py: ``self.SKIRTOR2016.disk *= ext_fac``) and integrates
        # ``AGN1.disk (1 - ext_fac)`` alone for the absorbed power. Feeding the
        # torus in as well was an inconsistency with no physics behind it;
        # measured, it inflated the absorbed integral by 1.7% (SMC extinction
        # is nearly transparent in the IR, which is why it stayed small and
        # unnoticed) while making the torus carry a screen it should not see.
        #
        # R63: the integrand is the UNMASKED disc. The cone dust re-emits what
        # it absorbed isotropically -- the absorbed power does not depend on
        # where the observer stands -- so ``sed_agn_polar`` is present at full
        # strength at Type-2 sightlines and the Stage-4.5 ``_central_mask``
        # must not reach this term. Only the LOS reddening is Type-1 only, and
        # ``polar_dust_extinction`` already gates that half itself. CIGALE
        # agrees (verified against a live skirtor2016 run: its polar blackbody
        # is added unconditionally, only ``disk *= ext_fac`` is gated on
        # ``i <= 90 - oa``); see ``polar_dust_reemission_lnu``'s docstring for
        # the measured per-inclination numbers. Re-applying the mask here
        # divides the Type-2 (i=80) ``sed_agn_polar`` by 98.7x under
        # ``independent`` and 67.7x under ``conserving`` at the fiducial with
        # ``agn_polar_ebv=0.3``.
        #
        # R60: which cone factor applies depends on what the disc the polar
        # dust reprocesses represents -- the two frames are exactly 18/7
        # apart, so this is stated, never inherited. And it is decided by the
        # SAME predicate that decided the disc's frame at Stage 4: the
        # *traced* ``agn_ir_frac > 0``.
        #   * fracAGN > 0: Stage 4's R-tie normalized the disc to CIGALE's
        #     inclination-specific ``disk`` (``agn_power x R``), so the
        #     reference is CIGALE's face-on ``disk`` and the factor is ``g``.
        #     ``g`` is referenced to ``int L(theta=0) dlambda`` -- NOT the
        #     observer-inclination, R-tied ``L_lambda_disc`` Stage 4 just
        #     built (which carries both the ``disk(i)/disk(0)`` reweighting
        #     and the anisotropy-scaled ``_disc_R``), so the face-on array is
        #     rebuilt from the PRE-Stage-4 disc shape (``_disc_intrinsic``)
        #     renormalized and rescaled by ``agn_power * _disc_R_faceon``
        #     (``skirtor_disc_dust_ratio``'s ``int_disk0 / int_dust``,
        #     "the ratio the polar l_ext proxy needs"), mirroring exactly how
        #     Stage 4 built ``_disc_scaled`` via ``agn_power * _disc_R``.
        #   * fracAGN = 0 (the registry default): there is no R-tie, the disc
        #     in the SED is the bolometric-frame disc debited by
        #     ``(1 - agn_torus_frac)``, and the factor is ``f_cone`` on that
        #     very array.
        # A STATIC branch here (on ``agn_norm``/``agn_torus_block`` alone)
        # applied ``g`` to a rebuilt face-on array in BOTH regimes, so at the
        # default ``agn_ir_frac = 0`` ``sed_agn_polar`` came out 1.58x high
        # and bit-identical across a 2.84x change in the disc it reprocesses.
        # Both branches are cheap (one extinction curve, one graybody), so
        # they are both evaluated and selected with ``jnp.where`` on the
        # tracer -- a Python ``if`` on ``_agn_fracAGN`` would not trace.
        _polar_params = {k: v for k, v in params.items() if k != "agn_polar_reference"}
        L_nu_reemit = polar_dust_reemission_lnu(
            wave,
            L_lambda_disc,
            agn_polar_reference="bolometric",
            **_polar_params,
        )
        if _disc_R_faceon is not None:
            # ``R_faceon = int_disk0/int_dust`` was derived on the SKIRTOR
            # templates' NATIVE grid, with ``skirtor_disc_dust_ratio``'s own
            # note that resampling those templates onto a caller grid moves
            # the integrals ~10%. So the shape it multiplies must be
            # unit-normalized on that SAME grid, and the absorbed-power
            # integral taken there too -- which is also what CIGALE does
            # (``l_ext = g(oa) * trapezoid(AGN1.disk * (1 - ext_fac),
            # x=AGN1.wl)``, the template grid). ``skirtor_disc_dust_ratio``
            # hands both out (``faceon_shape_native``, ``wave_native``), so
            # the pairing cannot drift apart. Unit-normalizing the same shape
            # on ``wave`` instead made the polar share depend on the caller's
            # wavelength extent -- 11.0% for the ``skirtor`` disc block
            # between an 8-1e8 A and a 500-1e8 A grid, neither of which
            # truncates the SKIRTOR templates at all.
            _polar_disc_face_on = _disc_shape_faceon * (_agn_power * _disc_R_faceon)
            L_nu_reemit = jnp.where(
                _agn_fracAGN > 0.0,
                polar_dust_reemission_lnu(
                    wave,
                    _polar_disc_face_on,
                    l_in_wavelength=_disc_wave_native,
                    agn_polar_reference="face_on",
                    **_polar_params,
                ),
                L_nu_reemit,
            )

        # R59: under the joint and conserving policies the AGN dust budget
        # INCLUDES the polar re-emission -- torus + polar = the budget -- so
        # the AGN dust total is invariant in E(B-V), exactly as CIGALE's is
        # (skirtor2016.py adds the polar blackbody to ``dust`` BEFORE
        # ``norm = 1/int dust``, then splits: ``lumin_dust = agn_power``,
        # ``lumin_torus = agn_power - lumin_polar_dust``). The torus block has
        # already normalized itself to the budget, so the share comes out of
        # it rather than being added on top. ``share`` is < 1 by construction,
        # so no E(B-V) can drive the torus negative.
        #
        # Under 'independent' each component stays on its own luminosity scale
        # -- that is the policy's contract -- so the re-emission is additive
        # and the total grows by the absorbed polar power.
        # Both budgets in the SAME measure. The torus is an L_lambda density
        # and the graybody an L_nu one, and on a finite grid the two
        # quadratures of one spectrum differ by ~1e-6 relative -- enough to
        # leave the "invariant total" drifting at that level. Converting the
        # torus to L_nu first and integrating both over nu makes the split
        # exact by construction.
        _nu = C_AA_PER_S / wave
        if _agn_norm in ("cigale_joint", "conserving"):
            _agn_dust_budget = jnp.abs(jnp.trapezoid(L_lambda_torus * _conv, _nu))
            _polar_power = jnp.abs(jnp.trapezoid(L_nu_reemit, _nu))
            # Degenerate case: BOTH budgets are zero -- no torus emission and
            # no polar re-emission on this grid (agn_polar_ebv = 0 with a
            # torus block that contributes nothing here) -- so there is no
            # budget to split. Documented value: share 0, i.e. the torus
            # keeps the whole (zero) budget and the split is a no-op.
            #
            # The denominator is SELECTED before the divide rather than
            # floored. An outer jnp.where does not protect the reverse pass:
            # both branches are differentiated, and division's VJP carries
            # -num/den**2, so a floored denominator squares to exactly 0.0 in
            # float32 and feeds 0 * inf = NaN back into the surviving branch.
            #
            # A NaN budget (e.g. agn_polar_ebv itself NaN) also lands in the
            # not-live branch, because NaN > 0.0 is False -- so without a
            # further check the documented "share 0" answer would silently
            # narrow the NaN budget to a finite share, and the torus
            # sub-component would come back clean while the SED and the
            # polar component still carry the NaN. The extra jnp.isnan
            # selection keeps that view honest: it is a boolean select on a
            # non-differentiable predicate (isnan, like the `> 0.0` above,
            # carries no gradient), so it changes only the forward value at
            # a NaN input and leaves the true (finite) zero/zero degenerate
            # case, and its selected-denominator gradient, untouched.
            _dust_total = _agn_dust_budget + _polar_power
            _dust_total_live = _dust_total > 0.0
            _dust_total_is_nan = jnp.isnan(_dust_total)
            _share = jnp.where(
                _dust_total_live,
                _polar_power / jnp.where(_dust_total_live, _dust_total, 1.0),
                jnp.where(_dust_total_is_nan, jnp.nan, 0.0),
            )
            _torus_factor = 1.0 - _share
            # Renormalize the graybody from its own absorbed power to the
            # budget share, so torus + polar integrates to _agn_dust_budget.
            # Degenerate case: no polar power at all -- then ``L_nu_reemit``
            # is identically zero and ``_share`` is zero with it, so the
            # rescale is 0/0. Documented value: factor 1.0, which leaves the
            # zero re-emission exactly zero.
            _polar_live = _polar_power > 0.0
            L_nu_reemit = L_nu_reemit * jnp.where(
                _polar_live,
                _agn_dust_budget * _share / jnp.where(_polar_live, _polar_power, 1.0),
                1.0,
            )
        else:
            _torus_factor = 1.0
        L_lambda_total = L_lambda_central * factor + L_lambda_torus * _torus_factor
    else:
        L_nu_reemit = jnp.zeros_like(wave)
        _torus_factor = factor
        L_lambda_total = (L_lambda_central + L_lambda_torus) * factor

    L_nu_atten = L_lambda_total * _conv
    L_nu_result = L_nu_atten + L_nu_reemit

    # Public per-sub-block rest-frame SEDs (NAMING_CONTRACT §4b.5):
    # ``sed_agn_disc``, ``sed_agn_torus``, ``sed_agn_polar``, ``sed_agn_lines``
    # (nlr + blr + feii). See :func:`_agn_sed_components`.
    components = (
        _agn_sed_components(
            L_lambda_disc=L_lambda_disc,
            L_lambda_torus=L_lambda_torus,
            L_lambda_lines_aniso=L_lambda_lines_aniso,
            L_lambda_feii=L_lambda_feii,
            L_lambda_lines_iso=L_lambda_lines_iso,
            central_mask=_central_mask,
            atten_factor=factor,
            torus_factor=_torus_factor,
            l_nu_conv=_conv,
            L_nu_polar=L_nu_reemit,
        )
        if return_components
        else None
    )

    # Return with optional L_2500_intrinsic/L_4400_intrinsic and per-sub-block
    # components tuples.
    if return_l2500 and return_components:
        return (L_nu_result, L_2500_intrinsic, L_4400_intrinsic, components)
    elif return_l2500:
        return (L_nu_result, L_2500_intrinsic, L_4400_intrinsic)
    elif return_components:
        return (L_nu_result, components)
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
    return_components: bool = False,
    **params,
) -> Array | tuple[Array, float]:
    r"""AGN_MODELS["composable"] entry point: :data:`L_ν` in erg/s/Hz.

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
    return_components : bool, optional
        When True, additionally return a ``components`` dict (see
        :func:`compose_l_nu`) as the last element of the returned tuple,
        with every entry scaled by ``agn_lum_ratio`` (matching ``L_nu``).
        Default: False.
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
    components : dict, optional
        When ``return_components=True``, ``{"disc", "torus", "lines",
        "polar"}`` -> ndarray, each scaled by ``agn_lum_ratio``. Returned as
        the last element of the tuple.

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
        return_components=return_components,
        **params,
    )
    if return_l2500 and return_components:
        L_nu, L_2500_intrinsic, L_4400_intrinsic, components = result
        components = {k: agn_lum_ratio * v for k, v in components.items()}
        return (agn_lum_ratio * L_nu, L_2500_intrinsic, L_4400_intrinsic, components)
    elif return_l2500:
        L_nu, L_2500_intrinsic, L_4400_intrinsic = result
        return (agn_lum_ratio * L_nu, L_2500_intrinsic, L_4400_intrinsic)
    elif return_components:
        L_nu, components = result
        components = {k: agn_lum_ratio * v for k, v in components.items()}
        return (agn_lum_ratio * L_nu, components)
    else:
        return agn_lum_ratio * result
