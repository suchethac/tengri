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
from tengri.components.agn.blocks.atten_blocks import polar_dust_reemission_lnu
from tengri.components.agn.blocks.masking import (
    sigmoid_visibility_mask,
    split_lines_result,
)
from tengri.components.agn.blocks.torus_screen import (
    TORUS_SCREEN_PARAMS,
    torus_screen_transmission,
)
from tengri.components.agn.polar_dust import _RV_SMC, smc_extinction_curve
from tengri.components.agn.skirtor import skirtor_disc_dust_ratio
from tengri.utils.physics_constants import L_SUN

#: Torus selectors that do NOT receive the grey Type-1/2 visibility mask:
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
# (i.e. compatible with GRAHSP-style downstream blocks that normalise to
# λL_λ(5100Å)). Block impls outside this set may emit zero or NaN at
# 5100Å, breaking the downstream normalisation silently.
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
    "feii": frozenset({"grahsp"}),
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
        "richards2006",
    }
)

#: Speed of light in Å × Hz, used for L_λ → L_ν conversion.
C_AA_PER_S: float = 2.99792458e18

#: Selector keys recognised by the runner. Match the canonical pipeline order.
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
            f"normalise to lambda*L_lambda(5100A) of the disc, which "
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
                f"normalises to the disc's lambda*L_lambda(5100A), but "
                f"disc_block={selectors['disc']!r} is not in the set of "
                f"impls known to produce a meaningful UV/optical continuum "
                f"at 5100A: {sorted(_DISCS_WITH_5100A_CONTINUUM)}. "
                f"Verify your disc impl emits sensible flux at 5100A."
            )

    # Rule 7: polar dust selected but E(B-V) defaults to 0 (no-op).
    if (
        selectors["attenuation"] == "polar_dust"
        and params is not None
        and float(params.get("agn_polar_ebv", 0.0)) == 0.0
    ):
        _emit(
            "Composable AGN: agn_attenuation_block='polar_dust' but "
            "agn_polar_ebv=0 (no extinction applied). Either set "
            "agn_polar_ebv > 0 or pick agn_attenuation_block='none'."
        )

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
    **params,
) -> Array:
    r"""Compose AGN-side :math:`L_\nu` from per-stage block implementations.

    Pipeline (paper §2.1.6 / upstream module order)::

        L_λ_total = L_disc + L_nlr + L_blr + L_feii + L_torus
        L_λ_atten = L_λ_total × attenuation_factor
        L_ν       = L_λ_atten × λ²/c

    The disc stage runs first so its 5100Å luminosity can scale the NLR /
    BLR / FeII / torus normalisations (matching upstream GRAHSP convention).

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
    **params
        Per-impl free parameters. Each block consumes the keys it
        recognises and ignores the rest.

    Returns
    -------
    L_nu : ndarray, shape (n_wave,)
        Total AGN-side :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    JIT-compatible (selectors are static). The order of operations matches
    the upstream GRAHSP ``ActivateGTorus``/``ActivateNarrowLines``/
    ``ActivateBroadLines``/``ActivatePL``/``BiAttenuationLaw`` chain so
    an all-grahsp selection is numerically equivalent to
    :func:`tengri.components.agn.grahsp.compute_grahsp_sed`.
    """
    wave = jnp.asarray(wavelength)
    # If templates are pre-loaded, forward them under a stable kwarg name
    # blocks recognise (``templates``). When None, blocks fall back to their
    # own lru_cache load. We strip the kwarg afterwards so blocks that don't
    # take it never see it.
    grahsp_templates = template_state.get("grahsp") if template_state is not None else None

    # Stage 1: disc continuum (L_lambda [erg/s/Å]).
    disc_fn = resolve_agn_block("disc", agn_disc_block)
    L_lambda_disc = disc_fn(
        wave,
        agn_log_lbol=agn_log_lbol,
        templates=grahsp_templates,
        **params,
    )

    # Disc extinction (CIGALE skirtor2016.py:341-348). For Type-1 viewing
    # (i <= 90 - oa, i.e. cos_inc >= sin(oa)) the line-of-sight disc is
    # reddened: ``disk *= ext_fac``. The energy the reddening removes is
    # routed to the polar greybody (the SKIRTOR torus block normalises
    # disc+torus+polar jointly to agn_power — see below). Gated on
    # ``agn_polar_ebv > 0`` → models without polar dust are untouched.
    _polar_ebv = jnp.asarray(params.get("agn_polar_ebv", 0.0))
    _cos_inc = jnp.asarray(params.get("agn_cos_inc", 0.86602540378443864))
    _oa_deg = jnp.asarray(params.get("agn_oa_skirtor", params.get("agn_polar_oa", 40.0)))
    _is_type1 = _cos_inc >= jnp.sin(jnp.deg2rad(_oa_deg))
    _disc_ext = jnp.exp(-0.921 * _polar_ebv * _RV_SMC * smc_extinction_curve(wave))
    _disc_ext = jnp.where(_is_type1 & (_polar_ebv > 0.0), _disc_ext, 1.0)

    # CIGALE single-reference disc normalisation (#556). In fracAGN-coupled
    # mode with the SKIRTOR torus, CIGALE ties the disc to the SAME
    # ``agn_power`` as the dust via the fixed template ratio
    # ``R = lumin_disk/lumin_dust`` (skirtor2016.py ``norm = 1/∫dust``), so
    # disc, torus and polar all scale together and the disc-reddening
    # absorbed power is conserved into the polar greybody. ``R`` carries the
    # anisotropy factor ``η(i) = cos(i)(1+2cos(i))/3``. We capture R from the
    # *un-reddened* disc shape here (CIGALE normalises before reddening) and
    # apply the disc renormalisation after the torus block fixes agn_power.
    # Static dispatch on the torus name (JIT-safe); the fracAGN>0 gate is
    # applied branchlessly after the torus block (below).
    _agn_fracAGN = jnp.asarray(params.get("agn_fracAGN", 0.0))
    _disc_R = None
    _disc_incl = None
    # ``agn_norm`` policy (#556): "cigale_joint" ties disc/torus/polar to the
    # single agn_power reference (only meaningful for the SKIRTOR torus, whose
    # template ratios define R); "independent" keeps the legacy per-component
    # scaling. Static string (JIT-safe Python branch).
    _agn_norm = params.get("agn_norm", "cigale_joint")
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
        # the L_absorbed-coupled value the torus block normalises to
        # (``agn_torus_frac × L_bol``), available *before* the torus runs so
        # there is no circular dependency on ∫torus. Branchless: fall back to
        # the legacy proxy where ``agn_fracAGN == 0``.
        _agn_torus_frac = jnp.asarray(params.get("agn_torus_frac", 0.5))
        _Lbol = 10.0**agn_log_lbol * L_SUN
        _agn_power_pre = _agn_torus_frac * _Lbol
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

    # Compute lambda*L_lambda(5100Å) for downstream block normalisations.
    # L_lambda is on the user's wave grid; jnp.interp pulls the value at 5100Å.
    l5100_disc = jnp.interp(5100.0, wave, L_lambda_disc) * 5100.0

    # Stage 2a: narrow-line region.
    nlr_fn = resolve_agn_block("nlr", agn_nlr_block)
    nlr_aniso, nlr_iso = split_lines_result(
        nlr_fn(
            wave,
            agn_log_lbol=agn_log_lbol,
            l5100_disc=l5100_disc,
            templates=grahsp_templates,
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
            templates=grahsp_templates,
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
        templates=grahsp_templates,
        **params,
    )

    # Stage 4: IR torus.
    torus_fn = resolve_agn_block("torus", agn_torus_block)
    L_lambda_torus = torus_fn(
        wave,
        agn_log_lbol=agn_log_lbol,
        l5100_disc=l5100_disc,
        templates=grahsp_templates,
        **params,
    )

    # CIGALE single-reference disc normalisation (#556), part 2. The SKIRTOR
    # torus block fixes ``agn_power = ∫L_lambda_torus`` (disc+torus+polar
    # share this budget). Tie the disc bolometric to ``agn_power × R`` so
    # the disc scales with the same reference as the dust — energy conserved,
    # inclination-correct via the ``η(i)`` baked into R. Applied branchlessly:
    # only when ``agn_fracAGN > 0`` (the CIGALE-coupled mode); otherwise the
    # disc keeps its independent ``agn_log_lbol`` scaling.
    if _disc_R is not None:
        _agn_power = jnp.trapezoid(L_lambda_torus, wave)
        # Apply the wavelength-dependent ``disk(i)/disk(0)`` inclination
        # attenuation to the disc *shape* (CIGALE ``SKIRTOR.disk(i)/AGN1.disk(0)``)
        # so the disc spectrum is inclination-correct, then renormalise the
        # reweighted shape to the agn_power-tied bolometric ``agn_power × R``.
        _disc_reweighted = L_lambda_disc * _disc_incl
        _disc_int = jnp.maximum(jnp.trapezoid(_disc_reweighted, wave), 1e-30)
        _disc_scaled = _disc_reweighted * (_agn_power * _disc_R) / _disc_int
        L_lambda_disc = jnp.where(_agn_fracAGN > 0.0, _disc_scaled, L_lambda_disc)

    # Stage 4.5: Type-1/2 obscuration of the *anisotropic* central engine (disc +
    # broad lines + FeII). The isotropic NLR is added back afterwards, so it stays
    # visible at all inclinations. Each torus carries ONE obscuration model (no
    # double-counting): dusty-screen tori (fritz/skirtor, #294) apply a
    # wavelength-dependent screen; every other non-"none" torus applies the grey
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
    # as a geometry-independent FIR greybody. Compute and add this to the SED.
    # Static dispatch on agn_attenuation_block (a Python string) is JIT-safe.
    if agn_attenuation_block == "polar_dust":
        # Compute reemission in L_nu from the pre-attenuation SED (torus-screened
        # central engine + torus IR).
        L_lambda_pre_atten = L_lambda_central + L_lambda_torus
        L_nu_reemit = polar_dust_reemission_lnu(wave, L_lambda_pre_atten, **params)
        return L_nu_atten + L_nu_reemit
    else:
        return L_nu_atten


def composable_agn_l_nu(
    wavelength: Array,
    agn_log_lbol: float = 45.0,
    agn_frac: float = 1.0,
    agn_disc_block: str = "none",
    agn_nlr_block: str = "none",
    agn_blr_block: str = "none",
    agn_feii_block: str = "none",
    agn_torus_block: str = "none",
    agn_attenuation_block: str = "none",
    template_state: dict | None = None,
    **params,
) -> Array:
    r"""AGN_MODELS["composable"] entry point — :data:`L_ν` in erg/s/Hz.

    Thin wrapper around :func:`compose_l_nu` matching the AGN_MODELS
    registry signature::

        fn(wavelength, agn_log_lbol, agn_frac, **kwargs) -> L_nu

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float, optional
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`. Default ``45.0``.
    agn_frac : float, optional
        Overall AGN fraction scaling [dimensionless]. Default ``1.0``.
    agn_disc_block, agn_nlr_block, agn_blr_block, agn_feii_block, \
agn_torus_block, agn_attenuation_block : str, optional
        Per-stage block selectors. Default ``"none"`` for every stage
        (a no-op pipeline; users **must** opt in by name).
    **params
        Per-impl free parameters forwarded to every block.

    Returns
    -------
    L_nu : ndarray, shape (n_wave,)
        Total AGN :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    JIT-compatible (block selectors are static). Validation runs once at
    Python entry, so the JIT cache picks up changes only on selector
    changes (which trigger a recompile anyway).
    """
    validate_block_recipe(
        agn_disc_block=agn_disc_block,
        agn_nlr_block=agn_nlr_block,
        agn_blr_block=agn_blr_block,
        agn_feii_block=agn_feii_block,
        agn_torus_block=agn_torus_block,
        agn_attenuation_block=agn_attenuation_block,
        params=params,
    )
    return agn_frac * compose_l_nu(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_disc_block=agn_disc_block,
        agn_nlr_block=agn_nlr_block,
        agn_blr_block=agn_blr_block,
        agn_feii_block=agn_feii_block,
        agn_torus_block=agn_torus_block,
        agn_attenuation_block=agn_attenuation_block,
        template_state=template_state,
        **params,
    )
