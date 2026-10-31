# SPDX-License-Identifier: BSD-3-Clause
"""Recipe-level bolometric normalization of the composable GRAHSP BBB (owner ruling 16).

Upstream GRAHSP sets one ``lum5100`` for the whole accretion-powered side: every non-torus
``activate*`` component, integrated above 91.1753 nm (``activatebol``). The composable
blocks used to normalize the disc alone, so a recipe with a GRAHSP disc and GRAHSP lines
differed from the monolithic model by the line and FeII share (1.30x for SBPL, 1.26x for
Netzer in l5100).

:func:`grahsp_recipe_log_l5100` evaluates the unit-l5100 BBB side of the selected GRAHSP
blocks (disc, narrow lines, broad lines + Balmer continuum, FeII) on the shared
publication integrals, and returns the explicit ``agn_grahsp_log_l5100`` that makes their
sum equal ``10**agn_log_lbol * L_sun``. The runner hands it to every block as the disc's
explicit normalization; the line blocks read ``l5100`` from that disc, so one scale covers
all of them.

The closed forms registered below (``LINE_ENERGY_BLOCKS``, ``DISC_POWER_BLOCKS``) give the
same pieces in the conserving ledger.
"""

from __future__ import annotations

import inspect
import math

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_disc_power, register_line_energy
from tengri.components.agn.blocks.grahsp_blocks import (
    grahsp_blr_block,
    grahsp_feii_block,
    grahsp_netzer_disc_block,
    grahsp_nlr_block,
    grahsp_sbpl_disc_block,
    grahsp_veroncetty_feii_block,
    netzer_unit_fn,
    sbpl_unit_fn,
)
from tengri.components.agn.grahsp.lines import AGN_TYPE_BL
from tengri.components.agn.grahsp.normalization import (
    balmer_continuum_unit_power,
    broad_line_unit_power,
    feii_forest_unit_power,
    narrow_line_unit_power,
)
from tengri.components.agn.grahsp.publication import bolometric_unit_nm
from tengri.components.agn.grahsp.templates import load_grahsp_templates
from tengri.utils.physics_constants import L_SUN as LSUN_ERG
from tengri.utils.scale import representable_floor

__all__ = [
    "GRAHSP_DISC_BLOCKS",
    "GRAHSP_FEII_BLOCKS",
    "grahsp_recipe_log_l5100",
]

GRAHSP_DISC_BLOCKS: frozenset[str] = frozenset({"grahsp_sbpl", "grahsp_netzer"})
"""Disc blocks whose normalization the recipe sets."""

GRAHSP_FEII_BLOCKS: frozenset[str] = frozenset({"grahsp", "grahsp_veroncetty"})
"""FeII blocks that belong to the GRAHSP BBB side."""

_LOG10_LSUN_ERG: float = math.log10(LSUN_ERG)


def _kw(fn, name: str, params):
    """Value of block keyword ``name``: from ``params``, else the block's own default."""
    if name in params:
        return params[name]
    return inspect.signature(fn).parameters[name].default


def _templates_or_load(templates):
    """The pre-loaded GRAHSP bundle, else the cached load."""
    return load_grahsp_templates() if templates is None else templates


def _disc_unit(disc_block: str, params, templates) -> Array:
    """Unit-l5100 disc power above the lumBolBBB cut [erg/s], as the disc block integrates it."""
    if disc_block == "grahsp_netzer":
        unit_fn = netzer_unit_fn(
            templates,
            log_mbh=_kw(grahsp_netzer_disc_block, "agn_grahsp_netzer_log_mbh", params),
            spin=_kw(grahsp_netzer_disc_block, "agn_grahsp_netzer_spin", params),
            log_mdot=_kw(grahsp_netzer_disc_block, "agn_grahsp_netzer_log_mdot", params),
        )
        return bolometric_unit_nm(unit_fn, (templates.disc_wave_nm,))
    fn = grahsp_sbpl_disc_block
    unit_fn = sbpl_unit_fn(
        uvslope=_kw(fn, "agn_grahsp_uvslope", params),
        plslope=_kw(fn, "agn_grahsp_plslope", params),
        plbendloc_nm=_kw(fn, "agn_grahsp_plbendloc_nm", params),
        plbendwidth=_kw(fn, "agn_grahsp_plbendwidth", params),
        cutoff_nm=_kw(fn, "agn_grahsp_cutoff_nm", params),
    )
    return bolometric_unit_nm(unit_fn)


def _nlr_unit(params, templates) -> Array:
    """Unit-l5100 narrow-line power above the lumBolBBB cut [erg/s]."""
    fn = grahsp_nlr_block
    return narrow_line_unit_power(
        templates,
        agn_type=_kw(fn, "agn_type", params),
        a_lines=_kw(fn, "agn_grahsp_a_lines", params),
        linewidth_kms=_kw(fn, "agn_grahsp_linewidth_kms", params),
    )


def _blr_unit(params, templates) -> Array:
    """Unit-l5100 broad-line plus Balmer-continuum power above the cut [erg/s]."""
    fn = grahsp_blr_block
    if _kw(fn, "agn_type", params) != AGN_TYPE_BL:
        return jnp.asarray(0.0)
    linewidth = _kw(fn, "agn_grahsp_linewidth_kms", params)
    return broad_line_unit_power(
        templates, a_lines=_kw(fn, "agn_grahsp_a_lines", params), linewidth_kms=linewidth
    ) + balmer_continuum_unit_power(
        a_bc=_kw(fn, "agn_grahsp_a_bc", params), linewidth_kms=linewidth
    )


def _feii_unit(feii_block: str, params, templates) -> Array:
    """Unit-l5100 FeII forest power above the lumBolBBB cut [erg/s]."""
    if feii_block == "grahsp_veroncetty":
        fn = grahsp_veroncetty_feii_block
        twave, tlumin = templates.feii_vc04_wave_nm, templates.feii_vc04_lumin
        field = "feii_vc04_wave_nm"
    else:
        fn = grahsp_feii_block
        twave, tlumin = templates.feii_wave_nm, templates.feii_lumin
        field = "feii_wave_nm"
    if _kw(fn, "agn_type", params) != AGN_TYPE_BL:
        return jnp.asarray(0.0)
    return feii_forest_unit_power(
        twave,
        tlumin,
        a_feii=_kw(fn, "agn_grahsp_a_feii", params),
        a_lines=_kw(fn, "agn_grahsp_a_lines", params),
        wave_field=field,
    )


def _lines_unit(nlr_block: str, blr_block: str, feii_block: str, params, templates) -> Array:
    """Summed unit-l5100 power of the GRAHSP line and FeII blocks that are selected."""
    total = jnp.asarray(0.0)
    if nlr_block == "grahsp":
        total = total + _nlr_unit(params, templates)
    if blr_block == "grahsp":
        total = total + _blr_unit(params, templates)
    if feii_block in GRAHSP_FEII_BLOCKS:
        total = total + _feii_unit(feii_block, params, templates)
    return total


def grahsp_recipe_log_l5100(
    agn_log_lbol,
    params,
    *,
    disc_block: str,
    nlr_block: str,
    blr_block: str,
    feii_block: str,
    templates=None,
):
    """Explicit ``log10(lambda L_lambda(5100 A) / (erg/s))`` of a GRAHSP BBB recipe.

    Sets the BBB-side bolometric, summed over the selected GRAHSP blocks above the
    lumBolBBB cut, to ``10**agn_log_lbol * L_sun``. ``None`` when the disc is not a GRAHSP
    disc (the disc then keeps its own normalization).

    Parameters
    ----------
    agn_log_lbol : float
        True :math:`\\log_{10}(L_\\mathrm{bol}/L_\\odot)` (not the float32 reference value).
    params : mapping
        Block keyword arguments (``agn_grahsp_*`` keys, ``agn_type``).
    disc_block, nlr_block, blr_block, feii_block : str
        Selected block names of each stage.
    templates : GRAHSPTemplates, optional
        Pre-loaded bundle; the cached load when ``None``.

    Returns
    -------
    ndarray, shape () or None
        log10 of the unit-free l5100 in erg/s, or ``None``.

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes, on the unit integrals.
    """
    if disc_block not in GRAHSP_DISC_BLOCKS:
        return None
    if templates is None:
        templates = load_grahsp_templates()
    total = _disc_unit(disc_block, params, templates) + _lines_unit(
        nlr_block, blr_block, feii_block, params, templates
    )
    return (
        jnp.asarray(agn_log_lbol)
        + _LOG10_LSUN_ERG
        - jnp.log10(jnp.maximum(total, representable_floor(1e-300)))
    )


# ──────────────────────────────────────────────────────────────────────
# Closed forms for the conserving ledger (LINE_ENERGY_BLOCKS, DISC_POWER_BLOCKS)
# ──────────────────────────────────────────────────────────────────────


@register_line_energy("nlr", "grahsp")
def _grahsp_nlr_energy(agn_log_lbol, l5100_disc, *, templates=None, **params):
    """Narrow-line power [erg/s]: linear in the disc's l5100."""
    return l5100_disc * _nlr_unit(params, _templates_or_load(templates))


@register_line_energy("blr", "grahsp")
def _grahsp_blr_energy(agn_log_lbol, l5100_disc, *, templates=None, **params):
    """Broad-line plus Balmer-continuum power [erg/s]: linear in the disc's l5100."""
    return l5100_disc * _blr_unit(params, _templates_or_load(templates))


@register_line_energy("feii", "grahsp")
def _grahsp_feii_energy(agn_log_lbol, l5100_disc, *, templates=None, **params):
    """FeII forest power [erg/s] (Bruhweiler & Verner): linear in the disc's l5100."""
    return l5100_disc * _feii_unit("grahsp", params, _templates_or_load(templates))


@register_line_energy("feii", "grahsp_veroncetty")
def _grahsp_veroncetty_energy(agn_log_lbol, l5100_disc, *, templates=None, **params):
    """FeII forest power [erg/s] (Veron-Cetty & Joly): linear in the disc's l5100."""
    return l5100_disc * _feii_unit("grahsp_veroncetty", params, _templates_or_load(templates))


def _disc_power_fraction(disc_block: str):
    """Closed-form disc share of L_acc for one GRAHSP disc block (the full GRAHSP recipe).

    The share is the disc's power over the BBB-side total, both above the lumBolBBB cut.
    It assumes the narrow lines, broad lines and FeII are GRAHSP too, as the registry
    cannot see the other stages' names.
    """

    def fraction(agn_log_lbol, *, templates=None, **params):
        templates = _templates_or_load(templates)
        disc = _disc_unit(disc_block, params, templates)
        total = disc + _lines_unit("grahsp", "grahsp", "grahsp", params, templates)
        shape = params.get("agn_log_lbol_shape", agn_log_lbol)
        explicit = params.get("agn_grahsp_log_l5100")
        log_l5100 = (
            shape + _LOG10_LSUN_ERG - jnp.log10(jnp.maximum(total, representable_floor(1e-300)))
            if explicit is None
            else explicit
        )
        # disc power = disc * l5100, divided by L_acc = 10**shape * L_sun (the shape frame).
        return disc * jnp.power(10.0, log_l5100 - shape) / LSUN_ERG

    return fraction


for _disc in sorted(GRAHSP_DISC_BLOCKS):
    register_disc_power(_disc)(_disc_power_fraction(_disc))
