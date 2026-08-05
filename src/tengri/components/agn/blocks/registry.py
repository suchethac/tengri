# SPDX-License-Identifier: BSD-3-Clause
"""Register the composable AGN runner in :data:`AGN_MODELS`.

Importing this module side-effects ``AGN_MODELS["composable"]``. It lives
in its own file (rather than runner.py) to avoid an import cycle: the
runner imports the block protocol, which imports nothing from
:mod:`tengri.components.agn.unified`, while the registration must depend
on :func:`register_agn_model` from there.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL
from tengri.components.agn.blocks.runner import composable_agn_l_nu
from tengri.components.agn.unified import register_agn_model

__all__ = ["composable"]


@register_agn_model(
    "composable",
    citation="(no single paper — block recipe of registered tengri AGN blocks)",
    short_doc="Composable AGN: pick one block per stage (disc/nlr/blr/feii/torus/atten)",
)
def composable(
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_lum_ratio: float = 1.0,
    agn_disc_block: str = "none",
    agn_nlr_block: str = "none",
    agn_blr_block: str = "none",
    agn_feii_block: str = "none",
    agn_torus_block: str = "none",
    agn_attenuation_block: str = "none",
    agn_norm: str = "cigale_joint",
    **params,
) -> jnp.ndarray:
    r"""Composable AGN — registered AGN_MODELS entry.

    Thin wrapper around :func:`composable_agn_l_nu`; see that function
    and :mod:`tengri.components.agn.blocks._protocol` for the full
    contract.

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
        Per-stage block selectors (default ``"none"`` everywhere — the
        user **must** opt in by name, and a warning is emitted if the
        recipe is degenerate).
    agn_norm : {"cigale_joint", "conserving", "independent"}, optional
        Cross-block normalization policy (single-sourced from
        :data:`AGN_NORM_POLICIES`). ``"cigale_joint"`` (current default)
        ties the disc, torus, and polar dust to a *single* ``agn_power``
        reference via the fixed SKIRTOR template ratios — bit-identical to
        X-CIGALE's energy balance (Stalevski+2016). ``"conserving"`` debits
        the disc by the reprocessed fraction so ``disc(1-f) + torus(f)``
        conserves ``L_bol`` for *every* torus (reproduces the monolithic
        models). ``"independent"`` keeps the legacy two-reference scaling
        (disc on ``agn_log_lbol``, dust on the absorbed-energy ``agn_power``),
        which does *not* conserve energy across the disc/dust boundary.
    **params
        Per-impl free parameters forwarded to every block.

    Returns
    -------
    L_nu : ndarray, shape (n_wave,)
        Total AGN :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    JIT-compatible. Selectors are static; param values are dynamic.
    """
    return composable_agn_l_nu(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_lum_ratio=agn_lum_ratio,
        agn_disc_block=agn_disc_block,
        agn_nlr_block=agn_nlr_block,
        agn_blr_block=agn_blr_block,
        agn_feii_block=agn_feii_block,
        agn_torus_block=agn_torus_block,
        agn_attenuation_block=agn_attenuation_block,
        agn_norm=agn_norm,
        **params,
    )
