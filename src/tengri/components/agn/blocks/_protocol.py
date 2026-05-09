# SPDX-License-Identifier: BSD-3-Clause
"""AGN block protocol — the contract each pluggable AGN sub-component obeys.

Tengri's existing AGN registry (:data:`tengri.components.agn.AGN_MODELS`)
holds *monolithic* models — qsogen, skirtor, kubota_done, GRAHSP — each one
a complete recipe (disc + lines + torus + attenuation) bundled into a single
function. Picking ``agn_model="qsogen"`` means inheriting *all* of qsogen's
pieces; users cannot mix QSOgen's BBB with SKIRTOR's torus without writing
custom glue code.

This module introduces a finer-grained registry for pluggable **blocks** —
the natural spectral decomposition of an AGN SED::

    [disc]  →  [lines]  →  [feii]  →  [torus]  →  [attenuation]
                                                       │
                                                       ▼
                                                 final L_nu

Each *category* (`disc`, `lines`, `feii`, `torus`, `attenuation`) hosts a
named registry of implementations. A user composes an AGN by picking one
implementation per category, and :func:`composable_agn` runs them in the
canonical order above.

This coexists with the legacy :data:`AGN_MODELS`: monolithic recipes stay
addressable by their old name (``"qsogen"``, ``"grahsp"``, …) while users
who want mix-and-match can opt into ``agn_model="composable"`` plus
per-category selectors.

Block signature
---------------
Each block is a pure JAX function with one of three signatures, depending
on its category::

    # disc — produces the AGN UV/optical continuum.
    disc(wavelength, agn_log_lbol, **params) -> L_lambda  [erg/s/Å]

    # lines / feii / torus — additive contributions, normalised to the
    # disc-side λL_λ(5100Å) already computed by the disc block.
    lines(wavelength, agn_log_lbol, l5100_disc, **params) -> L_lambda
    feii (wavelength, agn_log_lbol, l5100_disc, **params) -> L_lambda
    torus(wavelength, agn_log_lbol, l5100_disc, **params) -> L_lambda

    # attenuation — multiplicative wavelength factor in [0, 1].
    attenuation(wavelength, **params) -> factor  [dimensionless]

Why ``L_λ`` rather than ``L_ν``? Because every AGN piece in tengri's
existing codebase is naturally written in :math:`L_\\lambda` (the
GRAHSP, qsogen, and Mullaney models all live there); converting once at
the runner output (``L_ν = L_λ λ²/c``) is cheaper and less error-prone
than per-block conversions.

Why ``l5100_disc`` shared state? The lines / feii / torus normalisations
in upstream GRAHSP are tied to :math:`\\lambda L_\\lambda(5100\\,\\mathrm{\\AA})`
of the disc; extracting it once and threading it through is more efficient
than each block recomputing.

Parameter contract
------------------
Each block-impl owns its own parameter prefix (e.g. ``agn_grahsp_*``,
``agn_qsogen_*``, ``agn_torus_simple_*``), exactly as today. The runner
forwards ``**params`` to every block; each block is responsible for
extracting its own keys (and ignoring the rest).

Block names should be unique across categories so they can be inferred
from the first selector hit (``agn_disc_block="grahsp_sbpl"``,
``agn_torus_block="grahsp"`` etc.).

Notes
-----
**JIT-compatible**: yes — registry lookups happen at trace-time on Python
strings (static); the dispatched callables are pure JAX.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
from jax import Array

__all__ = [
    "AGN_BLOCKS",
    "BLOCK_CATEGORIES",
    "BlockCategory",
    "register_agn_block",
    "resolve_agn_block",
]

BlockCategory = Literal["disc", "lines", "feii", "torus", "attenuation"]
BLOCK_CATEGORIES: tuple[str, ...] = ("disc", "lines", "feii", "torus", "attenuation")
"""Fixed canonical pipeline order; do not reorder without updating runner."""

# Two-level dict: category -> name -> callable.
AGN_BLOCKS: dict[str, dict[str, Callable]] = {
    cat: {} for cat in BLOCK_CATEGORIES
}


def register_agn_block(category: BlockCategory, name: str) -> Callable:
    """Decorator factory: register a block implementation in
    :data:`AGN_BLOCKS`.

    Parameters
    ----------
    category : {"disc", "lines", "feii", "torus", "attenuation"}
        Pipeline stage this block implements.
    name : str
        Unique identifier within the category. Examples: ``"grahsp"``,
        ``"powerlaw"``, ``"none"``, ``"smc_prevot"``.

    Returns
    -------
    decorator : callable
        Inner decorator that registers the function and returns it
        unchanged.

    Raises
    ------
    KeyError
        If ``category`` is not a valid :data:`BLOCK_CATEGORIES` value.
    ValueError
        If ``(category, name)`` is already registered.

    Examples
    --------
    >>> @register_agn_block("torus", "grahsp")
    ... def grahsp_torus_block(wavelength, agn_log_lbol, l5100_disc, **params):
    ...     ...
    """
    if category not in AGN_BLOCKS:
        raise KeyError(
            f"Unknown block category {category!r}; expected one of {BLOCK_CATEGORIES}."
        )

    def decorator(fn: Callable) -> Callable:
        if name in AGN_BLOCKS[category]:
            raise ValueError(
                f"AGN block {category}/{name!r} is already registered "
                f"(by {AGN_BLOCKS[category][name].__module__})."
            )
        AGN_BLOCKS[category][name] = fn
        return fn

    return decorator


def resolve_agn_block(category: BlockCategory, name: str) -> Callable:
    """Look up a registered block by (category, name).

    Parameters
    ----------
    category : str
    name : str

    Returns
    -------
    block_fn : callable

    Raises
    ------
    ValueError
        If ``name`` is not registered in ``category``.
    """
    if category not in AGN_BLOCKS:
        raise KeyError(f"Unknown category {category!r}; expected {BLOCK_CATEGORIES}.")
    if name not in AGN_BLOCKS[category]:
        available = sorted(AGN_BLOCKS[category])
        raise ValueError(
            f"Unknown {category} block {name!r}. Available: {available}."
        )
    return AGN_BLOCKS[category][name]


# ──────────────────────────────────────────────────────────────────────
# Built-in "none" blocks — return zeros / identity factor. Useful as
# defaults when the user wants to skip a stage.
# ──────────────────────────────────────────────────────────────────────


@register_agn_block("disc", "none")
def _disc_none(wavelength: Array, agn_log_lbol: float, **_params) -> Array:
    r"""Skip the disc stage: emit zero L_lambda."""
    return jnp.zeros_like(jnp.asarray(wavelength))


@register_agn_block("lines", "none")
def _lines_none(
    wavelength: Array, agn_log_lbol: float, l5100_disc: Array, **_params
) -> Array:
    r"""Skip the line stage: emit zero L_lambda."""
    return jnp.zeros_like(jnp.asarray(wavelength))


@register_agn_block("feii", "none")
def _feii_none(
    wavelength: Array, agn_log_lbol: float, l5100_disc: Array, **_params
) -> Array:
    r"""Skip the FeII stage: emit zero L_lambda."""
    return jnp.zeros_like(jnp.asarray(wavelength))


@register_agn_block("torus", "none")
def _torus_none(
    wavelength: Array, agn_log_lbol: float, l5100_disc: Array, **_params
) -> Array:
    r"""Skip the torus stage: emit zero L_lambda."""
    return jnp.zeros_like(jnp.asarray(wavelength))


@register_agn_block("attenuation", "none")
def _attenuation_none(wavelength: Array, **_params) -> Array:
    r"""Skip attenuation: return identity factor (1.0 everywhere)."""
    return jnp.ones_like(jnp.asarray(wavelength))
