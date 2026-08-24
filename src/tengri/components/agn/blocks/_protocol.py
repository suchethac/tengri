# SPDX-License-Identifier: BSD-3-Clause
"""AGN block protocol, the contract each pluggable AGN sub-component obeys.

Tengri's existing AGN registry (:data:`tengri.components.agn.AGN_MODELS`)
holds *monolithic* models: qsogen, skirtor, kubota_done, GRAHSP: each one
a complete recipe (disc + lines + torus + attenuation) bundled into a single
function. Picking ``agn_model="qsogen"`` means inheriting *all* of qsogen's
pieces; users cannot mix QSOgen's BBB with SKIRTOR's torus without writing
custom glue code.

This module introduces a finer-grained registry for pluggable **blocks** :
the natural spectral decomposition of an AGN SED::

    disc → nlr → blr → feii → torus → attenuation

Each *category* (`disc`, `nlr`, `blr`, `feii`, `torus`, `attenuation`) hosts a
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

    # disc: produces the AGN UV/optical continuum.
    disc(wavelength, agn_log_lbol, **params) -> L_lambda  [erg/s/Å]

    # nlr / blr / feii / torus: additive contributions, normalized to the
    # disc-side λL_λ(5100Å) already computed by the disc block.
    nlr (wavelength, agn_log_lbol, l5100_disc, **params) -> L_lambda or (L_maskable, L_isotropic)
    blr (wavelength, agn_log_lbol, l5100_disc, **params) -> L_lambda or (L_maskable, L_isotropic)
    feii(wavelength, agn_log_lbol, l5100_disc, **params) -> L_lambda
    torus(wavelength, agn_log_lbol, l5100_disc, **params) -> L_lambda

    # attenuation: multiplicative wavelength factor in [0, 1].
    attenuation(wavelength, **params) -> factor  [dimensionless]

Why ``L_λ`` rather than ``L_ν``? Because every AGN piece in tengri's
existing codebase is naturally written in :math:`L_\\lambda` (the
GRAHSP, qsogen, and Mullaney models all live there); converting once at
the runner output (``L_ν = L_λ λ²/c``) is cheaper and less error-prone
than per-block conversions.

Why ``l5100_disc`` shared state? The lines / feii / torus normalizations
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
**JIT-compatible**: yes, registry lookups happen at trace-time on Python
strings (static); the dispatched callables are pure JAX.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
from jax import Array

__all__ = [
    "AGN_BLOCKS",
    "AGN_BLOCK_META",
    "AGN_NORM_POLICIES",
    "BLOCK_CATEGORIES",
    "BlockCategory",
    "collect_block_templates",
    "register_agn_block",
    "resolve_agn_block",
]

BlockCategory = Literal["disc", "nlr", "blr", "feii", "torus", "attenuation"]
BLOCK_CATEGORIES: tuple[str, ...] = ("disc", "nlr", "blr", "feii", "torus", "attenuation")
"""Fixed canonical pipeline order; do not reorder without updating runner."""

# Two-level dict: category -> name -> callable.
AGN_BLOCKS: dict[str, dict[str, Callable]] = {cat: {} for cat in BLOCK_CATEGORIES}

# Parallel dict: (category, name) -> metadata dict
# Stores citation, status, short_doc alongside the callable in AGN_BLOCKS.
# Enables introspection (list_agn_blocks, describe_agn_block) while keeping
# resolve_agn_block() returning a bare callable for backward compatibility.
AGN_BLOCK_META: dict[tuple[str, str], dict[str, str]] = {}

# Parallel dict: (category, name) -> zero-arg callable returning the block's
# template library as a JAX pytree.
#
# This is the threading registry. A block backed by a template library MUST
# declare its loader here (via ``register_agn_block(template_loader=...)``),
# because that is the only thing that tells the forward model which grids to
# hoist out of the trace and hand to ``jax.jit`` as arguments. A block that
# omits it and instead calls its module-level cached loader from inside the
# trace freezes the entire library into the graph as ``Constant`` ops: 31 MB
# for SKIRTOR, 17 MB for Fritz. See ``collect_block_templates``.
AGN_BLOCK_TEMPLATE_LOADERS: dict[tuple[str, str], Callable[[], object]] = {}

# Cross-block normalization policies (``agn_norm``). Single source of truth
# shared by the runner (``compose_l_nu``) and the grammar validator
# (``parameters/groups.py``) so the two can never drift, the repo's recurring
# "wired in one layer but not the other" footgun. Each value is a one-line
# description for the ``describe``/``list`` surfaces. ``name -> description``.
AGN_NORM_POLICIES: dict[str, str] = {
    "cigale_joint": (
        "Single agn_power reference; disc/torus/polar tied by the SKIRTOR "
        "template ratio R (X-CIGALE, Yang+2020). Energy-conserving for the "
        "SKIRTOR torus. Current default."
    ),
    "conserving": (
        "Disc debited by the reprocessed fraction so disc(1-f)+torus(f) "
        "conserves L_bol for every torus. Reproduces the monolithic models."
    ),
    "independent": (
        "Each component on its own luminosity scale; no cross-block energy "
        "tie (AGNfitter-style). For cross-code comparison, not conserving."
    ),
}


def register_agn_block(
    category: BlockCategory,
    name: str,
    citation: str = "",
    status: str = "production",
    short_doc: str = "",
    template_loader: Callable[[], object] | None = None,
) -> Callable:
    """Decorator factory: register a block implementation in
    :data:`AGN_BLOCKS`.

    Parameters
    ----------
    category : {"disc", "nlr", "blr", "feii", "torus", "attenuation"}
        Pipeline stage this block implements.
    name : str
        Unique identifier within the category. Examples: ``"grahsp"``,
        ``"powerlaw"``, ``"none"``, ``"smc_prevot"``.
    citation : str, optional
        Academic citation (e.g., paper title, authors, journal reference).
        Default ``""``.
    status : str, optional
        Block maturity: ``"production"``, ``"experimental"``, ``"demo"``,
        or ``"deprecated"``. Default ``"production"``.
    short_doc : str, optional
        One-line description (e.g., "Power-law continuum with 2 free params").
        Default ``""``.
    template_loader : callable, optional
        Zero-argument callable returning this block's template library as a
        JAX pytree (e.g. :func:`~tengri.components.agn.silva04.load_silva04_default_grid`).
        Declaring it makes the forward model load the library **outside** the
        JIT trace and pass it to the block as a traced argument. Blocks with
        no template library leave this ``None`` (default). The block must
        then accept the library via its ``templates`` keyword.

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
    >>> @register_agn_block(
    ...     "torus",
    ...     "grahsp",
    ...     citation="Nenkova et al. 2008",
    ...     short_doc="Clumpy toroidal dust model",
    ... )
    ... def grahsp_torus_block(wavelength, agn_log_lbol, l5100_disc, **params): ...
    """
    if category not in AGN_BLOCKS:
        raise KeyError(f"Unknown block category {category!r}; expected one of {BLOCK_CATEGORIES}.")

    def decorator(fn: Callable) -> Callable:
        if name in AGN_BLOCKS[category]:
            raise ValueError(
                f"AGN block {category}/{name!r} is already registered "
                f"(by {AGN_BLOCKS[category][name].__module__})."
            )
        AGN_BLOCKS[category][name] = fn
        AGN_BLOCK_META[(category, name)] = {
            "citation": citation,
            "status": status,
            "short_doc": short_doc,
        }
        if template_loader is not None:
            AGN_BLOCK_TEMPLATE_LOADERS[(category, name)] = template_loader
        return fn

    return decorator


def collect_block_templates(recipe: dict[str, str]) -> dict[str, object]:
    """Load the template libraries a block recipe needs, outside any trace.

    Parameters
    ----------
    recipe : dict
        Maps block category to selected block name, e.g.
        ``{"torus": "skirtor", "disc": "multicolor"}``. Unknown categories
        and unregistered names are ignored.

    Returns
    -------
    dict
        Maps ``"<category>/<name>"`` to that block's template pytree. Blocks
        that declare no loader are absent. Empty if nothing needs threading.

    Notes
    -----
    **JIT-compatible**: no, deliberately: this performs the HDF5 I/O that
    must happen *before* tracing so the arrays can be passed in as
    arguments. Calling it inside a trace defeats its entire purpose.

    A loader that raises (missing data file, unreadable grid) is skipped
    rather than propagated: the block will then fall back to its own
    on-disk load and merely bake, which is slow but still correct. Failing
    the whole model build here would turn a performance regression into an
    outage.
    """
    templates: dict[str, object] = {}
    for category, name in recipe.items():
        loader = AGN_BLOCK_TEMPLATE_LOADERS.get((category, name))
        if loader is None:
            continue
        try:
            bundle = loader()
        except Exception:
            continue
        # A loader may legitimately return None when its library is absent in a
        # given install (e.g. the v2 SKIRTOR grid has no separate disc column,
        # and nthcomp templates are optional). Storing the None would put a key
        # in the dict that reads as "threaded" to anything doing a membership
        # test, so leave it out entirely.
        if bundle is not None:
            templates[f"{category}/{name}"] = bundle
    return templates


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
        raise ValueError(f"Unknown {category} block {name!r}. Available: {available}.")
    return AGN_BLOCKS[category][name]


# ──────────────────────────────────────────────────────────────────────
# Built-in "none" blocks: return zeros / identity factor. Useful as
# defaults when the user wants to skip a stage.
# ──────────────────────────────────────────────────────────────────────


@register_agn_block("disc", "none")
def _disc_none(wavelength: Array, agn_log_lbol: float, **_params) -> Array:
    r"""Skip the disc stage: emit zero L_lambda."""
    return jnp.zeros_like(jnp.asarray(wavelength))


@register_agn_block("nlr", "none")
def _nlr_none(wavelength: Array, agn_log_lbol: float, l5100_disc: Array, **_params) -> Array:
    r"""Skip the NLR stage: emit zero L_lambda."""
    return jnp.zeros_like(jnp.asarray(wavelength))


@register_agn_block("blr", "none")
def _blr_none(wavelength: Array, agn_log_lbol: float, l5100_disc: Array, **_params) -> Array:
    r"""Skip the BLR stage: emit zero L_lambda."""
    return jnp.zeros_like(jnp.asarray(wavelength))


@register_agn_block("feii", "none")
def _feii_none(wavelength: Array, agn_log_lbol: float, l5100_disc: Array, **_params) -> Array:
    r"""Skip the FeII stage: emit zero L_lambda."""
    return jnp.zeros_like(jnp.asarray(wavelength))


@register_agn_block("torus", "none")
def _torus_none(wavelength: Array, agn_log_lbol: float, l5100_disc: Array, **_params) -> Array:
    r"""Skip the torus stage: emit zero L_lambda."""
    return jnp.zeros_like(jnp.asarray(wavelength))


@register_agn_block("attenuation", "none")
def _attenuation_none(wavelength: Array, **_params) -> Array:
    r"""Skip attenuation: return identity factor (1.0 everywhere)."""
    return jnp.ones_like(jnp.asarray(wavelength))
