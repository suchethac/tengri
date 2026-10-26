# SPDX-License-Identifier: BSD-3-Clause
"""Refusal of keyword arguments the composable runner's selected blocks do not read.

``compose_l_nu(..., **params)`` hands every keyword to every stage, and each block takes the
keys it knows and ignores the rest. A key no selected block reads was therefore dropped in
silence: ``compose_l_nu(..., agn_disc_block='adaf_lopez2024', agn_cigale_disk_delta=0.3)``
returned the default ADAF blend after #2993 renamed that block's weight to
``agn_cigale_adaf_delta``. The grammar refuses such keys since #2994; this is the same rule on
the keyword path, with the same CIGALE spelling hints.
"""

from __future__ import annotations

import difflib
import inspect
from collections.abc import Callable, Mapping

from tengri.components.agn.blocks._consumes import agn_active_param_set
from tengri.components.agn.blocks._protocol import (
    DISC_POWER_BLOCKS,
    DISC_SPLIT_BLOCKS,
    DISC_STATE_BLOCKS,
    LINE_ENERGY_BLOCKS,
    TORUS_ANISOTROPY_BLOCKS,
    resolve_agn_block,
)
from tengri.config.exceptions import ParameterError

__all__ = [
    "PASS_THROUGH_KEYS",
    "RUNNER_READ_KEYS",
    "RUNNER_SUPPLIED_KEYS",
    "recipe_read_keys",
    "refuse_unread_keys",
]

#: Keys the runner itself reads, whatever the blocks: the policy, the shared normalization and
#: geometry, the CIGALE tie and the polar-dust budget.
RUNNER_READ_KEYS: frozenset[str] = frozenset(
    {
        "agn_norm",
        "agn_lum_ratio",
        "agn_cos_inc",
        "agn_torus_frac",
        "agn_ir_frac",
        "agn_ebv_disc",
        "agn_polar_oa",
        "agn_polar_law",
        "agn_polar_ebv",
        "agn_theta_torus",
        "agn_grahsp_log_l5100",
    }
)

#: Documented pass-through keys: the float32 shape hand-off (``agn_log_lbol_shape``), threaded
#: libraries (``templates``, ``_template``) and the solved disc state. They configure the
#: evaluation, not the model, and are accepted with every recipe.
PASS_THROUGH_KEYS: frozenset[str] = frozenset(
    {"agn_log_lbol_shape", "templates", "_template", "disc_state"}
)

#: Block keywords the runner fills itself (the grid, the dtype, the disc's 5100 A
#: luminosity), never a caller's.
RUNNER_SUPPLIED_KEYS: frozenset[str] = frozenset({"wavelength", "dtype", "l5100_disc"})

_STAGES: tuple[tuple[str, str], ...] = (
    ("disc", "agn_disc_block"),
    ("nlr", "agn_nlr_block"),
    ("blr", "agn_blr_block"),
    ("feii", "agn_feii_block"),
    ("torus", "agn_torus_block"),
    ("attenuation", "agn_attenuation_block"),
)


def _named_keywords(fn: Callable) -> set[str]:
    """The named (not ``**kwargs``) parameters of ``fn``."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return set()
    return {
        name
        for name, p in signature.parameters.items()
        if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }


def recipe_read_keys(selectors: Mapping[str, str]) -> frozenset[str]:
    """Every keyword the recipe's blocks and the runner read.

    Parameters
    ----------
    selectors : mapping
        ``{"agn_disc_block": ..., ..., "agn_attenuation_block": ...}``.

    Returns
    -------
    frozenset of str
        The union of :data:`RUNNER_READ_KEYS`, the recipe's consumed set
        (``_consumes.agn_active_param_set``) and the named keywords of every selected block
        and of the closed forms it registers (disc power, state and split, line energies,
        torus anisotropy), less the keywords the runner fills itself
        (:data:`RUNNER_SUPPLIED_KEYS`).
    """
    reads = set(RUNNER_READ_KEYS)
    reads |= agn_active_param_set({"agn_model": "composable", **selectors})
    for category, kwarg in _STAGES:
        name = selectors.get(kwarg, "none")
        reads |= _named_keywords(resolve_agn_block(category, name))
        for table in (
            (DISC_POWER_BLOCKS, DISC_STATE_BLOCKS, DISC_SPLIT_BLOCKS)
            if category == "disc"
            else (TORUS_ANISOTROPY_BLOCKS,)
            if category == "torus"
            else ()
        ):
            if name in table:
                reads |= _named_keywords(table[name])
        if (category, name) in LINE_ENERGY_BLOCKS:
            reads |= _named_keywords(LINE_ENERGY_BLOCKS[(category, name)])
    # Every named keyword of the selected blocks, not only the ``agn_*`` ones: the multicolor
    # disc's ``euv_tail`` is a model switch a caller passes by its own name.
    return frozenset(reads - RUNNER_SUPPLIED_KEYS)


def _spelling_hint(key: str, reads: frozenset[str]) -> str:
    """A CIGALE spelling hint and the closest read key, for one refused ``key``."""
    from tengri.parameters.agn_ownership import cigale_spelling_hint

    parts = key.split("_")
    cigale = next(
        (h for i in range(len(parts)) if (h := cigale_spelling_hint("_".join(parts[i:])))), ""
    )
    close = difflib.get_close_matches(key, sorted(reads), n=1, cutoff=0.8)
    nearest = f" Did you mean {close[0]!r}?" if close else ""
    return f"{nearest}{cigale}"


def refuse_unread_keys(selectors: Mapping[str, str], params: Mapping) -> None:
    """Raise when ``params`` holds a key no selected block and not the runner reads.

    Parameters
    ----------
    selectors : mapping
        The six block selectors.
    params : mapping
        The keyword arguments handed to the runner.

    Raises
    ------
    ParameterError
        Naming each unread key, the keys the recipe reads, and a spelling hint where one
        exists (the CIGALE table of #2994, or the closest read key).

    Notes
    -----
    **JIT-compatible**: composition-time only (string keys).
    """
    reads = recipe_read_keys(selectors)
    unread = sorted(k for k in params if k not in reads and k not in PASS_THROUGH_KEYS)
    if not unread:
        return
    recipe = ", ".join(f"{k}={v!r}" for k, v in selectors.items())
    hints = "".join(_spelling_hint(k, reads) for k in unread)
    noun = "keys" if len(unread) > 1 else "key"
    raise ParameterError(
        f"compose_l_nu: no block of the recipe ({recipe}) reads the {noun} "
        f"{', '.join(repr(k) for k in unread)}, so it would be silently ignored. The recipe "
        f"reads: {', '.join(sorted(reads))}.{hints}"
    )
