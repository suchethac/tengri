# SPDX-License-Identifier: BSD-3-Clause
"""AGN block subsystem: fine-grained, composable AGN sub-components.

See :mod:`._protocol` for the contract and :mod:`.runner` for the
:func:`composable_agn_l_nu` runner. Importing this package side-effects
registration of all built-in blocks (GRAHSP suite, simple alternates,
``none`` no-ops).

Public API
----------

- :data:`AGN_BLOCKS`: two-level registry (category → name → callable).
- :func:`register_agn_block`: decorator for new block impls.
- :func:`resolve_agn_block`: registry lookup.
- :func:`composable_agn_l_nu`: the AGN_MODELS["composable"] entry point.
- :func:`validate_block_recipe`: construction-time recipe validator.
- :class:`RecipeWarning`: warning subclass for suspicious combos.

"""

# Importing these submodules side-effects @register_agn_block calls.
# Side-effect: registers AGN_MODELS["composable"]. Imported last to avoid
# circular imports: registry depends on the runner being defined.
# composable_precompute is the third (precomputed-lookup) evaluation mode;
# mirror of qsogen_precompute / skirtor_precompute for the composable runner.
from tengri.components.agn.blocks import (
    alternates,
    atten,
    blr,
    composable_precompute,
    disc,
    feii,
    grahsp_blocks,
    nlr,
    qsogen_blocks,
    registry,
    torus,
)
from tengri.components.agn.blocks._protocol import (
    AGN_BLOCKS,
    AGN_NORM_POLICIES,
    BLOCK_CATEGORIES,
    register_agn_block,
    resolve_agn_block,
)
from tengri.components.agn.blocks.recipe import Recipe
from tengri.components.agn.blocks.runner import (
    BLOCK_SELECTOR_KEYS,
    DEFAULT_BLOCK_SELECTORS,
    RecipeWarning,
    composable_agn_l_nu,
    compose_l_nu,
    validate_block_recipe,
)

__all__ = [
    "AGN_BLOCKS",
    "AGN_NORM_POLICIES",
    "BLOCK_CATEGORIES",
    "BLOCK_SELECTOR_KEYS",
    "DEFAULT_BLOCK_SELECTORS",
    "Recipe",
    "RecipeWarning",
    "composable_agn_l_nu",
    "composable_precompute",
    "compose_l_nu",
    "register_agn_block",
    "resolve_agn_block",
    "validate_block_recipe",
]
