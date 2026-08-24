# SPDX-License-Identifier: BSD-3-Clause
"""AGN-block view of the component-generic grid-support registry.

The concept, the helpers and the registry now live in
:mod:`tengri.components.grid_support`, because the same defect exists wherever
a template-backed component clips a parameter onto a grid axis -- dust IR
libraries as much as AGN disc grids (#1586). This module keeps the AGN-block
spelling of that registry: the runner selects blocks by bare ``category``
(``'disc'``), while the general table is keyed by the dotted selector a user
writes in the build grammar (``'agn.disc'``).

Nothing here duplicates the general logic; it only translates the key.
"""

from __future__ import annotations

from tengri.components.grid_support import (
    GridSupportFn,
    describe_clipping,
    grid_support,
    is_contained,
    live_fraction,
)

# There is deliberately no AGN-local copy of the registry. An earlier draft
# exported one, derived from GRID_SUPPORT by a comprehension -- which meant
# registering or monkeypatching an entry there had no effect, because lookups
# went to the general table. Two dicts that must agree is the same drift
# generator this module exists to stop. Register in
# :data:`tengri.components.grid_support.GRID_SUPPORT`, keyed ``('agn.<category>',
# name)``.


def block_grid_support(category: str, name: str) -> dict[str, tuple[float, float]]:
    """Return the grid support of one AGN block, or ``{}`` if it has none.

    Parameters
    ----------
    category : str
        Pipeline stage, e.g. ``'disc'``.
    name : str
        Block name, e.g. ``'slone_netzer'``.

    Returns
    -------
    support : dict[str, tuple[float, float]]
        ``{param_name: (lo, hi)}``. Empty when the block is not
        template-backed **or** when its grid is not installed.

    Notes
    -----
    **JIT-compatible**: not applicable -- composition-time only.
    """
    return grid_support(f"agn.{category}", name)


__all__ = [
    "GridSupportFn",
    "block_grid_support",
    "describe_clipping",
    "is_contained",
    "live_fraction",
]
