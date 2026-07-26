# SPDX-License-Identifier: BSD-3-Clause
"""There is exactly one ``make_log_age_grid`` (#1402).

It was defined twice — in ``utils/grid.py`` and again in
``components/stellar/sfh/gp_sfh.py`` — and *both* copies had live importers on
opposite sides of the forward/inference boundary:

===========================================  ===================================
``utils/grid.py``                            ``inference/standardized.py``
``components/stellar/sfh/gp_sfh.py``         ``components/stellar/component.py``
===========================================  ===================================

The copies were numerically identical when this was found (``max|A-B| = 0`` at
n_grid 16, 64, 256), so nothing was wrong at the time. The hazard was structural:
this is the SFH age grid, and a correction landing in one copy would have
silently desynced the forward model's age weights from the grid inference
standardizes against. Nothing would raise; the fit would just be biased.

Identity is asserted rather than equality of values. Two copies that agree today
pass a value comparison, which is exactly how this survived — so the assertion
has to be that there is one object, not that two objects match.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.contract


def test_every_import_path_yields_the_same_object():
    """A re-fork fails here, not in a bias nobody notices."""
    import tengri
    from tengri.components.stellar.sfh.gp_sfh import make_log_age_grid as gp_sfh_copy
    from tengri.components.stellar.sfh.registry import make_log_age_grid as registry_copy
    from tengri.utils.grid import make_log_age_grid as canonical

    assert gp_sfh_copy is canonical, (
        "components.stellar.sfh.gp_sfh.make_log_age_grid is a second definition; "
        "re-export tengri.utils.grid.make_log_age_grid instead (#1402)."
    )
    assert registry_copy is canonical
    assert tengri.make_log_age_grid is canonical, (
        "the public tengri.make_log_age_grid must be the canonical object"
    )


def test_the_forward_and_inference_consumers_agree():
    """Pin the two modules whose disagreement would be the actual bug."""
    from tengri.components.stellar import component as forward_side
    from tengri.inference import standardized as inference_side
    from tengri.utils.grid import make_log_age_grid as canonical

    assert forward_side.make_log_age_grid is canonical
    assert inference_side.make_log_age_grid is canonical


def test_grid_contract_is_unchanged():
    """The collapse must not have moved the grid — defaults and endpoints."""
    from tengri.utils.grid import make_log_age_grid

    grid = np.asarray(make_log_age_grid())
    assert grid.shape == (256,)
    assert float(grid[0]) == pytest.approx(6.0)
    assert float(grid[-1]) == pytest.approx(10.14)

    # Uniform in log10(age/yr) — the property every consumer relies on.
    steps = np.diff(grid)
    assert np.allclose(steps, steps[0])

    for n in (16, 64, 256):
        assert np.asarray(make_log_age_grid(n)).shape == (n,)
