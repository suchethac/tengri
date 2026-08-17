# SPDX-License-Identifier: BSD-3-Clause
"""There is exactly one ``make_log_age_grid`` (#1402).

It was defined twice — in ``utils/grid.py`` and again in
``components/stellar/sfh/gp_sfh.py`` — and the copies had importers on the
forward model side. The inference side (formerly in ``inference/standardized.py``,
now inlined in ``loss_functions``) also consumed the grid. The hazard was
structural: a correction landing in one copy would have silently desynced the
forward model's age weights from the grid inference used. Nothing would raise;
the fit would just be biased.

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


def test_the_forward_consumer_uses_canonical():
    """The forward-model consumer must use the canonical grid (#1402)."""
    from tengri.components.stellar import component as forward_side
    from tengri.utils.grid import make_log_age_grid as canonical

    assert forward_side.make_log_age_grid is canonical


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


def test_the_bounds_have_one_source_too():
    """Unifying the function is not enough — ``log_age_grid_step`` needs the same.

    ``gp_sfh`` re-declared ``LOG_AGE_MIN`` / ``LOG_AGE_MAX`` as literals while
    ``make_log_age_grid`` defaults off ``utils/grid``'s. That is the same
    duplication one level down, and it is the level that survives unifying the
    function.
    """
    from tengri.components.stellar.sfh.gp_sfh import LOG_AGE_MAX, LOG_AGE_MIN
    from tengri.utils.grid import DEFAULT_LOG_AGE_MAX, DEFAULT_LOG_AGE_MIN

    assert LOG_AGE_MIN == DEFAULT_LOG_AGE_MIN
    assert LOG_AGE_MAX == DEFAULT_LOG_AGE_MAX


@pytest.mark.parametrize("n_grid", [16, 64, 256, 257])
def test_analytic_step_matches_the_grid_it_describes(n_grid):
    """The step must track the grid at ANY bounds, not just the current ones.

    ``test_grid_contract_is_unchanged`` above pins the endpoints at 6.0 / 10.14,
    so it catches an *accidental* bound change. It cannot catch a *deliberate*
    one, because updating that hardcoded 10.14 is the first thing a maintainer
    making the change would do. Measured on this branch before the bounds were
    aliased: widening ``DEFAULT_LOG_AGE_MAX`` to 10.20 and updating that
    assertion left the whole file green while ``log_age_grid_step(256)`` sat
    1.43 % away from the real spacing — an error that flows straight into
    ``StellarSEDComponent.apply``'s age weights under JIT.

    This assertion is relative, so it holds whatever the bounds become.
    """
    from tengri.components.stellar.sfh.gp_sfh import log_age_grid_step
    from tengri.utils.grid import make_log_age_grid

    grid = np.asarray(make_log_age_grid(n_grid))
    assert log_age_grid_step(n_grid) == pytest.approx(float(grid[1] - grid[0]), rel=1e-12)
