# SPDX-License-Identifier: BSD-3-Clause
"""``make_log_age_grid`` has exactly one definition (#1402).

It used to be defined twice — ``utils/grid.py`` and
``components/stellar/sfh/gp_sfh.py`` — as two distinct function objects with
live importers on opposite sides of the forward/inference boundary:

* ``components/stellar/component.py`` and ``sfh/_prior_sampling.py`` took the
  ``gp_sfh`` copy (the forward model's age weights),
* ``inference/standardized.py`` took the ``utils.grid`` copy (the parameter
  standardization inference runs on).

The two agreed numerically, so there was no bug to observe. That is precisely
what made it dangerous: this is the SFH age grid, and a correction landing in
one copy and not the other would silently desync the forward model from
inference. Nothing raises; the fit is just biased.

Identity is the assertion that cannot rot — comparing *values* would keep
passing after someone re-forks the function, which is the failure mode.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.contract


def test_single_function_object_across_both_import_paths():
    """The forward-side and inference-side imports must be the same object."""
    from tengri.components.stellar.sfh.gp_sfh import make_log_age_grid as forward_side
    from tengri.utils.grid import make_log_age_grid as inference_side

    assert forward_side is inference_side, (
        "make_log_age_grid has been re-forked: gp_sfh and utils.grid hold "
        "distinct function objects again (#1402)"
    )


def test_public_namespaces_agree():
    """``tengri.make_log_age_grid`` and ``tengri.utils.make_log_age_grid`` are one.

    Before #1402 these were genuinely different objects — the top-level export
    resolved to the ``gp_sfh`` copy while ``tengri.utils`` resolved to the
    ``utils.grid`` copy — even though ``docs/api/utils.rst`` documents the
    top-level name under a *utils* heading.
    """
    import tengri
    import tengri.utils as tengri_utils
    from tengri.utils.grid import make_log_age_grid as canonical

    assert tengri.make_log_age_grid is canonical
    assert tengri_utils.make_log_age_grid is canonical


def test_only_one_module_defines_it():
    """Guard the *definition*, not just the bindings.

    A re-export satisfies the identity checks above; so would a module that
    defines its own copy and then rebinds the name. Checking ``__module__``
    pins where the single definition actually lives.
    """
    from tengri.components.stellar.sfh.gp_sfh import make_log_age_grid

    assert make_log_age_grid.__module__ == "tengri.utils.grid", (
        f"make_log_age_grid is defined in {make_log_age_grid.__module__!r}; "
        "utils.grid is the single home (#1402)"
    )


def test_grid_bounds_share_one_source():
    """``log_age_grid_step``'s defaults must track the grid's, not restate them.

    ``log_age_grid_step`` computes the spacing analytically from ``n_grid`` and
    the bounds, so it is only correct while its bounds equal the ones
    ``make_log_age_grid`` actually uses. Re-declaring 6.0 / 10.14 in gp_sfh
    would reintroduce the same desync one level down.
    """
    from tengri.components.stellar.sfh.gp_sfh import LOG_AGE_MAX, LOG_AGE_MIN
    from tengri.utils.grid import DEFAULT_LOG_AGE_MAX, DEFAULT_LOG_AGE_MIN

    assert LOG_AGE_MIN == DEFAULT_LOG_AGE_MIN
    assert LOG_AGE_MAX == DEFAULT_LOG_AGE_MAX


@pytest.mark.parametrize("n_grid", [16, 64, 256, 257])
def test_analytic_step_matches_the_grid_it_describes(n_grid):
    """The two must agree at every n, including odd n.

    This is the property the shared-bounds test protects, asserted end to end:
    if the bounds ever diverge, the analytic step silently stops describing the
    grid the forward model is evaluated on.
    """
    from tengri.components.stellar.sfh.gp_sfh import log_age_grid_step
    from tengri.utils.grid import make_log_age_grid

    grid = np.asarray(make_log_age_grid(n_grid))
    assert log_age_grid_step(n_grid) == pytest.approx(float(grid[1] - grid[0]), rel=1e-12)
