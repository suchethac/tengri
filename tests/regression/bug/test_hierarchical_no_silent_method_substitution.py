# SPDX-License-Identifier: BSD-3-Clause
"""``PopulationFitter.run`` must never substitute a different algorithm.

``_HIERARCHICAL_OVERRIDES = {"mcmc_ess": "native_vi_linear"}`` rewrote the
caller's method before dispatch. Someone who asked for elliptical slice
sampling got MGVI instead — a different algorithm, registered
``tier="broken"`` after #231, with nothing in ``diagnostics`` recording the
swap.

Renaming a deprecated *spelling* is fine; ``resolve_method`` does exactly
that and warns. Swapping the *algorithm* is not: the two answer different
questions, and the caller has no way to find out which one ran.

The repair is to let an unsupported method reach the existing
``ValueError``. That message must name the method the caller actually
asked for — an error naming ``native_vi_linear`` to someone who typed
``mcmc_ess`` sends them to debug a backend they never selected.

The supported list in that message is derived from the dispatch table
rather than hand-written, so it cannot drift from what the code accepts
(the same defect #1394 found in the ``forward_chunk_size`` warning).
"""

from __future__ import annotations

import inspect

import pytest

from tengri.inference import hierarchical
from tengri.inference.hierarchical import PopulationFitter

pytestmark = pytest.mark.regression_bug


def test_no_override_table_remains():
    """The substitution table is gone from the dispatch source."""
    src = inspect.getsource(PopulationFitter.run)
    assert "_HIERARCHICAL_OVERRIDES" not in src, (
        "A method-substitution table reappeared in PopulationFitter.run. "
        "Support the method or raise; never swap the algorithm silently."
    )


def test_unsupported_method_error_names_the_requested_method():
    """``mcmc_ess`` must raise naming itself, not the backend it mapped onto."""
    src = inspect.getsource(hierarchical)
    # The old table is the only place these two names sat adjacent.
    assert '"mcmc_ess": "native_vi_linear"' not in src
    assert "'mcmc_ess': 'native_vi_linear'" not in src


@pytest.mark.parametrize("method", ["mcmc_ess", "mcmc_nuts", "map", "nss"])
def test_methods_absent_from_the_table_are_reported_as_unsupported(method):
    """Every registry method missing from ``_method_map`` reaches the raise.

    These four are all genuinely absent from the hierarchical dispatch. The
    point is not that they *should* work — it is that asking for them gets a
    message about the method asked for.
    """
    src = inspect.getsource(PopulationFitter.run)
    table_line = [ln for ln in src.splitlines() if "_method_map = {" in ln]
    assert table_line, "dispatch table not found — did run() get restructured?"
    assert f'"{method}"' not in src.split("_method_map = {")[1].split("}")[0], (
        f"{method!r} is now in _method_map; this test's premise is stale. "
        "If the hierarchical path gained it, drop it from the parametrize list."
    )


def test_supported_list_is_derived_not_handwritten():
    """The ValueError must build its list from the table, not a literal.

    A hand-written enumeration of what the code supports is a stale claim
    waiting to happen — #1394 shipped exactly that defect in the
    ``forward_chunk_size`` warning, steering callers off a working path.
    """
    src = inspect.getsource(PopulationFitter.run)
    raise_block = src.split("Unknown method")[1][:600]
    assert "_method_map" in raise_block, (
        "The unsupported-method message hardcodes its list. Derive it from "
        "_method_map so it cannot drift from the dispatch table."
    )
