# SPDX-License-Identifier: BSD-3-Clause
"""#2484: the result record names the metric that ran, not the one requested.

``dense_mass_matrix=True`` above the D cap falls back to a diagonal metric with
a warning. The posterior's diagnostics must say ``"diag"`` for that run, so a
later census cannot mistake it for a dense one.
"""

from __future__ import annotations

import warnings

import pytest

from tengri.inference.backends.mcmc.nuts import (
    DENSE_MASS_MAX_DIM,
    mass_matrix_label,
    resolve_dense_mass_gate,
)

pytestmark = pytest.mark.regression_bug


def test_label_names_the_metric_that_ran():
    assert mass_matrix_label(True) == "dense"
    assert mass_matrix_label(False) == "diag"


def test_d31_dense_request_resolves_to_diag_and_is_recorded_as_diag():
    """A D=31 request for a dense metric runs diagonal, so the record says diag."""
    n_dim = DENSE_MASS_MAX_DIM + 1
    assert n_dim == 31
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        use_dense = resolve_dense_mass_gate(True, n_dim, method="mcmc_nuts")
    assert use_dense is False
    assert any("DIAGONAL" in str(w.message) for w in rec)
    assert mass_matrix_label(use_dense) == "diag"


def test_dense_below_cap_is_recorded_as_dense():
    use_dense = resolve_dense_mass_gate(True, DENSE_MASS_MAX_DIM, method="mcmc_nuts")
    assert use_dense is True
    assert mass_matrix_label(use_dense) == "dense"
