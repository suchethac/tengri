# SPDX-License-Identifier: BSD-3-Clause
"""``_resolve_dense_mass_matrix``'s D<=12, ``dense_basis``-aware auto-policy.

Replaces the D < 8 cliff with dense at D <= 12 unless the spec's SFH is
``dense_basis``, measured on ``ctl-dpl`` (D=8 photometry, 14 bands, DPL SFH):
the dense window adaptation uses 1.1-1.9 GB RSS and costs 3.4x fewer
gradients per effective sample than diagonal (dense 34 g/draw, ESS 83;
diagonal 550 g/draw, ESS 113; six-seed sweep). ``dense_basis`` keeps its
diagonal fallback at any D, unchanged, because its per-sample derived
quantity publishing -- not dimensionality on its own -- is the historical
20+ GB warmup driver (#319, 22.78 GB peak at D=8).

See ``tests/unit/inference/test_dense_mass_gate.py`` for the shared
``resolve_dense_mass_gate``/``DENSE_MASS_MAX_DIM`` cap tests, which this file
does not duplicate.
"""

from __future__ import annotations

import pytest

from tengri.inference.backends.mcmc.nuts import (
    _resolve_dense_mass_matrix,
    _spec_uses_dense_basis,
    run_nuts,
)

pytestmark = pytest.mark.contract


class _StubSpec:
    """Minimal stand-in for ``Parameters``, carrying only ``mean_sfh_type``."""

    def __init__(self, mean_sfh_type):
        self.mean_sfh_type = mean_sfh_type


@pytest.mark.parametrize("n_dim", [7, 8, 12])
def test_dense_at_and_below_twelve(n_dim):
    """No spec (or a non-``dense_basis`` spec) is dense up to D = 12."""
    assert _resolve_dense_mass_matrix(None, n_dim) is True
    assert _resolve_dense_mass_matrix(None, n_dim, spec=_StubSpec("dpl")) is True


def test_diagonal_above_twelve():
    """D = 13 crosses the new threshold into diagonal."""
    assert _resolve_dense_mass_matrix(None, 13) is False
    assert _resolve_dense_mass_matrix(None, 13, spec=_StubSpec("dpl")) is False


@pytest.mark.parametrize("n_dim", [7, 8, 12])
def test_dense_basis_is_diagonal_throughout_the_dense_band(n_dim):
    """A ``dense_basis`` SFH stays diagonal even where D <= 12 would be dense."""
    assert _resolve_dense_mass_matrix(None, n_dim, spec=_StubSpec("dense_basis")) is False
    assert _resolve_dense_mass_matrix(None, n_dim, spec=_StubSpec(["dpl", "dense_basis"])) is False


def test_dense_basis_predicate_handles_string_list_and_missing_attribute():
    """``_spec_uses_dense_basis`` -- the predicate reused from the OOM warning."""
    assert _spec_uses_dense_basis(_StubSpec("dense_basis")) is True
    assert _spec_uses_dense_basis(_StubSpec(["dpl", "dense_basis"])) is True
    assert _spec_uses_dense_basis(_StubSpec("dpl")) is False
    assert _spec_uses_dense_basis(None) is False

    class BareSpec:
        pass

    assert _spec_uses_dense_basis(BareSpec()) is False


def test_explicit_true_false_round_trip_regardless_of_dim_or_spec():
    """An explicit request always wins, at any D and with any spec."""
    assert _resolve_dense_mass_matrix(True, 20) is True
    assert _resolve_dense_mass_matrix(True, 8, spec=_StubSpec("dense_basis")) is True
    assert _resolve_dense_mass_matrix(False, 3) is False
    assert _resolve_dense_mass_matrix(False, 8, spec=_StubSpec("dpl")) is False


def test_run_nuts_default_is_still_none():
    """``run_nuts`` must keep exposing the auto-policy as its default."""
    import inspect

    sig = inspect.signature(run_nuts)
    assert sig.parameters["dense_mass_matrix"].default is None
