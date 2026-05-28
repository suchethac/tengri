# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #319 — NUTS warmup OOM at D >= 8 + dense mass.

The trace graph for full mass-matrix NUTS adaptation grows quadratically
in D and is amplified by ``mean_sfh_type='dense_basis'`` (per-sample
derived publishing). The documented worst case is a 22.78 GB peak on
a D=8 photometry fit. Small D <= 7 fits peak at 3-6 GB and are fine.

Until the underlying memory pressure is solved (separate issue), the
heuristic warning at least lets the user know to switch to
``dense_mass_matrix=False`` or ``method='mcmc_hmc'`` before their
machine OOMs.
"""

from __future__ import annotations

import warnings

import pytest

from tengri.inference.backends.mcmc.nuts import _maybe_warn_high_memory_nuts

pytestmark = pytest.mark.regression_bug


class _StubSpec:
    """Minimal stub matching the InferenceContext.spec attributes the
    heuristic reads (``stochastic`` and ``mean_sfh_type``)."""

    def __init__(self, mean_sfh_type=None, stochastic=False):
        self.mean_sfh_type = mean_sfh_type
        self.stochastic = stochastic


def _count_oom_warnings(warns):
    return [w for w in warns if "20+ GB" in str(w.message)]


class TestBug319HighMemoryWarning:
    def test_d6_dense_silent(self):
        """D=7 dense fits peak at 3-6 GB; no warning expected."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _maybe_warn_high_memory_nuts(n_dim=6, dense_mass_matrix=True, spec=_StubSpec("dpl"))
        assert not _count_oom_warnings(w)

    def test_d7_dense_silent(self):
        """Boundary case — D=7 is still inside the safe zone."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _maybe_warn_high_memory_nuts(n_dim=7, dense_mass_matrix=True, spec=_StubSpec("dpl"))
        assert not _count_oom_warnings(w)

    def test_d8_dense_warns(self):
        """D=8 — the documented OOM threshold."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _maybe_warn_high_memory_nuts(n_dim=8, dense_mass_matrix=True, spec=_StubSpec("dpl"))
        hits = _count_oom_warnings(w)
        assert hits, "#319: D=8 with dense_mass_matrix=True should warn"
        msg = str(hits[0].message)
        # The warning has to give the user a concrete escape hatch.
        assert "dense_mass_matrix=False" in msg
        assert "mcmc_hmc" in msg
        assert "#319" in msg

    def test_d20_dense_warns(self):
        """Larger D still warns — the warning is dimensionality-monotone."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _maybe_warn_high_memory_nuts(n_dim=20, dense_mass_matrix=True, spec=_StubSpec("dpl"))
        assert _count_oom_warnings(w)

    def test_diagonal_mass_silent_regardless_of_dim(self):
        """Diagonal mass-matrix adaptation is the recommended escape;
        don't second-guess users who pick it."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _maybe_warn_high_memory_nuts(n_dim=20, dense_mass_matrix=False, spec=_StubSpec("dpl"))
        assert not _count_oom_warnings(w)

    def test_stochastic_sfh_silent(self):
        """Stochastic-SFH fits hit a separate, more aggressive warning
        higher up in ``run_nuts``; don't double-warn here."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _maybe_warn_high_memory_nuts(
                n_dim=10, dense_mass_matrix=True, spec=_StubSpec("dpl", stochastic=True)
            )
        assert not _count_oom_warnings(w)

    def test_dense_basis_amplifier_hint(self):
        """When mean_sfh_type includes ``dense_basis`` the warning calls
        out the specific 22.78 GB number from the original report."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _maybe_warn_high_memory_nuts(
                n_dim=8,
                dense_mass_matrix=True,
                spec=_StubSpec(["dpl", "dense_basis"]),
            )
        hits = _count_oom_warnings(w)
        assert hits
        msg = str(hits[0].message)
        assert "dense_basis" in msg
        assert "22.78 GB" in msg

    def test_non_dense_basis_sfh_no_amplifier_hint(self):
        """Don't put the dense_basis amplifier hint on plain SFHs."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _maybe_warn_high_memory_nuts(n_dim=8, dense_mass_matrix=True, spec=_StubSpec("dpl"))
        hits = _count_oom_warnings(w)
        assert hits
        assert "dense_basis" not in str(hits[0].message)

    def test_string_or_list_mean_sfh_type_both_handled(self):
        """``mean_sfh_type`` can be a string or a list of strings — both
        should detect 'dense_basis' equivalently."""
        for sfh in ("dense_basis", ["dpl", "dense_basis"], ["dense_basis"]):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _maybe_warn_high_memory_nuts(n_dim=8, dense_mass_matrix=True, spec=_StubSpec(sfh))
            hits = _count_oom_warnings(w)
            assert hits, f"mean_sfh_type={sfh!r}: no warning"
            assert "dense_basis" in str(hits[0].message), f"mean_sfh_type={sfh!r}: missing hint"

    def test_no_mean_sfh_type_attribute_safe(self):
        """If the spec doesn't expose ``mean_sfh_type`` (defensive case)
        the heuristic must not crash."""

        class BareSpec:
            stochastic = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _maybe_warn_high_memory_nuts(n_dim=8, dense_mass_matrix=True, spec=BareSpec())
        assert _count_oom_warnings(w)
