# SPDX-License-Identifier: BSD-3-Clause
"""The non-negativity helpers must reject an all-zero array.

That is the whole point of them. 235 tests in this tree assert
``jnp.all(x >= 0)`` and nothing else, and every one of those passes on
``zeros_like(x)`` — the exact output a misapplied mask or an unfilled buffer
produces. If ``assert_non_negative`` did not fail there, converting those call
sites would change nothing.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tests._bounds import (
    assert_in_unit_interval,
    assert_non_negative,
    assert_positive,
    not_all_zero,
)

pytestmark = pytest.mark.contract


class TestAssertNonNegative:
    def test_accepts_a_normal_positive_array(self):
        assert_non_negative(jnp.array([0.0, 1.0, 2.0]))

    def test_rejects_an_all_zero_array(self):
        """The failure the old assertion admitted."""
        with pytest.raises(AssertionError, match="entirely zero"):
            assert_non_negative(jnp.zeros(8))

    def test_the_old_assertion_passes_on_that_same_array(self):
        """Proves the upgrade is not cosmetic.

        Deliberately spelled the old way — this is the assertion the 197
        converted call sites used to make, and it must be seen passing on the
        array the new helper rejects.
        """
        zeros = jnp.zeros(8)
        assert bool(jnp.all(zeros >= 0.0))

    def test_rejects_negatives(self):
        with pytest.raises(AssertionError, match="negative element"):
            assert_non_negative(jnp.array([1.0, -1e-9, 2.0]))

    def test_reports_how_many_and_how_negative(self):
        """A bound failure is much easier to act on with the worst offender."""
        with pytest.raises(AssertionError, match=r"2 negative element\(s\) of 4"):
            assert_non_negative(jnp.array([1.0, -0.5, -2.0, 3.0]))

    def test_all_zero_is_allowed_when_stated(self):
        """Zero SFR, a fully absorbed SED — legitimate, but say so."""
        assert_non_negative(jnp.zeros(4), allow_all_zero=True)

    def test_a_single_positive_element_is_enough(self):
        """The guard is against a uniformly dead output, not against zeros."""
        assert_non_negative(jnp.array([0.0, 0.0, 0.0, 1e-30]))

    def test_an_empty_array_does_not_trip_the_zero_guard(self):
        """Emptiness is a different defect; this helper should not claim it."""
        assert_non_negative(jnp.array([]))

    def test_the_name_reaches_the_message(self):
        with pytest.raises(AssertionError, match="sed_dust_ir"):
            assert_non_negative(jnp.zeros(3), name="sed_dust_ir")


class TestAssertPositive:
    def test_accepts_strictly_positive(self):
        assert_positive(jnp.array([1e-30, 1.0]))

    def test_rejects_a_zero(self):
        """Distinct from non-negativity: here zero is itself the defect."""
        with pytest.raises(AssertionError, match="non-positive"):
            assert_positive(jnp.array([1.0, 0.0]))


class TestAssertInUnitInterval:
    def test_accepts_a_transmission_curve(self):
        assert_in_unit_interval(jnp.linspace(0.0, 1.0, 5))

    @pytest.mark.parametrize("bad", [-1e-9, 1.0 + 1e-9])
    def test_rejects_leaving_the_interval(self, bad):
        with pytest.raises(AssertionError, match="unit interval"):
            assert_in_unit_interval(jnp.array([0.5, bad]))

    def test_rejects_total_extinction(self):
        """An all-zero transmission is a bug, not a bound."""
        with pytest.raises(AssertionError, match="entirely zero"):
            assert_in_unit_interval(jnp.zeros(4))


class TestNotAllZeroPredicate:
    @pytest.mark.parametrize(
        "arr,expected",
        [
            (jnp.zeros(3), False),
            (jnp.array([0.0, 1.0]), True),
            (jnp.array([0.0, 1e-300]), True),
        ],
    )
    def test_predicate(self, arr, expected):
        assert not_all_zero(arr) is expected

    def test_it_returns_a_python_bool(self):
        """A traced or numpy bool leaking out would make `assert x is True`
        style checks silently wrong at call sites."""
        assert isinstance(not_all_zero(jnp.ones(2)), bool)
        assert not isinstance(not_all_zero(jnp.ones(2)), np.bool_)
