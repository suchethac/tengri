# SPDX-License-Identifier: BSD-3-Clause
"""Tests for FREE and FIXED sentinel objects.

27 tests become 8. `TestFreeSingleton` and `TestFixedSingleton` were the same
ten tests written twice, differing only in which sentinel they named, so they
are one parametrized class here; the four "not equal to True / False / None / 0"
tests per sentinel are one test over a table of look-alikes.

Two did not survive:

* `test_free_identity_across_module_reloads` was named for a property that does
  not hold, and did not test it. Its body read ``sentinels_module.FREE`` twice
  and asserted the two were identical -- ``x is x``. Measured 2026-08-17: after
  ``importlib.reload`` the sentinel is a **different, non-equal** object, which
  is ordinary for a module-level singleton and is now stated below rather than
  asserted wrongly.
* `test_sentinels_have_stable_hash` asserted ``isinstance(hash(x), int)``, which
  is true of every hashable object, and then re-asserted the distinctness that
  `test_the_sentinels_are_distinguishable` already covers.
"""

import copy
import pickle

import pytest

from tengri.parameters.sentinels import FIXED, FREE

pytestmark = pytest.mark.contract

#: Values a sentinel must not compare equal to. `FREE == True` would be the
#: dangerous one: the grammar accepts booleans in some positions, so a sentinel
#: that equalled ``True`` would silently pass a truthiness check meant for a
#: real flag.
_LOOKALIKES = [True, False, None, 0, 0.0, "", [], {}]


@pytest.mark.parametrize("sentinel", [FREE, FIXED], ids=repr)
class TestEachSentinel:
    """Properties both sentinels must satisfy, stated once."""

    def test_is_a_singleton(self, sentinel):
        """Importing the name again yields the same object."""
        from tengri.parameters import sentinels

        assert getattr(sentinels, repr(sentinel)) is sentinel

    def test_repr_is_its_own_name(self, sentinel):
        """``repr`` is the bare name, which is what `spec.summary()` prints."""
        assert repr(sentinel) in {"FREE", "FIXED"}

    def test_survives_deepcopy(self, sentinel):
        assert copy.deepcopy(sentinel) is sentinel

    def test_survives_pickle(self, sentinel):
        assert pickle.loads(pickle.dumps(sentinel)) is sentinel

    def test_is_hashable_as_dict_value_and_set_member(self, sentinel):
        assert {"param": sentinel}["param"] is sentinel
        assert sentinel in {sentinel}

    def test_is_truthy_and_unequal_to_every_lookalike(self, sentinel):
        """Truthy, and distinct from every falsy value it could be mistaken for.

        One test over a table rather than four near-identical ones per
        sentinel, and the table is wider than the four the originals checked --
        ``0.0``, ``""``, ``[]`` and ``{}`` were not covered and are exactly the
        values an ``if not value:`` check would confuse a sentinel with.
        """
        assert bool(sentinel) is True
        for other in _LOOKALIKES:
            assert sentinel != other, f"{sentinel!r} compares equal to {other!r}"


class TestTheTwoTogether:
    """Properties about the pair, which the parametrized class cannot express."""

    def test_the_sentinels_are_distinguishable(self):
        """Different objects, unequal, and separately hashable."""
        assert FREE is not FIXED
        assert FREE != FIXED
        assert hash(FREE) != hash(FIXED)

    def test_they_keep_their_identity_inside_a_nested_structure(self):
        """deepcopy and pickle of a container must not clone the singletons.

        This is the property that matters in practice: a `Parameters` spec is a
        nested dict of these, and a fitter that deep-copied it into new objects
        would break every ``is FREE`` check downstream.
        """
        nested = {"a": {"b": {"free": FREE, "fixed": FIXED, "list": [FREE, FIXED, 1.0]}}}

        for restored in (copy.deepcopy(nested), pickle.loads(pickle.dumps(nested))):
            inner = restored["a"]["b"]
            assert inner["free"] is FREE
            assert inner["fixed"] is FIXED
            assert inner["list"][0] is FREE
            assert inner["list"][1] is FIXED
            assert inner["list"][2] == 1.0
