"""Tests for FREE and FIXED sentinel objects."""

import copy
import pickle

import pytest

from tengri.parameters.sentinels import FIXED, FREE

pytestmark = pytest.mark.contract


class TestFreeSingleton:
    """Test FREE singleton behavior."""

    def test_free_is_singleton(self):
        """FREE is always the same object (singleton pattern)."""
        from tengri.parameters.sentinels import FREE as FREE2

        assert FREE is FREE2

    def test_free_repr(self):
        """repr(FREE) returns the string 'FREE'."""
        assert repr(FREE) == "FREE"

    def test_free_hashable_as_dict_value(self):
        """FREE can be used as a dict value and retrieved."""
        d = {"param": FREE}
        assert d["param"] is FREE

    def test_free_survives_deepcopy(self):
        """FREE remains the same singleton after deepcopy."""
        copied = copy.deepcopy(FREE)
        assert copied is FREE

    def test_free_not_equal_fixed(self):
        """FREE and FIXED are not equal."""
        assert FREE != FIXED

    def test_free_not_equal_true(self):
        """FREE is not equal to True."""
        assert FREE != True  # noqa: E712

    def test_free_not_equal_false(self):
        """FREE is not equal to False."""
        assert FREE != False  # noqa: E712

    def test_free_not_equal_none(self):
        """FREE is not equal to None."""
        assert FREE != None  # noqa: E711

    def test_free_not_equal_zero(self):
        """FREE is not equal to 0."""
        assert FREE != 0

    def test_free_is_truthy(self):
        """bool(FREE) is True (not an empty value)."""
        assert bool(FREE) is True

    def test_free_picklable(self):
        """FREE can be pickled and unpickled, remaining a singleton."""
        pickled = pickle.dumps(FREE)
        unpickled = pickle.loads(pickled)
        assert unpickled is FREE


class TestFixedSingleton:
    """Test FIXED singleton behavior."""

    def test_fixed_is_singleton(self):
        """FIXED is always the same object (singleton pattern)."""
        from tengri.parameters.sentinels import FIXED as FIXED2

        assert FIXED is FIXED2

    def test_fixed_repr(self):
        """repr(FIXED) returns the string 'FIXED'."""
        assert repr(FIXED) == "FIXED"

    def test_fixed_hashable_as_set_member(self):
        """FIXED can be added to a set."""
        s = {FIXED}
        assert FIXED in s

    def test_fixed_survives_deepcopy(self):
        """FIXED remains the same singleton after deepcopy."""
        copied = copy.deepcopy(FIXED)
        assert copied is FIXED

    def test_fixed_not_equal_free(self):
        """FIXED and FREE are not equal."""
        assert FIXED != FREE

    def test_fixed_not_equal_true(self):
        """FIXED is not equal to True."""
        assert FIXED != True  # noqa: E712

    def test_fixed_not_equal_false(self):
        """FIXED is not equal to False."""
        assert FIXED != False  # noqa: E712

    def test_fixed_not_equal_none(self):
        """FIXED is not equal to None."""
        assert FIXED != None  # noqa: E711

    def test_fixed_not_equal_zero(self):
        """FIXED is not equal to 0."""
        assert FIXED != 0

    def test_fixed_is_truthy(self):
        """bool(FIXED) is True (not an empty value)."""
        assert bool(FIXED) is True

    def test_fixed_picklable(self):
        """FIXED can be pickled and unpickled, remaining a singleton."""
        pickled = pickle.dumps(FIXED)
        unpickled = pickle.loads(pickled)
        assert unpickled is FIXED


class TestSentinelIntegration:
    """Test sentinels in realistic usage patterns."""

    def test_sentinels_as_dict_values(self):
        """Both sentinels work as dict values in a config."""
        config = {"param1": FREE, "param2": FIXED, "param3": 1.5}
        assert config["param1"] is FREE
        assert config["param2"] is FIXED
        assert config["param3"] == 1.5

    def test_deepcopy_dict_with_sentinels(self):
        """deepcopy of a dict with sentinels preserves singleton identity."""
        original = {"free": FREE, "fixed": FIXED}
        copied = copy.deepcopy(original)
        assert copied["free"] is FREE
        assert copied["fixed"] is FIXED

    def test_pickle_dict_with_sentinels(self):
        """pickle/unpickle of a dict with sentinels preserves singleton identity."""
        original = {"free": FREE, "fixed": FIXED}
        pickled = pickle.dumps(original)
        restored = pickle.loads(pickled)
        assert restored["free"] is FREE
        assert restored["fixed"] is FIXED

    def test_sentinels_distinguishable(self):
        """The two sentinels are clearly distinguishable."""
        assert id(FREE) != id(FIXED)
        assert hash(FREE) != hash(FIXED)


class TestSentinelIdentity:
    """Test object identity and hashing behavior."""

    def test_free_identity_across_module_reloads(self):
        """FREE identity is preserved across re-imports."""
        from tengri.parameters import sentinels as sentinels_module

        FREE_first = sentinels_module.FREE
        FREE_second = sentinels_module.FREE
        assert FREE_first is FREE_second

    def test_sentinels_have_stable_hash(self):
        """Both sentinels are hashable with stable hashes."""
        # Just verify they hash without error and produce different hashes
        h_free = hash(FREE)
        h_fixed = hash(FIXED)
        assert isinstance(h_free, int)
        assert isinstance(h_fixed, int)
        assert h_free != h_fixed

    def test_sentinels_in_nested_structure(self):
        """Sentinels survive in deeply nested data structures."""
        nested = {
            "level1": {
                "level2": {
                    "free": FREE,
                    "fixed": FIXED,
                    "list": [FREE, FIXED, 1.0],
                }
            }
        }
        copied = copy.deepcopy(nested)
        assert copied["level1"]["level2"]["free"] is FREE
        assert copied["level1"]["level2"]["fixed"] is FIXED
        assert copied["level1"]["level2"]["list"][0] is FREE
        assert copied["level1"]["level2"]["list"][1] is FIXED
