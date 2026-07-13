# SPDX-License-Identifier: BSD-3-Clause
"""The ``--shard I/N`` split must lose no tests.

Sharding a tree across CI runners is only safe if the shards *partition* the
suite: every test lands in exactly one shard. The failure this guards against is
the quiet one — an off-by-one in the round-robin drops a slice of the tree, every
shard still reports green, and the suite silently stops testing what it claims to.
A red CI is cheap; a green CI that runs fewer tests than it says is not.
"""

import pytest

from tests.conftest import parse_shard, select_shard

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("total", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("n_items", [0, 1, 5, 258, 1000])
def test_shards_partition_the_suite_exactly(n_items, total):
    """Union of all shards == the original selection, with no test in two shards."""
    items = [f"test_{i}" for i in range(n_items)]

    shards = [select_shard(items, i, total)[0] for i in range(1, total + 1)]
    union = [item for shard in shards for item in shard]

    assert sorted(union) == sorted(items), (
        f"{total} shards over {n_items} items did not reproduce the suite: "
        f"{len(union)} selected vs {len(items)} collected — tests were dropped or duplicated."
    )
    assert len(set(union)) == len(union), "a test was claimed by more than one shard"


@pytest.mark.parametrize("total", [2, 3, 4])
def test_selected_and_deselected_are_complementary(total):
    """Each shard's (selected, deselected) covers the whole list, disjointly."""
    items = [f"test_{i}" for i in range(100)]
    for index in range(1, total + 1):
        selected, deselected = select_shard(items, index, total)
        assert set(selected).isdisjoint(deselected)
        assert sorted(selected + deselected) == sorted(items)


def test_shards_are_balanced_to_within_one():
    """Round-robin must not hand one runner a disproportionate slice.

    Balance by count is not balance by wall clock, but a count imbalance would be
    a bug in the partition itself — that is what is checked here.
    """
    items = [f"test_{i}" for i in range(258)]
    sizes = [len(select_shard(items, i, 4)[0]) for i in range(1, 5)]
    assert max(sizes) - min(sizes) <= 1, f"round-robin shards are lopsided: {sizes}"


def test_one_shard_is_the_identity():
    """``--shard 1/1`` must select everything — the no-op case CI relies on."""
    items = [f"test_{i}" for i in range(50)]
    selected, deselected = select_shard(items, 1, 1)
    assert selected == items
    assert deselected == []


@pytest.mark.parametrize("spec", ["1/2", "2/2", "4/4", "1/1"])
def test_parse_shard_accepts_well_formed_specs(spec):
    index, total = parse_shard(spec)
    assert 1 <= index <= total


@pytest.mark.parametrize("spec", ["0/2", "3/2", "-1/4", "1/0", "abc", "1", "", "1/x", "1/2/3"])
def test_parse_shard_rejects_bad_specs(spec):
    """A malformed shard spec must raise, never quietly degrade.

    Falling back to "run everything" would multiply CI cost while looking green;
    falling back to "run nothing" would look green while testing nothing.
    """
    with pytest.raises(ValueError):
        parse_shard(spec)
