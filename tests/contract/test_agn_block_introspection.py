# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for composable AGN block introspection.

Verifies that blocks are discoverable via list_agn_blocks() and
describe_agn_block(), with consistent metadata (citation, status, short_doc).
"""

import pytest

import tengri


@pytest.mark.contract
def test_list_agn_blocks_returns_all_registered():
    """list_agn_blocks() enumerates all registered blocks across all categories."""
    from tengri.components.agn.blocks._protocol import AGN_BLOCKS

    blocks = tengri.list_agn_blocks()
    block_count = sum(len(names) for names in AGN_BLOCKS.values())
    assert len(blocks) == block_count, (
        f"Expected {block_count} blocks, got {len(blocks)}. "
        f"_RegistryTable should have exactly one row per (category, name) pair."
    )


@pytest.mark.contract
def test_list_agn_blocks_category_filter():
    """list_agn_blocks(category=...) filters to a single category."""
    for category in ("disc", "nlr", "blr", "feii", "torus", "attenuation"):
        blocks = tengri.list_agn_blocks(category=category)
        # Every block in the result should match the requested category.
        for block in blocks:
            assert block["category"] == category


@pytest.mark.contract
def test_list_agn_blocks_has_metadata():
    """Every block in list_agn_blocks() has status, citation, short_doc fields."""
    blocks = tengri.list_agn_blocks()
    assert len(blocks) > 0, "No blocks registered (sanity check)."

    for block in blocks:
        # Metadata fields should be present and typed correctly.
        assert "name" in block
        assert "category" in block
        assert "status" in block
        assert isinstance(block["status"], str)
        assert "citation" in block
        assert isinstance(block["citation"], str)
        assert "short_doc" in block
        assert isinstance(block["short_doc"], str)
        # Status must be one of the allowed values.
        assert block["status"] in (
            "production",
            "experimental",
            "demo",
            "deprecated",
        ), f"Unknown status '{block['status']}' for block {block['name']}"


@pytest.mark.contract
def test_describe_agn_block_with_category():
    """describe_agn_block(name, category=...) returns a single record."""
    # Assuming "none" is registered in multiple categories.
    disc_rec = tengri.describe_agn_block("none", category="disc")
    assert isinstance(disc_rec, dict)
    assert disc_rec["name"] == "none"
    assert disc_rec["category"] == "disc"


@pytest.mark.contract
def test_describe_agn_block_ambiguous_returns_list():
    """describe_agn_block(name) with ambiguous name returns list of records."""
    # Assuming "none" appears in multiple categories.
    result = tengri.describe_agn_block("none")
    # If "none" is ambiguous (appears in >1 category), result is a list.
    if isinstance(result, list):
        assert len(result) > 1, "Expected >1 match if result is a list"
        for rec in result:
            assert rec["name"] == "none"
    else:
        # If unambiguous, result is a single _DescribeRecord.
        assert result["name"] == "none"


@pytest.mark.contract
def test_describe_agn_block_unknown_raises_keyerror():
    """describe_agn_block() raises KeyError for unknown block names."""
    with pytest.raises(KeyError, match="Unknown AGN block"):
        tengri.describe_agn_block("this_block_definitely_does_not_exist_xyz")


@pytest.mark.contract
def test_list_agn_blocks_symmetric_with_describe():
    """Every block in list_agn_blocks() is describable via describe_agn_block()."""
    blocks = tengri.list_agn_blocks()
    for block in blocks:
        # Should be able to describe with category specified.
        rec = tengri.describe_agn_block(block["name"], category=block["category"])
        assert rec["name"] == block["name"]
        assert rec["category"] == block["category"]


@pytest.mark.contract
def test_generic_describe_finds_agn_blocks():
    """tengri.describe(name) returns AGN block records in cross-kind search."""
    # Neither the ``if blocks:`` nor the ``except KeyError`` below it survived
    # measurement: 49 blocks are registered and ``describe`` returns every key,
    # so both branches were dead. Together they let this test pass having
    # asserted nothing -- the ``if`` skipped the body, and the handler turned a
    # missing key (a real defect in ``describe``) into a skip whose message
    # blamed registration instead.
    blocks = tengri.list_agn_blocks()
    assert blocks, "no AGN blocks registered, so generic describe() is unverified"

    test_name = blocks[0]["name"]
    rec = tengri.describe(test_name)
    assert rec["name"] == test_name
    assert rec.get("kind") == "agn_block"


@pytest.mark.contract
def test_agn_block_metadata_not_empty_except_citation():
    """AGN block metadata fields have non-empty status, with citation/short_doc
    allowed to be empty (backfilled in later waves).
    """
    blocks = tengri.list_agn_blocks()
    for block in blocks:
        # status is required and must be non-empty.
        assert block["status"], (
            f"Block {block['name']} in category {block['category']} has empty status."
        )
        # citation and short_doc are allowed to be empty (backfilled later).
        # Just verify they exist and are strings.
        assert isinstance(block["citation"], str)
        assert isinstance(block["short_doc"], str)


@pytest.mark.contract
def test_agn_block_category_field_exists():
    """Every block row includes a 'category' field for filtering."""
    blocks = tengri.list_agn_blocks()
    from tengri.components.agn.blocks._protocol import BLOCK_CATEGORIES

    categories_seen = {b["category"] for b in blocks}
    # Sanity: we should see at least some of the defined categories.
    assert categories_seen.issubset(BLOCK_CATEGORIES), (
        f"Unexpected categories: {categories_seen - BLOCK_CATEGORIES}"
    )
