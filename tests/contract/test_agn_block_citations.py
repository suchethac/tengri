# SPDX-License-Identifier: BSD-3-Clause
"""Conformance tests for AGN block metadata (citations, status, documentation).

Every registered AGN block must carry citation information, a status tag,
and a one-line description — matching the monolithic registry structure.

This test ensures:
1. Every real (non-`none`) block has a non-empty citation.
2. Monolithic models also have proper citations.
3. Block metadata is accessible via introspection API.

pytestmark: contract
"""

import pytest

import tengri
from tengri.components.agn.blocks._protocol import AGN_BLOCK_META


@pytest.mark.contract
class TestAGNBlockCitations:
    """Citations and metadata conformance for composable AGN blocks."""

    def test_all_real_blocks_have_citations(self):
        """Every non-none block must have a non-empty citation unless explicitly allowed.

        Generic phenomenological blocks with no canonical reference are allowed
        to have empty citations if they have clear short_doc descriptions.
        """
        # Explicitly allowed blocks with no canonical reference
        _GENERIC_NO_CITATION = {
            ("disc", "powerlaw"),  # Phenomenological single-slope power-law
            ("torus", "simple"),  # Phenomenological single-temperature graybody
            ("torus", "two_temperature"),  # Phenomenological two-temperature graybody
        }

        missing_citations = []
        for (category, name), meta in AGN_BLOCK_META.items():
            # Skip the `none` placeholder blocks
            if name == "none":
                continue
            # Skip explicitly-allowed generic blocks
            if (category, name) in _GENERIC_NO_CITATION:
                continue
            citation = meta.get("citation", "").strip()
            if not citation:
                missing_citations.append((category, name))

        assert not missing_citations, f"Blocks missing citations: {missing_citations}"

    def test_all_blocks_have_status(self):
        """Every block must declare a status."""
        missing_status = []
        for (category, name), meta in AGN_BLOCK_META.items():
            status = meta.get("status", "").strip()
            if not status:
                missing_status.append((category, name))

        assert not missing_status, f"Blocks missing status: {missing_status}"

    def test_all_blocks_have_short_doc(self):
        """Every block must have a one-line short_doc."""
        missing_doc = []
        for (category, name), meta in AGN_BLOCK_META.items():
            if name == "none":
                continue  # no-op placeholder blocks carry no metadata
            short_doc = meta.get("short_doc", "").strip()
            if not short_doc:
                missing_doc.append((category, name))

        assert not missing_doc, f"Blocks missing short_doc: {missing_doc}"

    def test_monolithic_models_have_citations(self):
        """Every monolithic AGN model exposes a non-empty citation.

        Uses the canonical ``list_agn_models()`` introspection (the composable
        meta-model ``composable`` is exempt — it is a dispatcher, not a cited
        physics model).
        """
        missing = [
            row["name"]
            for row in tengri.list_agn_models()
            if row.get("name") != "composable" and not str(row.get("citation", "")).strip()
        ]
        assert not missing, f"Monolithic models missing citations: {missing}"

    def test_block_introspection_api(self):
        """Verify list_agn_blocks() and describe_agn_block() work."""
        # list_agn_blocks should return a _RegistryTable (list-like)
        blocks_table = tengri.list_agn_blocks()
        assert hasattr(blocks_table, "__iter__"), "list_agn_blocks() should return iterable"
        assert len(blocks_table) > 0, "Should have at least one block"

        # Each row should have category and name
        categories_seen = set()
        for block in blocks_table:
            assert isinstance(block, dict), "Each block should be a dict"
            assert "category" in block, "Block should have 'category' key"
            assert "name" in block, "Block should have 'name' key"
            categories_seen.add(block["category"])

        # describe_agn_block should fetch metadata for each block
        for block in blocks_table:
            category = block["category"]
            block_name = block["name"]
            if block_name == "none":
                continue
            desc = tengri.describe_agn_block(block_name, category=category)
            assert desc is not None
            assert "citation" in desc
            assert "status" in desc
            assert "short_doc" in desc
