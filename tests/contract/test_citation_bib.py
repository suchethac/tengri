# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the BibTeX parser and .bib-driven citation registry."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract


def test_registry_loaded_from_bib():
    """REGISTRY is populated by parsing references.bib, not by hard-coded entries."""
    from tengri.citations.registry import BIB_PATH, REGISTRY

    assert BIB_PATH.exists()
    assert len(REGISTRY) >= 10
    # references.bib must declare the canonical registry_key (allowing flexible
    # whitespace around '=' for readable alignment).
    import re

    text = BIB_PATH.read_text(encoding="utf-8")
    for key in ("calzetti2000", "dsps", "tengri"):
        assert key in REGISTRY, f"{key} missing from registry"
        pattern = rf"registry_key\s*=\s*\{{\s*{re.escape(key)}\s*\}}"
        assert re.search(pattern, text), f"{key} not found in .bib"


def test_parser_simple_entry():
    """Minimal @article with standard fields parses cleanly."""
    from tengri.citations._bibparser import parse_bibtex

    src = """
    @article{Calzetti2000,
      author = {Calzetti, D. and Kinney, A. L.},
      title  = {The Dust Content},
      year   = {2000},
      doi    = {10.1086/308692},
    }
    """
    entries = parse_bibtex(src)
    assert len(entries) == 1
    e = entries[0]
    assert e["entry_type"] == "article"
    assert e["bibtex_key"] == "Calzetti2000"
    assert e["author"] == "Calzetti, D. and Kinney, A. L."
    assert e["year"] == "2000"
    assert e["doi"] == "10.1086/308692"


def test_parser_nested_braces():
    """Values with nested braces (common in BibTeX titles) are preserved."""
    from tengri.citations._bibparser import parse_bibtex

    src = "@article{K, title = {Foo {Bar} Baz}, year = 2020}"
    entries = parse_bibtex(src)
    assert entries[0]["title"] == "Foo {Bar} Baz"


def test_parser_quoted_value():
    """Double-quoted values are supported."""
    from tengri.citations._bibparser import parse_bibtex

    src = '@article{K, author = "Doe, J.", year = 2020}'
    entries = parse_bibtex(src)
    assert entries[0]["author"] == "Doe, J."


def test_parser_comments_ignored():
    """Lines starting with ``%`` are skipped."""
    from tengri.citations._bibparser import parse_bibtex

    src = """
    % this is a comment
    @article{K, title = {T}, year = 2020}
    % another comment
    """
    entries = parse_bibtex(src)
    assert len(entries) == 1


def test_parser_custom_fields():
    """Non-standard fields (registry_key, role, upstream_code) flow through."""
    from tengri.citations._bibparser import parse_bibtex

    src = """
    @article{K,
      title = {T}, year = 2020,
      registry_key = {my_key},
      role = {A role},
      upstream_code = {foo/bar},
    }
    """
    entry = parse_bibtex(src)[0]
    assert entry["registry_key"] == "my_key"
    assert entry["role"] == "A role"
    assert entry["upstream_code"] == "foo/bar"


def test_parser_multiple_entries():
    """Multiple entries parse independently."""
    from tengri.citations._bibparser import parse_bibtex

    src = """
    @article{A, title = {T1}, year = 2020}
    @misc{B,   title = {T2}, year = 2021}
    """
    entries = parse_bibtex(src)
    assert [e["bibtex_key"] for e in entries] == ["A", "B"]
    assert entries[1]["entry_type"] == "misc"


def test_reload_refreshes_registry():
    """After reload(), the registry still contains the canonical entries."""
    from tengri.citations.registry import REGISTRY, reload

    keys_before = set(REGISTRY.keys())
    reload()
    assert set(REGISTRY.keys()) == keys_before


def test_entry_roundtrip_to_citation():
    """Every parsed entry maps to a valid Citation with core fields populated."""
    from tengri.citations.registry import REGISTRY

    for key in ("calzetti2000", "dsps", "nifty"):
        c = REGISTRY[key]
        assert c.key == key
        assert c.authors
        assert c.year
        assert c.title
        assert c.short
        assert c.role


def test_malformed_entry_raises():
    """A truncated entry raises ValueError, not a silent mis-parse."""
    from tengri.citations._bibparser import parse_bibtex

    src = "@article{K, author = {unterminated"
    with pytest.raises(ValueError):
        parse_bibtex(src)
