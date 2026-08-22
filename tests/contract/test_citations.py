# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the citations module."""

import pytest

from tengri.citations import REGISTRY, Citation, cite, cite_all, register
from tengri.citations.registry import format_list

pytestmark = pytest.mark.contract


class TestRegistry:
    """Tests for registry operations."""

    def test_registry_nonempty(self) -> None:
        """Registry must have at least 10 seed entries."""
        assert len(REGISTRY) >= 10

    def test_every_entry_has_required_fields(self) -> None:
        """Every registry entry must have key, short, role, authors, year."""
        for key, citation in REGISTRY.items():
            assert citation.key == key
            assert citation.short, f"{key}: short is empty"
            assert citation.role, f"{key}: role is empty"
            assert citation.authors, f"{key}: authors is empty"
            assert citation.year > 1900, f"{key}: year invalid"
            assert citation.title, f"{key}: title is empty"
            assert citation.bibtex_key, f"{key}: bibtex_key is empty"

    def test_cite_known_key(self) -> None:
        """cite() must return Citation for known keys."""
        citation = cite("calzetti2000")
        assert isinstance(citation, Citation)
        assert citation.key == "calzetti2000"
        assert "Calzetti" in citation.short

    def test_cite_unknown_key_raises_with_suggestions(self) -> None:
        """cite() must raise KeyError with suggestions for unknown keys."""
        with pytest.raises(KeyError) as exc_info:
            cite("nonexistent_key_xyz")
        error_msg = str(exc_info.value)
        assert "nonexistent_key_xyz" in error_msg
        assert "Available keys:" in error_msg
        # Should suggest at least some real keys. The message lists the first
        # few sorted registry keys, so assert against those (robust to new
        # registry additions rather than pinning specific names).
        from tengri.citations.registry import REGISTRY

        assert any(key in error_msg for key in sorted(REGISTRY.keys())[:5])

    def test_cite_all_returns_sorted_list(self) -> None:
        """cite_all() must return all citations sorted by key."""
        all_cites = cite_all()
        assert len(all_cites) >= 10
        keys = [c.key for c in all_cites]
        assert keys == sorted(keys)

    def test_register_duplicate_raises(self) -> None:
        """register() must raise KeyError if key already exists."""
        dup = Citation(
            key="calzetti2000",
            short="Duplicate",
            role="Test",
            authors="Test",
            year=2000,
            title="Test",
            journal="Test",
            doi=None,
            arxiv=None,
            bibtex_key="TestKey",
        )
        with pytest.raises(KeyError) as exc_info:
            register(dup)
        assert "already registered" in str(exc_info.value)

    def test_format_list_short(self) -> None:
        """format_list with fmt='short' produces one-liner per citation."""
        some_cites = [cite("calzetti2000"), cite("dsps")]
        formatted = format_list(some_cites, fmt="short")
        assert len(formatted) > 0
        # Should have multiple lines
        assert formatted.count("\n") == 1  # Two cites = one newline

    def test_format_list_bibtex(self) -> None:
        """format_list with fmt='bibtex' produces BibTeX per citation."""
        some_cites = [cite("calzetti2000")]
        formatted = format_list(some_cites, fmt="bibtex")
        assert "@article" in formatted
        # BibTeX key follows ADS convention (e.g. "Calzetti_2000")
        assert "Calzetti" in formatted

    def test_format_list_invalid_format_raises(self) -> None:
        """format_list() must raise ValueError for unknown format."""
        with pytest.raises(ValueError) as exc_info:
            format_list([cite("calzetti2000")], fmt="invalid")
        assert "Unknown format" in str(exc_info.value)


class TestCitationRecord:
    """Tests for Citation dataclass."""

    def test_citation_str_contains_role_and_short(self) -> None:
        """Citation.__str__() must contain role and short."""
        c = cite("calzetti2000")
        s = str(c)
        assert c.role in s
        assert c.short in s

    def test_citation_str_includes_doi_if_present(self) -> None:
        """Citation.__str__() includes DOI if present."""
        c = cite("calzetti2000")
        s = str(c)
        if c.doi:
            assert "DOI:" in s

    def test_citation_str_includes_arxiv_if_present(self) -> None:
        """Citation.__str__() includes arXiv ID if no DOI."""
        c = cite("dsps")
        s = str(c)
        if c.arxiv and not c.doi:
            assert "arXiv:" in s

    def test_bibtex_roundtrip(self) -> None:
        """to_bibtex() must include bibtex_key and year."""
        c = cite("calzetti2000")
        bibtex = c.to_bibtex()
        assert c.bibtex_key in bibtex
        assert str(c.year) in bibtex
        assert "@article" in bibtex

    def test_citation_frozen(self) -> None:
        """Citation must be frozen (immutable)."""
        from dataclasses import FrozenInstanceError

        c = cite("calzetti2000")
        with pytest.raises(FrozenInstanceError):
            c.key = "modified"

    def test_citation_upstream_code_optional(self) -> None:
        """Citation.upstream_code can be None."""
        c = cite("tengri")
        # tengri itself should have upstream_code=None
        assert c.upstream_code is None

    def test_citation_with_upstream_code(self) -> None:
        """Citations with upstream_code must store it."""
        c = cite("dsps")
        assert c.upstream_code == "ArgonneCPAC/dsps"


class TestSeedEntries:
    """Tests for specific seed entries."""

    def test_calzetti2000_exists(self) -> None:
        """Calzetti2000 dust attenuation seed must exist."""
        c = cite("calzetti2000")
        assert c.year == 2000
        assert "Calzetti" in c.short

    def test_dsps_exists(self) -> None:
        """DSPS seed must exist."""
        c = cite("dsps")
        assert "DSPS" in c.title or "Differentiable" in c.title

    def test_fsps_exists(self) -> None:
        """FSPS seed must exist."""
        c = cite("fsps")
        assert "FSPS" in c.title or "Conroy" in c.authors

    def test_jax_exists(self) -> None:
        """JAX seed must exist."""
        c = cite("jax")
        assert "JAX" in c.title or "Bradbury" in c.short

    def test_prospector_exists(self) -> None:
        """Prospector seed must exist."""
        c = cite("prospector")
        assert "Prospector" in c.title or "Johnson" in c.short


class TestPerFitCitationSurface:
    """The documented per-fit BibTeX surface (README, docs/citation.md).

    Regression for the fresh-user audit (2026-07): README and docs taught
    ``print(tengri.cite_all(result))``, but ``cite_all()`` takes NO argument
    and returns the whole registry — the per-fit emitter is
    ``print_components_bibtex(result)`` / ``cite_components(result)``. This
    guard keeps the taught call working and the semantics honest.
    """

    def test_cite_all_is_registry_wide_not_per_fit(self) -> None:
        # Passing a model/result (the old doc bug) must not silently "work".
        with pytest.raises(TypeError):
            cite_all(object())

    def test_print_components_bibtex_emits_for_a_model(self, synthetic_ssp_wide, capsys) -> None:
        import tengri
        from tengri import FREE, Fixed, SEDModel

        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "dpl", "all_params": FREE},
            neb={"type": "none"},
            redshift=Fixed(0.1),
        )
        tengri.print_components_bibtex(model)
        out = capsys.readouterr().out
        assert "@" in out, "print_components_bibtex emitted no BibTeX entries"
