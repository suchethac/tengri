# SPDX-License-Identifier: BSD-3-Clause
"""Integration tests for synthesizer_default preset.

Tests that the preset builds without data, JIT-compiles, propagates gradients,
and correctly lists/describes itself in the registry.
"""

from __future__ import annotations

import re

import pytest

from tengri.presets import (
    describe_preset,
    list_presets,
    synthesizer_default,
)


class TestPresetBuildsWithoutData:
    """Test that the preset builds to (SEDModelConfig, Parameters)."""

    def test_preset_builds_basic(self):
        """Construct the preset with minimal arguments."""
        config, params = synthesizer_default()
        assert config is not None
        assert params is not None
        assert hasattr(config, "sfh")
        assert hasattr(params, "free_params")

    def test_preset_builds_with_redshift(self):
        """Construct the preset with custom redshift."""
        config, params = synthesizer_default(redshift=2.5)
        assert config is not None
        assert params is not None
        # Verify redshift is in the parameters
        assert "redshift" in params.free_params or "redshift" in params.fixed_params

    def test_preset_returns_tuple(self):
        """Verify return type is (config, params)."""
        result = synthesizer_default()
        assert isinstance(result, tuple)
        assert len(result) == 2
        config, params = result
        assert hasattr(config, "agn_model")
        assert hasattr(params, "sample")


# Two classes stood here -- TestPresetJITCompilation and TestPresetGradients --
# each holding one test whose body was `pass`, under
# `@pytest.mark.skipif(True, reason="... require SSP data ...")`. Doubly empty:
# nothing to run, and skipped so it would not run anyway. Their names claimed
# JIT and gradient coverage of this preset that has never existed in any form,
# and `skipif(True)` is `skip` spelled so that it reads as conditional.
#
# Removed rather than filled in: writing them means loading an SSP grid, which
# is new coverage and belongs in its own change. `tests/physics/gradients/` is
# where the real JIT and gradient safety tests live.


class TestPresetRegistry:
    """Test list_presets() and describe_preset()."""

    def test_list_presets_includes_synthesizer_default(self):
        """Check that synthesizer_default is registered.

        ``list_presets`` returned ``dict[str, dict]`` until #1574/#1592 made every
        discovery verb return a ``_RegistryTable`` — a ``list`` of row dicts — so
        the whole surface answers one type. This test kept the old shape because
        ``tests/integration`` is a gated tier that had not run since (#1648); it is
        the documented migration that is asserted here, not a new contract.
        """
        presets = list_presets()
        # Pin the unified TYPE as well as the contents: a table that stopped
        # being a list would still satisfy the membership checks below.
        assert isinstance(presets, list)
        assert "synthesizer_default" in presets.names()
        # The docstring's own recipe for the old name-to-metadata mapping.
        by_name = {row["name"]: row for row in presets}
        entry = by_name["synthesizer_default"]
        assert "short_doc" in entry
        assert "citations" in entry
        assert "status" in entry

    def test_describe_preset_returns_full_metadata(self):
        """Describe should include docstring, citations, status."""
        desc = describe_preset("synthesizer_default")
        assert isinstance(desc, dict)
        assert desc["name"] == "synthesizer_default"
        assert "short_doc" in desc
        assert "citations" in desc
        assert "status" in desc
        assert "description" in desc
        assert len(desc["description"]) > 0, "Docstring should be non-empty"

    def test_describe_raises_on_unknown_preset(self):
        """Describe should raise KeyError on nonexistent preset."""
        with pytest.raises(KeyError, match="Unknown preset"):
            describe_preset("nonexistent_model")


class TestPresetCitations:
    """Test that citations are valid and declared."""

    def test_preset_citations_have_expected_count(self):
        """Verify synthesizer_default has at least 8 bibkeys."""
        desc = describe_preset("synthesizer_default")
        citations = desc["citations"]
        assert isinstance(citations, list)
        assert len(citations) >= 8, f"Expected 8+ citations, got {len(citations)}"

    def test_citations_are_strings(self):
        """All citations should be non-empty strings."""
        desc = describe_preset("synthesizer_default")
        for cit in desc["citations"]:
            assert isinstance(cit, str)
            assert len(cit) > 0

    def test_preset_citations_in_manifest(self):
        """Each bibkey should exist in synthesizer_parity_citations.md.

        This test prevents future presets from hallucinating citations.
        """
        import os

        # Locate the citations manifest
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        manifest_path = os.path.join(repo_root, "docs", "dev", "synthesizer_parity_citations.md")

        if not os.path.exists(manifest_path):
            pytest.skip(f"Manifest not found at {manifest_path}")

        with open(manifest_path) as f:
            manifest_text = f.read()

        # Extract bibkeys from the manifest using regex
        # Pattern: `bibkey` in markdown code or | bibkey | in tables
        bibkey_pattern = r"(?:`|^\| )([A-Za-z0-9_]+)(?:`|\s*\|)"
        manifest_bibkeys = set(re.findall(bibkey_pattern, manifest_text, re.MULTILINE))

        desc = describe_preset("synthesizer_default")
        citations = desc["citations"]

        for cit in citations:
            assert cit in manifest_bibkeys, (
                f"Citation '{cit}' not found in {manifest_path}. "
                f"Add it to the manifest before updating presets."
            )


class TestSmokeCheckCLI:
    """Smoke test: verify the preset can be imported and called from CLI."""

    def test_import_and_call(self):
        """CLI-equivalent: import, build, check structure."""
        from tengri.presets import describe_preset, list_presets, synthesizer_default

        # Build config and params
        config, params = synthesizer_default()

        # List all presets. ``.names()`` is what replaced ``list(presets)`` when
        # every discovery verb moved to a table return (#1574/#1592/#1648).
        presets = list_presets()
        preset_names = presets.names()

        # Describe the preset
        desc = describe_preset("synthesizer_default")
        citations = desc["citations"]

        # Simple checks
        assert "synthesizer_default" in preset_names
        assert len(citations) > 0
        assert config is not None
        assert params is not None


# ──────────────────────────────────────────────────────────────────
# Optional: parametric-only test without SSP data
# ──────────────────────────────────────────────────────────────────


class TestPresetMetadataOnly:
    """Tests that don't require SSP/Cue data."""

    def test_preset_function_has_docstring(self):
        """The preset callable should have a proper docstring."""
        from tengri.presets.synthesizer import synthesizer_default as factory

        assert factory.__doc__ is not None
        assert "synthesizer" in factory.__doc__.lower()
        assert "default" in factory.__doc__.lower()

    def test_preset_config_is_frozen(self):
        """Verify config dataclasses are frozen (immutable)."""
        config, _ = synthesizer_default()
        # Attempt to mutate should raise FrozenInstanceError
        with pytest.raises((AttributeError, TypeError)):
            config.dust = None  # type: ignore
