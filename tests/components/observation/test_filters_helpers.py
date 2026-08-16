# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for filter discovery helpers (tengri.observation.filters namespace).

Frozen: list_filter_aliases() sorted and non-empty; instrument filtering
case-insensitive; load() raises on unknown filters; describe() includes
wavelength info; suggest() returns sorted by λ_eff, raises on unknown coverage,
accepts documented presets.

Called ``list_filters`` until #1574, when it was renamed to say which question
it answers — ``tengri.list_filters`` lists SVO curve-file stems, this lists the
short aliases the loaders take. The old name is a deprecated alias.
"""

import numpy as np
import pytest

from tengri.observation.filters import (
    compute_effective_wavelength,
    describe,
    list_filter_aliases,
    load,
    suggest,
)

pytestmark = pytest.mark.bounds


class TestListFilterAliases:
    """list_filter_aliases: sorted, non-empty, instrument filtering."""

    def test_list_filter_aliases_nonempty_and_sorted(self):
        """list_filter_aliases().names() returns a non-empty sorted list."""
        names = list_filter_aliases().names()
        if not names:
            pytest.skip("Filter library is empty in this environment")
        assert len(names) > 0
        assert names == sorted(names)
        assert all(isinstance(name, str) for name in names)

    def test_list_filter_aliases_instrument_filter_case_insensitive(self):
        """Instrument filter is case-insensitive: sdss == SDSS."""
        result = list_filter_aliases(instrument="sdss").names()
        result_upper = list_filter_aliases(instrument="SDSS").names()
        assert result == result_upper

    def test_list_filter_aliases_instrument_filter_sdss(self):
        """Instrument='sdss' returns SDSS filters (sdss_*) only."""
        names = list_filter_aliases(instrument="sdss").names()
        if not names:
            pytest.skip("No SDSS filters in registry")
        assert all("sdss" in name.lower() for name in names)

    def test_list_filter_aliases_instrument_no_matches(self):
        """Unknown instrument returns an empty table."""
        result = list_filter_aliases(instrument="nonexistent_instrument_xyz")
        assert result.names() == []


class TestLoad:
    """load: raises on unknown filters."""

    def test_load_unknown_filter_raises(self):
        """load() raises KeyError for unknown filter."""
        with pytest.raises(KeyError):
            load(["nonexistent_filter_xyz"])


class TestDescribe:
    """describe: raises on unknown, includes wavelength info."""

    def test_describe_raises_on_unknown(self):
        """Unknown filter raises, like its sibling ``load`` (#1611).

        This used to assert the opposite — that an unknown name returned a
        fallback string. The string was ``"<name>: (filter found; no summary
        available)"``, which states the filter *was* found, and made an unknown
        name indistinguishable from a curve that failed to load. ``load()``
        directly above already raises ``KeyError`` for the same input, so the
        fallback was the odd one out rather than a policy.
        """
        with pytest.raises(KeyError):
            describe("nonexistent_filter_xyz")

    def test_describe_includes_wavelength_info(self):
        """describe() includes wavelength information (λ_eff or range)."""
        result = describe("sdss_r")
        # Should mention wavelength or effective wavelength in some form
        assert any(
            term in result.lower()
            for term in ["λ_eff", "lambda", "angstrom", "å", "μm", "micron", "range"]
        )


class TestSuggest:
    """suggest: sorted by λ_eff, rejects unknown coverage, accepts presets."""

    def test_suggest_returns_sorted_by_wavelength(self):
        """suggest() returns filters sorted by effective wavelength."""
        result = suggest(redshift=0.0, coverage="visible_to_nir")
        if len(result) < 2:
            pytest.skip("Not enough filters to check sorting")

        # Load every returned filter and verify they are sorted by lambda_eff.
        #
        # A failed load used to be swallowed by `except Exception: pass` and the
        # filter dropped. Dropping elements can turn an unsorted sequence into a
        # sorted one, so a load failure could mask the very ordering bug this
        # test exists to catch — and if every load failed, `wavelengths` was
        # empty and the guarded assertion below never ran at all.
        wavelengths = []
        failed = []
        for name in result:
            try:
                fc = load([name])[2][0]
                wave_np = np.asarray(fc.wave)
                trans_np = np.asarray(fc.trans)
                wavelengths.append(compute_effective_wavelength(wave_np, trans_np))
            except Exception as exc:
                failed.append(f"{name}: {type(exc).__name__}")

        assert not failed, f"suggest() returned filters that will not load: {failed}"
        assert len(wavelengths) == len(result), "a filter was dropped before the sort check"
        assert wavelengths == sorted(wavelengths), (
            f"suggest() is not sorted by effective wavelength: {wavelengths}"
        )

    def test_suggest_unknown_coverage_raises(self):
        """Unknown coverage preset raises ValueError."""
        with pytest.raises(ValueError):
            suggest(redshift=0.0, coverage="unknown_coverage_xyz")

    def test_suggest_accepts_documented_presets(self):
        """All documented coverage presets are accepted (visible, visible_to_nir, etc)."""
        presets = ["visible", "visible_to_nir", "uv_to_ir", "jwst_cover"]
        for preset in presets:
            # Should not raise
            result = suggest(redshift=0.0, coverage=preset)
            assert isinstance(result, list)
