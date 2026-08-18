# SPDX-License-Identifier: BSD-3-Clause
"""NSS preset expansion: fast vs. accurate evidence settings."""

import pytest

from tengri.inference.backends import evidence


class TestResolveNssSettings:
    """Tests for _resolve_nss_settings: preset-based parameter expansion."""

    def test_resolve_nss_all_none_uses_accurate_defaults(self):
        """All-None arguments yield the accurate (default) preset values."""
        n_live, num_delete, log_evidence_tol, max_shrinkage = evidence._resolve_nss_settings(
            preset=None, n_live=None, num_delete=None, log_evidence_tol=None, max_shrinkage=None
        )
        assert n_live == 500
        assert num_delete == 50
        assert log_evidence_tol == -3.0
        assert max_shrinkage == 20

    def test_resolve_nss_preset_fast(self):
        """preset='fast' yields the fast-tier values."""
        n_live, num_delete, log_evidence_tol, max_shrinkage = evidence._resolve_nss_settings(
            preset="fast", n_live=None, num_delete=None, log_evidence_tol=None, max_shrinkage=None
        )
        assert n_live == 100
        assert num_delete == 20
        assert log_evidence_tol == -2.0
        assert max_shrinkage == 10

    def test_resolve_nss_preset_accurate(self):
        """preset='accurate' yields the accurate-tier values."""
        n_live, num_delete, log_evidence_tol, max_shrinkage = evidence._resolve_nss_settings(
            preset="accurate",
            n_live=None,
            num_delete=None,
            log_evidence_tol=None,
            max_shrinkage=None,
        )
        assert n_live == 500
        assert num_delete == 50
        assert log_evidence_tol == -3.0
        assert max_shrinkage == 20

    def test_resolve_nss_none_preset_same_as_accurate(self):
        """preset=None is identical to preset='accurate'."""
        result_none = evidence._resolve_nss_settings(
            preset=None, n_live=None, num_delete=None, log_evidence_tol=None, max_shrinkage=None
        )
        result_accurate = evidence._resolve_nss_settings(
            preset="accurate",
            n_live=None,
            num_delete=None,
            log_evidence_tol=None,
            max_shrinkage=None,
        )
        assert result_none == result_accurate

    def test_resolve_nss_explicit_overrides_preset(self):
        """Explicit non-None arguments override the preset."""
        n_live, num_delete, log_evidence_tol, max_shrinkage = evidence._resolve_nss_settings(
            preset="fast", n_live=300, num_delete=None, log_evidence_tol=None, max_shrinkage=None
        )
        assert n_live == 300
        assert num_delete == 20
        assert log_evidence_tol == -2.0
        assert max_shrinkage == 10

    def test_resolve_nss_multiple_overrides(self):
        """Multiple explicit arguments override the preset independently."""
        n_live, num_delete, log_evidence_tol, max_shrinkage = evidence._resolve_nss_settings(
            preset="accurate",
            n_live=200,
            num_delete=30,
            log_evidence_tol=-2.5,
            max_shrinkage=None,
        )
        assert n_live == 200
        assert num_delete == 30
        assert log_evidence_tol == -2.5
        assert max_shrinkage == 20

    def test_resolve_nss_all_explicit_overrides_preset(self):
        """All explicit arguments override the preset completely."""
        n_live, num_delete, log_evidence_tol, max_shrinkage = evidence._resolve_nss_settings(
            preset="fast", n_live=250, num_delete=40, log_evidence_tol=-2.8, max_shrinkage=15
        )
        assert n_live == 250
        assert num_delete == 40
        assert log_evidence_tol == -2.8
        assert max_shrinkage == 15

    def test_resolve_nss_unknown_preset_raises(self):
        """ValueError for unknown preset name."""
        with pytest.raises(ValueError, match=r"(fast|accurate|unknown|preset)"):
            evidence._resolve_nss_settings(
                preset="bogus",
                n_live=None,
                num_delete=None,
                log_evidence_tol=None,
                max_shrinkage=None,
            )

    def test_resolve_nss_preset_fast_partial_override(self):
        """Fast preset with one override."""
        n_live, num_delete, log_evidence_tol, max_shrinkage = evidence._resolve_nss_settings(
            preset="fast", n_live=None, num_delete=25, log_evidence_tol=None, max_shrinkage=None
        )
        assert n_live == 100
        assert num_delete == 25
        assert log_evidence_tol == -2.0
        assert max_shrinkage == 10

    def test_resolve_nss_none_preset_all_override(self):
        """preset=None (defaults to accurate) with all values overridden."""
        n_live, num_delete, log_evidence_tol, max_shrinkage = evidence._resolve_nss_settings(
            preset=None, n_live=150, num_delete=35, log_evidence_tol=-2.2, max_shrinkage=12
        )
        assert n_live == 150
        assert num_delete == 35
        assert log_evidence_tol == -2.2
        assert max_shrinkage == 12
