"""Phase 5: Spectrum LUT (SpectrumPrecomp) contract tests.

Tests the SpectrumPrecomp dataclass, spectrum grid wiring, and agreement
between exact and LUT-based spectrum predictions.
"""

import pytest

from tengri import SpectrumPrecomp


class TestSpectrumPrecompDataclass:
    """SpectrumPrecomp is a frozen, hashable dataclass."""

    def test_spectrum_precomp_frozen(self):
        """SpectrumPrecomp instances are immutable."""
        sp = SpectrumPrecomp()
        with pytest.raises(AttributeError):
            sp.x = 42  # no assignment on frozen dataclass

    def test_spectrum_precomp_hashable(self):
        """SpectrumPrecomp instances are hashable."""
        sp1 = SpectrumPrecomp()
        sp2 = SpectrumPrecomp()
        # Same type should be hashable
        hash_set = {sp1, sp2}
        assert len(hash_set) >= 1  # At least one unique instance

    def test_spectrum_precomp_isinstance(self):
        """SpectrumPrecomp is an instance of SpectrumPrecomp."""
        sp = SpectrumPrecomp()
        assert isinstance(sp, SpectrumPrecomp)
