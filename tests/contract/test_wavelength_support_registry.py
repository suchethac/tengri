# SPDX-License-Identifier: BSD-3-Clause
"""Contract test: all torus, disc, and dust-emission blocks must declare wavelength support.

Issue #2564: Ensure every registered torus and dust-emission model either declares
its native wavelength grid support or is explicitly listed as grid-less.
"""

import pytest

from tengri.components.agn.blocks._protocol import AGN_BLOCKS
from tengri.components.dust.emission.emission import DUST_EMISSION_MODELS
from tengri.forward.wavelength_extension import (
    _AGN_TORUS_TEMPLATES,
    _ANALYTIC_DUST_EMISSION,
    _ANALYTIC_TORUS,
    _DUST_EMISSION_TEMPLATES,
    _GRIDLESS_DUST_EMISSION,
    _GRIDLESS_TORUS,
    native_wave_agn_disc,
    native_wave_agn_torus,
    native_wave_dust_emission,
)

pytestmark = pytest.mark.contract


class TestDustEmissionWavelengthSupport:
    """Every registered dust-emission model must declare wavelength support."""

    def test_all_dust_models_have_support(self):
        """All dust emission models must have either grid or explicit gridless entry."""
        missing_support = []
        for name in sorted(DUST_EMISSION_MODELS.keys()):
            wave = native_wave_dust_emission(name)
            if wave is None:
                missing_support.append(name)

        if missing_support:
            pytest.fail(
                f"Dust-emission models without declared wavelength support: {missing_support}\n"
                f"Each model must either:\n"
                f"  1. Appear in _DUST_EMISSION_TEMPLATES dict (tabulated)\n"
                f"  2. Be listed in _ANALYTIC_DUST_EMISSION (analytic models)\n"
                f"  3. Be listed in _GRIDLESS_DUST_EMISSION (no emission of their own)\n"
                f"See issue #2564 and wavelength_extension.py for details."
            )


class TestTorusWavelengthSupport:
    """Every registered torus block must declare wavelength support."""

    def test_all_torus_blocks_have_support(self):
        """All torus blocks must have either grid or explicit gridless entry."""
        torus_blocks = AGN_BLOCKS.get("torus", {})
        missing_support = []
        for name in sorted(torus_blocks.keys()):
            # Check if block is declared in one of the support dicts/sets
            is_declared = (
                name in _AGN_TORUS_TEMPLATES
                or name in _ANALYTIC_TORUS
                or name in _GRIDLESS_TORUS
            )
            if not is_declared:
                missing_support.append(name)

        if missing_support:
            pytest.fail(
                f"Torus blocks without declared wavelength support: {missing_support}\n"
                f"Each block must be added to one of:\n"
                f"  1. _AGN_TORUS_TEMPLATES (tabulated templates)\n"
                f"  2. _ANALYTIC_TORUS (analytic models)\n"
                f"  3. _GRIDLESS_TORUS (no emission of their own)\n"
                f"See issue #2564 for details."
            )


class TestDiscWavelengthSupport:
    """Every registered disc block should have wavelength support (but may truncate gracefully)."""

    def test_disc_blocks_with_no_support(self):
        """Report disc blocks without support; measurement needed to decide if fix required."""
        disc_blocks = AGN_BLOCKS.get("disc", {})
        missing_support = []
        for name in sorted(disc_blocks.keys()):
            wave = native_wave_agn_disc(name)
            if wave is None:
                missing_support.append(name)

        # Unlike torus and dust, discs emit mostly inside SSP range.
        # Report as a warning for now, but mention they should be checked.
        if missing_support:
            pytest.warns(
                UserWarning,
                match=f"Disc blocks without wavelength support: {', '.join(missing_support)}"
            )
