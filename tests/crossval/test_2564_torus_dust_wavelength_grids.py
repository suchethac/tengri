# SPDX-License-Identifier: BSD-3-Clause
"""Regression test #2564: Torus and dust-emission native wavelength grids.

Issue #2564 adds wavelength-grid support declarations for all torus and
dust-emission blocks. This test verifies that the declared native grids exist
and have correct properties (non-empty, sorted, finite).
"""

import numpy as np
import pytest

from tengri.forward.wavelength_extension import (
    native_wave_agn_torus,
    native_wave_dust_emission,
)

pytestmark = pytest.mark.crossval


class TestTorusAgnfitterLineGrids:
    """AGNfitter-lineage torus blocks should have accessible native grids."""

    @pytest.mark.parametrize(
        "block_name",
        [
            "nenkova_agnfitter",
            "nenkova_agnfitter_2p",
            "nenkova_agnfitter_3p",
            "skirtor_agnfitter",
            "skirtor_agnfitter_1p",
            "skirtor_agnfitter_2p",
        ],
    )
    def test_torus_agnfitter_native_grid(self, block_name):
        """Native grid exists and is well-formed."""
        wave = native_wave_agn_torus(block_name)
        assert wave is not None, f"{block_name} should have native wavelength support"
        assert len(wave) > 0, f"{block_name} native grid should not be empty"
        assert np.all(np.isfinite(wave)), f"{block_name} native grid has non-finite values"
        assert np.all(np.diff(wave) > 0), f"{block_name} native grid should be strictly ascending"

    @pytest.mark.parametrize(
        "block_name",
        [
            "fritz",
            "cat3d_wind",
            "cat3d_wind_lowfwd",
        ],
    )
    def test_torus_template_native_grid(self, block_name):
        """Template-based torus blocks should have accessible native grids."""
        wave = native_wave_agn_torus(block_name)
        assert wave is not None, f"{block_name} should have native wavelength support"
        assert len(wave) > 0, f"{block_name} native grid should not be empty"


class TestDustEmissionGrids:
    """Schreiber2018 and dh02_ce01 dust models should use declared grids."""

    @pytest.mark.parametrize("model_name", ["schreiber2018", "dh02_ce01"])
    def test_dust_emission_native_grid(self, model_name):
        """Dust emission model's native grid exists and is well-formed."""
        wave = native_wave_dust_emission(model_name)
        assert wave is not None, f"{model_name} should have native wavelength support"
        assert len(wave) > 0, f"{model_name} native grid should not be empty"
        assert np.all(np.isfinite(wave)), f"{model_name} native grid has non-finite values"
        assert np.all(np.diff(wave) > 0), f"{model_name} native grid should be strictly ascending"
