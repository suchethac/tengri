# SPDX-License-Identifier: BSD-3-Clause
"""Tests for LineList emission line registry."""

from __future__ import annotations

import chex
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.bounds

from tengri.observation.line_list import LineList


class TestLineListDefault13:
    def test_has_correct_count(self):
        cat = LineList.default_13()
        assert cat.n_lines == 13

    def test_has_halpha(self):
        cat = LineList.default_13()
        assert "Halpha" in cat.names

    def test_has_hbeta(self):
        cat = LineList.default_13()
        assert "Hbeta" in cat.names

    def test_wavelengths_shape(self):
        cat = LineList.default_13()
        chex.assert_shape(cat.wavelengths, (13,))


class TestLineListDefaultOptical:
    def test_count_in_range(self):
        cat = LineList.default_optical()
        assert 35 <= cat.n_lines <= 45

    def test_has_bpt_lines(self):
        cat = LineList.default_optical()
        names = set(cat.names)
        bpt_lines = [
            "Halpha",
            "Hbeta",
            "OIII_5007",
            "NII_6584",
            "SII_6717",
            "SII_6731",
            "OII_3726",
        ]
        for line in bpt_lines:
            assert line in names, f"Missing BPT-critical line: {line}"

    def test_wavelengths_sorted(self):
        cat = LineList.default_optical()
        assert jnp.all(jnp.diff(cat.wavelengths) > 0), "Wavelengths must be sorted"

    def test_has_oiii_doublet(self):
        cat = LineList.default_optical()
        oiii_doublets = [d for d in cat.doublets if cat.names[d.primary_idx] == "OIII_5007"]
        assert len(oiii_doublets) == 1
        assert abs(oiii_doublets[0].ratio - 2.98) < 0.01

    def test_balmer_flags(self):
        cat = LineList.default_optical()
        ha_idx = list(cat.names).index("Halpha")
        hb_idx = list(cat.names).index("Hbeta")
        oiii_idx = list(cat.names).index("OIII_5007")
        assert cat.is_balmer[ha_idx]
        assert cat.is_balmer[hb_idx]
        assert not cat.is_balmer[oiii_idx]


class TestConstraintMatrix:
    def test_shape(self):
        cat = LineList.default_optical()
        C = cat.build_constraint_matrix()
        chex.assert_shape(C, (cat.n_lines, cat.n_independent))
        assert cat.n_independent == cat.n_lines - len(cat.doublets)

    def test_oiii_ratio_encoded(self):
        cat = LineList.default_optical()
        C = cat.build_constraint_matrix()
        i_5007 = list(cat.names).index("OIII_5007")
        i_4959 = list(cat.names).index("OIII_4959")
        # Find which column corresponds to OIII_5007's independent amplitude
        primary_col = int(jnp.argmax(C[i_5007, :]))
        assert abs(float(C[i_5007, primary_col]) - 1.0) < 1e-5
        assert abs(float(C[i_4959, primary_col]) - 1.0 / 2.98) < 1e-4

    def test_identity_for_no_doublets(self):
        """Single-line catalog has identity constraint matrix."""
        # Build a minimal catalog with just one line (no doublets)
        cat = LineList(
            names=("Halpha",),
            wavelengths=jnp.array([6562.80]),
            species=("H1",),
            doublets=(),
            is_balmer=(True,),
            is_broad_candidate=(True,),
            is_strong=(True,),
            plot_group=("halpha_nii_6548_48",),
        )
        C = cat.build_constraint_matrix()
        chex.assert_shape(C, (1, 1))
        assert abs(float(C[0, 0]) - 1.0) < 1e-6


class TestSelect:
    def test_wavelength_range(self):
        cat = LineList.default_optical()
        sub = cat.select(wave_min=4000.0, wave_max=7000.0)
        assert all(4000.0 <= float(w) <= 7000.0 for w in sub.wavelengths)
        assert sub.n_lines < cat.n_lines

    def test_names_filter(self):
        cat = LineList.default_optical()
        bpt = cat.select(names=["Halpha", "Hbeta", "OIII_5007", "NII_6584"])
        assert set(bpt.names) == {"Halpha", "Hbeta", "OIII_5007", "NII_6584"}
        assert bpt.n_lines == 4

    def test_wavelengths_filter(self):
        cat = LineList.default_optical()
        sub = cat.select(wavelengths=[6562.80, 4861.33])
        # Should find Halpha and Hbeta
        names = set(sub.names)
        assert "Halpha" in names
        assert "Hbeta" in names

    def test_unknown_name_raises(self):
        cat = LineList.default_optical()
        with pytest.raises(ValueError, match="not found"):
            cat.select(names=["NotALine"])

    def test_doublets_preserved_when_both_selected(self):
        cat = LineList.default_optical()
        # Select both OIII lines — doublet should be preserved
        sub = cat.select(names=["OIII_5007", "OIII_4959"])
        assert len(sub.doublets) == 1

    def test_doublets_dropped_when_one_filtered(self):
        cat = LineList.default_optical()
        # Select only OIII_5007, not OIII_4959 — doublet should be dropped
        sub = cat.select(names=["Halpha", "Hbeta", "OIII_5007"])
        oiii_doublets = [d for d in sub.doublets if sub.names[d.primary_idx] == "OIII_5007"]
        assert len(oiii_doublets) == 0  # OIII_4959 was filtered out
