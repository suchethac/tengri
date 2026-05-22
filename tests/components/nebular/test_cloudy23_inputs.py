# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for Cloudy 23 input-deck generator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular.cloudy23_inputs import (
    build_cloudy23_deck,
)

pytestmark = pytest.mark.bounds


class TestCloudy23DeckRender:
    """Test Cloudy23Deck.render() output."""

    def test_minimal_deck_renders(self):
        """Verify minimal deck produces sensible Cloudy 23 syntax."""
        deck = build_cloudy23_deck(
            log_u=-3.0,
            log_n_h=2.0,
            log_z_gas=-0.3,
            sed_keyword="blackbody, T=40000 K",
        )
        rendered = deck.render()

        # Check key lines are present
        assert "title" in rendered
        assert "ionizing source table blackbody, T=40000 K" in rendered
        assert "ionization parameter -3.0" in rendered
        assert "hden 2.0" in rendered
        assert "iterate 30 times" in rendered
        assert "stop temperature 100.0 K" in rendered
        assert "metals" in rendered
        assert "grains ism" in rendered
        assert rendered.endswith("\n")

    def test_custom_iterations_and_stop_temp(self):
        """Verify custom iteration and stop-temperature values appear."""
        deck = build_cloudy23_deck(
            log_u=-2.0,
            log_n_h=1.0,
            log_z_gas=0.0,
            sed_keyword="blackbody, T=50000 K",
            cloudy_iterations=50,
            stop_temperature_k=500.0,
        )
        rendered = deck.render()

        assert "iterate 50 times" in rendered
        assert "stop temperature 500.0 K" in rendered

    def test_save_lines_and_continuum_commands(self):
        """Verify save lines and save continuum commands appear."""
        deck = build_cloudy23_deck(
            log_u=-3.0,
            log_n_h=2.0,
            log_z_gas=-0.3,
            sed_keyword="blackbody, T=40000 K",
            save_lines_path="lines.txt",
            save_continuum_path="continuum.txt",
        )
        rendered = deck.render()

        assert "save lines column lines.txt" in rendered
        assert "save continuum continuum.txt" in rendered

    def test_abundance_set_gass10(self):
        """Verify gass10 (default) produces minimal abundance commands."""
        deck = build_cloudy23_deck(
            log_u=-3.0,
            log_n_h=2.0,
            log_z_gas=-0.3,
            sed_keyword="blackbody, T=40000 K",
            abundance_set="gass10",
        )
        rendered = deck.render()

        # gass10 is Cloudy default; no explicit "set abundances" command
        assert "set abundances" not in rendered
        assert "metals" in rendered

    def test_abundance_set_ism(self):
        """Verify ism abundance set produces correct command."""
        deck = build_cloudy23_deck(
            log_u=-3.0,
            log_n_h=2.0,
            log_z_gas=-0.3,
            sed_keyword="blackbody, T=40000 K",
            abundance_set="ism",
        )
        rendered = deck.render()

        assert "set abundances ism" in rendered
        assert "metals" in rendered

    def test_abundance_set_h_ii(self):
        """Verify h_ii abundance set produces correct command."""
        deck = build_cloudy23_deck(
            log_u=-3.0,
            log_n_h=2.0,
            log_z_gas=-0.3,
            sed_keyword="blackbody, T=40000 K",
            abundance_set="h_ii",
        )
        rendered = deck.render()

        assert "set abundances H II regions" in rendered
        assert "metals" in rendered

    def test_extra_commands_appended(self):
        """Verify extra_commands are appended at the end."""
        extra = ("set nFnu = 4", "print lines")
        deck = build_cloudy23_deck(
            log_u=-3.0,
            log_n_h=2.0,
            log_z_gas=-0.3,
            sed_keyword="blackbody, T=40000 K",
            extra_commands=extra,
        )
        rendered = deck.render()

        for cmd in extra:
            assert cmd in rendered

    def test_no_grains_when_grain_set_none(self):
        """Verify grains command is omitted when grain_set=None."""
        deck = build_cloudy23_deck(
            log_u=-3.0,
            log_n_h=2.0,
            log_z_gas=-0.3,
            sed_keyword="blackbody, T=40000 K",
            grain_set=None,
        )
        rendered = deck.render()

        assert "grains" not in rendered

    def test_grain_set_agn(self):
        """Verify agn grain set produces correct command."""
        deck = build_cloudy23_deck(
            log_u=-3.0,
            log_n_h=2.0,
            log_z_gas=-0.3,
            sed_keyword="blackbody, T=40000 K",
            grain_set="agn",
        )
        rendered = deck.render()

        assert "grains agn" in rendered


class TestValidation:
    """Test input validation."""

    def test_logu_too_low_raises(self):
        """Verify log_u < -5 raises ValueError."""
        with pytest.raises(ValueError, match="log_u"):
            build_cloudy23_deck(
                log_u=-10.0,
                log_n_h=2.0,
                log_z_gas=-0.3,
                sed_keyword="blackbody, T=40000 K",
            )

    def test_logu_too_high_raises(self):
        """Verify log_u > 0 raises ValueError."""
        with pytest.raises(ValueError, match="log_u"):
            build_cloudy23_deck(
                log_u=1.0,
                log_n_h=2.0,
                log_z_gas=-0.3,
                sed_keyword="blackbody, T=40000 K",
            )

    def test_lognh_too_low_raises(self):
        """Verify log_n_h < -2 raises ValueError."""
        with pytest.raises(ValueError, match="log_n_h"):
            build_cloudy23_deck(
                log_u=-3.0,
                log_n_h=-10.0,
                log_z_gas=-0.3,
                sed_keyword="blackbody, T=40000 K",
            )

    def test_lognh_too_high_raises(self):
        """Verify log_n_h > 6 raises ValueError."""
        with pytest.raises(ValueError, match="log_n_h"):
            build_cloudy23_deck(
                log_u=-3.0,
                log_n_h=10.0,
                log_z_gas=-0.3,
                sed_keyword="blackbody, T=40000 K",
            )

    def test_logz_too_low_raises(self):
        """Verify log_z_gas < -4 raises ValueError."""
        with pytest.raises(ValueError, match="log_z_gas"):
            build_cloudy23_deck(
                log_u=-3.0,
                log_n_h=2.0,
                log_z_gas=-10.0,
                sed_keyword="blackbody, T=40000 K",
            )

    def test_logz_too_high_raises(self):
        """Verify log_z_gas > 1 raises ValueError."""
        with pytest.raises(ValueError, match="log_z_gas"):
            build_cloudy23_deck(
                log_u=-3.0,
                log_n_h=2.0,
                log_z_gas=2.0,
                sed_keyword="blackbody, T=40000 K",
            )

    def test_iterations_less_than_1_raises(self):
        """Verify cloudy_iterations < 1 raises ValueError."""
        with pytest.raises(ValueError, match="cloudy_iterations"):
            build_cloudy23_deck(
                log_u=-3.0,
                log_n_h=2.0,
                log_z_gas=-0.3,
                sed_keyword="blackbody, T=40000 K",
                cloudy_iterations=0,
            )

    def test_stop_temperature_negative_raises(self):
        """Verify stop_temperature_k <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="stop_temperature_k"):
            build_cloudy23_deck(
                log_u=-3.0,
                log_n_h=2.0,
                log_z_gas=-0.3,
                sed_keyword="blackbody, T=40000 K",
                stop_temperature_k=-100.0,
            )

    def test_no_sed_source_raises(self):
        """Verify that providing neither sed_table nor sed_keyword raises."""
        with pytest.raises(ValueError, match="Either sed_table or sed_keyword"):
            build_cloudy23_deck(
                log_u=-3.0,
                log_n_h=2.0,
                log_z_gas=-0.3,
            )

    def test_unknown_abundance_set_raises(self):
        """Verify unknown abundance_set raises ValueError."""
        with pytest.raises(ValueError, match="abundance_set"):
            build_cloudy23_deck(
                log_u=-3.0,
                log_n_h=2.0,
                log_z_gas=-0.3,
                sed_keyword="blackbody, T=40000 K",
                abundance_set="unknown",
            )


class TestTabulatedSED:
    """Test tabulated ionizing spectrum."""

    def test_tabulated_sed_in_command(self):
        """Verify tabulated SED produces correct Cloudy command."""
        wave_aa = jnp.array([100.0, 200.0, 300.0])
        j_lambda = jnp.array([1.0, 2.0, 1.5])
        sed_table = {"wave_aa": wave_aa, "j_lambda": j_lambda}

        deck = build_cloudy23_deck(
            log_u=-3.0,
            log_n_h=2.0,
            log_z_gas=-0.3,
            sed_table=sed_table,
        )
        rendered = deck.render()

        # Should reference the .sed sidecar file
        assert f'table SED "{deck.prefix}.sed"' in rendered
        # sed_keyword should not appear
        assert "blackbody" not in rendered

    def test_tabulated_sed_sidecar_written(self, tmp_path):
        """Verify write() produces both .in and .sed files."""
        wave_aa = jnp.array([100.0, 200.0, 300.0])
        j_lambda = jnp.array([1.0, 2.0, 1.5])
        sed_table = {"wave_aa": wave_aa, "j_lambda": j_lambda}

        deck = build_cloudy23_deck(
            log_u=-3.0,
            log_n_h=2.0,
            log_z_gas=-0.3,
            sed_table=sed_table,
        )
        in_file = deck.write(tmp_path)

        # Check .in file exists
        assert in_file.exists()

        # Check .sed file exists
        sed_file = tmp_path / f"{deck.prefix}.sed"
        assert sed_file.exists()

        # Load and verify .sed file contents
        sed_data = np.loadtxt(sed_file)
        np.testing.assert_allclose(sed_data[:, 0], np.array(wave_aa))
        np.testing.assert_allclose(sed_data[:, 1], np.array(j_lambda))


class TestWrite:
    """Test Cloudy23Deck.write() method."""

    def test_write_creates_directory(self):
        """Verify write() creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "path"
            assert not nested_path.exists()

            deck = build_cloudy23_deck(
                log_u=-3.0,
                log_n_h=2.0,
                log_z_gas=-0.3,
                sed_keyword="blackbody, T=40000 K",
            )
            in_file = deck.write(nested_path)

            assert nested_path.exists()
            assert in_file.exists()

    def test_write_returns_absolute_path(self, tmp_path):
        """Verify write() returns absolute path to .in file."""
        deck = build_cloudy23_deck(
            log_u=-3.0,
            log_n_h=2.0,
            log_z_gas=-0.3,
            sed_keyword="blackbody, T=40000 K",
        )
        in_file = deck.write(tmp_path)

        assert in_file.is_absolute()
        assert in_file.name.endswith(".in")

    def test_round_trip_write_then_read(self, tmp_path):
        """Verify round-trip: write deck, read back, rendering matches."""
        deck = build_cloudy23_deck(
            log_u=-3.0,
            log_n_h=2.0,
            log_z_gas=-0.3,
            sed_keyword="blackbody, T=40000 K",
            save_lines_path="lines.txt",
            abundance_set="ism",
        )
        rendered_before = deck.render()
        in_file = deck.write(tmp_path)

        # Read back
        rendered_after = in_file.read_text()

        assert rendered_before == rendered_after


class TestIntegration:
    """Integration tests."""

    def test_representative_realistic_deck(self, tmp_path):
        """Verify a realistic (logU, n_H, Z) point deck is sensible."""
        deck = build_cloudy23_deck(
            log_u=-3.5,
            log_n_h=2.5,
            log_z_gas=-0.5,
            sed_keyword="blackbody, T=45000 K",
            cloudy_iterations=40,
            stop_temperature_k=150.0,
            save_lines_path="emission_lines.txt",
            save_continuum_path="continuum_flux.txt",
            abundance_set="h_ii",
            grain_set="ism",
            extra_commands=("print lines all", "print continuum"),
        )

        # Write and verify file
        in_file = deck.write(tmp_path)
        content = in_file.read_text()

        # Spot-check key elements
        assert "logU-3.5" in deck.prefix
        assert "logn2.5" in deck.prefix
        assert "logZ-0.5" in deck.prefix
        assert "ionizing source table blackbody, T=45000 K" in content
        assert "ionization parameter -3.5" in content
        assert "hden 2.5" in content
        assert "iterate 40 times" in content
        assert "stop temperature 150.0 K" in content
        assert "save lines column emission_lines.txt" in content
        assert "save continuum continuum_flux.txt" in content
        assert "set abundances H II regions" in content
        assert "grains ism" in content
        assert "print lines all" in content
        assert "print continuum" in content
