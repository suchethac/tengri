# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the GRAHSP Netzer accretion-disc template."""

from __future__ import annotations

from pathlib import Path

import chex
import numpy as np
import pytest

from tests._bounds import assert_non_negative

pytestmark = pytest.mark.bounds

FIXTURE = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "grahsp" / "netzer_disc.npz"


@pytest.fixture(scope="module")
def fixture():
    return np.load(FIXTURE, allow_pickle=True)


def test_fixture_shapes(fixture):
    assert fixture["disc_spectra"].shape[0] == 8
    assert fixture["disc_spectra"].shape[1] == fixture["wave_disc_nm"].size
    assert fixture["wave_disc_nm"].size == 1950


def test_netzer_disc_matches_upstream(fixture):
    """Test that netzer_disc matches l5100 * disc_lumin on native grid."""
    from tengri.components.agn.grahsp.disc import netzer_disc
    from tengri.components.agn.grahsp.templates import load_grahsp_templates

    templates = load_grahsp_templates()
    wave_nm = fixture["wave_disc_nm"]
    expected = fixture["disc_spectra"]

    for i, expected_spectrum in enumerate(expected):
        # Extract parameters from fixture
        params_dict = dict(fixture["params"][i])
        l5100 = float(params_dict["l5100"])
        model_idx = int(params_dict["model_idx"])

        # Get the disc template for this model
        disc_lumin_model = templates.disc_lumin[model_idx, :]

        # Compute using netzer_disc
        out = np.asarray(
            netzer_disc(
                wave_nm=wave_nm,
                l5100=l5100,
                disc_wave_nm=templates.disc_wave_nm,
                disc_lumin_model=disc_lumin_model,
            )
        )

        # On the native grid, interpolation is identity, so should match exactly.
        # Use rtol=1e-8 to account for floating-point rounding on very small values.
        np.testing.assert_allclose(
            out, expected_spectrum, rtol=1e-8, atol=0.0, err_msg=f"case {i}"
        )


def test_select_disc_model():
    """Test that select_disc_model returns correct indices and raises on bad labels."""
    from tengri.components.agn.grahsp.disc import select_disc_model
    from tengri.components.agn.grahsp.templates import load_grahsp_templates

    templates = load_grahsp_templates()

    # Test a few valid models
    idx = select_disc_model(
        templates.disc_m, templates.disc_a, templates.disc_mdot, m="6.0", a="0.998", mdot="0.3"
    )
    assert idx == 0

    idx = select_disc_model(
        templates.disc_m, templates.disc_a, templates.disc_mdot, m="8.0", a="0", mdot="0.3"
    )
    assert idx == 10

    idx = select_disc_model(
        templates.disc_m, templates.disc_a, templates.disc_mdot, m="9.0", a="0", mdot="0.03"
    )
    assert idx == 15

    # Test invalid model raises ValueError
    with pytest.raises(ValueError, match="not found"):
        select_disc_model(
            templates.disc_m,
            templates.disc_a,
            templates.disc_mdot,
            m="5.0",  # Invalid M
            a="0",
            mdot="0.3",
        )


def test_netzer_disc_output_properties(fixture):
    """Test that netzer_disc output is finite and non-negative."""
    from tengri.components.agn.grahsp.disc import netzer_disc
    from tengri.components.agn.grahsp.templates import load_grahsp_templates

    templates = load_grahsp_templates()
    wave_nm = fixture["wave_disc_nm"]

    # Test one case
    l5100 = 1e43
    model_idx = 2
    disc_lumin_model = templates.disc_lumin[model_idx, :]

    out = np.asarray(
        netzer_disc(
            wave_nm=wave_nm,
            l5100=l5100,
            disc_wave_nm=templates.disc_wave_nm,
            disc_lumin_model=disc_lumin_model,
        )
    )

    # Should be finite and non-negative (ignoring padding zeros)
    chex.assert_tree_all_finite(out)
    assert_non_negative(out, name="out", msg="Disc spectrum should be non-negative")


def test_netzer_disc_interpolation():
    """Test interpolation to a different wavelength grid."""
    from tengri.components.agn.grahsp.disc import netzer_disc
    from tengri.components.agn.grahsp.templates import load_grahsp_templates

    templates = load_grahsp_templates()

    # Create a coarser output grid
    wave_out = np.logspace(2.0, 5.0, 100)  # 100-10000 nm
    l5100 = 1e43
    model_idx = 2

    disc_lumin_model = templates.disc_lumin[model_idx, :]

    out = np.asarray(
        netzer_disc(
            wave_nm=wave_out,
            l5100=l5100,
            disc_wave_nm=templates.disc_wave_nm,
            disc_lumin_model=disc_lumin_model,
        )
    )

    assert out.shape == wave_out.shape
    chex.assert_tree_all_finite(out)
    assert_non_negative(out, name="out")


def test_jit_compatible():
    """Test that netzer_disc is JIT-compatible."""
    import jax
    import jax.numpy as jnp

    from tengri.components.agn.grahsp.disc import netzer_disc
    from tengri.components.agn.grahsp.templates import load_grahsp_templates

    templates = load_grahsp_templates()
    wave_nm = jnp.linspace(100.0, 100000.0, 50)
    l5100 = 1e43
    disc_lumin_model = templates.disc_lumin[2, :]

    # JIT compile
    jitted = jax.jit(netzer_disc)
    out = np.asarray(jitted(wave_nm, l5100, templates.disc_wave_nm, disc_lumin_model))

    assert out.shape == wave_nm.shape
    chex.assert_tree_all_finite(out)
