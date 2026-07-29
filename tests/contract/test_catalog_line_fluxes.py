# SPDX-License-Identifier: BSD-3-Clause
"""Catalog fitting with per-galaxy emission-line fluxes (#1480)."""

from __future__ import annotations

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.contract


def test_two_galaxies_with_different_line_fluxes_get_different_posteriors(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """The defining behavior: per-galaxy line data must reach the likelihood.

    Before #1480, both galaxies were scored against the template Observation's
    line fluxes, so their posteriors were identical up to photometric noise.
    """
    pytest.importorskip("blackjax")
    from tests.contract._line_catalog_fixture import build_two_galaxy_catalog

    cat, _ = build_two_galaxy_catalog(
        halpha=(1.0e-16, 4.0e-16), ssp=synthetic_ssp_wide, obs_base=synthetic_tophat_obs
    )
    post = cat.fit(
        "mcmc_hmc",
        key=jax.random.PRNGKey(0),
        n_warmup=60,
        n_samples=60,
        n_leapfrog_steps=8,
    )
    # Get medians by accessing via the public interface
    med = np.array([np.median(post[i].properties["sfr_10myr"]) for i in range(2)])
    ratio = float(med[1] / med[0])
    # Galaxy with 4x Halpha MUST recover higher SFR. Before #1480, both galaxies
    # were scored against template line fluxes, giving ratio ≈ 1.00 (silent subst).
    assert ratio > 1.5, (
        f"galaxy with 4x Halpha recovered SFR ratio {ratio:.2f}; "
        "line fluxes are not reaching the per-galaxy likelihood"
    )


def test_swapped_halpha_flips_the_ratio(synthetic_ssp_wide, synthetic_tophat_obs):
    """Verify that swapping Halpha values swaps which galaxy has higher SFR.

    This test catches transposition bugs (e.g. line_flux_obs and line_flux_err
    swapped, or galaxy dict keys mis-assigned). Before #1480, swapping Halpha
    had no effect (ratio stayed ~1.00). After the fix, swapping must flip the
    ratio to the other side of 1.0.
    """
    pytest.importorskip("blackjax")
    from tests.contract._line_catalog_fixture import build_two_galaxy_catalog

    # Original: galaxy 0 has 1x, galaxy 1 has 4x
    cat_orig, _ = build_two_galaxy_catalog(
        halpha=(1.0e-16, 4.0e-16),
        ssp=synthetic_ssp_wide,
        obs_base=synthetic_tophat_obs,
    )
    post_orig = cat_orig.fit(
        "mcmc_hmc",
        key=jax.random.PRNGKey(0),
        n_warmup=60,
        n_samples=60,
        n_leapfrog_steps=8,
    )
    med_orig = np.array([np.median(post_orig[i].properties["sfr_10myr"])
                         for i in range(2)])
    ratio_orig = float(med_orig[1] / med_orig[0])

    # Swapped: galaxy 0 has 4x, galaxy 1 has 1x
    cat_swap, _ = build_two_galaxy_catalog(
        halpha=(4.0e-16, 1.0e-16),  # Reversed
        ssp=synthetic_ssp_wide,
        obs_base=synthetic_tophat_obs,
    )
    post_swap = cat_swap.fit(
        "mcmc_hmc",
        key=jax.random.PRNGKey(0),
        n_warmup=60,
        n_samples=60,
        n_leapfrog_steps=8,
    )
    med_swap = np.array([np.median(post_swap[i].properties["sfr_10myr"])
                         for i in range(2)])
    ratio_swap = float(med_swap[1] / med_swap[0])

    # The ratios must flip to opposite sides of 1.0.
    # If not, per-galaxy line data is NOT reaching the likelihood.
    assert ratio_orig > 1.5, (
        f"original (1x, 4x) ratio {ratio_orig:.4f} not > 1.5; "
        "line data may not be reaching likelihood"
    )
    assert ratio_swap < 0.67, (
        f"swapped (4x, 1x) ratio {ratio_swap:.4f} not < 0.67; "
        "swapping Halpha did not flip the ratio as expected"
    )
    # Also check they're roughly reciprocals (within ~20% tolerance for MCMC noise)
    reciprocal_ratio = 1.0 / ratio_orig
    tolerance = abs(ratio_swap - reciprocal_ratio) / reciprocal_ratio
    assert tolerance < 0.2, (
        f"swapped ratio {ratio_swap:.4f} is not reciprocal of "
        f"{ratio_orig:.4f} (tolerance {tolerance:.2%} > 20%)"
    )


def test_line_column_count_must_match_the_observation(synthetic_ssp_wide, synthetic_tophat_obs):
    """Validate that line column count matches observation."""
    from tests.contract._line_catalog_fixture import build_two_galaxy_catalog

    with pytest.raises(ValueError, match="line"):
        build_two_galaxy_catalog(
            halpha=(1e-16, 4e-16),
            n_line_cols=3,
            ssp=synthetic_ssp_wide,
            obs_base=synthetic_tophat_obs,
        )
