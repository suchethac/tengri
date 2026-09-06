# SPDX-License-Identifier: BSD-3-Clause
"""Catalog fitting with per-galaxy emission-line fluxes (#1480).

Direct plumbing tests for per-galaxy line flux threading. Tests verify the
MECHANISM (data reaches likelihood), not the physics (model predictions).
synthetic_ssp_wide has no nebular emission (Halpha_pred=0), but per-galaxy
chi-squared terms ((obs - 0) / err)**2 differ, proving threading works.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.contract


def test_engine_receives_per_galaxy_line_data(synthetic_ssp_wide, synthetic_tophat_obs):
    """Test 1: Verify catalog ingests correct per-galaxy line data.

    Check that the catalog's internal _catalog_arrays has the right
    line_flux_obs for each galaxy before any engine runs.
    """
    from tests.contract._line_catalog_fixture import build_two_galaxy_catalog

    cat, _ = build_two_galaxy_catalog(
        halpha=(1.0e-16, 4.0e-16),
        ssp=synthetic_ssp_wide,
        obs_base=synthetic_tophat_obs,
    )

    ca = cat._catalog_arrays

    # Verify per-galaxy line fluxes were ingested correctly
    assert ca.line_flux_obs is not None, "line_flux_obs is None"
    assert ca.line_flux_obs.shape == (2, 1), f"Expected shape (2, 1), got {ca.line_flux_obs.shape}"

    # Verify the 4:1 contrast ratio (galaxy 1 is 4x galaxy 0)
    g0 = float(ca.line_flux_obs[0, 0])
    g1 = float(ca.line_flux_obs[1, 0])
    assert g0 != 0.0, f"Galaxy 0 flux is zero: {g0}"
    assert np.all(np.isfinite(g0)), (
        "`g0` is non-finite — non-zero is not enough, `nan != 0.0` is True "
        "and a NaN satisfies a non-zero assertion (#2178)"
    )
    ratio = g1 / g0
    assert np.allclose(ratio, 4.0), (
        f"Galaxy 1 is not 4x galaxy 0: ratio={ratio:.4f}, g0={g0:.4e}, g1={g1:.4e}"
    )


def test_likelihood_is_sensitive_to_line_data(synthetic_ssp_wide, synthetic_tophat_obs):
    """Test 2: Per-galaxy likelihood differs based on observed Halpha.

    Evaluate loss through the same 2-arg path run_one uses, WITH per-galaxy
    data_args. Probes that per-galaxy line fluxes reach and affect the objective.
    """
    from jax.flatten_util import ravel_pytree

    from tengri.inference.backends.mcmc._shared import _get_flat_logdensity
    from tengri.inference.fitter import Fitter
    from tests.contract._line_catalog_fixture import build_two_galaxy_catalog

    cat, truth = build_two_galaxy_catalog(
        halpha=(1.0e-16, 4.0e-16),
        ssp=synthetic_ssp_wide,
        obs_base=synthetic_tophat_obs,
    )

    ca = cat._catalog_arrays

    # Build ONE fitter and get the 2-arg loss function (same path run_one uses)
    fitter = Fitter(cat.fwd, ca.flux[0], ca.noise[0], data_type="photometry")
    log_posterior_2arg, _, _, template_data_args = _get_flat_logdensity(fitter, truth)
    init_flat, _ = ravel_pytree(truth)

    # Build per-galaxy data_args for SAME loss function
    data_args_g0 = dict(template_data_args)
    data_args_g0["data"] = ca.flux[0]
    data_args_g0["noise"] = ca.noise[0]
    data_args_g0["noise_inv"] = 1.0 / (ca.noise[0] ** 2)
    data_args_g0["line_flux_obs"] = ca.line_flux_obs[0]
    data_args_g0["line_flux_err"] = ca.line_flux_err[0]

    data_args_g1 = dict(template_data_args)
    data_args_g1["data"] = ca.flux[1]
    data_args_g1["noise"] = ca.noise[1]
    data_args_g1["noise_inv"] = 1.0 / (ca.noise[1] ** 2)
    data_args_g1["line_flux_obs"] = ca.line_flux_obs[1]
    data_args_g1["line_flux_err"] = ca.line_flux_err[1]

    # Evaluate loss with per-galaxy data_args
    nlp_g0 = float(log_posterior_2arg(init_flat, data_args_g0))
    nlp_g1 = float(log_posterior_2arg(init_flat, data_args_g1))

    # Expected: line chi2_g0 = 0 (obs_g0 = measured_base),
    # line chi2_g1 = (3*base / (0.05*base))**2 = 3600
    # The loss difference is proportional to the line chi2 contribution.
    # Even though photometry dominates the overall loss (40M+), the per-galaxy
    # difference should be detectable if per-galaxy line data reaches the objective.
    nlp_diff = nlp_g1 - nlp_g0
    # The losses must differ (not allclose) and the difference must be non-trivial
    # (> 1000 in loss units, roughly the expected line contribution scale)
    assert not np.allclose(nlp_g0, nlp_g1), (
        f"Loss identical for both galaxies; line flux data not reaching likelihood. "
        f"nlp_g0={nlp_g0:.4e}, nlp_g1={nlp_g1:.4e}, diff={nlp_diff:.4e}"
    )
    assert abs(nlp_diff) > 1000.0, (
        f"Loss difference too small ({abs(nlp_diff):.1f}): "
        f"nlp_g0={nlp_g0:.4e}, nlp_g1={nlp_g1:.4e}. "
        f"Per-galaxy line data not reaching objective."
    )


def test_swapped_halpha_flips_outcomes(synthetic_ssp_wide, synthetic_tophat_obs):
    """Test 3: Swapping Halpha values swaps likelihood differences.

    Build catalog with halpha SWAPPED and verify Test 1 and Test 2 outcomes
    exchange. Permanent check for transposition bugs.
    """
    from tengri.inference.fitter import Fitter
    from tests.contract._line_catalog_fixture import build_two_galaxy_catalog

    # Original: (1x, 4x)
    cat_orig, truth_orig = build_two_galaxy_catalog(
        halpha=(1.0e-16, 4.0e-16),
        ssp=synthetic_ssp_wide,
        obs_base=synthetic_tophat_obs,
    )
    ca_orig = cat_orig._catalog_arrays

    # Swapped: (4x, 1x)
    cat_swap, truth_swap = build_two_galaxy_catalog(
        halpha=(4.0e-16, 1.0e-16),
        ssp=synthetic_ssp_wide,
        obs_base=synthetic_tophat_obs,
    )
    ca_swap = cat_swap._catalog_arrays

    # Likelihood differences (g0 - g1)
    fitter_orig_g0 = Fitter(
        cat_orig.fwd, ca_orig.flux[0], ca_orig.noise[0], data_type="photometry"
    )
    fitter_orig_g1 = Fitter(
        cat_orig.fwd, ca_orig.flux[1], ca_orig.noise[1], data_type="photometry"
    )

    nlp_fn_orig_g0 = fitter_orig_g0._get_or_build_logdensity_fn()
    nlp_fn_orig_g1 = fitter_orig_g1._get_or_build_logdensity_fn()

    data_args_orig_g0 = dict(fitter_orig_g0._data_args)
    data_args_orig_g0["line_flux_obs"] = ca_orig.line_flux_obs[0]
    data_args_orig_g0["line_flux_err"] = ca_orig.line_flux_err[0]

    data_args_orig_g1 = dict(fitter_orig_g1._data_args)
    data_args_orig_g1["line_flux_obs"] = ca_orig.line_flux_obs[1]
    data_args_orig_g1["line_flux_err"] = ca_orig.line_flux_err[1]

    ratio_orig = float(
        nlp_fn_orig_g0(truth_orig, data_args_orig_g0)
        - nlp_fn_orig_g1(truth_orig, data_args_orig_g1)
    )

    # Same computation with swapped
    fitter_swap_g0 = Fitter(
        cat_swap.fwd, ca_swap.flux[0], ca_swap.noise[0], data_type="photometry"
    )
    fitter_swap_g1 = Fitter(
        cat_swap.fwd, ca_swap.flux[1], ca_swap.noise[1], data_type="photometry"
    )

    nlp_fn_swap_g0 = fitter_swap_g0._get_or_build_logdensity_fn()
    nlp_fn_swap_g1 = fitter_swap_g1._get_or_build_logdensity_fn()

    data_args_swap_g0 = dict(fitter_swap_g0._data_args)
    data_args_swap_g0["line_flux_obs"] = ca_swap.line_flux_obs[0]
    data_args_swap_g0["line_flux_err"] = ca_swap.line_flux_err[0]

    data_args_swap_g1 = dict(fitter_swap_g1._data_args)
    data_args_swap_g1["line_flux_obs"] = ca_swap.line_flux_obs[1]
    data_args_swap_g1["line_flux_err"] = ca_swap.line_flux_err[1]

    ratio_swap = float(
        nlp_fn_swap_g0(truth_swap, data_args_swap_g0)
        - nlp_fn_swap_g1(truth_swap, data_args_swap_g1)
    )

    # Ratios should flip sign (opposite sides of 0)
    assert ratio_orig * ratio_swap < 0, (
        f"Ratios did not flip: orig={ratio_orig:.4f}, swap={ratio_swap:.4f}"
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
