# SPDX-License-Identifier: BSD-3-Clause
"""Diagnostics module for hierarchical PSD recovery."""

import numpy as np
import pytest

pytestmark = pytest.mark.contract


def test_flat_widths_do_not_pass_the_scaling_criterion():
    """June's failure signature: 8192x more data, unchanged intervals."""
    from tengri.inference.population.diagnostics import interval_width_scaling

    n_values = np.array([50, 100, 200, 500])
    flat = np.array([1.80, 1.79, 1.81, 1.80])
    out = interval_width_scaling(flat, n_values)
    assert not out["excludes_zero_3sigma"], "a flat width must not pass"


def test_sqrt_n_widths_pass_the_scaling_criterion():
    from tengri.inference.population.diagnostics import interval_width_scaling

    n_values = np.array([50, 100, 200, 500])
    scaling = 12.0 / np.sqrt(n_values)
    out = interval_width_scaling(scaling, n_values)
    assert out["excludes_zero_3sigma"]
    assert abs(out["slope"] + 0.5) < 0.05, f"slope {out['slope']:.3f} should be -0.5"


def test_sqrt_n_with_noise_exercises_the_three_sigma_criterion():
    """1/sqrt(N) scaling plus scatter: slope_err is non-zero, so the 3-sigma
    comparison decides the outcome rather than being bypassed."""
    from tengri.inference.population.diagnostics import interval_width_scaling

    rng = np.random.default_rng(42)
    n_values = np.array([50, 100, 200, 500])
    widths = 12.0 / np.sqrt(n_values) * (1.0 + rng.normal(0.0, 0.04, 4))
    out = interval_width_scaling(widths, n_values)
    assert out["slope_err"] > 0.0, "perfect-fit data bypasses the criterion"
    assert -0.60 < out["slope"] < -0.40, f"slope {out['slope']:.3f} should be near -0.5"
    assert out["excludes_zero_3sigma"]


def test_interval_width_scaling_rejects_too_few_points():
    """Fewer than 3 points: cannot estimate slope_err (divides by n-2)."""
    from tengri.inference.population.diagnostics import interval_width_scaling

    with pytest.raises(ValueError, match="At least 3 data points required"):
        interval_width_scaling(np.array([1.0, 2.0]), np.array([10, 20]))


def test_interval_width_scaling_rejects_nan_input():
    """NaN in the input should raise, not silently return False."""
    from tengri.inference.population.diagnostics import interval_width_scaling

    n_values = np.array([50, 100, 200, 500])
    with pytest.raises(ValueError, match="NaN in input"):
        interval_width_scaling(np.array([1.8, np.nan, 1.81, 1.80]), n_values)

    with pytest.raises(ValueError, match="NaN in input"):
        interval_width_scaling(np.array([1.8, 1.79, 1.81, 1.80]), np.array([50, np.nan, 200, 500]))


def test_zero_divergence_warning():
    """Zero divergences across the population is a red flag, not a pass."""
    import warnings

    from tengri.inference.population.diagnostics import report
    from tengri.inference.population.estimator import ESSSummary

    # Create mock data
    interim_result = {"n_divergent": np.array([0, 0, 0])}
    ess_summary = ESSSummary(
        at_mode=np.array([100.0, 105.0, 98.0]),
        min_high_mass=np.array([95.0, 100.0, 92.0]),
    )
    shared_posterior = (None, ess_summary)

    # Should warn when divergences sum to zero
    with pytest.warns(UserWarning, match="Zero divergences across the whole population"):
        result = report(interim_result, shared_posterior)

    assert result["zero_divergence_flag"] is True

    # Should NOT warn when divergences are present
    interim_result_with_div = {"n_divergent": np.array([0, 1, 0])}
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        result = report(interim_result_with_div, shared_posterior)

    # Filter to UserWarnings only
    user_warnings = [w for w in warning_list if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 0, "Should not warn when divergences present"
    assert result["zero_divergence_flag"] is False


def test_credible_interval_non_square_grid():
    """Non-square grid (n_sigma != n_tau) must reshape correctly.

    Tests that a non-square grid (n_sigma=16, n_tau=24) is reshaped
    correctly when computing marginals. The C-ordered reshape is critical:
    getting axes backwards produces plausible-but-wrong marginals.
    """
    from tengri.inference.population.diagnostics import credible_interval
    from tengri.inference.population.estimator import SharedGrid

    grid = SharedGrid.uniform(
        sigma_bounds=(0.1, 2.0),
        tau_bounds_yr=(1e6, 1e8),
        n_sigma=16,
        n_tau=24,
    )

    # Create a log-posterior with a broad peak
    log_posterior = np.zeros(16 * 24)
    sigma_idx = np.repeat(np.arange(16), 24)
    tau_idx = np.tile(np.arange(24), 16)
    # Peak centered at (sigma[8], tau[12]) with Gaussian decay
    sigma_dist = (sigma_idx - 8.0) ** 2
    tau_dist = (tau_idx - 12.0) ** 2
    log_posterior = -0.5 * (sigma_dist / 4.0 + tau_dist / 9.0)

    result = credible_interval(log_posterior, grid, level=0.68)

    # Verify all required keys are present
    assert "sigma_lower" in result and "sigma_upper" in result
    assert "tau_lower_yr" in result and "tau_upper_yr" in result
    # Intervals should have non-zero width (major sanity check for reshape)
    assert result["sigma_lower"] < result["sigma_upper"], (
        f"sigma interval inverted: [{result['sigma_lower']}, "
        f"{result['sigma_upper']}], indicates reshape error"
    )
    assert result["tau_lower_yr"] < result["tau_upper_yr"], (
        f"tau interval inverted: [{result['tau_lower_yr']}, "
        f"{result['tau_upper_yr']}], indicates reshape error"
    )
    # Verify the level was stored correctly
    assert result["credible_levels"] == (0.68, 0.68)


def test_report_accesses_ess_summary_fields():
    """report() must read .at_mode and .min_high_mass from ESSSummary."""
    from tengri.inference.population.diagnostics import report
    from tengri.inference.population.estimator import ESSSummary

    interim_result = {"n_divergent": np.array([1, 2, 0])}
    ess_at_mode = np.array([100.5, 105.3, 98.7])
    ess_min_high = np.array([95.2, 100.1, 92.9])
    ess_summary = ESSSummary(at_mode=ess_at_mode, min_high_mass=ess_min_high)
    shared_posterior = (None, ess_summary)

    result = report(interim_result, shared_posterior)

    # Verify the fields were read correctly
    np.testing.assert_array_equal(result["ess_at_mode"], ess_at_mode)
    np.testing.assert_array_equal(result["ess_min_high_mass"], ess_min_high)
    assert result["zero_divergence_flag"] is False


def test_credible_interval_survives_a_large_log_posterior():
    """exp() without a max subtraction underflows once N is large.

    log_posterior sums one term per galaxy, so its magnitude grows with the
    population. At N ~ 64 it reaches the hundreds; np.exp of that underflows
    every node to 0 and the normalization returns NaN for the whole grid.
    """
    from tengri.inference.population.diagnostics import credible_interval
    from tengri.inference.population.estimator import SharedGrid

    grid = SharedGrid.uniform(
        sigma_bounds=(0.01, 1.0), tau_bounds_yr=(1.0e7, 5.0e8), n_sigma=20, n_tau=24
    )
    nodes = grid.nodes
    # A peaked posterior offset by a huge constant, as a real sum over galaxies is.
    peak = np.exp(-0.5 * ((np.asarray(nodes)[:, 0] - 0.5) / 0.1) ** 2)
    lp = np.log(peak + 1e-300) - 5000.0
    out = credible_interval(lp, grid)
    for k, v in out.items():
        if k == "credible_levels":
            continue
        assert np.isfinite(v), f"{k} is {v}; the grid failed to normalize"
    assert out["sigma_lower"] < out["sigma_upper"]


def test_credible_interval_is_not_quantized_to_grid_nodes():
    """Snapping to nodes quantized the WIDTH, which breaks width scaling.

    Two different population sizes both reported a sigma width of exactly
    0.117 — seven cells of a 0.01678 grid, twice.
    """
    from tengri.inference.population.diagnostics import credible_interval
    from tengri.inference.population.estimator import SharedGrid

    grid = SharedGrid.uniform(
        sigma_bounds=(0.01, 1.0), tau_bounds_yr=(1.0e7, 5.0e8), n_sigma=60, n_tau=60
    )
    spacing = float(np.asarray(grid.sigma)[1] - np.asarray(grid.sigma)[0])
    sig = np.asarray(grid.nodes)[:, 0]
    widths = []
    for center in (0.40, 0.42):  # shifted well under one grid cell
        lp = -0.5 * ((sig - center) / 0.08) ** 2
        out = credible_interval(lp, grid)
        widths.append(out["sigma_upper"] - out["sigma_lower"])
    # Endpoints must not land on grid nodes for every input.
    on_node = [abs((w / spacing) - round(w / spacing)) < 1e-9 for w in widths]
    assert not all(on_node), (
        f"widths {widths} are all exact multiples of the grid spacing {spacing}; "
        "credible_interval is still snapping to nodes"
    )
