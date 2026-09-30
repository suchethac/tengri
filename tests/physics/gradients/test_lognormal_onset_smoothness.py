# SPDX-License-Identifier: BSD-3-Clause
"""The log-normal SFH must stay differentiable in age at a wide width.

``mean_sfh.lognormal`` evaluates a density in cosmic time since formation,
:math:`T = \\mathrm{age} - t_{\\rm lb}`, and masks it where :math:`T \\le 0`.
The density is integrable at the onset for any width, but at a wide one it
approaches zero so slowly that it is effectively flat across the whole grid --
at width 2.14 dex the kernel varies by about a factor of four over four
decades in :math:`T`. A hard mask on a flat integrand means every age-grid
node switches on at a value comparable to its neighbors', and because the
shape is point-sampled and integrated by trapezoid, the integral jumps each
time the moving boundary crosses a node.

The photometry was a sawtooth in age and the gradient was wrong where it
mattered: measured on FSPS MIST C3K, 89 steps above four times the median step
over a 2401-point sweep, and a worst finite-difference disagreement of 7.7
against 0.0021 for delayed-tau on the same scan. NUTS cannot integrate that --
an energy error at any step -- and it showed up as divergences and an adapted
step size driven toward zero.

This is a quadrature defect rather than a physics one, so the fix is
quadrature: weight the boundary cell by the fraction of it inside the support,
smoothly. The scale is the local grid spacing, which the problem already
fixes, so nothing here is a tapering length chosen by hand.

Two properties are pinned, and the second is what makes the first safe: the
sawtooth is gone at a wide width, and nothing moves at a narrow one, where the
kernel was already small at onset and the old mask was already correct.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform

pytestmark = [pytest.mark.gradient, pytest.mark.regression_bug]

#: Wide enough that the kernel is flat across the grid. Row V of the paper's
#: CANDELS grid reached this in its posterior, which is how it was found.
WIDE_DEX = 2.14
#: Narrow enough that the kernel is already negligible at the onset.
NARROW_DEX = 0.30
PROBE_BAND = ["sdss_g", "sdss_r", "sdss_i", "2mass_j"]
#: Teeth fall about 20 Myr apart near this age, so this resolves them well.
AGES = np.linspace(1.30, 1.50, 201)


def _model(ssp, observation, width):
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        neb={"type": "none"},
        sfh={
            "type": "lnorm",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Fixed(10.0),
            "peak_gyr": Fixed(5.0),
            "width_gyr": Fixed(width),
            "age_gyr": Uniform(0.5, 3.0),
        },
        redshift=Fixed(1.0),
        approx=None,
    )


@pytest.fixture(scope="module")
def observation():
    return tengri.Observation(photometry=tengri.Photometry.from_names(PROBE_BAND))


def _sweep(model):
    f = jax.jit(lambda a: sum(model.predict_photometry({"sfh_lnorm_age_gyr": a})))
    return np.array([float(f(float(a))) for a in AGES])


def _step_excess(values: np.ndarray) -> float:
    """Largest single step, in units of the median step. Smooth curves sit near 1."""
    steps = np.abs(np.diff(values)) / np.abs(values[:-1])
    return float(steps.max() / np.median(steps))


def test_the_probe_would_see_a_sawtooth_if_one_were_there(ssp_data_wne, observation):
    """Guards the two below: a sweep too coarse to resolve teeth passes them."""
    values = _sweep(_model(ssp_data_wne, observation, WIDE_DEX))

    assert values.std() > 0.0, (
        "the photometry did not move across the age sweep at all, so this "
        "module measures nothing -- check that age_gyr is free and passed"
    )
    assert len(AGES) > 100, "too few samples to resolve node crossings"


def test_a_wide_lognormal_is_smooth_in_age(ssp_data_wne, observation):
    """The defect: the trapezoid jumped whenever the onset crossed a node."""
    excess = _step_excess(_sweep(_model(ssp_data_wne, observation, WIDE_DEX)))

    assert excess < 4.0, (
        f"the largest step across the age sweep is {excess:.1f} times the median "
        "step, so the photometry still has a sawtooth in age at width "
        f"{WIDE_DEX} dex. A hard onset mask on a nearly flat kernel switches "
        "grid nodes on at full value; weight the boundary cell instead."
    )


def test_a_narrow_lognormal_is_untouched(ssp_data_wne, observation):
    """The fix must be a no-op wherever the old mask was already right.

    At a narrow width the kernel is negligible by the time the onset is
    reached, so the boundary cell carries nothing and weighting it changes
    nothing. Measured against the unfixed code this arm was bit-identical, and
    a fix that perturbed it would be changing results it has no business
    touching.
    """
    excess = _step_excess(_sweep(_model(ssp_data_wne, observation, NARROW_DEX)))

    assert excess < 2.0, (
        f"a narrow log-normal shows a step {excess:.1f} times the median, which "
        "it did not before: the boundary weighting is reaching a regime where "
        "the onset was never the problem."
    )


def test_the_gradient_agrees_with_finite_differences_at_a_wide_width(ssp_data_wne, observation):
    """Autodiff must see what a finite difference sees.

    A masked branch contributes no derivative, so before the fix autodiff
    stepped straight over the discontinuity: the two disagreed by a factor of
    several while the function jumped underneath. The bar is measured here
    rather than asserted -- delayed-tau is scanned in the same run on the same
    grid, so the tolerance is whatever a well-behaved SFH achieves.
    """
    model = _model(ssp_data_wne, observation, WIDE_DEX)
    f = jax.jit(lambda a: sum(model.predict_photometry({"sfh_lnorm_age_gyr": a})))
    g = jax.jit(jax.grad(lambda a: sum(model.predict_photometry({"sfh_lnorm_age_gyr": a}))))

    # Dense over a narrow window rather than sparse over a wide one. A kink at
    # a node crossing is only a few times 1e-4 Gyr across, so a scan stepping
    # 2e-3 steps over it and reports agreement: the first version of this test
    # passed a linear partial-cell weight whose worst disagreement was 0.22,
    # measured elsewhere at an age it happened not to sample.
    h = 3e-5
    worst = 0.0
    for age in np.linspace(1.405, 1.420, 121):
        analytic = float(g(float(age)))
        if analytic == 0.0:
            continue
        numeric = float((f(float(age) + h) - f(float(age) - h)) / (2 * h))
        worst = max(worst, abs(numeric / analytic - 1.0))

    assert worst < 0.05, (
        f"worst |FD/AD - 1| is {worst:.4f} across the scan. Autodiff and a "
        "finite difference disagree, which at this width means the integrand "
        "still has a corner the mask is hiding from the derivative."
    )
