# SPDX-License-Identifier: BSD-3-Clause
"""No SFH form is a staircase in a time parameter that moves its own support.

A hard mask at a boundary that moves with a free parameter makes the
integrated photometry a step function of that parameter: a jump every time
an age-grid node crosses the moving boundary. Autodiff sees only the smooth
slope between steps, so the gradient disagrees with the function by orders of
magnitude at each one, and a gradient sampler reads it as an energy error at
any step size.

This sweeps every registered SFH family's onset/age/peak-time parameters
(``ParamDef.z_capped_onset``, derived from the registry the same way
``tests/contract/test_onset_age_z_narrowing_registry_derived.py`` does) plus
every other time parameter that moves a support boundary or a sharp feature
(burst ages, truncation and quench times, end times), on both age kernels, on
the tracked MILES grid. The statistic is
:func:`tests._step_excess.local_step_excess`, each step against the median of
its neighbors: the whole-sweep median of
:func:`tests._step_excess.step_excess` (which ``test_lognormal_onset_smoothness.py``
pins for the log-normal) reads a smooth curve whose slope varies by more than
the threshold across the window as a staircase (#2683).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tests._step_excess import local_step_excess

pytestmark = pytest.mark.gradient

#: Above this multiple of the median neighboring step, the curve is not smooth.
STAIRCASE = 4.0
#: Fine sweep width, matching test_lognormal_onset_smoothness.py's own scale.
N_POINTS = 201
PROBE_BANDS = ["sdss_g", "sdss_r", "sdss_i", "2mass_j"]
_AGE_KERNELS = ("cic", "dsps")

#: Non-zero SFR ratios for the post-starburst ladders. At the registry defaults
#: every ratio is 0, so every bin forms stars at one rate and moving an edge
#: between two equal bins moves no mass: the summed photometry changed by
#: ~1e-9 per sweep step, alternating in sign, and the statistic measured
#: quadrature round-off on a physically flat direction rather than the edge
#: (#2683). These values give every edge a real step, as
#: ``tests/regression/bug/test_bug_2683_followup_gradient.py`` does.
_PSB_FLEX_STEPS = {
    "ratio_young": Fixed(0.6),
    "ratio_flex_0": Fixed(0.5),
    "ratio_flex_1": Fixed(-0.4),
    "ratio_flex_2": Fixed(0.3),
    "ratio_flex_3": Fixed(-0.5),
    "ratio_old_0": Fixed(0.3),
    "ratio_old_1": Fixed(-0.2),
    "ratio_old_2": Fixed(0.2),
}
_PSB2022_STEPS = {
    "ratio_young": Fixed(0.6),
    "ratio_old_0": Fixed(0.3),
    "ratio_old_1": Fixed(-0.2),
    "ratio_old_2": Fixed(0.2),
}

#: (family, short param name, sweep window [Gyr], extra Fixed overrides needed
#: to make the swept boundary visible in the photometry, or to keep a
#: declared ordering constraint satisfied against the parameter the sweep
#: does not touch). One row per z_capped_onset parameter plus every other
#: registry parameter that moves a support boundary or a sharp feature.
_CASES: tuple[tuple[str, str, tuple[float, float], dict], ...] = (
    ("tsnorm", "peak_lbt_gyr", (1.3, 1.5), {}),
    ("snorm", "peak_lbt_gyr", (1.3, 1.5), {}),
    ("snorm_burst", "peak_lbt_gyr", (1.3, 1.5), {}),
    ("snorm_burst", "burst_age_gyr", (0.05, 0.25), {"burst_sfr": Fixed(2.0)}),
    ("tsnorm_burst", "peak_lbt_gyr", (1.3, 1.5), {}),
    ("tsnorm_burst", "burst_age_gyr", (0.05, 0.25), {"burst_sfr": Fixed(2.0)}),
    ("norm", "peak_lbt_gyr", (1.3, 1.5), {}),
    ("lnorm", "age_gyr", (1.3, 1.5), {}),
    ("dpl", "age_gyr", (1.3, 1.5), {}),
    ("dpl_lookback", "age_gyr", (1.3, 1.5), {"end_gyr": Fixed(0.1)}),
    ("dpl_lookback", "end_gyr", (0.05, 0.25), {"age_gyr": Fixed(5.0)}),
    ("const", "start_gyr", (1.3, 1.5), {"end_gyr": Fixed(0.1)}),
    ("const", "end_gyr", (0.05, 0.25), {"start_gyr": Fixed(5.0)}),
    ("exp", "start_gyr", (1.3, 1.5), {}),
    ("dexp", "start_gyr", (1.3, 1.5), {}),
    ("declining_exp", "age_gyr", (1.3, 1.5), {}),
    ("trunc_exp", "age_gyr", (1.3, 1.5), {"end_gyr": Fixed(0.1)}),
    ("trunc_exp", "end_gyr", (0.05, 0.25), {"age_gyr": Fixed(5.0)}),
    ("delayed", "age_gyr", (1.3, 1.5), {}),
    ("const_exp", "age_gyr", (1.3, 1.5), {"quench_gyr": Fixed(0.5)}),
    ("const_exp", "quench_gyr", (0.5, 0.7), {"age_gyr": Fixed(5.0)}),
    ("sfh2exp", "age_gyr", (1.3, 1.5), {}),
    ("sfh2exp", "burst_age_gyr", (0.05, 0.25), {"f_burst": Fixed(0.3)}),
    ("delayed_bq", "age_main_gyr", (1.3, 1.5), {}),
    ("delayed_bq", "age_bq_gyr", (0.05, 0.25), {"r_sfr": Fixed(0.1)}),
    ("periodic", "age_gyr", (1.3, 1.5), {}),
    ("periodic", "delta_bursts_gyr", (0.05, 0.25), {}),
    ("periodic", "tau_bursts_gyr", (0.05, 0.25), {}),
    ("buat08", "age_gyr", (1.3, 1.5), {}),
    ("psb", "age_gyr", (2.0, 2.2), {"burstage_gyr": Fixed(0.1)}),
    ("psb", "burstage_gyr", (0.05, 0.25), {"age_gyr": Fixed(5.0)}),
    ("psb_suess2022", "tflex_gyr", (1.3, 1.5), {"tlast_gyr": Fixed(0.5), **_PSB2022_STEPS}),
    ("psb_suess2022", "tlast_gyr", (0.05, 0.25), {"tflex_gyr": Fixed(2.0), **_PSB2022_STEPS}),
    ("psb_flex", "tflex_gyr", (1.3, 1.5), {"tlast_gyr": Fixed(0.5), **_PSB_FLEX_STEPS}),
    ("psb_flex", "tlast_gyr", (0.05, 0.25), {"tflex_gyr": Fixed(2.0), **_PSB_FLEX_STEPS}),
    # `top_hat` is registered but not yet validated against the DSPS forward
    # path, so `SEDModel.build` refuses it; excluded rather than xfailed,
    # since it is not reachable through the public builder at all.
)

#: Known, documented residual gaps, at (family, param, kernel) granularity --
#: a family can be smooth in one parameter or kernel and not another, so a
#: whole-family skip would hide passing cases. The two kernel names integrate
#: the same function since #2683, so every row is identical for ``cic`` and
#: ``dsps`` and each gap is listed for both. Both remaining gaps are the
#: model's own definition, not the quadrature: :func:`mean_sfh.periodic` (CIGALE
#: ``sfhperiodic``) switches every burst on with a jump in SFR at its onset,
#: lookback ``age - k * delta``. Whenever an onset crosses lookback zero, a new
#: burst starts forming the youngest, brightest stars, so the photometry is
#: continuous but its derivative jumps (a kink), and it then rises by several
#: per cent per Myr. The autodiff gradient matches the finite difference across
#: the kink (to 7 % at the steepest step, 2e-4 in the median), so the gradient
#: is the function's; no knot or sub-bin integration removes a kink the history
#: itself has. Local step excess on the MILES grid, ``origin/main`` (cic / dsps)
#: against this branch, measured at #2683:
#:
#: * ``periodic.age_gyr`` (6.9 / 7.6 on main, 14.6 here): the sweep crosses
#:   ``age = 14 * delta = 1.4`` Gyr, where the 15th burst switches on; every
#:   step above 4 lies in 1.400-1.407 Gyr. Main sampled the onset on the dense
#:   grid and smeared the kink over a cell, which read lower; the onset is an
#:   exact knot here.
#: * ``periodic.delta_bursts_gyr`` (9.1 / 9.3 on main, 9.3 here): the largest
#:   steps sit at ``delta = age / k`` (0.249, 0.237, 0.227, 0.217, 0.208, 0.199
#:   Gyr for ``k`` = 20-25), each an onset crossing lookback zero. Between them
#:   a 1 Myr step in ``delta`` moves the youngest onset by ``k`` Myr, so the
#:   sweep also undersamples the photometry's own oscillation.
#:
#: Not gaps, and smooth on this branch (local excess):
#:
#: * ``psb_flex`` / ``psb_suess2022`` ``tflex`` and ``tlast`` (1.0-1.1, both
#:   kernels): swept with non-zero SFR ratios (``_PSB_FLEX_STEPS``,
#:   ``_PSB2022_STEPS``). With those ratios main's ``dsps`` histogram reads
#:   2.5e9-2.0e10 on all four rows (``cic`` 1.0-1.1), the staircase #2683 removed.
#:   At the default all-zero ratios these rows had measured round-off on a flat
#:   direction (see ``_PSB_FLEX_STEPS``), not an edge: the 4.6-7.0 they read
#:   there, on main's ``cic`` as here, was noise.
#: * ``periodic.tau_bursts_gyr`` (1.13) and ``tsnorm_burst.burst_age_gyr``
#:   (1.14): the whole-sweep median read them at 4.3 and 5.0 because their
#:   slope varies that much across the window. The ``tau_bursts`` finite
#:   differences match the autodiff gradient to 6e-5 at all 201 points, and
#:   4.32 at a 512x integrand against 4.33 at 128x shows the curvature is the
#:   physics, not the quadrature.
_KNOWN_GAPS = {
    ("periodic", "age_gyr", "cic"),
    ("periodic", "age_gyr", "dsps"),
    ("periodic", "delta_bursts_gyr", "cic"),
    ("periodic", "delta_bursts_gyr", "dsps"),
}


def _ids():
    return [f"{fam}-{param}" for fam, param, _win, _extra in _CASES]


@pytest.fixture(scope="module")
def observation():
    return tengri.Observation(photometry=tengri.Photometry.from_names(PROBE_BANDS))


#: The registry's internal family-prefix spelling differs from the grammar's
#: `type` for these families.
_INTERNAL_PREFIX = {"const_exp": "cexp", "psb_suess2022": "psb2022"}


def _sweep_excess(ssp_data_fsps, observation, family, param, window, extra, kernel):
    lo, hi = window
    prefix = _INTERNAL_PREFIX.get(family, family)
    full_param = f"sfh_{prefix}_{param}"

    sfh_group = {
        "type": family,
        "all_params": Fixed(DEFAULT),
        param: Uniform(lo, hi),
        "age_kernel": kernel,
        **extra,
    }

    model = SEDModel.build(
        ssp_data=ssp_data_fsps,
        observation=observation,
        neb={"type": "none"},
        sfh=sfh_group,
        redshift=Fixed(0.1),
        approx=None,
    )
    f = jax.jit(jax.vmap(lambda a: jnp.sum(model.predict_photometry({full_param: a}))))
    values = np.asarray(f(jnp.linspace(lo, hi, N_POINTS)))
    if np.std(values) == 0.0:
        pytest.skip(f"{family}.{param} at kernel={kernel} is inert over this window")
    return local_step_excess(values)


@pytest.mark.parametrize("kernel", _AGE_KERNELS)
@pytest.mark.parametrize("family,param,window,extra", _CASES, ids=_ids())
def test_no_staircase_in_the_swept_time_parameter(
    ssp_data_fsps, observation, family, param, window, extra, kernel
):
    """``local_step_excess`` of summed photometry stays below 4 across a fine sweep.

    (family, param, kernel) triples in :data:`_KNOWN_GAPS` are measured and
    reported but not held to the threshold -- see that constant's docstring
    for each entry's mechanism.
    """
    excess = _sweep_excess(ssp_data_fsps, observation, family, param, window, extra, kernel)
    if (family, param, kernel) in _KNOWN_GAPS:
        pytest.skip(f"{family}.{param} kernel={kernel}: known gap, excess={excess:.3g}")
    assert excess < STAIRCASE, (
        f"{family}.{param} kernel={kernel}: local_step_excess={excess:.3g} >= {STAIRCASE} "
        f"over [{window[0]}, {window[1]}] Gyr -- staircase in the swept parameter."
    )
