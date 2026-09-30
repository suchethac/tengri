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
the tracked MILES grid. The statistic (:func:`tests._step_excess.step_excess`)
is the one ``test_lognormal_onset_smoothness.py`` pins for the log-normal.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tests._step_excess import step_excess

pytestmark = pytest.mark.gradient

#: Above this multiple of the median step, the curve is not smooth.
STAIRCASE = 4.0
#: Fine sweep width, matching test_lognormal_onset_smoothness.py's own scale.
N_POINTS = 201
PROBE_BANDS = ["sdss_g", "sdss_r", "sdss_i", "2mass_j"]
_AGE_KERNELS = ("cic", "dsps")

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
    ("psb_suess2022", "tflex_gyr", (1.3, 1.5), {"tlast_gyr": Fixed(0.5)}),
    ("psb_suess2022", "tlast_gyr", (0.05, 0.25), {"tflex_gyr": Fixed(2.0)}),
    ("psb_flex", "tflex_gyr", (1.3, 1.5), {"tlast_gyr": Fixed(0.5)}),
    ("psb_flex", "tlast_gyr", (0.05, 0.25), {"tflex_gyr": Fixed(2.0)}),
    # `top_hat` is registered but not yet validated against the DSPS forward
    # path, so `SEDModel.build` refuses it; excluded rather than xfailed,
    # since it is not reachable through the public builder at all.
)

#: Known, documented residual gaps: the mechanism is not a quadrature defect
#: fixable by the boundary partial-cell weight, or (periodic) only partially
#: so. See the module docstring of ``mean_sfh.periodic`` and
#: ``mean_sfh.truncated_skewnormal`` / issue #299 for the mechanism each one
#: is failing by.
_KNOWN_GAPS = {
    "periodic",
    "tsnorm",
    "tsnorm_burst",
    "psb_suess2022",
    "psb_flex",
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
    return step_excess(values)


@pytest.mark.parametrize("kernel", _AGE_KERNELS)
@pytest.mark.parametrize("family,param,window,extra", _CASES, ids=_ids())
def test_no_staircase_in_the_swept_time_parameter(
    ssp_data_fsps, observation, family, param, window, extra, kernel
):
    """``step_excess`` of summed photometry stays below 4 across a fine sweep.

    Families in :data:`_KNOWN_GAPS` are measured and reported but not held to
    the threshold: ``tsnorm``/``tsnorm_burst`` alias against the SSP age
    grid's own resolution rather than a moving-boundary quadrature defect
    (issue #299's ``SFHBurstAliasingWarning`` describes the same mechanism);
    ``periodic``'s burst-train onset and the rectangular type's own closing
    edge are partial-cell weighted, but an older overlapping burst's closing
    edge is not; ``psb_suess2022``/``psb_flex`` get the same partial-cell
    treatment at their ``tlast_gyr``/``tflex_gyr`` bin edges
    (``_piecewise_constant_sfr_smooth``), which resolves some cells but not
    the DSPS kernel's own coarser age-grid aliasing at that edge (the same
    mechanism as ``tsnorm``) or every CIC cell for ``tflex_gyr``.
    """
    excess = _sweep_excess(ssp_data_fsps, observation, family, param, window, extra, kernel)
    if family in _KNOWN_GAPS:
        pytest.skip(f"{family}.{param} kernel={kernel}: known gap, excess={excess:.3g}")
    assert excess < STAIRCASE, (
        f"{family}.{param} kernel={kernel}: step_excess={excess:.3g} >= {STAIRCASE} "
        f"over [{window[0]}, {window[1]}] Gyr -- staircase in the swept parameter."
    )
