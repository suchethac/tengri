# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #2645: PSB fixed old bins bounded to cosmic time.

``psb_flex`` and ``psb_suess2022`` (both :func:`psb_continuity_flex`) laid
their ``n_fixed`` fixed old bins out as equal-width intervals spanning
``[tflex_gyr, 13.7 Gyr]`` at every redshift. At z = 0.5 (age(z) = 8.59 Gyr)
the fixed section's oldest bins lie partly or entirely before the Big Bang:
the default model forms 33% of its stellar mass before the Big Bang
(``SFHBeforeBigBangWarning``) and publishes star formation to 13.3 Gyr
lookback, 4.7 Gyr older than the universe.

The fix bounds the fixed section to ``[tflex_gyr, age_universe_yr]``, the
same ``age_universe_yr`` injection :func:`psb_wild2020` already receives
(component.py's ``apply``/``compute_joint_weights``), for a fixed redshift at
build time. A free redshift is served by the ceiling convention #2567
documents (:class:`NonparametricBinEdgesAtRedshiftCeilingWarning` for the
other z-scaled nonparametric ladders) through the same ``age_at_z``-based
injection, since the injected ``age_universe_yr`` already tracks
``t_obs_gyr`` per draw at eager-forward time.

References: Suess et al. 2022, ApJ 935, 146, §3.1.4; Wild et al. 2020 Eq. 5
(psb_wild2020's own age_universe_yr injection); issue #2567 (the z-ceiling
convention for free-redshift nonparametric bin ladders).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar.component import SFHBeforeBigBangWarning
from tengri.components.stellar.sfh.nonparametric import PSB_FLEX_DEFAULT_MAX_AGE_GYR
from tengri.utils.cosmology import age_at_z

pytestmark = pytest.mark.regression_bug

PSB_FAMILIES = ["psb_flex", "psb_suess2022"]
AGE_KERNELS = ["cic", "dsps"]


def _build(ssp, sfh_type, redshift, *, age_kernel="cic", tflex_free=False):
    sfh_group = {
        "type": sfh_type,
        "all_params": Fixed(DEFAULT),
        "log_total_mass": Fixed(10.0),
        "tlast_gyr": Fixed(0.2),
        "age_kernel": age_kernel,
    }
    if tflex_free:
        from tengri import Uniform

        sfh_group["tflex_gyr"] = Uniform(1.0, 5.0, default=2.0)
    else:
        sfh_group["tflex_gyr"] = Fixed(2.0)
    return SEDModel.build(
        ssp_data=ssp,
        redshift=Fixed(redshift),
        neb={"type": "none"},
        sfh=sfh_group,
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
    )


@pytest.mark.parametrize("sfh_type", PSB_FAMILIES)
def test_default_psb_does_not_warn_before_big_bang(synthetic_ssp_wide, sfh_type):
    """The issue's own reproducer, at z = 0.5 (age(z) = 8.59 Gyr): no warning.

    Before the fix this prints ``SFHBeforeBigBangWarning: ... forms 33% of
    its stellar mass before the Big Bang`` and the oldest lookback with
    SFR > 0 is 13.3 Gyr. After the fix neither symptom reproduces.
    """
    z = 0.5
    m = _build(synthetic_ssp_wide, sfh_type, z)
    params = {}

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        pred = m.predict(params)
    hits = [w for w in rec if issubclass(w.category, SFHBeforeBigBangWarning)]
    assert not hits, f"{sfh_type}: unexpected SFHBeforeBigBangWarning(s): {hits}"

    st = m.predict_state(params)
    lbt_gyr = np.asarray(st.derived["sfh_grid_lbt_yr"]) / 1e9
    sfr = np.asarray(st.derived["sfr_history"])
    age_z = float(age_at_z(z))
    oldest_sf_lbt = float(lbt_gyr[sfr > 0].max())
    assert oldest_sf_lbt <= age_z + 1e-6, (
        f"{sfh_type}: oldest lookback with SFR > 0 ({oldest_sf_lbt:.3f} Gyr) exceeds "
        f"age(z={z}) = {age_z:.3f} Gyr"
    )
    assert np.isfinite(float(pred.stellar_mass)) and float(pred.stellar_mass) > 0.0


@pytest.mark.parametrize("sfh_type", PSB_FAMILIES)
def test_z0_oldest_edge_is_age_of_universe_not_the_old_constant(synthetic_ssp_wide, sfh_type):
    """z = 0: the fixed section's oldest edge is age(0) = 13.7869 Gyr, not 13.7.

    Pins both the shift (new edge != old 13.7 Gyr constant) and that the
    resulting published SFH actually reaches the new, slightly older edge.
    Exercises :func:`psb_continuity_flex` directly (not through the full
    model/SSP grid, whose own oldest template age is an independent 13.80 Gyr
    in :func:`synthetic_ssp_wide` and would otherwise mask the shift): the
    region between the old 13.7 Gyr constant and the new age(0) edge falls
    inside the oldest fixed bin only once ``age_universe_yr`` is supplied.
    """
    from tengri.components.stellar.sfh.nonparametric import psb_continuity_flex

    z = 0.0
    age_z = float(age_at_z(z))
    assert age_z != pytest.approx(PSB_FLEX_DEFAULT_MAX_AGE_GYR, abs=1e-4), (
        "test assumes age_at_z(0) differs measurably from the old 13.7 Gyr constant"
    )
    assert age_z > PSB_FLEX_DEFAULT_MAX_AGE_GYR  # the sampled point must lie between the two

    t_between_yr = jnp.array([0.5 * (PSB_FLEX_DEFAULT_MAX_AGE_GYR + age_z) * 1e9])
    kwargs = dict(log_total_mass=10.0, tlast_gyr=0.2, tflex_gyr=2.0)
    sfr_old_edge = float(psb_continuity_flex(t_between_yr, **kwargs, age_universe_yr=None)[0])
    sfr_new_edge = float(
        psb_continuity_flex(t_between_yr, **kwargs, age_universe_yr=age_z * 1e9)[0]
    )
    assert sfr_old_edge == 0.0, (
        f"{sfh_type}: the pre-#2645 (constant 13.7 Gyr) ladder must already be zero "
        f"between 13.7 Gyr and age(0)"
    )
    assert sfr_new_edge > 0.0, (
        f"{sfh_type}: the fixed section did not extend to age(0) = {age_z:.4f} Gyr "
        f"when age_universe_yr was supplied"
    )

    # The same shift through the full model (sanity check only, not the
    # primary pin above): formed mass must still close to the declared
    # total -- the model's own internal mass-conserving accounting, not a
    # naive external re-integration of sfr_history, which on
    # synthetic_ssp_wide's coarse 25-node grid picks up extra quadrature
    # error now that the new edge sits close to that grid's own oldest
    # template age (see test_psb_suess2022_forward.py's updated tolerance).
    m = _build(synthetic_ssp_wide, sfh_type, z)
    formed = float(m.predict_state({}).derived["log_mstar_formed"])
    assert formed == pytest.approx(10.0, abs=1e-3)


@pytest.mark.parametrize("sfh_type", PSB_FAMILIES)
@pytest.mark.parametrize("age_kernel", AGE_KERNELS)
def test_gradient_wrt_tflex_is_finite(synthetic_ssp_wide, sfh_type, age_kernel):
    """d(stellar_mass)/d(tflex_gyr) stays finite with the age(z)-bounded fixed section.

    ``dsps`` is finite-only here, not finite-and-nonzero: that kernel quantizes
    the flex/fixed boundary onto its own coarse age-bin grid, and a swept
    ``tflex_gyr`` can sit inside the same bin throughout, a pre-existing,
    independently-verified flat direction documented in
    ``tests/physics/gradients/test_sfh_time_parameter_smoothness.py``'s
    ``_KNOWN_GAPS`` (``("psb_flex"/"psb_suess2022", "tflex_gyr", "dsps")``),
    not something #2645 introduced.
    """
    m = _build(synthetic_ssp_wide, sfh_type, 0.5, age_kernel=age_kernel, tflex_free=True)
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    tflex_key = next(k for k in p if k.endswith("tflex_gyr"))

    def loss(pp):
        return jnp.sum(m.predict_state(pp).sed_intrinsic)

    g = jax.grad(loss)(p)
    grad_val = float(g[tflex_key])
    assert np.isfinite(grad_val), f"{sfh_type} {age_kernel}: d(loss)/d(tflex_gyr) is not finite"
    if age_kernel == "cic":
        assert grad_val != 0.0, (
            f"{sfh_type} {age_kernel}: d(loss)/d(tflex_gyr) is identically zero "
            "(finite is not enough -- a collapsed gradient gives a sampler no signal, #2100)"
        )
    # grad-assert: finite-only — dsps quantizes the flex/fixed boundary onto its
    # own coarse age-bin grid, a pre-existing documented flat direction for
    # psb_flex/psb_suess2022.tflex_gyr on this kernel (see docstring above).
