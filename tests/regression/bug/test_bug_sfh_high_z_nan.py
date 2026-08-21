# SPDX-License-Identifier: BSD-3-Clause
"""Regression: SFH must not NaN at high redshift (cosmic-time table monotonicity).

When the star formation history places star formation before the Big Bang
(``age_gyr`` exceeds the age of the universe at ``redshift``), the cosmic-time
table handed to DSPS could become non-monotone — boundary-valid bins clamp to
``T_TABLE_MIN`` *below* the invalid-bin ramp — and DSPS then returns an all-NaN
weight tensor, so ``age_weights`` and the whole SED were NaN.

:func:`enforce_increasing_cosmic_time` projects the table to strictly
increasing (an exact no-op for already-increasing tables), and an eager
:class:`SFHBeforeBigBangWarning` surfaces the truncation so the result is not
silently wrong.

See suchethac/tengri#683.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import FIXED, Fixed
from tengri.components.stellar.component import SFHBeforeBigBangWarning
from tengri.components.stellar.sps.dsps_wrapper import enforce_increasing_cosmic_time

pytestmark = pytest.mark.regression_bug


def test_enforce_increasing_is_noop_for_increasing_table():
    """A table already increasing by more than eps is returned unchanged."""
    t = jnp.linspace(0.01, 13.7, 93)
    out = np.asarray(enforce_increasing_cosmic_time(t))
    np.testing.assert_allclose(out, np.asarray(t), rtol=0.0, atol=1e-9)


def test_enforce_increasing_fixes_non_monotone_table():
    """An inverted/duplicated table (the high-z boundary case) becomes strict."""
    # ramp overshoots (0.015) then dips back to the floor (0.010), as happens
    # when boundary-valid bins clamp to T_TABLE_MIN below the invalid-bin ramp.
    t = jnp.array([0.0102, 0.012, 0.015, 0.010, 0.010, 0.5, 1.0])
    out = np.asarray(enforce_increasing_cosmic_time(t))
    assert np.all(np.diff(out) > 0.0), f"not strictly increasing: {out}"


def _high_z_model(ssp, redshift, age_gyr=3.0, free_mass=False):
    sfh = {
        "type": "dpl",
        "all_params": FIXED,
        "age_gyr": age_gyr,
        "tau_gyr": 5.0,
        "alpha": 2.0,
        "beta": 2.0,
        "log_total_mass": tengri.FREE if free_mass else 10.0,
    }
    return tengri.SEDModel.build(
        ssp_data=ssp,
        sfh=sfh,
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": FIXED,
            "tau_diff": 0.0,
            "tau_bc": 0.0,
        },
        redshift=Fixed(redshift),
    )


def test_high_z_sfh_sed_is_finite(synthetic_ssp_wide):
    """SFH onset older than cosmic age at z=5 must not produce a NaN SED."""
    m = _high_z_model(synthetic_ssp_wide, redshift=5.0, age_gyr=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SFHBeforeBigBangWarning)
        sed = np.asarray(m.predict_rest_sed(m.spec.get_fixed_values()).sed)
    assert np.all(np.isfinite(sed)), "high-z SFH SED must be finite (no NaN/Inf)"
    assert np.nanmax(sed) > 0.0


def test_high_z_gradient_is_finite(synthetic_ssp_wide):
    """Gradients must stay finite at high z so inference can sample z freely."""
    m = _high_z_model(synthetic_ssp_wide, redshift=5.0, age_gyr=3.0, free_mass=True)
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))

    def loss(pp):
        return jnp.nansum(m.predict_state(pp).sed_intrinsic)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SFHBeforeBigBangWarning)
        g = jax.grad(loss)(p)
    assert np.isfinite(float(g["sfh_dpl_log_total_mass"]))


def test_eager_warns_when_sfh_predates_big_bang(synthetic_ssp_wide):
    """Eager forward warns when a non-negligible mass forms before the Big Bang."""
    m = _high_z_model(synthetic_ssp_wide, redshift=5.0, age_gyr=3.0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m.predict_rest_sed(m.spec.get_fixed_values())
    hits = [w for w in rec if issubclass(w.category, SFHBeforeBigBangWarning)]
    assert len(hits) == 1, "expected exactly one SFHBeforeBigBangWarning"
    assert "Big Bang" in str(hits[0].message)


def test_eager_does_not_warn_at_low_redshift(synthetic_ssp_wide):
    """A physical low-z SFH must not warn."""
    m = _high_z_model(synthetic_ssp_wide, redshift=0.05, age_gyr=3.0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m.predict_rest_sed(m.spec.get_fixed_values())
    assert not any(issubclass(w.category, SFHBeforeBigBangWarning) for w in rec)


def test_jit_path_is_silent_and_finite(synthetic_ssp_wide):
    """Under jit the warning is skipped (tracer guard) and the result is finite."""
    m = _high_z_model(synthetic_ssp_wide, redshift=5.0, age_gyr=3.0)
    p = m.spec.get_fixed_values()

    def scalar(pp):
        return jnp.nansum(m.predict_state(pp).sed_intrinsic)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        v = jax.jit(scalar)(p)
    assert not any(issubclass(w.category, SFHBeforeBigBangWarning) for w in rec)
    assert np.isfinite(float(v))
