# SPDX-License-Identifier: BSD-3-Clause
"""#2640: every published SFR history is bounded to [0, age(z)] and carries the formed mass.

The stellar component's ``sfr_history`` and the ``SEDModel`` routes
(``predict_sfh``, ``predict_sfh_quantities``) publish ONE history: windowed to
the support by putting a node exactly at ``age(z)`` (nodes beyond collapse onto
it) and rescaled by a single factor, so the PLAIN integral over the published
grid equals ``10**log_mstar_formed`` (float precision on the native grid), no
node and no trapezoid area lies beyond ``age(z)``, and the SFR averages are
unbiased against an independent dense quadrature.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.cosmology import age_at_z

pytestmark = pytest.mark.regression_bug

Z = 0.1
LOG_MASS = 9.0


def _build(ssp, sfh):
    return SEDModel.build(
        ssp_data=ssp,
        sfh=sfh,
        met={"type": "delta", "logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(Z),
    )


def _cases():
    age = float(age_at_z(Z))
    for kernel in ("cic", "dsps"):
        yield pytest.param(
            {
                "type": "dpl",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Fixed(LOG_MASS),
                "age_gyr": Fixed(1.1 * age),
                "age_kernel": kernel,
            },
            {},
            id=f"dpl-{kernel}",
        )
        yield pytest.param(
            {
                "type": "delayed",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Fixed(LOG_MASS),
                "age_gyr": Fixed(1.1 * age),
                "age_kernel": kernel,
            },
            {},
            id=f"delayed-{kernel}",
        )
    xi = np.random.default_rng(3).standard_normal(256)
    yield pytest.param(
        {
            "type": ["dpl", "field"],
            "sfh_dpl_age_gyr": Fixed(1.1 * age),
            "sfh_dpl_log_total_mass": Fixed(LOG_MASS),
            "sfh_dpl_alpha": Fixed(1.0),
            "sfh_dpl_beta": Fixed(1.5),
            "sfh_dpl_tau_gyr": Fixed(3.0),
            "sfh_field_psd_sigma": Fixed(1.0),
            "sfh_field_psd_tau_myr": Fixed(50.0),
        },
        {"sfh_field_xi": xi},
        id="dpl+field-dsps",
    )
    for kernel in ("cic", "dsps"):
        t_myr = np.arange(3000.0)
        t_gyr = age - 3.0 + t_myr / 1e3
        sfr = np.exp(-t_myr / 800.0)
        sfr = sfr / (sfr.sum() * 1e6)
        yield pytest.param(
            {"type": "table", "all_params": Fixed(DEFAULT), "age_kernel": kernel},
            {"sfh_t_gyr": t_gyr, "sfh_sfr": sfr, "sfh_table_log_total_mass": LOG_MASS},
            id=f"table-{kernel}",
        )


@pytest.mark.parametrize(("sfh", "extra"), list(_cases()))
def test_published_histories_are_bounded_rescaled_and_agree(synthetic_ssp_wide, sfh, extra):
    model = _build(synthetic_ssp_wide, sfh)
    age_yr = float(age_at_z(Z)) * 1e9
    st = model.predict_state(extra)
    lbt = np.asarray(st.derived["sfh_grid_lbt_yr"])
    sfr = np.asarray(st.derived["sfr_history"])
    formed = 10 ** float(st.derived["log_mstar_formed"])

    # No node and no trapezoid area beyond the Big Bang: the grid ends at
    # age(z) (nodes beyond it collapse onto it, zero-width cells).
    assert lbt.max() <= age_yr * (1.0 + 1e-12)
    over = lbt > age_yr * (1.0 + 1e-12)
    assert not over.any()
    assert np.all(np.diff(lbt) >= 0.0)

    # The PLAIN trapezoid is the formed mass to round-off.
    np.testing.assert_allclose(np.trapezoid(sfr, lbt), formed, rtol=1e-8)

    # predict_sfh (native grid) is the same history, and its mean is bounded
    # and rescaled to the same mass.
    out = model.predict_sfh(extra, grid="native")
    np.testing.assert_allclose(np.asarray(out["t_gyr"]) * 1e9, lbt, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(out["sfr_full"]), sfr, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(np.trapezoid(np.asarray(out["sfr_mean"]), lbt), formed, rtol=1e-8)


@pytest.mark.parametrize(("sfh", "extra"), [c for c in _cases() if "table" not in c.id])
@pytest.mark.parametrize("n_linear", [200, 1000, 5000])
def test_resampled_linear_grid_is_bounded_and_carries_the_mass(
    synthetic_ssp_wide, sfh, extra, n_linear
):
    """``predict_sfh(grid='linear')``: exactly zero beyond age(z); integral ~ formed.

    The rescaled history is resampled on the redshift-independent axis and
    the OUTPUT grid's own cell-overlap weight of ``[0, age(z)]`` is applied
    (zero at every node beyond), so no interpolation ramp crosses age(z). The
    integral differs from the formed mass only by the resampling error of the
    output grid: < 1e-3 for smooth families at 1000 nodes; a bursty field
    sampled on a coarse grid is genuine sampling error (here < 1e-2 at 1000,
    < 1e-1 at 200 nodes).
    """
    model = _build(synthetic_ssp_wide, sfh)
    age_gyr = float(age_at_z(Z))
    st = model.predict_state(extra)
    formed = 10 ** float(st.derived["log_mstar_formed"])
    out = model.predict_sfh(extra, n_linear=n_linear)
    t = np.asarray(out["t_gyr"])
    for key in ("sfr_mean", "sfr_full"):
        s = np.asarray(out[key])
        assert t[s > 0].max() <= age_gyr, f"{key}: SFR > 0 beyond age(z)"
        np.testing.assert_allclose(
            np.trapezoid(s, t * 1e9), formed, rtol=1e-2 if n_linear >= 1000 else 1e-1
        )


def _dense_averages(model, z, nd=400001):
    """Independent dense quadrature of the SFR averages of the declared-mass history."""
    import jax.numpy as jnp

    p = model._get_internal_params({})
    kw = {
        k: v
        for k, v in p.items()
        if k in model._sfh_internal_names or k in model._sfh_public_names
    }
    age = float(age_at_z(z)) * 1e9
    t = np.concatenate([np.linspace(1.0, age, nd), np.geomspace(age * 1.0000001, 1.4e10, 2000)])
    s = np.asarray(model._sfh_fn(jnp.asarray(t), **kw))
    inside = t <= age
    scale = 10**LOG_MASS / np.trapezoid(s[inside], t[inside])
    out = {}
    for name, win in (("sfr_10myr", 1e7), ("sfr_100myr", 1e8)):
        m = t <= win
        out[name] = scale * np.trapezoid(s[m], t[m]) / win
    return out


@pytest.mark.parametrize("z", [0.5, 2.5, 6.0])
@pytest.mark.parametrize("family", ["dpl", "delayed"])
def test_published_sfr_averages_match_dense_quadrature(synthetic_ssp_wide, family, z):
    """sfr_10myr / sfr_100myr are unbiased against a dense quadrature, also at z = 6.

    Measured bias of the 256-node log grid: <= 1.3e-3 (the grid's own
    quadrature of the history, here 1.2e-3 at z = 6); tolerance is 3x that.
    A zeroed node plus a partial weight at age(z) biased these by +1.4e-2 at z = 6.
    """
    sfh = {
        "type": family,
        "all_params": Fixed(DEFAULT),
        "log_total_mass": Fixed(LOG_MASS),
    }
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        sfh=sfh,
        met={"type": "delta", "logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(z),
    )
    ref = _dense_averages(model, z)
    d = model.predict_state({}).derived
    for name in ("sfr_10myr", "sfr_100myr"):
        assert float(d[name]) == pytest.approx(ref[name], rel=4e-3), name


@pytest.mark.parametrize("z", [0.5, 6.0])
def test_readers_tolerate_the_repeated_abscissa_at_age_of_universe(synthetic_ssp_wide, z):
    """Readers of the published grid (repeated nodes at age(z)) stay correct.

    ``Prediction`` resamples the history onto ``model.age_yr``: zero beyond
    age(z), the history's own value at and just inside it. The
    mass-weighted-metallicity bins (``gradient`` widths) and the SFR
    averages integrate, so zero-width cells contribute nothing.
    """
    import jax.numpy as jnp

    sfh = {"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(LOG_MASS)}
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        sfh=sfh,
        met={"type": "delta", "logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(z),
    )
    age_yr = float(age_at_z(z)) * 1e9
    pred = model.predict({})
    pred._ensure_sfh()
    sfr = np.asarray(pred._cache["sfr"])
    ages = np.asarray(model.age_yr)
    assert np.all(sfr[ages > age_yr] == 0.0)
    assert sfr[ages <= age_yr].min() >= 0.0 and sfr[ages <= age_yr][-1] > 0.0

    d = model.predict_state({}).derived
    t, s = np.asarray(d["sfh_grid_lbt_yr"]), np.asarray(d["sfr_history"])
    for x in (age_yr, age_yr * (1.0 - 1e-6)):
        v = float(jnp.interp(x, t, s))
        assert np.isfinite(v) and v == pytest.approx(s[t <= age_yr][-1], rel=1e-3)
    q = model._predict_sfh_quantities({})
    assert np.isfinite(float(q.mass_weighted_metallicity))
    assert float(q.sfr_100myr) == pytest.approx(float(d["sfr_100myr"]), rel=1e-12)
