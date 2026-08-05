# SPDX-License-Identifier: BSD-3-Clause
"""#1522: a tabulated SFH silently drops mass older than the oldest SSP age bin.

``sfh={'type': 'table'}`` is consumed by interpolating the user's history onto
the **SSP lookback-age grid** (``_tabulated_sfh``, ``component.py``), and the
grid stops at its oldest bin. Any mass the table carries beyond that age has no
bin to land in, so it is dropped — no warning, no NaN, finite photometry that
simply belongs to a lighter galaxy than the one the caller described.

The same axis is guarded at its other end. ``_build_dsps_sfh_table`` clamps and
zeroes SSP ages *older than the universe* at the observation redshift, and the
component raises ``SFHBeforeBigBangWarning`` when it does (#683). One edge of
one axis is loud; the other is silent.

Measured with the real PARSEC/MILES grid (oldest bin 12.589 Gyr), the loss is
zero above z≈0.09 — where cosmic time has not yet outrun the grid — and rises to
46 % at z=0 for an exponential history anchored at the Big Bang. A parametric
SFH on the identical grid is immune: it renormalizes its age weights to
``10**log_total_mass`` after landing them on the grid, so whatever falls off the
end is scaled back in.

Why no existing test caught it: the W6 suite runs on ``synthetic_ssp_wide``,
whose oldest bin is 13.80 Gyr — beyond cosmic time at any z > 0, so the edge is
never in play — and asserts galaxy-to-galaxy **ratios**, which divide out a
common truncation factor exactly. The fixture below is that same fixture with
one number changed: where the oldest age bin sits. No ``data/`` needed, so this
runs on CI rather than only on a laptop that happens to have the real grid.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.conservation

_Z_OBS = 0.05  # cosmic age 13.114 Gyr — comfortably past the grid below
_OLDEST_BIN_GYR = 8.0  # oldest SSP age bin, deliberately inside cosmic time
_LOG_MASS = 10.0
_TAU_GYR = 2.0
_AGE_GYR = 10.0  # formation lookback — puts real mass beyond the oldest bin
_N_T = 512


@pytest.fixture(scope="module")
def truncating_ssp():
    """``synthetic_ssp_wide``, with the oldest age bin moved inside cosmic time.

    Constructed identically to the shipped fixture — same wavelength grid, same
    separable flux, same metallicities — except that ``ssp_lg_age_gyr`` tops out
    at 8 Gyr instead of 13.8 Gyr. That one change is the entire experiment: at
    z=0.05 the universe is 13.114 Gyr old, so a history reaching back to the Big
    Bang now carries mass the grid cannot represent.
    """
    import jax.numpy as jnp

    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    wave = jnp.logspace(2.0, 7.0, 1600)
    lg_age = jnp.linspace(-3.0, float(np.log10(_OLDEST_BIN_GYR)), 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (lg_age - lg_age.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(flux) + 1e-12,
        ssp_lg_age_gyr=lg_age,
        ssp_lgmet=lgmet,
    )


def _build(ssp, obs, sfh):
    from tengri import FIXED, ForwardModel, SEDModel
    from tengri.parameters.priors import Fixed

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sed = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh=sfh,
            dust={
                "type": "two_component",
                "all_params": FIXED,
                "tau_bc": 0.0,
                "tau_diff": 0.0,
            },
            neb={"type": "none"},
            redshift=Fixed(_Z_OBS),
        )
        return ForwardModel.build(sed=sed, observation=obs), sed


def _parametric_sfh():
    """The same delayed-tau SFH the table below is sampled from, all params fixed."""
    from tengri import FIXED

    return {
        "type": "delayed",
        "all_params": FIXED,
        "log_total_mass": _LOG_MASS,
        "tau_gyr": _TAU_GYR,
        "age_gyr": _AGE_GYR,
    }


def _history():
    """The delayed-tau SFH as a (t [Gyr], SFR [Msun/yr]) table, ascending in t.

    ``sfhdelayed`` normalizes by a trapezoid over the lookback grid it is given,
    so build ascending in lookback and flip.
    """
    from tengri.components.stellar.sfh.mean_sfh import sfhdelayed
    from tengri.cosmology import age_at_z

    t_univ = float(age_at_z(_Z_OBS))
    t_lb = np.linspace(0.0, t_univ, _N_T)
    sfr_lb = np.asarray(sfhdelayed(t_lb * 1e9, _LOG_MASS, _TAU_GYR * 1e9, _AGE_GYR * 1e9))
    return (t_univ - t_lb)[::-1], np.maximum(sfr_lb[::-1], 0.0)


def test_the_input_history_carries_the_full_mass():
    """Assert the setup: the table handed in really does integrate to 1e10 Msun.

    Without this, a failure below could be blamed on the history rather than on
    what the model does with it.
    """
    t_gyr, sfr = _history()
    formed = float(np.trapezoid(sfr, t_gyr * 1e9))
    assert np.isclose(formed / 10**_LOG_MASS, 1.0, rtol=1e-6), (
        f"the test history forms {formed:.6e} Msun, not {10**_LOG_MASS:.6e}"
    )


def test_parametric_arm_conserves_mass_on_the_same_truncating_grid(
    truncating_ssp, synthetic_tophat_obs
):
    """The matched control: the grid is not the problem, the representation is.

    Identical SSP, identical SFH, identical redshift — expressed parametrically
    instead of as a table. This arm reports the full 1e10 Msun, because it
    renormalizes its age weights after landing them on the grid. So the mass
    loss below cannot be attributed to the SSP grid being short; it is specific
    to how the tabulated path lands mass on that grid.
    """
    import jax.numpy as jnp

    fwd, sed = _build(truncating_ssp, synthetic_tophat_obs, _parametric_sfh())
    params = {k: jnp.asarray(v) for k, v in fwd.spec.get_fixed_values().items()}
    mass = float(
        np.asarray(sed.predict_properties(params, names=("stellar_mass",))["stellar_mass"])
    )
    assert np.isclose(mass / 10**_LOG_MASS, 1.0, rtol=1e-6), (
        f"parametric arm formed {mass:.6e} Msun on the truncating grid"
    )


def test_tabulated_sfh_conserves_mass_across_the_grid_edge(truncating_ssp, synthetic_tophat_obs):
    """The conservation law: sum(age_weights) must equal the table's own integral.

    ``stellar_mass`` IS the age-weight quadrature (``log_mstar_formed =
    log10(sum(age_weights))``), and the SED is that same weighted sum — so this
    was never a bookkeeping label that happened to disagree: before the fix
    every band fell by the same 0.7246 as the mass, and a caller got a galaxy
    1.4x lighter than the one they simulated with no diagnostic of any kind.

    Fixed by extending the CIC integrand past the oldest template
    (:func:`_extend_integrand_to_history`), so those parcels land on the oldest
    template instead of falling off the grid.
    """
    from tengri import Catalog

    tab_fwd, _sed = _build(truncating_ssp, synthetic_tophat_obs, {"type": "table"})
    par_fwd, _par_sed = _build(truncating_ssp, synthetic_tophat_obs, _parametric_sfh())

    t_gyr, sfr = _history()
    mock = Catalog.from_histories(tab_fwd, t_gyr=t_gyr[None, :], sfr=sfr[None, :]).simulate(
        properties=("stellar_mass",)
    )
    mass = float(np.asarray(mock.properties["stellar_mass"])[0])

    assert np.isclose(mass / 10**_LOG_MASS, 1.0, rtol=1e-3), (
        f"tabulated arm formed {mass:.6e} Msun of the requested "
        f"{10**_LOG_MASS:.6e} — {100 * (1 - mass / 10**_LOG_MASS):.1f} % lost"
    )

    # The photometry must no longer carry the deficit either. It does not come
    # back exactly equal to the parametric arm, and should not: the two adopt
    # different — both defensible — policies for mass the grid cannot represent.
    # The table accumulates it onto the oldest template; the parametric family
    # drops it and renormalizes the rest. Measured, that policy gap is 0.72 %
    # here with 27.5 % of the mass relocated, and it collapses to 0.0022 % on a
    # grid where nothing falls off the end (the round-trip in
    # tests/unit/inference/test_catalog_histories.py measures exactly that), so
    # it scales with the relocated fraction rather than being a residual defect.
    # Before the fix this ratio was 0.7246 — the mass deficit, in light.
    import jax.numpy as jnp

    p_par = {k: jnp.asarray(v) for k, v in par_fwd.spec.get_fixed_values().items()}
    f_par = np.asarray(par_fwd.predict_photometry(p_par))
    f_tab = np.asarray(mock.photometry)[0]
    assert np.allclose(f_tab / f_par, 1.0, rtol=2e-2), f"flux ratio {f_tab / f_par}"


def test_the_remaining_approximation_is_announced(truncating_ssp, synthetic_tophat_obs):
    """Mass is conserved, but those stars wear the oldest template's colors — say so.

    The fix removes the mass loss; it cannot invent templates older than the
    grid. What is left is a color approximation, and #1522's original sin was
    silence, so it must not stay invisible.

    Eager paths only. The check casts to ``float``, which raises
    ``ConcretizationTypeError`` under jit/vmap — so ``Catalog.simulate``, being
    jitted, does not warn. That is the same limitation
    :class:`SFHBeforeBigBangWarning` carries at the other end of this axis, and
    the reason this test drives ``predict_photometry`` directly.
    """
    import jax.numpy as jnp

    from tengri.components.stellar.component import SFHBeyondSSPGridWarning

    tab_fwd, _sed = _build(truncating_ssp, synthetic_tophat_obs, {"type": "table"})
    t_gyr, sfr = _history()
    params = {"sfh_t_gyr": jnp.asarray(t_gyr), "sfh_sfr": jnp.asarray(sfr)}

    with pytest.warns(SFHBeyondSSPGridWarning, match="older than the oldest SSP template"):
        tab_fwd.predict_photometry(params)


def test_a_history_inside_the_grid_does_not_warn(synthetic_ssp_wide, synthetic_tophat_obs):
    """The negative control: no warning when nothing falls off the end.

    Without this the guard above passes on a warning that fires unconditionally,
    which would be its own silent-failure mode — every table catalog crying wolf.
    Same history, same redshift; only the SSP grid's oldest bin moves (13.80 Gyr,
    past cosmic time at z=0.05, so the extension is a no-op).
    """
    import jax.numpy as jnp

    from tengri.components.stellar.component import SFHBeyondSSPGridWarning

    tab_fwd, _sed = _build(synthetic_ssp_wide, synthetic_tophat_obs, {"type": "table"})
    t_gyr, sfr = _history()
    params = {"sfh_t_gyr": jnp.asarray(t_gyr), "sfh_sfr": jnp.asarray(sfr)}

    with warnings.catch_warnings():
        warnings.simplefilter("error", SFHBeyondSSPGridWarning)
        tab_fwd.predict_photometry(params)
