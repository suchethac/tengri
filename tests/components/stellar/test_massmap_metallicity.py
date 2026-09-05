# SPDX-License-Identifier: BSD-3-Clause
"""massmap_lin and massmap_box metallicity modes.

Verifies:

1. Shape and monotonicity: Z(age) runs from Zstart (oldest) to Zfinal (present)
2. Boundary conditions: Z at the oldest age is Zstart, at present Zfinal
3. Limiting cases: massmap_box -> massmap_lin in the small-enrichment limit,
   and -- the other half of that claim -- the two are *not* the same model
   outside it
4. Gradient safety: autodiff w.r.t. parameters matches finite difference
5. Integration: models build and produce a finite history via SEDModel

The two modes had a class each, five tests apiece, differing only in which
function they called. They share one table now, which also gave massmap_box
the zero-SFR case only massmap_lin had.

Two assertions were doing less than they read as:

* ``test_finite_values`` (once per mode) asserted only ``jnp.all(isfinite)``.
  It is subsumed: a non-finite entry makes ``dz <= tolerance`` false in the
  monotonicity test and fails the ``assert_allclose`` in the boundary test, so
  a NaN or inf cannot reach the end of this file unnoticed.
* ``test_zero_sfr_safe`` asserted finiteness on an all-zero SFH. Measured, both
  modes return exactly ``log_z_final`` at every age -- a claim worth making,
  where "is finite" would also accept a garbage constant. In log space a
  returned array of zeros means Z = 1, i.e. fifty times solar, and is finite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.stellar.sfh.metallicity_history import (
    massmap_box_metallicity,
    massmap_lin_metallicity,
)
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform

pytestmark = pytest.mark.bounds


def fd_grad(f, x: float, eps: float = 1e-5) -> float:
    """Central finite difference gradient: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


#: The two modes share a signature: (lg_age_gyr, ages_yr, sfr, log_z_start,
#: log_z_final). massmap_box takes an optional trailing yield_rho.
_MASSMAP_MODES = [("lin", massmap_lin_metallicity), ("box", massmap_box_metallicity)]
_MODE_IDS = [name for name, _fn in _MASSMAP_MODES]


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def simple_sfh():
    """Simple exponentially declining SFH on a log-age grid.

    Returns
    -------
    ssp_ages_yr : ndarray, shape (n_age,)
        Age in years, ascending lookback order (youngest first).
    sfr_on_ssp : ndarray, shape (n_age,)
        SFR at each age in Msun/yr.
    ssp_lg_age_gyr : ndarray, shape (n_age,)
        log10(age/Gyr) of each age.
    """
    ssp_lg_age_gyr = jnp.linspace(-3.0, 1.114, 20)  # 1 Myr to ~13 Gyr
    ssp_ages_yr = 10.0 ** (ssp_lg_age_gyr + 9.0)
    tau = 5.0e9  # 5 Gyr timescale
    sfr_on_ssp = jnp.exp(-ssp_ages_yr / tau)
    sfr_on_ssp = sfr_on_ssp / jnp.sum(sfr_on_ssp)
    return ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr


@pytest.fixture
def log_z_abs_values():
    """Typical absolute log10(Z) range.

    Returns
    -------
    log_z_start : float
        log10(Z) at oldest age (1e-4 in linear space).
    log_z_final : float
        log10(Z) at present day (0.02 in linear space, roughly solar).
    """
    return jnp.log10(1e-4), jnp.log10(0.02)


# ── Shared by both modes ──────────────────────────────────────────


@pytest.mark.parametrize(("mode", "metallicity_fn"), _MASSMAP_MODES, ids=_MODE_IDS)
def test_output_shape(mode, metallicity_fn, simple_sfh, log_z_abs_values):
    """Output shape matches the input SSP age grid."""
    ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
    log_z_start, log_z_final = log_z_abs_values
    result = metallicity_fn(ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final)
    assert result.shape == ssp_lg_age_gyr.shape


@pytest.mark.parametrize(("mode", "metallicity_fn"), _MASSMAP_MODES, ids=_MODE_IDS)
def test_monotonic_decreasing(mode, metallicity_fn, simple_sfh, log_z_abs_values):
    """Z(age) decreases monotonically from youngest to oldest.

    cmf runs from 1 (present) to 0 (oldest) and Z = Zstart + (Zfinal - Zstart)
    * cmf, so Z must fall from Zfinal at the youngest age to Zstart at the
    oldest.

    This is also where a non-finite entry is caught: ``nan <= tolerance`` is
    False, so NaN or inf anywhere in the history fails here.
    """
    ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
    log_z_start, log_z_final = log_z_abs_values
    result = metallicity_fn(ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final)

    z_linear = 10.0**result
    dz = jnp.diff(z_linear)
    tolerance = 1e-14 * jnp.mean(z_linear)
    assert jnp.all(dz <= tolerance), f"Non-monotonic: dz = {dz}, tolerance = {tolerance}"


@pytest.mark.bounds
@pytest.mark.parametrize(("mode", "metallicity_fn"), _MASSMAP_MODES, ids=_MODE_IDS)
def test_boundary_conditions(mode, metallicity_fn, simple_sfh, log_z_abs_values):
    """Z at the oldest age is Zstart, at present Zfinal."""
    ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
    log_z_start, log_z_final = log_z_abs_values
    result = metallicity_fn(ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final)

    # Oldest age is the last element in ascending lookback order.
    assert_allclose(result[-1], log_z_start, atol=0.1)
    assert_allclose(result[0], log_z_final, atol=0.1)


@pytest.mark.parametrize(("mode", "metallicity_fn"), _MASSMAP_MODES, ids=_MODE_IDS)
def test_gradient_wrt_zfinal(mode, metallicity_fn, simple_sfh, log_z_abs_values):
    """d<Z>/dZfinal matches central finite difference.

    Finiteness alone would not do here: the canonical safe-divide guard writes
    exactly 0.0, and zero is finite.
    """
    ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
    log_z_start, log_z_final = log_z_abs_values

    def fn(z_final):
        return jnp.mean(
            metallicity_fn(ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, z_final)
        )

    grad_auto = float(jax.grad(fn)(log_z_final))
    assert_allclose(grad_auto, fd_grad(fn, log_z_final), rtol=1e-3)

    # Non-vacuity: a detached gradient is 0.0, which no rtol comparison
    # against an equally-zero difference quotient would catch.
    assert abs(grad_auto) > 1e-3, f"{mode}: Zfinal has no effect on the mean, grad={grad_auto}"


@pytest.mark.parametrize(("mode", "metallicity_fn"), _MASSMAP_MODES, ids=_MODE_IDS)
def test_zero_sfr_pins_the_history_at_zfinal(mode, metallicity_fn, log_z_abs_values):
    """With no star formation anywhere, every age carries Zfinal.

    The division by total formed mass is guarded, and the guard's value is the
    thing worth pinning. massmap_box had no zero-SFR case at all before this
    was a table.
    """
    ssp_lg_age_gyr = jnp.linspace(-3.0, 1.114, 10)
    ssp_ages_yr = 10.0 ** (ssp_lg_age_gyr + 9.0)
    sfr_on_ssp = jnp.zeros_like(ssp_ages_yr)
    log_z_start, log_z_final = log_z_abs_values

    result = metallicity_fn(ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final)

    assert_allclose(result, jnp.full_like(result, log_z_final), rtol=1e-6)


# ── massmap_box only ──────────────────────────────────────────────


@pytest.mark.limit
def test_small_enrichment_limit(simple_sfh):
    """When (Zfinal - Zstart) << yield, the box model reduces to the linear one."""
    ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
    log_z_start = jnp.log10(1e-4)
    log_z_final = jnp.log10(1e-4 + 1e-5)  # only 10% enrichment
    yield_rho = 0.03  # large relative to the enrichment

    z_lin = 10.0 ** massmap_lin_metallicity(
        ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final
    )
    z_box = 10.0 ** massmap_box_metallicity(
        ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final, yield_rho
    )

    rel_err = jnp.abs(z_box - z_lin) / (z_lin + 1e-20)
    assert jnp.all(rel_err < 0.1), f"Limit test failed: max rel_err = {jnp.max(rel_err)}"


@pytest.mark.limit
def test_box_and_lin_are_not_the_same_model(simple_sfh, log_z_abs_values):
    """Non-vacuity for the limit above: away from it the two must diverge.

    ``test_small_enrichment_limit`` asserts the two agree where they should.
    On its own that is satisfied by a massmap_box that quietly calls the linear
    path -- the failure mode where a selector accepts a value and ignores it.
    Measured at the fixture's enrichment (1e-4 to 0.02): the two differ by
    0.115 dex, 23% in Z. The 0.05 dex floor here is well clear of that and far
    above numerical noise.
    """
    ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
    log_z_start, log_z_final = log_z_abs_values

    lin = massmap_lin_metallicity(
        ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final
    )
    box = massmap_box_metallicity(
        ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final
    )

    spread = float(jnp.max(jnp.abs(box - lin)))
    assert spread > 0.05, f"massmap_box is indistinguishable from massmap_lin ({spread:.4f} dex)"


def test_yield_effect(simple_sfh, log_z_abs_values):
    """A smaller yield slows metallicity growth (log-linear effect)."""
    ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
    log_z_start, log_z_final = log_z_abs_values

    z_large = 10.0 ** massmap_box_metallicity(
        ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final, yield_rho=0.05
    )
    z_small = 10.0 ** massmap_box_metallicity(
        ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final, yield_rho=0.01
    )

    mid_idx = len(ssp_lg_age_gyr) // 2
    assert float(z_large[mid_idx]) > float(z_small[mid_idx])


# ── Integration via Parameters ────────────────────────────────────


class TestMassmapIntegration:
    """massmap modes integrated into Parameters."""

    @pytest.mark.contract
    def test_massmap_lin_explicit_mode(self):
        """met_mode='massmap_lin' builds with both endpoints free."""
        params = Parameters(
            met_mode="massmap_lin",
            met_logzsol_start=Uniform(-4.0, -2.0),
            met_logzsol_final=Uniform(-2.0, 0.0),
        )
        assert params.met_mode == "massmap_lin"

        free_params = params.free_params
        assert any("met_logzsol_start" in p for p in free_params)
        assert any("met_logzsol_final" in p for p in free_params)

    @pytest.mark.contract
    def test_massmap_box_with_yield(self):
        """met_mode='massmap_box' with an explicit yield.

        met_mode is set explicitly because massmap_box's keys are a superset of
        massmap_lin's, which is otherwise ambiguous.
        """
        params = Parameters(
            met_mode="massmap_box",
            met_logzsol_start=Uniform(-4.0, -2.0),
            met_yield=Fixed(0.03),
        )
        assert params.met_mode == "massmap_box"
        assert any("met_logzsol_start" in p for p in params.free_params)


@pytest.mark.regression_bug
def test_massmap_lin_is_linear_in_Z_not_logZ():
    """massmap_lin maps Z linearly in cumulative mass (ProSpect Zfunc_massmap_lin).

    Regression for the log-vs-linear bug: the original code interpolated in
    log10(Z), giving a *geometric* map (~2x off vs ProSpect at the half-mass
    point). For a constant SFR on a uniform age grid, cmf is linear in age, so Z
    at the mid-age must be the *arithmetic* midpoint (Zstart+Zfinal)/2 -- not the
    geometric mean sqrt(Zstart*Zfinal).
    """
    n = 401
    ages_yr = jnp.linspace(1.0e6, 13.0e9, n)  # uniform grid
    lg_age_gyr = jnp.log10(ages_yr / 1e9)
    sfr = jnp.ones(n)  # constant SFR -> cmf linear in age
    z_start, z_final = 1.0e-4, 2.0e-2
    log_z = np.asarray(
        massmap_lin_metallicity(
            lg_age_gyr, ages_yr, sfr, float(np.log10(z_start)), float(np.log10(z_final))
        )
    )
    z = 10.0**log_z
    z_mid = float(z[n // 2])  # cmf ~ 0.5
    arithmetic = 0.5 * (z_start + z_final)
    geometric = float(np.sqrt(z_start * z_final))

    # The message used to sit in a tuple beside the call -- `(assert_allclose(...),
    # f"...")` -- which builds a 2-tuple and discards it, so it could never be
    # shown. err_msg is the parameter that reaches a failure report.
    assert_allclose(
        z_mid,
        arithmetic,
        rtol=0.02,
        err_msg=(
            f"massmap_lin at half-mass Z={z_mid:.3e} should be the arithmetic midpoint "
            f"{arithmetic:.3e} (ProSpect linear map), not the geometric {geometric:.3e}"
        ),
    )
    # Endpoints: present-day -> Zfinal, oldest -> Zstart.
    assert_allclose(z[0], z_final, rtol=1e-3)
    assert_allclose(z[-1], z_start, rtol=5e-2)


@pytest.mark.regression_bug
def test_massmap_box_builds_via_group_dict_grammar(synthetic_ssp):
    """massmap_box is reachable through SEDModel.build's dict grammar.

    Regression for two coupled bugs: (1) ProSpect's ``yield`` param is a Python
    keyword and could not be a builder group key -- renamed to ``met_yield``;
    (2) inference raised "ambiguous" (massmap_box's keys are a superset of
    massmap_lin's) even when ``met_mode`` was set explicitly. Either one made
    ``SEDModel.build(met={'type': 'massmap_box', ...})`` crash.
    """
    from tengri import DEFAULT, Fixed, SEDModel

    model = SEDModel.build(
        ssp_data=synthetic_ssp,
        met={
            "type": "massmap_box",
            "logzsol_start": Fixed(-2.15),
            "logzsol_final": Fixed(0.15),
            "yield": Fixed(0.03),
            "all_params": Fixed(DEFAULT),
        },
        sfh={"type": "const", "log_total_mass": Fixed(10.0), "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.0),
    )
    state = model.predict_state({})
    z_hist = np.asarray(state.derived["log_metallicity_history"])
    assert np.isfinite(z_hist).all()

    # Monotonic enrichment: present-day (youngest) >= oldest.
    age = np.asarray(state.derived["sfh_grid_lbt_yr"])
    assert z_hist[np.argmin(age)] >= z_hist[np.argmax(age)]
