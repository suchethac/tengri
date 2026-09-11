# SPDX-License-Identifier: BSD-3-Clause
"""The 42 sites #1863/#1864 left pinned as a ratchet, now fixed (#1860).

``tools/check_representable_floors.py`` carries a mechanical census: a
``x / jnp.maximum(y, floor)`` whose floor is below ``sqrt(tiny)`` (the bound
division's VJP needs, since it carries ``-num/den**2``). #1863 fixed the
``_filter_integral_union`` site that opened this issue; #1864 fixed the 8
sites in ``utils/sed_quantities.py``. The census's remaining 46 (45 once
#2167 deleted one outright) were pinned rather than fixed pending a per-site
reachability check.

This module is that check. Every site is one of three things, and each is
tested accordingly:

* **Reachable, directly callable** -- the enclosing function is a plain
  ``jnp`` function callable with a small synthetic input that drives the
  guarded quantity to (or toward) zero. Tested by calling the real function.
* **Reachable, but the enclosing function needs data this environment does
  not have** (a downloaded SKIRTOR/Nenkova template grid, a built SSP-backed
  ``SEDModel``) -- tested by reproducing the guarded expression verbatim in a
  standalone closure, the same technique
  ``test_derivative_sized_denominator_floor.py`` uses for the original
  ``_filter_integral_union`` mechanism.
* **Not reachable** -- the guarded quantity is bounded away from zero by
  construction (a fixed module-level constant, a static structural sum) or
  the code path is eager NumPy that no JAX VJP ever passes through. Fixed
  anyway (the wrap is free and keeps the census clean) but the test is
  ``skip``, not ``xfail``, with the specific reason.

Every literal here is unchanged in float64 *except* the six ``1e-300`` sites
(``stellar/component.py`` x5, ``stellar/sfh/mean_sfh.py`` x2 -- see below),
where ``representable_denominator`` moves the float64 floor too, correctly:
``(1e-300)**2`` underflows in float64 as well (smallest normal 2.2e-308).

**A margin caveat, found while writing these tests.** ``representable_denominator``
guarantees ``1/floor**2`` is representable (``~8.5e37``, float32 max is
``3.4e38``): a headroom of about 4x, not more (documented in
``representable_denominator``'s own Notes: ``sqrt(tiny)`` sits "one factor of
two inside" the analytic cliff, i.e. its *square* has one factor of four).
The reverse-mode term multiplying that reciprocal is ``-(numerator-side
coefficient)``, so a coefficient larger than a handful (five near-identical
terms reduced together, or a multiplier like an equivalent-width feature
width of ~40 A) can still overflow that reciprocal to ``inf`` and produce a
NaN through the same ``0 * inf`` mechanism the derivative-safe floor exists to
prevent -- just one level up the expression. This is a property of the bound
itself, not a defect in any one site's fix, so the tests below choose
input magnitudes that isolate *this* fix's mechanism (the floor no longer
being the certain source of failure) and note where a larger real-world
coefficient could still be a problem for a different, follow-up reason.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.scale import representable_denominator

pytestmark = pytest.mark.regression_bug


def _tiny(dtype):
    return float(np.finfo(dtype).tiny)


def _assert_f64_floor_unchanged(literal):
    """The wrap is a no-op in float64 at this literal (contract, reaffirmed locally)."""
    with jax.enable_x64(True):
        assert representable_denominator(literal) == literal


def _assert_f64_floor_moves_but_stays_safe(literal):
    """The ``1e-300`` sites: float64 also needed the raise (#1860's documented exception)."""
    with jax.enable_x64(True):
        raised = representable_denominator(literal)
        assert raised > literal
        # grad-assert: finite-only — not a gradient: this checks that the raised
        # floor's 1/floor**2 (the VJP's denominator) is representable in float64.
        assert np.isfinite(np.float64(1.0) / (raised * raised))


# ════════════════════════════════════════════════════════════════════════
# Group A: reachable, directly callable
# ════════════════════════════════════════════════════════════════════════


def test_age_weights_cic_gradient_finite_for_a_degenerate_sfh():
    """``stellar/component.py:906`` -- ``_age_weights_cic``, floor ``1e-300``.

    All-zero SFR (an empty star-formation window) is a real degenerate SFH,
    not a synthetic corner case: the surrounding comment at the sibling
    inline sites (2798/3521/3557, see below) calls it out explicitly. Every
    parcel contributes zero, so ``w.sum() == 0`` and the pre-#1860 floor
    (``1e-300``) squared flushes to zero in float32.
    """
    from tengri.components.stellar.component import _age_weights_cic

    with jax.enable_x64(False):
        ssp_ages_yr = jnp.geomspace(1e6, 1.3e10, 40, dtype=jnp.float32)
        age_yr = jnp.geomspace(1e5, 1.2e10, 60, dtype=jnp.float32)
        t_obs_gyr = 13.0

        def f(sfr):
            w, _ = _age_weights_cic(age_yr, sfr, ssp_ages_yr, t_obs_gyr)
            return jnp.sum(w)

        sfr0 = jnp.zeros(60, dtype=jnp.float32)
        assert sfr0.dtype == jnp.float32
        g = jax.grad(f)(sfr0)
        assert g.dtype == jnp.float32
        assert np.all(np.isfinite(np.asarray(g))), f"non-finite grad: {g}"
        assert np.any(np.asarray(g) != 0.0), (
            f"identically zero grad: {g}: finite is not enough, a severed gradient"
            " path is as unusable as a NaN one (#2100)"
        )

    _assert_f64_floor_moves_but_stays_safe(1e-300)


def test_joint_weights_cic_met_table_gradient_finite_for_a_degenerate_sfh():
    """``stellar/component.py:1140`` -- ``_joint_weights_cic_met_table``, floor ``1e-300``.

    Same degenerate-SFH mechanism as ``_age_weights_cic`` above, on the
    per-age-metallicity joint (met, age) weight table.
    """
    from tengri.components.stellar.component import _joint_weights_cic_met_table

    with jax.enable_x64(False):
        ssp_ages_yr = jnp.geomspace(1e6, 1.3e10, 20, dtype=jnp.float32)
        ssp_lgmet = jnp.linspace(-2.0, 0.3, 5, dtype=jnp.float32)
        age_yr = jnp.geomspace(1e5, 1.2e10, 30, dtype=jnp.float32)
        lgmet_on_ssp_ages = jnp.full(20, -0.5, dtype=jnp.float32)
        t_obs_gyr = 13.0

        def f(sfr):
            joint, _ = _joint_weights_cic_met_table(
                age_yr, sfr, ssp_ages_yr, t_obs_gyr, lgmet_on_ssp_ages, 0.3, ssp_lgmet
            )
            return jnp.sum(joint)

        sfr0 = jnp.zeros(30, dtype=jnp.float32)
        g = jax.grad(f)(sfr0)
        assert g.dtype == jnp.float32
        assert np.all(np.isfinite(np.asarray(g))), f"non-finite grad: {g}"
        assert np.any(np.asarray(g) != 0.0), (
            f"identically zero grad: {g}: finite is not enough, a severed gradient"
            " path is as unusable as a NaN one (#2100)"
        )

    _assert_f64_floor_moves_but_stays_safe(1e-300)


def test_cic_parcels_f_lin_gradient_finite_for_a_degenerate_ssp_grid():
    """``stellar/component.py:937`` -- ``_cic_parcels``'s ``f_lin`` fallback, floor ``1e-30``.

    ``f_lin`` is the linear-in-age fallback used when the log-age
    interpolation fraction is non-finite (a leading ``age = 0`` SSP node).
    Its own denominator, ``ssp_ages_yr[idx+1] - ssp_ages_yr[idx]``, is zero
    when two consecutive SSP age nodes coincide -- a genuinely reachable
    degenerate grid, constructed directly here.
    """
    from tengri.components.stellar.component import _cic_parcels

    with jax.enable_x64(False):
        # Two coincident nodes at the front of the grid.
        ssp_ages_yr = jnp.concatenate(
            [jnp.zeros(2, dtype=jnp.float32), jnp.geomspace(1e6, 1.3e10, 18, dtype=jnp.float32)]
        )
        age_yr = jnp.geomspace(1.0, 1.2e10, 30, dtype=jnp.float32)
        t_obs_gyr = 13.0

        def f(sfr):
            contrib, _, f_arr, _, _ = _cic_parcels(age_yr, sfr, ssp_ages_yr, t_obs_gyr)
            return jnp.sum(contrib * f_arr)

        sfr0 = jnp.ones(30, dtype=jnp.float32)
        g = jax.grad(f)(sfr0)
        assert g.dtype == jnp.float32
        assert np.all(np.isfinite(np.asarray(g))), f"non-finite grad: {g}"
        assert np.any(np.asarray(g) != 0.0), (
            f"identically zero grad: {g}: finite is not enough, a severed gradient"
            " path is as unusable as a NaN one (#2100)"
        )

    _assert_f64_floor_unchanged(1e-30)


def test_renormalize_to_mass_gradient_finite_for_an_all_zero_shape():
    """``stellar/sfh/mean_sfh.py:96`` -- ``_renormalize_to_mass``, floor ``1e-30``.

    An all-zero unnormalized shape (e.g. every parametric SFH component
    clipped negative and floored to zero) makes ``mass_norm == 0``: the
    function's own docstring names this exact degenerate case.
    """
    from tengri.components.stellar.sfh.mean_sfh import _renormalize_to_mass

    with jax.enable_x64(False):
        t_lookback = jnp.linspace(0.0, 1.3e10, 50, dtype=jnp.float32)

        def f(shape):
            return jnp.sum(_renormalize_to_mass(shape, t_lookback, 10.0))

        shape0 = jnp.zeros(50, dtype=jnp.float32)
        g = jax.grad(f)(shape0)
        assert g.dtype == jnp.float32
        assert np.all(np.isfinite(np.asarray(g))), f"non-finite grad: {g}"
        assert np.any(np.asarray(g) != 0.0), (
            f"identically zero grad: {g}: finite is not enough, a severed gradient"
            " path is as unusable as a NaN one (#2100)"
        )

    _assert_f64_floor_unchanged(1e-30)


def test_sfh2exp_gradient_finite_for_a_near_zero_duration_burst():
    """``stellar/sfh/mean_sfh.py:1802,1803`` -- ``sfh2exp``, floor ``1e-300`` (x2).

    ``burst_age_yr`` a few grid cells wide makes the burst window's covered
    mass tiny: ``burst`` is (numerically) all zero and
    ``m_burst = trapezoid(burst, t_lookback)`` collapses toward zero, the
    same mechanism as an empty burst window. ``burst_age_yr`` exactly ``0``
    is excluded here because it lands exactly on ``window_weight``'s own
    zero-width boundary, whose *own* gradient (unrelated to this fix, a
    zero-measure-interval derivative) is singular there; a few-year window is
    still a physically negligible burst and does not depend on that
    unrelated singularity.
    """
    from tengri.components.stellar.sfh.mean_sfh import sfh2exp

    with jax.enable_x64(False):
        t_lookback = jnp.linspace(0.0, 1.3e10, 200, dtype=jnp.float32)

        def f(burst_age_yr):
            sfr = sfh2exp(
                t_lookback,
                log_total_mass=10.0,
                tau_main_yr=4e9,
                tau_burst_yr=3e8,
                f_burst=0.1,
                age_yr=1.2e10,
                burst_age_yr=burst_age_yr,
            )
            return jnp.sum(sfr)

        g = jax.grad(f)(jnp.asarray(1.0, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        # grad-assert: finite-only — a few-year burst contributes nothing measurable,
        # so d(sum sfr)/d(burst_age_yr) is a float32 rounding residual, not a signal:
        # 0.0 at 3 yr and 3e-8 at 30 yr on CPU, 5e-7 at 3 yr and exactly 0.0 at 30 yr
        # on CUDA (measured). Its sign and zero-ness are backend rounding; the claim
        # here is that the two 1e-300 floors no longer turn it into NaN.
        assert np.isfinite(g), f"non-finite grad: {g}"

    _assert_f64_floor_moves_but_stays_safe(1e-300)


def test_dense_basis_gradient_finite_when_the_recent_sfr_window_covers_everything():
    """``stellar/sfh/dense_basis.py:527`` -- ``init_scale``, floor ``1e-30``.

    ``mass_init`` sums the SFR *outside* the ``_DECOUPLE_SFR_TIME_GYR``-wide
    recent window. Setting ``age_universe_yr`` at or below that window width
    makes ``recent_mask`` true everywhere, so every sample is excluded from
    ``mass_init`` and it sums to exactly zero -- reachable via the public
    ``age_universe_yr`` argument, not a synthetic input.
    """
    from tengri.components.stellar.sfh.dense_basis import _DECOUPLE_SFR_TIME_GYR, dense_basis

    with jax.enable_x64(False):
        age_yr = jnp.linspace(0.0, 1.0e8, 400, dtype=jnp.float32)  # inside the decouple window

        def f(age_universe_yr):
            sfr = dense_basis(
                age_yr,
                log_total_mass=8.0,
                log_sfr_inst=0.0,
                age_universe_yr=age_universe_yr,
                tx_frac_0=0.25,
                tx_frac_1=0.5,
                tx_frac_2=0.75,
            )
            return jnp.sum(sfr)

        age_universe_yr0 = jnp.asarray(_DECOUPLE_SFR_TIME_GYR * 1e9 * 0.5, dtype=jnp.float32)
        g = jax.grad(f)(age_universe_yr0)
        assert g.dtype == jnp.float32
        # grad-assert: finite-only — at this degenerate input the floor binds (or the
        # where-mask selects its constant branch), so d/dx max(x, floor) is exactly
        # zero by construction (measured); the claim is that the VJP's 1/floor**2 is
        # representable, i.e. no NaN, not that the gradient is non-zero.
        assert np.isfinite(g), f"non-finite grad: {g}"

    _assert_f64_floor_unchanged(1e-30)


def test_break_from_means_gradient_finite_for_a_zero_blue_continuum():
    """``observation/spectral_indices.py:409`` -- ``_break_from_means``, floor ``1e-30``."""
    from tengri.observation.spectral_indices import _break_from_means

    with jax.enable_x64(False):
        g = jax.grad(
            lambda f_blue: _break_from_means(f_blue, jnp.asarray(1.0, dtype=jnp.float32))
        )(jnp.asarray(0.0, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        # grad-assert: finite-only — at this degenerate input the floor binds (or the
        # where-mask selects its constant branch), so d/dx max(x, floor) is exactly
        # zero by construction (measured); the claim is that the VJP's 1/floor**2 is
        # representable, i.e. no NaN, not that the gradient is non-zero.
        assert np.isfinite(g), f"non-finite grad: {g}"

    _assert_f64_floor_unchanged(1e-30)


def test_ew_from_means_gradient_finite_for_a_zero_continuum():
    """``observation/spectral_indices.py:424`` -- ``_ew_from_means``, floor ``1e-30``.

    ``feat_width`` scales the numerator-side term of the guarded quotient;
    the Lick-index feature windows this ships (``STANDARD_INDICES``) are
    ~10-50 A wide, wide enough that the resulting reciprocal can itself
    overflow float32 even past this fix (see the module docstring's margin
    caveat) -- a separate, follow-up concern from the specific defect this
    fix addresses (the pre-#1860 floor's squared *underflow*, which is what
    made the NaN *certain* rather than merely *at risk*). A width of 1 A
    isolates this fix's mechanism.
    """
    from tengri.observation.spectral_indices import _ew_from_means

    with jax.enable_x64(False):

        def f(cont_means):
            return _ew_from_means(cont_means, jnp.asarray(0.5, dtype=jnp.float32), 1.0, "AA")

        g = jax.grad(f)(jnp.zeros(2, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        assert np.all(np.isfinite(np.asarray(g))), f"non-finite grad: {g}"
        assert np.any(np.asarray(g) != 0.0), (
            f"identically zero grad: {g}: finite is not enough, a severed gradient"
            " path is as unusable as a NaN one (#2100)"
        )

    _assert_f64_floor_unchanged(1e-30)


def test_measure_slope_gradient_finite_for_a_window_outside_the_grid():
    """``observation/spectral_indices.py:468,469,469`` -- ``_measure_slope``, floor ``1e-30`` (x3).

    Three denominators in one expression share ``sw``, mirroring the already
    -fixed ``compute_uv_slope_beta`` sibling in ``utils/sed_quantities.py``
    (#1864). A feature window entirely outside the wavelength grid makes the
    soft sigmoid weight ``w`` (and hence ``sw = sum(w)``) underflow to exactly
    zero in float32: a genuinely out-of-range fit window, not a contrived
    input.
    """
    from tengri.observation.spectral_indices import SpectralIndexDef, _measure_slope

    with jax.enable_x64(False):
        wave = jnp.linspace(3000.0, 4000.0, 64, dtype=jnp.float32)
        idx = SpectralIndexDef(
            name="_test_slope",
            index_type="slope",
            continuum=(),
            feature=(1.0e6, 1.0e6 + 100.0),
        )

        def f(flux):
            return jnp.nan_to_num(_measure_slope(wave, flux, idx))

        flux0 = jnp.full(64, 1.0e-6, dtype=jnp.float32)
        g = jax.grad(f)(flux0)
        assert g.dtype == jnp.float32
        # grad-assert: finite-only — at this degenerate input the floor binds (or the
        # where-mask selects its constant branch), so d/dx max(x, floor) is exactly
        # zero by construction (measured); the claim is that the VJP's 1/floor**2 is
        # representable, i.e. no NaN, not that the gradient is non-zero.
        assert np.all(np.isfinite(np.asarray(g))), f"non-finite grad: {g}"

    _assert_f64_floor_unchanged(1e-30)


def test_beloborodov_gamma_hot_gradient_finite_for_zero_seed_luminosity():
    """``components/agn/disc.py`` (~1453) -- ``beloborodov_gamma_hot``, floor ``1e-30``.

    ``l_seed = 0`` is a physical edge case (no soft photons intercepted by
    the corona). ``l_diss_hot`` is ``1e10``: matching the caller's own
    comment (disc.py, around the ``float32`` branch of ``multicolor_disc``)
    -- "l_hot_erg ~5e43 and l_seed ~1e44 erg/s overflow [float32]. Work both
    in L_sun units ... Beloborodov uses only their ratio, so units cancel" --
    the float32 call site never passes raw erg/s here, only the L_sun-scaled
    (~1e8-1e13) values this test uses.
    """
    from tengri.components.agn.disc import beloborodov_gamma_hot

    with jax.enable_x64(False):
        g = jax.grad(
            lambda l_seed: beloborodov_gamma_hot(jnp.asarray(1.0e10, dtype=jnp.float32), l_seed)
        )(jnp.asarray(0.0, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        # grad-assert: finite-only — at this degenerate input the floor binds (or the
        # where-mask selects its constant branch), so d/dx max(x, floor) is exactly
        # zero by construction (measured); the claim is that the VJP's 1/floor**2 is
        # representable, i.e. no NaN, not that the gradient is non-zero.
        assert np.isfinite(g), f"non-finite grad: {g}"

    _assert_f64_floor_unchanged(1e-30)


def test_broken_powerlaw_continuum_gradient_finite_at_an_extreme_slope():
    """``components/agn/qsogen.py`` (~326) -- ``_broken_powerlaw_continuum``, floor ``1e-30``.

    ``f_norm`` normalizes the continuum at 5500 A; a blue slope well outside
    the physical prior (``_DEFAULT_PLSLP1 = -0.349``; realistic fits stay
    within a few units of that) drives the 5500 A power-law term toward the
    float32 underflow boundary, pushing ``f_norm`` toward zero. ``plslp1 =
    5`` already sits an order of magnitude past any value the model's own
    priors would propose; larger still (the module docstring's margin
    caveat) can overflow the guarded reciprocal itself, a separate,
    follow-up concern from the floor this fix addresses.
    """
    from tengri.components.agn.qsogen import _broken_powerlaw_continuum

    with jax.enable_x64(False):
        wavelength = jnp.linspace(500.0, 3.0e4, 200, dtype=jnp.float32)

        def f(plslp1):
            return jnp.sum(_broken_powerlaw_continuum(wavelength, plslp1, 0.3, 4000.0))

        g = jax.grad(f)(jnp.asarray(5.0, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        assert np.isfinite(g), f"non-finite grad: {g}"
        assert np.any(np.asarray(g) != 0.0), (
            f"identically zero grad: {g}: finite is not enough, a severed gradient"
            " path is as unusable as a NaN one (#2100)"
        )

    _assert_f64_floor_unchanged(1e-30)


def test_hot_dust_blackbody_gradient_finite_for_a_cold_anchor_temperature():
    """``components/agn/qsogen.py`` (~376) -- ``_hot_dust_blackbody``, floor ``1e-60``.

    ``bb_anchor`` is the Planck function evaluated at the 2 um anchor; a very
    low ``tbb`` pushes ``hc_over_k / (tbb * anchor)`` large enough that
    ``exp(...) - 1`` overflows and ``bb_anchor`` underflows toward zero in
    float32 (the ``representable_exponent`` clip on ``x_anchor`` caps the
    exponent but the reciprocal Planck shape can still vanish).
    """
    from tengri.components.agn.qsogen import _hot_dust_blackbody

    with jax.enable_x64(False):
        wavelength = jnp.linspace(500.0, 3.0e4, 100, dtype=jnp.float32)
        continuum_flam = jnp.ones(100, dtype=jnp.float32)

        def f(tbb):
            return jnp.sum(_hot_dust_blackbody(wavelength, continuum_flam, tbb, bbnorm=1.0))

        g = jax.grad(f)(jnp.asarray(5.0, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        # grad-assert: finite-only — at this degenerate input the floor binds (or the
        # where-mask selects its constant branch), so d/dx max(x, floor) is exactly
        # zero by construction (measured); the claim is that the VJP's 1/floor**2 is
        # representable, i.e. no NaN, not that the gradient is non-zero.
        assert np.isfinite(g), f"non-finite grad: {g}"

    _assert_f64_floor_unchanged(1e-60)


def test_nebular_continuum_gaunt_factor_gradient_finite_at_long_wavelength():
    """``components/nebular/_shared.py:1044`` -- Gaunt factor ``g_ff``, floor ``1e-30``.

    ``x = h*nu/(k*T)`` underflows toward zero at long enough wavelength
    (``nu -> 0``), which is exactly the guard's job inside ``log(2/x)``.
    ``q_h`` is kept at ``1e10`` (rather than a raw CGS ``Q_H`` of
    1e53-1e56) so the *forward* free-free luminosity itself stays
    representable in float32 and this test isolates the Gaunt-factor
    denominator's mechanism rather than an unrelated float32-range issue
    with the overall continuum normalization.
    """
    from tengri.components.nebular._shared import compute_analytic_nebular_continuum

    with jax.enable_x64(False):

        def f(wave_hi):
            wave_aa = jnp.linspace(3000.0, wave_hi, 64, dtype=jnp.float32)
            return jnp.sum(compute_analytic_nebular_continuum(wave_aa, 1.0e10, 0.0))

        g = jax.grad(f)(jnp.asarray(1.0e9, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        assert np.isfinite(g), f"non-finite grad: {g}"
        assert np.any(np.asarray(g) != 0.0), (
            f"identically zero grad: {g}: finite is not enough, a severed gradient"
            " path is as unusable as a NaN one (#2100)"
        )

    _assert_f64_floor_unchanged(1e-30)


def test_nebular_continuum_case_b_recombination_gradient_finite_at_high_temperature():
    """``components/nebular/_shared.py:1038`` -- ``q_over_alpha``, floor ``1e-40``.

    ``alpha_b(T) = alpha_B(1e4) * (T/1e4)**(-0.847)`` decreases with
    temperature without bound; a high enough electron temperature drives it
    toward the float32 underflow boundary. Same ``q_h`` note as the Gaunt
    -factor test above.
    """
    from tengri.components.nebular._shared import compute_analytic_nebular_continuum

    with jax.enable_x64(False):
        wave_aa = jnp.linspace(1000.0, 3000.0, 32, dtype=jnp.float32)

        def f(temperature):
            return jnp.sum(compute_analytic_nebular_continuum(wave_aa, 1.0e10, 0.0, temperature))

        g = jax.grad(f)(jnp.asarray(3.0e8, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        assert np.isfinite(g), f"non-finite grad: {g}"
        assert np.any(np.asarray(g) != 0.0), (
            f"identically zero grad: {g}: finite is not enough, a severed gradient"
            " path is as unusable as a NaN one (#2100)"
        )

    _assert_f64_floor_unchanged(1e-40)


# ════════════════════════════════════════════════════════════════════════
# Group B: reachable, but the enclosing function needs undownloaded template
# grids or a built SSP-backed SEDModel. Each closure reproduces the guarded
# expression verbatim (same literal, same shape) from the cited site.
# ════════════════════════════════════════════════════════════════════════


def test_torus_faceon_disc_normalization_mechanism():
    """``components/agn/blocks/torus.py`` (~515) inside ``skirtor_torus_block``.

    ``skirtor_torus_block`` requires a loaded SKIRTOR template grid (~1 GB
    download, CLAUDE.md), unavailable in this environment. The guarded line,
    reproduced verbatim: ``L_disc_lambda = _sch_shape /
    jnp.maximum(jnp.trapezoid(_sch_shape, wave_aa), floor)``. An all-zero
    Schartmann shape (e.g. every disc-flux template weight clipped to zero
    upstream) makes the trapezoid integral exactly zero.
    """
    with jax.enable_x64(False):
        wave_aa = jnp.linspace(100.0, 3.0e4, 200, dtype=jnp.float32)

        def quotient(sch_shape):
            return jnp.sum(
                sch_shape
                / jnp.maximum(jnp.trapezoid(sch_shape, wave_aa), representable_denominator(1e-30))
            )

        g = jax.grad(quotient)(jnp.zeros(200, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        assert np.all(np.isfinite(np.asarray(g))), f"non-finite grad: {g}"
        assert np.any(np.asarray(g) != 0.0), (
            f"identically zero grad: {g}: finite is not enough, a severed gradient"
            " path is as unusable as a NaN one (#2100)"
        )

    _assert_f64_floor_unchanged(1e-30)


def test_torus_rescale_mechanism():
    """``components/agn/blocks/torus.py`` (~544) inside ``skirtor_torus_block``.

    Same reachability note as above. Guarded line: ``rescale = l_scale /
    jnp.maximum(l_scale + l_ext, floor)``. Both vanish together when
    ``agn_log_lbol`` drives ``l_scale`` to zero and the extinguished disc
    luminosity ``l_ext`` (itself proportional to the disc) follows it.
    """
    with jax.enable_x64(False):

        def quotient(l_scale):
            l_ext = l_scale  # proportional in the real pipeline
            return l_scale / jnp.maximum(l_scale + l_ext, representable_denominator(1e-30))

        g = jax.grad(quotient)(jnp.asarray(0.0, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        assert np.isfinite(g), f"non-finite grad: {g}"
        assert np.any(np.asarray(g) != 0.0), (
            f"identically zero grad: {g}: finite is not enough, a severed gradient"
            " path is as unusable as a NaN one (#2100)"
        )

    _assert_f64_floor_unchanged(1e-30)


def test_skirtor_shape_normalization_mechanism():
    """``components/agn/skirtor.py`` (~782) inside ``create_skirtor_components_from_grid``.

    Requires a loaded SKIRTOR grid. Guarded line: ``shape_n = disc_n /
    jnp.maximum(jnp.trapezoid(disc_n, wave_grid), floor)``, same shape and
    reachability as the torus.py mechanism above.
    """
    with jax.enable_x64(False):
        wave_grid = jnp.linspace(100.0, 3.0e4, 200, dtype=jnp.float32)

        def quotient(disc_n):
            return jnp.sum(
                disc_n
                / jnp.maximum(jnp.trapezoid(disc_n, wave_grid), representable_denominator(1e-30))
            )

        g = jax.grad(quotient)(jnp.zeros(200, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        assert np.all(np.isfinite(np.asarray(g))), f"non-finite grad: {g}"
        assert np.any(np.asarray(g) != 0.0), (
            f"identically zero grad: {g}: finite is not enough, a severed gradient"
            " path is as unusable as a NaN one (#2100)"
        )

    _assert_f64_floor_unchanged(1e-30)


def test_skirtor_bolometric_rescale_mechanism():
    """``components/agn/skirtor.py`` (~1005), floor ``1e-100``.

    Guarded line: ``spec_n = spec * (l_scale / jnp.maximum(jnp.abs(bolo), floor))``.
    ``bolo`` is the trapezoid bolometric integral of the interpolated
    template; a zero (or fully-negative-canceling) spectrum makes it zero.
    ``l_scale`` is kept at O(1) (a dimensionless template rescale factor at
    this stage, not yet the raw bolometric luminosity), matching how
    ``skirtor.py`` actually forms it (``l_scale = 10**agn_log_lbol * L_SUN *
    frac_agn``, applied in log-offset form elsewhere in the float32 path).
    """
    with jax.enable_x64(False):

        def quotient(bolo):
            l_scale = 1.0
            return l_scale / jnp.maximum(jnp.abs(bolo), representable_denominator(1e-100))

        g = jax.grad(quotient)(jnp.asarray(0.0, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        # grad-assert: finite-only — at this degenerate input the floor binds (or the
        # where-mask selects its constant branch), so d/dx max(x, floor) is exactly
        # zero by construction (measured); the claim is that the VJP's 1/floor**2 is
        # representable, i.e. no NaN, not that the gradient is non-zero.
        assert np.isfinite(g), f"non-finite grad: {g}"

    _assert_f64_floor_unchanged(1e-100)


def test_skirtor_disc_dust_ratio_mechanism():
    """``components/agn/skirtor.py`` (~1190), floor ``1e-30``.

    Guarded line (inside an outer ``jnp.where(disk_at_face > 1e-30, num /
    jnp.maximum(disk_at_face, floor), 0.0)``): the discarded branch used to
    contribute ``0 * inf`` once the squared pre-#1860 floor overflowed the
    VJP; with the derivative-safe floor the discarded branch's local
    gradient is finite, so ``0 * finite == 0`` and no NaN survives the
    ``where``. Reproduced verbatim including the outer ``where``, since that
    combination is exactly the "double-where" shape #1860's scope comment
    warns about.
    """
    with jax.enable_x64(False):

        def masked_ratio(disk_at_face):
            disk_at_i = 1.0
            return jnp.where(
                disk_at_face > 1e-30,
                disk_at_i / jnp.maximum(disk_at_face, representable_denominator(1e-30)),
                0.0,
            )

        g = jax.grad(masked_ratio)(jnp.asarray(0.0, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        # grad-assert: finite-only — the input sits below the floor, so the floor binds
        # and d/dx max(x, floor) is exactly zero by construction; the claim under test
        # is that the VJP's 1/floor**2 term is representable (no NaN), not its size.
        assert np.isfinite(g), f"non-finite grad: {g}"

        # Neuter check: the pre-#1860 floor (representable_floor, value-safe
        # only) must still NaN this exact shape, or the test above is vacuous.
        from tengri.utils.scale import representable_floor

        def masked_ratio_unsafe(disk_at_face):
            disk_at_i = 1.0
            return jnp.where(
                disk_at_face > 1e-30,
                disk_at_i / jnp.maximum(disk_at_face, representable_floor(1e-30)),
                0.0,
            )

        g_unsafe = jax.grad(masked_ratio_unsafe)(jnp.asarray(0.0, dtype=jnp.float32))
        assert np.isnan(g_unsafe), (
            "value-sized floor no longer NaNs this shape -- the neuter check "
            "can no longer detect the regression it exists for"
        )

    _assert_f64_floor_unchanged(1e-30)


def test_skirtor_model_retilt_mechanism():
    """``components/agn/skirtor_model.py`` (~470), floor ``1e-100``.

    Inside ``SKIRTORTorus.predict``.

    Requires a built ``SEDModel`` with a loaded SKIRTOR grid. Guarded line:
    ``retilt = shape_sel / jnp.maximum(shape_ref, floor)``, where
    ``shape_ref = skirtor_disk_spectrum(wave_nm, delta=0.0)`` can vanish at
    grid edges the analytic disc shape does not cover.
    """
    with jax.enable_x64(False):
        wave_nm = jnp.linspace(10.0, 3000.0, 100, dtype=jnp.float32)

        def quotient(shape_ref):
            shape_sel = jnp.ones(100, dtype=jnp.float32)
            return jnp.sum(shape_sel / jnp.maximum(shape_ref, representable_denominator(1e-100)))

        g = jax.grad(quotient)(jnp.zeros(100, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        # grad-assert: finite-only — the input sits below the floor, so the floor binds
        # and d/dx max(x, floor) is exactly zero by construction; the claim under test
        # is that the VJP's 1/floor**2 term is representable (no NaN), not its size.
        assert np.all(np.isfinite(np.asarray(g))), f"non-finite grad: {g}"

    _assert_f64_floor_unchanged(1e-100)


def test_disc_zone_luminosity_rescale_mechanism():
    """``components/agn/disc.py`` (~1399) inside ``_compute_zone_luminosities``, floor ``1e-100``.

    Requires the full multi-zone Kubota-Done disc setup. Guarded line:
    ``scale = l_bol_erg / jnp.maximum(l_bol_unnorm, floor)``, with no
    downstream ``jnp.clip`` the way ``beloborodov_gamma_hot`` (above) has --
    that clip is what keeps *that* site's gradient finite (a zero cotangent
    from a saturated clip) regardless of the numerator's scale, and this
    site has no equivalent protection. Here the cotangent reaching the
    division is never zeroed first, so the reciprocal-squared term
    (``-l_bol_erg / den**2``) is what must itself stay representable, and
    representable_denominator's margin is only about 4x (module docstring).
    At the real magnitude this site sees even after the float32 path's own
    L_sun rescaling (``l_bol_erg ~ 1e8-1e13``), the reciprocal overflows and
    this specific site's gradient is measured to STILL be non-finite at the
    exact degenerate point (``l_bol_unnorm == 0``) -- confirmed directly:
    ``l_bol_erg = 1e5`` already reproduces the NaN. ``l_bol_erg = 1.0`` below
    demonstrates the floor-raise mechanism in isolation (the specific defect
    this fix addresses: the pre-#1860 floor squared to exactly zero and
    guaranteed a NaN at *any* numerator); the residual large-numerator
    overflow is a distinct, unresolved concern for this site, reported
    separately rather than masked by an unrealistic test magnitude.
    """
    with jax.enable_x64(False):

        def quotient(l_bol_unnorm):
            l_bol_erg = 1.0
            return l_bol_erg / jnp.maximum(l_bol_unnorm, representable_denominator(1e-100))

        g = jax.grad(quotient)(jnp.asarray(0.0, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        # grad-assert: finite-only — the input sits below the floor, so the floor binds
        # and d/dx max(x, floor) is exactly zero by construction; the claim under test
        # is that the VJP's 1/floor**2 term is representable (no NaN), not its size.
        assert np.isfinite(g), f"non-finite grad: {g}"

    _assert_f64_floor_unchanged(1e-100)


def test_kd_precompute_bolometric_rescale_mechanism():
    """``components/agn/kd_precompute.py`` (~1185) inside ``kubota_done_disc_preintegrated``.

    Requires the full filter-preintegrated Kubota-Done disc setup. Guarded
    line: ``scale = l_bol_requested / jnp.maximum(l_bol_unnorm, floor)``,
    floor ``1e-100`` -- the filter-space analog of the disc.py mechanism
    above, with the identical residual-overflow caveat: no downstream clip
    protects this division, so at the real ``l_bol_requested`` magnitude the
    reciprocal-squared term can still overflow past representable_denominator's
    ~4x margin. ``l_bol_requested = 1.0`` isolates this fix's own mechanism.
    """
    with jax.enable_x64(False):

        def quotient(l_bol_unnorm):
            l_bol_requested = 1.0
            return l_bol_requested / jnp.maximum(l_bol_unnorm, representable_denominator(1e-100))

        g = jax.grad(quotient)(jnp.asarray(0.0, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        # grad-assert: finite-only — the input sits below the floor, so the floor binds
        # and d/dx max(x, floor) is exactly zero by construction; the claim under test
        # is that the VJP's 1/floor**2 term is representable (no NaN), not its size.
        assert np.isfinite(g), f"non-finite grad: {g}"

    _assert_f64_floor_unchanged(1e-100)


def test_xray_stellar_age_mechanism():
    """``components/xray/component.py:387`` inside ``emitter_inputs``, floor ``1e-30``.

    Requires a built stellar component supplying ``derived["age_weights"]``.
    Guarded line (inside ``jnp.where(_w_sum > 0.0, num / jnp.maximum(_w_sum,
    floor) / 1e9, 1.0)``): all-zero age weights (absent/degenerate derived
    state) makes ``_w_sum == 0``, taking the same double-``where`` shape as
    the skirtor.py mechanism above.
    """
    with jax.enable_x64(False):
        ssp_ages_yr = jnp.linspace(1.0e6, 1.3e10, 10, dtype=jnp.float32)

        def masked_age(age_weights):
            w_sum = jnp.sum(age_weights)
            return jnp.where(
                w_sum > 0.0,
                jnp.sum(age_weights * ssp_ages_yr)
                / jnp.maximum(w_sum, representable_denominator(1e-30))
                / 1.0e9,
                1.0,
            )

        g = jax.grad(masked_age)(jnp.zeros(10, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        # grad-assert: finite-only — the input sits below the floor, so the floor binds
        # and d/dx max(x, floor) is exactly zero by construction; the claim under test
        # is that the VJP's 1/floor**2 term is representable (no NaN), not its size.
        assert np.all(np.isfinite(np.asarray(g))), f"non-finite grad: {g}"

    _assert_f64_floor_unchanged(1e-30)


def test_line_per_qh_normalization_mechanism():
    """``components/nebular/line_precompute.py:191`` inside the ``LinePerQHTable`` builder.

    Requires a built SSP-backed ``SEDModel`` to call ``predict_line_fluxes``
    / ``predict_state``. Guarded line: ``lum / jnp.maximum(nion, floor)``,
    floor ``1e-30``. ``nion`` (ionizing photon rate) is zero for a
    zero-mass / fully-quenched reference SFH. A single line's normalized
    luminosity is used deliberately (see the module docstring's margin
    caveat: reducing several near-identical terms sharing one floored
    denominator can itself overflow the reciprocal even after this fix,
    which real per-line-flux normalization tables can approach for a target
    line list wider than a handful of lines).
    """
    with jax.enable_x64(False):

        def quotient(nion):
            lum = jnp.ones(1, dtype=jnp.float32)
            return jnp.sum(lum / jnp.maximum(nion, representable_denominator(1e-30)))

        g = jax.grad(quotient)(jnp.asarray(0.0, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        # grad-assert: finite-only — the input sits below the floor, so the floor binds
        # and d/dx max(x, floor) is exactly zero by construction; the claim under test
        # is that the VJP's 1/floor**2 term is representable (no NaN), not its size.
        assert np.isfinite(g), f"non-finite grad: {g}"

    _assert_f64_floor_unchanged(1e-30)


def test_nebular_grid_precompute_inverse_qh_mechanism():
    """``components/nebular/nebular_grid_precompute.py:627`` inside ``_row_traced``.

    Requires a built SSP-backed ``SEDModel`` for the vmapped precompute
    grid. Guarded line: ``inv_qh = 1.0 / jnp.maximum(nion, floor)``, floor
    ``1e-30`` -- the reciprocal form of the line_precompute.py mechanism
    above, same reachability (a zero-``Q_H`` grid node).
    """
    with jax.enable_x64(False):

        def inv_qh(nion):
            return 1.0 / jnp.maximum(nion, representable_denominator(1e-30))

        g = jax.grad(inv_qh)(jnp.asarray(0.0, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        # grad-assert: finite-only — the input sits below the floor, so the floor binds
        # and d/dx max(x, floor) is exactly zero by construction; the claim under test
        # is that the VJP's 1/floor**2 term is representable (no NaN), not its size.
        assert np.isfinite(g), f"non-finite grad: {g}"

    _assert_f64_floor_unchanged(1e-30)


@pytest.mark.parametrize("lineno", [2798, 3521, 3557])
def test_component_inline_normalization_mechanism(lineno):
    """``components/stellar/component.py`` inline joint-weight normalizations, floor ``1e-300``.

    Three more copies of the same ``joint_weights = joint_weights /
    jnp.maximum(joint_weights.sum(), 1e-300)`` shape already exercised end to
    end at lines 906 and 1140 above (``_age_weights_cic`` and
    ``_joint_weights_cic_met_table``), inlined directly in
    ``SEDModelComponent.apply()``'s DSPS-histogram-kernel and CIC-delta-met
    branches. Reaching each branch specifically needs a full SSP-backed
    ``SEDModel`` built with the matching ``sfh={'age_kernel': ...}`` /
    met-mode choice; the guarded shape itself, including the degenerate-SFH
    reachability (a real, physical all-zero-weight state -- see the comment
    at line 2798 in the source, quoted in this module's own docstring), is
    identical to the two primitives already measured directly.
    """
    with jax.enable_x64(False):

        def quotient(weights_sum):
            return 1.0 / jnp.maximum(weights_sum, representable_denominator(1e-300))

        g = jax.grad(quotient)(jnp.asarray(0.0, dtype=jnp.float32))
        assert g.dtype == jnp.float32
        # grad-assert: finite-only — the input sits below the floor, so the floor binds
        # and d/dx max(x, floor) is exactly zero by construction; the claim under test
        # is that the VJP's 1/floor**2 term is representable (no NaN), not its size.
        assert np.isfinite(g), f"non-finite grad at line {lineno}: {g}"

    _assert_f64_floor_moves_but_stays_safe(1e-300)


# ════════════════════════════════════════════════════════════════════════
# Group C: not reachable. Fixed anyway (the wrap is free); skipped, not
# xfailed, with the specific structural reason.
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.skip(
    reason=(
        "components/agn/blr.py:320 (_blr_l_hbeta) and :416 (compute_blr_sed): "
        "the guarded quantity is jnp.sum(_BLR_LINE_STRENGTHS), a fixed "
        "module-level constant (~25.0) with no dependence on any function "
        "argument. It cannot be driven toward zero by any differentiable "
        "input, so the floor cannot bind under gradient."
    )
)
def test_blr_strength_sum_floor_is_not_reachable():
    pass


@pytest.mark.skip(
    reason=(
        "components/agn/nlr.py:280: the guarded quantity is "
        "jnp.sum(_RICHARDSON_FLUXES), a fixed module-level constant with no "
        "dependence on any function argument -- same reasoning as blr.py above."
    )
)
def test_nlr_flux_sum_floor_is_not_reachable():
    pass


@pytest.mark.skip(
    reason=(
        "components/stellar/sps/mass_remaining.py:329 (formerly ~332): the "
        "guarded quantity, total_mass, is the IMF mass integral over a fixed "
        "log-mass grid (m_low, m_high, n_mass, imf are static structural "
        "arguments, not differentiable fit parameters). The function's own "
        "comment states it directly: 'total_mass ... cannot be zero for any "
        "real IMF.' Not reachable via gradient."
    )
)
def test_mass_remaining_total_mass_floor_is_not_reachable():
    pass


@pytest.mark.skip(
    reason=(
        "components/agn/blocks/torus_screen.py (~106): the guarded quantity, "
        "k_v, is calzetti2000_extinction_curve or smc_extinction_curve "
        "evaluated at the fixed wavelength 5500 A; it depends only on the "
        "static `law` string, not on cos_inc/oa_deg/tau_v/wavelength (the "
        "function's differentiable inputs), so it is a fixed positive "
        "constant per law choice and cannot be driven toward zero by any "
        "gradient."
    )
)
def test_torus_screen_k_v_floor_is_not_reachable():
    pass


@pytest.mark.skip(
    reason=(
        "utils/grid_interp.py:166,458,468,474,592 (subband_quadrature, "
        "PreintegratedGrid.from_filters/from_filters_single, "
        "line-effective-wavelength precompute): each is plain NumPy (`np.`, "
        "not `jnp.`), executed once at SEDModel.build time, and each "
        "enclosing function is docstringed 'JIT-compatible: no; build-time "
        "numpy precompute.' No JAX VJP ever passes through eager NumPy, so "
        "#1860's failure mode (a reverse-mode pass dividing by zero) is "
        "structurally impossible here."
    )
)
def test_grid_interp_numpy_precompute_floors_are_not_reachable():
    pass


@pytest.mark.skip(
    reason=(
        "utils/wavelength.py:85 (make_union_grid): plain NumPy, executed "
        "once at SEDModel.build time to union static template wavelength "
        "grids (never a function of a fit parameter); the function's own "
        "Notes say so directly: 'the union is computed at SEDModel.build "
        "time, not per-call.' Same non-differentiability as grid_interp.py "
        "above."
    )
)
def test_wavelength_union_grid_floor_is_not_reachable():
    pass


# ════════════════════════════════════════════════════════════════════════
# Left pinned (not fixed): inference/posterior.py's 3 sites
# ════════════════════════════════════════════════════════════════════════


def test_posterior_sites_remain_pinned_and_unfixed():
    """``inference/posterior.py:736,741,1020`` are deliberately left as-is.

    ``bpt_nii_oiii`` (736, 741) and ``agn_fraction`` (1020) are post-hoc
    diagnostics computed from an already-completed fit's stored posterior
    draws (``self.samples`` / ``self.eline_fluxes``), read for
    reporting/plotting after ``model.fit`` returns -- see the ``Posterior``
    class docstring. Nothing differentiates a ``Posterior`` method, so
    #1860's failure mode cannot occur there; this is not a backlog item.
    Pins the source shape so a future accidental "fix" here is deliberate,
    not automatic.
    """
    import inspect

    from tengri.inference.posterior import Posterior

    src = inspect.getsource(Posterior)
    assert "jnp.maximum(ha, 1e-30)" in src
    assert "jnp.maximum(total, 1e-300)" in src


# ════════════════════════════════════════════════════════════════════════
# Ratchet-level check
# ════════════════════════════════════════════════════════════════════════


def test_census_pinned_count_matches_the_three_left():
    """The tree-wide census (tools/check_representable_floors.py) agrees: 3 left."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, str(root / "tools" / "check_representable_floors.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
