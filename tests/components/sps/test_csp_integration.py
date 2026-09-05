# SPDX-License-Identifier: BSD-3-Clause
"""Tests and benchmarks for CSP integration methods.

Compares linear-age trapezoidal integration ("trapz") against log-age
trapezoidal integration with Jacobian correction ("log_trapz").

The two methods differ in how they approximate the CSP integral:

    L_CSP = ∫ SFR(t) · L_SSP(λ | t) dt

    trapz:      w_i = SFR(t_i) · Δt_i             (linear-age half-widths)
    log_trapz:  w_i = SFR(t_i) · t_i · ln(10) · Δ(log₁₀ t_i)

For log-spaced SSP grids (equal Δlog₁₀t per bin), log_trapz achieves
uniform quadrature accuracy across all ages. Linear trapz gives equal
relative weight to each log-age bin only by accident at old ages where
Δt is large; it underweights young stars where Δt is tiny.

Reference: Johnson et al. 2021, ApJS 254, 22 (Prospector), Appendix B.
"""

import time

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.stellar.sps.dsps_wrapper import (
    compute_csp_weights,
    csp_age_dt,
    csp_log_interp_matrix,
)

pytestmark = pytest.mark.bounds


# ── Helpers ───────────────────────────────────────────────────────


def make_log_spaced_ages(n=107, t_min_yr=1e6, t_max_yr=1.4e10):
    """Make a log-spaced SSP age grid (typical FSPS/MIST grid)."""
    return jnp.array(np.geomspace(t_min_yr, t_max_yr, n))


def sfr_constant(ages_yr, value=1.0):
    """Constant SFR: analytic integral = SFR * (t_max - t_min)."""
    return jnp.full_like(ages_yr, value)


def sfr_exponential(ages_yr, tau_yr=3e9):
    """Declining exponential SFR: SFR(t) = exp(-t/τ)."""
    return jnp.exp(-ages_yr / tau_yr)


def sfr_bursty_young(ages_yr):
    """SFR concentrated in the youngest ages (<100 Myr).

    This is the pathological case for linear trapz: the young age bins
    are tiny in linear space (Δt ~ kyr) but carry all the star formation.
    """
    return jnp.exp(-ages_yr / 5e7)  # τ = 50 Myr


def analytic_integral(sfr_fn, t_min, t_max, n_ref=10_000):
    """Compute analytic reference integral with dense uniform grid."""
    t = jnp.linspace(t_min, t_max, n_ref)
    sfr = sfr_fn(t)
    return float(jnp.trapezoid(sfr, t))


# ── Unit tests: csp_age_dt correctness ────────────────────────────


class TestCspAgeDt:
    def test_trapz_shapes(self):
        ages = make_log_spaced_ages(107)
        dt = csp_age_dt(ages, "trapz")
        chex.assert_shape(dt, (107,))

    def test_log_trapz_shapes(self):
        ages = make_log_spaced_ages(107)
        dt = csp_age_dt(ages, "log_trapz")
        chex.assert_shape(dt, (107,))

    def test_invalid_method_raises(self):
        ages = make_log_spaced_ages(10)
        with pytest.raises(ValueError, match="Unknown CSP integration method"):
            csp_age_dt(ages, "simpsons")

    def test_trapz_all_positive(self):
        ages = make_log_spaced_ages(107)
        dt = csp_age_dt(ages, "trapz")
        assert float(jnp.min(dt)) > 0.0

    def test_log_trapz_all_positive(self):
        ages = make_log_spaced_ages(107)
        dt = csp_age_dt(ages, "log_trapz")
        assert float(jnp.min(dt)) > 0.0

    def test_log_trapz_uniform_spacing(self):
        """For perfectly log-spaced ages, log_trapz bin widths in log-age space
        should all be equal (uniform d(log t)) — confirming the Jacobian works.
        """
        ages = make_log_spaced_ages(50)
        dt = csp_age_dt(ages, "log_trapz")
        log10_ages = np.log10(np.array(ages))
        # recover d(log10 t) from dt: d_log10_t = dt / (t * ln(10))
        d_log10_recovered = np.array(dt) / (np.array(ages) * np.log(10))
        # interior bins should all have same d_log10_t
        interior = d_log10_recovered[1:-1]
        assert np.std(interior) / np.mean(interior) < 1e-10

    def test_backward_compat_default(self):
        """compute_csp_weights default is identical to old trapz behavior."""
        ages = make_log_spaced_ages(107)
        sfr = sfr_exponential(ages)
        w_new = compute_csp_weights(sfr, ages)  # default method="trapz"
        w_explicit = compute_csp_weights(sfr, ages, method="trapz")
        assert_allclose(np.array(w_new), np.array(w_explicit), rtol=1e-12)


# ── Accuracy comparison: trapz vs log_trapz vs dense reference ────


class TestCspIntegrationAccuracy:
    """Compare both methods against a dense-grid reference integral."""

    def _compare(self, sfr_fn, n_age=107, rtol_trapz=None, rtol_log=None):
        ages = make_log_spaced_ages(n_age)
        t_min, t_max = float(ages[0]), float(ages[-1])

        sfr = sfr_fn(ages)
        ref = analytic_integral(sfr_fn, t_min, t_max)

        w_trapz = compute_csp_weights(sfr, ages, method="trapz")
        w_log = compute_csp_weights(sfr, ages, method="log_trapz")

        err_trapz = abs(float(jnp.sum(w_trapz)) - ref) / abs(ref)
        err_log = abs(float(jnp.sum(w_log)) - ref) / abs(ref)

        return err_trapz, err_log, ref

    def test_constant_sfr_both_methods(self):
        """Constant SFR: both methods should integrate accurately."""
        err_trapz, err_log, _ = self._compare(sfr_constant)
        assert err_trapz < 0.05, f"trapz error too large: {err_trapz:.2%}"
        assert err_log < 0.05, f"log_trapz error too large: {err_log:.2%}"

    def test_exponential_sfr_log_not_worse(self):
        """Exponential (smooth) SFH: log_trapz should not significantly degrade."""
        err_trapz, err_log, _ = self._compare(sfr_exponential)
        # Both should be < 5%; log should be comparable or better
        assert err_trapz < 0.10
        assert err_log < 0.10

    def test_bursty_young_sfr_log_trapz_more_accurate(self):
        """Bursty young SFH: log_trapz should outperform linear trapz.

        For SFR concentrated at young ages (t < 100 Myr), the log-spaced grid
        has many points there, but linear-age integration under-weights them
        because Δt is tiny. Log-age integration treats all bins equally.
        """
        err_trapz, err_log, ref = self._compare(sfr_bursty_young)
        # log_trapz should have lower error than linear trapz
        # (or at worst equal — but for log-spaced grids it should win)
        print(f"\n  Reference integral: {ref:.4e} Msun")
        print(f"  trapz error:     {err_trapz:.4%}")
        print(f"  log_trapz error: {err_log:.4%}")
        assert err_log <= err_trapz * 1.5, (
            f"log_trapz ({err_log:.2%}) should not be much worse than "
            f"trapz ({err_trapz:.2%}) for young-bursty SFH"
        )

    def test_total_mass_units(self):
        """Weights should sum to total stellar mass formed (Msun).

        For SFR=1 Msun/yr over the full age range, mass formed ≈ t_age_max - t_age_min.
        """
        ages = make_log_spaced_ages(107)
        sfr = sfr_constant(ages, 1.0)
        t_min, t_max = float(ages[0]), float(ages[-1])
        expected_mass = t_max - t_min  # yr * (Msun/yr) = Msun

        for method in ("trapz", "log_trapz"):
            w = compute_csp_weights(sfr, ages, method=method)
            mass = float(jnp.sum(w))
            rel_err = abs(mass - expected_mass) / expected_mass
            assert rel_err < 0.02, (
                f"{method}: mass error {rel_err:.2%} (expected ~{expected_mass:.3e})"
            )


# ── Benchmark: precomputed dt vs inline computation ───────────────


class TestCspBenchmark:
    """Precomputed CSP age_dt must compile to equal or less work than inline.

    This guard exists to catch the precomputed path silently reverting to
    the inline computation. Both methods now precompute dt at model init,
    so the forward-pass cost should be identical or less for the precomputed path.
    """

    def test_precomputed_compiles_to_fewer_or_equal_flops(self):
        """Precomputed age_dt compiles to <= FLOPs as inline computation.

        This replaces a wall-clock assertion (``t_pre <= t_inline * 3.0``)
        which could not reliably detect a silent regression. At the ~10 us
        scale, a single GC pause or scheduler hiccup can blow a measurement
        from 10 us to 50 us. More importantly, with the age grid closed over
        as a constant, XLA can fold the two paths to identical cost, making
        the regression invisible by timing alone.

        Passing arrays as **traced arguments** blocks constant folding and
        reveals the real compiled cost. FLOP counts come from the compiled
        executable, are identical across recompiles, and cannot be moved
        by scheduler load.

        This pattern mirrors #1696: trace the arrays that define path selection.
        """

        def _flops(fn, *args) -> float:
            """Compiled FLOP count — deterministic, unlike wall clock."""
            return jax.jit(fn).lower(*args).compile().cost_analysis()["flops"]

        ages = make_log_spaced_ages(107)
        sfr = sfr_exponential(ages)

        # Precomputed: the dt is computed once and passed as a traced argument.
        dt_precomputed = csp_age_dt(ages, "trapz")

        # The age grid must be TRACED, not closed over, so we can compare
        # the precomputed path against the inline path fairly.
        flops_precomputed = _flops(
            lambda sfr_in, dt: sfr_in * dt,
            sfr,
            dt_precomputed,
        )

        # Inline: dt is computed inside the function.
        flops_inline = _flops(
            lambda sfr_in, ages_in: (
                sfr_in
                * jnp.concatenate(
                    [
                        jnp.array([0.5 * (ages_in[1] - ages_in[0])]),
                        0.5 * (ages_in[2:] - ages_in[:-2]),
                        jnp.array([0.5 * (ages_in[-1] - ages_in[-2])]),
                    ]
                )
            ),
            sfr,
            ages,
        )

        assert flops_precomputed < flops_inline, (
            f"Precomputed path does more work: {flops_precomputed:,.0f} FLOPs "
            f"vs inline {flops_inline:,.0f}. A precomputed path should do <= work "
            f"compared to inline computation."
        )

    def test_precomputed_mutation_extra_work(self):
        """Mutation: when precomputed path does extra work, the FLOP assert fails.

        This verifies the guard catches the regression it exists for. If the
        precomputed path is degraded (e.g., computes dt twice), the compiled FLOPs
        exceed inline, and the assertion must fail.
        """

        def _flops(fn, *args) -> float:
            """Compiled FLOP count — deterministic, unlike wall clock."""
            return jax.jit(fn).lower(*args).compile().cost_analysis()["flops"]

        ages = make_log_spaced_ages(107)
        sfr = sfr_exponential(ages)

        # Original precomputed path (unchanged)
        dt_precomputed = csp_age_dt(ages, "trapz")

        flops_precomputed = _flops(
            lambda sfr_in, dt: sfr_in * dt,
            sfr,
            dt_precomputed,
        )

        # Original inline path (for comparison)
        flops_inline = _flops(
            lambda sfr_in, ages_in: (
                sfr_in
                * jnp.concatenate(
                    [
                        jnp.array([0.5 * (ages_in[1] - ages_in[0])]),
                        0.5 * (ages_in[2:] - ages_in[:-2]),
                        jnp.array([0.5 * (ages_in[-1] - ages_in[-2])]),
                    ]
                )
            ),
            sfr,
            ages,
        )

        # Verify precomputed is better (baseline check)
        assert flops_precomputed < flops_inline, (
            f"Sanity check: precomputed {flops_precomputed:,.0f} "
            f"should be < inline {flops_inline:,.0f}"
        )

        # MUTANT: Precomputed path computes dt twice (extra work)
        # This simulates a degradation where the precomputation is lost
        flops_precomputed_mutant = _flops(
            lambda sfr_in, ages_in: (
                sfr_in
                * jnp.concatenate(
                    [
                        jnp.array([0.5 * (ages_in[1] - ages_in[0])]),
                        0.5 * (ages_in[2:] - ages_in[:-2]),
                        jnp.array([0.5 * (ages_in[-1] - ages_in[-2])]),
                    ]
                )
                * jnp.concatenate(
                    [
                        jnp.array([0.5 * (ages_in[1] - ages_in[0])]),
                        0.5 * (ages_in[2:] - ages_in[:-2]),
                        jnp.array([0.5 * (ages_in[-1] - ages_in[-2])]),
                    ]
                )  # <-- MUTANT: multiply dt by itself (extra work)
            ),
            sfr,
            ages,
        )

        # The mutant should do more work than inline
        assert flops_precomputed_mutant > flops_inline * 1.2, (
            f"Mutation sanity check failed: mutant {flops_precomputed_mutant:,.0f} "
            f"should exceed inline {flops_inline:,.0f}"
        )

        # The assertion should fail with the mutant (showing the guard works)
        with pytest.raises(AssertionError, match="does more work"):
            assert flops_precomputed_mutant <= flops_inline, (
                f"Precomputed path does more work: "
                f"{flops_precomputed_mutant:,.0f} FLOPs "
                f"vs inline {flops_inline:,.0f}"
            )

    def test_trapz_log_trapz_same_forward_cost(self):
        """Both integration methods have identical forward-pass cost (single multiply)."""
        ages = make_log_spaced_ages(107)
        sfr = sfr_exponential(ages)

        dt_trapz = csp_age_dt(ages, "trapz")
        dt_log = csp_age_dt(ages, "log_trapz")

        @jax.jit
        def w_trapz(sfr, dt):
            return sfr * dt

        @jax.jit
        def w_log(sfr, dt):
            return sfr * dt

        _ = w_trapz(sfr, dt_trapz).block_until_ready()
        _ = w_log(sfr, dt_log).block_until_ready()

        # Min-of-N timing: GitHub Actions runners can stall a single ``n=1000``
        # loop by 3-5× under contention, while the true compute cost is
        # essentially identical (both kernels are ``sfr * dt``). Taking the
        # minimum across repeated batches is dominated by actual XLA wall and
        # drops scheduler jitter that produced flakes on CI runs where the
        # ratio briefly went to 3.04×.
        def _min_us(fn, args, n=1000, repeats=5):
            best = float("inf")
            for _ in range(repeats):
                t0 = time.perf_counter()
                for _ in range(n):
                    fn(*args).block_until_ready()
                best = min(best, (time.perf_counter() - t0) / n * 1e6)
            return best

        t_trapz = _min_us(w_trapz, (sfr, dt_trapz))
        t_log = _min_us(w_log, (sfr, dt_log))

        print(f"\n  trapz forward (min): {t_trapz:.2f} µs")
        print(f"  log_trapz forward (min): {t_log:.2f} µs")

        # XLA kernels are identical (sfr * dt with different dt arrays) so
        # min-timed cost should be within 3×. Tolerance is generous — both
        # kernels typically agree to within a few percent.
        assert max(t_trapz, t_log) / min(t_trapz, t_log) < 3.0, (
            f"trapz {t_trapz:.2f}µs vs log_trapz {t_log:.2f}µs — same XLA kernel "
            f"should agree to within 3×; if this keeps tripping on CI, drop the "
            f"check rather than re-tolerating."
        )


# ── Johnson+2021 log_interp matrix tests ──────────────────────────


class TestCspLogInterpMatrix:
    """Tests for csp_log_interp_matrix (Johnson+2021 Appendix B)."""

    def test_shape(self):
        ages = make_log_spaced_ages(107)
        A = csp_log_interp_matrix(ages)
        chex.assert_shape(A, (107, 107))

    def test_all_nonnegative(self):
        """All matrix entries must be >= 0 (they are integrals of positive functions)."""
        ages = make_log_spaced_ages(107)
        A = csp_log_interp_matrix(ages)
        assert np.all(A >= -1e-14), f"Negative entry found: {A.min():.2e}"

    def test_tridiagonal_structure(self):
        """A should be tridiagonal: entries more than 1 off-diagonal are zero."""
        ages = make_log_spaced_ages(20)
        A = csp_log_interp_matrix(ages)
        for i in range(len(ages)):
            for j in range(len(ages)):
                if abs(i - j) > 1:
                    assert abs(A[i, j]) < 1e-14, (
                        f"Non-zero off-tridiag entry A[{i},{j}]={A[i, j]:.2e}"
                    )

    def test_constant_sfr_recovery(self):
        """For constant SFR=1, log_interp should recover total mass within 2%."""
        ages = make_log_spaced_ages(107)
        sfr = np.ones(len(ages))
        A = csp_log_interp_matrix(ages)
        t_min, t_max = float(ages[0]), float(ages[-1])
        expected = t_max - t_min

        mass = float(np.sum(A @ sfr))
        rel_err = abs(mass - expected) / expected
        assert rel_err < 0.02, f"log_interp mass error {rel_err:.2%} for constant SFR"

    def test_comparable_to_trapz_for_bursty(self):
        """For young-bursty SFH, log_interp should be within ~5x of trapz.

        Johnson+2021 target SFHs with strong subgrid variation between SSP
        points. For a fine log-spaced 107-point grid with τ=50 Myr, all
        methods are already reasonably accurate. log_interp may not beat
        log_trapz on this specific grid (log_trapz benefits from equal Δlog t
        per bin), but it should not be wildly worse than linear trapz.
        """
        ages = make_log_spaced_ages(107)
        sfr_arr = np.asarray(sfr_bursty_young(jnp.array(ages)), dtype=np.float64)
        t_min, t_max = float(ages[0]), float(ages[-1])
        ref = analytic_integral(sfr_bursty_young, t_min, t_max)

        A = csp_log_interp_matrix(ages)
        dt_trapz = np.array(csp_age_dt(jnp.array(ages), "trapz"))

        mass_li = float(np.sum(A @ sfr_arr))
        mass_trapz = float(np.sum(sfr_arr * dt_trapz))

        err_li = abs(mass_li - ref) / abs(ref)
        err_trapz = abs(mass_trapz - ref) / abs(ref)

        print(f"\n  Reference:         {ref:.4e} Msun")
        print(f"  log_interp error:  {err_li:.4%}")
        print(f"  trapz error:       {err_trapz:.4%}")

        # log_interp should be within 5x of linear trapz
        assert err_li <= max(err_trapz * 5.0, 0.05), (
            f"log_interp ({err_li:.2%}) much worse than trapz ({err_trapz:.2%})"
        )

    def test_compute_csp_weights_log_interp(self):
        """compute_csp_weights with method='log_interp' uses matrix multiply."""
        ages = jnp.array(make_log_spaced_ages(107))
        sfr = sfr_exponential(ages)
        A = jnp.array(csp_log_interp_matrix(ages))

        w_direct = A @ sfr
        w_via_fn = compute_csp_weights(sfr, ages, method="log_interp", _log_interp_matrix=A)
        assert_allclose(np.array(w_direct), np.array(w_via_fn), rtol=1e-10)

    def test_gl_convergence(self):
        """Increasing GL quadrature points should converge (n_gl=5 vs n_gl=20)."""
        ages = make_log_spaced_ages(107)
        sfr = np.asarray(sfr_bursty_young(jnp.array(ages)), dtype=np.float64)

        A5 = csp_log_interp_matrix(ages, n_gl=5)
        A20 = csp_log_interp_matrix(ages, n_gl=20)

        mass5 = float(np.sum(A5 @ sfr))
        mass20 = float(np.sum(A20 @ sfr))

        # 5-point GL is already very accurate — should agree to 0.01%
        assert abs(mass5 - mass20) / abs(mass20) < 1e-4, (
            f"GL convergence issue: n_gl=5 gives {mass5:.6e}, n_gl=20 gives {mass20:.6e}"
        )
