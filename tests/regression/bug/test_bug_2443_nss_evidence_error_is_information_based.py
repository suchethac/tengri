# SPDX-License-Identifier: BSD-3-Clause
"""#2443: the NSS evidence error is sqrt(H / n_eff), not sqrt(ESS) / n_live.

The standard deviation of log Z from a nested-sampling run is
sigma = sqrt(H / n_live) (Skilling 2006), with H the information in nats. For
batch deletion of k points per iteration the live count is replaced by
n_eff = sum_j 1/(n-j) / sum_j 1/(n-j)^2. H is a property of the problem; the
effective sample size is not, so sqrt(ESS)/n_live tracks sqrt(H/n_live) only when
ESS happens to be close to n_live * H.

The tests build real nested-sampling runs in NumPy for a problem with a known
answer: a prior uniform in a d-ball of radius R and log L = -r^2/2. The enclosed
prior volume is X(r) = (r / R)^d, so a uniform draw inside the contour r < r*
is r = r* u^(1/d). The evidence is

    log Z = log[(2 pi)^(d/2) Gamma(d/2 + 1) / (pi^(d/2) R^d)]

(the Gaussian is untruncated at R = 10 to better than 1e-15) and the information
is H = -d/2 - log Z. Two radii are used: R = 10 and R = 100. At R = 100 the
posterior is the same but the prior is 10^d larger, so H is larger by d ln 10
while ESS / n_live is unchanged, and the two error formulas separate.
"""

from __future__ import annotations

from fractions import Fraction
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gammaln, logsumexp

from tengri.inference.backends.evidence import _nss_information_and_error
from tengri.inference.backends.nested.base import NSInfo, StateWithLogLikelihood
from tengri.inference.backends.nested.utils import log_weights

pytestmark = pytest.mark.regression_bug

D = 5
N_REALIZATIONS = 100
EXTRA_SHRINKAGE = 20.0  # nats of ln X beyond H, so the live remainder is < 1e-6 of Z
N_CALIBRATION_RUNS = 40


def analytic(radius, d=D):
    """Return (log Z, H) in nats for the d-ball prior and log L = -r^2 / 2."""
    log_z = (
        0.5 * d * np.log(2 * np.pi)
        + gammaln(d / 2 + 1)
        - 0.5 * d * np.log(np.pi)
        - d * np.log(radius)
    )
    return float(log_z), float(-d / 2 - log_z)


def iterations_needed(radius, n_live, num_delete):
    """Iterations that compress ln X to H + EXTRA_SHRINKAGE."""
    _, h = analytic(radius)
    return int(np.ceil((h + EXTRA_SHRINKAGE) * n_live / num_delete))


def make_run(n_live, num_delete, n_iter, seed, radius):
    """Nested-sampling run on the d-ball problem, as an ``NSInfo``.

    Each iteration removes the ``num_delete`` outermost live points and replaces
    them by uniform draws inside the contour of the innermost removed point, whose
    log-likelihood is stored as their birth contour. The final live points are
    appended as dead points with their own birth contours.
    """
    rng = np.random.default_rng(seed)
    r = radius * rng.random(n_live) ** (1.0 / D)
    birth = np.full(n_live, -np.inf)
    dead_logl, dead_birth = [], []
    for _ in range(n_iter):
        removed = np.argpartition(-r, num_delete - 1)[:num_delete]
        removed = removed[np.argsort(-r[removed])]
        dead_logl.append(-0.5 * r[removed] ** 2)
        dead_birth.append(birth[removed])
        r_star = r[removed[-1]]
        r[removed] = r_star * rng.random(num_delete) ** (1.0 / D)
        birth[removed] = -0.5 * r_star**2
    dead_logl.append(-0.5 * r**2)
    dead_birth.append(birth)
    logl = jnp.asarray(np.concatenate(dead_logl))
    zeros = jnp.zeros_like(logl)
    particles = StateWithLogLikelihood(
        {"r": zeros}, zeros, logl, jnp.asarray(np.concatenate(dead_birth))
    )
    return NSInfo(particles, None)


@cache
def run_summary(radius, n_live, num_delete, n_iter, seed):
    """Helper outputs and diagnostics for one run (tuple, hashable and immutable)."""
    run = make_run(n_live, num_delete, n_iter, seed, radius)
    log_w = np.asarray(log_weights(jax.random.PRNGKey(seed), run, shape=N_REALIZATIONS))
    logl = np.asarray(run.particles.loglikelihood)
    information, err, n_eff = _nss_information_and_error(log_w, logl, n_live, num_delete)
    log_z_k = logsumexp(log_w, axis=0)
    # ESS as defined by tengri.inference.backends.nested.utils.ess (mean log weight).
    log_w_mean = log_w.mean(axis=-1)
    log_w_mean = log_w_mean - log_w_mean.max()
    ess = float(np.exp(2 * logsumexp(log_w_mean) - logsumexp(2 * log_w_mean)))
    live_fraction = float(np.mean(np.exp(logsumexp(log_w[-n_live:], axis=0) - log_z_k)))
    return {
        "H": information,
        "err": err,
        "n_eff": n_eff,
        "ess": ess,
        "log_z": float(np.mean(log_z_k)),
        "live_fraction": live_fraction,
    }


def n_eff_exact(n_live, num_delete):
    """n_eff from exact rational arithmetic, independent of the implementation."""
    s1 = sum(Fraction(1, n_live - j) for j in range(num_delete))
    s2 = sum(Fraction(1, (n_live - j) ** 2) for j in range(num_delete))
    return float(s1 / s2)


def test_a_information_matches_analytic():
    """(a) H of a 400-point run agrees with -d/2 - log Z.

    The scatter of H over 5 seeds is measured in the test and the tolerance is
    3x the scatter of the mean over those seeds.
    """
    radius, n_live = 10.0, 400
    log_z_true, h_true = analytic(radius)
    n_iter = iterations_needed(radius, n_live, 1)
    runs = [run_summary(radius, n_live, 1, n_iter, seed) for seed in range(5)]
    h_values = np.array([run["H"] for run in runs])
    h_mean = h_values.mean()
    scatter_of_mean = h_values.std(ddof=1) / np.sqrt(len(h_values)) / h_true
    tol = 3 * scatter_of_mean
    rel_err = abs(h_mean - h_true) / h_true
    msg = (
        f"analytic log Z={log_z_true:.6f}, H={h_true:.6f}; measured H={h_mean:.4f} "
        f"(single-run std {h_values.std(ddof=1):.4f}, mean's relative scatter "
        f"{scatter_of_mean:.2%}); relative error {rel_err:.2%}, tolerance 3x scatter = {tol:.2%}"
    )
    print(f"\n(a) {msg}")
    assert max(run["live_fraction"] for run in runs) < 1e-6, msg
    assert rel_err < tol, msg


def test_b_error_scales_as_inverse_sqrt_n_live():
    """(b) sigma(400) / sigma(100) = sqrt(100/400) = 0.5 within 10 %."""
    radius = 10.0
    errs = {}
    for n_live in (100, 400):
        n_iter = iterations_needed(radius, n_live, 1)
        errs[n_live] = np.mean(
            [run_summary(radius, n_live, 1, n_iter, s)["err"] for s in range(5)]
        )
    ratio = errs[400] / errs[100]
    msg = f"err(100)={errs[100]:.5f}, err(400)={errs[400]:.5f}, ratio={ratio:.4f} (expect 0.5)"
    print(f"\n(b) {msg}")
    assert abs(ratio - 0.5) < 0.05, msg


@pytest.mark.parametrize(
    ("radius", "old_formula_is_off"),
    [(10.0, False), (100.0, True)],
    ids=["R10", "R100"],
)
def test_c_error_independent_of_run_length(radius, old_formula_is_off):
    """(c) Running 1.5x and 3x as long leaves the error unchanged.

    The reference is sigma = sqrt(H / n_live) with the analytic H (Skilling 2006).
    For R = 100 the old expression sqrt(ESS)/n_live is far from it; for R = 10
    ESS is close to n_live * H, so the two formulas agree there and nothing is
    asserted about the old one.
    """
    n_live, seed = 100, 7
    _, h_true = analytic(radius)
    base = iterations_needed(radius, n_live, 1)
    short = run_summary(radius, n_live, 1, int(1.5 * base), seed)
    long = run_summary(radius, n_live, 1, 3 * base, seed)
    truth = np.sqrt(h_true / n_live)
    old_short, old_long = np.sqrt(short["ess"]) / n_live, np.sqrt(long["ess"]) / n_live
    change = abs(long["err"] - short["err"]) / short["err"]
    msg = (
        f"R={radius:g}: sqrt(H/n)={truth:.4f}; new err {short['err']:.4f} (1.5x) / "
        f"{long['err']:.4f} (3x), change {change:.2%}; old sqrt(ESS)/n {old_short:.4f} / "
        f"{old_long:.4f}, old/truth = {old_short / truth:.2f}"
    )
    print(f"\n(c) {msg}")
    assert change < 0.02, msg
    assert abs(short["err"] / truth - 1) < 0.10, msg
    if old_formula_is_off:
        assert abs(old_short / truth - 1) > 0.25, msg


@pytest.mark.parametrize("radius", [10.0, 100.0], ids=["R10", "R100"])
@pytest.mark.parametrize("num_delete", [1, 20])
def test_d_error_calibrated_against_scatter_of_log_z(radius, num_delete):
    """(d) std of log Z over 40 independent runs / reported error is in [0.75, 1.33].

    Each run's log Z is the mean over its 100 simulated volume sequences of
    log Z_k, which is the run's best estimate; its spread over independent runs
    is the true scatter the reported error must describe. The ratio with n_live in
    place of n_eff is printed for comparison.
    """
    n_live = 100
    log_z_true, _ = analytic(radius)
    n_iter = iterations_needed(radius, n_live, num_delete)
    runs = [
        run_summary(radius, n_live, num_delete, n_iter, 1000 + s)
        for s in range(N_CALIBRATION_RUNS)
    ]
    std_log_z = np.std([r["log_z"] for r in runs], ddof=1)
    mean_err = np.mean([r["err"] for r in runs])
    mean_h = np.mean([r["H"] for r in runs])
    ratio = std_log_z / mean_err
    ratio_n_live = std_log_z / np.sqrt(mean_h / n_live)
    mc_error = ratio / np.sqrt(2 * (N_CALIBRATION_RUNS - 1))
    msg = (
        f"R={radius:g}, k={num_delete}: analytic log Z={log_z_true:.4f}; std(log Z)="
        f"{std_log_z:.4f}, mean err={mean_err:.4f}, ratio={ratio:.3f} +- {mc_error:.3f} "
        f"(n_eff={runs[0]['n_eff']:.2f}); ratio with n_live instead of n_eff = {ratio_n_live:.3f}"
    )
    print(f"\n(d) {msg}")
    assert 0.75 <= ratio <= 1.33, msg


def test_e_effective_live_count():
    """(e) n_eff = n for single deletion; 89.7634 for (100, 20); 474.6236 for (500, 50).

    The expected values come from exact rational arithmetic and are pinned to
    four decimals.
    """
    flat_logl = np.zeros(4)
    flat_log_w = np.log(np.full((4, 2), 0.25))
    got = {
        (n, k): _nss_information_and_error(flat_log_w, flat_logl, n, k)[2]
        for n, k in [(100, 1), (100, 20), (500, 50)]
    }
    print(f"\n(e) n_eff: {got}")
    assert got[(100, 1)] == pytest.approx(100.0, rel=1e-13)
    assert got[(100, 20)] == pytest.approx(89.7634, abs=5e-5)
    assert got[(500, 50)] == pytest.approx(474.6236, abs=5e-5)
    for (n, k), value in got.items():
        assert value == pytest.approx(n_eff_exact(n, k), rel=1e-12)


def test_f_flat_likelihood_has_zero_information():
    """(f) A constant likelihood gives H = 0 and a finite zero error."""
    rng = np.random.default_rng(0)
    d_x = rng.dirichlet(np.ones(50), size=N_REALIZATIONS).T  # (50, K), columns sum to 1
    logl = np.full(50, -3.0)
    information, err, n_eff = _nss_information_and_error(
        np.log(d_x) + logl[:, None], logl, 100, 20
    )
    print(f"\n(f) flat likelihood: H={information:.3e}, err={err:.3e}, n_eff={n_eff:.4f}")
    assert information == pytest.approx(0.0, abs=1e-10)
    assert np.isfinite(err)
    assert err == pytest.approx(0.0, abs=1e-5)


def test_g_negative_information_is_an_error():
    """(g) Weights inconsistent with the likelihoods (H < 0) raise rather than clamp."""
    logl = np.array([-10.0, -10.0])
    log_w = np.zeros((2, 3))  # posterior weights 0.5, so H = -10 - log 2 < 0
    with pytest.raises(ValueError, match="negative"):
        _nss_information_and_error(log_w, logl, 100, 1)


def test_h_normalizes_each_realization_before_averaging():
    """H is the mean over columns of each column's own information.

    Columns whose log Z_k differ by several nats must be normalized one by one;
    normalizing the column-average instead gives a different number.
    """
    logl = np.array([-4.0, -2.0, -1.0, -0.5])
    # log(L_i dX_ik): volume elements sum to 1, e^-3 and e^-6 in the three columns.
    d_x = np.array([[0.1, 0.3, 0.4], [0.2, 0.3, 0.3], [0.3, 0.2, 0.2], [0.4, 0.2, 0.1]])
    d_x = d_x / d_x.sum(axis=0) * np.exp([0.0, -3.0, -6.0])
    log_w = np.log(d_x) + logl[:, None]
    per_column = []
    for k in range(log_w.shape[1]):
        p = np.exp(log_w[:, k])
        z = p.sum()
        per_column.append(np.sum(p / z * logl) - np.log(z))
    expected = float(np.mean(per_column))
    mean_w = np.mean(np.exp(log_w), axis=1)
    averaged_first = float(np.sum(mean_w / mean_w.sum() * logl) - np.log(mean_w.sum()))
    information, _, _ = _nss_information_and_error(log_w, logl, 100, 1)
    print(
        f"\n(h) per-column mean H={expected:.9f}, helper H={information:.9f}, "
        f"averaged-first H={averaged_first:.9f}"
    )
    assert information == pytest.approx(expected, abs=1e-12)
    assert abs(averaged_first - expected) > 1e-3


@pytest.mark.parametrize(("n_live", "num_delete"), [(100, 100), (100, 150), (100, 0)])
def test_i_num_delete_outside_live_set_raises(n_live, num_delete):
    """(i) num_delete must satisfy 1 <= num_delete < n_live; the message names both."""
    with pytest.raises(ValueError, match=rf"{num_delete}.*{n_live}"):
        _nss_information_and_error(np.zeros((2, 2)), np.zeros(2), n_live, num_delete)
