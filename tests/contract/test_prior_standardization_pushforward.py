"""Contract: every Distribution's unstandardize is the exact prior pushforward.

The standardized parameterization (Knollmüller & Enßlin 2019 [1]_, Eqs.
18-25; Frank, Leike & Enßlin 2021 [2]_, Eqs. 2-3) requires

    theta = unstandardize(xi) = F_prior^{-1}(Phi(xi)),   xi ~ N(0, 1),

so the ½ξᵀξ prior term in the information Hamiltonian
(``inference/loss_functions.build_loss_fn``) corresponds to the *declared*
prior. If ``unstandardize`` is any other bijection (sigmoid, clipped
affine, variance-matched approximation), MAP/NUTS/VI silently fit a
different prior than ``log_prob``/``sample`` describe — and disagree with
the physical-space prior used by nested sampling and evidence backends.

Three-way consistency enforced here for every distribution:

  1. pushforward(N(0,1)) ≡ sample()   (two-sample KS)
  2. pushforward(N(0,1)) ≡ exp(log_prob)  (one-sample KS vs analytic CDF)
  3. standardize(unstandardize(xi)) ≈ xi  (round-trip bijectivity)

Regression for the 2026-07 audit: LogUniform used a sigmoid map
(logit-normal effective prior, tails compressed ~2x at the 99th
percentile), StudentT used a variance-matched Gaussian (heavy tails of the
df=2 Leja+2019 continuity-SFH ratio prior discarded), and truncated
Gaussian/LogNormal clipped at the bounds (point masses, zero gradient,
non-invertible round-trip).

References
----------
.. [1] Knollmüller, J. & Enßlin, T. A., "Metric Gaussian Variational
   Inference", arXiv:1901.11033.
.. [2] Frank, P., Leike, R., & Enßlin, T. A. 2021, Entropy 23, 853,
   "Geometric Variational Inference", arXiv:2105.10470,
   doi:10.3390/e23070853.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as st

from tengri.parameters.priors import (
    Gaussian,
    Laplace,
    LogNormal,
    LogUniform,
    StudentT,
    Uniform,
)

pytestmark = pytest.mark.contract

N = 100_000
KEY = jax.random.PRNGKey(20260705)
# n=1e5 two-sample KS: D ~ 0.006 at p=0.05; require p > 1e-3 to keep the
# test deterministic-stable while still catching any wrong-transform bug
# (the 2026-07 regressions had D = 0.04-0.18, p < 1e-100).
P_MIN = 1e-3


def _loguniform_cdf(lo, hi):
    return lambda x: (np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo))


CASES = [
    pytest.param(Uniform(2.0, 5.0), lambda x: st.uniform.cdf(x, 2.0, 3.0), id="uniform"),
    pytest.param(Gaussian(1.5, 0.7), lambda x: st.norm.cdf(x, 1.5, 0.7), id="gaussian"),
    pytest.param(
        Gaussian(0.0, 1.0, lo=-1.0, hi=2.0),
        lambda x: st.truncnorm.cdf(x, -1.0, 2.0),
        id="gaussian-truncated",
    ),
    pytest.param(LogUniform(1e-2, 1e2), _loguniform_cdf(1e-2, 1e2), id="loguniform"),
    pytest.param(LogUniform(1.0, 300.0), _loguniform_cdf(1.0, 300.0), id="loguniform-psd-tau"),
    pytest.param(LogNormal(0.0, 0.5), lambda x: st.lognorm.cdf(x, 0.5), id="lognormal"),
    pytest.param(
        LogNormal(0.0, 0.5, lo=0.5, hi=3.0),
        lambda x: (
            (st.lognorm.cdf(np.clip(x, 0.5, 3.0), 0.5) - st.lognorm.cdf(0.5, 0.5))
            / (st.lognorm.cdf(3.0, 0.5) - st.lognorm.cdf(0.5, 0.5))
        ),
        id="lognormal-truncated",
    ),
    pytest.param(
        StudentT(0.0, 0.3, 2.0),
        lambda x: st.t.cdf(x / 0.3, 2.0),
        id="studentt-df2-continuity-prior",
    ),
    pytest.param(StudentT(0.0, 1.0, 5.0), lambda x: st.t.cdf(x, 5.0), id="studentt-df5-table"),
    pytest.param(StudentT(0.0, 1.0, 1.0), lambda x: st.t.cdf(x, 1.0), id="studentt-df1-cauchy"),
    pytest.param(
        StudentT(0.0, 0.3, 2.0, lo=-1.0, hi=1.0),
        lambda x: (
            (st.t.cdf(np.clip(x, -1.0, 1.0) / 0.3, 2.0) - st.t.cdf(-1.0 / 0.3, 2.0))
            / (st.t.cdf(1.0 / 0.3, 2.0) - st.t.cdf(-1.0 / 0.3, 2.0))
        ),
        id="studentt-truncated",
    ),
    pytest.param(Laplace(0.0, 0.5), lambda x: st.laplace.cdf(x, 0.0, 0.5), id="laplace"),
    pytest.param(
        Laplace(0.2, 0.3, lo=-0.5, hi=0.8),
        lambda x: (
            (st.laplace.cdf(np.clip(x, -0.5, 0.8), 0.2, 0.3) - st.laplace.cdf(-0.5, 0.2, 0.3))
            / (st.laplace.cdf(0.8, 0.2, 0.3) - st.laplace.cdf(-0.5, 0.2, 0.3))
        ),
        id="laplace-truncated",
    ),
]


def _pushforward(dist, n=N):
    xi = jax.random.normal(KEY, (n,))
    return xi, np.asarray(jax.vmap(dist.unstandardize)(xi))


@pytest.mark.parametrize("dist,cdf", CASES)
def test_pushforward_matches_sample(dist, cdf):
    """unstandardize(N(0,1)) and sample() draw from the same distribution."""
    _, push = _pushforward(dist)
    draws = np.asarray(jax.vmap(dist.sample)(jax.random.split(KEY, N)))
    res = st.ks_2samp(push, draws)
    assert res.pvalue > P_MIN, (
        f"{dist!r}: pushforward != sample() (KS D={res.statistic:.4f}, "
        f"p={res.pvalue:.2e}) — unstandardize is not the prior's quantile map"
    )


@pytest.mark.parametrize("dist,cdf", CASES)
def test_pushforward_matches_log_prob(dist, cdf):
    """unstandardize(N(0,1)) follows the analytic CDF implied by log_prob."""
    _, push = _pushforward(dist)
    res = st.kstest(push, cdf)
    assert res.pvalue > P_MIN, (
        f"{dist!r}: pushforward != declared density (KS D={res.statistic:.4f}, "
        f"p={res.pvalue:.2e}) — latent-space and physical-space priors disagree"
    )


@pytest.mark.parametrize("dist,cdf", CASES)
def test_round_trip_bijective(dist, cdf):
    """standardize(unstandardize(xi)) recovers xi (smooth bijection, no clip).

    Domain |xi| <= 4.5: beyond ~5.2 the p = Phi(xi) guard clip at 1e-7
    saturates by design (CDF-space quantization), so the bijection claim
    is over the +-4.5-sigma latent range that samplers explore.
    """
    xi = jnp.linspace(-4.5, 4.5, 4096)
    push = jax.vmap(dist.unstandardize)(xi)
    back = np.asarray(jax.vmap(dist.standardize)(push))
    np.testing.assert_allclose(back, np.asarray(xi), atol=1e-5, rtol=0)


def test_truncated_gaussian_gradient_nonzero_at_bounds():
    """Truncation must keep gradients alive near the bounds (no clip plateau)."""
    dist = Gaussian(0.0, 1.0, lo=-1.0, hi=2.0)
    g = jax.grad(lambda x: dist.unstandardize(x))
    for xi in [-3.0, 0.0, 3.0]:
        val = float(g(jnp.asarray(xi)))
        assert np.isfinite(val) and val > 0.0, (
            f"d(unstandardize)/dxi at xi={xi} is {val}; clipping would zero it"
        )


def test_studentt_df2_heavy_tails_preserved():
    """The df=2 continuity-SFH ratio prior must keep its Student-t tails.

    Regression: the variance-matched Gaussian approximation put the
    99.87th percentile (xi=3) at mu + 3*sigma*scale with scale=3.0 —
    the exact df=2 quantile at p=Phi(3) is t=13.97, i.e. 4.19 in
    physical units for sigma=0.3, not 2.7.
    """
    dist = StudentT(0.0, 0.3, 2.0)
    theta = float(dist.unstandardize(jnp.asarray(3.0)))
    exact = 0.3 * float(st.t.ppf(st.norm.cdf(3.0), 2.0))
    np.testing.assert_allclose(theta, exact, rtol=1e-6)
