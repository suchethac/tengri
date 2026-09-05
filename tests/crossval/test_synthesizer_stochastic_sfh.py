# SPDX-License-Identifier: BSD-3-Clause
r"""Cross-validation: tengri's ``field`` compositor vs Synthesizer's ``SFH.Stochastic``.

Synthesizer's ``main`` branch carries ``SFH.Stochastic``, which draws fluctuations
of :math:`\log_{10}(\mathrm{SFR})` about a base SFH from a Gaussian process on a
uniform cosmic-time grid running from the Big Bang to the observation epoch. The
only kernel it ships is the damped random walk,
:math:`C(\Delta t) = \sigma^2 \exp(-|\Delta t| / \tau)` in
:math:`\mathrm{dex}^2`, and one realization is drawn per seed and frozen.

tengri's ``field`` compositor is the same Gaussian process, entered from the
opposite side: ``sfh={'type': ['<mean>', 'field']}`` multiplies the mean SFH by
:math:`\exp(x(t) - K_0/2)`, where :math:`x` is a damped random walk in natural-log
SFR with covariance :math:`K(\Delta t) = (\sigma \ln 10)^2 \exp(-|\Delta t|/\tau)`.
Dividing by :math:`(\ln 10)^2` puts that in :math:`\mathrm{dex}^2` and it *is*
Synthesizer's kernel, so ``psd_sigma`` and Synthesizer's ``sigma`` mean the same
thing, and ``psd_tau_myr`` is Synthesizer's ``tau`` in Myr.

The one convention difference is the :math:`-K_0/2` term. tengri applies the
log-normal bias correction, so the ensemble mean of the modulated SFH equals the
mean SFH; Synthesizer does not, so at :math:`\sigma = 0.3` dex its ensemble-mean
*linear* SFR sits ``exp(K_0/2) - 1 = +26.95 %`` above its base SFH. Both are
defensible — the correction preserves the mean SFR, its absence preserves the
mean log SFR — but a normalization compared across the two codes must account
for it.

This file needs no Synthesizer install. The reference below is an independent
NumPy implementation of the damped-random-walk covariance and the Cholesky
sampler that Synthesizer's Stochastic SFH documents (its ``main`` branch as of
2026-09-06); ``TestReferenceKernel`` checks that reference against the closed
form before anything else compares against it.

Run with::

    pytest -m crossval tests/crossval/test_synthesizer_stochastic_sfh.py

References
----------
.. [1] C. C. Lovell et al., "Synthesizer: a Software Package for Synthetic
   Astronomical Observables," Open Journal of Astrophysics, 8 (2025).
   https://doi.org/10.33232/001c.145766
.. [2] W. J. Roper et al., "Synthesizer: Synthetic Observables for Modern
   Astronomy," Journal of Open Source Software, 11, 9436 (2026).
   https://doi.org/10.21105/joss.09436
.. [3] K. G. Iyer et al., "Nonparametric star formation history reconstruction
   with Gaussian processes I: counting major episodes of star formation,"
   arXiv:2208.05938 (2024). [the GP + PSD formalism both codes follow]
.. [4] N. Caplar & S. Tacchella, "Stochastic modeling of star-formation
   histories I: the scatter of the star-forming main sequence," MNRAS, 487,
   3845 (2019). arXiv:1901.07556. https://doi.org/10.1093/mnras/stz1449
"""

from __future__ import annotations

import warnings
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, make_log_age_grid
from tengri.components.stellar.sfh.registry import compute_field_gp

pytestmark = pytest.mark.crossval

LN10 = np.log(10.0)

#: Synthesizer's documented Stochastic SFH example: DampedRandomWalk(sigma=0.3,
#: tau=1 Gyr) observed at z = 1.
SIGMA_DEX = 0.3
TAU_YR = 1.0e9

#: K(0)/2 in natural-log units — tengri's log-normal bias correction, and the
#: log of Synthesizer's mean-linear-SFR bias.
K0_HALF = 0.5 * (SIGMA_DEX * LN10) ** 2


# ── Reference: the documented Synthesizer construction ────────────────
#
# An independent NumPy implementation of what Synthesizer's Stochastic SFH
# documents (main branch, 2026-09-06): a Toeplitz covariance
# sigma^2 exp(-|dt|/tau) on a uniform cosmic-time grid and a Cholesky draw of
# the log10 SFR fluctuations. Units and input validation are out of scope here.


def synthesizer_drw_covariance(delta_t, sigma, tau):
    """``DampedRandomWalk.covariance``: ``sigma**2 * exp(-|dt| / tau)``.

    Parameters
    ----------
    delta_t : array_like
        Time lag(s) [yr].
    sigma : float
        Standard deviation of the log10(SFR) fluctuations [dex].
    tau : float
        Correlation timescale [yr].

    Returns
    -------
    ndarray
        Auto-covariance at each lag [dex^2].
    """
    return sigma**2 * np.exp(-np.abs(delta_t) / tau)


def synthesizer_build_covariance_matrix(tarr, sigma, tau):
    """``Kernel.build_covariance_matrix``: the symmetric Toeplitz fill.

    Parameters
    ----------
    tarr : array_like, shape (n,)
        A regularly spaced time grid [yr].
    sigma, tau : float
        Kernel parameters, as in :func:`synthesizer_drw_covariance`.

    Returns
    -------
    ndarray, shape (n, n)
        The covariance matrix [dex^2].
    """
    tarr = np.asarray(tarr, dtype=np.float64)
    n = tarr.size
    cov_deltat = np.asarray(
        synthesizer_drw_covariance(tarr - tarr[0], sigma, tau), dtype=np.float64
    )
    cov_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        col = np.roll(cov_deltat, i)
        col[:i] = np.flip(cov_deltat[1 : i + 1], 0)
        cov_matrix[:, i] = col
    return cov_matrix


def synthesizer_sample_multivariate_normal(cov, rng, size=1):
    """``_sample_multivariate_normal``: zero-mean draws via Cholesky.

    Parameters
    ----------
    cov : array_like, shape (n, n)
        Covariance matrix.
    rng : numpy.random.Generator
        Random number generator.
    size : int, optional
        Number of independent draws. Synthesizer draws one; the batch is a
        loop over ``random_seed`` there and a second axis here.

    Returns
    -------
    ndarray, shape (size, n)
        Draws from ``N(0, cov)``.
    """
    n = cov.shape[0]
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        jitter = 1e-10 * np.trace(cov) / n
        chol = np.linalg.cholesky(cov + jitter * np.eye(n))
    return (chol @ rng.standard_normal((n, size))).T


# ── tengri-side helpers ───────────────────────────────────────────────


def _uniform_cosmic_grid(step_gyr=0.1, n=100):
    """Uniform cosmic-time grid [yr] and its log10, as Synthesizer builds one.

    Synthesizer samples the GP on ``linspace(0, t_univ, n_grid)``. tengri's field
    takes ``log10(age/yr)``, so the grid starts one step above zero to keep the
    logarithm finite; the kernel depends only on lags, which are unaffected.
    """
    t_yr = (np.arange(1, n + 1) * step_gyr) * 1e9
    return t_yr, np.log10(t_yr)


def _field_operator(log_t):
    """The exact linear map ``gp_x = M xi`` the ``field`` compositor applies.

    Runs :func:`compute_field_gp` on every basis vector of ``xi``. The map is
    linear in ``xi``, so the columns of ``M`` are the images of the basis and
    ``M M^T`` is the realized covariance in natural-log units — no Monte Carlo,
    no tolerance beyond floating point.
    """
    n = log_t.size
    basis = jnp.eye(n)
    grid = jnp.asarray(log_t)
    cols = jax.vmap(lambda e: _field_gp(e, grid, n))(basis)
    return np.asarray(cols).T


def _field_gp_and_k0(xi, log_age_grid, n_grid):
    """One ``field`` GP realization [natural log] and its bias correction.

    ``d_log_age`` is unused by the ``drw`` branch, which builds its covariance in
    physical time from ``log_age_grid``; it is passed as 0.0 to say so.
    """
    return compute_field_gp(
        xi,
        SIGMA_DEX,
        TAU_YR,
        n_grid,
        0.0,
        field_model="drw",
        log_age_grid=log_age_grid,
    )


def _field_gp(xi, log_age_grid, n_grid):
    """The realization alone, for the covariance and amplitude checks."""
    return _field_gp_and_k0(xi, log_age_grid, n_grid)[0]


# ── Part 0: the reference itself ──────────────────────────────────────


class TestReferenceKernel:
    """A reference is only a reference once it reproduces its closed form."""

    def test_toeplitz_fill_equals_the_direct_kernel(self):
        t_yr, _ = _uniform_cosmic_grid()
        built = synthesizer_build_covariance_matrix(t_yr, SIGMA_DEX, TAU_YR)
        direct = synthesizer_drw_covariance(t_yr[:, None] - t_yr[None, :], SIGMA_DEX, TAU_YR)
        assert np.max(np.abs(built - direct)) < 1e-15, (
            f"the reference Toeplitz fill differs from sigma^2 exp(-|dt|/tau) by "
            f"{np.max(np.abs(built - direct)):.3e} dex^2"
        )

    def test_cholesky_sampler_realizes_the_covariance(self):
        t_yr, _ = _uniform_cosmic_grid()
        cov = synthesizer_build_covariance_matrix(t_yr, SIGMA_DEX, TAU_YR)
        draws = synthesizer_sample_multivariate_normal(cov, np.random.default_rng(42), size=20000)
        emp = np.cov(draws, rowvar=False)
        rel = np.abs(emp - cov).max() / SIGMA_DEX**2
        assert rel < 0.05, f"empirical covariance of the reference sampler off by {rel:.3f}"


# ── Part 1 (a): kernel parity ─────────────────────────────────────────


class TestKernelParity:
    """tengri's field covariance, in dex^2, IS Synthesizer's DRW kernel."""

    def test_named_lags(self):
        """Lags 0, 0.1, 1 and 5 Gyr at sigma = 0.3 dex, tau = 1 Gyr.

        Measured: every lag agrees to <= 4.5e-15 relative — floating point, not
        a physics tolerance.
        """
        _, log_t = _uniform_cosmic_grid()
        m = _field_operator(log_t)
        k_dex = (m @ m.T) / LN10**2

        worst = 0.0
        for lag_gyr in (0.0, 0.1, 1.0, 5.0):
            j = round(lag_gyr / 0.1)
            got = k_dex[0, j]
            want = float(synthesizer_drw_covariance(lag_gyr * 1e9, SIGMA_DEX, TAU_YR))
            rel = abs(got / want - 1.0)
            worst = max(worst, rel)
            assert rel < 1e-12, (
                f"lag {lag_gyr} Gyr: tengri {got:.12e} dex^2 vs Synthesizer "
                f"{want:.12e} dex^2 ({rel:.3e} relative)"
            )
        assert worst < 1e-12

    def test_whole_covariance_matrix(self):
        """Not just the named lags: every entry of the 100x100 Toeplitz matrix.

        Measured worst relative deviation 2.7e-14.
        """
        t_yr, log_t = _uniform_cosmic_grid()
        m = _field_operator(log_t)
        k_dex = (m @ m.T) / LN10**2
        k_synth = synthesizer_build_covariance_matrix(t_yr, SIGMA_DEX, TAU_YR)
        rel = np.abs(k_dex / k_synth - 1.0).max()
        assert rel < 1e-11, f"worst entry differs by {rel:.3e} relative"

    def test_amplitude_convention_is_dex_not_natural_log(self):
        """K(0) = (sigma ln10)^2, so the modulation std is sigma DEX.

        The failure this guards is silent: a field built with ``psd_sigma``
        read as a natural-log amplitude is a factor ln10 = 2.303 too quiet, and
        nothing raises.
        """
        _, log_t = _uniform_cosmic_grid()
        m = _field_operator(log_t)
        k0 = (m @ m.T)[0, 0]
        assert abs(k0 / (SIGMA_DEX * LN10) ** 2 - 1.0) < 1e-12, (
            f"K(0) = {k0:.9f} natural-log^2; expected {(SIGMA_DEX * LN10) ** 2:.9f} "
            f"(sigma in dex). A natural-log reading would give {SIGMA_DEX**2:.9f}."
        )


# ── Part 2 (b): Monte Carlo through the compositor's own construction ─


class TestMonteCarloAmplitude:
    """4000 latent draws through :func:`compute_field_gp` on the field's grid."""

    N_DRAWS = 4000
    N_GRID = 256

    @staticmethod
    def _draws(n_draws, n_grid):
        grid = np.asarray(make_log_age_grid(n_grid))
        xi = jax.random.normal(jax.random.PRNGKey(20260906), (n_draws, n_grid))
        gp = jax.vmap(lambda x: _field_gp(x, jnp.asarray(grid), n_grid))(xi)
        return np.asarray(gp), 10.0**grid

    def test_variance_is_sigma_squared_in_dex(self):
        """Grid-averaged sample variance of log10(SFR) fluctuations vs sigma^2.

        Measured 0.089844 dex^2 against sigma^2 = 0.09, a 0.17 % deviation. The
        assertion is at 5 %: single-node sample variances scatter by ~+-4.5 % at
        this ensemble size, so the grid average is the statistic with the
        headroom, and the seed is fixed so the number is reproducible.
        """
        gp, _ = self._draws(self.N_DRAWS, self.N_GRID)
        var = (gp / LN10).var(axis=0, ddof=1)
        rel = abs(var.mean() / SIGMA_DEX**2 - 1.0)
        assert rel < 0.05, (
            f"grid-averaged variance {var.mean():.6f} dex^2 vs sigma^2 = "
            f"{SIGMA_DEX**2:.6f} ({100 * rel:.2f} % off)"
        )

    def test_acf_at_lag_tau_is_one_over_e(self):
        """C(tau)/C(0) = e^-1 for a damped random walk, on the field's own grid.

        The log-age grid does not carry tau as an exact lag, so the check takes
        the node pair whose physical separation is closest to it (1.000047 Gyr
        against tau = 1 Gyr). Measured correlation 0.35931 vs e^-1 = 0.36788, a
        difference of 0.0086; the assertion is at 0.05, roughly four times the
        ensemble standard error.
        """
        gp, t_field = self._draws(self.N_DRAWS, self.N_GRID)
        dex = gp / LN10
        lag = np.abs(t_field[:, None] - t_field[None, :])
        lag[np.tril_indices_from(lag)] = np.inf
        i, j = np.unravel_index(np.argmin(np.abs(lag - TAU_YR)), lag.shape)
        corr = float(np.corrcoef(dex[:, i], dex[:, j])[0, 1])
        assert abs(lag[i, j] / TAU_YR - 1.0) < 0.01, "no grid pair sits near tau"
        assert abs(corr - 1.0 / np.e) < 0.05, (
            f"correlation at lag {lag[i, j] / 1e9:.6f} Gyr is {corr:.5f}; "
            f"a DRW gives e^-1 = {1 / np.e:.5f}"
        )


# ── Part 3 (c): the public build grammar ──────────────────────────────


class TestPublicApiDraw:
    """The amplitude survives the whole public path, not just the primitive.

    ``sfh={'type': ['const', 'field'], 'psd_sigma': ..., 'psd_tau_myr': ...}``
    plus ``spec.sample(key)``, which is where ``sfh_field_xi`` — the N(0, I)
    latent, one value per SFH grid node — is drawn.
    """

    N_SEEDS: ClassVar[int] = 64

    @staticmethod
    def _common(ssp):
        return dict(
            ssp_data=ssp,
            met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "power_law",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.05),
            n_grid=256,
        )

    #: A flat mean SFH switched on before the oldest grid node, so every node the
    #: statistic uses has a finite mean SFR to fluctuate about.
    CONST: ClassVar[dict] = {
        "log_total_mass": Fixed(10.0),
        "start_gyr": Fixed(13.0),
        "end_gyr": Fixed(0.0),
    }

    def test_field_draw_amplitude_in_dex(self, synthetic_ssp_wide):
        """std of log10(SFR / SFR_mean) = 0.3 dex through the built model.

        The estimator is the pairwise difference ``r_i - r_j`` at node pairs
        separated by more than 6 tau, whose variance is ``2 sigma^2 (1 - rho)``
        with ``rho -> 0``. Differencing is what makes it exact: the pipeline
        renormalizes each realization to the requested ``log_total_mass``, and
        that per-draw constant cancels in the difference while biasing a naive
        within-realization spatial std low (measured 0.203 dex).

        Measured 0.3039 dex over 64 seeds, and 0.2956 / 0.2968 over two further
        disjoint seed sets — so the estimator itself is good to ~1.5 %. The
        assertion is nonetheless at 15 %, because the amplitude is already
        pinned to 0.17 % by the Monte Carlo above; what this one has to
        discriminate is a *convention*, and the two live candidates are far
        outside it — 0.130 dex if ``psd_sigma`` were read as a natural-log
        amplitude, 0.691 dex if it were multiplied by ln10 instead of the
        field being.
        """
        common = self._common(synthetic_ssp_wide)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_field = SEDModel.build(
                sfh={
                    "type": ["const", "field"],
                    **self.CONST,
                    "psd_sigma": Fixed(SIGMA_DEX),
                    "psd_tau_myr": Fixed(TAU_YR / 1e6),
                    "all_params": Fixed(DEFAULT),
                },
                **common,
            )
            m_mean = SEDModel.build(
                sfh={"type": "const", **self.CONST, "all_params": Fixed(DEFAULT)},
                **common,
            )

            state_mean = m_mean.predict_state(m_mean.spec.sample(jax.random.PRNGKey(0)))
            sfr_mean = np.asarray(state_mean.derived["sfr_history"])
            t_lbt = np.asarray(state_mean.derived["sfh_grid_lbt_yr"])

            ratios = []
            for seed in range(self.N_SEEDS):
                params = m_field.spec.sample(jax.random.PRNGKey(seed))
                assert np.shape(params["sfh_field_xi"]) == (256,)
                sfr = np.asarray(m_field.predict_state(params).derived["sfr_history"])
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratios.append(np.log10(sfr / sfr_mean))
        r = np.array(ratios)

        ok = np.nonzero(np.isfinite(r).all(axis=0))[0]
        pairs = [
            (i, j)
            for a, i in enumerate(ok)
            for j in ok[a + 1 :]
            if abs(t_lbt[i] - t_lbt[j]) > 6.0 * TAU_YR
        ]
        assert len(pairs) > 100, "not enough well-separated node pairs to estimate sigma"
        diffs = np.array([r[:, i] - r[:, j] for i, j in pairs])
        sigma_hat = float(np.sqrt(diffs.var(axis=1, ddof=1).mean() / 2.0))

        assert abs(sigma_hat / SIGMA_DEX - 1.0) < 0.15, (
            f"public-API field draw has std {sigma_hat:.4f} dex; psd_sigma = "
            f"{SIGMA_DEX} dex was requested (a natural-log reading would give "
            f"{SIGMA_DEX / LN10:.4f}, an ln10-scaled one {SIGMA_DEX * LN10:.4f})"
        )


# ── Part 4 (d): the log-normal centering ──────────────────────────────


class TestLogNormalCentering:
    """tengri's -K(0)/2 term, and the bias Synthesizer carries without it."""

    N_DRAWS = 4000
    N_GRID = 256

    def test_tengri_modulation_is_mean_preserving(self):
        """E[SFR / SFR_mean] = 1, which is what -K(0)/2 buys.

        The correction is taken from :func:`compute_field_gp`'s own second
        return value, not recomputed here: the composed SFH applies
        ``exp(gp_x - k0_half)`` with exactly that number, so a build that
        stopped returning it — Synthesizer's convention — has to be visible.

        Measured 0.99377 grid-averaged over 4000 draws (worst node 0.9732).
        A log-normal with sigma_ln = 0.691 has std 0.782, so one node's
        standard error at this ensemble size is 0.0124; the assertion is at
        0.05 grid-averaged, four standard errors, with the seed fixed.
        """
        grid = jnp.asarray(make_log_age_grid(self.N_GRID))
        xi = jax.random.normal(jax.random.PRNGKey(20260906), (self.N_DRAWS, self.N_GRID))
        gp, k0_half = jax.vmap(lambda x: _field_gp_and_k0(x, grid, self.N_GRID))(xi)
        k0_half = np.asarray(k0_half)
        assert np.allclose(k0_half, K0_HALF, rtol=1e-12), (
            f"the field returns K(0)/2 = {k0_half.flat[0]:.9f}; (sigma ln10)^2 / 2 = {K0_HALF:.9f}"
        )
        modulation = np.exp(np.asarray(gp) - k0_half[:, None])
        mean = modulation.mean(axis=0)
        assert abs(mean.mean() - 1.0) < 0.05, (
            f"grid-averaged E[SFR/SFR_mean] = {mean.mean():.5f}; the -K(0)/2 "
            f"correction should make it 1"
        )

    def test_synthesizer_linear_mean_is_biased_high(self):
        """Synthesizer omits the correction, so its mean linear SFR is high.

        ``sfr_cosmic = 10**(log10(base) + fluctuations)`` with zero-mean
        fluctuations has ensemble mean ``exp(K(0)/2) * base``. At sigma = 0.3 dex
        that is ``exp(0.238585) = 1.26945``, i.e. **+26.95 %**. Measured through
        the reference sampler: 1.2696.

        This is the one place a normalization carried between the two codes
        needs an explicit conversion; recorded in
        ``reproduction/synthesizer/README.md``.
        """
        t_yr, _ = _uniform_cosmic_grid()
        cov = synthesizer_build_covariance_matrix(t_yr, SIGMA_DEX, TAU_YR)
        fluct = synthesizer_sample_multivariate_normal(
            cov, np.random.default_rng(20260906), size=40000
        )
        mean_linear = (10.0**fluct).mean(axis=0)

        analytic = np.exp(K0_HALF)
        assert abs(mean_linear.mean() / analytic - 1.0) < 0.02, (
            f"Synthesizer's mean linear SFR is {mean_linear.mean():.5f} x base; "
            f"exp(K(0)/2) = {analytic:.5f}"
        )
        assert abs(analytic - 1.26945) < 1e-4, "the recorded +26.95 % convention offset has moved"
