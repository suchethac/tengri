# SPDX-License-Identifier: BSD-3-Clause
"""Leja+2017 Dirichlet SFH: the prior lives on the *SFR* fractions.

Leja et al. 2017 (ApJ 837, 170, Sect. 2.3 and Appendix) parameterizes the
Dirichlet SFH with ``N - 1`` auxiliary variables that stick-break the
**SFR fractions**, not the mass fractions, so that the vector of SFR
fractions is exactly symmetric ``Dirichlet(1, ..., 1)``. Bin masses then
follow as ``m_i \\propto sfr\\_frac_i \\Delta t_i``.

tengri exposes the auxiliaries as ``sfh_dir_z_i ~ Uniform(0, 1)`` and maps
each one through the ``Beta(1, N-1-i)`` quantile inside
:func:`~tengri.components.stellar.sfh.nonparametric.dirichlet`.

Before this contract was pinned, the stick-breaking was applied directly to
the *mass* fractions with uniform (rather than Beta-quantile-mapped)
auxiliaries. That is not a symmetric Dirichlet in either variable: over
uniform draws the mean mass fractions were
``[0.500, 0.251, 0.125, 0.062, 0.030, 0.016, 0.015]`` and the mean SFR
fractions ``[0.720, 0.223, 0.048, 0.0072, 0.0013, 0.00044, 0.00017]``,
i.e. the prior placed ~72% of the star formation in the youngest bin.

References
----------
.. [1] J. Leja et al., "Deriving Physical Properties from Broadband
   Photometry with Prospector: Description of the Model and a Demonstration
   of its Accuracy Using 129 Galaxies in the Local Universe," ApJ, 837, 170
   (2017). arXiv:1609.09073. https://doi.org/10.3847/1538-4357/aa5ffe
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sfh.nonparametric import DEFAULT_BIN_EDGES_GYR, dirichlet

pytestmark = pytest.mark.regression_paper


_EDGES_GYR = np.asarray(DEFAULT_BIN_EDGES_GYR)
_N_BINS = _EDGES_GYR.shape[0] - 1  # 7
_BIN_CENTERS_YR = jnp.asarray(0.5 * (_EDGES_GYR[:-1] + _EDGES_GYR[1:]) * 1e9)
_BIN_WIDTHS_YR = jnp.asarray(np.diff(_EDGES_GYR) * 1e9)

_N_DRAWS = 20_000
_SEED = 20260905

# Var[X] for X ~ Beta(1, N-1) with N = 7: 6 / (7^2 * 8).
_BETA_1_6_VARIANCE = 6.0 / (7.0**2 * 8.0)


def _sfr_at_bin_centers(latents: jnp.ndarray) -> jnp.ndarray:
    """SFR in each of the 7 default bins for one latent vector, shape (6,)."""
    kwargs = {f"z_frac_{i}": latents[i] for i in range(_N_BINS - 1)}
    return dirichlet(age_yr=_BIN_CENTERS_YR, log_total_mass=0.0, **kwargs)


@pytest.fixture(scope="module")
def sfr_fractions() -> np.ndarray:
    """Monte-Carlo SFR fractions, shape (_N_DRAWS, 7), from uniform latents."""
    rng = np.random.default_rng(_SEED)
    latents = jnp.asarray(rng.uniform(0.0, 1.0, size=(_N_DRAWS, _N_BINS - 1)))
    sfr = jax.vmap(_sfr_at_bin_centers)(latents)
    return np.asarray(sfr / jnp.sum(sfr, axis=1, keepdims=True))


def test_uniform_latents_give_symmetric_dirichlet_sfr_fractions(sfr_fractions):
    """Mean SFR fraction is 1/N in every bin (Dirichlet(1,...,1) marginal mean)."""
    means = sfr_fractions.mean(axis=0)
    print("measured mean SFR fractions:", np.array2string(means, precision=6, separator=", "))
    expected = 1.0 / _N_BINS
    assert means.shape == (_N_BINS,)
    for i, m in enumerate(means):
        assert abs(m - expected) < 5e-3, (
            f"bin {i}: mean SFR fraction {m:.6f} differs from 1/{_N_BINS} = "
            f"{expected:.6f} by more than 5e-3; full means {means}"
        )


def test_first_sfr_fraction_matches_beta_1_nminus1_variance(sfr_fractions):
    """Marginal of f_0 under Dirichlet(1,...,1) is Beta(1, N-1); check its variance."""
    var_f0 = float(sfr_fractions[:, 0].var())
    print(f"measured var(f_0) = {var_f0:.6f}; Beta(1,6) variance = {_BETA_1_6_VARIANCE:.6f}")
    rel = abs(var_f0 - _BETA_1_6_VARIANCE) / _BETA_1_6_VARIANCE
    assert rel < 0.15, (
        f"var(f_0) = {var_f0:.6f} is {rel:.1%} away from the Beta(1, 6) "
        f"variance {_BETA_1_6_VARIANCE:.6f}"
    )


@pytest.mark.conservation
@pytest.mark.parametrize(
    "latents",
    [
        (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (0.1, 0.9, 0.3, 0.7, 0.2, 0.8),
        (0.05, 0.05, 0.05, 0.05, 0.05, 0.05),
        (0.95, 0.95, 0.95, 0.95, 0.95, 0.95),
    ],
)
@pytest.mark.parametrize("log_total_mass", [8.0, 10.0, 11.5])
def test_total_mass_and_positivity(latents, log_total_mass):
    """The binned SFR integrates to 10**log_total_mass and is non-negative."""
    kwargs = {f"z_frac_{i}": v for i, v in enumerate(latents)}
    sfr_bins = dirichlet(age_yr=_BIN_CENTERS_YR, log_total_mass=log_total_mass, **kwargs)

    assert np.all(np.asarray(sfr_bins) >= 0.0), f"negative SFR: {sfr_bins}"
    mass = float(jnp.sum(sfr_bins * _BIN_WIDTHS_YR))
    expected = 10.0**log_total_mass
    assert mass == pytest.approx(expected, rel=1e-10, abs=0.0)

    # Piecewise constant: two ages inside the same bin give the same SFR.
    lo = jnp.asarray(_EDGES_GYR[:-1] * 1e9) * 0.999 + jnp.asarray(_EDGES_GYR[1:] * 1e9) * 0.001
    hi = jnp.asarray(_EDGES_GYR[:-1] * 1e9) * 0.001 + jnp.asarray(_EDGES_GYR[1:] * 1e9) * 0.999
    sfr_lo = dirichlet(age_yr=lo, log_total_mass=log_total_mass, **kwargs)
    sfr_hi = dirichlet(age_yr=hi, log_total_mass=log_total_mass, **kwargs)
    np.testing.assert_allclose(np.asarray(sfr_lo), np.asarray(sfr_bins), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(sfr_hi), np.asarray(sfr_bins), rtol=1e-12)


# Frozen SFR fractions at u = (0.5,)*6, derived independently from the
# Leja+2017 construction: v_i = 1 - (1 - u_i)**(1/(N-1-i)), then
# sfr_frac_0 = v_0, sfr_frac_i = v_i * prod_{j<i}(1 - v_j),
# sfr_frac_{N-1} = prod_j (1 - v_j).
_FROZEN_HALF_LATENT_SFR_FRACTIONS = np.array(
    [
        0.10910128186,
        0.11532633722,
        0.12339634603,
        0.13454357296,
        0.15161103793,
        0.18301071199,
        0.18301071199,
    ]
)


def test_frozen_sfr_fractions_at_half_latents():
    """Deterministic pin so any future change of convention is loud."""
    kwargs = {f"z_frac_{i}": 0.5 for i in range(_N_BINS - 1)}
    sfr_bins = dirichlet(age_yr=_BIN_CENTERS_YR, log_total_mass=0.0, **kwargs)
    fractions = np.asarray(sfr_bins / jnp.sum(sfr_bins))
    np.testing.assert_allclose(fractions, _FROZEN_HALF_LATENT_SFR_FRACTIONS, rtol=0.0, atol=1e-8)
