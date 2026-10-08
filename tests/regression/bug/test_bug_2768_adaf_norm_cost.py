# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for #2768: the ADAF normalization must be accurate and cheap under grad.

#2728 normalized ``adaf_spectrum`` on an 8193-node fixed grid and left the ADAF disc without a
closed-form power, so the composable runner integrated it again on its 13001-node budget
grid whenever a line block debited the disc. Gradient FLOPs (compiled-graph cost analysis,
caller grid 1000 nodes, ``agn_log_lbol`` and ``agn_log_mbh`` traced):

========================================  ==========  =========  ==========
graph                                     before      this fix   budget
========================================  ==========  =========  ==========
graph                                     main        this fix   budget
========================================  ==========  =========  ==========
``adaf_spectrum``                         1 423 587   393 826    710 000
``compose_l_nu``, ADAF only               1 438 905   409 538    710 000
``compose_l_nu``, ADAF + NLR + BLR        5 364 822   1 040 749  1 870 000
========================================  ==========  =========  ==========

Budgets are absolute, 1.8 x the value measured with this JAX/XLA (so a compiler upgrade that
moves ``cost_analysis`` accounting by tens of percent cannot flake them) and still 2 x below the
cost on main for the first two rows and 2.9 x for the last. Before #2728 the spectrum alone
cost 383 365 (436 640 on the caller grid of the original measurement). The NLR and BLR line
profiles add about 630 000 to the last row for every disc (a multicolor disc gains 670 000
from them); that part is not the ADAF's, so its budget is the measured total.

Accuracy is measured against an independent quadrature: 64 panels of 16-point Gauss-Legendre
on each of the spectrum's four log-frequency segments, which resolves the integrand to
round-off.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import roots_legendre

from tengri.components.agn import adaf as adaf_module
from tengri.components.agn.adaf import adaf_spectrum
from tengri.components.agn.blocks import _protocol
from tengri.components.agn.blocks.disc import adaf_disc_block
from tengri.components.agn.blocks.runner import compose_l_nu
from tengri.utils.physics_constants import C_AA, L_SUN

pytestmark = pytest.mark.regression_bug

_ACCURACY = 1.0e-7
_N_PRIOR_SAMPLES = 200
_SPECTRUM_FLOP_BUDGET = 710_000  # 1.8 x 393 826 measured
_COMPOSITION_FLOP_BUDGET = 710_000  # 1.8 x 409 538 measured
_COMPOSITION_LINES_FLOP_BUDGET = 1_870_000  # 1.8 x 1 040 749 measured
_CALLER_GRID = np.geomspace(1.0e3, 1.0e8, 1000)  # [Angstrom]
_PROBES = np.geomspace(1.0e3, 2.0e5, 24)  # [Angstrom]

# Declared priors of the ADAF parameters, and the black-hole mass / luminosity range a fit spans.
_RANGES = {
    "agn_log_lbol": (8.0, 14.0),  # [log10 Lsun]
    "agn_log_mbh": (6.0, 10.0),  # [log10 Msun]
    "agn_adaf_alpha": (0.05, 0.5),
    "agn_adaf_beta": (0.1, 0.9),
    "agn_adaf_delta": (0.001, 0.5),
}
_NAMES = tuple(_RANGES)


def _parameter_sets() -> np.ndarray:
    """200 prior draws followed by the 32 corners of the parameter box, shape (232, 5)."""
    rng = np.random.default_rng(2768)
    lo = np.array([_RANGES[n][0] for n in _NAMES])
    hi = np.array([_RANGES[n][1] for n in _NAMES])
    draws = lo + (hi - lo) * rng.random((_N_PRIOR_SAMPLES, len(_NAMES)))
    corners = np.array(np.meshgrid(*zip(lo, hi), indexing="ij")).reshape(len(_NAMES), -1).T
    return np.concatenate([draws, corners])


def _spectrum_at_probes(x):
    """``adaf_spectrum`` at the probe wavelengths for the parameter vector ``x``."""
    return adaf_spectrum(
        jnp.asarray(_PROBES),
        agn_log_lbol=x[0],
        agn_log_mbh=x[1],
        agn_adaf_alpha=x[2],
        agn_adaf_beta=x[3],
        agn_adaf_delta=x[4],
    )


def _gl_panels(log_lo, log_hi, n_panels=64, order=16):
    """Nodes and weights of ``n_panels`` Gauss-Legendre panels over ``[log_lo, log_hi]``."""
    x, w = roots_legendre(order)
    edges = log_lo + (log_hi - log_lo) * jnp.linspace(0.0, 1.0, n_panels + 1)
    half = 0.5 * (edges[1:] - edges[:-1])
    mid = 0.5 * (edges[1:] + edges[:-1])
    nodes = (mid[:, None] + half[:, None] * jnp.asarray(x)[None, :]).ravel()
    weights = (half[:, None] * jnp.asarray(w)[None, :]).ravel()
    return nodes, weights


def _reference_spectrum_at_probes(x):
    """The spectrum normalized by the dense composite quadrature, at the probe wavelengths."""
    m = adaf_module
    log_lbol, log_mbh, alpha, beta, delta = x
    mass = 10.0**log_mbh
    mdot = m._adaf_mdot_from_lbol(10.0**log_lbol * m._LSUN_ERG, mass, alpha, beta, delta)
    t_e = m._adaf_electron_temperature(mass, mdot, alpha, beta, delta)
    x_m = m._adaf_x_m(t_e, mass, mdot, alpha, beta)
    alpha_c = m._adaf_alpha_c(m._adaf_tau_es(mdot, alpha), t_e)
    nu_p = m._adaf_nu_peak(t_e, x_m, mass, mdot, alpha, beta)
    l_nu_p = m._adaf_lnu_peak(t_e, nu_p, mass)
    l_brems0 = m._adaf_lbrems0(t_e, mass, mdot, alpha)
    nu_min = nu_p * (m._R_MIN / m._R_MAX) ** 1.25
    nu_max_c = 3.0 * m._K_BOLTZ * t_e / m._H_PLANCK

    def total(nu):
        ratio = nu / nu_p
        shape = jnp.where(nu <= nu_p, ratio**0.4, ratio ** (-alpha_c))
        shape = shape * jnp.exp(-nu_min / nu) * jnp.exp(-jnp.clip(nu / nu_max_c, 0.0, 500.0))
        brems = l_brems0 * jnp.exp(-jnp.clip(m._H_PLANCK * nu / (m._K_BOLTZ * t_e), 0.0, 500.0))
        return l_nu_p * shape + brems

    breaks = jnp.log(
        jnp.stack([0.02 * nu_min, nu_min, nu_p, nu_max_c, 100.0 * m._K_BOLTZ * t_e / m._H_PLANCK])
    )
    integral = 0.0
    for i in range(4):
        log_nu, weight = _gl_panels(breaks[i], breaks[i + 1])
        nu = jnp.exp(log_nu)
        integral = integral + jnp.sum(weight * total(nu) * nu)
    nu_probe = m._wavelength_to_nu(jnp.asarray(_PROBES))
    return 10.0**log_lbol * m._LSUN_ERG * total(nu_probe) / integral


@pytest.fixture(scope="module")
def _parameters():
    return jnp.asarray(_parameter_sets())


class TestNormalizationAccuracy:
    """The segmented quadrature reproduces the dense reference to 1e-7."""

    def test_prior_samples_and_corners(self, _parameters):
        """Max |SED / reference - 1| over 200 prior draws and the 32 corners is below 1e-7."""
        got = np.asarray(jax.jit(jax.vmap(_spectrum_at_probes))(_parameters))
        ref = np.asarray(jax.jit(jax.vmap(_reference_spectrum_at_probes))(_parameters))
        assert got.shape == (len(_parameter_sets()), _PROBES.size)
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref))
        keep = ref > 1.0e-6 * ref.max(axis=1, keepdims=True)
        worst = float(np.max(np.abs(got[keep] / ref[keep] - 1.0)))
        assert worst < _ACCURACY, f"max relative error {worst:.3e} exceeds {_ACCURACY:.0e}"


class TestClosedFormPower:
    """The ADAF disc registers its power, so the runner never integrates it on a grid."""

    def test_registered(self):
        """``DISC_POWER_BLOCKS`` carries ``adaf`` and returns 1 in units of L_acc."""
        assert "adaf" in _protocol.DISC_POWER_BLOCKS
        assert float(_protocol.DISC_POWER_BLOCKS["adaf"](11.5)) == 1.0

    @pytest.mark.parametrize("log_mbh", [6.5, 8.0, 9.5])
    def test_matches_the_integral_of_the_block(self, log_mbh):
        """The block's L_lambda integrates to ``L_acc`` over 1e-5 A - 1e12 A to 1e-4."""
        wave = jnp.asarray(np.geomspace(1.0e-5, 1.0e12, 400_001))
        l_lambda = adaf_disc_block(wave, agn_log_lbol=11.5, agn_log_mbh=log_mbh)
        nu = C_AA / np.asarray(wave)
        l_nu = np.asarray(l_lambda) * np.asarray(wave) ** 2 / C_AA
        power = float(np.trapezoid(l_nu[::-1], nu[::-1]))
        assert power / (10.0**11.5 * L_SUN) == pytest.approx(1.0, abs=1.0e-4)


def _flops(fn, x0) -> float:
    return jax.jit(jax.grad(fn)).lower(x0).compile().cost_analysis()["flops"]


class TestGradientCost:
    """Compiled gradient FLOPs stay within the budgets in the module docstring."""

    def test_adaf_spectrum_alone(self):
        wave = jnp.asarray(_CALLER_GRID)

        def fn(x):
            return jnp.sum(adaf_spectrum(wave, agn_log_lbol=x[0], agn_log_mbh=x[1]))

        flops = _flops(fn, jnp.array([11.5, 8.0]))
        assert flops <= _SPECTRUM_FLOP_BUDGET, f"{flops:,.0f} > {_SPECTRUM_FLOP_BUDGET:,}"

    @pytest.mark.parametrize(
        ("lines", "budget"),
        [
            (("none", "none"), _COMPOSITION_FLOP_BUDGET),
            (("analytic", "analytic"), _COMPOSITION_LINES_FLOP_BUDGET),
        ],
        ids=["no_lines", "nlr_blr"],
    )
    def test_conserving_composition(self, lines, budget):
        """``compose_l_nu`` with the conserving ledger never integrates the ADAF on a grid."""
        wave = jnp.asarray(_CALLER_GRID)

        def fn(x):
            return jnp.sum(
                compose_l_nu(
                    wave,
                    x[0],
                    agn_cos_inc=0.5,
                    agn_log_mbh=x[1],
                    agn_disc_block="adaf",
                    agn_torus_block="none",
                    agn_attenuation_block="none",
                    agn_norm="conserving",
                    agn_nlr_block=lines[0],
                    agn_blr_block=lines[1],
                    agn_feii_block="none",
                )
            )

        flops = _flops(fn, jnp.array([11.5, 8.0]))
        assert flops <= budget, f"{flops:,.0f} > {budget:,}"
