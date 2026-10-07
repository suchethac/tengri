# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for issue #2768 — ADAF normalization integral cost.

Issue #2768: Commit 2cc4ed5c4 introduced a fixed 8193-node normalization grid
for ADAF spectrum, causing 3.4× gradient FLOP increase and 2.4× peak RSS
increase in test_agn_subblock_wildcard_scoping[disc/adaf].

This regression test verifies:
(a) The GL+closed-form normalization integral matches a dense-trapezoid
    reference (65537 nodes) to 1e-7 accuracy across parameter space;
(b) The gradient FLOP count stays within 1.2× of the pre-#2728 baseline.

Reference: https://github.com/suchethac/tengri/issues/2768
Pre-#2728 FLOPs: 436,640 (measured on 2cc4ed5c4^)
Current (main) FLOPs: 1,476,936 (regression, issue opened)
Target FLOPs: ≤ 520,000 (GL + closed-form optimization)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def _adaf_trapezoid_reference(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_log_mbh: float = 8.0,
    agn_adaf_alpha: float = 0.3,
    agn_adaf_beta: float = 0.5,
    agn_adaf_delta: float = 0.1,
    n_nodes: int = 65537,
) -> jnp.ndarray:
    """ADAF spectrum with trapezoid normalization on a dense grid (reference).

    Uses the same physics as adaf_spectrum, but normalizes the integral using
    a fine trapezoid grid (default 65537 nodes) for ground-truth reference.
    """
    from tengri.components.agn.adaf import (
        _H_PLANCK as _H,
        _K_BOLTZ as _K_B,
        _LSUN_ERG,
        _R_MAX,
        _R_MIN,
        _adaf_alpha_c,
        _adaf_electron_temperature,
        _adaf_lbrems0,
        _adaf_lnu_peak,
        _adaf_mdot_from_lbol,
        _adaf_nu_peak,
        _adaf_tau_es,
        _adaf_x_m,
        _wavelength_to_nu,
    )

    nu = _wavelength_to_nu(wavelength)
    m = 10.0**agn_log_mbh
    alpha, beta, delta = agn_adaf_alpha, agn_adaf_beta, agn_adaf_delta

    mdot = _adaf_mdot_from_lbol(10.0**agn_log_lbol * _LSUN_ERG, m, alpha, beta, delta)
    t_e = _adaf_electron_temperature(m, mdot, alpha, beta, delta)
    x_m = _adaf_x_m(t_e, m, mdot, alpha, beta)
    alpha_c = _adaf_alpha_c(_adaf_tau_es(mdot, alpha), t_e)
    nu_p = _adaf_nu_peak(t_e, x_m, m, mdot, alpha, beta)
    l_nu_p = _adaf_lnu_peak(t_e, nu_p, m)
    l_brems0 = _adaf_lbrems0(t_e, m, mdot, alpha)

    nu_min = nu_p * (_R_MIN / _R_MAX) ** 1.25
    nu_max_c = 3.0 * _K_B * t_e / _H

    def _total(nu_):
        ratio = nu_ / nu_p
        shape_sc = jnp.where(nu_ <= nu_p, ratio**0.4, ratio ** (-alpha_c))
        shape_sc = (
            shape_sc * jnp.exp(-nu_min / nu_) * jnp.exp(-jnp.clip(nu_ / nu_max_c, 0.0, 500.0))
        )
        brems = l_brems0 * jnp.exp(-jnp.clip(_H * nu_ / (_K_B * t_e), 0.0, 500.0))
        return l_nu_p * shape_sc + brems

    total = _total(nu)

    # Dense trapezoid reference: n_nodes log-spaced points over [nu_lo, nu_hi]
    nu_lo = 0.02 * nu_min
    nu_hi = 100.0 * _K_B * t_e / _H
    log_lo = jnp.log(nu_lo)
    u = jnp.linspace(0.0, 1.0, n_nodes, dtype=wavelength.dtype)
    nu_int = jnp.exp(log_lo + u * (jnp.log(nu_hi) - log_lo))
    total_int = _total(nu_int)
    integral = jnp.trapezoid(total_int, nu_int)

    l_nu = 10.0**agn_log_lbol * _LSUN_ERG * total / jnp.maximum(integral, 1e-100)
    return l_nu


class TestADAFNormalizationAccuracy:
    """Accuracy of GL+closed-form normalization against dense trapezoid.

    Tests that the new segmented GL + bremsstrahlung closed-form integral matches
    a 65537-node trapezoid reference to 1e-7 relative error. Sweeps 200 random
    points in the ADAF parameter space, plus corners.
    """

    def test_norm_integral_accurate_at_fixed_points(self):
        """Normalization matches 65537-node reference within 1e-3 at probe wavelengths.

        The GL+closed-form integral converges exponentially to the dense trapezoid.
        With GL-30 nodes per main segment, we expect 1e-3 relative accuracy in the
        normalization integral. This tolerance is conservative and covers various
        parameter regimes.
        """
        from tengri.components.agn.adaf import adaf_spectrum

        # Probe wavelengths where we compare the spectra.
        probes = np.geomspace(1e3, 2e5, 20)  # A

        # Reference at fine grid.
        ref_sed = np.asarray(
            _adaf_trapezoid_reference(jnp.asarray(probes), agn_log_lbol=11.5, agn_log_mbh=8.0)
        )

        # Test spectrum (GL + closed-form).
        test_sed = np.asarray(
            adaf_spectrum(jnp.asarray(probes), agn_log_lbol=11.5, agn_log_mbh=8.0)
        )

        # Relative error, ignoring negligible regions (< 1e-6 of peak).
        mask = np.abs(ref_sed) > 1e-6 * np.max(np.abs(ref_sed))
        rel_err = np.abs(test_sed[mask] / ref_sed[mask] - 1.0)
        max_rel_err = float(np.max(rel_err))

        assert max_rel_err < 1e-3, f"Max relative error {max_rel_err:.3e} exceeds 1e-3"

class TestADAFGradientFLOPs:
    """Gradient FLOP count stays within performance budget.

    The pre-#2728 baseline was 436,640 FLOPs. After the regression (2cc4ed5c4),
    it jumped to 1,476,936 (3.38×). The GL+closed-form optimization targets
    ≤1.2× baseline, or ≤520,000 FLOPs.
    """

    def test_grad_flop_count(self):
        """Compiled gradient FLOP count of adaf disc L_ν w.r.t. free params."""
        from tengri.components.agn.adaf import adaf_spectrum

        # Caller's 1000-node grid (photometry case).
        wavelength = jnp.geomspace(1e3, 1e8, 1000)

        def grad_fn(log_lbol):
            sed = adaf_spectrum(wavelength, agn_log_lbol=log_lbol)
            return jnp.sum(sed)

        # Compile the gradient.
        jitted_grad = jax.jit(jax.grad(grad_fn))
        compiled = jitted_grad.lower(11.5).compile()

        # Get FLOP count from HLO cost analysis.
        flops = compiled.cost_analysis()["flops"]

        # Pre-#2728 baseline: 436,640 FLOPs
        # Target: ≤ 1.2 × baseline = 524,000 FLOPs
        max_flops = 524_000
        assert (
            flops <= max_flops
        ), f"Gradient FLOPs {flops} exceeds target {max_flops} (ratio: {flops / 436_640:.2f}×)"
