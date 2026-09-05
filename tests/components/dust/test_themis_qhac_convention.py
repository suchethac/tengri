# SPDX-License-Identifier: BSD-3-Clause
"""Regression: THEMIS ``qhac`` parameter is honored, not clipped to grid min.

CIGALE's THEMIS ``qhac`` (a-C(:H) aromatic carbon mass fraction) sweeps
``[0.02, 0.40]`` and defaults to 0.17 (the diffuse-ISM standard); tengri's
``ThemisIRSEDComponent`` exposes the same convention (``qhac = Fixed(0.17)``).

The shipped template grid (``data/themis_templates.h5``) stores its qhac axis
in FSPS scaling (CIGALE value x 100/2.2, i.e. ``[0.909 .. 18.18]``). The
interpolator looked up the user-facing CIGALE value directly against that
axis, so every physical qhac < 0.909 (which is the entire CIGALE range,
including the 0.17 default) silently clipped to the grid minimum and selected
the wrong grain composition. That flattened the whole sweep and shifted the
mid-IR PAH strength and FIR peak by tens of percent versus CIGALE.

These tests pin the user-facing convention: distinct qhac values must give
distinct grain models, and the default must not be the grid minimum.
"""

import numpy as np
import pytest

from tengri.components.dust.emission import themis as themis_emission

pytestmark = pytest.mark.regression_bug

_WAVE = np.logspace(3.5, 7.0, 1200)  # 3162 A .. 1e7 A
_L_IR = 1.0  # normalized; templates conserve absorbed energy


def _emit(qhac, alpha=2.0):
    return np.asarray(
        themis_emission(
            _WAVE, _L_IR, dust_umin=1.0, dust_gamma_dl=0.1, dust_qhac=qhac, dust_alpha=alpha
        )
    )


def _rel_maxdiff(a, b):
    """Peak-relative max |a - b|.

    The SED is energy-normalized to L_ir=1, so absolute values are ~1e-13 and a
    default-atol ``np.allclose`` would pass trivially. Compare against the SED
    peak so the assertion bites at the true scale.
    """
    peak = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(a - b))) / peak


def test_default_qhac_is_not_grid_minimum():
    """qhac=0.17 (CIGALE default) must not equal the smallest grid grain model.

    Pre-fix, 0.17 clipped to the grid minimum, so 0.17 and 0.02 produced an
    identical spectrum. They must differ by more than a percent of the peak.
    """
    rel = _rel_maxdiff(_emit(0.17), _emit(0.02))
    assert rel > 0.01, (
        f"qhac=0.17 collapses to the minimum grain model (rel diff {rel:.3e}) — "
        "CIGALE convention not applied before grid interpolation"
    )


def test_qhac_sweep_changes_spectrum():
    """Distinct qhac across CIGALE's [0.02, 0.40] range give distinct spectra."""
    seds = [_emit(q) for q in (0.02, 0.10, 0.17, 0.30, 0.40)]
    for i in range(len(seds) - 1):
        rel = _rel_maxdiff(seds[i], seds[i + 1])
        assert rel > 0.01, (
            f"qhac sweep step {i} produced no change (rel diff {rel:.3e}) — values are clipping"
        )


def test_qhac_energy_conserved_across_sweep():
    """Grain composition changes the shape but not the absorbed-energy budget."""
    c_aa = 2.998e18
    bands = []
    for q in (0.02, 0.17, 0.40):
        sed = _emit(q)
        bands.append(np.trapezoid((c_aa / _WAVE) * sed, np.log(_WAVE)))
    bands = np.array(bands)
    # THEMIS is energy-balance-normalized: the band-integrated luminosity is
    # nearly invariant under qhac (shape redistributes, total is conserved).
    np.testing.assert_allclose(bands, bands[0], rtol=5e-2)


def test_qhac_axis_conversion_under_jit():
    """Regression #2114: _qhac_axis_to_cigale must not raise under trace.

    The component chain is built lazily inside the first traced call to
    predict_photometry. If _qhac_axis_to_cigale runs inside a trace,
    float(jnp.max(...)) on a traced array raises ConcretizationTypeError.
    The fix keeps the convention select inside the traced graph via jnp.where.
    """
    import jax
    import jax.numpy as jnp

    from tengri.components.dust.emission_templates import _qhac_axis_to_cigale

    # FSPS-scaled grid: max > 0.5 → needs conversion
    fsps_grid = jnp.array([0.909, 4.545, 9.091, 13.636, 18.18])
    cigale_expected = fsps_grid * 2.2 / 100.0

    # JIT the conversion to trace all array operations
    jitted_convert = jax.jit(_qhac_axis_to_cigale)
    result = jitted_convert(fsps_grid)
    np.testing.assert_allclose(result, cigale_expected, rtol=1e-14)

    # CIGALE-scaled grid: max <= 0.5 → no conversion
    cigale_grid = jnp.array([0.02, 0.10, 0.17, 0.30, 0.40])
    result = jitted_convert(cigale_grid)
    np.testing.assert_allclose(result, cigale_grid, rtol=1e-14)

    # Gradient through the conversion: this would fail with ConcretizationTypeError
    # in the pre-fix code when running inside jax.grad
    def wrapper(qhac_grid):
        converted = _qhac_axis_to_cigale(qhac_grid)
        return jnp.sum(converted)  # Scalar for grad

    # This must not raise even though qhac_grid is under trace, and the
    # select resolves to one branch, so the gradient is the conversion
    # factor on an FSPS grid and exactly 1 on a CIGALE grid.
    grad_fn = jax.grad(wrapper)
    np.testing.assert_allclose(grad_fn(fsps_grid), 2.2 / 100.0, rtol=1e-14)
    np.testing.assert_allclose(grad_fn(cigale_grid), 1.0, rtol=1e-14)
