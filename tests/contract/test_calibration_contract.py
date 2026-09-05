# SPDX-License-Identifier: BSD-3-Clause
"""Contract: calibration coefficients, the kernel cache, and the two exclusive modes.

Three seams around the spectroscopic Chebyshev calibration, each of which failed
silently at some point:

1. **Kernel-cache color-leak.** ``SEDModel.compile_signature()`` did not carry
   ``calibration_order``, so two models differing only in that order shared a
   compiled kernel. The compiled closure captures an ``Observation`` whose
   projector reads ``cal_c1..cN`` out of the param dict, so the second model
   inherited the first's coefficient lookup — raising ``KeyError: 'cal_c1'`` on a
   dict that rightly has no such key, or (worse, in the reverse order) applying a
   calibration the caller never supplied.

2. **Coefficient-name drift.** The names are declared in one place
   (``get_calibration_params``) and consumed in another
   (``calibration_coeffs``). When those drifted apart, the plotting layer went
   looking for a ``cal_c0`` the model never produces.

3. **Double-counted calibration.** Fitting ``cal_c1..cN`` explicitly *and*
   marginalizing the polynomial analytically applies it twice.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, Observation, SEDModel, Spectroscopy
from tengri.config.exceptions import ConfigError

pytestmark = pytest.mark.contract

_WAVE_OBS = jnp.linspace(4000.0, 9000.0, 300)
_SFH = {"type": "dpl", "all_params": Fixed(DEFAULT)}
_DUST = {"type": "single_component", "law": "calzetti", "all_params": Fixed(DEFAULT)}


@pytest.fixture(scope="module")
def ssp():
    try:
        return tengri.load_ssp()
    except FileNotFoundError as exc:
        pytest.skip(f"SSP data not on disk (CI runner): {exc}")


def _model(ssp, calibration_order: int, **spec_kw) -> SEDModel:
    spectroscopy = Spectroscopy(
        wave_obs=_WAVE_OBS,
        resolution=1000.0,
        calibration_order=calibration_order,
        **spec_kw,
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=spectroscopy),
        sfh=_SFH,
        dust_attenuation=_DUST,
        redshift=Fixed(0.1),
    )


def _params(model: SEDModel, **cal) -> dict:
    base = {name: float(v) for name, v in model.spec.get_fixed_values().items()}
    return {**base, **cal}


def test_compile_signature_separates_calibration_order(ssp):
    """Two models differing only in calibration_order must not share a cache slot."""
    cal2 = _model(ssp, calibration_order=2)
    cal0 = _model(ssp, calibration_order=0)
    assert cal2.compile_signature() != cal0.compile_signature()


def test_calibration_order_zero_model_survives_a_warm_kernel_cache(ssp):
    """Build+run a calibrated model FIRST, then an uncalibrated one, in one process.

    Order matters: the leak only shows when the calibrated model warms the cache
    first. Pre-fix, the second ``predict_spectrum`` raised ``KeyError: 'cal_c1'``
    because it reused the first model's compiled kernel.
    """
    cal2 = _model(ssp, calibration_order=2)
    warmed = np.asarray(cal2.predict_spectrum(_params(cal2, cal_c1=0.3, cal_c2=0.0)))
    assert np.all(np.isfinite(warmed))

    cal0 = _model(ssp, calibration_order=0)
    plain = np.asarray(cal0.predict_spectrum(_params(cal0)))
    assert np.all(np.isfinite(plain))

    # The uncalibrated model must NOT have inherited the 0.3 tilt.
    flat = np.asarray(cal2.predict_spectrum(_params(cal2, cal_c1=0.0, cal_c2=0.0)))
    np.testing.assert_allclose(plain, flat, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_calibration_coeffs_reads_exactly_the_declared_keys(order):
    """The assembler consumes precisely the names the declaration registers."""
    spectroscopy = Spectroscopy(wave_obs=_WAVE_OBS, calibration_order=order)
    declared = list(spectroscopy.get_calibration_params())

    assert declared == [f"cal_c{i + 1}" for i in range(order)]
    assert "cal_c0" not in declared  # the constant is fixed at 1, never a parameter

    params = {name: 0.1 * (i + 1) for i, name in enumerate(declared)}
    coeffs = spectroscopy.calibration_coeffs(params)

    if order == 0:
        assert coeffs is None
    else:
        np.testing.assert_allclose(
            np.asarray(coeffs), [0.1 * (i + 1) for i in range(order)], rtol=1e-12
        )


def test_calibration_wave_range_is_the_configured_instrument_grid():
    """The polynomial is anchored to wave_obs, so a coefficient keeps its meaning."""
    spectroscopy = Spectroscopy(wave_obs=_WAVE_OBS, calibration_order=2)
    lo, hi = spectroscopy.calibration_wave_range
    assert float(lo) == pytest.approx(4000.0)
    assert float(hi) == pytest.approx(9000.0)


def test_marginalize_plus_explicit_calibration_raises_config_error(ssp):
    """calibration_marginalize XOR calibration_order>0 — both would double-count.

    The Fitter builds its likelihood eagerly, so the conflict surfaces at fit
    setup rather than at the first likelihood evaluation. That is the point: the
    user learns before burning a sampling run.
    """
    model = _model(ssp, calibration_order=3)
    flux = model.predict_spectrum(_params(model, cal_c1=0.0, cal_c2=0.0, cal_c3=0.0))
    noise = 0.05 * jnp.abs(flux).mean() * jnp.ones_like(flux)

    with pytest.raises(ConfigError, match="calibration"):
        tengri.Fitter(
            model,
            data=flux,
            noise=noise,
            data_type="spectroscopy",
            calibration_marginalize=True,
        )


def test_marginalize_alone_is_still_allowed(ssp):
    """The analytic path must stay usable: order=0 + marginalize is the valid pairing."""
    model = _model(ssp, calibration_order=0)
    flux = model.predict_spectrum(_params(model))
    noise = 0.05 * jnp.abs(flux).mean() * jnp.ones_like(flux)

    fitter = tengri.Fitter(
        model,
        data=flux,
        noise=noise,
        data_type="spectroscopy",
        calibration_marginalize=True,
    )
    assert fitter is not None
