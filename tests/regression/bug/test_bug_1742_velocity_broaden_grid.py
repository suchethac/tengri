# SPDX-License-Identifier: BSD-3-Clause
"""``velocity_broaden`` must refuse a grid it cannot broaden correctly (#1742).

The convolution is a *constant* Gaussian in ``ln(lambda)`` — that is what makes
one FFT correct for the whole array — and it is exact only when the grid is
uniform in ``ln(lambda)``. On a linear grid ``d(ln lambda) = d(lambda)/lambda``
varies as ``1/lambda``, so a width read off the first pixel pair sets the kernel
by the bluest pixel while the feature sits elsewhere, and the recovered width is
low by exactly ``wave[0] / lambda_line``.

Nothing signaled that: no exception, no NaN, just a smaller number. The
docstring made it worse by telling users to pass "uniformly spaced" — the grid
that breaks it.

This module pins both halves: the log grid stays exact, and the linear grid now
raises with the factor it would have been wrong by.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.spectrum import velocity_broaden

pytestmark = pytest.mark.regression_bug

_C_KM_S = 299792.458
_LAMBDA_LINE = 5000.0
_WAVE_LO, _WAVE_HI = 4500.0, 5500.0


def _log_grid(n=4096):
    return jnp.logspace(np.log10(_WAVE_LO), np.log10(_WAVE_HI), n)


def _linear_grid(n=4096):
    return jnp.linspace(_WAVE_LO, _WAVE_HI, n)


def _delta_at_line(wave):
    """A unit spike at the line, so the output's second moment IS the kernel."""
    idx = int(np.argmin(np.abs(np.asarray(wave) - _LAMBDA_LINE)))
    return jnp.zeros_like(wave).at[idx].set(1.0)


def _recovered_sigma_v(wave, flux):
    """Velocity width of a broadened profile, by second moment in ln(lambda)."""
    w = np.asarray(wave, dtype=np.float64)
    f = np.asarray(flux, dtype=np.float64)
    total = f.sum()
    lnw = np.log(w)
    mean = (f * lnw).sum() / total
    var = (f * (lnw - mean) ** 2).sum() / total
    return np.sqrt(var) * _C_KM_S


# ── the log grid is exact, and must stay so ───────────────────────


@pytest.mark.parametrize("sigma_v", [100.0, 200.0, 300.0, 500.0])
def test_log_grid_recovers_the_requested_width(sigma_v):
    """The implementation is exact when its precondition holds (#1742)."""
    wave = _log_grid()
    out = velocity_broaden(_delta_at_line(wave), wave, sigma_v)

    recovered = _recovered_sigma_v(wave, out)
    assert recovered == pytest.approx(sigma_v, rel=2e-3), (
        f"log-uniform grid should reproduce sigma_v exactly; asked {sigma_v}, "
        f"recovered {recovered:.3f}"
    )


# ── the linear grid is refused, loudly and with the number ────────


def test_linear_grid_raises_instead_of_under_broadening():
    """A linear grid must raise rather than return a plausible wrong answer."""
    wave = _linear_grid()

    with pytest.raises(ValueError, match="uniform in ln\\(lambda\\)") as excinfo:
        velocity_broaden(_delta_at_line(wave), wave, 200.0)

    message = str(excinfo.value)
    assert "#1742" in message
    # The remedy must be in the message: a user who hits this needs the next step,
    # not just the diagnosis.
    assert "logspace" in message, f"error did not name the remedy: {message}"


def test_the_error_quotes_the_factor_it_would_have_been_wrong_by():
    """0.900 = 4500/5000 is the whole mechanism; the message must carry it."""
    wave = _linear_grid()

    with pytest.raises(ValueError) as excinfo:
        velocity_broaden(_delta_at_line(wave), wave, 200.0)

    # wave[0] / lambda_mid, with lambda_mid the array center ~5000 A.
    assert "0.9" in str(excinfo.value), (
        f"error did not report the under-broadening factor: {excinfo.value}"
    )


# ── the guard must not fire on things that are fine ───────────────


def test_short_grids_are_not_rejected():
    """Too few pixels to characterize spacing is not the same as bad spacing."""
    wave = jnp.array([4999.0, 5000.0])
    velocity_broaden(jnp.array([0.0, 1.0]), wave, 100.0)


def test_traced_grid_is_not_rejected():
    """A traced grid cannot be inspected; the guard must skip, not crash.

    It must not raise ``ConcretizationTypeError`` trying to look at values that
    do not exist yet — that would turn a correctness guard into a JIT bug.
    """
    wave = _log_grid(256)
    flux = _delta_at_line(wave)

    out = jax.jit(velocity_broaden)(flux, wave, 150.0)
    assert np.all(np.isfinite(np.asarray(out)))


def test_jit_with_a_closed_over_linear_grid_still_raises():
    """The common JIT shape — a fixed instrument grid — must stay guarded.

    A spectroscopic grid is normally closed over rather than traced, so the check
    still has values to inspect and the guard keeps working under ``jit``.
    """
    wave = _linear_grid(256)
    flux = _delta_at_line(wave)

    with pytest.raises(ValueError, match="uniform in ln"):
        jax.jit(lambda f: velocity_broaden(f, wave, 150.0))(flux)
