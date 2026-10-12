# SPDX-License-Identifier: BSD-3-Clause
"""A traced wavelength grid must take the general (piecewise) LSF path (#2847).

``_is_log_uniform`` used to answer ``True`` for a tracer, so a linear grid passed
through ``jax.jit`` as an argument took the single-FFT constant-R path, which is
only correct on a grid uniform in ln(lambda). An unknown pixel scale must take the
general path, never the fast one.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation import spectrum as spectrum_module
from tengri.observation.spectrum import _is_log_uniform, apply_lsf

pytestmark = pytest.mark.bounds


@pytest.fixture
def linear_wave():
    """Linearly spaced grid: d(ln lambda) varies strongly across it."""
    return np.linspace(4000.0, 9000.0, 2000)


@pytest.fixture
def spectrum_in(linear_wave):
    """A narrow emission-like bump so the LSF shape is visible."""
    x = np.arange(linear_wave.size)
    return jnp.asarray(1.0 + 5.0 * np.exp(-0.5 * ((x - 900.0) / 3.0) ** 2))


def test_is_log_uniform_is_false_for_a_tracer(linear_wave):
    """A tracer has no values to inspect, so it is unknown and takes the general path."""
    seen = []

    def probe(w):
        seen.append(_is_log_uniform(w))
        return w.sum()

    jax.jit(probe)(jnp.asarray(linear_wave))
    assert seen == [False]


def test_is_log_uniform_still_true_for_concrete_log_grid():
    """Concrete log-uniform grids keep the single-FFT fast path."""
    assert _is_log_uniform(np.geomspace(4000.0, 9000.0, 2000)) is True


def test_is_log_uniform_still_false_for_concrete_linear_grid(linear_wave):
    assert _is_log_uniform(linear_wave) is False


def test_jit_with_traced_linear_grid_matches_eager(linear_wave, spectrum_in):
    """jit over a linear grid must equal the eager (general-path) result.

    On the unchanged code the traced grid took the constant-R FFT path, which is
    off by about 8e-3 (relative max) on this linear grid at this resolution.
    """
    resolution = 2000.0
    eager = apply_lsf(spectrum_in, jnp.asarray(linear_wave), resolution=resolution)

    def f(w):
        return apply_lsf(spectrum_in, w, resolution=resolution)

    jitted = jax.jit(f)(jnp.asarray(linear_wave))
    # The traced grid pads by n - 1 while the concrete one pads by the 5-sigma bound,
    # so the eager result drops up to ~6e-7 of the kernel tail (the documented
    # truncation). Measured max relative difference here is 7.5e-9; the wrong
    # constant-R path is off by 7.9e-3 on this grid. rtol 1e-6 sits between the two.
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), rtol=1e-6, atol=0.0)


def test_jit_traced_grid_uses_general_path_not_constant_r(linear_wave, spectrum_in, monkeypatch):
    """The traced call must reach the piecewise kernel, not the constant-R one."""
    calls = {"constant": 0, "variable": 0}
    orig_const = spectrum_module._apply_lsf_constant_r
    orig_var = spectrum_module._apply_lsf_variable_r

    def spy_const(*a, **k):
        calls["constant"] += 1
        return orig_const(*a, **k)

    def spy_var(*a, **k):
        calls["variable"] += 1
        return orig_var(*a, **k)

    monkeypatch.setattr(spectrum_module, "_apply_lsf_constant_r", spy_const)
    monkeypatch.setattr(spectrum_module, "_apply_lsf_variable_r", spy_var)

    jax.jit(lambda w: apply_lsf(spectrum_in, w, resolution=2000.0))(jnp.asarray(linear_wave))
    assert calls == {"constant": 0, "variable": 1}


def test_velocity_broaden_refuses_a_traced_grid(linear_wave):
    """velocity_broaden has no general path, so an unverifiable grid must raise.

    The log-uniformity check used to return silently on a tracer, leaving the
    single-FFT path to run on a grid it could not inspect.
    """
    from tengri.observation.spectrum import velocity_broaden

    flux = jnp.ones(linear_wave.size)
    with pytest.raises(ValueError, match="concrete"):
        jax.jit(lambda w: velocity_broaden(flux, w, 150.0))(jnp.asarray(linear_wave))
