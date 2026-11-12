# SPDX-License-Identifier: BSD-3-Clause
"""``equivalent_width`` is finite and float64-accurate in float32 at physical L_nu scale.

``equivalent_width`` converts L_nu [erg/s/Hz] to F_lambda with ``C_AA / wavelength**2``
(about 3e14 at 100 Angstrom). A galaxy-scale L_nu of 1e33 therefore gives a product of
about 6e47, above the float32 maximum (3.4e38), so the float32 result is NaN. The equivalent
width is a ratio, so the overall flux scale must cancel exactly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.analysis.diagnostics.spectral import equivalent_width

pytestmark = pytest.mark.contract

LINE_AA = 6564.61
WINDOW_AA = 20.0
CONTINUUM_AA = 50.0


def _spectrum():
    wave = np.linspace(6000.0, 7200.0, 2400)
    continuum = 2.0e33 * (wave / LINE_AA) ** -1.5
    line = 0.05 * continuum * np.exp(-0.5 * ((wave - LINE_AA) / 4.0) ** 2)
    return wave, continuum + line


def test_equivalent_width_float32_is_finite_and_matches_float64():
    wave, l_nu = _spectrum()
    ref = float(
        equivalent_width(jnp.asarray(wave), jnp.asarray(l_nu), LINE_AA, WINDOW_AA, CONTINUUM_AA)
    )
    with jax.enable_x64(False):
        out = equivalent_width(
            jnp.asarray(wave, dtype=jnp.float32),
            jnp.asarray(l_nu, dtype=jnp.float32),
            LINE_AA,
            WINDOW_AA,
            CONTINUUM_AA,
        )
    out = float(out)
    assert np.isfinite(out), "float32 equivalent width is non-finite"
    assert out != 0.0
    # The width is i_line / f_cont - window_width, a difference of ~40.5 Angstrom and ~40.0.
    # float32 round-off on the ~40.5 ratio (about 2e-7 relative, measured abs 7.5e-6 Angstrom)
    # is the error floor, so the tolerance is absolute: 1e-4 Angstrom.
    np.testing.assert_allclose(out, ref, rtol=0.0, atol=1e-4)
