# SPDX-License-Identifier: BSD-3-Clause
r"""The ADAF spectral shape must not depend on the SED dtype (#2783).

At the clipped-accretion-rate corner of the ADAF model (``agn_log_lbol_shape``
high, so ``mdot`` is clipped to ``mdot_crit``) the electron-temperature and
``x_M`` solves used to run in float32 and moved ``T_e`` by enough to change the
spectral *shape* by up to ~1 relative in the Wien tail, while the total power
still agreed. The shape ``L_nu / int L_nu dnu`` is therefore compared between
float32 and float64 over the issue's points and every corner of the declared
ADAF prior box, with the tolerance the disc-shape guards use.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_WAVE = np.geomspace(1.0e2, 1.0e11, 4000)  # Angstrom; covers the ADAF support
_C_ANG_HZ = 2.99792458e18  # c in Angstrom Hz

# Declared prior box (components/agn/_params.py): the four ADAF corners and the
# shape luminosity at both ends of agn_log_lbol's Uniform(8, 14).
_MBH_BOX = (6.0, 10.0)
_ALPHA_BOX = (0.05, 0.5)
_BETA_BOX = (0.1, 0.9)
_DELTA_BOX = (0.001, 0.5)
_LBOL_SHAPE_BOX = (8.0, 14.0)

# The issue's two points, at the clipped-mdot corner (log_lbol 2, shape 14).
_ISSUE_POINTS = (
    {"agn_log_mbh": 6.0},
    {"agn_log_mbh": 10.0, "agn_adaf_alpha": 0.5},
)


def _corner_points():
    points = [
        {
            "agn_log_mbh": mbh,
            "agn_adaf_alpha": alpha,
            "agn_adaf_beta": beta,
            "agn_adaf_delta": delta,
            "agn_log_lbol_shape": shape,
        }
        for mbh, alpha, beta, delta, shape in itertools.product(
            _MBH_BOX, _ALPHA_BOX, _BETA_BOX, _DELTA_BOX, _LBOL_SHAPE_BOX
        )
    ]
    for extra in _ISSUE_POINTS:
        points.append({"agn_log_lbol_shape": 14.0, **extra})
    return points


def _adaf_shape(point, dtype):
    """Return ``L_nu / int L_nu dnu`` on ``_WAVE`` for ``point`` in ``dtype``."""
    from tengri.components.agn.adaf import adaf_spectrum

    w = jnp.asarray(_WAVE, dtype=dtype)
    spec = np.asarray(adaf_spectrum(w, agn_log_lbol=2.0, **point), dtype=np.float64)
    nu = _C_ANG_HZ / _WAVE
    power = abs(np.trapezoid(spec * nu, x=np.log(_WAVE)))
    return spec / power


def _shape_pair(point):
    with jax.enable_x64(True):
        ref = _adaf_shape(point, jnp.float64)
    with jax.enable_x64(False):
        f32 = _adaf_shape(point, jnp.float32)
    return ref, f32


@pytest.mark.parametrize("point", _corner_points(), ids=lambda p: str(sorted(p.items())))
def test_adaf_shape_float32_matches_float64(point):
    """The float32 ADAF shape matches float64 to the disc-shape tolerance."""
    ref, f32 = _shape_pair(point)
    assert np.all(np.isfinite(f32)), f"non-finite float32 ADAF shape at {point}"
    peak = np.abs(ref).max()
    live = np.abs(ref) > 1e-6 * peak
    rel = np.abs(f32[live] - ref[live]) / np.abs(ref[live])
    assert rel.max() < 1e-3, (
        f"ADAF shape float32 vs float64 max rel = {rel.max():.3e} at {point} "
        "(#2783: the T_e / x_M solves must not depend on the SED dtype)"
    )
