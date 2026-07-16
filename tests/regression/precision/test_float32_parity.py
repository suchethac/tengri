import numpy as np
import pytest
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from .conftest import Z_MASS_GRID, build_model, forward_outputs
from tengri.observation.redshift_kernel import shift_to_obs_frame

pytestmark = pytest.mark.regression_bug


@pytest.mark.parametrize("z,log10_mass", Z_MASS_GRID)
def test_f64_reference_is_finite(ssp_bare, z, log10_mass):
    """Baseline: the current f64 path produces finite outputs on the grid."""
    model = build_model(ssp_bare, "float64")
    out = forward_outputs(model, z, log10_mass)
    for k, v in out.items():
        assert np.all(np.isfinite(v)), f"{k} non-finite at z={z}, logM={log10_mass}"


def _flux_case(dtype):
    wave_rest = jnp.linspace(1e2, 1e5, 2000, dtype=dtype)
    L_nu = jnp.asarray(2.4e34, dtype=dtype) * jnp.exp(
        -((wave_rest - 5000.0) ** 2) / (2 * 800.0 ** 2))
    wave_obs = jnp.linspace(1e2, 6e5, 3000, dtype=dtype)
    return wave_rest, L_nu, wave_obs


@pytest.mark.parametrize("z", [0.01, 0.5, 1.0, 6.0])
def test_flux_seam_f64_exact_and_f32_finite(z):
    wr, L, wo = _flux_case(jnp.float64)
    ref = np.asarray(shift_to_obs_frame(wr, L, wo, jnp.float64(z)))
    assert np.all(np.isfinite(ref)) and ref.max() > 0
    with jax.enable_x64(False):  # JAX ≥0.9 context manager (was jax.experimental.enable_x64)
        wr32, L32, wo32 = _flux_case(jnp.float32)
        got = np.asarray(shift_to_obs_frame(wr32, L32, wo32, jnp.float32(z)))
    assert np.all(np.isfinite(got)), f"f32 flux seam non-finite at z={z}"
    m = ref > ref.max() * 1e-6
    assert_allclose(got[m], ref[m], rtol=2e-3)
