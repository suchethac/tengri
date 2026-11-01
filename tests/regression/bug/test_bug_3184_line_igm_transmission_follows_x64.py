# SPDX-License-Identifier: BSD-3-Clause
"""#3184: _line_igm_transmission returns the trace's canonical dtype, not its input's.

The early return for a chain with no IGM component was ``jnp.ones_like(line_wave_rest)``, so a
float64 wavelength array materialized under x64 came back as float64 ones under
``jax.enable_x64(False)``, and the next ``jnp.log10`` failed with a float64/float32 ``lax.mul``
mismatch. The function now states its wavelength input with ``device_table`` (the #3086
contract), so both the early return and the quadrature return a dtype the trace can hold.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed, SEDModel, recipes

pytestmark = pytest.mark.regression_bug

_WAVES = np.array([1215.67, 1549.0, 6564.61])
_PARAMS = {"neb_eline_sigma_kms": 100.0}


def _build(ssp_data, igm_type, z=3.0):
    recipe = dict(recipes.star_forming_photometry(), redshift=Fixed(z))
    recipe["igm"] = {"type": igm_type}
    return SEDModel.build(ssp_data=ssp_data, observation=None, **recipe)


@pytest.mark.parametrize("igm_type", ["none", "inoue14"])
def test_x64_off_output_is_float32_and_log10_works(synthetic_ssp_wide, igm_type):
    model = _build(synthetic_ssp_wide, igm_type)
    with jax.enable_x64(True):
        wave64 = jnp.asarray(_WAVES)
        assert wave64.dtype == np.float64
        reference = np.asarray(model._line_igm_transmission(_PARAMS, wave64, 3.0), np.float64)
    with jax.enable_x64(False):
        eager = model._line_igm_transmission(_PARAMS, wave64, 3.0)
        assert eager.dtype == np.float32
        np.asarray(jnp.log10(eager))
        jitted = jax.jit(lambda z: model._line_igm_transmission(_PARAMS, wave64, z))(3.0)
        assert jitted.dtype == np.float32
        np.asarray(jnp.log10(jitted))
    for out in (eager, jitted):
        np.testing.assert_allclose(np.asarray(out, np.float64), reference, rtol=0, atol=1e-6)
    if igm_type == "none":
        np.testing.assert_array_equal(np.asarray(eager), np.ones(len(_WAVES), np.float32))


@pytest.mark.parametrize("igm_type", ["none", "inoue14"])
def test_float32_input_under_x64_on_is_a_dtype_x64_can_hold(synthetic_ssp_wide, igm_type):
    """x64 holds both float dtypes; the seam only rules out a dtype the trace cannot hold.

    Whether the quadrature promotes to float64 is ordinary JAX promotion, not part of the contract.
    """
    model = _build(synthetic_ssp_wide, igm_type)
    with jax.enable_x64(True):
        # The float64 reference uses the float32-rounded wavelengths, so the check measures the
        # dtype seam rather than the input rounding (1215.67 is ~6e-5 A off in float32).
        rounded64 = jnp.asarray(np.array(_WAVES, dtype=np.float32).astype(np.float64))
        reference = np.asarray(model._line_igm_transmission(_PARAMS, rounded64, 3.0), np.float64)
        wave32 = jnp.asarray(np.array(_WAVES, dtype=np.float32))
        assert wave32.dtype == np.float32
        out = model._line_igm_transmission(_PARAMS, wave32, 3.0)
        assert out.dtype in (np.float32, np.float64)
        np.asarray(jnp.log10(out))
        np.testing.assert_allclose(np.asarray(out, np.float64), reference, rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    ("x64", "wave_dtype"),
    [
        (True, np.float64),
        (False, np.float64),
        (False, np.float32),
        (True, np.float32),
    ],
)
def test_no_igm_early_return_never_leaves_a_non_canonical_dtype(
    synthetic_ssp_wide, x64, wave_dtype
):
    """The branch the float32 fast-nebular build hits: no IGM, ones in a dtype the trace holds.

    A float64 input under x64-off comes back float32; a float32 input keeps float32.
    """
    model = _build(synthetic_ssp_wide, "none")
    with jax.enable_x64(True):
        wave = jnp.asarray(np.array(_WAVES, dtype=wave_dtype))
    with jax.enable_x64(x64):
        expected = np.float32 if wave_dtype == np.float32 else (np.float64 if x64 else np.float32)
        out = model._line_igm_transmission(_PARAMS, wave, 3.0)
        assert out.dtype == expected
        np.testing.assert_array_equal(np.asarray(out), np.ones(len(_WAVES)))
