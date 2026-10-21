# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSP templates are host arrays that must follow the trace's x64 state (#3086).

``load_grahsp_templates`` caches NumPy arrays (#1462). JAX 0.11 memoizes a numpy object's
device conversion by object identity with the dtype of its *first* conversion, so a bare
``jnp.asarray(template)`` or ``scalar * template`` made under ``jax.enable_x64(True)`` returned
a float64 buffer to a later ``jax.enable_x64(False)`` computation, which XLA rejected
(``Executable(jit_multiply) expected parameter 1 ... f32[36] but got ... f64[36]``). The failure
depended on which test ran first in a process, so these tests toggle x64 explicitly in one
process, in both orders, and are independent of shard order.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.grahsp import GRAHSPSEDComponent
from tengri.components.agn.grahsp.templates import load_grahsp_templates
from tengri.protocols.component import ForwardState

pytestmark = pytest.mark.regression_bug

_PARAMS = {
    "agn_grahsp_log_l5100": 44.0,
    "agn_grahsp_uvslope": 0.0,
    "agn_grahsp_plslope": -1.7,
    "agn_grahsp_plbendloc_nm": 100.0,
    "agn_grahsp_plbendwidth": 1.0,
    "agn_grahsp_cutoff_nm": 10000.0,
    "agn_grahsp_a_lines": 1.0,
    "agn_grahsp_a_feii": 5.0,
    "agn_grahsp_linewidth_kms": 5000.0,
    "agn_grahsp_fcov": 0.4,
    "agn_grahsp_si": 0.0,
    "agn_grahsp_cool_lam_um": 17.0,
    "agn_grahsp_cool_width": 0.45,
    "agn_grahsp_hot_lam_um": 2.0,
    "agn_grahsp_hot_width": 0.5,
    "agn_grahsp_hot_fcov": 1.0,
    "agn_grahsp_ebv": 0.05,
    "agn_grahsp_ebv_agn": 0.05,
}
_WAVE = np.geomspace(0.5, 1.0e4, 120)


def _apply(x64):
    dtype = jnp.float64 if x64 else jnp.float32
    with jax.enable_x64(x64):
        wave = jnp.asarray(_WAVE, dtype=dtype)
        params = {k: jnp.asarray(v, dtype=dtype) for k, v in _PARAMS.items()}
        out = GRAHSPSEDComponent().apply(ForwardState(wave=wave), params)
        return out.sed_intrinsic.dtype, np.asarray(out.sed_intrinsic, np.float64)


def test_float32_apply_after_the_model_path_converted_the_templates_in_float64(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """The failing sequence of #3086, in one process: model predictions, then float32 GRAHSP.

    ``predict_photometry`` / ``predict`` run the GRAHSP blocks under float64 and convert the
    cached host templates there; the float32 ``apply`` that follows must not receive those
    float64 buffers. The templates are read from the same cached bundle in both.

    JAX keeps the dtype of a numpy object's *first* device conversion, so this test is placed
    before the others in the file, which also convert the bundle's arrays; it clears the
    loader cache so that the model path makes the first conversion of fresh arrays.
    """
    import warnings

    from tengri import DEFAULT, Fixed, SEDModel

    load_grahsp_templates.cache_clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            neb={"type": "none"},
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw"},
                "blr": {"type": "grahsp"},
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(2.0),
        )
        params = dict(model.spec.sample(jax.random.PRNGKey(0)))
        np.asarray(model.predict_photometry(params))
        np.asarray(model.predict(params).photometry())

    dtype, sed = _apply(False)
    assert dtype == jnp.float32
    assert np.all(np.isfinite(sed))
    _, sed64 = _apply(True)
    peak = float(np.max(np.abs(sed64)))
    np.testing.assert_allclose(sed, sed64, rtol=1e-4, atol=1e-6 * peak)


@pytest.mark.parametrize("order", [(True, False), (False, True), (True, False, True)])
def test_apply_follows_the_x64_state_in_either_order(order):
    """Finite results and the right dtype after every x64 toggle, in one process."""
    results = {}
    for x64 in order:
        dtype, sed = _apply(x64)
        assert dtype == (jnp.float64 if x64 else jnp.float32)
        assert np.all(np.isfinite(sed))
        results[x64] = sed
    if len(results) == 2:
        peak = float(np.max(np.abs(results[True])))
        # float32 flushes the far Gaussian tails (< 1e-38 of the peak) to zero.
        np.testing.assert_allclose(results[False], results[True], rtol=1e-4, atol=1e-6 * peak)


def test_template_use_after_a_float64_conversion_of_the_same_array():
    """The loader's own arrays: convert under x64, then use under float32."""
    templates = load_grahsp_templates()
    from tengri.utils.host_array import device_table

    with jax.enable_x64(True):
        assert device_table(templates.line_broad).dtype == jnp.float64
        # Populate JAX's identity memo for this exact array object with a float64 copy.
        assert jnp.asarray(templates.line_broad).dtype == jnp.float64
    with jax.enable_x64(False):
        scale = jnp.asarray(2.0, dtype=jnp.float32)
        assert (scale * device_table(templates.line_broad)).dtype == jnp.float32
    with jax.enable_x64(True):
        assert device_table(templates.line_broad).dtype == jnp.float64
