# SPDX-License-Identifier: BSD-3-Clause
"""DSPS kernels must not be handed mixed float32/float64 operands (#1448).

The SSP grids (``ssp_lgmet``, ``ssp_lg_age_gyr``, ``ssp_flux``) are cached host
arrays built once at load time, so they stay float64 even inside
``jax.enable_x64(False)``, while fitted parameters arrive as float32 tracers.
DSPS then sizes its internal buffers from the float64 grids and scatters a
float32-derived value into them::

    FutureWarning: scatter inputs have incompatible types: cannot safely cast
    value from dtype=float32 to dtype=float64 with
    jax_numpy_dtype_promotion=standard. In future JAX releases this will
    result in an error.

"In future JAX releases this will result in an error" is the point: today it
warns, tomorrow the float32 path stops working. Every DSPS call site now routes
its operands through ``canonical_dsps_kwargs`` first.

Both SFH paths are covered because they reach *different* DSPS entry points:

* parametric / tabulated -> tengri's own cloud-in-cell age weights plus
  ``calc_lgmet_weights_from_lognormal_mdf`` for the metallicity axis;
* field (stochastic GP) -> DSPS's ``calc_rest_sed_sfh_table_*`` kernels, which
  do the core SSP integration.

The field path was missed by the first fix precisely because the parametric
tests never reach those kernels -- measured six warnings there after the
lognormal-MDF site was already clean.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED

pytestmark = pytest.mark.regression_bug


def _build(ssp, sfh):
    from tengri import Fixed, SEDModel
    from tengri.observation import Observation, Photometry

    obs = Observation(photometry=Photometry.from_names(["sdss_r", "sdss_i", "wise_w1"]))
    return SEDModel.build(ssp_data=ssp, observation=obs, sfh=sfh, redshift=Fixed(0.5), approx=None)


_SFH_CASES = {
    # tengri CIC age weights + DSPS lognormal-MDF metallicity weights
    "parametric": {"type": "dpl", "*": FIXED},
    # DSPS calc_rest_sed_sfh_table_* — the core SSP integration
    "field": {"type": "dpl", "field": {"*": FIXED}, "*": FIXED},
}


@pytest.mark.parametrize("case", sorted(_SFH_CASES), ids=sorted(_SFH_CASES))
def test_no_mixed_dtype_scatter_in_pure_float32(ssp_bare, case):
    """A pure-float32 forward pass must emit no scatter FutureWarning."""
    model = _build(ssp_bare, _SFH_CASES[case])
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))

    with jax.enable_x64(False):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            phot = model.predict_photometry(params)
        scatter = [
            w
            for w in caught
            if issubclass(w.category, FutureWarning) and "scatter inputs" in str(w.message)
        ]

    assert jnp.all(jnp.isfinite(phot)), "float32 photometry must stay finite"
    assert not scatter, (
        f"{len(scatter)} mixed-dtype scatter FutureWarning(s) on the {case!r} SFH path — "
        "a DSPS call site is being handed a cached float64 grid alongside float32 "
        f"tracers. First: {str(scatter[0].message)[:160] if scatter else ''}"
    )


@pytest.mark.parametrize("case", sorted(_SFH_CASES), ids=sorted(_SFH_CASES))
def test_float64_photometry_is_unchanged(ssp_bare, case):
    """Canonicalizing must be a no-op under x64, where the canonical float is f64.

    That property is what makes the pattern safe to apply at every DSPS
    boundary, so it is pinned rather than assumed. Reference values were
    measured at full ``repr`` against the pre-change tree and are bit-identical.
    """
    model = _build(ssp_bare, _SFH_CASES[case])
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))

    with jax.enable_x64(True):
        phot = np.asarray(model.predict_photometry(params), dtype=np.float64)

    expected = {
        "parametric": [4.340050276662975e-30, 7.543893013950031e-30, 2.7959839525213e-29],
        "field": [4.08152632752286e-30, 7.322717885517473e-30, 2.8190906432470604e-29],
    }[case]

    np.testing.assert_array_equal(
        phot,
        np.array(expected, dtype=np.float64),
        err_msg=(
            f"float64 photometry moved on the {case!r} path; the dtype "
            "canonicalization must be a no-op under x64"
        ),
    )
