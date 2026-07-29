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


def test_canonicalization_is_a_noop_under_x64():
    """Under x64 the canonical float already IS float64, so every cast is a no-op.

    That property is the whole safety argument for applying this at every DSPS
    boundary, so it is pinned directly on the helper.

    An earlier version of this test pinned specific float64 photometry values
    instead, measured at full ``repr`` against a pre-change worktree. That was
    the wrong assertion: it pins a *consequence* of the invariant rather than
    the invariant, so it imports the numerics of the entire SED pipeline — and
    the host's BLAS and XLA codegen — into the comparison. It held on the
    machine it was recorded on and failed in CI, which is the expected
    behavior of a cross-machine bit-exact assertion on a full forward pass,
    not evidence about the canonicalization.

    Same-machine end-to-end float64 invariance was measured separately (both
    SFH paths, byte-for-byte at full ``repr``, against a detached worktree at
    the pre-change commit) and is recorded in the commit message; the rest of
    the precision suite would redden if float64 genuinely moved.
    """
    from tengri.components.stellar.sps.dsps_wrapper import canonical_dsps_kwargs

    with jax.enable_x64(True):
        raw = {
            "grid": jnp.linspace(-4.0, -1.0, 8),
            "scalar": jnp.asarray(0.5),
            "python_float": 0.25,
            "already_f32": jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        }
        out = canonical_dsps_kwargs(**raw)

    assert set(out) == set(raw), "canonicalization must not add or drop keys"
    for key, value in raw.items():
        assert out[key].dtype == jnp.float64, (
            f"{key!r} came back as {out[key].dtype}, not float64, under x64"
        )
        # Widening f32 -> f64 is exact, so every operand must compare equal to
        # its input; nothing may be rounded on the way through.
        np.testing.assert_array_equal(
            np.asarray(out[key]),
            np.asarray(jnp.asarray(value, dtype=jnp.float64)),
            err_msg=f"{key!r} was altered by a cast that must be a no-op under x64",
        )


def test_canonicalization_unifies_a_mixed_dtype_call_in_float32():
    """The actual job: one dtype out, even when a cached f64 grid meets f32."""
    from tengri.components.stellar.sps.dsps_wrapper import canonical_dsps_kwargs

    with jax.enable_x64(False):
        # A cached host grid stays float64 even here — that is the whole problem.
        cached_grid = jnp.asarray(np.linspace(-4.0, -1.0, 8), dtype=jnp.float64)
        out = canonical_dsps_kwargs(grid=cached_grid, param=jnp.asarray(0.5))

    dtypes = {key: value.dtype for key, value in out.items()}
    assert len(set(dtypes.values())) == 1, f"operands left on mixed dtypes: {dtypes}"
