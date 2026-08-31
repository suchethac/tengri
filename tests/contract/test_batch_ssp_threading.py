# SPDX-License-Identifier: BSD-3-Clause
"""The batch surfaces must thread the SSP grid too (#1793).

#1753 / #1787 gave the scalar JIT-safe surfaces a ``ssp_data=`` channel so a
caller who wraps them in their own ``jax.jit`` keeps the grid out of the
compiled program. The batch helpers were left behind: both are
``jax.vmap(model.predict_X)``, which closure-captures ``model``, so the grid
reached the trace as a constant with no way to pass it in.

Measured on the 3x25x1600 synthetic grid before the fix -- largest constant
baked into a user-jitted trace:

    model.predict_photometry_batch(batch)                     120000  (the grid)
    jax.vmap(lambda p: predict_photometry(p, ssp_data=g))(b)    1600

The second row is what a caller could already hand-roll *because* #1787 landed;
the helper should offer the same thing rather than making the hand-roll the only
threadable option. On a real grid this is the 58.87 MB -> 0.19 MB
persistent-cache difference #1787 measured, per batched program.

The grids ride ``in_axes=None`` -- one shared table, never a per-galaxy copy.
The shape assertions below would catch a regression to ``in_axes=0``, which
would still thread but would silently ask for N copies of the SSP grid.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform

from ._jaxpr_consts import baked_bytes

pytestmark = pytest.mark.contract

_N_BATCH = 3


def _baked(fn, *args) -> int:
    """Bytes of array constants frozen into ``fn``'s trace, at any depth."""
    return baked_bytes(jax.make_jaxpr(fn)(*args))


def _model(ssp, observation):
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(8, 12)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )


def _batch(model, key=0):
    p = model.spec.sample(jax.random.PRNGKey(key))
    return {k: jnp.tile(jnp.asarray(v)[None], (_N_BATCH,)) for k, v in p.items()}


def test_batch_photometry_is_not_baked_when_threaded(synthetic_ssp_wide, synthetic_tophat_obs):
    """The regression: threading the grid must keep it out of the batch trace."""
    model = _model(synthetic_ssp_wide, synthetic_tophat_obs)
    batch = _batch(model)
    ssp_bytes = int(np.asarray(synthetic_ssp_wide.ssp_flux).nbytes)

    def threaded(grid, params_batch):
        return model.predict_photometry_batch(params_batch, ssp_data=grid)

    baked = _baked(threaded, synthetic_ssp_wide, batch)
    assert baked < ssp_bytes, (
        f"the caller threaded the grid into predict_photometry_batch and {baked} bytes "
        f"of constants remain frozen into the trace, >= the grid's own {ssp_bytes}. "
        f"On a real grid that is 58.87 MB inlined per batched program (#1793)."
    )


def test_unthreaded_batch_still_bakes_the_grid(synthetic_ssp_wide, synthetic_tophat_obs):
    """Neuter for the test above, so it cannot pass by the grid being small."""
    model = _model(synthetic_ssp_wide, synthetic_tophat_obs)
    batch = _batch(model)
    ssp_bytes = int(np.asarray(synthetic_ssp_wide.ssp_flux).nbytes)

    baked = _baked(model.predict_photometry_batch, batch)
    assert baked >= ssp_bytes, (
        f"expected the un-threaded batch call to bake the grid ({ssp_bytes} bytes) so "
        f"the threaded test has something to prove; baked only {baked}"
    )


@pytest.mark.parametrize("threaded", [False, True], ids=["baked", "threaded"])
def test_threading_does_not_change_batch_photometry(
    synthetic_ssp_wide, synthetic_tophat_obs, threaded
):
    """Threading moves where the grid enters, never the answer or the shape."""
    model = _model(synthetic_ssp_wide, synthetic_tophat_obs)
    batch = _batch(model, key=1)

    kw = {"ssp_data": synthetic_ssp_wide} if threaded else {}
    out = np.asarray(model.predict_photometry_batch(batch, **kw), dtype=np.float64)
    reference = np.asarray(model.predict_photometry_batch(batch), dtype=np.float64)

    assert out.shape[0] == _N_BATCH, (
        f"batch axis lost: got shape {out.shape}, expected leading {_N_BATCH}"
    )
    assert np.all(np.isfinite(out)), "batched photometry is not finite"
    np.testing.assert_allclose(
        out,
        reference,
        rtol=1e-12,
        atol=0.0,
        err_msg="threading changed the physics, not just the calling convention",
    )


def test_grid_is_shared_across_the_batch_not_copied(synthetic_ssp_wide, synthetic_tophat_obs):
    """``in_axes=None`` on the grids: one table, not N copies.

    A regression to ``in_axes=0`` would still thread -- the const test above
    would pass -- while silently demanding a per-galaxy copy of the SSP grid.
    vmap rejects that outright here, because the grid has no batch axis to map,
    so calling with a mapped axis is the failure this pins against.
    """
    model = _model(synthetic_ssp_wide, synthetic_tophat_obs)
    batch = _batch(model, key=2)

    out = model.predict_photometry_batch(batch, ssp_data=synthetic_ssp_wide)
    assert out.shape[0] == _N_BATCH

    # The single-galaxy surface, threaded, must agree row-for-row with the batch.
    single = model.spec.sample(jax.random.PRNGKey(2))
    one = np.asarray(
        model.predict_photometry(single, ssp_data=synthetic_ssp_wide), dtype=np.float64
    )
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float64)[0],
        one,
        rtol=1e-12,
        atol=0.0,
        err_msg="batch row 0 disagrees with the threaded scalar surface on identical params",
    )
