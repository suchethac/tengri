# SPDX-License-Identifier: BSD-3-Clause
"""``predict_sfh`` is jit- and vmap-safe (#715).

``SEDModel.predict_sfh`` resamples the star-formation history onto a uniform
lookback grid through :func:`tengri.utils.grid.interpolate_to_linear_time`.
That function called ``float()`` on an element of the log-age grid. Under
``jax.jit`` a constant array's elements are staged tracers, so
``jax.jit(jax.vmap(model.predict_sfh))`` raised ``ConcretizationTypeError``
on every model, and the three SFH analysis figures could only loop over draws
eagerly.

The contract tested here:

* ``jit(vmap(predict_sfh))`` traces and runs for a parametric, a continuity and
  an age-anchored family, on the linear and the native grid;
* its per-draw result equals the eager per-draw loop to ``rtol=1e-10`` (float64);
* a float32 run is finite.

The reference is the eager loop over the same model, so the comparison checks
the batched path against the unbatched one; it does not depend on any frozen
number.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FREE, Fixed, SEDModel

pytestmark = pytest.mark.contract

N_DRAWS = 5
FAMILIES = ("dpl", "continuity", "psb_flex")


def _build(ssp, sfh_type):
    return SEDModel.build(
        ssp_data=ssp,
        redshift=Fixed(0.5),
        neb={"type": "none"},
        sfh={"type": sfh_type, "all_params": FREE},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
    )


def _batch(model, n=N_DRAWS):
    """Distinct draws of every free SFH parameter, shape ``(n,)`` each."""
    names = list(model.spec.free_params)
    offsets = jnp.linspace(-0.2, 0.2, n)
    return {k: 0.5 + offsets for k in names}


def _draw(batch, i):
    return {k: v[i] for k, v in batch.items()}


@pytest.mark.parametrize("sfh_type", FAMILIES)
@pytest.mark.parametrize("grid", ["linear", "native"])
def test_jit_vmap_predict_sfh_matches_eager_per_draw(synthetic_ssp_wide, sfh_type, grid):
    """Batched and eager per-draw SFH agree to rtol 1e-10 in float64."""
    model = _build(synthetic_ssp_wide, sfh_type)
    batch = _batch(model)

    def fn(p):
        out = model.predict_sfh(p, grid=grid)
        return out["t_gyr"], out["sfr_mean"], out["sfr_full"]

    t_b, mean_b, full_b = jax.jit(jax.vmap(fn))(batch)
    eager = [fn(_draw(batch, i)) for i in range(N_DRAWS)]
    t_e = np.stack([np.asarray(e[0]) for e in eager])
    mean_e = np.stack([np.asarray(e[1]) for e in eager])
    full_e = np.stack([np.asarray(e[2]) for e in eager])

    np.testing.assert_allclose(np.asarray(t_b), t_e, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(np.asarray(mean_b), mean_e, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(np.asarray(full_b), full_e, rtol=1e-10, atol=0.0)
    assert np.all(np.isfinite(mean_e))


def test_jit_vmap_predict_sfh_float32_is_finite(synthetic_ssp_wide_physical):
    """The jit/vmap path also runs in pure float32 and returns finite curves."""
    with jax.enable_x64(False):
        model = _build(synthetic_ssp_wide_physical, "dpl")
        batch = {k: v.astype(jnp.float32) for k, v in _batch(model).items()}
        out = jax.jit(jax.vmap(lambda p: model.predict_sfh(p)["sfr_mean"]))(batch)
        assert out.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(out)))
