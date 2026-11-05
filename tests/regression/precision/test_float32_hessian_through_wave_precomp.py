# SPDX-License-Identifier: BSD-3-Clause
r"""The float32 Hessian of a photometric objective through the WavePrecomp LUT is finite.

Every entry of that Hessian was NaN. The cause was the sub-band node wavelength,
:math:`\lambda_k = \sum(w\lambda\phi)/\sum(w\phi)`, whose denominator is a per-Msun
band luminosity of 3e-21 to 1e-17. Its custom rule keeps the first derivative in
range, but the second derivative of a ratio carries :math:`1/\mathrm{den}^2`, above
float32's maximum (3.4e38) for :math:`\mathrm{den} < 10^{-19}`; the zero cotangent
of an unconsumed node times that ``inf`` is NaN.

The objective is the residual / sigma form of a real likelihood. A bare
``sum(photometry)`` has a float32 second derivative of ~1e-60 (underflow), which
reads as an all-zero Hessian whatever the code does, so it cannot tell a repair
from the defect.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.components.stellar.component import _flux_weighted_node

pytestmark = pytest.mark.regression_bug

_BANDS = ["des_g", "des_r", "des_z", "wise_w1"]
_SSP = "data/fsps_prsc_miles_chabrier.h5"


def _build():
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return SEDModel.build(
        ssp_data=load_ssp_data(_SSP),
        observation=Observation(photometry=Photometry.from_names(_BANDS)),
        sfh={"type": "continuity", "log_total_mass": Fixed(10.0), "all_params": FREE},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": Uniform(0.0, 2.0),
        },
        neb={"type": "none"},
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )


def _chi2_hessian(model):
    """Hessian of ``sum(((model - data) / sigma)**2)`` at one draw, data at another."""
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    other = dict(model.spec.sample(jax.random.PRNGKey(1)))
    free = {k: v for k, v in params.items() if k in set(model.spec.free_params)}
    data = model.predict_photometry({**params, **other})
    sigma = 0.1 * jnp.abs(data)

    def chi2(q):
        return jnp.sum(((model.predict_photometry({**params, **q}) - data) / sigma) ** 2)

    return jax.hessian(chi2)(free)


def test_float32_hessian_is_finite_and_nonzero():
    with jax.enable_x64(False):
        hess = _chi2_hessian(_build())
        leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(hess)]
    assert leaves and leaves[0].dtype == np.float32, "precondition: genuinely float32"
    n_bad = sum(int(np.sum(~np.isfinite(x))) for x in leaves)
    assert n_bad == 0, f"{n_bad} non-finite Hessian entries through the WavePrecomp LUT"
    assert any(np.any(x != 0.0) for x in leaves), "Hessian is identically zero"


def test_flux_weighted_node_second_derivative_in_float32():
    """The node's own second derivative at a per-Msun-sized denominator."""

    def f(num, den):
        return jnp.sum(_flux_weighted_node(num, den) * den)

    with jax.enable_x64(False):
        den = jnp.asarray([3.0e-21, 1.0e-19, 1.4e-17], dtype=jnp.float32)
        num = 5000.0 * den
        # d2/d(den)2 of q*den with q = num/den is d2(num)/d(den)2 = 0; the cross
        # term d2/d(num)d(den) is 1/den-sized, 3e20, which float32 holds.
        hess = jax.hessian(lambda d: f(5000.0 * d, d))(den)
        cross = jax.jacfwd(jax.grad(f, argnums=0), argnums=1)(num, den)
    assert np.all(np.isfinite(np.asarray(hess))), hess
    assert np.all(np.isfinite(np.asarray(cross))), cross


def test_flux_weighted_node_float64_is_bit_identical_to_the_plain_ratio():
    with jax.enable_x64(True):
        num = jnp.asarray([5000.0e-20, 5.0, 5000.0, 4321.0e-17])
        den = jnp.asarray([1.0e-20, 1.0e-3, 1.0, 1.0e-17])
        assert jnp.array_equal(_flux_weighted_node(num, den), num / den)
        g = jax.grad(lambda n, d: jnp.sum(_flux_weighted_node(n, d)), argnums=(0, 1))(num, den)
        p = jax.grad(lambda n, d: jnp.sum(n / d), argnums=(0, 1))(num, den)
        np.testing.assert_allclose(g[0], p[0], rtol=1e-15)
        np.testing.assert_allclose(g[1], p[1], rtol=1e-15)
