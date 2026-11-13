# SPDX-License-Identifier: BSD-3-Clause
"""#3404: profiled mass stays finite when the best-fit amplitude is non-positive.

A non-detection (negative or zero photometric fluxes) gives a best-fit
dimensionless amplitude ``a_star <= 0``. The profiled quadrature used
``log10(a_star)`` for its window, which is NaN there, and the NaN poisoned the
marginal likelihood, the reinserted mass draws and the MAP point. With a
non-positive amplitude the conditional posterior over ``ell`` piles against the
lower prior edge, so every output must stay finite and the draws must sit there.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import ForwardModel, Observation, Photometry, SEDModel, generate_mock, recipes
from tengri.inference.fitter import Fitter
from tengri.inference.mass_profile import (
    _QUAD_HALF_WIDTH_SIGMAS,
    _REINSERT_QUAD_NODES,
    _log_mass_integral,
    _quad_nodes,
    _sample_log_mass,
)
from tengri.parameters.priors import Uniform

pytestmark = pytest.mark.regression_bug

_FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "des_g", "des_r", "des_i"]
_MASS_NAME = "sfh_tsnorm_log_total_mass"
_ELL_LO = 7.0
_ELL_HI = 12.5
_ELL_REF = 10.0
#: Large coefficient: the amplitude posterior is narrow (sigma_a = 1/sqrt(A)),
#: so a non-detection piles the mass at the lower edge. At A = 1e6 the tail
#: beyond 0.5 dex still holds about 0.5% of the draws, so the edge check needs 1e8.
_A_REF = 1.0e8
_NON_POSITIVE_AMPLITUDES = [-3.0, -1e-3, 0.0, 1e-12]


def _prior():
    return Uniform(_ELL_LO, _ELL_HI)


def _original_quad_nodes(A_ref, a_star, ell_lo, ell_hi, ell_ref, quad_nodes):
    """The pre-#3404 grid, verbatim, for the bit-identity check."""
    u_lo = ell_lo - ell_ref
    u_hi = ell_hi - ell_ref
    center = jnp.clip(jnp.log10(a_star), u_lo, u_hi)
    sigma_u = 1.0 / (jnp.sqrt(A_ref) * a_star * jnp.log(10.0))
    lo = jnp.maximum(u_lo, center - _QUAD_HALF_WIDTH_SIGMAS * sigma_u)
    hi = jnp.minimum(u_hi, center + _QUAD_HALF_WIDTH_SIGMAS * sigma_u)
    u_nodes = jnp.linspace(lo, hi, quad_nodes)
    d_u = (hi - lo) / (quad_nodes - 1)
    weights = jnp.ones(quad_nodes).at[0].set(0.5).at[-1].set(0.5) * d_u
    return u_nodes, weights


@pytest.mark.parametrize("a_star", _NON_POSITIVE_AMPLITUDES)
def test_quad_nodes_finite_for_non_positive_amplitude(a_star):
    u_nodes, weights = _quad_nodes(_A_REF, a_star, _ELL_LO, _ELL_HI, _ELL_REF)
    assert bool(jnp.all(jnp.isfinite(u_nodes)))
    assert bool(jnp.all(jnp.isfinite(weights)))
    assert bool(jnp.all(weights >= 0.0))
    # The grid starts at the prior's lower edge, offset from the reference.
    assert float(u_nodes[0]) == pytest.approx(_ELL_LO - _ELL_REF)


@pytest.mark.parametrize("a_star", _NON_POSITIVE_AMPLITUDES)
def test_log_mass_integral_finite_for_non_positive_amplitude(a_star):
    value = _log_mass_integral(_A_REF, a_star, _ELL_LO, _ELL_HI, _prior(), _ELL_REF)
    assert bool(jnp.isfinite(value)), f"a_star={a_star}: log integral = {value}"


@pytest.mark.parametrize("a_star", _NON_POSITIVE_AMPLITUDES)
def test_sample_log_mass_draws_pile_at_lower_edge(a_star):
    keys = jax.random.split(jax.random.PRNGKey(0), 200)

    def draw(k):
        return _sample_log_mass(k, _A_REF, a_star, _ELL_LO, _ELL_HI, _prior(), _ELL_REF)

    draws = np.asarray(jax.vmap(draw)(keys))
    assert np.all(np.isfinite(draws))
    assert np.all((draws >= _ELL_LO) & (draws <= _ELL_HI))
    assert np.all(draws - _ELL_LO < 0.5), (
        f"max draw {draws.max():.3f} is more than 0.5 dex above the lower edge"
    )


def test_quad_nodes_bit_identical_inside_window():
    a_star = 1.7
    assert _ELL_LO <= _ELL_REF + np.log10(a_star) <= _ELL_HI
    for quad_nodes in (48, _REINSERT_QUAD_NODES):
        new_u, new_w = _quad_nodes(
            400.0, a_star, _ELL_LO, _ELL_HI, _ELL_REF, quad_nodes=quad_nodes
        )
        old_u, old_w = _original_quad_nodes(400.0, a_star, _ELL_LO, _ELL_HI, _ELL_REF, quad_nodes)
        np.testing.assert_array_equal(np.asarray(new_u), np.asarray(old_u))
        np.testing.assert_array_equal(np.asarray(new_w), np.asarray(old_w))


def test_log_mass_integral_gradient_finite_for_non_positive_amplitude():
    def f(a_star):
        return _log_mass_integral(_A_REF, a_star, _ELL_LO, _ELL_HI, _prior(), _ELL_REF)

    grad = jax.grad(f)(jnp.asarray(-0.5))
    assert bool(jnp.isfinite(grad)), f"d log integral / d a_star = {grad}"


def _non_detection_fitter(ssp_data):
    """Photometry-only fitter whose fluxes are all -|noise| (a non-detection)."""
    obs = Observation(photometry=Photometry.from_names(_FILTERS))
    model = SEDModel.build(ssp_data=ssp_data, observation=obs, **recipes.mock_recovery_minimal())
    forward = ForwardModel.build(sed=model)
    truth = model.spec.sample(jax.random.PRNGKey(5))
    mock = generate_mock(model, truth, key=jax.random.PRNGKey(6), snr=30.0)
    noise = jnp.asarray(mock["noise"])
    flux = -jnp.abs(noise)
    return Fitter(forward, data=flux, noise=noise, profile_mass=True)


def test_map_mass_is_finite_and_at_lower_bound_for_non_detection(ssp_data_fsps):
    fitter = _non_detection_fitter(ssp_data_fsps)
    lo, _ = fitter._profile_mass_bounds
    with pytest.warns(RuntimeWarning, match=_MASS_NAME):
        post = fitter.run("map", key=jax.random.PRNGKey(42))
    mass = float(post.params[_MASS_NAME])
    assert np.isfinite(mass)
    assert mass == pytest.approx(lo, abs=1e-9)
    assert post.diagnostics["profile_mass_at_bound"] is True


def test_laplace_mass_is_finite_and_near_lower_bound_for_non_detection(ssp_data_fsps):
    fitter = _non_detection_fitter(ssp_data_fsps)
    lo, _ = fitter._profile_mass_bounds
    post = fitter.run("laplace", key=jax.random.PRNGKey(42))
    draws = np.asarray(post.samples[_MASS_NAME])
    assert np.all(np.isfinite(draws))
    assert np.all(draws >= lo)
    assert float(np.mean(draws)) - lo < 0.5
