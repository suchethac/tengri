# SPDX-License-Identifier: BSD-3-Clause
"""The dust energy-balance table works on a grid with an [alpha/Fe] axis (#2250).

The energy-balance LUT is linear in the SSP cube. On a 4D ``(n_met, n_alpha, n_age, n_wave)``
grid its ``B``/``G`` families carry the alpha axis, and the consumers collapse it at the
stellar component's bracket before the contraction. Contract:

a. WavePrecomp agrees with the exact path at several [alpha/Fe] values, with the same error
   the 3D alpha-free slab shows, for two-component and single-screen dust, fixed and free z;
b. at a node value of [alpha/Fe] the 4D WavePrecomp model equals the 3D model of that slab;
c. the gradient of WavePrecomp photometry with respect to a free ``met_alpha_fe`` is finite,
   nonzero and matches a central finite difference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, SEDModel, Uniform, WavePrecomp
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.parameters import Fixed
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.contract

ALPHA_NODES = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])
#: The 3D alpha-free slab's own LUT-vs-exact error is the yardstick; the 4D error must stay
#: within this factor of it (and below an absolute ceiling).
SLAB_FACTOR = 2.0
RTOL_CEILING = 1e-2
FLOOR = 1e-12

DUST = {
    "two_component": {"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT)},
    "single_component": {
        "type": "single_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
    },
}
EMISSION = {"type": "dale2014", "all_params": Fixed(DEFAULT)}


@pytest.fixture(scope="module")
def ssp4d():
    """Build 3 Z x 5 [alpha/Fe] x 20 ages x 300 wavelengths (as in test_alpha_fe_4d_luts)."""
    n_met, n_age, n_wave = 3, 20, 300
    wave = jnp.linspace(3000.0, 10000.0, n_wave)
    ages = jnp.linspace(-1.0, 1.14, n_age)
    key = jax.random.PRNGKey(7)
    base = jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-3
    feature = jnp.exp(-0.5 * ((wave - 5175.0) / 40.0) ** 2)
    cubes = [base * (1.0 - 0.25 * (1.0 + a) * feature)[None, None, :] for a in ALPHA_NODES]
    flux = jnp.stack(cubes, axis=1)  # (n_met, n_alpha, n_age, n_wave)
    lgmet = jnp.array([-1.5, -0.5, 0.0]) + LOG10_ZSUN
    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux,
        ssp_lg_age_gyr=ages,
        ssp_lgmet=lgmet,
        ssp_alpha_fe=jnp.asarray(ALPHA_NODES),
    )


@pytest.fixture(scope="module")
def observation():
    return tengri.Observation(
        photometry=tengri.Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]),
    )


def _model(ssp, observation, *, dust, alpha_spec, redshift, approx):
    met = {"logzsol": Fixed(-0.2), "all_params": Fixed(DEFAULT)}
    if alpha_spec is not None:
        met["alpha_fe"] = alpha_spec
    kwargs = dict(
        ssp_data=ssp,
        observation=observation,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
        met=met,
        dust_attenuation=dust,
        dust_emission=EMISSION,
        redshift=redshift,
        neb={"type": "none"},
    )
    if approx is not None:
        kwargs["approx"] = approx
    return SEDModel.build(**kwargs)


def _full_params(model, alpha=None, z=None):
    """A complete parameter set (free parameters drawn at a fixed key), then overridden."""
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    if alpha is not None:
        params["met_alpha_fe"] = jnp.asarray(alpha, dtype=jnp.float64)
    if z is not None:
        params["redshift"] = jnp.asarray(z, dtype=jnp.float64)
    return params


def _observables(model, alpha=None, z=None):
    """Photometry, absorbed and re-emitted luminosity of one parameter point.

    The energy-balance table reaches the properties only through the template data the JIT
    path threads, so it is passed explicitly: without it a plain ``predict_properties`` call
    runs the exact integral and would not test the table at all.
    """
    params = _full_params(model, alpha, z)
    phot = np.asarray(model.predict_photometry(params))
    props = model.predict_properties(
        params,
        names=("l_dust_absorbed", "l_tir"),
        template_data=model._template_data_for_jit(),
    )
    return phot, np.asarray(props["l_dust_absorbed"]), np.asarray(props["l_tir"])


def _max_rel(lut, exact):
    return float(np.max(np.abs(lut / exact - 1.0)))


def _error(lut_obs, exact_obs):
    """Largest relative LUT-vs-exact error over photometry, absorbed and re-emitted."""
    return max(
        _max_rel(np.atleast_1d(lut), np.atleast_1d(exact))
        for lut, exact in zip(lut_obs, exact_obs, strict=True)
    )


@pytest.mark.parametrize("redshift", [Fixed(0.1), Uniform(0.05, 0.5)], ids=["fixed_z", "free_z"])
@pytest.mark.parametrize("dust_name", ["two_component", "single_component"])
def test_wave_precomp_matches_exact_at_several_alpha(ssp4d, observation, dust_name, redshift):
    dust = DUST[dust_name]
    slab = ssp4d._replace(ssp_flux=ssp4d.ssp_flux[:, 1], ssp_alpha_fe=None)
    z = 0.2 if not isinstance(redshift, Fixed) else None

    slab_lut = _model(
        slab, observation, dust=dust, alpha_spec=None, redshift=redshift, approx=WavePrecomp()
    )
    slab_exact = _model(
        slab, observation, dust=dust, alpha_spec=None, redshift=redshift, approx=None
    )
    slab_err = _error(_observables(slab_lut, None, z), _observables(slab_exact, None, z))
    budget = max(SLAB_FACTOR * slab_err, FLOOR)

    lut_m = _model(
        ssp4d,
        observation,
        dust=dust,
        alpha_spec=Uniform(-0.2, 0.6),
        redshift=redshift,
        approx=WavePrecomp(),
    )
    exact_m = _model(
        ssp4d,
        observation,
        dust=dust,
        alpha_spec=Uniform(-0.2, 0.6),
        redshift=redshift,
        approx=None,
    )
    worst = max(
        _error(_observables(lut_m, a, z), _observables(exact_m, a, z)) for a in (0.0, 0.3, 0.6)
    )
    assert worst < RTOL_CEILING, f"4D LUT vs exact {worst:.3e} exceeds the ceiling"
    assert worst <= budget, f"4D LUT vs exact {worst:.3e} exceeds {budget:.3e} (2x 3D slab)"


@pytest.mark.parametrize("dust_name", ["two_component", "single_component"])
def test_wave_precomp_on_node_equals_the_node_slab(ssp4d, observation, dust_name):
    """At alpha = 0.2 (node 2) the 4D model's absorbed luminosity equals the 3D slab's."""
    dust = DUST[dust_name]
    m4 = _model(
        ssp4d,
        observation,
        dust=dust,
        alpha_spec=Uniform(-0.2, 0.6),
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )
    slab = ssp4d._replace(ssp_flux=ssp4d.ssp_flux[:, 2], ssp_alpha_fe=None)
    m3 = _model(
        slab, observation, dust=dust, alpha_spec=None, redshift=Fixed(0.1), approx=WavePrecomp()
    )
    for got, want in zip(_observables(m4, 0.2), _observables(m3, None), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=0)


@pytest.mark.parametrize("dust_name", ["two_component", "single_component"])
def test_gradient_in_alpha_through_the_energy_balance_table(ssp4d, observation, dust_name):
    m = _model(
        ssp4d,
        observation,
        dust=DUST[dust_name],
        alpha_spec=Uniform(-0.2, 0.6),
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )
    base = _full_params(m, 0.3, None)

    def total(a):
        params = dict(base, met_alpha_fe=a)
        return 1e12 * jnp.sum(m.predict_photometry(params))

    a0 = jnp.asarray(0.3)
    g = float(jax.grad(total)(a0))
    h = 1e-4
    fd = float((total(a0 + h) - total(a0 - h)) / (2 * h))
    assert np.isfinite(g), "non-finite d(LUT photometry)/d(met_alpha_fe)"
    assert g != 0.0, "identically zero d(LUT photometry)/d(met_alpha_fe)"
    np.testing.assert_allclose(g, fd, rtol=1e-4)
