# SPDX-License-Identifier: BSD-3-Clause
"""SpectrumPrecomp works on a grid with an [alpha/Fe] axis (#2250).

Before this contract, ``approx=SpectrumPrecomp()`` on a 4D ``(n_met, n_alpha, n_age, n_wave)``
grid failed in two places: the fixed-redshift lookup table assumed a 3D cube (an IndexError at
build), and the per-draw route used by a free redshift refused a 4D cube outright. Contract:

a. for each redshift and dispersion configuration, the spectrum from the lookup table agrees
   with the exact path at several [alpha/Fe] values, within twice the error the same
   configuration shows on the alpha = 0 slab in the same test (and below 1e-2);
b. at a node value of [alpha/Fe] the 4D spectrum equals the 3D model built on that node's slab
   (bit-for-bit up to float reassociation);
c. ``d(spectrum)/d(met_alpha_fe)`` is finite, nonzero and matches a central finite difference.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.forward.sed_model import SpectrumPrecomp
from tengri.parameters import Fixed
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.contract

ALPHA_NODES = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])
ALPHA_NODE_INDEX = {float(a): k for k, a in enumerate(ALPHA_NODES)}
# Between-node values exercise the bracket interpolation; 0.2 is a node.
ALPHA_PROBES = (0.0, 0.3, 0.6)
TOL_CEILING = 1e-2

# (redshift prior, sigma_v prior or None, params that fix the redshift and dispersion)
CONFIGS = {
    "fixed_z_sigma_fixed": (Fixed(0.05), Fixed(150.0), {}),
    "fixed_z_sigma_free": (Fixed(0.05), Uniform(0.0, 2000.0), {"sigma_v_kms": 150.0}),
    "free_z_sigma_undeclared": (Uniform(0.01, 0.5, "redshift"), None, {"redshift": 0.1}),
    "free_z_sigma_fixed": (Uniform(0.01, 0.5, "redshift"), Fixed(150.0), {"redshift": 0.1}),
}


@functools.cache
def _ssp4d():
    """Build 3 Z x 5 [alpha/Fe] x 20 ages x 300 wavelengths (as in test_alpha_fe_4d_luts)."""
    n_met, n_age, n_wave = 3, 20, 300
    wave = jnp.linspace(3000.0, 10000.0, n_wave)
    ages = jnp.linspace(-1.0, 1.14, n_age)
    key = jax.random.PRNGKey(7)
    base = jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-3
    feature = jnp.exp(-0.5 * ((wave - 5175.0) / 40.0) ** 2)  # a Mg b-like absorption trough
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


def _slab(ssp4d, k):
    return SSPData(
        ssp_wave=ssp4d.ssp_wave,
        ssp_flux=ssp4d.ssp_flux[:, k],
        ssp_lg_age_gyr=ssp4d.ssp_lg_age_gyr,
        ssp_lgmet=ssp4d.ssp_lgmet,
    )


@functools.cache
def _observation():
    wave_obs = jnp.linspace(4000.0, 8000.0, 400)
    return tengri.Observation(
        photometry=tengri.Photometry.from_names(["sdss_g", "sdss_r"]),
        spectroscopy=tengri.Spectroscopy(wave_obs=wave_obs, resolution=jnp.full(400, 2000.0)),
    )


def _model(ssp, config, *, alpha, approx):
    """Model for one configuration: 4D when ``alpha`` is True, else a 3D slab."""
    z, sigma, _ = CONFIGS[config]
    met = {"logzsol": Fixed(-0.3)}
    if alpha:
        met["alpha_fe"] = Uniform(-0.2, 0.6)
    kwargs = dict(
        ssp_data=ssp,
        observation=_observation(),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": Fixed(2.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
        },
        met=met,
        redshift=z,
        neb={"type": "ssp"},
    )
    if sigma is None:
        return SEDModel.build(**kwargs, approx=approx)
    # The dispersion is merged into the spec before the lookup table is built.
    base = SEDModel.build(**kwargs, approx=None)
    spec = base.spec.merge_observation_params(sigma_v_kms=sigma)
    return SEDModel(spec, ssp, observation=_observation(), approx=approx)


def _spectrum(model, config, alpha=None):
    params = dict(CONFIGS[config][2])
    if alpha is not None:
        params["met_alpha_fe"] = alpha
    return np.asarray(model.predict_observables(params).spec_fnu)


def _rel_err(a, b):
    """Largest pixel error relative to the largest exact pixel."""
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("config", list(CONFIGS))
def test_spectrum_precomp_matches_exact_within_control(config):
    ssp = _ssp4d()
    control_exact = _spectrum(_model(_slab(ssp, 1), config, alpha=False, approx=None), config)
    control_lut = _spectrum(
        _model(_slab(ssp, 1), config, alpha=False, approx=SpectrumPrecomp()), config
    )
    control = _rel_err(control_lut, control_exact)
    # The floor keeps the bound meaningful when the control is at round-off.
    tol = min(max(2.0 * control, 1e-12), TOL_CEILING)
    assert control > 0.0, "the 3D control shows no LUT error; the tolerance would be vacuous"

    exact4 = _model(ssp, config, alpha=True, approx=None)
    lut4 = _model(ssp, config, alpha=True, approx=SpectrumPrecomp())
    worst = 0.0
    for a in ALPHA_PROBES:
        err = _rel_err(_spectrum(lut4, config, a), _spectrum(exact4, config, a))
        worst = max(worst, err)
        assert err <= tol, (
            f"{config} alpha={a}: LUT vs exact {err:.3e} exceeds {tol:.3e} "
            f"(twice the alpha = 0 control {control:.3e})"
        )
    assert worst < TOL_CEILING


@pytest.mark.parametrize("config", ["fixed_z_sigma_fixed", "free_z_sigma_fixed"])
@pytest.mark.parametrize("alpha", [-0.2, 0.2, 0.6])
def test_node_spectrum_equals_the_node_slab_model(config, alpha):
    ssp = _ssp4d()
    k = ALPHA_NODE_INDEX[alpha]
    lut4 = _model(ssp, config, alpha=True, approx=SpectrumPrecomp())
    lut3 = _model(_slab(ssp, k), config, alpha=False, approx=SpectrumPrecomp())
    np.testing.assert_allclose(
        _spectrum(lut4, config, alpha), _spectrum(lut3, config), rtol=1e-10, atol=0
    )


@pytest.mark.parametrize("config", ["fixed_z_sigma_fixed", "free_z_sigma_undeclared"])
def test_alpha_gradient_on_the_lut_matches_finite_difference(config):
    lut4 = _model(_ssp4d(), config, alpha=True, approx=SpectrumPrecomp())
    scale = float(np.max(np.abs(_spectrum(lut4, config, 0.25))))

    def loss(a):
        params = dict(CONFIGS[config][2])
        params["met_alpha_fe"] = a
        return jnp.sum(lut4.predict_observables(params).spec_fnu) / scale

    g = float(jax.grad(loss)(0.25))
    h = 1e-3
    fd = float((loss(0.25 + h) - loss(0.25 - h)) / (2 * h))
    assert np.isfinite(g), "a non-finite d(spectrum)/d(met_alpha_fe) through the LUT"
    assert g != 0.0, "an identically zero d(spectrum)/d(met_alpha_fe) through the LUT"
    np.testing.assert_allclose(g, fd, rtol=1e-4)


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.6])
def test_pre_broadened_cube_route_equals_the_per_draw_route(alpha):
    """With a free redshift and a fixed dispersion the cube is broadened once at build.

    On a grid with an [alpha/Fe] axis that cube carries the axis, and the route that
    uses it must give the spectrum of the per-draw route, which broadens the summed
    spectrum instead. The per-draw result comes from the same model with the built
    cube removed from the stellar component's state.
    """
    import dataclasses

    from tengri.inference._model_cache import _default_owner

    config = "free_z_sigma_fixed"
    model = _model(_ssp4d(), config, alpha=True, approx=SpectrumPrecomp())
    stellar = model._cached_component_chain[0]
    cube = stellar._state.spec_ssp_broadened
    assert cube is not None and np.ndim(cube) == 4, "the 4D pre-broadened cube was not built"
    pre_broadened = _spectrum(model, config, alpha)

    object.__setattr__(
        stellar, "_state", dataclasses.replace(stellar._state, spec_ssp_broadened=None)
    )
    _default_owner.clear()
    jax.clear_caches()
    per_draw = _spectrum(model, config, alpha)
    _default_owner.clear()

    assert _rel_err(pre_broadened, per_draw) < 1e-10
