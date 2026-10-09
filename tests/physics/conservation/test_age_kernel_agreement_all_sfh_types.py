# SPDX-License-Identifier: BSD-3-Clause
"""Both SFH age kernels build, conserve mass and agree for every SFH type (#2683, #2684).

``age_kernel='cic'`` shares each mass parcel of the dense, edge-resolved integrand
between the two bracketing SSP ages (first order); ``'dsps'`` names the same
integration (#2683), because the histogram assignment it once meant mis-places any
structure narrower than the SSP node spacing and no table refinement removes it.
For every SFH type the registry offers, at z in {0, 2.5} and default parameters,
both kernels must build, form the declared mass, and agree in the four bands
(FUV, u, r, H) to round-off. (Earlier bounds of 0.5-26 % per family, which
documented the histogram kernel's placement error, are gone with it.)

The class at the end repeats the checks on a 4-D alpha-enhanced SSP library: the
alpha axis is collapsed before the age kernel runs, so the kernels see the same
cube as on a 3-D grid.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.components.stellar.sfh.registry import SFH_REGISTRY
from tengri.components.stellar.sps.dsps_wrapper import interpolate_alpha_only

pytestmark = pytest.mark.conservation

BANDS = ["galex_fuv", "sdss_u", "sdss_r", "2mass_h"]
COMMON_BOUND = 1e-9
ZS = (0.0, 2.5)

#: The registry's own gate: these types are refused at build, on both kernels,
#: with the stated message; ``burst`` and ``field`` are modifiers that need a
#: smooth base (covered by the composite and field tests), ``table`` needs a
#: runtime history.
REFUSALS = {
    "bursty_continuity": "not yet validated",
    "gaussian_burst": "not yet validated",
    "prospector_beta": "not yet validated",
    "top_hat": "not yet validated",
    "burst": "additive",
    "field": "additive",
    "table": "runtime arrays",
}


def _unique_families():
    seen = {}
    for key, spec in SFH_REGISTRY.items():
        seen.setdefault(id(spec), key)
    return sorted(seen.values())


FAMILIES = _unique_families()


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp()


@pytest.fixture(scope="module")
def obs():
    return tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))


def _build(ssp, obs, kernel, z, sfh, **kw):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        neb={"type": "none"},
        redshift=Fixed(z),
        sfh={**sfh, "all_params": Fixed(DEFAULT), "age_kernel": kernel},
        **kw,
    )


def _formed(model, params=None):
    return float(model.predict_state(params or {}).derived["log_mstar_formed"])


@pytest.mark.parametrize("z", ZS)
@pytest.mark.parametrize("family", FAMILIES)
def test_kernels_agree_for_every_sfh_type(ssp, obs, family, z):
    if family in REFUSALS:
        with pytest.raises((ValueError, TypeError), match=REFUSALS[family]):
            _build(ssp, obs, "cic", z, {"type": family}).predict_photometry({})
        return
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kernel in ("cic", "dsps"):
            m = _build(ssp, obs, kernel, z, {"type": family})
            out[kernel] = (_formed(m), np.asarray(m.predict_photometry({})))
    declared = out["cic"][0]
    assert abs(out["dsps"][0] - declared) <= 1e-8 * abs(declared), f"{family} z={z}: mass"
    err = np.abs(out["dsps"][1] / out["cic"][1] - 1.0)
    assert np.all(err <= COMMON_BOUND), (
        f"{family} z={z}: |dsps/cic-1| in {BANDS} = {np.round(err * 100, 6)} % exceeds "
        f"{COMMON_BOUND * 100:.0e} %"
    )


# --- 4-D alpha-enhanced library -------------------------------------------------

ALPHA_NODES = jnp.array([-0.2, 0.0, 0.2, 0.4])
SFH_4D = {
    "delayed_5gyr": {
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(10.0),
    },
    "continuity": {"type": "continuity"},
    "field": {"type": "delayed", "field": True},
}


@pytest.fixture(scope="module")
def ssp4(ssp):
    """The shipped 3-D library with an alpha axis: flux x (1 + 0.4 [alpha/Fe]).

    The [alpha/Fe] = 0 node equals the 3-D library exactly. (The two fixtures in
    tests/components/sps carry 10 and 8 age nodes: too coarse to compare kernels.)
    """
    flux4 = ssp.ssp_flux[:, None, :, :] * (1.0 + 0.4 * ALPHA_NODES)[None, :, None, None]
    return ssp._replace(ssp_flux=flux4, ssp_alpha_fe=ALPHA_NODES)


def _build4(ssp4, obs, kernel, z, sfh, alpha, **kw):
    return SEDModel.build(
        ssp_data=ssp4,
        observation=obs,
        neb={"type": "none"},
        redshift=Fixed(z),
        sfh={**sfh, "all_params": Fixed(DEFAULT), "age_kernel": kernel},
        met={"type": "delta", "alpha_fe": alpha, "all_params": Fixed(DEFAULT)},
        **kw,
    )


@pytest.mark.parametrize("alpha", [0.0, 0.3])
@pytest.mark.parametrize("z", ZS)
@pytest.mark.parametrize("name", sorted(SFH_4D))
def test_4d_grid_both_kernels(ssp4, obs, name, z, alpha):
    sfh = SFH_4D[name]
    out = {}
    for kernel in ("cic", "dsps"):
        m = _build4(ssp4, obs, kernel, z, sfh, Fixed(alpha))
        p = m.spec.sample(jax.random.PRNGKey(0)) if name == "field" else {}
        phot = np.asarray(m.predict_photometry(p))
        assert np.all(np.isfinite(phot)), f"{kernel}: non-finite photometry"
        out[kernel] = (_formed(m, p), phot)
    assert abs(out["dsps"][0] - out["cic"][0]) <= 1e-8 * abs(out["cic"][0])
    err = np.abs(out["dsps"][1] / out["cic"][1] - 1.0)
    assert np.all(err <= COMMON_BOUND), f"{name} z={z} alpha={alpha}: {np.round(err * 100, 6)} %"


@pytest.mark.parametrize("kernel", ["cic", "dsps"])
def test_4d_at_scaled_solar_equals_3d(ssp, ssp4, obs, kernel):
    sfh = SFH_4D["delayed_5gyr"]
    p4 = np.asarray(_build4(ssp4, obs, kernel, 0.0, sfh, Fixed(0.0)).predict_photometry({}))
    p3 = np.asarray(_build(ssp, obs, kernel, 0.0, sfh).predict_photometry({}))
    np.testing.assert_allclose(p4, p3, rtol=1e-10, atol=0)


@pytest.mark.parametrize("kernel", ["cic", "dsps"])
def test_4d_apply_matches_fast_path_on_the_collapsed_cube(ssp4, obs, kernel):
    """The fast path refuses a 4-D library; on the collapsed 3-D cube it matches apply.

    Compared as the stellar SED: the fast-path weights folded with the collapsed
    flux cube must be proportional to the apply SED at every wavelength.
    """
    sfh = SFH_4D["delayed_5gyr"]
    m = SEDModel.build(
        ssp_data=ssp4,
        observation=obs,
        neb={"type": "none"},
        redshift=Fixed(0.0),
        sfh={**sfh, "all_params": Fixed(DEFAULT), "age_kernel": kernel},
        met={"type": "delta", "alpha_fe": Fixed(0.3), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
    )
    sed_apply = np.asarray(m.predict_rest_sed({}).sed)
    stellar = next(c for c in m._build_component_chain() if type(c).__name__.startswith("Stellar"))
    params = {}
    for name in m.spec.all_params:
        value = getattr(m.spec.get_distribution(name), "default", None)
        if value is not None:
            params[name] = jnp.asarray(float(value))
    with pytest.raises(ValueError, match="alpha-Fe"):
        stellar.compute_joint_weights(params, ssp_data=ssp4)
    flat = ssp4._replace(
        ssp_flux=interpolate_alpha_only(ssp4.ssp_flux, ssp4.ssp_alpha_fe, jnp.asarray(0.3)),
        ssp_alpha_fe=None,
    )
    # The 3-D fast path folds [alpha/Fe] into the metallicity (effective Z); the 4-D
    # apply collapsed the cube instead, so hand it alpha = 0.
    params["met_alpha_fe"] = jnp.asarray(0.0)
    jw_fast, _, _ = stellar.compute_joint_weights(params, ssp_data=flat)
    folded = np.einsum("ma,maw->w", np.asarray(jw_fast), np.asarray(flat.ssp_flux))
    ok = folded > 0
    ratio = sed_apply[ok] / folded[ok]
    assert ratio.max() / ratio.min() - 1.0 <= 1e-8, (
        f"apply/fast-path SED ratio varies {ratio.max() / ratio.min() - 1:.2e}"
    )


def test_4d_alpha_gradient_is_kernel_independent(ssp4, obs):
    g = {}
    for kernel in ("cic", "dsps"):
        m = _build4(ssp4, obs, kernel, 0.0, SFH_4D["delayed_5gyr"], Uniform(-0.2, 0.4))

        def band(a, model=m):
            return model.predict_photometry({"met_alpha_fe": a})[2]

        g[kernel] = float(jax.grad(band)(0.1))
    assert np.isfinite(g["cic"]) and np.isfinite(g["dsps"])
    assert g["cic"] != 0.0 and g["dsps"] != 0.0, g
    assert abs(g["dsps"] - g["cic"]) <= 1e-9 * abs(g["cic"]), g


@pytest.mark.parametrize("kernel", ["cic", "dsps"])
def test_4d_evolving_metallicity_route(ssp4, obs, kernel):
    """Per-age metallicity on a 4-D library builds on both kernels and conserves mass."""
    m = SEDModel.build(
        ssp_data=ssp4,
        observation=obs,
        neb={"type": "none"},
        redshift=Fixed(0.0),
        sfh={**SFH_4D["delayed_5gyr"], "all_params": Fixed(DEFAULT), "age_kernel": kernel},
        met={"type": "ramp", "all_params": Fixed(DEFAULT)},
    )
    assert np.all(np.isfinite(np.asarray(m.predict_photometry({}))))
    assert abs(10.0 ** _formed(m) / 1e10 - 1.0) <= 1e-8
