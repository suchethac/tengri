# SPDX-License-Identifier: BSD-3-Clause
"""WavePrecomp photometry works on a grid with an [alpha/Fe] axis (#2250).

WavePrecomp (the default of every photometric fit) builds photometry lookup tables from the
SSP grid. On a 4D ``(n_met, n_alpha, n_age, n_wave)`` grid those tables carry the alpha axis,
and the photometry contractions assumed a 3D layout. Contract:

a. fixed redshift: the LUT agrees with the exact path at several [alpha/Fe] values;
b. free redshift: the same, through the z-table;
c. the gradient of the LUT photometry with respect to ``met_alpha_fe`` is finite, nonzero and
   matches a central finite difference, at fixed and at free redshift;
d. at a node value of [alpha/Fe] the 4D LUT equals the LUT of the 3D model built on that slab,
   which pins the collapse to the right plane;
e. the 3D path is unchanged: the LUT agrees with the exact path on an alpha-free slab;
f. the free-redshift z-table round-trips through the on-disk precomp cache.
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
# Relative LUT-vs-exact tolerance per band. Measured maxima are recorded in the report.
LUT_RTOL = 5e-3


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


def _model(ssp, observation, *, alpha_spec, redshift, approx):
    met = {"logzsol": Fixed(-0.2), "all_params": Fixed(DEFAULT)}
    if alpha_spec is not None:
        met["alpha_fe"] = alpha_spec
    kwargs = dict(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
        },
        met=met,
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


def _phot(model, alpha, z=None):
    return np.asarray(model.predict_photometry(_full_params(model, alpha, z)))


def _max_rel(lut, exact):
    return float(np.max(np.abs(lut / exact - 1.0)))


def test_fixed_z_lut_matches_exact_at_several_alpha(ssp4d, observation):
    lut_m = _model(
        ssp4d,
        observation,
        alpha_spec=Uniform(-0.2, 0.6),
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )
    exact_m = _model(
        ssp4d, observation, alpha_spec=Uniform(-0.2, 0.6), redshift=Fixed(0.1), approx=None
    )
    worst = max(_max_rel(_phot(lut_m, a), _phot(exact_m, a)) for a in (0.0, 0.3, 0.6))
    assert worst < LUT_RTOL, f"fixed-z LUT vs exact max relative difference {worst:.3e}"


def test_fixed_alpha_builds_and_matches_exact(ssp4d, observation):
    lut_m = _model(
        ssp4d, observation, alpha_spec=Fixed(0.3), redshift=Fixed(0.1), approx=WavePrecomp()
    )
    exact_m = _model(ssp4d, observation, alpha_spec=Fixed(0.3), redshift=Fixed(0.1), approx=None)
    worst = _max_rel(_phot(lut_m, None), _phot(exact_m, None))
    assert worst < LUT_RTOL, f"Fixed(0.3) LUT vs exact max relative difference {worst:.3e}"


def test_free_z_lut_matches_exact_at_several_alpha_and_z(ssp4d, observation):
    z_spec = Uniform(0.05, 0.5)
    lut_m = _model(
        ssp4d, observation, alpha_spec=Uniform(-0.2, 0.6), redshift=z_spec, approx=WavePrecomp()
    )
    exact_m = _model(
        ssp4d, observation, alpha_spec=Uniform(-0.2, 0.6), redshift=z_spec, approx=None
    )
    worst = max(
        _max_rel(_phot(lut_m, a, z), _phot(exact_m, a, z))
        for z in (0.1, 0.3)
        for a in (0.0, 0.3, 0.6)
    )
    assert worst < LUT_RTOL, f"free-z LUT vs exact max relative difference {worst:.3e}"


@pytest.mark.parametrize("z", [0.1, None], ids=["fixed_z", "free_z"])
def test_gradient_in_alpha_is_finite_and_matches_fd(ssp4d, observation, z):
    redshift = Fixed(0.1) if z is not None else Uniform(0.05, 0.5)
    m = _model(
        ssp4d, observation, alpha_spec=Uniform(-0.2, 0.6), redshift=redshift, approx=WavePrecomp()
    )

    base = _full_params(m, 0.3, None if z is not None else 0.2)

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


def test_lut_on_node_equals_lut_of_the_node_slab(ssp4d, observation):
    """At alpha = 0.2 (node 2) the 4D LUT equals the 3D LUT of that slab, to rounding."""
    m4 = _model(
        ssp4d,
        observation,
        alpha_spec=Uniform(-0.2, 0.6),
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )
    slab = ssp4d._replace(ssp_flux=ssp4d.ssp_flux[:, 2], ssp_alpha_fe=None)
    m3 = _model(slab, observation, alpha_spec=None, redshift=Fixed(0.1), approx=WavePrecomp())
    np.testing.assert_allclose(_phot(m4, 0.2), _phot(m3, None), rtol=1e-12, atol=0)


@pytest.mark.parametrize("redshift", [Fixed(0.1), Uniform(0.05, 0.5)], ids=["fixed_z", "free_z"])
def test_dusty_lut_between_nodes_equals_the_lut_of_the_interpolated_cube(
    ssp4d, observation, redshift
):
    """Under dust the tables' sub-band wavelengths matter; they must follow the spectrum.

    At [alpha/Fe] = 0.3, between two nodes, the 4D model's table photometry must equal the
    table photometry of a 3D model built on the cube interpolated at 0.3. Fluxes are
    linear in the cube. A sub-band wavelength is a flux-weighted mean, so it agrees only
    if ``wavelength * flux`` is interpolated, not the wavelength itself.
    """
    from tengri.components.stellar.sps.dsps_wrapper import interpolate_alpha_only

    dust = {"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT)}

    def build(ssp, alpha_spec):
        met = {"logzsol": Fixed(-0.2), "all_params": Fixed(DEFAULT)}
        if alpha_spec is not None:
            met["alpha_fe"] = alpha_spec
        return SEDModel.build(
            ssp_data=ssp,
            observation=observation,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
            met=met,
            dust_attenuation=dust,
            redshift=redshift,
            neb={"type": "none"},
            approx=WavePrecomp(),
        )

    flat = ssp4d._replace(
        ssp_flux=interpolate_alpha_only(ssp4d.ssp_flux, ssp4d.ssp_alpha_fe, jnp.asarray(0.3)),
        ssp_alpha_fe=None,
    )
    params = {} if isinstance(redshift, Fixed) else {"redshift": jnp.asarray(0.3)}
    on_4d = np.asarray(build(ssp4d, Fixed(0.3)).predict_photometry(params))
    on_flat = np.asarray(build(flat, None).predict_photometry(params))
    np.testing.assert_allclose(on_4d, on_flat, rtol=1e-10, atol=0.0)


def test_3d_slab_lut_matches_exact(ssp4d, observation):
    """The alpha-free path is unchanged in kind: the LUT agrees with the exact path on a slab."""
    slab = ssp4d._replace(ssp_flux=ssp4d.ssp_flux[:, 1], ssp_alpha_fe=None)
    lut_m = _model(slab, observation, alpha_spec=None, redshift=Fixed(0.1), approx=WavePrecomp())
    exact_m = _model(slab, observation, alpha_spec=None, redshift=Fixed(0.1), approx=None)
    worst = _max_rel(_phot(lut_m, None), _phot(exact_m, None))
    assert worst < LUT_RTOL, f"3D slab LUT vs exact max relative difference {worst:.3e}"


def test_free_z_table_round_trips_through_disk_cache(ssp4d, observation, tmp_path, monkeypatch):
    monkeypatch.delenv("TENGRI_DISABLE_PRECOMP_CACHE", raising=False)
    monkeypatch.setenv("TENGRI_PRECOMP_CACHE_DIR", str(tmp_path))
    z_spec = Uniform(0.05, 0.5)

    def build():
        return _model(
            ssp4d,
            observation,
            alpha_spec=Uniform(-0.2, 0.6),
            redshift=z_spec,
            approx=WavePrecomp(),
        )

    first = _phot(build(), 0.3, 0.2)
    assert list(tmp_path.glob("ztable_*.npz")), "the 4D z-table was not written to the cache"
    second = _phot(build(), 0.3, 0.2)
    assert np.array_equal(first, second)


# --- The exact IGM fold on a 4D grid (#2250) -------------------------------------------
# The exact sub-band fold (WavePrecomp(igm_fold="exact")) multiplies per-node tables that
# carry the alpha axis. These tests put the Lyman break inside sdss_u at z = 3.5 and
# require the 4D fold to be as accurate as the 3D fold on the same alpha = 0 slab.

IGM_BANDS = ["sdss_u", "sdss_g", "sdss_r"]
IGM_ALPHAS = (0.0, 0.3, 0.6)
IGM_RTOL = 1e-2
N_Z_EXACT = 16


@pytest.fixture(scope="module")
def ssp4d_uv():
    """3 Z x 5 [alpha/Fe] x 20 ages x 300 wavelengths reaching 800 A (Lyman limit in grid)."""
    n_met, n_age, n_wave = 3, 20, 300
    wave = jnp.linspace(800.0, 10000.0, n_wave)
    ages = jnp.linspace(-1.0, 1.14, n_age)
    key = jax.random.PRNGKey(11)
    base = jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-3
    feature = jnp.exp(-0.5 * ((wave - 1500.0) / 300.0) ** 2)
    cubes = [base * (1.0 + 0.3 * a * feature)[None, None, :] for a in ALPHA_NODES]
    flux = jnp.stack(cubes, axis=1)
    lgmet = jnp.array([-1.5, -0.5, 0.0]) + LOG10_ZSUN
    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux,
        ssp_lg_age_gyr=ages,
        ssp_lgmet=lgmet,
        ssp_alpha_fe=jnp.asarray(ALPHA_NODES),
    )


@pytest.fixture(scope="module")
def uv_observation():
    return tengri.Observation(photometry=tengri.Photometry.from_names(IGM_BANDS))


def _igm_model(ssp, observation, *, alpha_spec, redshift, approx):
    met = {"logzsol": Fixed(-0.2), "all_params": Fixed(DEFAULT)}
    if alpha_spec is not None:
        met["alpha_fe"] = alpha_spec
    kwargs = dict(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
        },
        met=met,
        redshift=redshift,
        neb={"type": "none"},
        igm={"type": "inoue"},
    )
    if approx is not None:
        kwargs["approx"] = approx
    return SEDModel.build(**kwargs)


def _exact_lut(n_z=N_Z_EXACT):
    return WavePrecomp(igm_fold="exact", n_z=n_z)


def _worst_rel(lut_m, exact_m, alphas, zs):
    return max(_max_rel(_phot(lut_m, a, z), _phot(exact_m, a, z)) for z in zs for a in alphas)


def _slab3d(ssp, index):
    return ssp._replace(ssp_flux=ssp.ssp_flux[:, index], ssp_alpha_fe=None)


def test_exact_igm_fixed_z_4d_as_accurate_as_3d(ssp4d_uv, uv_observation):
    alpha_spec = Uniform(-0.2, 0.6)
    lut4 = _igm_model(
        ssp4d_uv, uv_observation, alpha_spec=alpha_spec, redshift=Fixed(3.5), approx=_exact_lut()
    )
    exact4 = _igm_model(
        ssp4d_uv, uv_observation, alpha_spec=alpha_spec, redshift=Fixed(3.5), approx=None
    )
    err4 = _worst_rel(lut4, exact4, IGM_ALPHAS, (None,))
    slab = _slab3d(ssp4d_uv, 1)
    lut3 = _igm_model(
        slab, uv_observation, alpha_spec=None, redshift=Fixed(3.5), approx=_exact_lut()
    )
    exact3 = _igm_model(slab, uv_observation, alpha_spec=None, redshift=Fixed(3.5), approx=None)
    err3 = _worst_rel(lut3, exact3, (None,), (None,))
    assert err4 < IGM_RTOL, f"4D exact-IGM fixed-z error {err4:.3e} >= {IGM_RTOL}"
    assert err4 <= 2.0 * err3, f"4D error {err4:.3e} exceeds twice the 3D error {err3:.3e}"


def test_exact_igm_free_z_4d_as_accurate_as_3d(ssp4d_uv, uv_observation):
    z_spec = Uniform(2.5, 4.0)
    zs = (3.0, 3.5)
    lut4 = _igm_model(
        ssp4d_uv,
        uv_observation,
        alpha_spec=Uniform(-0.2, 0.6),
        redshift=z_spec,
        approx=_exact_lut(),
    )
    exact4 = _igm_model(
        ssp4d_uv, uv_observation, alpha_spec=Uniform(-0.2, 0.6), redshift=z_spec, approx=None
    )
    err4 = _worst_rel(lut4, exact4, IGM_ALPHAS, zs)
    slab = _slab3d(ssp4d_uv, 1)
    lut3 = _igm_model(slab, uv_observation, alpha_spec=None, redshift=z_spec, approx=_exact_lut())
    exact3 = _igm_model(slab, uv_observation, alpha_spec=None, redshift=z_spec, approx=None)
    err3 = _worst_rel(lut3, exact3, (None,), zs)
    assert err4 < IGM_RTOL, f"4D exact-IGM free-z error {err4:.3e} >= {IGM_RTOL}"
    assert err4 <= 2.0 * err3, f"4D free-z error {err4:.3e} exceeds twice the 3D error {err3:.3e}"


@pytest.mark.parametrize("redshift", [Fixed(3.5), Uniform(2.5, 4.0)], ids=["fixed_z", "free_z"])
def test_exact_igm_on_node_equals_the_node_slab(ssp4d_uv, uv_observation, redshift):
    z = 3.5
    m4 = _igm_model(
        ssp4d_uv,
        uv_observation,
        alpha_spec=Uniform(-0.2, 0.6),
        redshift=redshift,
        approx=_exact_lut(),
    )
    m3 = _igm_model(
        _slab3d(ssp4d_uv, 2),
        uv_observation,
        alpha_spec=None,
        redshift=redshift,
        approx=_exact_lut(),
    )
    fixed = isinstance(redshift, Fixed)
    z_arg = None if fixed else z
    np.testing.assert_allclose(_phot(m4, 0.2, z_arg), _phot(m3, None, z_arg), rtol=1e-10, atol=0)


def test_exact_igm_disk_cache_second_build_loads_identical(
    ssp4d_uv, uv_observation, tmp_path, monkeypatch
):
    from tengri.components.igm import _subband_cache

    monkeypatch.delenv("TENGRI_DISABLE_PRECOMP_CACHE", raising=False)
    monkeypatch.setenv("TENGRI_PRECOMP_CACHE_DIR", str(tmp_path))
    z_spec = Uniform(2.5, 4.0)
    alpha_spec = Uniform(-0.2, 0.6)

    def build(redshift):
        return _igm_model(
            ssp4d_uv,
            uv_observation,
            alpha_spec=alpha_spec,
            redshift=redshift,
            approx=_exact_lut(),
        )

    for redshift, z in ((z_spec, 3.5), (Fixed(3.5), None)):
        _subband_cache.clear_memo()
        first = _phot(build(redshift), 0.3, z)
        assert list(tmp_path.glob(f"{_subband_cache.EXACT_FOLD_PREFIX}_*.npz")), (
            "the exact-fold table was not written to the cache"
        )
        _subband_cache.clear_memo()
        second = _phot(build(redshift), 0.3, z)
        assert np.array_equal(first, second)
