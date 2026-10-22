# SPDX-License-Identifier: BSD-3-Clause
"""The template tori publish an energy ledger that closes on the accretion power (#2851).

Owner rulings (2026-10-13) for every AGN component, applied to the torus-only template
libraries (CAT3D-Wind, SKIRTOR-AGNfitter, Silva+04 as standalone components; Fritz+06,
the AGNfitter-rX CLUMPY reductions and ``SKIRTOR_mean_1p`` through their SED kernels):

* C1: ``log_L_agn_torus`` is angle-integrated, the sphere mean of the power the torus
  emits along every sightline; ``log_L_agn_torus_los`` is the sightline value.
* C2: ``log_L_agn_torus`` plus ``log_L_agn_disc_unemitted`` (the ``(1 - f) L_bol`` disc share
  a torus-only component does not emit) equals ``10**agn_log_lbol L_sun``.
* C3: ``log_L_agn_torus_los`` is the frequency integral of the emitted torus SED.
* C4: the keys are finite log10 powers in pure float32 and agree with float64.

The sphere mean is checked against a dense trapezoid in ``mu = cos i`` of the emitted powers,
not the component's Gauss-Legendre quadrature; the spectrum integral is a trapezoid in
``ln nu`` on a wide dense grid, not the native power-law integral the component uses.
"""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri._data_setup import find_data
from tengri.utils.physics_constants import L_SUN
from tests.physics._agn_publication import ln_nu_integral
from tests.physics._agn_published import (
    box_points,
    build_component,
    declared_box,
    predict,
    relative_error,
)

pytestmark = pytest.mark.contract

#: Wide dense grid [Angstrom] spanning every torus library's native support (10 A to 3.6e8 A).
_WIDE = np.geomspace(5.0, 1.0e9, 24000)
#: Dense ``mu`` grid of the reference sphere mean.
_MU = np.linspace(0.0, 1.0, 401)
_LEDGER = ("log_L_agn_torus", "log_L_agn_torus_los", "log_L_agn_disc_unemitted")

_STANDALONE = ("cat3d", "skirtor_agnfitter", "silva04")


def _lbol_erg(log_lbol):
    return 10.0 ** float(log_lbol) * L_SUN


def _linear(published, key):
    """``10**published[key]`` [erg/s]; an absent piece (``-inf``) is 0."""
    return 10.0 ** float(published[key])


# ── standalone components (normalized per sightline: provenance not established) ────


@pytest.fixture(scope="module", params=_STANDALONE)
def standalone(request):
    comp = build_component(request.param)
    return request.param, comp, declared_box(comp)


_JITTED: dict = {}


def _published64(comp, params, wave):
    """``comp.predict`` in float64 under one ``jit`` per (component, grid)."""
    key = (id(comp), wave.size)
    if key not in _JITTED:
        w = jnp.asarray(wave)
        _JITTED[key] = jax.jit(lambda p: comp.predict(p, jnp.zeros_like(w), w))
    p = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in params.items()}
    sed, pub = _JITTED[key](p)
    return np.asarray(sed), {k: float(v) for k, v in pub.items()}


def test_standalone_tori_declare_the_ledger(standalone):
    """Every torus-only component declares the three ledger keys and their Lsun siblings."""
    name, comp, _ = standalone
    declared = {k.name: k.units for k in comp.outputs()}
    for key in _LEDGER:
        assert declared[key] == "dex", (name, key)
        assert declared[f"{key[len('log_') :]}_lsun"] == "Lsun", (name, key)


def test_standalone_ledger_closes_on_the_accretion_power(standalone):
    """C2: torus + unemitted disc = 10**log_lbol L_sun, at the box center and every edge."""
    name, comp, box = standalone
    for params in box_points(box, n_random=3):
        _, pub = _published64(comp, params, np.geomspace(100.0, 1e7, 50))
        total = _linear(pub, "log_L_agn_torus") + _linear(pub, "log_L_agn_disc_unemitted")
        assert total == pytest.approx(_lbol_erg(params["log_lbol"]), rel=1e-12), (name, params)


def test_standalone_sightline_key_is_the_spectrum_integral(standalone):
    """C3, and C1 for a per-sightline library: ``_los`` integrates the SED; it equals the mean."""
    name, comp, box = standalone
    for params in box_points(box, n_random=3):
        sed, pub = _published64(comp, params, _WIDE)
        spectrum = ln_nu_integral(sed, _WIDE)
        if float(params["torus_frac"]) == 0.0:
            assert spectrum == 0.0 and pub["log_L_agn_torus_los"] == -np.inf
            continue
        assert _linear(pub, "log_L_agn_torus_los") == pytest.approx(spectrum, rel=1e-6), name
        assert pub["log_L_agn_torus_los"] == pytest.approx(pub["log_L_agn_torus"], abs=1e-9)


def test_standalone_ledger_is_float32_safe(standalone):
    """C4: every ledger key and its Lsun sibling is finite in float32 and agrees with float64."""
    name, comp, box = standalone
    mid = {k: 0.5 * (lo + hi) for k, (lo, hi) in box.items()}
    pub32, _ = predict(comp, mid, float32=True)
    pub64, _ = predict(comp, mid, float32=False)
    for key in _LEDGER:
        assert np.isfinite(pub32[key]), (name, key, pub32[key])
        assert relative_error(key, pub32[key], pub64[key]) < 1e-4, (name, key)
        lsun = f"{key[len('log_') :]}_lsun"
        assert np.isfinite(pub32[lsun]) and pub32[lsun] > 0.0, (name, lsun)


# ── restored libraries: the SED kernels the composable torus blocks call ─────────────


def _nenkova(module, extra):
    def sed(grid, wave, mu, **kw):
        return getattr(module, f"{module.__name__.rsplit('.', 1)[1]}_sed_from_grid")(
            grid, wave, agn_cos_inc=mu, **extra, **kw
        )

    def coords(mu):
        return (mu, *extra.values())

    return sed, coords


def _kernel_case(name):
    """``(dtype -> anisotropy, sed(grid, wave, mu, **kw), coords(mu), inclination, grid)``."""
    if name == "fritz":
        from tengri.components.agn import fritz as mod

        if find_data("fritz2006_torus_grid.h5") is None:
            pytest.skip("template data unavailable (fritz2006_torus_grid.h5)")
        grid = mod.load_fritz_default_grid()
        shape = {
            "agn_fritz_r_ratio": 60.0,
            "agn_fritz_tau": 1.5,
            "agn_fritz_beta": -0.5,
            "agn_fritz_gamma": 4.0,
            "agn_fritz_oa": 40.0,
        }

        def sed(g, wave, mu, **kw):
            return mod.fritz_sed_from_grid(
                g, wave, agn_fritz_psy=mod.fritz_psy_from_cos_inc(mu), **shape, **kw
            )

        def coords(mu):
            return (*shape.values(), mod.fritz_psy_from_cos_inc(mu))

        return (
            lambda dtype: mod.fritz_anisotropy(grid),
            sed,
            coords,
            mod.TORUS_INCLINATION,
            grid,
        )
    mod = importlib.import_module(f"tengri.components.agn.{name}")
    files = {
        "nenkova_agnfitter": "nenkova_agnfitter_torus_grid.h5",
        "nenkova_agnfitter_2p": "nenkova_agnfitter_2p_torus_grid.h5",
        "nenkova_agnfitter_3p": "nenkova_agnfitter_3p_torus_grid.h5",
        "skirtor_agnfitter_1p": "skirtor_mean1p_torus_grid.h5",
    }
    found = find_data(files[name])
    if found is None:
        pytest.skip(f"template data unavailable ({files[name]})")
    grid = getattr(mod, f"load_{name}_grid")(str(found))
    if name == "skirtor_agnfitter_1p":

        def sed(g, wave, mu, **kw):
            return mod.skirtor_agnfitter_1p_sed_from_grid(g, wave, agn_cos_inc=mu, **kw)

        from tengri.components.agn._skirtor_agnfitter_sightline import (
            skirtor_incl_from_cos_inc,
        )

        def coords(mu):
            # The kernel's own key, i = arccos(mu), as for Fritz's psi above (#3085).
            return (skirtor_incl_from_cos_inc(mu),)

        return _aniso(grid, mod), sed, coords, mod.TORUS_INCLINATION, grid
    extra = {
        "nenkova_agnfitter": {},
        "nenkova_agnfitter_2p": {"agn_oa_nenkova": 37.5},
        "nenkova_agnfitter_3p": {"agn_oa_nenkova": 37.5, "agn_tv_nenkova": 50.0},
    }[name]
    sed, coords = _nenkova(mod, extra)
    return _aniso(grid, mod), sed, coords, mod.TORUS_INCLINATION, grid


_RESTORED = (
    "fritz",
    "nenkova_agnfitter",
    "nenkova_agnfitter_2p",
    "nenkova_agnfitter_3p",
    "skirtor_agnfitter_1p",
)
_KW = {"agn_log_lbol": 11.0, "agn_torus_frac": 0.4}


@pytest.fixture(scope="module", params=_RESTORED)
def restored(request):
    return request.param, *_kernel_case(request.param)


def _aniso(grid, mod):
    """``dtype -> AnisotropyGrid`` of a restored AGNfitter-rX kernel's library."""
    from tengri.components.agn._torus_inclination import library_anisotropy

    return lambda dtype: library_anisotropy(grid, mod.TORUS_INCLINATION, dtype)


def _ledger(case, mu, dtype=jnp.float64):
    """``torus_ledger_from_grid`` at ``mu`` (jitted in ``mu``), as floats."""
    from tengri.components.agn._torus_inclination import torus_ledger_from_grid

    _, make_aniso, _, coords, inc, _ = case

    @jax.jit
    def _at(m):
        return torus_ledger_from_grid(make_aniso(dtype), tuple(coords(m)), inc, **_KW)

    pub = _at(jnp.asarray(mu, dtype=dtype))
    return {k: float(v) for k, v in pub.items()}


@pytest.mark.parametrize("mu", [0.0, 0.2, 0.5, 0.866, 1.0])
def test_restored_ledger_closes_and_integrates_the_spectrum(restored, mu):
    """C2 and C3 along five sightlines from edge-on to face-on."""
    name, _, sed_fn, _, _, grid = restored
    pub = _ledger(restored, mu)
    total = _linear(pub, "log_L_agn_torus") + _linear(pub, "log_L_agn_disc_unemitted")
    assert total == pytest.approx(_lbol_erg(_KW["agn_log_lbol"]), rel=1e-12), name
    sed = np.asarray(jax.jit(lambda m: sed_fn(grid, jnp.asarray(_WIDE), m, **_KW))(mu))
    spectrum = ln_nu_integral(sed, _WIDE)
    assert _linear(pub, "log_L_agn_torus_los") == pytest.approx(spectrum, rel=1e-6), (name, mu)


def test_restored_main_key_is_the_sphere_mean_of_the_emitted_power(restored):
    """C1: ``log_L_agn_torus`` is the mean over ``mu`` of the emitted torus power (1e-5).

    The reference is a 401-point trapezoid in ``mu`` of the spectrum integrals; the sightlines
    differ (the libraries keep their anisotropy), so the mean is not any one of them.
    """
    name, _, sed_fn, _, _, grid = restored
    fn = jax.jit(lambda m: sed_fn(grid, jnp.asarray(_WIDE), m, **_KW))
    powers = np.array([ln_nu_integral(np.asarray(fn(m)), _WIDE) for m in _MU])
    mean = float(np.trapezoid(powers, _MU))
    pub = _ledger(restored, 0.5)
    assert _linear(pub, "log_L_agn_torus") == pytest.approx(mean, rel=1e-5), name
    assert powers.max() / powers.min() > 1.1, f"{name}: no anisotropy left"


def test_restored_ledger_is_float32_safe(restored):
    """C4: the restored ledger is finite in pure float32 and agrees with float64."""
    name = restored[0]
    with jax.enable_x64(False):
        pub32 = _ledger(restored, 0.4, dtype=jnp.float32)
    pub64 = _ledger(restored, 0.4)
    for key, value in pub64.items():
        assert np.isfinite(pub32[key]), (name, key)
        assert relative_error(key, pub32[key], value) < 1e-4, (name, key)


#: Public SED entry points of the restored libraries, as the composable torus blocks call
#: them: (module, function, kwargs placing the sightline at ``mu = cos i``).
_PUBLIC_SEDS = {
    "fritz": ("fritz", "fritz_sed", lambda m: {"agn_fritz_psy": float(np.degrees(np.arcsin(m)))}),
    "nenkova_agnfitter": (
        "nenkova_agnfitter",
        "nenkova_agnfitter_sed",
        lambda m: {"agn_cos_inc": m},
    ),
    "nenkova_agnfitter_2p": (
        "nenkova_agnfitter_2p",
        "nenkova_agnfitter_2p_sed",
        lambda m: {"agn_cos_inc": m},
    ),
    "nenkova_agnfitter_3p": (
        "nenkova_agnfitter_3p",
        "nenkova_agnfitter_3p_sed",
        lambda m: {"agn_cos_inc": m},
    ),
    "skirtor_agnfitter_1p": (
        "skirtor_agnfitter_1p",
        "skirtor_agnfitter_1p_sed",
        lambda m: {"agn_cos_inc": float(m)},
    ),
}


@pytest.mark.parametrize("name", sorted(_PUBLIC_SEDS))
def test_restored_public_sed_is_angle_normalized_and_anisotropic(name):
    """C1 through the public SED: the mean over ``mu`` is ``f L_bol``, face-on is brighter.

    Uses only the libraries' public ``*_sed`` entry points (no ledger helper), so the
    assertion is about the physics: a library normalized per sightline carries ``f L_bol``
    along every sightline and has a face-on/edge-on power ratio of exactly 1.
    """
    module, fn_name, sightline = _PUBLIC_SEDS[name]
    fn = getattr(importlib.import_module(f"tengri.components.agn.{module}"), fn_name)
    wave = jnp.asarray(_WIDE)
    mus = np.linspace(0.0, 1.0, 101)
    try:
        powers = np.array(
            [ln_nu_integral(np.asarray(fn(wave, **_KW, **sightline(m))), _WIDE) for m in mus]
        )
    except (FileNotFoundError, OSError) as err:
        pytest.skip(f"{name}: template data unavailable ({err})")
    target = 10.0 ** _KW["agn_log_lbol"] * L_SUN * _KW["agn_torus_frac"]
    assert float(np.trapezoid(powers, mus)) == pytest.approx(target, rel=1e-4), name
    assert powers[-1] / powers[0] > 1.1, f"{name}: face-on/edge-on {powers[-1] / powers[0]:.3f}"
