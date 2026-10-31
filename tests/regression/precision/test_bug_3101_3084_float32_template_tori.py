# SPDX-License-Identifier: BSD-3-Clause
r"""Every template torus is finite and float64-accurate in pure float32 (#3101, #3084).

One class of bug, three symptoms, each an un-normalized scale carried linearly through
float32:

* #3084: the kernels formed the torus power ``10**agn_log_lbol * L_sun * f`` as a linear
  erg/s value, ``inf`` in float32 beyond ``agn_log_lbol`` of about 4.9, so a bare kernel
  call returned ``inf`` across the declared prior.
* #3101: the anisotropy table of the Nenkova and SKIRTOR ``mean_1p`` libraries was formed
  after casting record powers of ~1e-55 to float32, which flushes them to zero: 0/0, NaN at
  every luminosity, in the kernels and in the precompute lookups.
* The same libraries store their records at ~1e-68 to 1e-71. A host table is rescaled
  before the cast, but a table threaded through ``jit`` as an argument (the composable
  runner's path) was cast first and the torus came back as exactly 0, silently.

Each case is measured in float32 against float64 *inside the test*
(``jax.enable_x64(False)``), with every grid built fresh in its own mode, over
``agn_log_lbol`` in {6, 8, 10, 12, 14} and several sightlines: the eager kernel, the kernel
jitted with its grid threaded as an argument, forward and reverse derivatives, and the
precompute photometry lookup. The float64 output is pinned bit for bit to the composition
the kernels had before the fix.
"""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_LBOLS = np.array([6.0, 8.0, 10.0, 12.0, 14.0])
_COS = (0.02, 0.55, 1.0)
_WAVE = np.geomspace(1.0e2, 1.0e7, 400)  # Angstrom, UV to the far-IR tail
_FRAC = 0.4
_Z = 0.5

#: Float32 vs float64, ``max|Delta| / max|L_nu|`` over the grid, per point. Measured
#: maximum over every case here: see the PR (the floor is float32 rounding through the
#: PCHIP lookup and the log-power resampling, a few 1e-6).
_RTOL_SED = 1.5e-5

# name: module, grid loader, grid finder, kernel, sightline key, sightlines, extra kwargs,
# precompute module (None when the library has no photometry table).
_CASES = {
    "fritz": (
        "fritz", "load_fritz_grid", "_find_fritz_grid", "fritz_sed_from_grid",
        "agn_fritz_psy", (0.001, 41.3, 89.99), {}, None,
    ),
    "nenkova_agnfitter": (
        "nenkova_agnfitter", "load_nenkova_agnfitter_grid", "_find_nenkova_agnfitter_grid",
        "nenkova_agnfitter_sed_from_grid", "agn_cos_inc", _COS, {},
        "nenkova_agnfitter_precompute",
    ),
    "nenkova_agnfitter_2p": (
        "nenkova_agnfitter_2p", "load_nenkova_agnfitter_2p_grid",
        "_find_nenkova_agnfitter_2p_grid", "nenkova_agnfitter_2p_sed_from_grid",
        "agn_cos_inc", _COS, {"agn_oa_nenkova": 35.0}, "nenkova_agnfitter_2p_precompute",
    ),
    "nenkova_agnfitter_3p": (
        "nenkova_agnfitter_3p", "load_nenkova_agnfitter_3p_grid",
        "_find_nenkova_agnfitter_3p_grid", "nenkova_agnfitter_3p_sed_from_grid",
        "agn_cos_inc", _COS, {"agn_oa_nenkova": 35.0, "agn_tv_nenkova": 60.0},
        "nenkova_agnfitter_3p_precompute",
    ),
    "skirtor_agnfitter": (
        "skirtor_agnfitter", "load_skirtor_agnfitter_grid", "_find_skirtor_agnfitter_grid",
        "skirtor_agnfitter_sed_from_grid", "agn_cos_inc", _COS, {},
        "skirtor_agnfitter_precompute",
    ),
    "skirtor_agnfitter_1p": (
        "skirtor_agnfitter_1p", "load_skirtor_agnfitter_1p_grid",
        "_find_skirtor_agnfitter_1p_grid", "skirtor_agnfitter_1p_sed_from_grid",
        "agn_cos_inc", _COS, {}, "skirtor_agnfitter_1p_precompute",
    ),
    "skirtor_agnfitter_2p": (
        "skirtor_agnfitter_2p", "load_skirtor_agnfitter_2p_grid",
        "_find_skirtor_agnfitter_2p_grid", "skirtor_agnfitter_2p_sed_from_grid",
        "agn_cos_inc", _COS, {}, "skirtor_agnfitter_2p_precompute",
    ),
    "cat3d_wind": (
        "cat3d_wind", "load_cat3d_wind_grid", "_find_cat3d_grid", "cat3d_wind_sed_from_grid",
        "agn_cos_inc", _COS, {"agn_a_cat3d": -1.6, "agn_fwd_cat3d": 1.4}, "cat3d_precompute",
    ),
    "cat3d_wind_lowfwd": (
        "cat3d_wind_lowfwd", "load_cat3d_wind_lowfwd_grid", "_find_cat3d_wind_lowfwd_grid",
        "cat3d_wind_lowfwd_sed_from_grid", "agn_cos_inc", _COS,
        {"agn_a_cat3d_lowfwd": -2.0, "agn_fwd_cat3d_lowfwd": 0.45},
        "cat3d_wind_lowfwd_precompute",
    ),
    "silva04": (
        "silva04", "load_silva04_grid", "_find_silva04_grid", "silva04_sed_from_grid",
        "agn_log_nh_silva", (22.0, 23.3, 24.0), {}, "silva04_precompute",
    ),
}  # fmt: skip


def _fritz_point(psy):
    """The Fritz kernel's default six-axis point at elevation ``psy`` [deg]."""
    return (60.0, 1.0, -0.5, 4.0, 60.0, psy)


def _skirtor_1p_point(cos_inc):
    from tengri.components.agn._skirtor_agnfitter_sightline import skirtor_incl_from_cos_inc

    return (skirtor_incl_from_cos_inc(cos_inc),)


#: Anisotropic libraries: the coordinates their kernel interpolates the anisotropy table at,
#: for the published ledger (``torus_ledger_from_grid``).
_LEDGER_POINT = {
    "fritz": _fritz_point,
    "nenkova_agnfitter": lambda c: (c,),
    "nenkova_agnfitter_2p": lambda c: (c, 35.0),
    "nenkova_agnfitter_3p": lambda c: (c, 35.0, 60.0),
    "skirtor_agnfitter_1p": _skirtor_1p_point,
}


def _ledger(name, mod, grid, lbol, sight):
    """``log_L_agn_torus_los`` [dex re erg/s] the library's torus block publishes."""
    from tengri.components.agn._torus_inclination import library_anisotropy, torus_ledger_from_grid

    inc = mod.TORUS_INCLINATION
    if name == "fritz":
        aniso = mod.fritz_anisotropy(grid)
    else:
        aniso = library_anisotropy(grid, inc, jnp.result_type(float))
    ledger = torus_ledger_from_grid(
        aniso, _LEDGER_POINT[name](sight), inc, agn_log_lbol=lbol, agn_torus_frac=_FRAC
    )
    return ledger["log_L_agn_torus_los"]


def _module(name):
    return importlib.import_module(f"tengri.components.agn.{name}")


def _clear_caches(*modules):
    """Drop every cached loader of ``modules``, so each precision mode builds its own grid."""
    for mod in modules:
        for value in vars(mod).values():
            if callable(getattr(value, "cache_clear", None)):
                value.cache_clear()


def _filters():
    """Five top-hat bands from 1.2 to 200 micron rest frame, observed at ``_Z``."""
    centers = np.geomspace(1.2e4, 2.0e6, 5) * (1.0 + _Z)
    waves = [np.linspace(c * 0.85, c * 1.15, 64) for c in centers]
    return waves, [np.ones(64) for _ in centers]


def _evaluate(name, x64):
    """Every measured output of one torus, in one precision mode, as float64 NumPy."""
    mod_name, loader, finder, kernel, key, sights, extra, pre_name = _CASES[name]
    mod = _module(mod_name)
    pre = _module(pre_name) if pre_name else None
    with jax.enable_x64(x64):
        _clear_caches(mod, *([pre] if pre else []))
        dtype = jnp.float64 if x64 else jnp.float32
        try:
            path = getattr(mod, finder)()
        except FileNotFoundError as exc:
            pytest.skip(f"{name} grid not on disk: {exc}")
        grid = getattr(mod, loader)(path)
        fn = getattr(mod, kernel)
        wave = jnp.asarray(_WAVE, dtype=dtype)
        points = np.array([(lb, s) for lb in _LBOLS for s in sights])

        def kwargs(lbol, sight):
            return {key: sight, "agn_log_lbol": lbol, "agn_torus_frac": _FRAC, **extra}

        def sed(g, lbol, sight):
            return fn(g, wave, **kwargs(lbol, sight))

        def per_point(g, p):
            lbol, sight = p[0], p[1]
            jac = jax.jacfwd(lambda a, b: sed(g, a, b), (0, 1))(lbol, sight)
            # A likelihood-sized cotangent: reverse mode through the luminosity scale.
            grad = jax.grad(lambda a, b: jnp.sum(sed(g, a, b)) * 1e-30, (0, 1))(lbol, sight)
            return sed(g, lbol, sight), jnp.stack(jac), jnp.stack(grad)

        # The grid is threaded through jit as an argument, as the composable runner does;
        # lax.map keeps one point's intermediates live at a time.
        threaded, jac, grad = jax.jit(lambda g, ps: jax.lax.map(lambda p: per_point(g, p), ps))(
            grid, jnp.asarray(points, dtype=dtype)
        )
        top = jnp.asarray(_LBOLS[-1], dtype=dtype)
        eager = [sed(grid, top, jnp.asarray(s, dtype=dtype)) for s in sights]
        out = {
            "threaded": np.asarray(threaded, np.float64),
            "jac": np.asarray(jac, np.float64),
            "grad": np.asarray(grad, np.float64),
            "eager": np.asarray(jnp.stack(eager), np.float64),
        }
        if name in _LEDGER_POINT:
            out["ledger"] = np.array(
                [float(_ledger(name, mod, grid, jnp.asarray(lb, dtype), s)) for lb, s in points]
            )
        if pre is not None:
            waves, trans = _filters()
            lookup = pre.build_lookup(pre.precompute(waves, trans, _Z, None, grid_path=path))
            out["lookup"] = np.stack(
                [
                    np.asarray(lookup(**kwargs(jnp.asarray(lb, dtype), jnp.asarray(s, dtype))))
                    for lb, s in points
                ]
            ).astype(np.float64)
        _clear_caches(mod, *([pre] if pre else []))
    return out


def _peak_relative(a, b):
    """``max|a - b| / max|b|`` per leading row."""
    axes = tuple(range(1, a.ndim))
    return np.max(np.abs(a - b), axis=axes) / np.max(np.abs(b), axis=axes)


@pytest.fixture(scope="module", params=sorted(_CASES))
def measured(request):
    name = request.param
    return name, _evaluate(name, True), _evaluate(name, False)


def test_float32_sed_is_finite_and_matches_float64(measured):
    """Threaded, eager and lookup outputs: finite, and float64's to ``_RTOL_SED``."""
    name, f64, f32 = measured
    for kind in ("threaded", "eager", "lookup"):
        if kind not in f64:
            continue
        assert np.all(np.isfinite(f32[kind])), f"{name} {kind}: non-finite in float32"
        # A threaded table cast to float32 before the fix came back as exactly 0.
        assert np.all(np.max(np.abs(f32[kind]), axis=-1) > 0), f"{name} {kind}: float32 zero"
        assert np.all(np.max(np.abs(f64[kind]), axis=-1) > 0), f"{name} {kind}: float64 zero"
        err = _peak_relative(f32[kind], f64[kind])
        assert err.max() <= _RTOL_SED, f"{name} {kind}: float32 off float64 by {err.max():.2e}"


def test_float32_published_sightline_power_matches_float64(measured):
    """The anisotropic libraries' ``log_L_agn_torus_los``: finite, float64's to 1e-5 dex."""
    name, f64, f32 = measured
    if "ledger" not in f64:
        pytest.skip(f"{name} carries no anisotropy table")
    # grad-assert: finite-only — a log power: 0 dex is a legitimate value, -inf is not.
    assert np.all(np.isfinite(f32["ledger"])), f"{name}: non-finite log_L_agn_torus_los"
    err = np.max(np.abs(f32["ledger"] - f64["ledger"]))
    assert err <= 1e-5, f"{name}: log_L_agn_torus_los off float64 by {err:.2e} dex"


def test_float32_derivatives_are_finite(measured):
    """Forward (jacfwd) and reverse (grad) derivatives in agn_log_lbol and the sightline."""
    name, f64, f32 = measured
    for kind in ("jac", "grad"):
        assert np.all(np.isfinite(f32[kind])), f"{name} {kind}: non-finite in float32"
        # The agn_log_lbol derivative is ln(10) times the SED: never identically zero.
        assert np.any(f32[kind] != 0.0), f"{name} {kind}: identically zero in float32"
        lbol_rows = f32[kind][:, 0].reshape(f32[kind].shape[0], -1)
        # grad-assert: nonzero-only — finiteness of the same array is asserted just above.
        assert np.all(np.any(lbol_rows != 0.0, axis=-1)), f"{name} {kind}: zero d/d agn_log_lbol"
    # d L_nu / d agn_log_lbol = ln(10) L_nu exactly, so the luminosity column is checked
    # against the float64 one to the SED tolerance.
    err = _peak_relative(f32["jac"][:, 0], f64["jac"][:, 0])
    assert err.max() <= _RTOL_SED, f"{name} d/d agn_log_lbol off by {err.max():.2e}"


@pytest.mark.parametrize("name", ["cat3d_wind", "nenkova_agnfitter_2p", "skirtor_agnfitter"])
def test_float64_shape_kernel_is_the_linear_product_bit_for_bit(name):
    """Float64 ``torus_lnu_from_grid`` is the pre-fix linear composition, to the bit."""
    from tengri.components.agn._template_grid import scale_to_lbol_native, torus_lnu_from_grid
    from tengri.utils.grid_interp import interp_nd_pchip, resample_template
    from tengri.utils.physics_constants import L_SUN

    mod_name, loader, finder, *_ = _CASES[name]
    mod = _module(mod_name)
    try:
        grid = getattr(mod, loader)(getattr(mod, finder)())
    except FileNotFoundError as exc:
        pytest.skip(f"{name} grid not on disk: {exc}")
    wave = jnp.asarray(_WAVE)
    coords = tuple(0.5 * (np.asarray(a)[0] + np.asarray(a)[-1]) for a in grid.axes)
    for lbol in _LBOLS:
        new = torus_lnu_from_grid(grid, wave, coords, agn_log_lbol=lbol, agn_torus_frac=_FRAC)
        template = interp_nd_pchip(
            jnp.asarray(grid.template), tuple(jnp.asarray(a) for a in grid.axes), coords
        )
        native = jnp.asarray(grid.wave_grid)
        sed = resample_template(wave, native, template, left=0.0, right=0.0)
        old = scale_to_lbol_native(template, native, sed, 10.0**lbol * L_SUN * _FRAC)
        np.testing.assert_array_equal(np.asarray(new), np.asarray(old))


@pytest.mark.parametrize("name", ["nenkova_agnfitter_3p", "skirtor_agnfitter_1p"])
def test_float64_anisotropy_table_is_unchanged_bit_for_bit(name):
    """Float64 ``anisotropy_values`` is the in-graph ``anisotropy_grid`` ratio, to the bit."""
    from tengri.components.agn._template_grid import native_bolometric_nu_np
    from tengri.components.agn._torus_inclination import anisotropy_grid, anisotropy_values

    mod_name, loader, finder, *_ = _CASES[name]
    mod = _module(mod_name)
    try:
        grid = getattr(mod, loader)(getattr(mod, finder)())
    except FileNotFoundError as exc:
        pytest.skip(f"{name} grid not on disk: {exc}")
    inc = mod.TORUS_INCLINATION
    template = np.asarray(grid.template)
    flat = template.reshape(-1, template.shape[-1])
    powers = np.array([native_bolometric_nu_np(r, grid.wave_grid, floor=0.0) for r in flat])
    old = anisotropy_grid(powers.reshape(template.shape[:-1]), grid.axes, inc).values
    new = anisotropy_values(template, grid.wave_grid, grid.axes, inc)
    np.testing.assert_array_equal(new, np.asarray(old))
    np.testing.assert_array_equal(np.asarray(grid.anisotropy), new)
