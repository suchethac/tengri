# SPDX-License-Identifier: BSD-3-Clause
r"""A cached AGN template loader never hands a float32 load to a float64 run (#3101, #3084).

Every template loader caches what it loads, and what it loads depends on the process
precision: ``jnp.array`` takes the dtype of the mode active at the first call, and
``loaded_template`` rescales a float32 table by its peak. A cache keyed on the path alone
therefore hands a float32-first load to every later float64 run in the same process: the
grid leaves are float32 and the float64 SED is off by ~6e-7 relative while its output dtype
still reads float64 (the Fritz loaders, before they were keyed), or NaN where a float32
anisotropy table was 0/0.

The contract: load in float32 first, then in float64 in the same process; the float64 load
holds only float64 arrays, and its SED equals a fresh, uncached float64 evaluation bit for
bit. Two source scans over every module under ``components/agn`` (subpackages included)
keep it complete:

* every loader decorated with ``cache_per_float_dtype`` (bare or attribute form) is a case;
* every other ``functools.cache`` / ``lru_cache`` there is on a recorded list of caches that
  hold no device array, each with its reason. A new path-only cache over a template fails
  until it is keyed on the dtype or argued onto the list.

One loader per scaling code path runs in the PR tier; the rest run in the ``slow`` tier.
"""

import ast
import functools
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

_AGN_SRC = Path(__file__).resolve().parents[2] / "src" / "tengri" / "components" / "agn"
_WAVE = np.geomspace(1.0e2, 1.0e7, 300)  # Angstrom
_POWER = {"agn_log_lbol": 10.0, "agn_torus_frac": 0.4}

_COS = {"agn_cos_inc": 0.55}
#: module: (grid finder, grid loader, kernel, kernel kwargs past the luminosity).
_LIBRARIES = {
    "fritz": ("_find_fritz_grid", "load_fritz_grid", "fritz_sed_from_grid",
              {"agn_fritz_psy": 41.3}),
    "nenkova_agnfitter": ("_find_nenkova_agnfitter_grid", "load_nenkova_agnfitter_grid",
                          "nenkova_agnfitter_sed_from_grid", _COS),
    "nenkova_agnfitter_2p": ("_find_nenkova_agnfitter_2p_grid", "load_nenkova_agnfitter_2p_grid",
                             "nenkova_agnfitter_2p_sed_from_grid",
                             {**_COS, "agn_oa_nenkova": 35.0}),
    "nenkova_agnfitter_3p": ("_find_nenkova_agnfitter_3p_grid", "load_nenkova_agnfitter_3p_grid",
                             "nenkova_agnfitter_3p_sed_from_grid",
                             {**_COS, "agn_oa_nenkova": 35.0, "agn_tv_nenkova": 60.0}),
    "skirtor_agnfitter": ("_find_skirtor_agnfitter_grid", "load_skirtor_agnfitter_grid",
                          "skirtor_agnfitter_sed_from_grid", _COS),
    "skirtor_agnfitter_1p": ("_find_skirtor_agnfitter_1p_grid", "load_skirtor_agnfitter_1p_grid",
                             "skirtor_agnfitter_1p_sed_from_grid", _COS),
    "skirtor_agnfitter_2p": ("_find_skirtor_agnfitter_2p_grid", "load_skirtor_agnfitter_2p_grid",
                             "skirtor_agnfitter_2p_sed_from_grid", _COS),
    "cat3d_wind": ("_find_cat3d_grid", "load_cat3d_wind_grid", "cat3d_wind_sed_from_grid",
                   {**_COS, "agn_a_cat3d": -1.6, "agn_fwd_cat3d": 1.4}),
    "cat3d_wind_lowfwd": ("_find_cat3d_wind_lowfwd_grid", "load_cat3d_wind_lowfwd_grid",
                          "cat3d_wind_lowfwd_sed_from_grid",
                          {**_COS, "agn_a_cat3d_lowfwd": -2.0, "agn_fwd_cat3d_lowfwd": 0.45}),
    "silva04": ("_find_silva04_grid", "load_silva04_grid", "silva04_sed_from_grid",
                {"agn_log_nh_silva": 23.3}),
}  # fmt: skip

#: Zero-argument loaders that return the kernel bound to the default grid.
_DEFAULTS = {
    "fritz": ("_load_fritz_default", "_load_fritz_components"),
    "nenkova_agnfitter": ("_load_nenkova_agnfitter_default",),
    "nenkova_agnfitter_2p": ("_load_nenkova_agnfitter_2p_default",),
    "nenkova_agnfitter_3p": ("_load_nenkova_agnfitter_3p_default",),
    "skirtor_agnfitter": ("_load_skirtor_agnfitter_default",),
    "skirtor_agnfitter_1p": ("_load_skirtor_agnfitter_1p_default",),
    "skirtor_agnfitter_2p": ("_load_skirtor_agnfitter_2p_default",),
    "cat3d_wind": ("_load_cat3d_default",),
    "cat3d_wind_lowfwd": ("_load_cat3d_wind_lowfwd_default",),
    "silva04": ("_load_silva04_default",),
}

#: Zero-argument loaders of other modules' grids: (module, loader) -> (finder module, finder,
#: kernel kwargs past the luminosity, index of the callable in what the loader returns).
_BOUND = {
    ("torus", "_load_nenkova_default"): ("torus", "_find_nenkova_grid", {"agn_tau": 30.0}, None),
    ("unified", "_load_skirtor_fn"): ("skirtor", "_find_skirtor_grid", _COS, None),
    ("unified", "_load_skirtor_raw_total_fn"): ("skirtor", "_find_skirtor_grid", _COS, 0),
}

#: One loader per scaling code path, in the PR tier.
_PR_TIER = {
    ("cat3d_wind", "load_cat3d_wind_grid"),  # torus_lnu_from_grid
    ("nenkova_agnfitter_2p", "load_nenkova_agnfitter_2p_grid"),  # torus_lnu_anisotropic
    ("fritz", "load_fritz_grid"),  # scale_by_torus_power
    ("silva04", "load_silva04_grid"),  # scale_to_torus_power_native
}

_CASES = (
    [(m, _LIBRARIES[m][1]) for m in _LIBRARIES]
    + [(m, name) for m, names in _DEFAULTS.items() for name in names]
    + list(_BOUND)
)
_TIERED = [
    pytest.param(*case, id=f"{case[0]}.{case[1]}")
    if case in _PR_TIER
    else pytest.param(*case, id=f"{case[0]}.{case[1]}", marks=pytest.mark.slow)
    for case in _CASES
]

#: ``functools.cache`` / ``lru_cache`` sites under ``components/agn`` that hold no device
#: array a later precision mode could inherit, with the reason. ``"dtype-keyed"`` marks a
#: cache whose first argument is the process float dtype name (#2275), checked below.
_HOST_ONLY_CACHES = {
    ("_nthcomp", "_load_nthcomp_templates_impl"): "NumPy host arrays (np.asarray), by design",
    ("_template_grid", "_for_dtype"): "the inner cache of cache_per_float_dtype; keyed on dtype",
    ("blr", "_fe2_spectral_tables"): "NumPy float64 host constants of the FeII path",
    ("disc", "_load_relagn_disc_grid"): "host_array tables; device conversion at the use",
    ("fritz", "_warn_legacy_unit_tag"): "warns once per path; returns None",
    ("grahsp.templates", "load_grahsp_templates"): "NumPy host arrays read from the HDF5",
    ("kd18_agnfitter", "load_kd18_agnfitter_grid"): "NumPy host arrays (np.asarray)",
    ("kd18_agnfitter", "_load_default"): "partial over the NumPy host grid above",
    ("kd18_agnfitter", "kd18_agnfitter_grid_support"): "Python float bounds",
    ("kd18_agnfitter", "load_kd18_agnfitter_warmindex_grid"): "NumPy host arrays (np.asarray)",
    ("kd18_agnfitter", "_load_default_warmindex"): "partial over the NumPy host grid above",
    ("kd18_agnfitter", "kd18_agnfitter_warmindex_grid_support"): "Python float bounds",
    ("slone_netzer", "load_slone_netzer_grid"): "NumPy host arrays (np.asarray)",
    ("slone_netzer", "_load_default"): "partial over the NumPy host grid above",
    ("slone_netzer", "slone_netzer_grid_support"): "Python float bounds",
    ("unified", "_load_relagn_fn"): "partial over the host_array RELAGN grid",
    ("skirtor", "_load_skirtor_default_grid_for"): "dtype-keyed",
    ("skirtor", "_load_raw_disk_dust_grid_for"): "dtype-keyed",
    ("skirtor", "_load_skirtor_default_for"): "dtype-keyed",
    ("skirtor", "_load_skirtor_components_for"): "dtype-keyed",
    ("skirtor", "_load_skirtor_disc_attenuation_for"): "dtype-keyed",
    ("skirtor", "load_skirtor_disc_atten_grid_for"): "dtype-keyed",
}


def _decorator_names(node):
    """Names of a function's decorators: ``x``, ``a.x`` and ``x(...)`` all give ``x``."""
    names = set()
    for d in node.decorator_list:
        d = d.func if isinstance(d, ast.Call) else d
        if isinstance(d, ast.Name):
            names.add(d.id)
        elif isinstance(d, ast.Attribute):
            names.add(d.attr)
    return names


def _cached_functions():
    """``(module, node, source)`` of every function defined under ``agn``, recursively."""
    for path in sorted(_AGN_SRC.rglob("*.py")):
        module = ".".join(path.relative_to(_AGN_SRC).with_suffix("").parts)
        text = path.read_text()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield module, node, text


def _decorated_loaders():
    """``(module, function)`` of every ``cache_per_float_dtype`` loader under ``agn``."""
    return {
        (m, n.name)
        for m, n, _ in _cached_functions()
        if "cache_per_float_dtype" in _decorator_names(n)
    }


def _path_only_caches():
    """``(module, function) -> (node, source)`` of every functools cache under ``agn``."""
    return {
        (m, n.name): (n, text)
        for m, n, text in _cached_functions()
        if _decorator_names(n) & {"cache", "lru_cache"}
    }


def _held_arrays(obj, seen=None):
    """Every array a loaded object holds, through partials, closures and containers."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return []
    seen.add(id(obj))
    if isinstance(obj, (np.ndarray, jax.Array)):
        return [obj]
    if isinstance(obj, functools.partial):
        parts = [obj.func, *obj.args, *obj.keywords.values()]
    elif isinstance(obj, (tuple, list)):
        parts = list(obj)
    elif isinstance(obj, dict):
        parts = list(obj.values())
    elif callable(obj) and getattr(obj, "__closure__", None):
        parts = [c.cell_contents for c in obj.__closure__]
    else:
        parts = [x for x in jax.tree_util.tree_leaves(obj) if x is not obj]
    return [a for p in parts for a in _held_arrays(p, seen)]


def _module(name):
    return importlib.import_module(f"tengri.components.agn.{name}")


def _skip_without(mod_name, finder):
    try:
        getattr(_module(mod_name), finder)()
    except FileNotFoundError as exc:
        pytest.skip(f"{mod_name} grid not on disk: {exc}")


def _evaluator(mod_name, loader_name):
    """``(loader args, fn(loaded, wave) -> L_nu)`` for one case."""
    if (mod_name, loader_name) in _BOUND:
        finder_mod, finder, extra, index = _BOUND[(mod_name, loader_name)]
        _skip_without(finder_mod, finder)

        def bound_other(loaded, wave):
            fn = loaded if index is None else loaded[index]
            return fn(wave, **_POWER, **extra)

        return (), bound_other
    finder, grid_loader, kernel, extra = _LIBRARIES[mod_name]
    kwargs = {**_POWER, **extra}
    if loader_name == grid_loader:
        try:
            args = (getattr(_module(mod_name), finder)(),)
        except FileNotFoundError as exc:
            pytest.skip(f"{mod_name} grid not on disk: {exc}")
        fn = getattr(_module(mod_name), kernel)
        return args, lambda grid, wave: fn(grid, wave, **kwargs)
    _skip_without(mod_name, finder)

    def bound(loaded, wave):
        return jnp.stack(jax.tree_util.tree_leaves(loaded(wave, **kwargs)))

    return (), bound


def _clear_caches(mod):
    """Drop every cached loader of ``mod``, so each case starts from an empty cache."""
    for value in vars(mod).values():
        if callable(getattr(value, "cache_clear", None)):
            value.cache_clear()


def test_case_table_covers_every_dtype_keyed_loader():
    """A new ``cache_per_float_dtype`` loader, anywhere under ``agn``, is a case here."""
    missing = _decorated_loaders() - set(_CASES)
    assert not missing, f"add these loaders to _LIBRARIES / _DEFAULTS / _BOUND: {sorted(missing)}"


def test_no_template_loader_is_left_on_a_path_only_cache():
    """Every other functools cache under ``agn`` is recorded as holding no device array."""
    caches = _path_only_caches()
    unrecorded = sorted(set(caches) - set(_HOST_ONLY_CACHES))
    assert not unrecorded, (
        f"path-only caches not recorded as host-only: {unrecorded}. A cache that holds a jnp "
        "array hands one precision mode's arrays to the other; key it with "
        "_template_grid.cache_per_float_dtype, or add it to _HOST_ONLY_CACHES with the reason."
    )
    stale = sorted(set(_HOST_ONLY_CACHES) - set(caches))
    assert not stale, f"_HOST_ONLY_CACHES names caches that no longer exist: {stale}"
    for key, reason in _HOST_ONLY_CACHES.items():
        node, text = caches[key]
        if reason == "dtype-keyed":
            first = node.args.args[0].arg if node.args.args else None
            assert first == "float_dtype_name", f"{key}: recorded dtype-keyed, first arg {first}"
            continue
        body = ast.get_source_segment(text, node) or ""
        assert "jnp.array(" not in body and "jnp.asarray(" not in body, (
            f"{key} is recorded as host-only ({reason}) but builds a jnp array"
        )


@pytest.mark.parametrize(("mod_name", "loader_name"), _TIERED)
def test_float32_first_load_does_not_reach_float64(mod_name, loader_name):
    """Float32 load, then float64 in one process: float64 leaves, bitwise the fresh value."""
    mod = _module(mod_name)
    loader = getattr(mod, loader_name)
    args, evaluate = _evaluator(mod_name, loader_name)
    _clear_caches(mod)
    try:
        with jax.enable_x64(False):
            first = loader(*args)
            np.asarray(evaluate(first, jnp.asarray(_WAVE, jnp.float32)))
        with jax.enable_x64(True):
            wave = jnp.asarray(_WAVE, jnp.float64)
            loaded = loader(*args)
            after = np.asarray(evaluate(loaded, wave))
            fresh = np.asarray(evaluate(loader.__wrapped__(*args), wave))
            held = _held_arrays(loaded)
            dtypes = {
                str(np.asarray(x).dtype)
                for x in held
                if np.issubdtype(np.asarray(x).dtype, np.floating)
            }
    finally:
        _clear_caches(mod)
    assert held, f"{mod_name}.{loader_name}: no array found in the loaded object"
    assert dtypes <= {"float64"}, f"{mod_name}.{loader_name}: float64 load holds {dtypes}"
    assert after.dtype == np.float64
    np.testing.assert_array_equal(after, fresh)


def test_keyword_and_positional_calls_share_one_entry():
    """``load(path)`` and ``load(grid_path=path)`` bind to one cache entry per dtype."""
    from tengri.components.agn import cat3d_wind

    loader = cat3d_wind.load_cat3d_wind_grid
    try:
        path = cat3d_wind._find_cat3d_grid()
    except FileNotFoundError as exc:
        pytest.skip(f"cat3d_wind grid not on disk: {exc}")
    loader.cache_clear()
    try:
        with jax.enable_x64(True):
            assert loader(grid_path=path) is loader(path)
            assert loader.cache_info().currsize == 1
        with jax.enable_x64(False):
            loader(grid_path=path)
            assert loader.cache_info().currsize == 2
    finally:
        loader.cache_clear()
