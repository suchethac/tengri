# SPDX-License-Identifier: BSD-3-Clause
r"""A cached torus loader never hands a float32 load to a float64 run (#3101, #3084).

Every template-torus loader caches what it loads, and what it loads depends on the process
precision: ``jnp.array`` takes the dtype of the mode active at the first call, and
``loaded_template`` rescales a float32 table by its peak. A cache keyed on the path alone
therefore hands a float32-first load to every later float64 run in the same process: the
grid leaves are float32 and the float64 SED is off by ~6e-7 relative while its output dtype
still reads float64 (the Fritz loaders, before they were keyed), or NaN where a float32
anisotropy table was 0/0.

The contract: load in float32 first, then in float64 in the same process; the float64 load
holds only float64 arrays, and its SED equals a fresh, uncached float64 evaluation bit for
bit. Every loader decorated with ``cache_per_float_dtype`` is a case (a source scan keeps
the table complete), plus the Fritz loaders, which carry the same state.
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

_CASES = [(m, _LIBRARIES[m][1]) for m in _LIBRARIES] + [
    (m, name) for m, names in _DEFAULTS.items() for name in names
]


def _decorated_loaders():
    """``(module, function)`` of every ``@cache_per_float_dtype`` in ``components/agn``."""
    found = set()
    for path in sorted(_AGN_SRC.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef) and any(
                isinstance(d, ast.Name) and d.id == "cache_per_float_dtype"
                for d in node.decorator_list
            ):
                found.add((path.stem, node.name))
    return found


def _held_arrays(obj):
    """Every array a loaded object holds: pytree leaves, partial arguments, closure cells."""
    if isinstance(obj, functools.partial):
        parts = [*obj.args, *obj.keywords.values()]
    elif callable(obj):
        parts = [c.cell_contents for c in (obj.__closure__ or ()) if not callable(c.cell_contents)]
    else:
        parts = [obj]
    leaves = jax.tree_util.tree_leaves(parts)
    return [x for x in leaves if isinstance(x, (np.ndarray, jax.Array))]


def _evaluator(mod_name, loader_name):
    """``(loader args, fn(loaded, wave) -> L_nu)`` for one case."""
    finder, grid_loader, kernel, extra = _LIBRARIES[mod_name]
    mod = importlib.import_module(f"tengri.components.agn.{mod_name}")
    kwargs = {"agn_log_lbol": 10.0, "agn_torus_frac": 0.4, **extra}
    if loader_name == grid_loader:
        try:
            args = (getattr(mod, finder)(),)
        except FileNotFoundError as exc:
            pytest.skip(f"{mod_name} grid not on disk: {exc}")
        fn = getattr(mod, kernel)
        return args, lambda grid, wave: fn(grid, wave, **kwargs)
    try:
        getattr(mod, finder)()
    except FileNotFoundError as exc:
        pytest.skip(f"{mod_name} grid not on disk: {exc}")

    def bound(loaded, wave):
        return jnp.stack(jax.tree_util.tree_leaves(loaded(wave, **kwargs)))

    return (), bound


def test_case_table_covers_every_dtype_keyed_loader():
    """A new ``@cache_per_float_dtype`` loader is a case here, not silently untested."""
    missing = _decorated_loaders() - set(_CASES)
    assert not missing, f"add these loaders to _LIBRARIES / _DEFAULTS: {sorted(missing)}"


@pytest.mark.parametrize(("mod_name", "loader_name"), _CASES, ids=[f"{m}.{n}" for m, n in _CASES])
def test_float32_first_load_does_not_reach_float64(mod_name, loader_name):
    """Float32 load, then float64 in one process: float64 leaves, bitwise the fresh value."""
    mod = importlib.import_module(f"tengri.components.agn.{mod_name}")
    loader = getattr(mod, loader_name)
    args, evaluate = _evaluator(mod_name, loader_name)
    loader.cache_clear()
    try:
        with jax.enable_x64(False):
            first = loader(*args)
            np.asarray(evaluate(first, jnp.asarray(_WAVE, jnp.float32)))
        with jax.enable_x64(True):
            wave = jnp.asarray(_WAVE, jnp.float64)
            loaded = loader(*args)
            after = np.asarray(evaluate(loaded, wave))
            fresh = np.asarray(evaluate(loader.__wrapped__(*args), wave))
            dtypes = {
                str(np.asarray(x).dtype)
                for x in _held_arrays(loaded)
                if np.issubdtype(np.asarray(x).dtype, np.floating)
            }
    finally:
        loader.cache_clear()
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
