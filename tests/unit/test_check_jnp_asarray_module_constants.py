# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tools/check_jnp_asarray_module_constants.py (#2774).

A dtype-free ``jnp.asarray`` on a module-level numpy constant caches a buffer whose
dtype depends on the x64 state at the first call. The guard must pass on src/tengri
and must flag a synthetic offender.
"""

import ast
import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_jnp_asarray_module_constants.py"
_spec = importlib.util.spec_from_file_location("check_jnp_asarray_module_constants", _TOOL)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _flags(source: str, extra_names: frozenset[str] = frozenset()) -> list[int]:
    tree = ast.parse(source)
    names = mod.module_numpy_constants(tree) | set(extra_names)
    return [lineno for lineno, _ in mod.check_tree(tree, names)]


def test_src_tree_is_clean():
    """The live src/tengri tree has no dtype-free jnp.asarray on a numpy constant."""
    assert mod.main([]) == 0


def test_flags_dtype_free_asarray_of_numpy_constant():
    source = (
        "import numpy as np\n"
        "import jax.numpy as jnp\n"
        "_TABLE = np.linspace(0.0, 1.0, 8)\n"
        "def f():\n"
        "    return jnp.asarray(_TABLE)\n"
    )
    assert _flags(source) == [5]


def test_flags_subscript_and_tuple_unpacked_constants():
    source = (
        "import numpy as np\n"
        "import jax.numpy as jnp\n"
        "_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(16)\n"
        "def f():\n"
        "    return jnp.asarray(_NODES), jnp.asarray(_WEIGHTS[:4])\n"
    )
    assert _flags(source) == [5, 5]


def test_flags_constant_from_host_array_helper():
    source = (
        "import jax.numpy as jnp\n"
        "from tengri.utils.host_array import host_array\n"
        "_TAB = host_array([1.0, 2.0])\n"
        "def f():\n"
        "    return jnp.asarray(_TAB)\n"
    )
    assert _flags(source) == [5]


def test_flags_constant_imported_from_another_module():
    source = (
        "import jax.numpy as jnp\n"
        "from tengri.x import _TAB\n"
        "def f():\n"
        "    return jnp.asarray(_TAB)\n"
    )
    assert _flags(source, extra_names=frozenset({"_TAB"})) == [4]


def test_explicit_dtype_and_device_table_are_not_flagged():
    source = (
        "import numpy as np\n"
        "import jax.numpy as jnp\n"
        "from tengri.utils.host_array import device_table\n"
        "_TABLE = np.linspace(0.0, 1.0, 8)\n"
        "def f():\n"
        "    a = jnp.asarray(_TABLE, dtype=jnp.float32)\n"
        "    b = device_table(_TABLE)\n"
        "    return a, b\n"
    )
    assert _flags(source) == []


def test_local_rebinding_is_not_a_module_constant():
    source = (
        "import numpy as np\n"
        "import jax.numpy as jnp\n"
        "_TABLE = np.linspace(0.0, 1.0, 8)\n"
        "def f(_TABLE):\n"
        "    return jnp.asarray(_TABLE)\n"
    )
    assert _flags(source) == []


def test_plain_python_constant_is_not_flagged():
    source = "import jax.numpy as jnp\n_X = 500.0\ndef f():\n    return jnp.asarray(_X)\n"
    assert _flags(source) == []
