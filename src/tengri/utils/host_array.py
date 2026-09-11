# SPDX-License-Identifier: BSD-3-Clause
"""Module-scope numeric tables: host-side at import, device-side per trace.

A constant written as ``jnp.array([...])`` at module scope is a device buffer
allocated while the module imports, so its dtype is fixed by the x64 state at
*import* time. That had two measured consequences (#2271):

* On a backend with no float64 (jax-mps / MLX on Apple GPU) the allocation
  itself raises and ``import tengri`` fails before any model is built.
* With ``JAX_ENABLE_X64=0`` the constants are float32 for the life of the
  process, and every float64 reference computed later (the whole pytest suite,
  which forces x64 back on) is silently contaminated by them.

A plain numpy table at module scope fixes the allocation but not the dtype:
JAX 0.11 memoizes a numpy object's device conversion by object identity, with
the dtype of its first conversion. The same module-level array reused across
traces under a different x64 state comes back with the old dtype: loudly (an
MLIR verifier error at ``broadcast_in_dim``) when it meets a tracer in an
operator, and silently from ``jnp.take`` / indexing. An explicit dtype at the
conversion is what breaks the memo, and ``__jax_array__`` cannot do it
implicitly: JAX refuses the protocol for jit arguments.

So the contract has two halves, both mechanical:

* :func:`host_array` at module scope: a plain, C-contiguous float64 numpy
  array. No device buffer exists until a trace asks for one. C-contiguous
  because jax-mps refuses a negative host stride at ``device_put`` (a flipped
  view; measured on the SKIRTOR cubes).
* :func:`device_table` at every use inside JAX code: the device copy in the
  canonical dtype of the current trace, made fresh at that use. Host-side
  uses (``np.asarray``, ``len``, ``.shape``) read the table directly.

``tools/check_host_tables.py`` enforces the second half; the census in
``tests/regression/precision/test_no_float64_constants_under_f32.py`` the first.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["device_table", "host_array"]


def host_array(values, dtype=np.float64) -> np.ndarray:
    """A module-scope table on the host: plain numpy, C-contiguous, no device buffer.

    Parameters
    ----------
    values : array_like
        The table. Any shape, including 0-d for a scalar that must stay
        strongly typed on the device (see :func:`device_table`).
    dtype : numpy dtype, optional
        Host dtype of the stored copy. Keep ``np.float64`` for physical tables
        so a float64 trace sees the full-precision values; a float32 trace
        rounds at :func:`device_table`.

    Returns
    -------
    ndarray
        The table, C-contiguous, owning its memory, with the rank of ``values``.
    """
    arr = np.array(values, dtype=dtype, copy=True)
    # np.ascontiguousarray promotes a 0-d array to shape (1,); a 0-d copy is
    # already contiguous and must stay a scalar (it enters jnp.stack calls).
    return arr if arr.ndim == 0 else np.ascontiguousarray(arr)


def device_table(table) -> jnp.ndarray:
    """The device copy of a host table in the canonical dtype of the current trace.

    Call it at the point of use inside JAX code, never at module scope. The
    dtype is passed explicitly so JAX's identity-keyed conversion memo cannot
    hand back a copy made under the other x64 state. A 0-d table becomes a
    strongly typed device scalar, which promotes a float32 operand to the
    process dtype exactly as the 0-d ``jnp`` constants it replaces did.

    Parameters
    ----------
    table : ndarray
        A :func:`host_array` (or any numpy array).

    Returns
    -------
    ndarray
        Device array of the same shape; ``float64`` under x64, ``float32``
        otherwise (integer tables keep their canonical integer width).
    """
    return jnp.asarray(table, dtype=jax.dtypes.canonicalize_dtype(np.asarray(table).dtype))
