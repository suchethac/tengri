# SPDX-License-Identifier: BSD-3-Clause
"""Galaxy-axis sharding for multi-device population fits.

A hierarchical fit is one high-dimensional problem, not N independent ones: the
per-galaxy latents and the shared PSD hyperparameters live in a single
parameter vector. Distributing it means splitting the *galaxy* axis across
devices and letting the likelihood's reduction combine the pieces.

The canonical population forward model reaches that axis through ``jax.vmap``
(:mod:`tengri.forward.population_sed_model`), which GSPMD partitions from a
sharding placed on the data alone — no ``shard_map`` and no manual
``jax.lax.psum``. This module therefore does one thing: commit the leading axis
of the runtime ``data_args`` to a device mesh. XLA propagates that sharding
backwards through the vmapped forward model and lowers the ``chi2`` sum to an
all-reduce.

That propagation is a property of batched primitives, not of sharding in
general. A ``lax.map`` galaxy loop is a sequential scan, and sharding a scan's
loop axis does *not* distribute it — GSPMD all-gathers and every device runs
every galaxy, measured at exactly ``n_devices`` slower with correct results and
zero speedup. Anything routed through this module must reach its galaxy axis by
``vmap``.

Notes
-----
**JIT-compatible**: :func:`shard_leading_axis` is an eager ``device_put`` on
concrete arrays and must be called *outside* ``jax.jit``. The sharding it
applies is part of the lowering key, so a jitted callable recompiles when handed
sharded arguments — the Python-level engine caches need no sharding-aware key.
"""

from __future__ import annotations

import jax
import numpy as np

__all__ = ["make_mesh", "replicate", "resolve_devices", "shard_leading_axis"]

#: Mesh axis name for the galaxy dimension.
GALAXY_AXIS = "gal"


def resolve_devices(devices):
    """Normalize the ``devices`` argument to a device list or ``None``.

    Parameters
    ----------
    devices : None or {'all'} or list of Device
        ``None`` selects the single-device path; ``'all'`` selects every
        :func:`jax.devices`; a list or tuple selects those devices.

    Returns
    -------
    list of Device or None
        The resolved device list, or ``None`` for the single-device path.

    Raises
    ------
    ValueError
        If ``devices`` is a string other than ``'all'``.
    """
    if devices is None:
        return None
    if isinstance(devices, str):
        if devices == "all":
            return list(jax.devices())
        raise ValueError(f"devices must be None, 'all', or a device list; got {devices!r}")
    return list(devices)


def make_mesh(devices):
    """Build the one-dimensional galaxy mesh, or ``None`` for one device.

    Parameters
    ----------
    devices : None or {'all'} or list of Device
        Devices to build the mesh over, as accepted by :func:`resolve_devices`.

    Returns
    -------
    jax.sharding.Mesh or None
        A mesh with the single named axis ``'gal'``, or ``None`` when the fit
        runs on one device and needs no sharding at all.
    """
    devs = resolve_devices(devices)
    if devs is None or len(devs) < 2:
        return None
    return jax.sharding.Mesh(np.asarray(devs, dtype=object), (GALAXY_AXIS,))


def replicate(x, mesh):
    """Pin ``x`` to a replicated layout so galaxy sharding cannot spread into it.

    Constraints here *contain* parallelism rather than create it. GSPMD
    propagates a sharding outwards from wherever it is introduced, and the
    optimizer state of a hierarchical fit — the flat parameter vector, the CG
    iterates, the PRNG keys — has no galaxy axis to split. Left unpinned, that
    propagation reaches a ``uint32[2]`` PRNG key and fails to lower with
    ``dim_size=2 is not divisible by axis_size=n_devices``.

    Parameters
    ----------
    x : array_like
        Value that must stay replicated across devices.
    mesh : jax.sharding.Mesh or None
        Mesh from :func:`make_mesh`. ``None`` makes this the identity, so
        single-device builds carry no sharding machinery at all.

    Returns
    -------
    ndarray
        ``x``, constrained to a replicated sharding when ``mesh`` is given.

    Notes
    -----
    **JIT-compatible** — this is the traced ``with_sharding_constraint``, and is
    intended to be called inside ``jax.jit``.
    """
    if mesh is None:
        return x
    return jax.lax.with_sharding_constraint(
        x, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    )


def shard_leading_axis(tree, devices):
    """Commit the leading (galaxy) axis of every array in ``tree`` to a mesh.

    Rank-0 leaves are left replicated: they carry no galaxy axis to split.

    Parameters
    ----------
    tree : pytree of ndarray
        Arrays whose axis 0 is the galaxy axis (e.g. a ``data_args`` dict with
        ``data``/``noise`` of shape ``(n_gal, n_data)``). Rank-0 leaves such as
        ``n_data`` are passed through untouched.
    devices : None or {'all'} or list of Device
        Devices to shard across, as accepted by :func:`resolve_devices`.

    Returns
    -------
    pytree of ndarray
        A new pytree with the same structure and values. Returned unchanged
        when ``devices`` resolves to ``None`` or to a single device.

    Raises
    ------
    ValueError
        If the galaxy axis is not divisible by the device count, or if leaves
        disagree about its length.

    Notes
    -----
    **Not JIT-compatible** — call outside ``jax.jit`` on concrete arrays.

    The galaxy axis must divide evenly across devices. Padding it is *not* safe
    here the way it is for an independent-galaxy catalog: a hierarchical fit
    couples every galaxy through the shared hyperparameters, so a padded dummy
    contributes to their gradient. Refusing to run is the honest failure.
    """
    devs = resolve_devices(devices)
    if devs is None or len(devs) < 2:
        return tree

    leaves = jax.tree.leaves(tree)
    batched = [np.shape(x)[0] for x in leaves if np.ndim(x) >= 1]
    if not batched:
        return tree
    if len(set(batched)) != 1:
        raise ValueError(
            f"leaves disagree on the galaxy-axis length: found {sorted(set(batched))}. "
            "Every batched leaf must share axis 0."
        )
    n_gal = batched[0]
    n_dev = len(devs)
    if n_gal % n_dev:
        raise ValueError(
            f"galaxy axis ({n_gal}) must be divisible by the device count ({n_dev}); "
            f"{n_gal % n_dev} galaxies would be left over. Pass a galaxy count that "
            "divides evenly, or fewer devices."
        )

    mesh = jax.sharding.Mesh(np.asarray(devs, dtype=object), (GALAXY_AXIS,))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(GALAXY_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def _put(x):
        """Shard a batched leaf; replicate a scalar one."""
        return jax.device_put(x, replicated if np.ndim(x) == 0 else sharding)

    return jax.tree.map(_put, tree)
