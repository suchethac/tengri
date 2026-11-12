# SPDX-License-Identifier: BSD-3-Clause
"""Workaround for an XLA CPU miscompile under ``jax.vmap(jax.jit(...))`` (#3350).

With an SSP grid whose flux is constant along wavelength, a batched
``einsum`` over a closure-captured constant is fused by XLA CPU into a
YNNPACK kernel (``__ynn_fusion``) that returns wrong values inside
``jit(vmap(...))``: ``predict_photometry`` came back as exact zeros. The same
arithmetic is correct when the array is an argument, when it is perturbed,
and in eager and single-draw ``jit``. Disabling that fusion type with
``--xla_cpu_experimental_ynn_fusion_type=`` restores the correct result.

Gate. The flag is applied only when the installed jax is in
:data:`AFFECTED_JAX_VERSIONS`, the versions where the miscompile was measured
(``0.11.1`` and ``0.12.0.1``, both on macOS arm64 CPU). Other versions are
untested and are left alone. A user who has set the flag in ``XLA_FLAGS``
keeps their own setting. The flag is appended to ``XLA_FLAGS``, never
overwriting it, and only while the XLA backend is still uninitialised, since
the backend reads ``XLA_FLAGS`` once. If the backend is already up the flag
would be silently ignored, so a :class:`RuntimeWarning` is raised instead.

Remove this module once an affected jax version is fixed upstream and the
version is dropped from :data:`AFFECTED_JAX_VERSIONS`.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import sys
import warnings
from collections.abc import Mapping
from typing import Literal

AFFECTED_JAX_VERSIONS: frozenset[str] = frozenset({"0.11.1", "0.12.0.1"})
"""jax releases where the closure-constant ``jit(vmap)`` miscompile was measured."""

YNN_FUSION_FLAG = "--xla_cpu_experimental_ynn_fusion_type="
"""Disables the YNNPACK fusion type; the empty value is the one measured to work."""

_FLAG_NAME = "xla_cpu_experimental_ynn_fusion_type"

Status = Literal["not_affected", "user_set", "applied", "too_late"]

_log = logging.getLogger(__name__)


def _installed_jax_version() -> str | None:
    """Version of the installed jax distribution, or ``None`` if absent."""
    try:
        return importlib.metadata.version("jax")
    except importlib.metadata.PackageNotFoundError:
        return None


def _backend_initialised() -> bool:
    """True when jax is imported and its XLA backend already exists.

    Reads ``sys.modules`` only, so tengri never imports jax to ask.
    """
    bridge = sys.modules.get("jax._src.xla_bridge")
    if bridge is None:
        return False
    return bool(bridge.backends_are_initialized())


def apply_ynn_fusion_workaround(
    environ: dict[str, str] | Mapping[str, str] | None = None,
    *,
    jax_version: str | None = None,
    backend_initialised: bool | None = None,
) -> Status:
    """Append the YNN fusion flag to ``XLA_FLAGS`` when the gate allows it.

    Parameters
    ----------
    environ : dict, optional
        Mutable environment. Defaults to ``os.environ``. Only ``XLA_FLAGS``
        is written.
    jax_version : str, optional
        Installed jax version. Defaults to the distribution metadata.
    backend_initialised : bool, optional
        Whether the XLA backend already exists. Defaults to a ``sys.modules``
        check.

    Returns
    -------
    str
        ``"not_affected"`` (version outside the gate), ``"user_set"`` (the
        user's ``XLA_FLAGS`` already names the flag), ``"applied"``, or
        ``"too_late"`` (backend up, flag not applied; a warning was raised).
    """
    env = os.environ if environ is None else environ
    version = _installed_jax_version() if jax_version is None else jax_version
    if version not in AFFECTED_JAX_VERSIONS:
        return "not_affected"
    current = env.get("XLA_FLAGS", "")
    if _FLAG_NAME in current:
        _log.debug("XLA CPU YNN fusion flag already set by the user; leaving XLA_FLAGS alone")
        return "user_set"
    initialised = _backend_initialised() if backend_initialised is None else backend_initialised
    if initialised:
        warnings.warn(
            f"tengri (#3350): jax {version} XLA CPU miscompiles jit(vmap) over a "
            "constant SSP flux, and the workaround flag was not applied because the "
            "XLA backend is already initialised. Import tengri before any jax "
            f"computation, or set XLA_FLAGS to include {YNN_FUSION_FLAG!r} yourself.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "too_late"
    env["XLA_FLAGS"] = f"{current} {YNN_FUSION_FLAG}".strip()
    _log.debug("applied XLA CPU YNN fusion workaround (#3350) for jax %s", version)
    return "applied"
