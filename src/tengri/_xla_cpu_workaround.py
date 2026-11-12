# SPDX-License-Identifier: BSD-3-Clause
"""Workaround for an XLA CPU miscompile under ``jax.vmap(jax.jit(...))`` (#3350).

With an SSP grid whose flux is constant along wavelength, a batched ``einsum``
over a closure-captured constant is fused by XLA CPU into a YNNPACK kernel
(``__ynn_fusion``) that returns wrong values inside ``jit(vmap(...))``:
``predict_photometry`` came back as exact zeros. The same arithmetic is correct
when the array is an argument, when it is perturbed, and in eager and
single-draw ``jit``. Disabling that fusion type with
``--xla_cpu_experimental_ynn_fusion_type=`` restores the correct result.

Gate. The XLA flag is registered by ``jaxlib`` (the XLA runtime), so the gate
reads the installed ``jaxlib`` version, not ``jax``. The flag is applied when:

- the platform is CPU (``JAX_PLATFORMS`` unset, or naming ``cpu``); and
- ``jaxlib >= FIRST_JAXLIB_WITH_FLAG``, the first release that accepts the flag.
  An unknown XLA flag aborts the process, so the lower bound is an
  availability requirement, not only an evidence statement. There is no upper
  bound: the newest measured release is still affected, so every later jaxlib
  is assumed affected until shown fixed.

The lower bound was measured (macOS 25.6.0 arm64, CPU backend, the
pure-JAX repro in ``upstream_issue.md``): see :data:`FIRST_JAXLIB_WITH_FLAG`.

A user who has already named the flag in ``XLA_FLAGS`` keeps their own setting.
The flag is appended to ``XLA_FLAGS``, never overwriting it, and only while the
XLA backend is still uninitialised, since the backend reads ``XLA_FLAGS`` once.
If the backend is already up the flag would be silently ignored, so a
:class:`RuntimeWarning` is raised instead.

Remove this module when the upstream miscompile is fixed in jaxlib and the
fixed release is added as an exclusion below :data:`FIRST_JAXLIB_WITH_FLAG`.
The upstream repro, versions and HLO evidence are in
``/Users/suchethacooray/.claude/jobs/cbd672d6/tmp/l3350/upstream_issue.md``
(not yet filed).
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import sys
import warnings
from collections.abc import MutableMapping
from typing import Literal

FIRST_JAXLIB_WITH_FLAG: tuple[int, ...] = (0, 8, 1)
"""Earliest jaxlib that accepts ``--xla_cpu_experimental_ynn_fusion_type``.

Evidence (pure-JAX probe, CPU, macOS arm64, one run per version):

- ``0.8.0``, ``0.7.2``, ``0.5.3`` abort XLA with ``Unknown flag`` when the flag is
  set (exit 134 or 1). ``0.7.1`` was also rejected. Those versions are never gated.
- ``0.8.1``, ``0.8.2``, ``0.8.3``, ``0.9.2`` accept the flag and return the correct
  value with or without it. No miscompile was observed there.
- ``0.10.2``, ``0.11.0``, ``0.11.1``, ``0.12.0``, ``0.12.0.1`` accept the flag and
  return the wrong value (1.875e44 against 1.263e45) without it, and the correct
  value with it. This is the miscompile window.

So the gate starts at the first accepting release (``0.8.1``), which is the
requirement that an unknown flag never reaches XLA. Releases ``0.8.1`` to ``0.9.x``
accept the flag but showed no miscompile, so they pay the flag's cost without a
fix. ``0.9.0``, ``0.9.0.1``, ``0.9.1``, ``0.10.0`` and ``0.10.1`` were not measured.
"""

YNN_FUSION_FLAG = "--xla_cpu_experimental_ynn_fusion_type="
"""Disables the YNNPACK fusion type; the empty value is the one measured to work."""

_FLAG_NAME = "xla_cpu_experimental_ynn_fusion_type"

Status = Literal["not_affected", "not_cpu", "user_set", "applied", "too_late"]

_log = logging.getLogger(__name__)


def _installed_jaxlib_version() -> str | None:
    """Version of the installed jaxlib distribution, or ``None`` if absent."""
    try:
        return importlib.metadata.version("jaxlib")
    except importlib.metadata.PackageNotFoundError:
        return None


def version_tuple(version: str) -> tuple[int, ...]:
    """Release digits of a version string, e.g. ``"0.12.0.1"`` -> ``(0, 12, 0, 1)``.

    Pre-release suffixes are dropped at the first non-digit component.
    """
    parts: list[int] = []
    for piece in version.split("."):
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
        if len(digits) != len(piece):
            break
    return tuple(parts)


def _is_cpu_platform(environ: MutableMapping[str, str]) -> bool:
    """True when ``JAX_PLATFORMS`` is unset or names ``cpu``."""
    platforms = environ.get("JAX_PLATFORMS", "").strip()
    if not platforms:
        return True
    return "cpu" in {name.strip() for name in platforms.split(",")}


def _backend_initialised() -> bool:
    """True when jax is imported and its XLA backend already exists.

    Reads ``sys.modules`` only, so tengri never imports jax to ask.
    """
    bridge = sys.modules.get("jax._src.xla_bridge")
    if bridge is None:
        return False
    return bool(bridge.backends_are_initialized())


def apply_ynn_fusion_workaround(
    environ: MutableMapping[str, str] | None = None,
    *,
    jaxlib_version: str | None = None,
    backend_initialised: bool | None = None,
) -> Status:
    """Append the YNN fusion flag to ``XLA_FLAGS`` when the gate allows it.

    Parameters
    ----------
    environ : MutableMapping, optional
        Mutable environment. Defaults to ``os.environ``. Only ``XLA_FLAGS`` is
        written; ``JAX_PLATFORMS`` is read.
    jaxlib_version : str, optional
        Installed jaxlib version. Defaults to the distribution metadata.
    backend_initialised : bool, optional
        Whether the XLA backend already exists. Defaults to a ``sys.modules``
        check.

    Returns
    -------
    str
        ``"not_cpu"`` (a non-CPU platform is requested), ``"not_affected"``
        (jaxlib below :data:`FIRST_JAXLIB_WITH_FLAG`, or not installed),
        ``"user_set"`` (``XLA_FLAGS`` already names the flag), ``"applied"``, or
        ``"too_late"`` (backend up, flag not applied; a warning was raised).
    """
    env = os.environ if environ is None else environ
    if not _is_cpu_platform(env):
        return "not_cpu"
    version = _installed_jaxlib_version() if jaxlib_version is None else jaxlib_version
    if version is None or version_tuple(version) < FIRST_JAXLIB_WITH_FLAG:
        return "not_affected"
    current = env.get("XLA_FLAGS", "")
    if _FLAG_NAME in current:
        _log.debug("XLA CPU YNN fusion flag already set by the user; leaving XLA_FLAGS alone")
        return "user_set"
    initialised = _backend_initialised() if backend_initialised is None else backend_initialised
    if initialised:
        warnings.warn(
            f"tengri (#3350): jaxlib {version} XLA CPU miscompiles jit(vmap) over a "
            "constant SSP flux, and the workaround flag was not applied because the "
            "XLA backend is already initialised. Import tengri before any jax "
            f"computation, or set XLA_FLAGS to include {YNN_FUSION_FLAG!r} yourself.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "too_late"
    env["XLA_FLAGS"] = f"{current} {YNN_FUSION_FLAG}".strip()
    _log.debug("applied XLA CPU YNN fusion workaround (#3350) for jaxlib %s", version)
    return "applied"
