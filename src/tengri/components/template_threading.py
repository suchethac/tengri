# SPDX-License-Identifier: BSD-3-Clause
"""The template-threading seam, shared by both component families.

A component backed by a template library must hand that library to the
compiled graph as a traced **argument**, not read it from module state while
tracing. Read at trace time, the whole library freezes into the HLO as XLA
``Constant`` ops — measured against a 0.05 MB bare-stellar floor: 66.6 MB for
Draine & Li 2014, 39.4 for THEMIS, 29.95 for SKIRTOR, 3.7 for the MAPPINGS V
shock grid (#1649, #1694).

Why this is a mixin rather than a method on one base class
-----------------------------------------------------------
``_REGISTRY`` holds two families. Most components subclass
:class:`~tengri.components.sed_model_component.SEDModelComponent` and write a
``predict(p, sed_in, wave, **inputs)``. Eight implement the bare
:class:`~tengri.protocols.component.SEDComponent` Protocol instead — they own
``apply`` outright because their shape does not fit ``predict`` (AGN, IGM,
nebular, radio, X-ray, the three dust attenuation components). That split is
deliberate and documented in ADR-0009/0011.

Threading has nothing to do with that split. Hanging the seam off
``SEDModelComponent`` alone made the registry heterogeneous in a way that bites
precisely the person it should help: a bare-Protocol author who copies a
sibling's ``accepts_threaded_templates = True`` gets an ``AttributeError`` from
the publisher, because the flag is readable by ``getattr`` but the method
behind it does not exist. Both families inherit this mixin instead, so the
answer to "can my component thread?" is the same everywhere.

Declaring only :data:`~typing.ClassVar` attributes and methods, it composes
with the ``@dataclass(frozen=True)`` declarations the bare-Protocol family uses
— dataclass field collection ignores ``ClassVar`` by construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

__all__ = ["TemplateThreading"]


class TemplateThreading:
    """Opt-in threading of a component's template library through ``jax.jit``.

    A component opts in by setting :attr:`accepts_threaded_templates` and
    accepting ``templates=None`` in whatever method consumes the library.
    ``SEDModel._template_data_for_jit`` then publishes the bundle, and the
    component reads it back with :meth:`threaded_templates`.

    Notes
    -----
    **JIT-compatible**: the methods here run at build time and during tracing,
    but perform no array work themselves — they select which object the caller
    reads.
    """

    #: Opt-in: when True, the component's template bundle is published by
    #: ``SEDModel._template_data_for_jit`` and passed to the consuming method
    #: as ``templates=``.
    #:
    #: Opt-in rather than automatic because consuming signatures are written
    #: per component; handing ``templates=`` to one that has no library at all
    #: (every closed-form law) would be a :class:`TypeError`.
    accepts_threaded_templates: ClassVar[bool] = False

    #: Namespace this component's bundle occupies in the threading dict, which
    #: is keyed ``template_data[namespace][component_name]``. Empty string means
    #: "use :attr:`name`", the right default for a component that owns its
    #: namespace outright. Subsystems with several sibling components share one
    #: namespace instead — dust emission pins ``"dust_ir"`` so all its backends
    #: land together beside the LUTs published for that subsystem.
    template_namespace: ClassVar[str] = ""

    def threaded_templates(self, template_data: Mapping[str, Any] | None) -> Any | None:
        """Return this component's template bundle, preferring the threaded one.

        Parameters
        ----------
        template_data : mapping, optional
            The nested threading dict published by
            ``SEDModel._template_data_for_jit``, keyed
            ``[namespace][component_name]``.

        Returns
        -------
        object or None
            The threaded bundle when present — its arrays are JIT arguments, so
            reading them costs nothing at compile time. Else ``self.data`` (set
            by ``precompute`` via ``load``), else ``None``, in which case the
            component falls back to its own module-level load and bakes.
        """
        if isinstance(template_data, dict):
            namespace = template_data.get(self.template_namespace or self.name)
            if isinstance(namespace, dict):
                found = namespace.get(self.name)
                if found is not None:
                    return found
        return getattr(self, "data", None)

    def templates_for_threading(self) -> Any | None:
        """Eagerly resolve the bundle to publish, for the threading collector.

        Called by ``SEDModel._template_data_for_jit`` *outside* any trace, so
        the load happens once at build time rather than once per compile.

        Returns
        -------
        object or None
            ``self.data`` when ``precompute`` already ran, else the result of a
            direct ``load`` call, else ``None`` — a component whose loader is
            unavailable (a missing optional data file) simply does not thread,
            exactly as before threading existed.
        """
        bundle = getattr(self, "data", None)
        if bundle is not None:
            return bundle
        load = getattr(self, "load", None)
        if load is None:
            # Bare-Protocol components define ``precompute``, not ``load``.
            # Opting in without a loader is not an error: publish nothing and
            # let the component keep reading its own module-level cache.
            return None
        try:
            return load(None)
        except Exception:
            # A component that cannot load here is no worse off than before
            # threading existed: it falls back to its in-predict load.
            return None
