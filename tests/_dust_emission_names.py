# SPDX-License-Identifier: BSD-3-Clause
"""Public ``dust_emission.type`` spelling for each ``sed_dust_ir`` registry component.

Ruling R82 (commit 8963b591f) correctly stopped
:func:`tengri.parameters.groups._valid_dust_emission_types` from accepting
``draine2021_pah_ir`` — the internal ``SEDModelComponent`` registry name for
the Draine+2021 PAHspec component, reachable in the public ``dust_emission``
grammar only through its alias ``draine2021_pah``. Three contract tests built
their enumeration directly from the registry and so fed that internal
spelling straight to the grammar. Their contracts are legitimate ("every
registered ``sed_dust_ir`` component is reachable and behaves"); only the
spelling they enumerated by was wrong.

This module is the one place that derives the translation, so all three
tests enumerate through it rather than each re-deriving (and re-drifting)
their own copy. Derived from the same two facts
:func:`~tengri.parameters.groups._valid_dust_emission_types` itself is
derived from — the live ``_REGISTRY`` and
:data:`tengri.forward.component_factory._EMISSION_TYPE_ALIASES` — so it
cannot drift from what the grammar actually accepts.
"""

from __future__ import annotations


def dust_ir_registry_to_public() -> dict[str, str]:
    """Map each ``sed_dust_ir``-publishing ``_REGISTRY`` key to its public spelling.

    Returns
    -------
    dict[str, str]
        Registry key -> the ``dust_emission.type`` value that reaches it: the
        alias key, for a registry name reachable ONLY through its alias (an
        :data:`~tengri.forward.component_factory._EMISSION_TYPE_ALIASES` value
        ending in ``"_ir"``, e.g. ``draine2021_pah_ir`` -> ``draine2021_pah``
        — mirroring the ``leaked_ir_targets`` exclusion in
        :func:`~tengri.parameters.groups._valid_dust_emission_types`); the
        registry name itself for everything else, including components that
        keep their un-suffixed registry name directly valid alongside a
        convenience alias (``draine_li2007``/``draine_li2014``/
        ``modified_blackbody`` beside ``dl07``/``dl14``/``mbb``).
    """
    from tengri.components.sed_model_component import _REGISTRY
    from tengri.forward.component_factory import _EMISSION_TYPE_ALIASES

    dust_ir_components = {
        name
        for name, cls in _REGISTRY.items()
        # getattr default skips non-emission registry entries (dust-attenuation
        # screens registered for dispatch in #844 lack _outputs_tuple).
        if "sed_dust_ir" in {o.name for o in getattr(cls, "_outputs_tuple", ())}
    }
    public_of_leaked_target = {
        target: alias for alias, target in _EMISSION_TYPE_ALIASES.items() if target.endswith("_ir")
    }
    return {name: public_of_leaked_target.get(name, name) for name in dust_ir_components}


def dust_emission_public_names() -> frozenset[str]:
    """The set of public ``dust_emission.type`` spellings (the map's values)."""
    return frozenset(dust_ir_registry_to_public().values())
