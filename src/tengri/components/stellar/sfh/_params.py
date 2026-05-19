"""Free-parameter declarations owned by the SFH sub-component.

Skeleton introduced in PR1 of the parameter-registry consolidation.
Empty until PR2 begins moving priors out of
``tengri.parameters._param_defs``. SFH parameters are currently produced
by ``tengri.components.stellar.sfh.registry.resolve_sfh`` and surfaced
through ``tengri.parameters._param_defs._build_param_registry``; PR2+
will decide whether to migrate them into this tuple or keep that
registry-driven path.
"""

from __future__ import annotations

from tengri.protocols.component import ParamDeclaration

PARAMS: tuple[ParamDeclaration, ...] = ()

__all__ = ["PARAMS"]
