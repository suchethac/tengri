"""Free-parameter declarations owned by the IGM component.

Skeleton introduced in PR1 of the parameter-registry consolidation.
Empty until PR2 begins moving priors out of
``tengri.parameters._param_defs``.
"""

from __future__ import annotations

from tengri.core.component import ParamDeclaration

PARAMS: tuple[ParamDeclaration, ...] = ()

__all__ = ["PARAMS"]
