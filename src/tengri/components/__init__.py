"""Forward model components: stellar, dust, nebular, AGN, IGM, radio, X-ray.

The ``sfh`` and ``sps`` subpackages were folded into ``stellar`` in
Phase II-2.1. The old dotted names remain importable as deprecation
shims (firing :class:`DeprecationWarning` on first use); they are not
preloaded here so that ``import tengri.components`` is warning-clean.
"""

# Import canonical submodules eagerly so users can write
# ``tengri.components.dust`` etc. without an explicit submodule import.
from tengri.components import agn, dust, igm, nebular, radio, stellar, xray

__all__ = [
    "agn",
    "dust",
    "igm",
    "nebular",
    "radio",
    "stellar",
    "xray",
]
