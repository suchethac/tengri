"""Fused SED kernels.

Currently houses the full kernel set in ``assembly.py`` (3174 lines, not yet
split). Phase 6 will split this by fusion strategy into ``exact.py``,
``compositional.py``, ``hybrid.py``, ``traceable.py``, and ``dispatch.py``.
Callers should import from ``tengri.forward.kernels.assembly`` until then.
"""

from tengri.forward.kernels.assembly import *  # noqa: F401, F403
