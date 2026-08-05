# SPDX-License-Identifier: BSD-3-Clause
"""Spatial physics blocks: surface-brightness profiles.

See :doc:`docs/dev/archive/forward-model-architecture.md` §3.2 for the
astronomer-facing authoring guide.
"""

from tengri.components.spatial.exponential import Exponential
from tengri.components.spatial.flat_slab import FlatSlab
from tengri.components.spatial.sersic import Sersic

__all__ = ["Exponential", "FlatSlab", "Sersic"]
