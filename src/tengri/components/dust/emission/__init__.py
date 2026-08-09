# SPDX-License-Identifier: BSD-3-Clause
"""Dust emission package.

Transitional layout (ADR-0019): the
historical ``emission.py`` module now lives at ``emission/emission.py`` so the
``analytic/`` subpackage can host the SEDModelComponents. This
``__init__`` preserves the full public + used surface of the old flat module,
so every existing ``from tengri.components.dust.emission import X`` keeps
working unchanged (the closures, registries, and template loaders).
"""

# Re-export the entire public surface of the implementation module so callers
# that did `from tengri.components.dust.emission import <name>` are unaffected.
# Import the analytic and template components so their SEDModelComponent
# subclasses register in _REGISTRY at ``import tengri`` time (registration is a
# __init_subclass__ side effect; the component modules import their closures
# lazily, so no import cycle).
from tengri.components.dust.emission import (
    analytic as _analytic,
    templates as _templates,
)
from tengri.components.dust.emission.emission import *  # noqa: F403

# Private module-level symbols that existing call sites and tests import by name.
from tengri.components.dust.emission.emission import (
    _find_data_file,
    _find_dl07_templates,
    _resolved,
)
