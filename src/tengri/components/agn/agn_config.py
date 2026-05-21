"""Re-export of :class:`~tengri.config.settings.AGNConfig`.

The canonical definition lives in :mod:`tengri.config.settings` alongside the
other ``*Config`` dataclasses; this module re-exports it so imports inside
:mod:`tengri.components.agn` do not have to reach across the package.
"""

from tengri.config.settings import AGNConfig

__all__ = ["AGNConfig"]
