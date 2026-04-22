"""Backward-compatibility shim for AGNConfig.

AGNConfig has been moved to src/tengri/config/settings.py to consolidate
all configuration classes. This module re-exports for backward compatibility
and to avoid circular imports within components/agn/.
"""

from tengri.config.settings import AGNConfig

__all__ = ["AGNConfig"]
