"""Forward model components: SFH, dust, SPS, observation."""

# Import submodules to make them available as components.agn, components.dust, etc.
from tengri.components import agn, dust, igm, nebular, radio, sfh, sps, xray

__all__ = [
    "agn",
    "dust",
    "igm",
    "nebular",
    "radio",
    "sfh",
    "sps",
    "xray",
]
