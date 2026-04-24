"""ASCII-art logo for tengri (shown in doctor() header and CLI banner).

Three sizes:
    LOGO         : default ~10-line compact rendition (used in doctor(), print_logo())
    LOGO_FULL    : full ~40-line sunburst; opt-in via print_logo(size="full")
    LOGO_BANNER  : one-liner for CLI --version

All three are plain Unicode block characters; no ANSI colour codes.
Design credit: Suchetha Cooray.
"""

from __future__ import annotations

# Default: hexagon with nested spiral — matches the official tengri mark.
# 7 lines, ~18 cols. Reads well in monospaced terminals.
LOGO = r"""    ╱▔▔▔▔▔▔▔▔▔╲
   ╱  ╭─────╮  ╲
  │  │ ╭───╮ │  │
  │  │ │ @ │ │  │
  │  │ ╰───╯ │  │
   ╲  ╰─────╯  ╱
    ╲▁▁▁▁▁▁▁▁▁╱"""


# Medium size — stronger spiral impression, 9 lines.
LOGO_MEDIUM = r"""     ╱▔▔▔▔▔▔▔▔▔▔▔╲
    ╱ ╭─────────╮ ╲
   ╱  │ ╭─────╮ │  ╲
  │   │ │ ╭─╮ │ │   │
  │   │ │ │@│ │ │   │
  │   │ │ ╰─╯ │ │   │
   ╲  │ ╰─────╯ │  ╱
    ╲ ╰─────────╯ ╱
     ╲▁▁▁▁▁▁▁▁▁▁▁╱"""


# Full detailed sunburst (original design) — opt-in via print_logo(size="full").
LOGO_FULL = r"""                         ▗▄▞▛▛▜▜▐▗▖
                      ▄▐▀▙▚▙▛▝▘▜▙▛▟▜▄▖
                   ▄▐▚▚▙▛▘▘       ▀▐▞▟▜▄▖
                ▗▞▜▐▐▞▀             ▝▝▙▚▜▜▄▖
             ▗▄▀▌▙▙▀                   ▝▀▟▖▛▜▄▖
          ▗▖▜▐▗▛▀▘                        ▝▀▙▄▚▜▗▖
       ▗▖▜▗▚▙▀▘       ▄▖▖▛▞▌▙▜▐▚▀▄▗▖▖        ▝▘▙▚▗▚▗▖
     ▄▜▖▌▙▀▘      ▖▖▛▙▚▙▀▘▀▀▝▘▀▘▀▝▚▟▞▜▝▄        ▝▚▙▘▞▚▄
  ▗▞▛▞▟▞▀      ▗▞▌▟▞▀                ▀▀▄▛▞▖▖       ▝▚▄▖▜▗▖
 ▗▌▙▙▀       ▗▐▄▙▀         ▗▗▗▗         ▝▐▟▝▄         ▀▄▚▐▖
▗▚▜▞        ▞▞▙▀     ▖▖▌▌▙▞▄▚▞▄▜▞▞▞▖▖      ▀▄▀▄         ▚▚▜
▐▐▐       ▗▚▚▛    ▖▙▚▙▀▝▀       ▝▀▝▟▐▞▚      ▀▄▀▖        ▌▛▌
▚▗▛      ▗▚█▘  ▗▗▜▞▝   ▗▗▗▞▙▜▐▚▜▗▗▖ ▝▐▚▚▚     ▝▚▐▚       ▚▜▟
▚▗▛      ▌▙▘ ▗▐▞▀   ▗▗▚▌▛▟▟▝▝▘▀▙▚▙▐▐▗ ▝▚▚▀▖     ▚▚▚      ▀▙▘
▚▗▙     ▞▜▘ ▞▞▘    ▟▞▛▟▝▘        ▝▚▙▚▚▖ ▚▚▚      ▀▞▄     ▜▐▀
▚▘▄     ▙▛▗▐▞▘   ▗▙▚▟▀ ▗▗▚▌▙▙▚▌▙▚▗ ▝▐▞▞▖ ▚▚▀      ▙▘▌    ▐▚▛
▌▞▌    ▚▌▗▚▛    ▗▌▙▀  ▞▞▙▛▝▘▘▀▘▚▚▚▚▖ ▐▐▐  ▚▀▘     ▗▜▝    ▀▙▘
▌▖▌   ▗▐▖▞▌     ▙▚▛ ▗▜▐▘▘▗▗▛▀▝▘▘▞▘▌▙ ▝▞▞▖ ▐▝▙      ▚▝▘   ▜▐▀
▌▞▞    ▛▞▀     ▝▞▟  ▌▙▀ ▖▌▌      ▝▐▗▘ ▚▚▜ ▗▀▄      ▌▝▘   ▐▙▜
▌▞▞   ▖▌▛▘     ▜▐▖ ▐▐▐▘▗▚▜        ▚▚▘ ▌▌▙ ▝▌▌     ▗▐▐▗   ▚▌█
▌▞▞   ▐▐▟      ▚▜▖ ▞▞▙ ▐▐▗▖       ▌▙▘▗▚▚▘ ▐▐▞     ▖▛▗    ▟▞█
▌▞▖   ▐▚▚      ▌▙▘ ▐▞▞ ▝▞▞▞▖    ▖▌▙▘ ▌▌▛ ▗▜▐     ▗▐ ▌▖   ▐▟▟
▌▞▞   ▝▞▛▖     ▐▝▌  ▌▛▌ ▜▐▝▞▀▐▟▟▝▝ ▖▙▚▛ ▗▚▚▛    ▗▚▘▐▖▘   ▜▄▜
▌▚▚    ▚▜▄     ▝▚▜  ▜▐▐  ▘▛▞▞▄▗▗▗▜▞▟▞▘ ▖▟▞▛    ▖▌▘ ▙▐    ▚▙▜
▌▌▌    ▝▙▐▖     ▚▚▜▖ ▚▌▛▄▖ ▀▀▐▙▜▞▘▀▘ ▗▞▌▙▀   ▗▐▟▘ ▟▐▘    ▜▞█
▚▐▞     ▝▟▞▖     ▚▚▚▖ ▝▞▄▚▚▖▖    ▄▗▄▀▙▚▀    ▄▚▌  ▞▌▌     ▜▟▟
▚▚▚       ▛▞▄     ▘▌▌▙  ▝▀▟▞▟▚▜▞▛▄▛▟▀▝   ▖▞▟▞▘  ▄▚▛      ▜▄▘
▚▚▚       ▝▚▌▙     ▝▐▄▀▌▄   ▝▀▘▀▀▘   ▗▄▐▚▜▝▘   ▞▙▛       ▛▞▛
▐▚▀▖        ▀▟▐▄      ▀▟▐▐▚▚▄▗▖▖▖▖▞▞▌▙▚▀▘    ▗▜▞▘       ▗▜▞▌
 ▙▜▜▖        ▝▚▄▀▄       ▀▝▘▚▙▚▛▟▝▝▀▘      ▄▞▙▛▘       ▗▜▐▞
  ▜▚▙▚▄        ▝▀▟▐▚▖                   ▗▄▜▄▌▘       ▄▞▌▌▛
   ▀▐▙▚▜▄▖        ▀▚▜▞▛▗▖▖          ▗▄▄▜▚▟▝▘      ▗▄▜▗▚▌▀
      ▀▙▌▛▜▗▖        ▀▝▙▜▄▜▜▀▙▀▚▜▜▀▛▄▙▌▀▘      ▗▄▜▚▚▌▀▘
        ▝▀▚▛▞▛▖▖          ▝▀▀▘▀▀▀▀▀▘        ▗▖▛▌▌▙▀
           ▝▀▟▞▌▛▄                       ▗▗▜▐▐▟▝▘
              ▝▀▟▞▛▙▄                  ▄▞▌▛▟▞▘
                 ▝▚▙▚▛▙▄            ▄▞▛▟▟▀▀
                    ▀▚▙▚▛▙▄▖    ▗▄▐▜▟▟▀▘
                       ▀▜▟▟▟▜▜▜▚█▞█▌▘
                          ▝█████▙▀"""


# Compact one-line banner for routine prints (e.g. CLI --version, logs).
# Uses the U+2B21 WHITE HEXAGON glyph to echo the official mark.
LOGO_BANNER = "⬡ tengri ⬡"


def _resolve(size: str) -> str:
    """Dispatch ``size`` to one of the logo constants."""
    if size in ("default", "small", None):
        return LOGO
    if size == "medium":
        return LOGO_MEDIUM
    if size == "full":
        return LOGO_FULL
    if size in ("banner", "compact"):
        return LOGO_BANNER
    raise ValueError(
        f"Unknown logo size '{size}'. "
        "Use 'default', 'medium', 'full', or 'compact'."
    )


def print_logo(size: str = "default", *, compact: bool | None = None) -> None:
    """Print the tengri logo.

    Parameters
    ----------
    size : {"default", "full", "compact"}
        Which size to print. "default" is the small ~10-line version.
        "full" is the detailed sunburst. "compact" is the one-line banner.
    compact : bool or None
        Deprecated alias for ``size="compact"``. Kept for backward compatibility.

    Notes
    -----
    Respects the ``TENGRI_NO_LOGO`` environment variable: if set to anything
    truthy, the function prints nothing.
    """
    import os
    import sys

    if os.environ.get("TENGRI_NO_LOGO"):
        return
    if compact is not None:
        size = "compact" if compact else "default"
    sys.stdout.write(_resolve(size) + "\n")
    sys.stdout.flush()


def logo_str(size: str = "default", *, compact: bool | None = None) -> str:
    """Return the logo as a string (no trailing newline).

    Parameters
    ----------
    size : {"default", "full", "compact"}
    compact : bool or None
        Deprecated alias for ``size="compact"``.

    Returns
    -------
    str
        The requested logo, or ``""`` if ``TENGRI_NO_LOGO`` is set.
    """
    import os

    if os.environ.get("TENGRI_NO_LOGO"):
        return ""
    if compact is not None:
        size = "compact" if compact else "default"
    return _resolve(size)
