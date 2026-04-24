"""ASCII-art logo for tengri — three sizes of the hex+spiral mark.

The three art constants are stippled renderings of the official tengri logo
(hexagon containing a spiral). Sizes:

    LOGO         : default, 21 lines, ~32 cols
    LOGO_MEDIUM  : 26 lines, ~40 cols — opt-in via print_logo(size="medium")
    LOGO_FULL    : 32 lines, ~50 cols — opt-in via print_logo(size="full")
    LOGO_BANNER  : plain "tengri" text — no logo below the 21-line threshold

Design credit: Suchetha Cooray.
"""

from __future__ import annotations

# Default: 21-line stippled hex+spiral. This is the smallest supported size —
# for any narrower context we fall back to the text banner (no fake-logo art).
LOGO = r"""            ▗▖▛▟▙▜▗▖
         ▗▖█▝▝    ▀▝▛▄▖
      ▗▗▚▌▘          ▘▚▛▖▖
   ▗▗▚▌▀  ▗▄▗▞▛▟▚▌▖▗▖   ▀▐▞▗▖
 ▖▌▌▀   ▖█▝▝       ▝▘▟ ▖   ▀▗▝▖
▐▐▘   ▖▙▀  ▖▖▟▞▟▞▟▗▗  ▝▗▘▖   ▝▞▌
▌▌   ▞▛ ▖▌▀ ▗▖▖▖▖▖▖▘▙▗  ▝▝▖   ▐▟
▌▌  ▞▞ ▌▘ ▖▜▝▝  ▝▀▐▖▖▚▚   ▚▘  ▗▜
▌▌  ▟▐▀  ▐▞▗▞▛▀▀▝▞▗▝▞▖▘▚   ▌▖ ▝█
▌▌ ▗▚▘  ▐▞ ▌▘▖▘  ▀▐ ▐▗▝▐   ▞▖ ▐▟
▌▌ ▗▜   ▞▖▝▞▝▜    ▚▌▞▞▝▄   ▀▖ ▐▐
▌▌ ▗▙   ▞▖▐▗▝▚▘▖▄▄▀▗▚▘▞▖  ▖▜  ▝█
▌▌  ▞▖  ▝▄ ▙▚▝▐▞▖▄▚▀▗▚▛ ▗▗▘▙  ▐▟
▌▌  ▝▞▖  ▚▚ ▘▙▖▄▖▄▗▚▌▘▗▗▞ ▟   ▐▟
▌▘   ▝▐▗  ▝▝▌▖▝▝▝▘▘▄▗▞▝ ▗▐▝   ▗▀
▝▜     ▘▞▖   ▀▘▀▝▝▝▝  ▗▄▀▘   ▗▜
 ▝▜▞▖▖   ▀▚▌▖▄     ▄▄▐▞   ▗▗▜▞▘
    ▀▐▞▄▖   ▝▘▘▘▀▘▀▝   ▗▖▛▞▝
       ▘▚▜▗         ▗▄▜▞▝
          ▀▜▙▄▖  ▗▖▛▙▀
             ▘▜▟█▙▀▘            """


# Medium: 26-line rendering.
LOGO_MEDIUM = r"""                ▖▞▜▞▙▜▄▗
             ▖▞▟▞▀▘  ▝▝▚▜▚▗
          ▖▞▟▞▀          ▘▜▞▌▄
       ▖▞▟▞▀                ▀▗▌▌▄
    ▖▞▞▞▀    ▗▗▞▟▟▄▙▞▟▞▌▖▄     ▘▗▘▘▖
 ▗▐▚▌▀    ▗▗▙▀▘         ▀▘▄▘▄    ▝▘▟▐▗
▗▚▛▘    ▗▚▛▘   ▖▖▗  ▖▗▗▗   ▝▚▞▗     ▝▚▜▖
▞▐     ▟▄▘  ▖▟▝▝▀    ▘▀▗▌▌▖   ▘▙▖     ▚▘
▌▜    ▐▞ ▗▞▝▘ ▖▖▙▜▞▛▟▞▌▄ ▘▟▗    ▞▌    ▜▜
▌▚   ▝▛▗▐▝  ▗▚▙▀▘▗▗▖▖ ▀▗▌▄▝▐▗   ▝▞▌   ▙▜
▌▜   ▙▚▝▘  ▟▐▘▗▐▐▞▘▀▐▚▚ ▝▖▌ ▌▖   ▝▞▖  ▞█
▌▚   ▙▚   ▗▚▘▗▚▌▚▖▀▝▘ ▘▜ ▀▞ ▐▐    ▞▖  ▙▜
▌▜  ▌▌▘   ▚▀ ▌▌ ▌▌    ▐▝▌▝▞ ▗▚    ▀▐  ▞█
▌▙  ▚▛    ▌▌ ▞▖▐▐▗    ▖▛ ▞▟ ▛▞   ▐▗▘  ▛▟
▌▞  ▐▀▖   ▚▚ ▌▛ ▚▚▀▗▄▘▀▗▐▞▘▞▟    ▌▐   ▙▜
▌▜   ▜▄   ▝▞▖▝▞▄ ▘▙▚▚▛▟▞▘▗▞▟   ▐▞▗▚▘  ▙▜
▛▞   ▝▖▙   ▚▞▖▝▐▚▌▄▖  ▖▖▞▙▀  ▖▛▘ ▙▘   ▙▜
▌▙    ▝▄▚    ▙▚▗▝▝▝▟▟▙▀▀▘ ▗▗▚▀ ▗▜▞    ▙▜
▚▚▖     ▚▚▄   ▘▚▀▌▖▖▖▖▖▗▐▞▘▘  ▄▚▘    ▗▞▛
▝▟▞▄     ▝▗▞▖▖    ▀▀▝▀▝     ▖█▞▘    ▄▚▜
  ▝▙▜▄▗     ▀▐▚▖▄▄     ▗▄▖▛█▝    ▄▐▚▌▘
     ▘█▞▄▖     ▀▘▘▘▙▄▙▝▘▘▀    ▄▝▌▟▝▘
       ▝▘▜▜▄▖              ▗▝▙▞▝▘
           ▘▜▜▄▖        ▗▄▜▙▀
             ▝▝▟▜▄▄▖▄▖▄▜▙▀▘
                ▝▚▙█▙█▛▘                """


# Full: 32-line rendering.
LOGO_FULL = r"""                    ▗▖▞▛▜▀▛▄▄
                 ▗▖▛▞▟▘▀▀▀▀▝▟▀▙▄
              ▗▖▜▖▙▀▘        ▝▘▙▜▚▄
           ▗▖▜▗▛▞▘              ▀▘▙▜▚▄
        ▗▖▟▗▙▀▘                    ▀▚▞▌▛▄
      ▄▐▖▙▌▀     ▗▖▖▛▞▛▙▜▟▞▞▚▗▄▖      ▀▐▖▌▌▄
   ▄▐▜▐▞▀     ▄▝▜▄▀▀▝      ▝▀▘▗▄▀▚▗      ▀▐▖▀▖▖
 ▗▚▙▙▀      ▖▙▚▀▘               ▝▘▙▀▗       ▀▐▞▚▖
▗▀▙▘      ▖▌▛     ▗ ▖▙▚▌▙▚▚▚▗▗▖    ▀▚▞▄       ▝▞▟
▝▞▌      ▄▙▀  ▗▖▌▙▘▀▀      ▀▘▚▞▞▖    ▝▄▀▖      ▐▞▌
▌▞▖    ▗▚▙▘ ▗▐▞▝▘  ▖▄▚▌▌▌▛▞▚▗▖ ▀▞▀▖    ▚▞▄     ▗▜▟
▌▞▌    ▗▛▘▗▐▞▘  ▗▐▞▛▞▘▀▘▀▀▝▚▙▐▚ ▝▚▚▚    ▝▞▄    ▐▐▟
▌▖▌   ▐▐▘▄▐    ▞▙▜▝ ▗▗▐▜▐▖▖▖ ▘▌▛▖ ▚▚▖    ▝▄▘   ▝▟▟
▌▞▖   ▐▘▟▞    ▄▚▛ ▄▜▞▛▝▝▘▀▟▐▗▖▝▞▞  ▄▐     ▚▞▖  ▐▗█
▌▞▖   ▛▐▘    ▄▚▛ ▐▐▞▘▖▟▀▝▘▘▞▞▞ ▐▝▙ ▗▚▘    ▝▖▌  ▝▞▟
▌▞▖  ▖▙▜    ▗▞▟ ▝▞▞ ▗▜     ▝▝▞▌ ▚▞ ▗▚▘    ▐▖▄  ▐▐▜
▌▚▘  ▙▐▘    ▗▚▘ ▚▀▌▝▌▌      ▖▌▘ ▌▟ ▗▚▘    ▖▚▗  ▗▚█
▌▌▘  ▝▞▌    ▗▚▘ ▌▛▖▐▐▐▗    ▖▟▞ ▞▞▘ ▌▛    ▗▀▝   ▐▐▟
▙▐   ▐▜▄    ▝▞▌ ▚▚▚ ▚▚▚▞▐▟▞▝▘▗▜▐▀ ▞▞▘   ▗▐ ▛▌  ▗▚█
▌▞▘   ▚▚▖    ▚▀▖ ▌▌▌ ▘▙▞▞▖▖▛▞▙▀▘ ▙▞▀   ▞▞▘▐▄   ▐▐▟
▌▌▌   ▝▙▜     ▛▞▖ ▜▐▀▖ ▝▀▝▀▝▘ ▖▄▙▌▘  ▗▚▀ ▝▚▘   ▝▙▜
▌▌▘    ▝▞▄▖   ▝▐▐▄ ▝▚▜▚▌▚▗ ▞▙▙▚▌▘  ▖▞▞▘ ▗▜▞    ▐▞█
▚▚▌     ▝▙▚▖    ▘▄▀▖▖ ▀▝▘▀▀▝▝   ▖▄▚▛▘  ▄▚▌     ▐▐▟
▚▚▚       ▚▌▙▖    ▀▐▐▚▚▗▄▗▗▖▄▞▛▟▞▝   ▗▐▞▘      ▞▌▛
 █▐▄       ▝▗▞▘▄     ▝▘▘▘▘▘▀▝▝     ▗▐▄▛▘      ▞▞▟▘
  ▜▞▛▌▖      ▝▀▄▜▗▖             ▗▄▜▚▌▘     ▄▐▜▐▀
   ▝▘▜▞▛▄▖      ▝▘█▞▛▌▗▄▄▄▄▄▄ ▛▛▟▖▀     ▗▞▌▌▌▀
      ▝▀▐▞▛▄▖       ▝▀▀▝▝▝▝▝▝▀▀      ▗▖▛▞▞▝▘
         ▝▘▙▚▛▄▖                  ▗▖▛▞▟▝▘
            ▝▘▙▚▜▗▖            ▗▖▛▟▟▝▘
               ▀▀▟▞▟▗▖      ▗▖▛▟▟▀▝
                  ▀▀▙▙▜▄▄▗▟▜▞█▞▘
                     ▀▜█▟█▟▛▀                     """


# Compact banner — plain text. Logo art is never drawn below LOGO's size.
LOGO_BANNER = "tengri"


def _resolve(size: str) -> str:
    """Map ``size`` to one of the logo constants.

    Recognised values:
        "default", "small"     → LOGO (21 lines)
        "medium"               → LOGO_MEDIUM (26 lines)
        "full", "large"        → LOGO_FULL (32 lines)
        "compact", "banner"    → LOGO_BANNER (plain text, no art)
    """
    if size in ("default", "small", None):
        return LOGO
    if size == "medium":
        return LOGO_MEDIUM
    if size in ("full", "large"):
        return LOGO_FULL
    if size in ("compact", "banner"):
        return LOGO_BANNER
    raise ValueError(
        f"Unknown logo size '{size}'. "
        "Use 'default' (small), 'medium', 'full', or 'compact'."
    )


def print_logo(size: str = "default", *, compact: bool | None = None) -> None:
    """Print the tengri logo.

    Parameters
    ----------
    size : {"default", "medium", "full", "compact"}
        Which rendering to print. "default" is the 21-line stippled version,
        the smallest rendered size. "compact" prints plain text ("tengri")
        because rendering the logo any smaller would misrepresent the mark.
    compact : bool or None
        Deprecated alias for ``size="compact"``. Kept for backward compatibility.

    Notes
    -----
    Respects the ``TENGRI_NO_LOGO`` environment variable. If set to anything
    truthy, this function writes nothing.
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
    """Return the logo string (no trailing newline).

    Parameters
    ----------
    size : {"default", "medium", "full", "compact"}
    compact : bool or None
        Deprecated alias for ``size="compact"``.

    Returns
    -------
    str
        The requested logo (or plain text for "compact"), or ``""`` if
        ``TENGRI_NO_LOGO`` is set.
    """
    import os

    if os.environ.get("TENGRI_NO_LOGO"):
        return ""
    if compact is not None:
        size = "compact" if compact else "default"
    return _resolve(size)
