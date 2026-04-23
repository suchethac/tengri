"""ASCII-art logo for tengri (shown in doctor() header and CLI banner)."""

from __future__ import annotations

# Unicode ASCII-art logo for tengri.
# Design credit: Suchetha Cooray.
LOGO = r"""                         ▗▄▞▛▛▜▜▐▗▖
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
                          ▝█████▙▀                          """


# Compact one-line banner for routine prints (e.g., CLI --version, logs).
LOGO_BANNER = "  ▗▖▛▟▞▘ tengri ▝▚▙▛▖▖"


def print_logo(compact: bool = False) -> None:
    """Print the tengri logo.

    Parameters
    ----------
    compact : bool
        If True, print the one-line banner. Default is the full logo.

    Notes
    -----
    Respects the TENGRI_NO_LOGO environment variable. If set, suppresses output.
    """
    import os
    import sys

    # Respect TENGRI_NO_LOGO env var for quiet runs.
    if os.environ.get("TENGRI_NO_LOGO"):
        return
    if compact:
        sys.stdout.write(LOGO_BANNER + "\n")
    else:
        sys.stdout.write(LOGO + "\n")
    sys.stdout.flush()


def logo_str(compact: bool = False) -> str:
    """Return the logo as a string (no newline at end).

    Parameters
    ----------
    compact : bool
        If True, return the one-line banner. Default is the full logo.

    Returns
    -------
    str
        The logo string, respecting TENGRI_NO_LOGO environment variable.
    """
    import os

    if os.environ.get("TENGRI_NO_LOGO"):
        return ""
    return LOGO_BANNER if compact else LOGO
