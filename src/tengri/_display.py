# SPDX-License-Identifier: BSD-3-Clause
"""Single sink for user-asked-for stdout output (citations, help, doctor reports).

Library consumers can silence all of this output by setting the
``TENGRI_QUIET=1`` environment variable, or by monkey-patching
``tengri._display._display`` to a custom handler.

Example:
    Silence all output::

        import os

        os.environ["TENGRI_QUIET"] = "1"
        import tengri

        tengri.help()  # prints nothing
"""

from __future__ import annotations

import os


def _display(text: str) -> None:
    """Print text to stdout unless TENGRI_QUIET=1 is set.

    Parameters
    ----------
    text : str
        Text to display.
    """
    if os.environ.get("TENGRI_QUIET") == "1":
        return
    print(text)
