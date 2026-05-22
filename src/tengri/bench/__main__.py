# SPDX-License-Identifier: BSD-3-Clause
"""``python -m tengri.bench`` entry point."""

from __future__ import annotations

import sys

from tengri.bench import run


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
