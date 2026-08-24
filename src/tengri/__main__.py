# SPDX-License-Identifier: BSD-3-Clause
"""Command-line entry point: ``python -m tengri <subcommand>``.

Available subcommands::

    python -m tengri summary           # one-line counts of every menu
    python -m tengri doctor            # env / install / SSP health check
    python -m tengri help              # the curated cheatsheet
    python -m tengri help <topic>      # topical cheatsheet
    python -m tengri search <query>    # cross-menu fuzzy search
    python -m tengri describe <name>   # universal lookup

Designed so a brand-new user can run ``python -m tengri summary`` right
after install to see what they have, without entering a REPL.
"""

from __future__ import annotations

import sys


def _print_help() -> None:
    print(
        "usage: python -m tengri <subcommand> [args]\n"
        "\n"
        "subcommands:\n"
        "  summary                one-line counts of every menu\n"
        "  doctor                 env / install / SSP health check\n"
        "  help [topic]           cheatsheet (full or topical)\n"
        "  search <query>         cross-menu fuzzy search\n"
        "  describe <name>        full metadata for one model / method\n"
        "\n"
        "topics for help: agn, dust, sfh, nebular, components, "
        "inference, filters\n"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return 0

    cmd, rest = args[0], args[1:]

    # Lazy import so e.g. ``--help`` doesn't pay JAX startup time.
    import tengri

    if cmd == "summary":
        tengri.summary()
        return 0

    if cmd == "doctor":
        tengri.doctor()
        return 0

    if cmd == "help":
        if rest:
            tengri.help(rest[0])
        else:
            tengri.help()
        return 0

    if cmd == "search":
        if not rest:
            print("error: search requires a query string", file=sys.stderr)
            return 2
        print(tengri.search(" ".join(rest)))
        return 0

    if cmd == "describe":
        if not rest:
            print("error: describe requires a name", file=sys.stderr)
            return 2
        try:
            print(tengri.describe(rest[0]))
            return 0
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    print(f"error: unknown subcommand {cmd!r}", file=sys.stderr)
    _print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
