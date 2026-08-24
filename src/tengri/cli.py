# SPDX-License-Identifier: BSD-3-Clause
"""Command-line interface for tengri.

Subcommands:
    tengri doctor          : print environment health check
    tengri cite            : list all registered citations (short form)
    tengri cite KEY        : print a specific citation (short + bibtex)
    tengri cite --bibtex   : list all citations in BibTeX format
    tengri --version       : print tengri version
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from tengri._logo import print_logo


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with all subcommands.

    """
    parser = argparse.ArgumentParser(
        prog="tengri",
        description="tengri SED fitting CLI",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("doctor", help="Environment health check")

    p_cite = sub.add_parser("cite", help="Look up citations")
    p_cite.add_argument(
        "key",
        nargs="?",
        help="Registry key (omit to list all)",
    )
    p_cite.add_argument(
        "--bibtex",
        action="store_true",
        help="Output BibTeX",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Parameters
    ----------
    argv: Sequence of str, optional
        Command-line arguments. If None, uses sys.argv[1:].

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).

    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        import tengri

        version = getattr(tengri, "__version__", "unknown")
        print_logo(compact=True)
        print(version)
        return 0

    if args.command == "doctor":
        from tengri import doctor

        doctor()
        return 0

    if args.command == "cite":
        from tengri.citations import cite, cite_all, format_list

        if args.key:
            try:
                c = cite(args.key)
            except KeyError as e:
                print(f"Unknown citation key: {args.key}", file=sys.stderr)
                print(str(e), file=sys.stderr)
                return 1

            if args.bibtex:
                print(c.to_bibtex())
            else:
                print(str(c))
                print()
                print(c.to_bibtex())

            return 0

        # List all citations
        fmt = "bibtex" if args.bibtex else "short"
        output = format_list(cite_all(), fmt=fmt)
        print(output)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
