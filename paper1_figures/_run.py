# SPDX-License-Identifier: BSD-3-Clause
"""Call a figure script's entry point in-process, whatever shape it has.

The scripts under ``analysis/paper1/`` do not share a CLI contract: some
``main`` take an ``argv`` list, others read ``sys.argv``, and one has no
``main`` at all. That is worth fixing in those scripts one day, but not from
here and not mid-paper; until then every notebook would otherwise need to know
which kind it is calling.

In-process rather than by subprocess, deliberately. A subprocess returns an
exit code and a wall of text; an import keeps the traceback, so a notebook that
fails says where.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Walk up until the directory holding ``analysis/paper1`` is found."""
    here = (start or Path.cwd()).resolve()
    while not (here / "analysis" / "paper1").is_dir():
        if here == here.parent:
            raise RuntimeError("not inside the tengri repository")
        here = here.parent
    return here


def run_figure(module_name: str, argv: list[str]) -> int:
    """Run ``analysis.paper1.<module_name>`` as if from the command line.

    Returns the script's exit status. A non-zero status is returned rather than
    raised so a notebook can report several figures and still finish; the
    caller decides what a failure means.
    """
    module = importlib.import_module(f"analysis.paper1.{module_name}")
    main = getattr(module, "main", None)
    if main is None:
        raise AttributeError(f"{module_name} has no main() to call")

    signature = inspect.signature(main)
    params = list(signature.parameters)

    # Three shapes exist in this tree and each needs different handling. The
    # middle one is the dangerous one: its argparse lives in the __main__
    # block, so main() ignores sys.argv entirely and silently falls back to its
    # own defaults. Called the wrong way it exits 0 and writes the figure
    # somewhere else, which is the quietest possible failure.
    if params and params[0] == "argv":
        return int(main(argv) or 0)

    if params:
        return int(main(**_as_kwargs(argv, signature)) or 0)

    saved = sys.argv
    sys.argv = [module_name, *argv]
    try:
        return int(main() or 0)
    finally:
        sys.argv = saved


def _as_kwargs(argv: list[str], signature: inspect.Signature) -> dict:
    """Map ``--flag value`` pairs onto a main() that takes named parameters.

    Refuses a flag with no matching parameter rather than dropping it: a
    silently ignored --out-dir is how a figure lands in the wrong directory
    while the call reports success.
    """
    pairs: dict[str, str] = {}
    items = list(argv)
    while items:
        flag = items.pop(0)
        if not flag.startswith("--"):
            raise ValueError(f"expected a --flag, got {flag!r}")
        if not items:
            raise ValueError(f"{flag} has no value")
        pairs[flag[2:].replace("-", "_")] = items.pop(0)

    unknown = sorted(set(pairs) - set(signature.parameters))
    if unknown:
        raise TypeError(
            f"main() has no parameter for {unknown}; it would have been ignored "
            "and the figure written to a default location"
        )

    out: dict[str, object] = {}
    for name, value in pairs.items():
        annotation = signature.parameters[name].annotation
        out[name] = Path(value) if "Path" in str(annotation) else value
    return out
