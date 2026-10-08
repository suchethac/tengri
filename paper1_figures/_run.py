# SPDX-License-Identifier: BSD-3-Clause
"""Call a figure script's entry point in-process, whatever shape it has.

The scripts under ``analysis/paper1/`` do not share a CLI contract: some
``main`` take an ``argv`` list, some take named parameters, others read
``sys.argv``, and one has no ``main`` at all and does its work at import.
That is worth fixing in those scripts one day, but not from here and not
mid-paper; until then every notebook would otherwise need to know which kind
it is calling.

In-process rather than by subprocess, deliberately. A subprocess returns an
exit code and a wall of text; an import keeps the traceback, so a notebook that
fails says where.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

#: Exit status a notebook uses when it cannot run for want of inputs, as
#: distinct from succeeding (0) or failing (anything else). A skip that exits 0
#: is indistinguishable from a figure that was produced, which is exactly the
#: reading that made this folder's README claim more than it delivered.
SKIPPED = 3


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
    dotted = f"analysis.paper1.{module_name}"

    # sys.argv is set BEFORE the import, not around the main() call. One script
    # here has no main at all -- its argparse and its savefig run at module
    # scope -- so by the time an import returns, that script has already read
    # sys.argv and written its figures. Patching afterwards would hand it the
    # notebook's own arguments.
    saved = sys.argv
    sys.argv = [module_name, *argv]
    try:
        already_imported = dotted in sys.modules
        module = importlib.import_module(dotted)
        main = getattr(module, "main", None)

        if main is None:
            # Shape four: the script is its own module body. The import above
            # did the work -- unless Python had already cached the module, in
            # which case the body did not re-run and nothing was written. That
            # would have reported success having produced no figure.
            if already_imported:
                importlib.reload(module)
            return 0

        signature = inspect.signature(main)
        params = list(signature.parameters)

        # Three further shapes, each needing different handling. The middle one
        # is the dangerous one: its argparse lives in the __main__ block, so
        # main() ignores sys.argv entirely and falls back to its own defaults.
        # Called the wrong way it exits 0 and writes the figure somewhere else,
        # which is the quietest possible failure.
        if params and params[0] == "argv":
            return int(main(argv) or 0)
        if params:
            return int(main(**_as_kwargs(argv, signature)) or 0)
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


#: Prefix of the per-figure line a family prints, which ``regenerate.py`` counts.
FIGURE_LINE = "FIGURE"


def figure_outcome(status: object, path: Path | None = None) -> str | None:
    """Return why a figure is not produced, or ``None`` when it is.

    Parameters
    ----------
    status : int or str
        What ``run_figure`` returned for the figure, or a ``refused: ...``
        string when the call raised.
    path : Path, optional
        The one file the figure should now exist as. ``None`` for a figure
        whose filenames the script does not know in advance, which is judged on
        ``status`` alone.

    Returns
    -------
    str or None
        ``None`` when the call succeeded and ``path`` is a file. Otherwise the
        reason, for the family's report.
    """
    if isinstance(status, str):
        return status
    if status != 0:
        return f"exit status {status}"
    if path is not None and not path.is_file():
        return f"not written to {path}"
    return None


def report_figures(outcomes: dict[str, str | None]) -> int:
    """Print one line per figure and return the family's exit status.

    A family's status is non-zero when any of its figures is refused or
    missing. Returning 0 with a figure absent is the defect this guards
    against: ``regenerate.py`` takes each family's status as its verdict.

    Parameters
    ----------
    outcomes : dict of str to str or None
        Figure name to its reason from :func:`figure_outcome`, or ``None`` when
        the figure was produced.

    Returns
    -------
    int
        ``0`` when every figure was produced, :data:`SKIPPED` otherwise.
    """
    complete = True
    for name, reason in outcomes.items():
        if reason is None:
            print(f"{FIGURE_LINE} {name} ok")
        else:
            complete = False
            print(f"{FIGURE_LINE} {name} missing: {reason}")
    return 0 if complete else SKIPPED
