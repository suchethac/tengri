#!/usr/bin/env python
"""Helper to build .ipynb notebook files from Python cell definitions.

Usage:
    from _nb_helper import md, code, write_notebook

    cells = [
        md('''# Title
        Some markdown text with $\\LaTeX$ equations.
        '''),
        code('''
        import jax
        import jax.numpy as jnp
        '''),
    ]
    write_notebook("notebooks/00_quickstart.ipynb", cells)
"""

import json
import textwrap
from pathlib import Path


def _to_source(text):
    """Convert a text block to .ipynb source format (list of lines with \\n)."""
    # Dedent to remove common leading whitespace (from triple-quoted strings)
    text = textwrap.dedent(text)
    # Strip leading/trailing blank lines but preserve internal structure
    lines = text.split("\n")
    # Strip leading blank lines
    while lines and lines[0].strip() == "":
        lines.pop(0)
    # Strip trailing blank lines
    while lines and lines[-1].strip() == "":
        lines.pop()
    if not lines:
        return [""]
    # Add \n to all lines except the last
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md(text):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _to_source(text),
    }


def code(text):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _to_source(text),
    }


def write_notebook(path, cells):
    """Write a list of cells to a .ipynb file."""
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"Wrote {path} — {len(cells)} cells ({n_md} md, {n_code} code)")
