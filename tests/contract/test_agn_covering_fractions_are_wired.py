# SPDX-License-Identifier: BSD-3-Clause
"""Every declared AGN covering-fraction parameter must be consumed by an AGN block.

``agn_feltre_cf`` was registered with a ``Uniform(0, 1)`` prior, routed to the
``agn.nlr`` group and advertised in the NLR/BLR builder docstrings — but no code ever
read it (#1263). It duplicated ``agn_nlr_cf``, which the Feltre block already passes to
``compute_nlr_sed_feltre`` as the single ``covering_fraction``.

That is worse than a dead knob. A silently-ignored parameter declared **FREE** gives the
sampler a dimension the likelihood does not depend on — an exactly flat direction, so
samples are wasted and convergence diagnostics report on something unidentifiable.

The physics admits exactly one covering fraction per region. The Feltre grid ships
normalized per ionizing photon (``log10(L_Hbeta / Q_H)``, from NEOGAL files stored at a
reference ``L_acc = 1e45 erg/s``), so precisely one external factor converts disc
bolometric luminosity into the intercepted accretion luminosity that sets ``Q_H``:

    l_acc_erg = covering_fraction * l_disc_bol_erg      # nlr_cloudy.py
    log_qh    = _log_qh_from_lacc(l_acc_erg, alpha_pl)

A second covering-fraction parameter could only double-count it or displace the working
one. Hence this guard: if an ``agn_*_cf`` parameter is declared, some AGN block must
take it as an argument.

References
----------
.. [1] A. Feltre, S. Charlot, J. Gutkin, MNRAS 456, 3354 (2016).
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

pytestmark = pytest.mark.contract

_BLOCKS_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "tengri"
    / "components"
    / "agn"
    / "blocks"
)


def _block_argument_names() -> set[str]:
    """Every argument name accepted by any function in the AGN blocks package."""
    names: set[str] = set()
    for path in sorted(_BLOCKS_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - blocks must parse
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                for arg in a.posonlyargs + a.args + a.kwonlyargs:
                    names.add(arg.arg)
    return names


def test_every_agn_covering_fraction_param_is_consumed_by_a_block():
    """A declared ``agn_*_cf`` with no block consuming it is a silent no-op."""
    from tengri.parameters.registry import registry

    declared = sorted(n for n in registry() if re.fullmatch(r"agn_\w+_cf", n))
    assert declared, "expected at least one agn_*_cf parameter in the registry"

    consumed = _block_argument_names()
    orphans = [n for n in declared if n not in consumed]

    assert not orphans, (
        "AGN covering-fraction parameter(s) declared but consumed by no block — a user "
        f"can set (or fit) these and nothing happens: {orphans}. "
        "Wire them into the relevant block, or remove the declaration; do not leave a "
        "samplable parameter that the likelihood ignores (#1263)."
    )
