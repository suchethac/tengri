# SPDX-License-Identifier: BSD-3-Clause
"""One derived list of every registered dust attenuation law.

Two files swept "every dust law" by hand and neither swept all of them:

* ``tests/regression/synthesizer_parity/test_dust_extrapolation.py`` carried
  five copies of a 21-name list (one per property tested).
* ``tests/components/dust/test_dust_attenuation_laws.py`` carried a 20-name
  list.

``DUST_LAWS`` has 22 entries. ``reddy15`` appeared in neither file, so it had no
normalization, finiteness, non-negativity, far-IR, UV-slope or bump coverage at
all. ``prevot_smc`` was missing from the second only because it is deliberately
excluded there (it does not follow the same k(V)=1 convention and has its own
test) — that exclusion is now explicit rather than implicit in an omission.

Seven hand-maintained enumerations of "every law" is seven chances to miss one.
This module derives the set once, so a law registered tomorrow is swept without
editing any test file. ``tests/contract/test_dust_law_sweep_is_complete.py``
guards that the derivation still matches the registry and is not empty.
"""

from __future__ import annotations

import importlib.util

import pytest

from tengri.components.dust.attenuation import DUST_LAWS

#: Laws backed by the optional ``dust-extinction`` package. Without it they
#: raise ImportError at call time, which reported 26 red tests across two files
#: on a machine merely missing an extra — noise that hides real breakage in the
#: same files.
GRAIN_MODEL_LAWS = frozenset({"wd01_smcbar", "wd01_mwrv31", "d03_mwrv31", "hd23_mwrv31"})

requires_dust_extinction = pytest.mark.skipif(
    importlib.util.find_spec("dust_extinction") is None,
    reason=(
        "grain-model dust laws are backed by the optional `dust-extinction` "
        "package (pip install dust-extinction)"
    ),
)


def law_names(exclude: frozenset[str] | set[str] = frozenset()) -> list[str]:
    """Every registered law name, sorted, minus ``exclude``."""
    return sorted(set(DUST_LAWS) - set(exclude))


def every_dust_law(exclude: frozenset[str] | set[str] = frozenset()) -> list:
    """Parametrize argument covering every registered law.

    Grain models carry a skipif so a missing optional backend reads as a skip
    rather than a failure. Pass ``exclude`` for laws a particular property
    genuinely does not apply to — and say why at the call site, because an
    exclusion nobody can see is how ``reddy15`` went untested.
    """
    return [
        pytest.param(name, marks=[requires_dust_extinction]) if name in GRAIN_MODEL_LAWS else name
        for name in law_names(exclude)
    ]


def swept_names(params: list) -> set[str]:
    """The plain names inside a parametrize list built by ``every_dust_law``."""
    return {p.values[0] if hasattr(p, "values") else p for p in params}
