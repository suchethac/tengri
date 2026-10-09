# SPDX-License-Identifier: BSD-3-Clause
"""Nothing may dispatch on a configuration key from a hand-written subset.

``fig06_code_overlay`` carried ``{"I": config_I, "II": config_II, "III":
config_III}`` -- written when the paper demonstrated three configurations, and
untouched when the table grew to six. Every panel for a later configuration
died on a bare ``KeyError: 'IV'``, and because the figure ran inside a compound
command whose last stage succeeded, the shell reported exit 0 and no PDF. The
same literal sat in ``postprocess_ppd``. ``fit_one``'s copy happened to be
complete, which is the point: three hand-maintained censuses, and only luck
decided which ones drifted.

The sweep below enumerates rather than lists -- it parses every module in the
package and judges each configuration-keyed mapping against ``CONFIG_KEYS``, so
a mapping added tomorrow is covered without editing this file.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

pytestmark = pytest.mark.contract

#: Every key the configuration table declares, read from the declaration.
from configs import CONFIG_KEYS

CENSUS = set(CONFIG_KEYS)


def _modules() -> list[Path]:
    """Every module in the package proper -- tests dispatch on keys legitimately."""
    return sorted(p for p in ANALYSIS_DIR.glob("*.py"))


def _config_keyed_dicts(tree: ast.AST):
    """Yield ``(lineno, keys)`` for every dict literal keyed purely by config keys.

    A mapping qualifies only when *all* of its keys are string constants drawn
    from the census. That deliberately ignores dicts that merely happen to
    contain one such key beside unrelated ones, and dict comprehensions -- a
    comprehension derives its keys, which is the shape this guard asks for.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict) or not node.keys:
            continue
        if any(k is None for k in node.keys):  # ``{**other}`` spread
            continue
        keys = [
            k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)
        ]
        if len(keys) != len(node.keys) or not set(keys) <= CENSUS:
            continue
        yield node.lineno, set(keys)


def test_every_configuration_keyed_mapping_covers_the_census() -> None:
    """A literal keyed by configuration must name every configuration."""
    incomplete = []
    for path in _modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for lineno, keys in _config_keyed_dicts(tree):
            missing = CENSUS - keys
            if missing:
                incomplete.append(f"{path.name}:{lineno} omits {sorted(missing)}")

    assert not incomplete, (
        "configuration-keyed mapping(s) that the table has outgrown:\n  "
        + "\n  ".join(incomplete)
        + "\nDerive the mapping from CONFIG_KEYS instead of restating it."
    )


def test_configs_exposes_a_builder_for_every_census_key() -> None:
    """The naming contract ``fig06`` resolves builders through.

    ``fig06_code_overlay`` now reaches its builder with
    ``getattr(configs, f"config_{key}")``. That works only while the module
    spells every builder that way, so the spelling is the contract, not an
    implementation detail.
    """
    import configs

    missing = [key for key in CONFIG_KEYS if not callable(getattr(configs, f"config_{key}", None))]
    assert not missing, f"configs.py declares no callable config_<KEY> for {missing}"
