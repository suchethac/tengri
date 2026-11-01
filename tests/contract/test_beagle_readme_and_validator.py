# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the BEAGLE README and the matched-physics validator keep their promises.

The README must name the setup variable, the registry pull and the Terms of Use
boundary, and it must cite the six references the BEAGLE Terms of Use makes mandatory.
The validator must print its help, and it must exit 2 with the export line when no
BEAGLE workdir is configured. Neither test starts Docker or needs the cache.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BEAGLE_DIR = _REPO_ROOT / "reproduction" / "beagle"
_README = _BEAGLE_DIR / "README.md"
_VALIDATOR = _BEAGLE_DIR / "validate_matched_physics.py"

_TOU_SENTENCE = "Section 6.2 of the BEAGLE Terms of Use bars"

# Terms of Use section 2.2 entries 1-6: first author and year, as the README cites them.
_MANDATORY_CITATIONS = (
    ("Chevallard", "2016"),
    ("Bruzual", "2003"),
    ("Gutkin", "2016"),
    ("Calzetti", "1994"),
    ("Charlot", "2000"),
    ("Chevallard", "2013"),
)

_BANNED = re.compile(
    r"#[0-9]{3,4}|\b(ported|copied|adapted) from|we changed|used to|previously|\bnow\b"
    r"|this was fixed|catalogue|colour|normalis"
)


def _readme() -> str:
    return _README.read_text(encoding="utf-8")


def _citing_section() -> str:
    text = _readme()
    start = text.index("## Citing")
    end = text.index("\n## ", start + 1)
    return text[start:end]


def _env_without_workdir() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "BEAGLE_WORKDIR"}
    env["PYTHONPATH"] = os.pathsep.join([str(_REPO_ROOT / "src"), str(_REPO_ROOT)])
    env.setdefault("JAX_PLATFORMS", "cpu")
    return env


def _run_validator(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_VALIDATOR), *args],
        cwd=_REPO_ROOT,
        env=_env_without_workdir(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_readme_names_workdir_and_pull() -> None:
    text = _readme()
    assert "BEAGLE_WORKDIR" in text
    assert "docker pull" in text


def test_readme_states_terms_of_use_boundary() -> None:
    assert _TOU_SENTENCE in _readme()


@pytest.mark.parametrize(("author", "year"), _MANDATORY_CITATIONS)
def test_readme_cites_mandatory_entry(author: str, year: str) -> None:
    citing = _citing_section()
    assert author in citing
    assert year in citing


def test_readme_banned_vocabulary_is_absent() -> None:
    hits = [m.group(0) for m in _BANNED.finditer(_readme())]
    assert hits == []


def test_validator_help_runs() -> None:
    proc = _run_validator("--help")
    assert proc.returncode == 0, proc.stderr
    assert "--tol" in proc.stdout


def test_validator_exits_2_without_workdir() -> None:
    proc = _run_validator()
    assert proc.returncode == 2
    assert "export BEAGLE_WORKDIR" in proc.stderr
