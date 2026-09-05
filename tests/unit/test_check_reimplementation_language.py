# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for ``tools/check_reimplementation_language.py``.

The guard has two jobs and both are tested here. It has to catch the claim
that tengri ported or copied someone else's source, and it has to stay quiet
on the honest uses of the same words: network ports, ``deepcopy``, an adapted
MCMC step size, ``exported from``. False positives matter as much as misses,
since a guard that cries wolf gets switched off.
"""

import importlib.util
import re
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_reimplementation_language.py"
_spec = importlib.util.spec_from_file_location("check_reimplementation_language", _TOOL)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def _reasons(text: str) -> list[str]:
    return [reason for _, _, reason in guard.scan_text(text)]


# --- the guard must fire on these -------------------------------------------

VIOLATIONS = [
    "This module is ported from CIGALE's dustatt_modified_starburst.",
    "Ported from ProSpect `massfunc_spline` (Robotham+2020).",
    "The disc spectrum is a port of Prospector's implementation.",
    "This is a faithful port of the Yallup, Kroupa & Handley (2026) algorithm.",
    "Coefficients copied from eazy-py.",
    "Torus templates adapted from AGNfitter-rX.",
    "The Kompaneets solver was translated from the original Fortran.",
    "A line-for-line port of the upstream routine.",
    "Reference SED-fitting frameworks (for comparison and porting credit)",
    "SKIRTOR port lives in components/agn/.",  # proximity rule
    "canonical ports for CIGALE-equivalent physics",  # proximity rule
]


@pytest.mark.parametrize("line", VIOLATIONS)
def test_flags_provenance_claims(line):
    assert _reasons(line), f"guard missed a provenance claim: {line!r}"


def test_detection_is_case_insensitive():
    """'Ported from' at the start of a sentence is the common real-world form."""
    assert _reasons("Ported from Prospector (Johnson+2021).")
    assert _reasons("PORTED FROM CIGALE.")


# --- the guard must stay silent on these ------------------------------------

INNOCENT = [
    # Network ports and hostnames.
    "port=3306",
    "  port:   3306",
    "Data source: http://www.icg.port.ac.uk/~maraston/M05",
    "port = _DB_PORT",
    # Python semantics, not provenance.
    "params = copy.deepcopy(defaults)",
    "the list is freshly copied — callers may not mutate it",
    # MCMC terms of art.
    "chains share one adapted step size and mass matrix",
    "the compiled kernel is independent of the adapted step_size",
    # Module plumbing: 'exported from' contains the letters 'ported from'.
    "Sentinels FREE / FIXED are singletons exported from `tengri`.",
    "Fundamental constants are imported from physics_constants.",
    "Multi-population is supported from day one.",
    "`components/` never imports from `forward/`",
    "Names re-exported from this namespace are canonical.",
    # The approved framings.
    "Implements the same model as Prospector (Johnson et al. 2021); validated against it.",
    "CIGALE's bundled Chabrier-IMF grid, repackaged in the DSPS HDF5 layout.",
    "Implements the Meiksin (2006) IGM model as CIGALE evaluates it.",
    "reimplemented in JAX following the published equations",
]


@pytest.mark.parametrize("line", INNOCENT)
def test_ignores_innocent_uses(line):
    assert not _reasons(line), f"guard false-positived on: {line!r}"


def test_bare_port_without_a_reference_code_is_not_flagged():
    """The proximity rule needs BOTH halves, or every network port trips it."""
    assert not _reasons("bind the server to a free port")
    assert _reasons("bind the CIGALE port")


# --- prose wrapped across a line break ---------------------------------------
#
# Docstrings here wrap at 99 columns and markdown at about 79, so a phrase as
# short as "ported from" straddles a line break often. A scan that reads one
# line at a time cannot see it: "ported" ends one line, "from CIGALE" starts
# the next, and neither half trips anything on its own.

WRAPPED_VIOLATIONS = [
    # The reference code sits on the second line, so the proximity rule on the
    # first line has nothing to pair with either.
    "The tabulated torus grid was ported\nfrom CIGALE's bundled library.\n",
    # No reference code anywhere: only the split phrase can catch this.
    "The attenuation coefficients were copied\nfrom the published tables.\n",
    "The birth-cloud treatment is adapted\nfrom the reference code.\n",
    # Indented continuation, the usual docstring shape.
    '"""Torus grid.\n\n    The grid was translated\n    from the original Fortran.\n    """\n',
]


@pytest.mark.parametrize("text", WRAPPED_VIOLATIONS)
def test_flags_provenance_claims_wrapped_across_lines(text):
    assert _reasons(text), f"guard missed a wrapped provenance claim: {text!r}"


WRAPPED_INNOCENT = [
    # 'exported'/'supported' contain the letters of 'ported'. The \b anchor has
    # to survive the join, or module plumbing lights up the whole build.
    "Sentinels FREE / FIXED are singletons exported\nfrom `tengri`.\n",
    "Multi-population is supported\nfrom day one.\n",
    # A blank line is a paragraph break, not a wrap. Joining across it would
    # invent "ported from-scratch" out of two unrelated sentences.
    "the grid was ported\n\nfrom-scratch rewrites are welcome\n",
    # A finished sentence is not a wrap either.
    "bind the server to a free port.\nCIGALE needs one too.\n",
    # The approved framing, wrapped.
    "CIGALE's bundled Chabrier-IMF grid, repackaged\nin the DSPS HDF5 layout.\n",
]


@pytest.mark.parametrize("text", WRAPPED_INNOCENT)
def test_wrapping_does_not_invent_violations(text):
    assert not _reasons(text), f"guard false-positived on wrapped prose: {text!r}"


def test_wrapped_hit_is_reported_once_with_a_usable_line_number():
    """One wrapped phrase is one finding, anchored on the line it starts."""
    text = "intro line\nThe torus grid was ported\nfrom CIGALE.\ntrailing line\n"
    hits = list(guard.scan_text(text))
    assert len(hits) == 1, f"expected exactly one finding, got {hits}"
    line_no, _, reason = hits[0]
    assert line_no == 2, f"should point at the line the phrase starts on, got {line_no}"
    assert "ported" in reason.lower()


# --- allowlist plumbing ------------------------------------------------------


def test_rule_stating_files_are_excluded():
    """Files that must quote the banned wording are allowlisted, not rewritten."""
    for rel in ("CLAUDE.md", "reproduction/CONTRACT.md", "docs/adr/0002-license-bsd3.md"):
        assert rel in guard.EXCLUDE_FILES


def test_upstream_copyright_headers_are_excluded():
    """Changing someone else's copyright notice is never a style sweep's call."""
    assert "src/tengri/inference/backends/nested".startswith(guard.EXCLUDE_DIRS)


def test_docstring_only_names_real_constants():
    """The first draft pointed at a ``PORT_NEAR_REFERENCE`` that never existed.

    A docstring that sends a reader looking for a symbol which is not there is
    worse than no docstring, and it survives every other check in this file.
    """
    named = set(re.findall(r"``([A-Z_]{3,})``", guard.__doc__))
    assert named, "docstring should reference the constants it describes"
    missing = sorted(n for n in named if not hasattr(guard, n))
    assert not missing, f"docstring names constants that do not exist: {missing}"


def test_repository_is_clean():
    """The whole repo passes the guard, so a regression fails loudly in CI.

    Issue #2075: pass explicit argv to guard.main() so pytest flags don't interfere.
    """
    assert guard.main([]) == 0
