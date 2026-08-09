# SPDX-License-Identifier: BSD-3-Clause
"""Sphinx cross-reference targets in docstrings must resolve (#1616).

``docs/api/*.rst`` are autodoc stubs, so **the docstrings are the API
reference**. A ``:func:`~tengri.X.Y``` in one renders as a link on the
published page, and ``nitpicky`` is off — so a target that does not exist is a
dead link that produces no build warning and no test failure. 19 were dead when
this check was added, including
``tengri.components.radio.RadioSEDComponent``, whose class had been renamed
``RadioPowerLawSEDComponent``.

Two subtleties produced false positives before they were handled, and any
rewrite must keep both:

* **Targets wrap.** A long path breaks across lines in a docstring and Sphinx
  joins it, so whitespace must be collapsed first. Ignoring this reported 24
  dead targets when 22 were dead.
* **Attributes shadow modules.** ``tengri.components.agn.qsogen`` is a
  submodule *and* a re-exported function; the function wins on attribute
  access. A plain ``getattr`` walk therefore called the live target
  ``...qsogen.compute_qsogen_sed`` dead — 22 reported, 19 real.

Both numbers were wrong in the direction that manufactures work, which is why
the resolver is pinned here rather than trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import check_doc_examples as cde


class TestTheResolverStillDiscriminates:
    """A resolver that says yes (or no) to everything makes the sweep useless."""

    def test_a_dead_target_is_rejected(self):
        assert not cde.xref_resolves("tengri.forward.SEDModel.build")
        assert not cde.xref_resolves("tengri.parameters._param_defs")

    def test_a_live_target_is_accepted(self):
        assert cde.xref_resolves("tengri.SEDModel.build")
        assert cde.xref_resolves("tengri.components.radio.component.RadioSEDComponent")

    def test_a_module_shadowed_by_a_function_still_resolves(self):
        """``agn.qsogen`` is both a submodule and a re-exported function."""
        assert cde.xref_resolves("tengri.components.agn.qsogen.compute_qsogen_sed")

    def test_a_bare_module_resolves(self):
        assert cde.xref_resolves("tengri.parameters._builders")


class TestTheParser:
    def test_a_wrapped_target_is_joined(self):
        assert cde.xref_targets(":func:`~tengri.a.\n    b`") == ["tengri.a.b"]

    def test_the_explicit_title_form_is_read(self):
        assert cde.xref_targets(":class:`the model <tengri.SEDModel>`") == ["tengri.SEDModel"]

    def test_trailing_parens_are_stripped(self):
        assert cde.xref_targets(":func:`~tengri.SEDModel.build()`") == ["tengri.SEDModel.build"]

    def test_non_tengri_targets_are_out_of_scope(self):
        assert cde.xref_targets(":class:`numpy.ndarray` :func:`jax.jit`") == []

    def test_every_role_we_claim_to_read_is_read(self):
        roles = ("func", "meth", "class", "attr", "data", "exc", "obj", "mod")
        text = " ".join(f":{r}:`~tengri.SEDModel`" for r in roles)
        assert len(cde.xref_targets(text)) == len(roles)


def test_the_repository_has_no_dead_cross_references():
    """The contract itself."""
    violations = [v for v in cde.check() if "cross-reference" in v]
    assert not violations, "dead cross-references:\n" + "\n".join(violations)
