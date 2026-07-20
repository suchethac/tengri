# SPDX-License-Identifier: BSD-3-Clause
"""``tengri.help()`` teaches a fit that actually works (#1284).

``help()`` is the first thing a new user is told to call. Its "Build a fit"
section used to teach::

    obs        = tengri.Observation(...)
    parameters = tengri.Parameters(...)        # <- consumed by nothing
    sed        = tengri.SEDModel.build(ssp_data=..., observation=obs, ...)
    forward    = tengri.ForwardModel.build(sed=sed, observation=obs)
    fitter     = tengri.Fitter(forward, data, noise)
    posterior  = fitter.run("map")             # <- a point estimate
    posterior.summary()                        # "median +- 68% CI per param"

Three defects: ``SEDModel.build`` has no ``parameters`` argument, so line 2
built an object and passed it nowhere (silently — nothing errors, you just get
default priors); the three-line engine form was taught instead of the canonical
``forward.fit(...)``; and the one method it headlined, ``"map"``, is the one
whose output does not match the credible-interval description beside it.

These tests assert the cheatsheet stays executable: every symbol it names must
resolve, and every keyword it shows must exist in the signature it belongs to.
"""

from __future__ import annotations

import inspect
import re

import pytest

import tengri

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def help_text(capsys_module=None) -> str:
    """The rendered cheatsheet."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tengri.help()
    text = buf.getvalue()
    assert len(text) > 500, "help() produced almost nothing — did rendering break?"
    return text


def _build_fit_section(text: str) -> str:
    body = text.split("3.  Build a fit")[1]
    return body.split("4.  Contribute")[0]


def test_the_dead_parameters_line_is_gone(help_text):
    """``SEDModel.build`` takes no ``parameters=``; teaching it built a no-op."""
    section = _build_fit_section(help_text)
    assert "parameters = tengri.Parameters" not in section, (
        "help() constructs a Parameters object that SEDModel.build cannot "
        "accept — the user gets default priors and no error."
    )
    assert "parameters" not in inspect.signature(tengri.SEDModel.build).parameters, (
        "SEDModel.build gained a `parameters` argument — this test is now stale"
    )


def test_it_teaches_the_canonical_fit_entry_point(help_text):
    section = _build_fit_section(help_text)
    assert "forward.fit(" in section, (
        "help() must teach `forward.fit(...)`, the documented canonical entry "
        "point, not only the ForwardModel -> Fitter -> run engine chain."
    )


def test_it_does_not_headline_map_as_a_posterior(help_text):
    """MAP is a point estimate; it must not sit under a 68%-CI description."""
    section = _build_fit_section(help_text)
    summary_lines = [ln for ln in section.splitlines() if "68%" in ln or "CI per param" in ln]
    assert summary_lines, "the summary() description vanished — test is stale"
    for ln in summary_lines:
        assert '"map"' not in ln and 'run("map")' not in ln

    assert "point estimate" in section, (
        "help() should say what 'map' actually returns, since it is the "
        "cheapest method and users reach for it first."
    )


def test_it_points_at_recipes(help_text):
    """The documented onboarding surface was never mentioned."""
    section = _build_fit_section(help_text)
    assert "recipes" in section, "help() must mention tengri.recipes"


def test_every_tengri_symbol_named_in_help_resolves(help_text):
    """A cheatsheet that names a nonexistent function is worse than none."""
    names = set(re.findall(r"\btengri\.([A-Za-z_][A-Za-z0-9_]*)", help_text))
    missing = sorted(n for n in names if not hasattr(tengri, n))
    assert not missing, f"help() names symbols that do not exist: {missing}"


def test_the_methods_help_recommends_are_real_and_not_broken(help_text):
    """It must not steer users to a quarantined backend (#1287)."""
    from tengri.inference._backend_registry import get_backend

    section = _build_fit_section(help_text)
    quoted = set(re.findall(r'"([a-z0-9_]+)"', section))
    broken = {row["name"] for row in tengri.list_inference_methods(tier="broken")}

    for name in quoted:
        try:
            get_backend(name)
        except (ValueError, KeyError):
            continue  # not a method name (e.g. a recipe or a filter)
        assert name not in broken, f"help() recommends {name!r}, which is registered tier='broken'"


def test_the_default_method_shown_is_the_real_default(help_text):
    """help() interpolates DEFAULT_METHOD rather than restating it."""
    from tengri.inference._backend_registry import DEFAULT_METHOD

    section = _build_fit_section(help_text)
    assert f'method="{DEFAULT_METHOD}"' in section, (
        f"help() should show method={DEFAULT_METHOD!r}, the live default. "
        "A hard-coded literal here is how the five surfaces drifted apart."
    )


def test_the_canonical_import_line_actually_imports(help_text):
    """The sub-namespace import taught at the top of the section must work."""
    section = _build_fit_section(help_text)
    line = next(
        (ln.strip() for ln in section.splitlines() if ln.strip().startswith("from tengri.")),
        None,
    )
    assert line, "help() should show the canonical sub-namespace import"
    stmt = line.split("#")[0].strip()
    exec(compile(stmt, "<help>", "exec"), {})


def test_the_section_parser_is_not_vacuous(help_text):
    """Guard the guard: if the heading changed, every test above would pass."""
    section = _build_fit_section(help_text)
    assert len(section.strip()) > 200, (
        "the 'Build a fit' section came back nearly empty — the split anchors "
        "have rotted, so these tests prove nothing."
    )
