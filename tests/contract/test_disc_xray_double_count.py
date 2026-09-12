# SPDX-License-Identifier: BSD-3-Clause
"""Contract: refuse an AGN-corona ``xray`` variant on a disc with its own corona.

Some composable ``agn.disc`` blocks already carry a hot corona baked into the
tabulated/analytic template (``kubota_done``, ``kd18_agnfitter``,
``kd18_agnfitter_warmindex``). Composing one of them with an ``xray`` group
selection that *also* adds an AGN corona (alpha_ox or alpha_IRX) would double
the coronal power, +51% over 0.5-10 keV for the Kubota & Done corona fraction.
:func:`tengri.components.xray._models.check_disc_xray_double_count` refuses
this combination at build time (``parameters/groups.py::_translate_structural``).

There is currently no registered host-XRB-only ``xray`` variant (one that
never adds a corona term): every active selection besides ``'none'`` --
``'simple'``/``'yang20'``/``'lopez24'``/``'xray_aird'``/``'agn_xray_corona'``
-- includes one (see ``AGN_CORONA_XRAY_VARIANTS`` in
``components/xray/_models.py``), so the "if one exists" host-XRB-only build
case from the task brief has no variant to exercise.
"""

from __future__ import annotations

import warnings

import pytest

from tengri.components.agn.blocks.runner import RecipeWarning
from tengri.parameters import FREE, parse_groups

pytestmark = pytest.mark.contract


def _build(disc: str, xray_type: str):
    """Build a minimal composable-AGN Parameters with one disc + xray selection.

    Advisory ``RecipeWarning``s (e.g. #1586 grid-support clipping) are muted:
    this module is about the hard ``ValueError`` double-count guard, not the
    warnings a wide wildcard may also raise for an unrelated reason.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RecipeWarning)
        return parse_groups(
            sfh={"type": "dpl"},
            agn={
                "type": "composable",
                "disc": {"type": disc},
                "norm": "independent",
                "all_params": FREE,
            },
            xray={"type": xray_type},
            redshift=0.1,
        )


_ERROR_TEXT = "already carries a hot corona"


def test_kubota_done_plus_corona_raises():
    """kubota_done's own corona + xray's alpha_ox corona double-counts."""
    with pytest.raises(ValueError, match=_ERROR_TEXT) as excinfo:
        _build("kubota_done", "simple")
    message = str(excinfo.value)
    assert "kubota_done" in message
    assert "xray={'type': 'simple'}" in message
    assert "0.5-10 keV" in message
    assert "xray={'type': 'none'}" in message


def test_kd18_agnfitter_plus_corona_raises():
    """The new grid-tabulated KD18 disc carries the same corona conflict."""
    with pytest.raises(ValueError, match=_ERROR_TEXT) as excinfo:
        _build("kd18_agnfitter", "yang20")
    assert "kd18_agnfitter" in str(excinfo.value)


def test_kd18_agnfitter_warmindex_plus_corona_raises():
    """The warmIndex variant of the KD18 disc also carries its own corona."""
    with pytest.raises(ValueError, match=_ERROR_TEXT) as excinfo:
        _build("kd18_agnfitter_warmindex", "lopez24")
    assert "kd18_agnfitter_warmindex" in str(excinfo.value)


def test_kubota_done_plus_xray_none_builds():
    """Disabling xray altogether is always compatible with a corona-carrying disc."""
    spec = _build("kubota_done", "none")
    assert spec is not None


def test_richards2006_plus_corona_builds():
    """A disc with no intrinsic X-rays composes fine with an xray corona."""
    spec = _build("richards2006", "simple")
    assert spec is not None


def test_agn_xray_corona_variant_also_raises():
    """The corona-only SEDModelComponent variant is refused too, not just the
    two function-based XRAY_MODELS entries."""
    with pytest.raises(ValueError, match=_ERROR_TEXT):
        _build("kubota_done", "agn_xray_corona")


def test_xray_aird_variant_also_raises():
    """``xray_aird`` reads the same corona term as 'simple'/'yang20'."""
    with pytest.raises(ValueError, match=_ERROR_TEXT):
        _build("kd18_agnfitter", "xray_aird")


def test_no_disc_no_xray_conflict():
    """No composable AGN disc at all: nothing to double-count."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RecipeWarning)
        spec = parse_groups(
            sfh={"type": "dpl"},
            xray={"type": "simple"},
            redshift=0.1,
        )
    assert spec is not None
