# SPDX-License-Identifier: BSD-3-Clause
"""Contract: every model choice triggers its citation on a *live* SEDModel.

Regression for the #938 audit. ``collect_citations`` must surface the right
paper for each configurable component of a live ``SEDModel`` — not only the
categories the config-based path covered. That path is effectively dead for a
SEDModel (``_find_model_config`` returns ``None``), so the nebular backend and
the X-ray / radio / shock components silently produced **no** citation until
they were mapped in ``_keys_from_live_registry``.

Data-gated (needs a real SSP grid); skips in CI.
"""

from __future__ import annotations

import pytest

from tengri import FIXED, FREE, Fixed, Observation, Photometry, SEDModel, builders
from tengri.citations import collect_citations

pytestmark = pytest.mark.contract


def _citation_keys(model) -> set[str]:
    return {c.key for c in collect_citations(model)}


def _build(ssp, **extra):
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))
    kwargs = dict(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "*": FREE},
        dust_attenuation={"type": "two_component", "law": "calzetti", "*": FIXED},
        redshift=Fixed(0.1),
    )
    kwargs.update(extra)
    return SEDModel.build(**kwargs)


@pytest.mark.parametrize(
    "extra, expected",
    [
        ({"neb": {"type": "cue", "*": FIXED}}, "cue"),
        (
            {
                "neb": {"type": "none"},
                "radio": {"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}},
            },
            "condon1992",
        ),
        ({"neb": {"type": "none"}, "shock": {"type": "mappings"}}, "mappings"),
        (
            {
                "neb": {"type": "none"},
                "agn": {"type": "composable", "disc": builders.agn.disc.multicolor(defaults=FREE)},
                "xray": {"type": "simple"},
            },
            "lehmer2016",
        ),
    ],
    ids=["nebular_cue", "radio_condon92", "shock_mappings", "xray_simple"],
)
def test_component_triggers_citation(ssp_data_fsps, extra, expected):
    keys = _citation_keys(_build(ssp_data_fsps, **extra))
    assert expected in keys, f"{expected!r} not triggered on live SEDModel; got {sorted(keys)}"


def test_igm_and_ssp_baseline_citations(ssp_data_fsps):
    """The default (IGM on, real SSP) run cites the IGM model + SSP provenance."""
    keys = _citation_keys(_build(ssp_data_fsps, neb={"type": "none"}))
    assert "inoue2014" in keys  # default IGM model
    assert "fsps" in keys  # SSP provenance
