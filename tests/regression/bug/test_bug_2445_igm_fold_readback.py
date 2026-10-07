# SPDX-License-Identifier: BSD-3-Clause
"""The IGM fold that ``WavePrecomp(igm_fold="auto")`` resolved to is reported (#2445).

``"auto"`` resolves at build time to the exact fold (the transmission integrated
inside the bandpass, so a band crossing the Lyman break gets
:math:`\\langle S T \\rangle`) or to the node fold (the transmission sampled at the
sub-band nodes, which forms :math:`\\langle S \\rangle \\langle T \\rangle` and is off by
tens of per cent at z >~ 3 when the break sits inside a band). The fit must be
able to say which quadrature it used: ``observed_facts["igm_fold"]`` carries the
declared mode beside the resolved one (``None`` when no fold was built).
"""

import pytest

import tengri
from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, WavePrecomp
from tengri.forward.precompute_report import precompute_engagement_report

pytestmark = pytest.mark.regression_bug

#: galex_fuv sits blueward of the Lyman break at z = 2.5.
BANDS = ["galex_fuv", "des_g"]


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp()


def _model(ssp, *, igm_fold="auto", igm=None):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(BANDS)),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Fixed(2.5),
        igm={"type": "inoue14"} if igm is None else igm,
        approx=WavePrecomp(igm_fold=igm_fold),
    )


def _fold_facts(ssp, **kwargs):
    """Build a fixed-z model and return ``(facts["igm_fold"], summary_text)``."""
    report = precompute_engagement_report(_model(ssp, **kwargs))
    return report.observed_facts["igm_fold"], report.summary_text()


def test_auto_on_a_smooth_fixed_z_transmission_reports_exact(ssp):
    """The exact fold is servable here, so ``"auto"`` must report it."""
    fact, _ = _fold_facts(ssp)
    assert fact == {"declared": "auto", "resolved": "exact"}


def test_explicit_node_is_reported_as_node(ssp):
    """The node fold is applied as named: declared and resolved agree."""
    fact, _ = _fold_facts(ssp, igm_fold="node")
    assert fact == {"declared": "node", "resolved": "node"}


def test_explicit_exact_where_servable_is_reported_as_exact(ssp):
    """A named exact fold is built and reported as such."""
    fact, _ = _fold_facts(ssp, igm_fold="exact")
    assert fact == {"declared": "exact", "resolved": "exact"}


def test_patchy_transmission_falls_back_to_node_under_auto_but_exact_raises(ssp):
    """Free-parameter transmission cannot be folded exactly at build time.

    ``"auto"`` takes the node fold and says so; naming ``"exact"`` still raises.
    """
    patchy = {"type": "inoue14", "patchy": True}
    fact, _ = _fold_facts(ssp, igm=patchy)
    assert fact == {"declared": "auto", "resolved": "node"}
    with pytest.raises(ValueError, match="fixed function of"):
        _model(ssp, igm=patchy, igm_fold="exact").predict_photometry({})


def test_no_igm_component_reports_no_fold_with_the_declared_mode(ssp):
    """Nothing is folded without an IGM component: ``resolved`` is None."""
    fact, _ = _fold_facts(ssp, igm={"type": "none"})
    assert fact == {"declared": "auto", "resolved": None}


def test_summary_text_names_the_declared_and_resolved_fold(ssp):
    """The one-line summary shows both modes."""
    _, summary = _fold_facts(ssp)
    assert "igm_fold: declared=auto, resolved=exact" in summary
