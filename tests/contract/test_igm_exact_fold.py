#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The opt-in exact IGM fold, against the exact wavelength-grid integrator.

``WavePrecomp(igm_fold=...)`` chooses how IGM transmission enters the sub-band
quadrature weights. The default ``"node"`` evaluates transmission at the
quadrature nodes and multiplies, forming <S><T>; ``"exact"`` carries it inside
the bandpass integral, <S*T>. The two differ only when transmission varies
within a band, which is exactly where a Lyman break falls inside a bandpass.

The feature shipped with no test at all. These pin the three things that can
regress silently:

1. the default stays ``"node"``. Roughly twenty peer sessions build against
   this code, so flipping the default changes their numbers without any
   diagnostic;
2. the refusals fire, rather than quietly returning the node answer, when the
   exact fold cannot be built;
3. **the exact fold actually is more accurate**, measured against
   ``approx=None`` -- the dense wavelength-grid integrator -- in a band that
   straddles the break.

Point 3 is measured on a **bare stellar** model. A dusty model's IGM error is
entangled with the dust screen's own sub-band error, and ranking the two folds
by |error| there measures a sign coincidence rather than the fold.
"""

from __future__ import annotations

import numpy as np
import pytest

import tengri
from tengri import DEFAULT, SEDModel, WavePrecomp
from tengri.parameters import Fixed

pytestmark = pytest.mark.contract

# At z=1 the Lyman limit lands at 912*(1+z) = 1824 A observed. GALEX NUV spans
# roughly 1691-3009 A and so contains it; the break is inside the bandpass and
# the two folds must disagree. An optical band is the control: transmission is
# flat across it, so both folds must agree with the integrator and each other.
STRADDLING_BAND = "galex_nuv"
CONTROL_BAND = "sdss_r"
# Photometry.from_names preserves the order it is given, and this module is the
# only thing that builds the observation, so the band order is known here.
BANDS = [STRADDLING_BAND, CONTROL_BAND]
STRADDLING = 0
CONTROL = 1
PROBE_Z = 1.0


@pytest.fixture(scope="module")
def ssp(ssp_data_wne):
    """The suite's own grid.

    `tests/conftest.py` disables the ancestor-directory walk in `data_dirs()`
    for hermeticity (#2329), so `load_ssp` by name finds nothing under pytest
    even when the grid is readable outside it. Both arms of every comparison
    below use this same grid, so the choice of library does not enter.
    """
    return ssp_data_wne


@pytest.fixture(scope="module")
def observation():
    return tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))


def _bare_stellar(ssp, observation, approx, z=PROBE_Z):
    """One model, identical in every respect but the approximation path.

    Deliberately no dust and no nebular group: this arm must isolate the IGM.
    A nebular group would also route through the LyC mask, which has its own
    open defect and would contaminate the comparison.
    """
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": Fixed(2.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
            "met_logzsol": Fixed(0.0),
        },
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=approx,
    )


def _photometry(model):
    return np.asarray(model.predict_photometry({}), dtype=np.float64)


def test_default_fold_is_node():
    """The default must not move: peer sessions build against it."""
    assert WavePrecomp().igm_fold == "node"
    assert WavePrecomp(n_subbands=5).igm_fold == "node"


def test_exact_is_an_accepted_value():
    assert WavePrecomp(igm_fold="exact").igm_fold == "exact"


def test_an_unknown_fold_is_refused_by_name():
    """A typo must not fall back to a default and be silently wrong."""
    with pytest.raises(ValueError) as excinfo:
        WavePrecomp(igm_fold="nodes")
    message = str(excinfo.value)
    assert "nodes" in message
    assert "'node'" in message and "'exact'" in message


def test_a_free_redshift_refuses_the_exact_fold(ssp, observation):
    """Refuse, rather than return the node answer under the exact label."""
    with pytest.raises(NotImplementedError) as excinfo:
        model = SEDModel.build(
            ssp_data=ssp,
            observation=observation,
            sfh={
                "type": "delayed",
                "all_params": Fixed(DEFAULT),
                "met_logzsol": Fixed(0.0),
            },
            redshift=tengri.Uniform(0.5, 1.5),
            igm={"type": "inoue"},
            approx=WavePrecomp(igm_fold="exact"),
        )
        model.predict_photometry({"redshift": PROBE_Z})
    assert "node" in str(excinfo.value)


@pytest.mark.parametrize("fold", ["node", "exact"])
def test_both_folds_agree_with_the_integrator_where_transmission_is_flat(ssp, observation, fold):
    """The control. No break in this band, so the folds cannot differ here.

    If this fails the disagreement below is not about the fold at all.
    """
    control = CONTROL

    truth = _photometry(_bare_stellar(ssp, observation, None))
    folded = _photometry(_bare_stellar(ssp, observation, WavePrecomp(igm_fold=fold)))

    relative = abs(folded[control] - truth[control]) / truth[control]
    assert relative < 1e-3, f"{fold} fold is {relative:.3%} off in a flat band"


def test_exact_fold_beats_the_node_fold_where_the_break_is_inside_the_band(ssp, observation):
    """The claim the feature exists to make, measured against the integrator.

    `approx=None` is the dense wavelength-grid path and is the reference. In a
    band containing the Lyman break the node fold forms <S><T> and the exact
    fold <S*T>; only the latter is the quantity the integrator computes.
    """
    straddling = STRADDLING

    truth = _photometry(_bare_stellar(ssp, observation, None))[straddling]
    node = _photometry(_bare_stellar(ssp, observation, WavePrecomp(igm_fold="node")))[straddling]
    exact = _photometry(_bare_stellar(ssp, observation, WavePrecomp(igm_fold="exact")))[straddling]

    node_error = abs(node - truth) / truth
    exact_error = abs(exact - truth) / truth

    # The two folds must actually differ here, or the band does not exercise
    # the feature and the comparison below is vacuous.
    assert abs(node - exact) / truth > 1e-5, (
        f"{STRADDLING_BAND} at z={PROBE_Z} does not separate the folds "
        f"(node vs exact {abs(node - exact) / truth:.3e}); the probe is vacuous"
    )
    assert exact_error < node_error, (
        f"exact fold {exact_error:.3%} is not closer to the integrator than "
        f"node fold {node_error:.3%} in {STRADDLING_BAND} at z={PROBE_Z}"
    )
