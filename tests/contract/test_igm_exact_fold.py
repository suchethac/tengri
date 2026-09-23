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


#: Node count for the two free-redshift builds below. Their subject is how the
#: fold is resolved when the redshift is free, not the z table's resolution, and
#: a default-resolution table cost ~45 s per build locally, which put the contract
#: shard past its 70-minute CI budget. Coarse is enough to have a table at all.
FREE_Z_NODES = 8


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
            approx=WavePrecomp(igm_fold="exact", n_z=FREE_Z_NODES),
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


def test_more_quadrature_nodes_cannot_substitute_for_the_exact_fold(ssp, observation):
    """Raising K converges the quadrature to the node fold's answer, not the
    integrator's.

    The sub-band quadrature order and the IGM fold are separate error axes. In
    a band whose transmission is structured, the fold sets a floor that no
    number of quadrature nodes crosses, because the nodes converge on
    <S><T> rather than <S*T>. Measured at z=0.5 across ten bands: K=5 exact
    reached 6.9e-4 in GALEX NUV where K=8 node reached only 1.2e-3.

    Without this, someone reading "converges as 1/K^2" reasonably concludes
    that a stubborn band just needs more nodes, and pays build time for an
    error that is not quadrature error.
    """
    truth = _photometry(_bare_stellar(ssp, observation, None))[STRADDLING]
    coarse_exact = _photometry(
        _bare_stellar(
            ssp,
            observation,
            WavePrecomp(band_integration="quadrature", n_subbands=5, igm_fold="exact"),
        )
    )[STRADDLING]
    fine_node = _photometry(
        _bare_stellar(
            ssp,
            observation,
            WavePrecomp(band_integration="quadrature", n_subbands=8, igm_fold="node"),
        )
    )[STRADDLING]

    err_coarse_exact = abs(coarse_exact - truth) / truth
    err_fine_node = abs(fine_node - truth) / truth

    assert err_coarse_exact < err_fine_node, (
        f"five exact-fold nodes ({err_coarse_exact:.3e}) should beat eight "
        f"node-fold nodes ({err_fine_node:.3e}) in {STRADDLING_BAND}: if they do "
        "not, the fold is no longer the floor and this test's premise has moved"
    )


def test_the_exact_fold_is_free_in_a_band_with_no_structure(ssp, observation):
    """It must not perturb bands it has no business touching.

    Across ten bands at z=0.5 only GALEX NUV moved; the other nine were
    bit-identical between the folds. A fold that shifted flat bands too would
    be changing the photometry for some other reason.
    """
    node = _photometry(_bare_stellar(ssp, observation, WavePrecomp(igm_fold="node")))
    exact = _photometry(_bare_stellar(ssp, observation, WavePrecomp(igm_fold="exact")))

    assert node[CONTROL] == exact[CONTROL], (
        f"the folds differ by {abs(node[CONTROL] - exact[CONTROL]) / node[CONTROL]:.3e} "
        f"in {CONTROL_BAND}, which carries no IGM structure"
    )


# ---------------------------------------------------------------------------
# "auto": the exact fold wherever it can be built, the node fold where it cannot.
#
# The exact fold raises for a free redshift and for a transmission carrying free
# parameters, so `"exact"` can never be proposed as the default while it is the
# only way to ask for the exact fold -- a blanket flip would break every free-z
# fit at once. `"auto"` is the mode that could be: it asks for the exact fold
# and accepts the node fold where the exact one is unavailable.
#
# The promise is one-sided and that is the whole risk. `"auto"` must never
# raise where `"node"` would have worked, which means the fall-back has to know
# every precondition the exact fold refuses on. A second, drifting copy of that
# list breaks the promise at run time and only for the configuration that
# happens to trigger it, so both read one predicate.


def test_auto_is_an_accepted_value():
    assert WavePrecomp(igm_fold="auto").igm_fold == "auto"


def test_adding_auto_does_not_move_the_default():
    """A new mode must not change what an unconfigured WavePrecomp does."""
    assert WavePrecomp().igm_fold == "node"


def test_auto_is_not_resolved_at_construction(ssp, observation):
    """The declared value survives on the config object.

    Resolution needs the redshift disposition and the transmission config,
    neither of which a bare ``WavePrecomp()`` has. A config that rewrote
    itself to "node" at construction would resolve every model the same way.
    """
    assert WavePrecomp(igm_fold="auto").igm_fold == "auto"
    model = _bare_stellar(ssp, observation, WavePrecomp(igm_fold="auto"))
    assert model._approx_config_wave.igm_fold == "auto"


def test_auto_takes_the_exact_fold_at_a_fixed_redshift(ssp, observation):
    """Where the exact fold is available, "auto" must actually use it."""
    auto = _photometry(_bare_stellar(ssp, observation, WavePrecomp(igm_fold="auto")))
    exact = _photometry(_bare_stellar(ssp, observation, WavePrecomp(igm_fold="exact")))
    node = _photometry(_bare_stellar(ssp, observation, WavePrecomp(igm_fold="node")))

    np.testing.assert_array_equal(auto, exact)
    # Without this the assertion above passes vacuously whenever the exact
    # fold happens to equal the node fold -- including if it never ran.
    assert auto[STRADDLING] != node[STRADDLING], (
        "auto is bit-identical to the node fold in the band that straddles the "
        "break, so it did not resolve to the exact fold"
    )


def _free_z_photometry(ssp, observation, fold):
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
        approx=WavePrecomp(igm_fold=fold, n_z=FREE_Z_NODES),
    )
    return np.asarray(model.predict_photometry({"redshift": PROBE_Z}), dtype=np.float64)


def test_auto_falls_back_to_the_node_fold_for_a_free_redshift(ssp, observation):
    """The precondition that makes "exact" unusable as a default.

    ``test_a_free_redshift_refuses_the_exact_fold`` pins that ``"exact"``
    raises here. ``"auto"`` must build the same model and return the node
    answer: not raise, and not quietly return a third thing.
    """
    np.testing.assert_array_equal(
        _free_z_photometry(ssp, observation, "auto"),
        _free_z_photometry(ssp, observation, "node"),
    )


def _patchy_model(ssp, observation, fold):
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
        redshift=Fixed(PROBE_Z),
        igm={"type": "inoue", "patchy": True},
        approx=WavePrecomp(igm_fold=fold),
    )


def test_a_free_transmission_refuses_the_exact_fold(ssp, observation):
    """The other precondition, pinned as a refusal before it is paired."""
    with pytest.raises(ValueError) as excinfo:
        _patchy_model(ssp, observation, "exact").predict_photometry({})
    assert "node" in str(excinfo.value)


def test_auto_falls_back_to_the_node_fold_for_a_free_transmission(ssp, observation):
    """Paired with the refusal above: same configuration, no raise."""
    auto = _photometry(_patchy_model(ssp, observation, "auto"))
    node = _photometry(_patchy_model(ssp, observation, "node"))
    np.testing.assert_array_equal(auto, node)


def test_the_refusals_and_the_fall_back_read_one_list_of_preconditions():
    """A third refusal must not make "auto" raise.

    ``"auto"`` promises to fall back wherever the exact fold cannot be built.
    That promise is only as good as the fall-back's knowledge of what the
    exact fold refuses on, and a refusal added to the exact fold alone would
    break it silently -- for one configuration, at run time, with no
    diagnostic. Both sides call ``_exact_fold_blocker``, so there is one list
    rather than two that can drift.
    """
    import ast
    import inspect
    import textwrap

    from tengri.forward import sed_model

    exact_src = inspect.getsource(sed_model._fold_igm_exact_into_subbands)
    resolver_src = inspect.getsource(sed_model._resolve_igm_fold)

    assert "_exact_fold_blocker(" in exact_src, "the exact fold does not consult the predicate"
    assert "_exact_fold_blocker(" in resolver_src, "the resolver does not consult the predicate"

    raises = [
        n for n in ast.walk(ast.parse(textwrap.dedent(exact_src))) if isinstance(n, ast.Raise)
    ]
    assert len(raises) == 1, (
        f"the exact fold raises from {len(raises)} sites, but only the one "
        "driven by _exact_fold_blocker is mirrored by the auto fall-back; the "
        f"others are at lines {[n.lineno for n in raises]} of the function"
    )


def test_auto_falls_back_when_the_exact_fold_would_do_nothing_at_all():
    """The second reason to fall back, and the one no exception announces.

    Without templates or filters the exact fold returns the state untouched,
    while the node fold still applies a fold. Resolving to "exact" there would
    silently skip work the node path does -- not a refusal, not a warning, just
    a missing fold. Asserted on the resolver directly because the condition is
    about arguments the dispatcher receives, not about any model's physics.
    """
    from tengri.forward.sed_model import _resolve_igm_fold

    class _NoBlocker:
        config = None

    comp, state = _NoBlocker(), object()

    assert _resolve_igm_fold("auto", comp, state, ssp_data=object(), filters=("f",)) == "exact"
    assert _resolve_igm_fold("auto", comp, state, ssp_data=None, filters=("f",)) == "node"
    assert _resolve_igm_fold("auto", comp, state, ssp_data=object(), filters=()) == "node"


def test_a_named_fold_is_never_downgraded_by_the_resolver():
    """ "node" and "exact" pass through untouched, whatever the configuration.

    The resolver exists to serve ``"auto"``. If it also rewrote an explicit
    ``"exact"``, a caller who asked for the exact fold by name would silently
    receive the node answer -- the substitution this seam exists to prevent.
    """
    from tengri.forward.sed_model import _resolve_igm_fold

    class _Blocked:
        class config:
            igm_patchy = True
            use_dla = False

    comp, state = _Blocked(), object()

    assert _resolve_igm_fold("exact", comp, state, ssp_data=None, filters=()) == "exact"
    assert _resolve_igm_fold("node", comp, state, ssp_data=object(), filters=("f",)) == "node"
