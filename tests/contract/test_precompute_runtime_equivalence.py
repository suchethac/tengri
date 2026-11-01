# SPDX-License-Identifier: BSD-3-Clause
"""WavePrecomp LUT <-> exact-path equivalence for radio, AGN disc and analytic dust-IR emission.

What is compared
----------------
Each case builds two SEDModels from the same ``SEDModel.build`` grammar on the real
wNE SSP (``data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5``, which covers
91 A - 1e8 A): ``approx=None`` (exact wave-grid path) and ``approx=WavePrecomp()``
(LUT). Both predict photometry on one drawn parameter vector. Each case asserts that
the LUT engaged (``model.approx.wave_precomp``) and that the exact path did not.

Only bands where the component under test adds at least 5% over a reference build
without it are compared. A case with no such band fails: it would compare the stellar
SED with itself.

Measured worst differences on the real grid, per case, with tolerances, are in the
comments above each test.

Cases not compared
------------------
* ``pah_drude``: the building block is rejected as a sole dust emitter
  (``ParameterError``, energy balance); it has no compliant standalone build.
The X-ray, silva04 and cat3d_wind cases use ``agn_log_lbol=12.0`` (prior 6.42-12.42)
and compare the mid-IR or X-ray bands they dominate. The shock case uses
``shock_log_lhalpha=43.0`` under ``norm='lhalpha'`` (prior 30-46); at the top of the
``shock_frac`` prior (1.0) shock adds only 4% per band, below the guard.

History
-------
The previous version built two identical exact models (``precompute=`` was a no-op),
so every assertion compared the exact path with itself. The shock test in
``test_nebular_shock_kernel_wiring.py`` had the same defect.
"""

from __future__ import annotations

import importlib
import itertools

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._data_skip import DATA_DIR as _REPO_DATA  # root computed once; see #1431
from tests.contract._heap import release_heap

pytestmark = pytest.mark.contract


def _assert_equivalent(precomp, runtime, label, *, rtol=1e-3):
    """Compare two photometry vectors with a floor set by the runtime signal.

    Used by the grid-torus equivalence tests below. The runtime vector must not be
    identically zero, otherwise nothing is being compared; the absolute floor is
    ``1e-10`` of its largest element.
    """
    runtime = np.asarray(runtime)
    precomp = np.asarray(precomp)
    scale = float(np.max(np.abs(runtime)))

    assert scale > 0.0, (
        f"{label}: the runtime photometry is identically zero, so this "
        f"comparison cannot distinguish a working precompute path from a "
        f"broken one. The band is probably off the SSP grid."
    )
    chex.assert_trees_all_close(
        precomp, runtime, rtol=rtol, atol=1e-10 * scale, custom_message=label
    )


_REAL_SSP = _REPO_DATA / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

#: A component must add at least this fraction over the reference build to be compared.
_DOMINANCE = 0.05


def _dust_att():
    """Two-component Calzetti attenuation shared by the radio and dust-IR cases."""
    from tengri import DEFAULT, Fixed

    return {
        "type": "two_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
        "tau_diff": 0.5,
    }


@pytest.fixture(scope="module")
def real_ssp():
    """The real wNE SSP, loaded once per module; skipped if the file is absent."""
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    if not _REAL_SSP.is_file():
        pytest.skip(f"real SSP grid not found: {_REAL_SSP}")
    return load_ssp_data(str(_REAL_SSP))


def _tophat_observation(centers, frac):
    """Top-hat filters centered on ``centers`` [A], each ``frac`` wide on each side."""
    import jax.numpy as jnp

    from tengri import Observation, Photometry
    from tengri.observation.photometry import FilterCurve

    curves = []
    for c in centers:
        wv = jnp.linspace(c * (1.0 - frac), c * (1.0 + frac), 48)
        tr = jnp.sin(jnp.linspace(0.0, jnp.pi, 48)) * 0.6
        curves.append(FilterCurve(wave=wv, trans=tr, name=f"b{int(c)}"))
    return Observation(photometry=Photometry(filters=tuple(curves)))


def _assert_lut_matches_exact(ssp, observation, component_on, component_off, *, rtol, label):
    """Compare the LUT with the exact path on the bands the component dominates.

    ``component_on`` and ``component_off`` are ``SEDModel.build`` keyword groups.
    The reference build ``component_off`` gives the band-dominance denominator, as in
    ``test_dust_ir_emission_precompute_parity.py``.
    """
    from tengri import DEFAULT, Fixed, SEDModel, WavePrecomp

    common = dict(
        ssp_data=ssp,
        observation=observation,
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
    )
    exact = SEDModel.build(approx=None, **{**common, **component_on})
    lut = SEDModel.build(approx=WavePrecomp(), **{**common, **component_on})
    reference = SEDModel.build(approx=None, **{**common, **component_off})

    assert lut.approx.wave_precomp, f"{label}: WavePrecomp did not engage"
    assert not exact.approx.wave_precomp, f"{label}: approx=None engaged the LUT"

    params = dict(exact.spec.sample(jax.random.PRNGKey(42)))
    phot_exact = np.asarray(exact.predict_photometry(params))
    phot_lut = np.asarray(lut.predict_photometry(params))
    phot_ref = np.asarray(reference.predict_photometry(params))

    share = np.abs(phot_exact - phot_ref) / np.maximum(np.abs(phot_ref), 1e-30)
    dominant = share >= _DOMINANCE
    assert dominant.any(), (
        f"{label}: the component adds at least 5% to no band (max {share.max():.2e}); "
        "the comparison would test the stellar SED alone"
    )
    rel = np.abs(phot_lut - phot_exact)[dominant] / np.maximum(np.abs(phot_exact)[dominant], 1e-30)
    assert rel.max() < rtol, (
        f"{label}: LUT vs exact max relative difference {rel.max():.3e} exceeds {rtol:.0e} "
        f"on the dominant bands (per band: {rel})"
    )


# ── Radio ─────────────────────────────────────────────────────────────────


def test_radio_bell2003_lut_matches_exact(real_ssp):
    """Radio synchrotron (bell2003 SFR channel): LUT matches exact on the radio bands.

    Bands 1e6, 1e7, 1e8 A. Measured on the real grid: max relative difference 2.0e-3
    on the two dominant bands (the 1e6 A band adds only 7e-5 and is excluded). Tolerance
    6e-3 = 3 x measured, rounded up.
    """
    obs = _tophat_observation([1e6, 1e7, 1e8], frac=0.18)
    on = dict(dust_attenuation=_dust_att(), radio={"sf": {"type": "bell2003"}})
    off = dict(
        dust_attenuation=_dust_att(), radio={"sf": {"type": "none"}, "agn": {"type": "none"}}
    )
    _assert_lut_matches_exact(real_ssp, obs, on, off, rtol=6e-3, label="radio bell2003")


# ── AGN disc ──────────────────────────────────────────────────────────────


def test_qsogen_disc_lut_matches_exact(real_ssp):
    """QSOgen AGN disc: LUT matches exact on UV-to-IR optical bands.

    Measured on the real grid: max relative difference 1.5e-14 over all five bands,
    each of which the disc dominates (>= 14%). Tolerance 5e-14 = 3 x measured, rounded up.
    """
    obs = _tophat_observation([3500.0, 4800.0, 6200.0, 7600.0, 9000.0], frac=0.18)
    on = dict(agn={"type": "composable", "disc": {"type": "qsogen"}})
    off = dict(agn={"type": "none"})
    _assert_lut_matches_exact(real_ssp, obs, on, off, rtol=5e-14, label="qsogen disc")


# ── X-ray, AGN torus and shock, at a raised luminosity ──────────────────────


def test_xray_simple_lut_matches_exact(real_ssp):
    """X-ray emission (simple) at ``agn_log_lbol=12.0``: LUT matches exact.

    The declared prior on ``agn_log_lbol`` is 6.42-12.42. Dominant bands are the 12.4 A
    and 1.24 A filters (share 7.3 and 1.1); the 124 A band adds 0.9% and is excluded.
    Measured worst relative difference 3.2e-15; tolerance 1e-14 = 3 x measured, rounded up.
    """
    from tengri import Fixed

    obs = _tophat_observation([124.0, 12.4, 1.24], frac=0.18)
    disc = {"type": "composable", "disc": {"type": "qsogen"}, "agn_log_lbol": Fixed(12.0)}
    _assert_lut_matches_exact(
        real_ssp,
        obs,
        dict(agn=disc, xray={"type": "simple"}),
        dict(agn=disc),
        rtol=1e-14,
        label="xray simple",
    )


# Measured mid-IR (5, 10, 20 um) at agn_log_lbol=12.0, all three bands dominant.
_TORUS_TOLERANCE = {
    # silva04 measured 4.5e-8 -> 3x = 1.4e-7 -> 2e-7
    "silva04": 2e-7,
    # cat3d_wind measured 4.7e-8 -> 3x = 1.4e-7 -> 2e-7
    "cat3d_wind": 2e-7,
}
_TORUS_GRID = {"silva04": "silva04_torus_grid.h5", "cat3d_wind": "cat3d_wind_torus_grid.h5"}


@pytest.mark.parametrize(
    "torus",
    [
        pytest.param(
            name,
            marks=pytest.mark.skipif(
                not (_REPO_DATA / _TORUS_GRID[name]).exists(),
                reason=f"{_TORUS_GRID[name]} not present",
            ),
        )
        for name in sorted(_TORUS_TOLERANCE)
    ],
)
def test_torus_lut_matches_exact(real_ssp, torus):
    """Torus template at ``agn_log_lbol=12.0``: LUT matches exact on mid-IR bands.

    Bands at 5, 10 and 20 um, where the torus dominates the stellar continuum by more
    than an order of magnitude. Tolerance per torus is in ``_TORUS_TOLERANCE``.
    """
    from tengri import Fixed

    obs = _tophat_observation([5e4, 1e5, 2e5], frac=0.18)
    on = dict(agn={"type": "composable", "torus": {"type": torus}, "agn_log_lbol": Fixed(12.0)})
    off = dict(agn={"type": "none"})
    _assert_lut_matches_exact(
        real_ssp, obs, on, off, rtol=_TORUS_TOLERANCE[torus], label=f"{torus} torus"
    )


@pytest.mark.skipif(
    not (_REPO_DATA / "mappings_templates.h5").exists(),
    reason="data/mappings_templates.h5 not present",
)
def test_shock_lhalpha_lut_matches_exact(real_ssp):
    """MAPPINGS shock emission at absolute ``shock_log_lhalpha=43.0``: LUT matches exact.

    ``norm='lhalpha'`` with the declared prior 30-46 on ``shock_log_lhalpha``. Dominant on
    all five optical bands (share 0.7-11). ``shock_frac`` cannot reach the guard: at its
    prior maximum of 1.0 shock adds 4% per band.
    Measured worst relative difference 9.6e-15; tolerance 3e-14 = 3 x measured, rounded up.
    """
    from tengri import Fixed

    obs = _tophat_observation([3500.0, 4800.0, 6200.0, 7600.0, 9000.0], frac=0.18)
    on = dict(
        neb={"type": "ssp"},
        shock={
            "norm": "lhalpha",
            "log_lhalpha": Fixed(43.0),
            "velocity": Fixed(300.0),
            "log_density": Fixed(0.0),
            "b_over_sqrt_n": Fixed(1.0),
        },
    )
    off = dict(neb={"type": "ssp"}, shock={"type": "none"})
    _assert_lut_matches_exact(real_ssp, obs, on, off, rtol=3e-14, label="shock lhalpha")


# ── Analytic dust-IR emission ─────────────────────────────────────────────
#
# Far-IR bands (50, 100, 200 um). At 2-10 um the modified blackbody and graybody add
# about 2e-6 of the flux, so the band guard excludes every band there.


@pytest.mark.skipif(
    not (_REPO_DATA / "silva04_torus_grid.h5").exists(),
    reason="Silva+04 torus grid not available; build via scripts/build_silva04_grid.py.",
)
@pytest.mark.parametrize("where", ["node", "off_node"])
def test_silva04_precompute_matches_exact_pchip(where) -> None:
    """``silva04_precompute.py``'s LUT agrees with the exact path at a node AND off-node.

    Regression test for task3 fix round 1, RULING R11: ``silva04_precompute.py``
    used triweight (a C²-smooth *smoother*) while the exact path (``silva04.py``)
    had already migrated to PCHIP (node-exact, C¹), so the two paths disagreed by
    construction -- most sharply at the grid's boundary nodes (~9% shape smear,
    ``torus.md`` D2), which this test targets directly.

    ``TestSilva04PrecomputeEquivalence.test_silva04`` above goes through a full
    ``SEDModel(precompute=True)``, which does NOT actually route this legacy
    monolithic ``agn_model='silva04'`` selection through
    ``silva04_precompute.py`` at all (checked directly: instrumenting
    ``precompute_silva04_photometry`` to print on entry, it is never called
    while building that test's ``model_precomp``) -- so that test's precompute
    and runtime models are silently both computing the exact path, and it
    cannot distinguish a working precompute LUT from a broken one, regardless
    of interpolation kernel. This test instead calls
    :func:`~tengri.components.agn.silva04_precompute.precompute_silva04_photometry`
    / :func:`~tengri.components.agn.silva04_precompute.build_silva04_photometry_lookup`
    directly, so it genuinely exercises the code RULING R11 migrated.

    Tolerance: measured directly against the fixed (both-PCHIP) code at the
    grid's two boundary nodes and the two adjacent off-node midpoints, the
    precompute-vs-exact ratio differs from 1 by at most ~0.25% -- a small,
    kernel-independent baseline (present identically at every node tested,
    so it is a filter-integration convention difference between this test's
    own reference quadrature and ``preintegrate_grid``'s, not an interpolation
    artifact). Reintroducing triweight in ``silva04_precompute.py`` alone
    (this test's mutant) pushes the boundary-node ratio to ~1.4% -- comfortably
    past the 0.5% threshold below, which stays comfortably above the measured
    baseline.
    """
    import h5py

    from tengri.components.agn.silva04 import load_silva04_grid, silva04_sed_from_grid
    from tengri.components.agn.silva04_precompute import (
        build_silva04_photometry_lookup,
        precompute_silva04_photometry,
    )

    grid_path = str(_REPO_DATA / "silva04_torus_grid.h5")
    with h5py.File(grid_path, "r") as f:
        log_nh_axis = np.asarray(f["silva04"]["log_nh_axis"][:], dtype=np.float64)
    grid = load_silva04_grid(grid_path)

    # Boundary nodes: where the triweight-vs-PCHIP kernel difference is
    # largest (torus.md D2); an interior node's difference is too small to
    # reliably separate from the baseline below. The off-node point sits 90%
    # of the way from the second-to-last to the last node -- close to, but
    # not on, the boundary node; the exact midpoint measured a smaller
    # triweight-vs-PCHIP gap (triweight's neighbor-averaging happens to be
    # closer to correct near a bin's center than near its edge).
    if where == "node":
        log_nh_value = float(log_nh_axis[-1])
    else:
        log_nh_value = float(log_nh_axis[-2] + 0.9 * (log_nh_axis[-1] - log_nh_axis[-2]))

    filter_waves = [np.geomspace(1.0e5, 1.0e6, 300), np.geomspace(1.0e6, 1.0e7, 300)]
    filter_trans = [np.ones(300), np.ones(300)]

    precomp = precompute_silva04_photometry(grid_path, filter_waves, filter_trans, redshift=0.0)
    lookup = build_silva04_photometry_lookup(precomp)
    phot_precomp = np.asarray(lookup(0.0, log_nh_value, 1.0))

    # Reference: the exact SED (silva04.py), evaluated on the template's own
    # native grid (no intermediate resample), integrated through the SAME
    # filters with the same BESSELL (1/lambda) weight
    # preintegrate_grid uses (see its docstring).
    wave = grid.wave_grid
    sed = np.asarray(
        silva04_sed_from_grid(
            grid,
            jax.numpy.asarray(wave),
            agn_log_lbol=0.0,
            agn_log_nh_silva=log_nh_value,
            agn_torus_frac=1.0,
        )
    )
    phot_exact = np.array(
        [
            np.trapezoid(np.interp(fw, wave, sed) * ft / fw, fw) / np.trapezoid(ft / fw, fw)
            for fw, ft in zip(filter_waves, filter_trans)
        ]
    )

    chex.assert_trees_all_close(
        phot_precomp,
        phot_exact,
        rtol=5.0e-3,
        atol=0.0,
        custom_message=f"silva04 precompute vs exact at log_nh_silva={log_nh_value:.4f} ({where})",
    )


# Measured on the real grid, far-IR bands. Tolerance = 3 x measured worst case, rounded up
# to one significant figure.
_FIR_CENTERS = [5e5, 1e6, 2e6]  # 50, 100, 200 um

_DUST_IR_TOLERANCE = {
    # measured 6.2e-8 -> 3x = 1.8e-7 -> 2e-7
    "dale2014": 2e-7,
    # measured 2.0e-6 -> 3x = 6.0e-6 -> 6e-6
    "modified_blackbody": 6e-6,
    # measured 3.8e-6 -> 3x = 1.1e-5 -> 2e-5
    "casey2012": 2e-5,
    # measured 6.0e-6 -> 3x = 1.8e-5 -> 2e-5
    "graybody": 2e-5,
}


@pytest.mark.parametrize("emitter", sorted(_DUST_IR_TOLERANCE))
def test_dust_ir_emitter_lut_matches_exact(real_ssp, emitter):
    """Analytic dust-IR emitter on the far-IR bands: LUT matches exact.

    The energy balance ties ``L_ir`` to the absorbed luminosity, so the emitter is
    compared with the two-component attenuation it sits behind. Only the far-IR bands
    are compared: at 2-10 um this emitter adds too little to clear the 5% guard.
    """
    from tengri import DEFAULT, Fixed

    obs = _tophat_observation(_FIR_CENTERS, frac=0.18)
    on = dict(
        dust_attenuation=_dust_att(),
        dust_emission={"type": emitter, "all_params": Fixed(DEFAULT)},
    )
    off = dict(dust_attenuation=_dust_att(), dust_emission={"type": "none"})
    _assert_lut_matches_exact(
        real_ssp,
        obs,
        on,
        off,
        rtol=_DUST_IR_TOLERANCE[emitter],
        label=f"{emitter} dust-IR (far-IR bands)",
    )


# pah_drude is not compared. Built as the sole dust emitter it raises ParameterError:
# it carries the aromatic-feature forest with no thermal continuum and fails the
# energy-balance check. It has no compliant standalone build to test.


# ── Grid-torus adapter precompute↔runtime equivalence (fix round 1) ──────
#
# Directly exercises the seven ``PrecomputeModule``-shaped torus adapters
# (five new AGNfitter-rX reductions + the two siblings whose pattern they
# mirror) rather than going through ``SEDModel``: the composable AGN runner's
# own WavePrecomp path (``composable_agn.blocks.composable_precompute``)
# re-evaluates the exact runner on an outer-product grid and never calls
# these adapters at all, and none of the five new reductions (nor
# ``nenkova_agnfitter``) has a monolithic ``agn_model=`` name, so building a
# full SEDModel through either surface would silently test something else.
# Calling ``module.precompute()`` / ``module.build_lookup()`` directly is the
# only way to test what these adapters actually promise (the
# ``PrecomputeModule`` Protocol contract), and is what the monkeypatch-to-
# raise proof below establishes: the test provably drives this exact call,
# not a bypass.
#
# "Exact" reference: the runtime ``*_sed_from_grid`` function (the same one
# the composable torus block calls) evaluated at the point under test on a model
# wavelength grid, then filter-integrated via
# :func:`tengri.observation.photometry.lnu_filter_integral`. The model grid is the
# template's native nodes plus every filter node in the rest frame
# (``_runtime_wave``): the runtime resamples the template log-log onto it, as onto any
# model grid, and the filter integral then sees the SED at its own nodes. A reference
# on the native grid alone integrates linearly between the coarse template
# wavelengths, the same quadrature the tables used to be built with, so it could not
# see those tables sit 5-33% off the runtime at their own nodes (#3160).
#
# Tolerance, measured (not guessed): at an exact grid node neither path
# interpolates, so agreement is rounding (measured worst case over every node of all
# nine libraries, at z = 0 and 1.2 with irregular filters, in
# ``test_torus_table_nodes_match_runtime``) -- rtol=1e-10.
# Off-node, the two paths do not commute (precompute interpolates
# NODE-PREINTEGRATED PHOTOMETRY via PCHIP; the exact path integrates the
# TEMPLATE interpolated in wavelength space, then integrates through the
# filter) so a real, larger residual is expected, held to the 5e-2 contract.
#
# Bug found by this table (fixed in this round, not merely tested around):
# ``skirtor_agnfitter{,_1p,_2p}_precompute.py`` each multiplied by
# ``_LSUN_ERG`` *twice* -- once folded into ``grid_phot`` at the precompute
# stage (correct) and again in ``l_scale`` at lookup time (double-counted) --
# over-scaling every SKIRTOR-family precompute photometry by exactly
# ``_LSUN_ERG`` (~3.83e33). The three ``l_scale = ... * _LSUN_ERG * ...``
# lines are now ``l_scale = ... * ...``; see the fix-round-1 report for the
# RED run this produced before the fix.

from tengri.observation.photometry import lnu_filter_integral


def _grid_torus_specs() -> list[dict]:
    """One entry per grid-torus adapter: (name, modules, loader/sed_fn names,
    grid path, axis kwarg names in ``grid.axes`` order, and ``deg_axes``: the axes
    the runtime grid stores as an inclination in degrees while the kwarg is its
    cosine, ``agn_cos_inc``)."""
    return [
        dict(
            name="nenkova_agnfitter_2p",
            runtime_mod="tengri.components.agn.nenkova_agnfitter_2p",
            precompute_mod="tengri.components.agn.nenkova_agnfitter_2p_precompute",
            loader="load_nenkova_agnfitter_2p_grid",
            sed_fn="nenkova_agnfitter_2p_sed_from_grid",
            grid_path=_REPO_DATA / "nenkova_agnfitter_2p_torus_grid.h5",
            axis_kwargs=("agn_cos_inc", "agn_oa_nenkova"),
        ),
        dict(
            name="nenkova_agnfitter_3p",
            runtime_mod="tengri.components.agn.nenkova_agnfitter_3p",
            precompute_mod="tengri.components.agn.nenkova_agnfitter_3p_precompute",
            loader="load_nenkova_agnfitter_3p_grid",
            sed_fn="nenkova_agnfitter_3p_sed_from_grid",
            grid_path=_REPO_DATA / "nenkova_agnfitter_3p_torus_grid.h5",
            axis_kwargs=("agn_cos_inc", "agn_oa_nenkova", "agn_tv_nenkova"),
        ),
        dict(
            name="skirtor_agnfitter_1p",
            runtime_mod="tengri.components.agn.skirtor_agnfitter_1p",
            precompute_mod="tengri.components.agn.skirtor_agnfitter_1p_precompute",
            loader="load_skirtor_agnfitter_1p_grid",
            sed_fn="skirtor_agnfitter_1p_sed_from_grid",
            grid_path=_REPO_DATA / "skirtor_mean1p_torus_grid.h5",
            axis_kwargs=("agn_cos_inc",),
            deg_axes=(0,),
        ),
        dict(
            name="skirtor_agnfitter_2p",
            runtime_mod="tengri.components.agn.skirtor_agnfitter_2p",
            precompute_mod="tengri.components.agn.skirtor_agnfitter_2p_precompute",
            loader="load_skirtor_agnfitter_2p_grid",
            sed_fn="skirtor_agnfitter_2p_sed_from_grid",
            grid_path=_REPO_DATA / "skirtor_mean2p_torus_grid.h5",
            axis_kwargs=("agn_oa_skirtor", "agn_cos_inc"),
            deg_axes=(1,),
        ),
        dict(
            name="cat3d_wind_lowfwd",
            runtime_mod="tengri.components.agn.cat3d_wind_lowfwd",
            precompute_mod="tengri.components.agn.cat3d_wind_lowfwd_precompute",
            loader="load_cat3d_wind_lowfwd_grid",
            sed_fn="cat3d_wind_lowfwd_sed_from_grid",
            grid_path=_REPO_DATA / "cat3d_wind_lowfwd_torus_grid.h5",
            axis_kwargs=("agn_cos_inc", "agn_a_cat3d_lowfwd", "agn_fwd_cat3d_lowfwd"),
        ),
        dict(
            name="nenkova_agnfitter",
            runtime_mod="tengri.components.agn.nenkova_agnfitter",
            precompute_mod="tengri.components.agn.nenkova_agnfitter_precompute",
            loader="load_nenkova_agnfitter_grid",
            sed_fn="nenkova_agnfitter_sed_from_grid",
            grid_path=_REPO_DATA / "nenkova_agnfitter_torus_grid.h5",
            axis_kwargs=("agn_cos_inc",),
        ),
        dict(
            name="skirtor_agnfitter",
            runtime_mod="tengri.components.agn.skirtor_agnfitter",
            precompute_mod="tengri.components.agn.skirtor_agnfitter_precompute",
            loader="load_skirtor_agnfitter_grid",
            sed_fn="skirtor_agnfitter_sed_from_grid",
            grid_path=_REPO_DATA / "skirtor_mean3p_torus_grid.h5",
            axis_kwargs=("agn_oa_skirtor", "agn_cos_inc", "agn_tv_skirtor"),
            deg_axes=(1,),
        ),
    ]


_GRID_TORUS_SPECS = _grid_torus_specs()
_GRID_TORUS_IDS = [s["name"] for s in _GRID_TORUS_SPECS]


def _node_and_offnode(grid):
    """Middle-node axis values, and one off-node value per axis (the midpoint
    to an adjacent node, so it genuinely requires interpolation on both
    paths)."""
    node_idx = tuple(len(ax) // 2 for ax in grid.axes)
    node_vals = tuple(float(grid.axes[i][node_idx[i]]) for i in range(len(grid.axes)))
    offnode_vals = []
    for i, ax in enumerate(grid.axes):
        j = node_idx[i]
        if j + 1 < len(ax):
            offnode_vals.append(float((ax[j] + ax[j + 1]) / 2.0))
        elif j > 0:
            offnode_vals.append(float((ax[j] + ax[j - 1]) / 2.0))
        else:
            offnode_vals.append(float(ax[j]))
    return node_vals, tuple(offnode_vals)


def _kwarg_values(spec, values):
    """Runtime-grid axis values as the kwargs take them: ``cos`` of a degree inclination."""
    deg_axes = spec.get("deg_axes", ())
    return tuple(
        float(np.cos(np.radians(v))) if i in deg_axes else v for i, v in enumerate(values)
    )


def _runtime_wave(grid, waves, redshift=0.0):
    """Model wavelength grid of the reference [Angstrom]: the template's native nodes and
    every filter node in the rest frame."""
    rest = [np.asarray(w, dtype=float) / (1.0 + redshift) for w in waves]
    return jnp.asarray(np.unique(np.concatenate([np.asarray(grid.wave_grid, float), *rest])))


def _exact_grid_torus_photometry(sed_fn, grid, axis_kwargs, values, waves, trans):
    """Filter-integrated photometry from the runtime SED function directly."""
    kwargs = dict(zip(axis_kwargs, values, strict=True))
    wave = _runtime_wave(grid, waves)
    L_nu = sed_fn(grid, wave, agn_log_lbol=0.0, agn_torus_frac=1.0, **kwargs)
    out = [
        float(lnu_filter_integral(L_nu, wave, jnp.asarray(w), jnp.asarray(t), 0.0))
        for w, t in zip(waves, trans, strict=True)
    ]
    return np.asarray(out)


def _precompute_grid_torus_photometry(
    precompute_mod, grid_path, axis_kwargs, values, waves, trans
):
    """Filter-integrated photometry via the adapter's own Protocol entry points."""
    preint = precompute_mod.precompute(
        list(waves), list(trans), 0.0, None, grid_path=str(grid_path)
    )
    lookup = precompute_mod.build_lookup(preint)
    kwargs = dict(zip(axis_kwargs, values, strict=True))
    return np.asarray(lookup(agn_log_lbol=0.0, agn_torus_frac=1.0, **kwargs))


@pytest.fixture(scope="module")
def grid_torus_filters():
    """UV-optical-IR filters, reused from :func:`agn_torus_filter_set`'s
    construction (module-scoped here: shared across every (block, point)
    case, cheap and immutable)."""
    centers = np.array([1500.0, 5500.0, 12000.0, 25000.0, 100000.0])
    widths = np.array([300.0, 1000.0, 2000.0, 5000.0, 20000.0])
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(c - 3 * w, c + 3 * w, 64)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)
    return waves, trans


@pytest.mark.parametrize("spec", _GRID_TORUS_SPECS, ids=_GRID_TORUS_IDS)
class TestGridTorusPrecomputeRuntimeEquivalence:
    """Precompute↔runtime equivalence for the seven grid-torus adapters.

    Five new AGNfitter-rX reductions (Task 4) plus the two siblings whose
    ``PrecomputeModule`` pattern they mirror (``nenkova_agnfitter``,
    ``skirtor_agnfitter``). See the module-level comment above this class for
    why these are tested by calling the adapter directly rather than through
    ``SEDModel``, and for the measured tolerances.
    """

    def _skip_if_missing(self, spec):
        if not spec["grid_path"].exists():
            pytest.skip(f"{spec['name']}: grid not found at {spec['grid_path']}")

    def _modules(self, spec):
        runtime_mod = importlib.import_module(spec["runtime_mod"])
        precompute_mod = importlib.import_module(spec["precompute_mod"])
        return runtime_mod, precompute_mod

    def test_node_matches_exact(self, spec, grid_torus_filters):
        """At an exact grid node, precompute and exact photometry agree to
        machine precision (measured worst case 8.6e-15; rtol=1e-10 here)."""
        self._skip_if_missing(spec)
        runtime_mod, precompute_mod = self._modules(spec)
        loader = getattr(runtime_mod, spec["loader"])
        sed_fn = getattr(runtime_mod, spec["sed_fn"])
        grid = loader(str(spec["grid_path"]))
        node_vals = _kwarg_values(spec, _node_and_offnode(grid)[0])
        waves, trans = grid_torus_filters

        exact = _exact_grid_torus_photometry(
            sed_fn, grid, spec["axis_kwargs"], node_vals, waves, trans
        )
        precomp = _precompute_grid_torus_photometry(
            precompute_mod, spec["grid_path"], spec["axis_kwargs"], node_vals, waves, trans
        )
        _assert_equivalent(precomp, exact, f"{spec['name']} node precompute<->exact", rtol=1e-10)

    def test_offnode_matches_exact(self, spec, grid_torus_filters):
        """Off a grid node, the two paths do not commute (interpolate-then-
        integrate vs integrate-then-interpolate); measured worst case 3.45e-2
        (cat3d_wind_lowfwd) -- rtol=5e-2 here."""
        self._skip_if_missing(spec)
        runtime_mod, precompute_mod = self._modules(spec)
        loader = getattr(runtime_mod, spec["loader"])
        sed_fn = getattr(runtime_mod, spec["sed_fn"])
        grid = loader(str(spec["grid_path"]))
        offnode_vals = _kwarg_values(spec, _node_and_offnode(grid)[1])
        waves, trans = grid_torus_filters

        exact = _exact_grid_torus_photometry(
            sed_fn, grid, spec["axis_kwargs"], offnode_vals, waves, trans
        )
        precomp = _precompute_grid_torus_photometry(
            precompute_mod, spec["grid_path"], spec["axis_kwargs"], offnode_vals, waves, trans
        )
        _assert_equivalent(
            precomp, exact, f"{spec['name']} off-node precompute<->exact", rtol=5e-2
        )

    def test_precompute_entry_point_is_actually_driven(
        self, spec, grid_torus_filters, monkeypatch
    ):
        """Vacuity proof: monkeypatch the adapter's own ``precompute`` to
        raise, and confirm the SAME call this test class makes above
        propagates that raise -- i.e. this test suite cannot pass by
        silently comparing two copies of the same exact-path computation, or
        any other bypass of the adapter under test (Task 3 found exactly
        this class of vacuity elsewhere in this file)."""
        self._skip_if_missing(spec)
        _, precompute_mod = self._modules(spec)
        waves, trans = grid_torus_filters

        def _raise(*_args, **_kwargs):
            raise RuntimeError(f"MONKEYPATCH_PROOF: {spec['name']}.precompute was called")

        monkeypatch.setattr(precompute_mod, "precompute", _raise)
        placeholder_vals = (0.0,) * len(spec["axis_kwargs"])
        with pytest.raises(RuntimeError, match="MONKEYPATCH_PROOF"):
            _precompute_grid_torus_photometry(
                precompute_mod,
                spec["grid_path"],
                spec["axis_kwargs"],
                placeholder_vals,
                waves,
                trans,
            )


# ---------------------------------------------------------------------------
# SKIRTOR-AGNfitter adapters: the inclination is interpolated where the runtime
# interpolates it (#3085)
# ---------------------------------------------------------------------------
#
# The runtime kernels read the library's inclination axis in degrees from the polar
# axis, at i = arccos(agn_cos_inc). The precompute adapters must interpolate the same
# variable: a table re-keyed by cos i is node-exact too, but between nodes it is a
# different interpolant (measured: up to 2.6e-3 off the degree read for SKIRTOR_mean_1p,
# 1.3e-2 for SKIRTOR_mean_3p). The single off-node point above cannot see that, so the
# inclination is swept densely here, three points per interval over all nine intervals.
#
# Two checks per adapter:
#
# * the lookup at agn_cos_inc = cos(i) is the 1-D PCHIP in degrees of the
#   preintegrated node table (SciPy's Fritsch-Carlson PCHIP, an independent
#   implementation), to 1e-9 on the interior intervals, with the other axes at their
#   middle node. The mean_2p and mean_3p tables are power-weighted (band channels over
#   a last bolometric-power channel, divided after interpolating), so their reference is
#   the same ratio of SciPy PCHIPs;
# * the precompute-vs-runtime sweep across the type-1/type-2 step is in the section on
#   tables normalized after interpolation, below.

_SKIRTOR_DENSE = {
    "skirtor_agnfitter_1p": 0,
    "skirtor_agnfitter_2p": 1,
    "skirtor_agnfitter": 1,
}
_SKIRTOR_DENSE_SPECS = [s for s in _GRID_TORUS_SPECS if s["name"] in _SKIRTOR_DENSE]
_SKIRTOR_DENSE_IDS = [s["name"] for s in _SKIRTOR_DENSE_SPECS]
# Adapters whose table carries a trailing bolometric-power channel.
_SKIRTOR_POWER_WEIGHTED = frozenset({"skirtor_agnfitter_2p", "skirtor_agnfitter"})


def _incl_fractions(nodes):
    """Three inclinations per interval [deg]."""
    return [a + (b - a) * f for a, b in itertools.pairwise(nodes) for f in (0.25, 0.5, 0.75)]


def _dense_incl_points(spec):
    """(kwarg values, inclination [deg]) for 3 points per interval, others at nodes."""
    runtime_mod = importlib.import_module(spec["runtime_mod"])
    grid = getattr(runtime_mod, spec["loader"])(str(spec["grid_path"]))
    incl_axis = _SKIRTOR_DENSE[spec["name"]]
    incls = _incl_fractions(np.asarray(grid.axes[incl_axis], dtype=float))
    others = [float(np.asarray(ax)[len(ax) // 2]) for ax in grid.axes]
    points = []
    for incl in incls:
        values = list(others)
        values[incl_axis] = float(np.cos(np.radians(incl)))
        points.append((tuple(values), incl))
    return runtime_mod, grid, points


@pytest.mark.parametrize("spec", _SKIRTOR_DENSE_SPECS, ids=_SKIRTOR_DENSE_IDS)
def test_skirtor_agnfitter_precompute_interpolates_inclination_in_degrees(
    spec, grid_torus_filters
):
    """The lookup at cos(i) is the degree-axis PCHIP of the node photometry (1e-9)."""
    from scipy.interpolate import PchipInterpolator

    if not spec["grid_path"].exists():
        pytest.skip(f"{spec['name']}: grid not found at {spec['grid_path']}")
    precompute_mod = importlib.import_module(spec["precompute_mod"])
    waves, trans = grid_torus_filters
    preint = precompute_mod.precompute(
        list(waves), list(trans), 0.0, None, grid_path=str(spec["grid_path"])
    )
    lookup = precompute_mod.build_lookup(preint)
    axes = [np.asarray(a, dtype=float) for a in preint["axes"]]
    incl_axis = _SKIRTOR_DENSE[spec["name"]]
    np.testing.assert_array_equal(axes[incl_axis], np.arange(0.0, 91.0, 10.0))

    _, _, points = _dense_incl_points(spec)
    phot = np.asarray(preint["grid_phot"], dtype=float)
    # The other axes sit at their middle node: index the table there.
    index = [len(a) // 2 for a in axes]
    index[incl_axis] = slice(None)
    profile = phot[tuple(index)]  # (n_incl, n_channels)
    pchip = PchipInterpolator(axes[incl_axis], profile, axis=0)
    if spec["name"] in _SKIRTOR_POWER_WEIGHTED:
        assert profile.shape[-1] == len(waves) + 1

        def reference(incl):
            row = pchip(incl)
            return row[:-1] / row[-1]
    else:
        reference = pchip
    for values, incl in points:
        if not axes[incl_axis][1] <= incl <= axes[incl_axis][-2]:
            continue  # end intervals: end-slope conventions differ between PCHIPs
        kwargs = dict(zip(spec["axis_kwargs"], values, strict=True))
        got = np.asarray(lookup(agn_log_lbol=0.0, agn_torus_frac=1.0, **kwargs))
        np.testing.assert_allclose(got, reference(incl), rtol=1e-9, err_msg=f"i = {incl}")


class _FixedInclination:
    """The slice of ``Parameters`` that ``collapse_fixed_axes`` reads: a Fixed inclination."""

    free_params = ()

    def __init__(self, cos_inc):
        self._cos_inc = cos_inc

    def get_fixed_values(self):
        return {"agn_cos_inc": self._cos_inc}


@pytest.mark.parametrize("spec", _SKIRTOR_DENSE_SPECS, ids=_SKIRTOR_DENSE_IDS)
def test_skirtor_agnfitter_fixed_cos_inc_collapses_at_its_degree_coordinate(
    spec, grid_torus_filters
):
    """A Fixed ``agn_cos_inc`` collapses the degree axis at ``arccos`` of it (1e-12).

    Read as degrees, cos(55 deg) = 0.574 would collapse the table next to face-on.
    """
    if not spec["grid_path"].exists():
        pytest.skip(f"{spec['name']}: grid not found at {spec['grid_path']}")
    precompute_mod = importlib.import_module(spec["precompute_mod"])
    waves, trans = grid_torus_filters
    cos_inc = float(np.cos(np.radians(55.0)))
    full = precompute_mod.build_lookup(
        precompute_mod.precompute(
            list(waves), list(trans), 0.0, None, grid_path=str(spec["grid_path"])
        )
    )
    collapsed_preint = precompute_mod.precompute(
        list(waves),
        list(trans),
        0.0,
        _FixedInclination(cos_inc),
        grid_path=str(spec["grid_path"]),
    )
    incl_axis = _SKIRTOR_DENSE[spec["name"]]
    assert collapsed_preint["_collapsed_axes"] == {incl_axis: pytest.approx(55.0, abs=1e-9)}
    collapsed = precompute_mod.build_lookup(collapsed_preint)
    others = {k: v for k, v in (("agn_oa_skirtor", 45.0), ("agn_tv_skirtor", 6.0))}
    free = [others[k] for k in spec["axis_kwargs"] if k != "agn_cos_inc"]
    got = np.asarray(collapsed(0.0, *free, agn_torus_frac=1.0))

    def _full_at(cos_value):
        kwargs = {k: (cos_value if k == "agn_cos_inc" else others[k]) for k in spec["axis_kwargs"]}
        return np.asarray(full(agn_log_lbol=0.0, agn_torus_frac=1.0, **kwargs))

    # mean_1p contracts the axis with the shared triweight slice, not the lookup's PCHIP
    # (a pre-existing, registered mismatch: test_precompute_axis_collapse.py), so the
    # collapsed table is compared for where it sits, not to round-off: it is the 55 deg
    # table, not the one at i = cos(55 deg) = 0.574 deg read as degrees. The
    # power-weighted mean_2p / mean_3p read the whole table there and agree exactly.
    def _gap(want):
        return float(np.max(np.abs(got / want - 1.0)))

    right = _gap(_full_at(cos_inc))
    wrong = _gap(_full_at(float(np.cos(np.radians(cos_inc)))))
    assert right < 0.05 and right < 0.2 * wrong, (right, wrong)


# ---------------------------------------------------------------------------
# Torus tables normalized after interpolation: the power-weighted table
# ---------------------------------------------------------------------------
#
# The runtime torus kernels interpolate the raw library templates and normalize the
# interpolated template by its own bolometric integral, which weights each node by its
# raw power. A table of node-normalized band photometry weights every node equally, and
# where the raw power changes by a large factor between nodes the two part. The tables
# below hold the raw band photometry plus a last raw-power channel and divide after
# interpolating (``_template_power_table``); the anisotropic libraries (Nenkova mean_2p /
# mean_3p) then multiply by the runtime's own sightline factor a / mean(a).
#
# The structured sweep: three points per interval on the inclination axis (degrees for
# SKIRTOR, cos i otherwise; the column density for Silva+04) crossed with every interval
# midpoint, and the middle node, of each other axis. A seeded uniform sweep over the
# whole hull adds the points between those lines. Every point must meet the unchanged
# 5e-2 contract. Measured maxima, node-normalized table -> power-weighted table, over
# the structured sweep without the node rows:
#
#   skirtor_agnfitter_2p   189 points  1.68e-1 -> 2.38e-2
#   skirtor_agnfitter      756 points  2.96e-1 -> 6.4e-3
#   cat3d_wind_lowfwd      360 points  2.63e-1 -> 3.62e-2
#   cat3d_wind             270 points  2.84e-1 -> 3.79e-2
#   nenkova_agnfitter_2p   297 points  4.22e-2 -> 3.8e-3
#   nenkova_agnfitter_3p  2376 points  3.98e-2 -> 9.7e-3
#   silva04                177 points  8.1e-3  -> 3.3e-4
#
# and over 1500 uniform random points of the hull (seed 0), power-weighted, at most
# 3.78e-2 (cat3d_wind); with five more mid-infrared bands the same library reaches 4.4e-2
# at cos i = 0.009, the largest gap measured anywhere.
#
# skirtor_agnfitter_1p (27 points, 1.67e-3) and nenkova_agnfitter (24 points, 6.4e-4) are
# power-weighted already, by an identity: with one library axis a is proportional to the
# record power P, so their tables, records times (a / mean a) / P, hold the raw band
# photometry over a constant (measured equal to the ratio-of-PCHIPs reference to 1e-15).
# They stay in the sweeps.
#
# A Fixed axis is read off the whole table at its coordinate, not contracted first: the
# tensor-product PCHIP depends on the order of the axes, and contracting a non-leading
# axis first read skirtor_agnfitter_2p at a Fixed agn_cos_inc 6.4e-2 off the runtime.

_TORUS_EXTRA_SPECS = [
    dict(
        name="cat3d_wind",
        runtime_mod="tengri.components.agn.cat3d_wind",
        precompute_mod="tengri.components.agn.cat3d_precompute",
        loader="load_cat3d_wind_grid",
        sed_fn="cat3d_wind_sed_from_grid",
        grid_path=_REPO_DATA / "cat3d_wind_torus_grid.h5",
        axis_kwargs=("agn_cos_inc", "agn_a_cat3d", "agn_fwd_cat3d"),
    ),
    dict(
        name="silva04",
        runtime_mod="tengri.components.agn.silva04",
        precompute_mod="tengri.components.agn.silva04_precompute",
        loader="load_silva04_grid",
        sed_fn="silva04_sed_from_grid",
        grid_path=_REPO_DATA / "silva04_torus_grid.h5",
        axis_kwargs=("agn_log_nh_silva",),
    ),
]
_TORUS_SWEEP_SIZE = {
    "skirtor_agnfitter_1p": 27,
    "skirtor_agnfitter_2p": 216,
    "skirtor_agnfitter": 1080,
    "nenkova_agnfitter": 24,
    "nenkova_agnfitter_2p": 324,
    "nenkova_agnfitter_3p": 2916,
    "cat3d_wind_lowfwd": 540,
    "cat3d_wind": 432,
    "silva04": 177,
}
_TORUS_RANDOM_POINTS = 1500
_TORUS_SWEEP_SPECS = [
    s for s in _GRID_TORUS_SPECS + _TORUS_EXTRA_SPECS if s["name"] in _TORUS_SWEEP_SIZE
]
_TORUS_SWEEP_IDS = [s["name"] for s in _TORUS_SWEEP_SPECS]
# The power-weighted tables; True marks the anisotropic ones, whose runtime multiplies
# the shape by a / mean(a).
_POWER_TABLE = {
    "skirtor_agnfitter_2p": False,
    "skirtor_agnfitter": False,
    "cat3d_wind_lowfwd": False,
    "cat3d_wind": False,
    "silva04": False,
    "nenkova_agnfitter_2p": True,
    "nenkova_agnfitter_3p": True,
}
_POWER_TABLE_SPECS = [s for s in _TORUS_SWEEP_SPECS if s["name"] in _POWER_TABLE]
_POWER_TABLE_IDS = [s["name"] for s in _POWER_TABLE_SPECS]
# (spec, Fixed axis) for every axis of every power-weighted table.
_FIXED_AXIS_CASES = [
    (spec, k) for spec in _POWER_TABLE_SPECS for k in range(len(spec["axis_kwargs"]))
]
_FIXED_AXIS_IDS = [
    f"{spec['name']}-fixed-{spec['axis_kwargs'][k]}" for spec, k in _FIXED_AXIS_CASES
]


def _library_axes(grid):
    """Node coordinates of every library axis, in axis order."""
    if hasattr(grid, "axes"):
        return [np.asarray(ax, dtype=float) for ax in grid.axes]
    return [np.asarray(grid.log_nh_axis, dtype=float)]


def _sweep_axis(spec):
    """The axis swept at three points per interval."""
    return _SKIRTOR_DENSE.get(spec["name"], 0)


def _other_axis_values(nodes):
    """Every interval midpoint of a non-swept axis, and its middle node."""
    return [(a + b) / 2.0 for a, b in itertools.pairwise(nodes)] + [float(nodes[len(nodes) // 2])]


def _torus_sweep(spec, grid):
    """Library-axis points: 3 per interval on the swept axis crossed with every
    interval midpoint and the middle node of each other axis."""
    swept = _sweep_axis(spec)
    per_axis = [
        _incl_fractions(nodes) if k == swept else _other_axis_values(nodes)
        for k, nodes in enumerate(_library_axes(grid))
    ]
    return np.array(list(itertools.product(*per_axis)))


def _random_hull(grid, n, seed=0):
    """``n`` library-axis points drawn uniformly over the node hull."""
    axes = _library_axes(grid)
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(a[0], a[-1], n) for a in axes])


def _load_torus(spec):
    """Runtime module, precompute module and runtime grid of one torus library."""
    if not spec["grid_path"].exists():
        pytest.skip(f"{spec['name']}: grid not found at {spec['grid_path']}")
    runtime_mod = importlib.import_module(spec["runtime_mod"])
    precompute_mod = importlib.import_module(spec["precompute_mod"])
    grid = getattr(runtime_mod, spec["loader"])(str(spec["grid_path"]))
    return runtime_mod, precompute_mod, grid


def _exact_photometry_fn(sed_fn, grid, names, waves, trans, redshift=0.0):
    """Runtime band photometry at kwarg values on the ``_runtime_wave`` model grid."""
    wave = _runtime_wave(grid, waves, redshift)

    def exact(*values):
        kwargs = dict(zip(names, values, strict=True))
        lnu = sed_fn(grid, wave, agn_log_lbol=0.0, agn_torus_frac=1.0, **kwargs)
        return jnp.stack(
            [
                lnu_filter_integral(lnu, wave, jnp.asarray(w), jnp.asarray(t), redshift)
                for w, t in zip(waves, trans, strict=True)
            ]
        )

    return exact


@pytest.fixture(autouse=True)
def _release_compiled_tables():
    """Hand each test's compile memory back (``_heap``): the file compiles one exact and one
    lookup graph per library and sweep."""
    yield
    release_heap()


#: Library-axis points per ``lax.map`` step of a sweep.
_SWEEP_BATCH = 64


def _compare_over(spec, library, lookup_fn, exact_fn, label):
    """Assert the 5e-2 contract at every library-axis point (rows of ``library``)."""
    kwarg_values = np.array([_kwarg_values(spec, tuple(row)) for row in library])
    stacked = jnp.asarray(kwarg_values)
    # ``lax.map`` in batches, not one ``vmap`` over every point: the exact runtime holds the
    # template on its native grid per point, and 1500 points at once peaked at 9 GB.
    want = np.asarray(
        jax.jit(lambda v: jax.lax.map(lambda r: exact_fn(*r), v, batch_size=_SWEEP_BATCH))(stacked)
    )
    got = np.asarray(
        jax.jit(lambda v: jax.lax.map(lambda r: lookup_fn(*r), v, batch_size=_SWEEP_BATCH))(
            stacked
        )
    )
    names = spec["axis_kwargs"]
    for row, g, e in zip(library, got, want, strict=True):
        _assert_equivalent(g, e, f"{label} at {dict(zip(names, row))}", rtol=5e-2)


def _full_lookup(spec, precompute_mod, waves, trans):
    lookup = precompute_mod.build_lookup(
        precompute_mod.precompute(
            list(waves), list(trans), 0.0, None, grid_path=str(spec["grid_path"])
        )
    )
    names = spec["axis_kwargs"]

    def precomputed(*values):
        return lookup(
            agn_log_lbol=0.0, agn_torus_frac=1.0, **dict(zip(names, values, strict=True))
        )

    return precomputed


@pytest.mark.parametrize("spec", _TORUS_SWEEP_SPECS, ids=_TORUS_SWEEP_IDS)
def test_torus_table_dense_offnode_sweep_matches_exact(spec, grid_torus_filters):
    """Every point of the structured sweep meets the 5e-2 contract against the runtime."""
    runtime_mod, precompute_mod, grid = _load_torus(spec)
    waves, trans = grid_torus_filters
    exact = _exact_photometry_fn(
        getattr(runtime_mod, spec["sed_fn"]), grid, spec["axis_kwargs"], waves, trans
    )
    library = _torus_sweep(spec, grid)
    assert library.shape[0] == _TORUS_SWEEP_SIZE[spec["name"]]
    _compare_over(
        spec, library, _full_lookup(spec, precompute_mod, waves, trans), exact, spec["name"]
    )


@pytest.mark.parametrize("spec", _TORUS_SWEEP_SPECS, ids=_TORUS_SWEEP_IDS)
def test_torus_table_random_hull_sweep_matches_exact(spec, grid_torus_filters):
    """1500 seeded uniform points over the node hull meet the 5e-2 contract."""
    runtime_mod, precompute_mod, grid = _load_torus(spec)
    waves, trans = grid_torus_filters
    exact = _exact_photometry_fn(
        getattr(runtime_mod, spec["sed_fn"]), grid, spec["axis_kwargs"], waves, trans
    )
    library = _random_hull(grid, _TORUS_RANDOM_POINTS)
    _compare_over(
        spec, library, _full_lookup(spec, precompute_mod, waves, trans), exact, spec["name"]
    )


class _FixedAxes:
    """The slice of ``Parameters`` that ``collapse_fixed_axes`` reads: Fixed values."""

    free_params = ()

    def __init__(self, values):
        self._values = dict(values)

    def get_fixed_values(self):
        return dict(self._values)


@pytest.mark.parametrize(("spec", "fixed_axis"), _FIXED_AXIS_CASES, ids=_FIXED_AXIS_IDS)
def test_torus_table_fixed_axis_lookup_matches_exact(spec, fixed_axis, grid_torus_filters):
    """With one axis Fixed off-node, the collapsed lookup meets the 5e-2 contract against
    the runtime off-node on every free axis: three Fixed values per axis, the free axes
    at every interval midpoint (three per interval on the swept axis)."""
    runtime_mod, precompute_mod, grid = _load_torus(spec)
    waves, trans = grid_torus_filters
    names = spec["axis_kwargs"]
    exact = _exact_photometry_fn(getattr(runtime_mod, spec["sed_fn"]), grid, names, waves, trans)
    axes = _library_axes(grid)
    swept = _sweep_axis(spec)
    nodes = axes[fixed_axis]
    picks = [len(nodes) // 4, len(nodes) // 2, (3 * len(nodes)) // 4]
    fixed_values = sorted({(nodes[i] + nodes[i + 1]) / 2.0 for i in picks if i + 1 < len(nodes)})
    for fixed_library in fixed_values:
        fixed_param = _kwarg_values(
            spec,
            tuple(fixed_library if k == fixed_axis else float(a[0]) for k, a in enumerate(axes)),
        )[fixed_axis]
        preint = precompute_mod.precompute(
            list(waves),
            list(trans),
            0.0,
            _FixedAxes({names[fixed_axis]: fixed_param}),
            grid_path=str(spec["grid_path"]),
        )
        assert set(preint["_collapsed_axes"]) == {fixed_axis}
        collapsed = precompute_mod.build_lookup(preint)
        per_axis = [
            [fixed_library]
            if k == fixed_axis
            else (
                _incl_fractions(a)
                if k == swept
                else [(x + y) / 2.0 for x, y in itertools.pairwise(a)]
            )
            for k, a in enumerate(axes)
        ]
        library = np.array(list(itertools.product(*per_axis)))

        def read(*values, collapsed=collapsed):
            free = [v for k, v in enumerate(values) if k != fixed_axis]
            return collapsed(0.0, *free, agn_torus_frac=1.0)

        label = f"{spec['name']} with {names[fixed_axis]} Fixed at {fixed_param:.4g}"
        _compare_over(spec, library, read, exact, label)


@pytest.mark.parametrize("spec", _POWER_TABLE_SPECS, ids=_POWER_TABLE_IDS)
def test_torus_table_is_power_weighted(spec, grid_torus_filters):
    """The lookup is the power-weighted reading of the raw library records, built here
    independently: band photometry of every raw record, resampled log-log onto the
    ``_runtime_wave`` model grid as the runtime emits it and integrated there with
    ``lnu_filter_integral``, and its bolometric power (``native_bolometric_nu_np``),
    SciPy's PCHIP of each along each axis in turn with the others at a node, their
    ratio, times the runtime's a / mean(a) for an anisotropic library.

    Tolerance 1e-9: the table and this reference integrate the bands by the same
    quadrature, so what remains is the two PCHIP implementations, as in the SKIRTOR
    degree-axis test above. An equal-weight table misses by 1e-2 and more; a wrong
    power channel by more still, and a table integrated linearly between the
    template's own wavelengths by up to 3e-1.
    """
    from scipy.interpolate import PchipInterpolator

    from tengri.components.agn._template_grid import native_bolometric_nu_np
    from tengri.components.agn._torus_inclination import library_anisotropy, log10_anisotropy
    from tengri.utils.grid_interp import resample_template
    from tengri.utils.physics_constants import L_SUN

    runtime_mod, precompute_mod, grid = _load_torus(spec)
    waves, trans = grid_torus_filters
    lookup = precompute_mod.build_lookup(
        precompute_mod.precompute(
            list(waves), list(trans), 0.0, None, grid_path=str(spec["grid_path"])
        )
    )
    axes = _library_axes(grid)
    wave_native = np.asarray(grid.wave_grid, dtype=float)
    wave = _runtime_wave(grid, waves)
    for swept in range(len(axes)):
        index = [len(a) // 2 for a in axes]
        index[swept] = slice(None)
        records = np.asarray(grid.template, dtype=float)[tuple(index)]  # (n_swept, n_wave)
        band = np.array(
            [
                [
                    float(
                        lnu_filter_integral(
                            resample_template(wave, jnp.asarray(wave_native), jnp.asarray(r)),
                            wave,
                            jnp.asarray(w),
                            jnp.asarray(t),
                            0.0,
                        )
                    )
                    for w, t in zip(waves, trans, strict=True)
                ]
                for r in records
            ]
        )
        power = np.array([native_bolometric_nu_np(r, wave_native) for r in records])
        band_pchip = PchipInterpolator(axes[swept], band, axis=0)
        power_pchip = PchipInterpolator(axes[swept], power)
        others = [float(a[len(a) // 2]) for a in axes]
        nodes = axes[swept]
        for x in _incl_fractions(nodes):
            if not nodes[1] <= x <= nodes[-2]:
                continue  # end intervals: end-slope conventions differ between PCHIPs
            point = list(others)
            point[swept] = x
            want = L_SUN * band_pchip(x) / power_pchip(x)
            if _POWER_TABLE[spec["name"]]:
                inc = runtime_mod.TORUS_INCLINATION
                aniso = library_anisotropy(grid, inc, jnp.float64)
                want = want * 10.0 ** float(log10_anisotropy(aniso, tuple(point), inc))
            kwargs = dict(zip(spec["axis_kwargs"], _kwarg_values(spec, tuple(point)), strict=True))
            got = np.asarray(lookup(agn_log_lbol=0.0, agn_torus_frac=1.0, **kwargs))
            np.testing.assert_allclose(
                got, want, rtol=1e-9, err_msg=f"{spec['name']} axis {swept} at {x}"
            )


# ---------------------------------------------------------------------------
# Every node of every torus table is the runtime's photometry (#3160)
# ---------------------------------------------------------------------------
#
# A table integrated linearly between the template's own wavelengths matched a reference
# integrated the same way, and sat 5-50% off the runtime at its own nodes: the runtime
# resamples the template log-log onto the model grid before the filter integral. Each
# table is now built on the runtime's quadrature, so at every node it is the runtime on
# the ``_runtime_wave`` model grid to rounding: measured worst case over all nodes of
# the nine libraries, both redshifts, 2.4e-14 -- rtol=1e-10. Measured before the
# rebuild, against this same reference: skirtor_agnfitter_1p 5.0e-1, cat3d_wind_lowfwd
# 3.3e-1 (both at z = 1.2).
#
# Irregular filters: unevenly spaced nodes, a transmission vanishing at the band edges,
# bands from 1300 A to 95 um, so at z = 1.2 the bluest falls below some libraries'
# template range and the reddest band reads the far-infrared tail.

_NODE_REDSHIFTS = (0.0, 1.2)


def _irregular_filters():
    """Ten bands, 1300 A - 95 um [Angstrom], seeded uneven node spacing."""
    rng = np.random.default_rng(3)
    bands = [
        (1300.0, 1800.0, 7),
        (2600.0, 3500.0, 25),
        (4800.0, 6300.0, 40),
        (10500.0, 13500.0, 31),
        (19500.0, 24000.0, 18),
        (30000.0, 39000.0, 22),
        (41000.0, 52000.0, 15),
        (75000.0, 160000.0, 50),
        (195000.0, 270000.0, 12),
        (550000.0, 950000.0, 20),
    ]
    waves, trans = [], []
    for lo, hi, n in bands:
        x = np.sort(np.concatenate([[lo, hi], rng.uniform(lo, hi, n - 2)]))
        u = (x - lo) / (hi - lo)
        waves.append(x)
        trans.append(np.sin(np.pi * u) ** 0.7 * (1.0 + 0.3 * np.sin(7.0 * u)))
    return waves, trans


@pytest.mark.parametrize("redshift", _NODE_REDSHIFTS, ids=[f"z{z}" for z in _NODE_REDSHIFTS])
@pytest.mark.parametrize("spec", _TORUS_SWEEP_SPECS, ids=_TORUS_SWEEP_IDS)
def test_torus_table_nodes_match_runtime(spec, redshift):
    """At every node, the table is the runtime's band photometry to 1e-10."""
    runtime_mod, precompute_mod, grid = _load_torus(spec)
    waves, trans = _irregular_filters()
    names = spec["axis_kwargs"]
    exact = _exact_photometry_fn(
        getattr(runtime_mod, spec["sed_fn"]), grid, names, waves, trans, redshift
    )
    lookup = precompute_mod.build_lookup(
        precompute_mod.precompute(
            list(waves), list(trans), redshift, None, grid_path=str(spec["grid_path"])
        )
    )

    def precomputed(*values):
        return lookup(
            agn_log_lbol=0.0, agn_torus_frac=1.0, **dict(zip(names, values, strict=True))
        )

    nodes = np.array(list(itertools.product(*_library_axes(grid))))
    stacked = jnp.asarray(np.array([_kwarg_values(spec, tuple(row)) for row in nodes]))

    def batched(fn):
        return jax.jit(lambda v: jax.lax.map(lambda r: fn(*r), v, batch_size=_SWEEP_BATCH))

    want = np.asarray(batched(exact)(stacked))
    got = np.asarray(batched(precomputed)(stacked))
    for row, g, e in zip(nodes, got, want, strict=True):
        _assert_equivalent(
            g, e, f"{spec['name']} node {dict(zip(names, row))} at z={redshift}", rtol=1e-10
        )
