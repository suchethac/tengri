# SPDX-License-Identifier: BSD-3-Clause
"""Precompute ↔ runtime equivalence tests for radio, X-ray, AGN and dust-IR components.

Each case builds two identical SEDModels — one with ``precompute=True`` (kernel
uses precompute lookup tables) and one with ``precompute=False`` (kernel uses
runtime wavelength physics), draws the same parameters into both, and compares
``predict_photometry``.

What is actually covered
------------------------
One radio configuration, one X-ray configuration, three AGN templates
(``qsogen``, ``silva04``, ``cat3d_wind``) and four analytic dust-IR models
(``modified_blackbody``, ``casey2012``, ``graybody``, ``pah_drude``).

The previous version of this docstring claimed rather more, and the gap is the
reason for most of this file's rewrite:

* It listed ``radio_synchrotron``, ``radio_freefree`` and ``radio_agn_jet`` as
  three covered models. **None is a model name** — the radio menu is
  ``condon92 / radio_dpl / radio_powerlaw / none`` — and all three tests called
  the helper with byte-identical arguments (``enable_radio=True``), so they
  built the same model three times. Same for ``xray_xrb`` / ``xray_corona`` /
  ``xray_corona_lopez24`` against a menu of ``simple / lopez24 / yang20 /
  xray_aird / agn_xray_corona / none``. Six tests, two configurations.
* It listed ``powerlaw_disc``, ``ss_disc`` and ``cigale_disc``. There were no
  such tests.

Selecting a specific radio or X-ray model is deliberately *not* attempted here.
That belongs with #1684, which is about whether the selector reaches the
component at all, and needs a fixture this file does not have (see below).

The tolerance, and why it was doing nothing
-------------------------------------------
Every assertion was ``chex.assert_trees_all_close(..., rtol=1e-3, atol=1e-30)``
against photometry that measured **~1e-44 (radio)** and **~1e-42 (X-ray)** — an
absolute floor twelve to fourteen orders of magnitude above the signal.
Substituting all-zeros for the precompute result passed unchanged; verified by
mutation on 2026-08-17. Comparisons now go through :func:`_assert_equivalent`,
whose floor is a fraction of the measured signal rather than a fixed constant,
and which refuses to compare two arrays that are identically zero.

The photometry was that small because the fixture was ``synthetic_ssp``, whose
grid spans **3000–10000 Å**, while the radio bands sit at 1e6–1e8 Å and the
X-ray bands at 1.24–124 Å. Neither band was on the grid. Radio now uses
``synthetic_ssp_wide`` (100 Å – 1 mm) and X-ray a local grid reaching 0.1 Å.
#1684 documents the same trap: on a grid that does not reach the band, every
X-ray model measures identical *for the wrong reason*.

References
----------
.. [1] Existing working example:
   tests/contract/test_agn_nebular_precompute_equivalence.py
"""

from __future__ import annotations

import importlib
import itertools

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract


def _assert_equivalent(precomp, runtime, label, *, rtol=1e-3):
    """Compare two photometry vectors against a floor set by the signal.

    The fixed ``atol=1e-30`` this replaces was unrelated to any scale in the
    problem. Against radio photometry of ~1e-44 it accepted anything, including
    an all-zeros precompute result.

    Two guards:

    * the runtime vector must not be identically zero — otherwise there is
      nothing to be equivalent *to*, and the comparison is vacuous however
      tight the tolerance;
    * the absolute floor is ``1e-10`` of the largest runtime element, so a
      near-zero band gets slack proportional to the signal rather than to a
      constant somebody guessed.
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


@pytest.fixture(scope="module")
def wide_band_ssp():
    """A coarse SSP grid spanning 0.1 A - 1 mm, so radio *and* X-ray are on it.

    Neither shared fixture covers both ends: ``synthetic_ssp`` spans
    3000-10000 A and ``synthetic_ssp_wide`` 100 A - 1 mm, while this file's
    filters sit at 1.24-124 A (X-ray) and 1e6-1e8 A (radio).

    Deliberately coarse. Same construction as ``synthetic_ssp_wide`` in
    ``tests/conftest.py`` with the blue edge moved to 0.1 A and the grid cut to
    260 points: an equivalence check between two kernels does not need spectral
    resolution, and at conftest's 1600 points this file took 17 minutes instead
    of the 1.5 it takes now. The 260 points still resolve every filter here,
    which ``test_the_*_band_is_not_empty`` below would catch if they did not.
    """
    import jax.numpy as jnp

    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    n_age = 12
    wave = jnp.logspace(-1.0, 7.0, 260)  # 0.1 A - 1 mm
    ages_gyr = jnp.linspace(-3.0, 1.14, n_age)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    flux = jnp.abs(flux) + 1e-12
    return SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def radio_filter_set():
    """Synthetic filter set covering FIR to radio (1 mm to 10 cm)."""
    # Centers in Angstrom: 1e6 (FIR), 1e7 (mm), 1e8 (cm)
    centers = np.array([1e6, 1e7, 1e8])
    widths = np.array([2e5, 2e6, 2e7])
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(max(c - 3 * w, 1e4), c + 3 * w, 64)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)
    return waves, trans


@pytest.fixture
def xray_filter_set():
    """Synthetic filter set covering hard X-rays (0.1, 1, 10 keV)."""
    # Convert keV to Angstrom: E(keV) <-> lambda(A) via E = 12398.4 / lambda
    # 0.1 keV ≈ 124 Å, 1 keV ≈ 12.4 Å, 10 keV ≈ 1.24 Å
    centers = np.array([124.0, 12.4, 1.24])
    widths = np.array([30.0, 3.0, 0.3])
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(max(c - 3 * w, 0.1), c + 3 * w, 64)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)
    return waves, trans


@pytest.fixture
def agn_disc_filter_set():
    """Synthetic filter set covering UV-optical-NIR (1500–12000 Å)."""
    centers = np.array([1500.0, 3500.0, 5500.0, 8500.0, 12000.0])
    widths = np.array([300.0, 500.0, 700.0, 1000.0, 1500.0])
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(c - 3 * w, c + 3 * w, 64)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)
    return waves, trans


@pytest.fixture
def agn_torus_filter_set():
    """Synthetic filter set covering UV-optical-IR (1500–100000 Å)."""
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


def _make_synth_observation(waves_list, trans_list):
    """Construct an Observation from filter waves and transmissions."""
    from tengri import Observation, Photometry
    from tengri.observation.photometry import FilterCurve

    curves = tuple(
        FilterCurve(wave=w, trans=t, name=f"band_{i}")
        for i, (w, t) in enumerate(zip(waves_list, trans_list))
    )
    return Observation(photometry=Photometry(filters=curves))


def _make_sed_model_pair(
    waves_list,
    trans_list,
    redshift=0.1,
    enable_radio=False,
    enable_xray=False,
    agn_disc_model=None,
    synthetic_ssp=None,
):
    """Build a pair of SEDModels (precompute=False/True) with identical config.

    Parameters
    ----------
    waves_list, trans_list : list
        Filter wavelengths and transmissions.
    redshift : float
        Redshift (default 0.1).
    enable_radio : bool
        If True, enable radio component.
    enable_xray : bool
        If True, enable X-ray component.
    agn_disc_model : str or None
        If set, enable AGN disc with this model.
    agn_torus_model : str or None
        If set, enable AGN torus with this model.
    synthetic_ssp : SSPData
        Minimal synthetic SSP (from conftest fixture).

    Returns
    -------
    tuple[SEDModel, SEDModel, Parameters]
        (runtime_model, precompute_model, spec)
    """
    from tengri import Fixed, Parameters, SEDModel

    observation = _make_synth_observation(waves_list, trans_list)

    # Build kwargs for Parameters
    kwargs = {
        "mean_sfh_type": "dpl",
        "sfh_dpl_alpha": Fixed(1.5),
        "sfh_dpl_beta": Fixed(2.0),
        "sfh_dpl_tau_gyr": Fixed(5.0),
        "sfh_dpl_log_total_mass": Fixed(0.0),
        "met_logzsol": Fixed(-0.5),
        "dust_tau_bc": Fixed(0.0),
        "dust_tau_diff": Fixed(0.0),
        "dust_slope": Fixed(-0.7),
        "redshift": Fixed(redshift),
    }

    # Add radio if requested
    if enable_radio:
        kwargs["radio"] = True

    # Add X-ray if requested
    if enable_xray:
        kwargs["xray"] = True

    # Add AGN model if requested (both disc and torus use agn_model kwarg)
    # One branch, not two. There used to be a second `agn_torus_model`
    # parameter with an identical body; no test ever passed it, and the three
    # torus templates are all passed through `agn_disc_model` anyway, so it was
    # dead in both senses.
    if agn_disc_model is not None:
        kwargs["agn_model"] = agn_disc_model
        kwargs["agn_log_lbol"] = Fixed(11.0)

    spec = Parameters(**kwargs)

    # Build both models
    model_runtime = SEDModel(spec, synthetic_ssp, observation=observation, precompute=False)
    model_precomp = SEDModel(spec, synthetic_ssp, observation=observation, precompute=True)

    return model_runtime, model_precomp, spec


# ── Radio and X-ray ──────────────────────────────────────────────
#
# Two configurations, not six tests. The six these replace called the helper
# with byte-identical arguments within each group -- ``enable_radio=True`` three
# times under three names that are not radio models, and ``enable_xray=True``
# three times likewise -- so each group built one configuration and asserted it
# three times over.


class TestRadioPrecomputeEquivalence:
    """One radio configuration, on a grid that reaches the radio band."""

    def test_radio(self, radio_filter_set, wide_band_ssp):
        """Radio emission: the precompute LUT must reproduce the runtime kernel.

        ``synthetic_ssp_wide`` (100 A - 1 mm), not ``synthetic_ssp``
        (3000-10000 A): the radio bands sit at 1e6-1e8 A, so on the narrow grid
        the photometry measured ~1e-44 and the shipped ``atol=1e-30`` accepted
        anything, including all zeros.
        """
        waves, trans = radio_filter_set
        model_runtime, model_precomp, spec = _make_sed_model_pair(
            waves, trans, redshift=0.1, enable_radio=True, synthetic_ssp=wide_band_ssp
        )
        params = spec.sample(jax.random.PRNGKey(42))

        _assert_equivalent(
            model_precomp.predict_photometry(params),
            model_runtime.predict_photometry(params),
            "radio precompute<->runtime",
        )

    def test_the_radio_band_is_not_empty(self, radio_filter_set, wide_band_ssp):
        """The equivalence above is only meaningful if radio reaches these bands.

        Without this, a change that stopped radio contributing at all would
        leave two vectors that agree because both are the same stellar SED.
        """
        waves, trans = radio_filter_set
        with_radio, _, spec_on = _make_sed_model_pair(
            waves, trans, redshift=0.1, enable_radio=True, synthetic_ssp=wide_band_ssp
        )
        without, _, spec_off = _make_sed_model_pair(
            waves, trans, redshift=0.1, enable_radio=False, synthetic_ssp=wide_band_ssp
        )
        key = jax.random.PRNGKey(42)
        on = np.asarray(with_radio.predict_photometry(spec_on.sample(key)))
        off = np.asarray(without.predict_photometry(spec_off.sample(key)))

        assert np.max(np.abs(on)) > 0.0, "radio photometry is identically zero"
        assert not np.array_equal(on, off), (
            "enabling radio does not change the photometry in these bands, so "
            "the equivalence test above compares the stellar SED to itself"
        )


class TestXRayPrecomputeEquivalence:
    """One X-ray configuration, on a grid that reaches the X-ray band."""

    def test_xray(self, xray_filter_set, wide_band_ssp):
        """X-ray emission: the precompute LUT must reproduce the runtime kernel.

        Uses the module-local 0.1 A grid. Neither shared SSP fixture reaches
        these bands (1.24-124 A), which is why the version this replaces
        compared numbers of order 1e-42 under an absolute floor of 1e-30.
        """
        waves, trans = xray_filter_set
        model_runtime, model_precomp, spec = _make_sed_model_pair(
            waves,
            trans,
            redshift=0.1,
            enable_xray=True,
            agn_disc_model="qsogen",
            synthetic_ssp=wide_band_ssp,
        )
        params = spec.sample(jax.random.PRNGKey(42))

        _assert_equivalent(
            model_precomp.predict_photometry(params),
            model_runtime.predict_photometry(params),
            "xray precompute<->runtime",
        )

    def test_the_xray_band_is_not_empty(self, xray_filter_set, wide_band_ssp):
        """Same guard as for radio, and the reason the local fixture exists."""
        waves, trans = xray_filter_set
        with_xray, _, spec_on = _make_sed_model_pair(
            waves,
            trans,
            redshift=0.1,
            enable_xray=True,
            agn_disc_model="qsogen",
            synthetic_ssp=wide_band_ssp,
        )
        without, _, spec_off = _make_sed_model_pair(
            waves,
            trans,
            redshift=0.1,
            enable_xray=False,
            agn_disc_model="qsogen",
            synthetic_ssp=wide_band_ssp,
        )
        key = jax.random.PRNGKey(42)
        on = np.asarray(with_xray.predict_photometry(spec_on.sample(key)))
        off = np.asarray(without.predict_photometry(spec_off.sample(key)))

        assert np.max(np.abs(on)) > 0.0, "xray photometry is identically zero"
        assert not np.array_equal(on, off), (
            "enabling xray does not change the photometry in these bands"
        )


# ── AGN Torus Tests ──────────────────────────────────────────────


class TestQSOgenPrecomputeEquivalence:
    """Test qsogen precompute↔runtime equivalence."""

    def test_qsogen(self, agn_torus_filter_set, synthetic_ssp):
        """QSOgen quasar SED: plslp1, ebv axes."""
        waves, trans = agn_torus_filter_set
        model_runtime, model_precomp, spec = _make_sed_model_pair(
            waves,
            trans,
            redshift=0.1,
            agn_disc_model="qsogen",
            synthetic_ssp=synthetic_ssp,
        )

        key = jax.random.PRNGKey(42)
        params_runtime = spec.sample(key)
        params_precomp = spec.sample(key)

        phot_runtime = model_runtime.predict_photometry(params_runtime)
        phot_precomp = model_precomp.predict_photometry(params_precomp)

        _assert_equivalent(phot_precomp, phot_runtime, "qsogen precompute↔runtime", rtol=1e-3)


from tests._data_skip import DATA_DIR as _REPO_DATA  # root computed once; see #1431


class TestSilva04PrecomputeEquivalence:
    """Test silva04 precompute↔runtime equivalence."""

    @pytest.mark.skipif(
        not (_REPO_DATA / "silva04_torus_grid.h5").exists(),
        reason="Silva+04 torus grid not available; build via scripts/build_silva04_grid.py.",
    )
    def test_silva04(self, agn_torus_filter_set, synthetic_ssp):
        """Silva+04 torus: log_NH axis."""
        waves, trans = agn_torus_filter_set
        model_runtime, model_precomp, spec = _make_sed_model_pair(
            waves,
            trans,
            redshift=0.1,
            agn_disc_model="silva04",
            synthetic_ssp=synthetic_ssp,
        )

        key = jax.random.PRNGKey(42)
        params_runtime = spec.sample(key)
        params_precomp = spec.sample(key)

        phot_runtime = model_runtime.predict_photometry(params_runtime)
        phot_precomp = model_precomp.predict_photometry(params_precomp)

        _assert_equivalent(phot_precomp, phot_runtime, "silva04 precompute↔runtime", rtol=1e-3)


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


class TestCAT3DWindPrecomputeEquivalence:
    """Test cat3d_wind precompute↔runtime equivalence."""

    @pytest.mark.skipif(
        not (_REPO_DATA / "cat3d_wind_torus_grid.h5").exists(),
        reason="CAT3D-Wind torus grid not available; build via scripts/build_cat3d_wind_grid.py.",
    )
    def test_cat3d_wind(self, agn_torus_filter_set, synthetic_ssp):
        """CAT3D-Wind torus: cos_inc, a, fwd axes."""
        waves, trans = agn_torus_filter_set
        model_runtime, model_precomp, spec = _make_sed_model_pair(
            waves,
            trans,
            redshift=0.1,
            agn_disc_model="cat3d_wind",
            synthetic_ssp=synthetic_ssp,
        )

        key = jax.random.PRNGKey(42)
        params_runtime = spec.sample(key)
        params_precomp = spec.sample(key)

        phot_runtime = model_runtime.predict_photometry(params_runtime)
        phot_precomp = model_precomp.predict_photometry(params_precomp)

        _assert_equivalent(phot_precomp, phot_runtime, "cat3d_wind precompute↔runtime", rtol=1e-3)


# ── Analytic Dust Emission Tests (PR 3) ──────────────────────────


@pytest.fixture
def dust_ir_filter_set():
    """Synthetic filter set covering FIR (5–100 μm)."""
    centers = np.array([5e4, 1e5, 2e4])  # 5, 10, 20 μm in Angstrom
    widths = np.array([1e4, 2e4, 5e3])
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(max(c - 3 * w, 1e4), c + 3 * w, 64)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)
    return waves, trans


def _make_dust_model_pair(
    waves_list,
    trans_list,
    dust_model="modified_blackbody",
    redshift=0.1,
    synthetic_ssp=None,
):
    """Build a pair of SEDModels (precompute=False/True) with dust_emission set.

    Parameters
    ----------
    waves_list, trans_list : list
        Filter wavelengths and transmissions.
    dust_model : str
        Dust emission model: "modified_blackbody", "casey2012", "graybody", "pah_drude".
    redshift : float
        Redshift (default 0.1).
    synthetic_ssp : SSPData
        Minimal synthetic SSP (from conftest fixture).

    Returns
    -------
    tuple[SEDModel, SEDModel, Parameters]
        (runtime_model, precompute_model, spec)
    """
    from tengri import Fixed, Parameters, SEDModel

    observation = _make_synth_observation(waves_list, trans_list)

    kwargs = {
        "mean_sfh_type": "dpl",
        "sfh_dpl_alpha": Fixed(1.5),
        "sfh_dpl_beta": Fixed(2.0),
        "sfh_dpl_tau_gyr": Fixed(5.0),
        "sfh_dpl_log_total_mass": Fixed(0.0),
        "met_logzsol": Fixed(-0.5),
        "dust_tau_bc": Fixed(0.5),  # Enable dust
        "dust_tau_diff": Fixed(0.2),
        "dust_slope": Fixed(-0.7),
        "dust_emission": dust_model,  # Set analytic dust model
        "redshift": Fixed(redshift),
    }

    # Add dust parameters based on model
    if dust_model == "modified_blackbody":
        kwargs["dust_T"] = Fixed(35.0)
        kwargs["dust_beta_ir"] = Fixed(1.6)
    elif dust_model == "casey2012":
        kwargs["dust_T"] = Fixed(40.0)
        kwargs["dust_beta_ir"] = Fixed(1.8)
        kwargs["dust_alpha_mir"] = Fixed(2.0)
    elif dust_model == "graybody":
        kwargs["dust_T"] = Fixed(35.0)
        kwargs["dust_beta_ir"] = Fixed(1.6)
        kwargs["dust_lambda_0_um"] = Fixed(150.0)
    # pah_drude has no free parameters (pure template)

    spec = Parameters(**kwargs)

    # Build both models
    model_runtime = SEDModel(spec, synthetic_ssp, observation=observation, precompute=False)
    model_precomp = SEDModel(spec, synthetic_ssp, observation=observation, precompute=True)

    return model_runtime, model_precomp, spec


class TestModifiedBlackbodyPrecomputeEquivalence:
    """Test modified_blackbody analytic dust precompute↔runtime equivalence."""

    def test_modified_blackbody(self, dust_ir_filter_set, synthetic_ssp):
        """Modified blackbody dust: dust_T, dust_beta_ir axes."""
        waves, trans = dust_ir_filter_set
        model_runtime, model_precomp, spec = _make_dust_model_pair(
            waves,
            trans,
            dust_model="modified_blackbody",
            redshift=0.1,
            synthetic_ssp=synthetic_ssp,
        )

        key = jax.random.PRNGKey(42)
        params_runtime = spec.sample(key)
        params_precomp = spec.sample(key)

        phot_runtime = model_runtime.predict_photometry(params_runtime)
        phot_precomp = model_precomp.predict_photometry(params_precomp)

        _assert_equivalent(
            phot_precomp, phot_runtime, "modified_blackbody precompute↔runtime", rtol=1e-3
        )


class TestCasey2012PrecomputeEquivalence:
    """Test casey2012 analytic dust precompute↔runtime equivalence."""

    def test_casey2012(self, dust_ir_filter_set, synthetic_ssp):
        """Casey+2012 MBB + mid-IR slope: dust_T, dust_beta_ir, dust_alpha_mir axes."""
        waves, trans = dust_ir_filter_set
        model_runtime, model_precomp, spec = _make_dust_model_pair(
            waves,
            trans,
            dust_model="casey2012",
            redshift=0.1,
            synthetic_ssp=synthetic_ssp,
        )

        key = jax.random.PRNGKey(42)
        params_runtime = spec.sample(key)
        params_precomp = spec.sample(key)

        phot_runtime = model_runtime.predict_photometry(params_runtime)
        phot_precomp = model_precomp.predict_photometry(params_precomp)

        _assert_equivalent(phot_precomp, phot_runtime, "casey2012 precompute↔runtime", rtol=1e-3)


class TestGraybodyPrecomputeEquivalence:
    """Test graybody analytic dust precompute↔runtime equivalence."""

    def test_graybody(self, dust_ir_filter_set, synthetic_ssp):
        """Graybody general-opacity model: dust_T, dust_beta_ir, dust_lambda_0_um axes."""
        waves, trans = dust_ir_filter_set
        model_runtime, model_precomp, spec = _make_dust_model_pair(
            waves,
            trans,
            dust_model="graybody",
            redshift=0.1,
            synthetic_ssp=synthetic_ssp,
        )

        key = jax.random.PRNGKey(42)
        params_runtime = spec.sample(key)
        params_precomp = spec.sample(key)

        phot_runtime = model_runtime.predict_photometry(params_runtime)
        phot_precomp = model_precomp.predict_photometry(params_precomp)

        _assert_equivalent(phot_precomp, phot_runtime, "graybody precompute↔runtime", rtol=1e-3)


class TestPAHDrudePrecomputeEquivalence:
    """Test pah_drude analytic dust precompute↔runtime equivalence."""

    def test_pah_drude(self, dust_ir_filter_set, synthetic_ssp):
        """PAH Drude template (pure shape, no free axes): scalar template scaling."""
        waves, trans = dust_ir_filter_set
        model_runtime, model_precomp, spec = _make_dust_model_pair(
            waves,
            trans,
            dust_model="pah_drude",
            redshift=0.1,
            synthetic_ssp=synthetic_ssp,
        )

        key = jax.random.PRNGKey(42)
        params_runtime = spec.sample(key)
        params_precomp = spec.sample(key)

        phot_runtime = model_runtime.predict_photometry(params_runtime)
        phot_precomp = model_precomp.predict_photometry(params_precomp)

        # PAH template has no grid axes, so numerical error should be < 1e-10
        _assert_equivalent(phot_precomp, phot_runtime, "pah_drude precompute↔runtime", rtol=1e-10)


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
# the composable torus block calls) evaluated at the point under test, then
# filter-integrated via :func:`tengri.observation.photometry.lnu_filter_integral`
# -- the same BESSELL-convention integral
# :func:`tengri.utils.grid_interp.preintegrate_grid` documents itself as
# matching, but a genuinely independent call (no shared code with the
# adapter under test).
#
# Tolerance, measured (not guessed): at an exact grid node neither path
# interpolates, so agreement is machine precision (measured worst case
# 8.6e-15 across all seven blocks) -- rtol=1e-10 leaves 5 orders of margin.
# Off-node, the two paths do not commute (precompute interpolates
# NODE-PREINTEGRATED PHOTOMETRY via PCHIP; the exact path integrates the
# TEMPLATE interpolated in wavelength space, then integrates through the
# filter) so a real, larger residual is expected: measured worst case 3.45e-2
# (``cat3d_wind_lowfwd``, whose axes are widest in relative terms) -- rtol
# =5e-2 leaves ~50% margin over every measured case.
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


def _exact_grid_torus_photometry(sed_fn, grid, axis_kwargs, values, waves, trans):
    """Filter-integrated photometry from the runtime SED function directly."""
    kwargs = dict(zip(axis_kwargs, values, strict=True))
    L_nu = sed_fn(
        grid, jnp.asarray(grid.wave_grid), agn_log_lbol=0.0, agn_torus_frac=1.0, **kwargs
    )
    out = [
        float(
            lnu_filter_integral(
                L_nu, jnp.asarray(grid.wave_grid), jnp.asarray(w), jnp.asarray(t), 0.0
            )
        )
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


def _exact_photometry_fn(sed_fn, grid, names, waves, trans):
    """Runtime band photometry at kwarg values, vmappable."""
    wave_native = jnp.asarray(grid.wave_grid)

    def exact(*values):
        kwargs = dict(zip(names, values, strict=True))
        lnu = sed_fn(grid, wave_native, agn_log_lbol=0.0, agn_torus_frac=1.0, **kwargs)
        return jnp.stack(
            [
                lnu_filter_integral(lnu, wave_native, jnp.asarray(w), jnp.asarray(t), 0.0)
                for w, t in zip(waves, trans, strict=True)
            ]
        )

    return exact


def _compare_over(spec, library, lookup_fn, exact_fn, label):
    """Assert the 5e-2 contract at every library-axis point (rows of ``library``)."""
    kwarg_values = np.array([_kwarg_values(spec, tuple(row)) for row in library])
    columns = [jnp.asarray(kwarg_values[:, k]) for k in range(kwarg_values.shape[1])]
    want = np.asarray(jax.jit(jax.vmap(exact_fn))(*columns))
    got = np.asarray(jax.jit(jax.vmap(lookup_fn))(*columns))
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
    independently: band photometry of every raw record (``lnu_filter_integral``) and
    its bolometric power (``native_bolometric_nu_np``), SciPy's PCHIP of each along the
    each axis in turn with the others at a node, their ratio, times the runtime's a / mean(a)
    for an anisotropic library.

    Tolerance 1e-4, measured: the table builder and ``lnu_filter_integral`` integrate
    the bands by different quadratures, 3e-5 apart at the nodes. An equal-weight table
    misses by 1e-2 and more; a wrong power channel by more still.
    """
    from scipy.interpolate import PchipInterpolator

    from tengri.components.agn._template_grid import native_bolometric_nu_np
    from tengri.components.agn._torus_inclination import library_anisotropy, log10_anisotropy
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
    for swept in range(len(axes)):
        index = [len(a) // 2 for a in axes]
        index[swept] = slice(None)
        records = np.asarray(grid.template, dtype=float)[tuple(index)]  # (n_swept, n_wave)
        band = np.array(
            [
                [
                    float(
                        lnu_filter_integral(
                            jnp.asarray(r),
                            jnp.asarray(wave_native),
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
                got, want, rtol=1e-4, err_msg=f"{spec['name']} axis {swept} at {x}"
            )
