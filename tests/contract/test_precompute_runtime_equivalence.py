# SPDX-License-Identifier: BSD-3-Clause
"""Precompute ↔ runtime equivalence tests for radio, X-ray, AGN and dust-IR components.

Each case builds two identical SEDModels — one with ``precompute=True`` (kernel
uses precompute lookup tables) and one with ``precompute=False`` (kernel uses
runtime wavelength physics), draws the same parameters into both, and compares
``predict_photometry``.

What is actually covered
------------------------
One radio configuration, one X-ray configuration, three AGN templates
(``qsogen``, ``silva04``, ``cat3d_wind``) and three analytic dust-IR models
(``modified_blackbody``, ``casey2012``, ``pah_drude``).

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

import chex
import jax
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


_REPO_DATA = __import__("pathlib").Path(__file__).resolve().parents[3] / "data"


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
        Dust emission model: "modified_blackbody", "casey2012", "pah_drude".
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
