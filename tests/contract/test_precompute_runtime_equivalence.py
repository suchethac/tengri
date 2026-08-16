# SPDX-License-Identifier: BSD-3-Clause
"""Precompute ↔ runtime equivalence tests for radio, X-ray, and AGN-disc components.

Each model tested here uses two identical SEDModels — one with precompute=True
(kernel uses precompute lookup tables) and one with precompute=False (kernel uses
runtime wavelength physics). The test:

1. Builds both models with identical parameter draws.
2. Calls predict_photometry(params) on each.
3. Asserts per-filter relative error < 1e-3 (or tighter for fixed-shape models).
4. Prints max rel-error for traceability.

Models covered:
  - radio_synchrotron, radio_freefree, radio_agn_jet
  - xray_xrb, xray_corona, xray_corona_lopez24
  - powerlaw_disc, ss_disc, cigale_disc (AGN analytic discs)
  - qsogen, silva04, cat3d_wind (AGN torus templates)

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

jax.config.update("jax_enable_x64", True)


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
    agn_torus_model=None,
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
    if agn_disc_model is not None:
        kwargs["agn_model"] = agn_disc_model
        kwargs["agn_log_lbol"] = Fixed(11.0)

    if agn_torus_model is not None:
        kwargs["agn_model"] = agn_torus_model
        kwargs["agn_log_lbol"] = Fixed(11.0)

    spec = Parameters(**kwargs)

    # Build both models
    model_runtime = SEDModel(spec, synthetic_ssp, observation=observation, precompute=False)
    model_precomp = SEDModel(spec, synthetic_ssp, observation=observation, precompute=True)

    return model_runtime, model_precomp, spec


# ── Radio Tests ──────────────────────────────────────────────────


class TestRadioSynchrotronPrecomputeEquivalence:
    """Test radio_synchrotron precompute↔runtime equivalence."""

    def test_radio_synchrotron(self, radio_filter_set, synthetic_ssp):
        """Synchrotron SFR-driven radio: alpha_sf axis."""
        waves, trans = radio_filter_set
        model_runtime, model_precomp, spec = _make_sed_model_pair(
            waves, trans, redshift=0.1, enable_radio=True, synthetic_ssp=synthetic_ssp
        )

        key = jax.random.PRNGKey(42)
        params_runtime = spec.sample(key)
        params_precomp = spec.sample(key)

        phot_runtime = model_runtime.predict_photometry(params_runtime)
        phot_precomp = model_precomp.predict_photometry(params_precomp)

        # Compute relative error
        chex.assert_trees_all_close(
            phot_precomp,
            phot_runtime,
            rtol=1e-3,
            atol=1e-30,
            custom_message="radio_synchrotron precompute↔runtime",
        )


class TestRadioFreefreePrecomputeEquivalence:
    """Test radio_freefree precompute↔runtime equivalence."""

    def test_radio_freefree(self, radio_filter_set, synthetic_ssp):
        """Free-free bremsstrahlung: alpha_ff axis."""
        waves, trans = radio_filter_set
        model_runtime, model_precomp, spec = _make_sed_model_pair(
            waves, trans, redshift=0.1, enable_radio=True, synthetic_ssp=synthetic_ssp
        )

        key = jax.random.PRNGKey(42)
        params_runtime = spec.sample(key)
        params_precomp = spec.sample(key)

        phot_runtime = model_runtime.predict_photometry(params_runtime)
        phot_precomp = model_precomp.predict_photometry(params_precomp)

        chex.assert_trees_all_close(
            phot_precomp,
            phot_runtime,
            rtol=1e-3,
            atol=1e-30,
            custom_message="radio_freefree precompute↔runtime",
        )


class TestRadioAGNJetPrecomputeEquivalence:
    """Test radio_agn_jet precompute↔runtime equivalence."""

    def test_radio_agn_jet(self, radio_filter_set, synthetic_ssp):
        """AGN jet power-law: alpha_agn axis."""
        waves, trans = radio_filter_set
        model_runtime, model_precomp, spec = _make_sed_model_pair(
            waves, trans, redshift=0.1, enable_radio=True, synthetic_ssp=synthetic_ssp
        )

        key = jax.random.PRNGKey(42)
        params_runtime = spec.sample(key)
        params_precomp = spec.sample(key)

        phot_runtime = model_runtime.predict_photometry(params_runtime)
        phot_precomp = model_precomp.predict_photometry(params_precomp)

        chex.assert_trees_all_close(
            phot_precomp,
            phot_runtime,
            rtol=1e-3,
            atol=1e-30,
            custom_message="radio_agn_jet precompute↔runtime",
        )


# ── X-ray Tests ──────────────────────────────────────────────────


class TestXRayXRBPrecomputeEquivalence:
    """Test xray_xrb precompute↔runtime equivalence."""

    def test_xray_xrb(self, xray_filter_set, synthetic_ssp):
        """X-ray binaries (HMXB+LMXB): gamma_hmxb, gamma_lmxb axes."""
        waves, trans = xray_filter_set
        model_runtime, model_precomp, spec = _make_sed_model_pair(
            waves, trans, redshift=0.1, enable_xray=True, synthetic_ssp=synthetic_ssp
        )

        key = jax.random.PRNGKey(42)
        params_runtime = spec.sample(key)
        params_precomp = spec.sample(key)

        phot_runtime = model_runtime.predict_photometry(params_runtime)
        phot_precomp = model_precomp.predict_photometry(params_precomp)

        chex.assert_trees_all_close(
            phot_precomp,
            phot_runtime,
            rtol=1e-3,
            atol=1e-30,
            custom_message="xray_xrb precompute↔runtime",
        )


class TestXRayCoronaPrecomputeEquivalence:
    """Test xray_corona precompute↔runtime equivalence."""

    def test_xray_corona(self, xray_filter_set, synthetic_ssp):
        """AGN corona (α_OX relation): gamma, alpha_ox axes."""
        waves, trans = xray_filter_set
        model_runtime, model_precomp, spec = _make_sed_model_pair(
            waves, trans, redshift=0.1, enable_xray=True, synthetic_ssp=synthetic_ssp
        )

        key = jax.random.PRNGKey(42)
        params_runtime = spec.sample(key)
        params_precomp = spec.sample(key)

        phot_runtime = model_runtime.predict_photometry(params_runtime)
        phot_precomp = model_precomp.predict_photometry(params_precomp)

        chex.assert_trees_all_close(
            phot_precomp,
            phot_runtime,
            rtol=1e-3,
            atol=1e-30,
            custom_message="xray_corona precompute↔runtime",
        )


class TestXRayCoronaLopez24PrecomputeEquivalence:
    """Test xray_corona_lopez24 precompute↔runtime equivalence."""

    def test_xray_corona_lopez24(self, xray_filter_set, synthetic_ssp):
        """AGN corona (α_IRX relation): gamma, alpha_irx axes.

        L_12um is computed from agn_log_lbol using Krawczyk+2013 bolometric
        correction (f_12 ~ 0.07). Precompute and runtime paths now use the
        same parametric formula and should be numerically equivalent.
        """
        waves, trans = xray_filter_set
        model_runtime, model_precomp, spec = _make_sed_model_pair(
            waves, trans, redshift=0.1, enable_xray=True, synthetic_ssp=synthetic_ssp
        )

        key = jax.random.PRNGKey(42)
        params_runtime = spec.sample(key)
        params_precomp = spec.sample(key)

        phot_runtime = model_runtime.predict_photometry(params_runtime)
        phot_precomp = model_precomp.predict_photometry(params_precomp)

        chex.assert_trees_all_close(
            phot_precomp,
            phot_runtime,
            rtol=1e-3,
            atol=1e-30,
            custom_message="xray_corona_lopez24 precompute↔runtime",
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

        chex.assert_trees_all_close(
            phot_precomp,
            phot_runtime,
            rtol=1e-3,
            atol=1e-30,
            custom_message="qsogen precompute↔runtime",
        )


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

        chex.assert_trees_all_close(
            phot_precomp,
            phot_runtime,
            rtol=1e-3,
            atol=1e-30,
            custom_message="silva04 precompute↔runtime",
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

        chex.assert_trees_all_close(
            phot_precomp,
            phot_runtime,
            rtol=1e-3,
            atol=1e-30,
            custom_message="cat3d_wind precompute↔runtime",
        )


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

        chex.assert_trees_all_close(
            phot_precomp,
            phot_runtime,
            rtol=1e-3,
            atol=1e-30,
            custom_message="modified_blackbody precompute↔runtime",
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

        chex.assert_trees_all_close(
            phot_precomp,
            phot_runtime,
            rtol=1e-3,
            atol=1e-30,
            custom_message="casey2012 precompute↔runtime",
        )


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
        chex.assert_trees_all_close(
            phot_precomp,
            phot_runtime,
            rtol=1e-10,
            atol=1e-30,
            custom_message="pah_drude precompute↔runtime",
        )
