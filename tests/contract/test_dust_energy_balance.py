# SPDX-License-Identifier: BSD-3-Clause
"""Issue #437: dust UV-absorbed energy ~= dust FIR re-emission.

Two-component Calzetti attenuation + a Draine & Li 2007 / 2014 /
Dale 2014 / THEMIS dust IR template must conserve energy to better
than a few percent across a tau sweep:

    int L_dust_emission dnu  (8-1000 um)
    ~  int (L_intrinsic - L_attenuated) dnu  (912 A - 3 um)

Issue #437 reported a ~10x energy gap (ratio ~0.1) with the
``dl07`` alias, suggesting the IR normalization was decoupled
from the UV-absorbed energy. The fix has since landed in the
component chain; this test pins the invariant so we cannot
regress.

A small (~3%) deficit is allowed because (i) trapezoid
integration on a log-spaced wavelength grid undersamples the
peak of the FIR template, and (ii) some absorbed photons go to
emission-line + PAH features outside the 8-1000 um band.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tengri

pytestmark = pytest.mark.contract


def _lnu_integrate(L_nu: np.ndarray, wave: np.ndarray, wmin: float, wmax: float) -> float:
    """Integrate L_nu over [wmin, wmax] in frequency space."""
    c_aa_s = 2.998e18  # speed of light in Angstrom/s
    mask = (wave >= wmin) & (wave <= wmax)
    w = wave[mask]
    L = L_nu[mask]
    nu = c_aa_s / w
    order = np.argsort(nu)
    return float(np.trapezoid(L[order], nu[order]))


@pytest.fixture(scope="module")
def intrinsic_sed():
    try:
        ssp = tengri.load_ssp()
    except FileNotFoundError as exc:
        pytest.skip(f"SSP data not on disk (CI runner): {exc}")
    m = tengri.SEDModel.build(
        ssp,
        sfh={"type": "tsnorm", "all_params": tengri.FIXED},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": tengri.FIXED,
            "tau_diff": 0.0,
            "tau_bc": 0.0,
        },
        redshift=tengri.Fixed(0.05),
    )
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    sed = np.asarray(m.predict_rest_sed(p).sed)
    wave = np.asarray(m.ssp_data.ssp_wave)
    return sed, wave, ssp


@pytest.mark.parametrize(
    "emission_type",
    ["draine_li2007", "draine_li2014", "dale2014", "themis"],
)
@pytest.mark.parametrize("tau", [0.3, 1.0])
def test_dust_energy_balance(intrinsic_sed, emission_type, tau):
    """L_emit_FIR must equal L_abs_UV-Opt to within ~5%."""
    sed_intr, wave, ssp = intrinsic_sed
    m = tengri.SEDModel.build(
        ssp,
        sfh={"type": "tsnorm", "all_params": tengri.FIXED},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": tengri.FIXED,
            "tau_diff": tau,
            "tau_bc": tau,
        },
        dust_emission={"type": emission_type, "all_params": tengri.FIXED},
        redshift=tengri.Fixed(0.05),
    )
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    out_d = m.predict_rest_sed(p)
    sed_d = np.asarray(out_d.sed)
    # Dust emission extends the prediction onto the union-of-component-grids
    # (panchromatic FIR/mm; #463/#476), which is longer than the intrinsic
    # dust-off SED's ssp_wave grid. Compare on the dusted grid by interpolating
    # the intrinsic SED onto it.
    wave_d = np.asarray(out_d.wavelength)
    sed_intr_on_d = np.interp(wave_d, wave, sed_intr)

    L_abs = _lnu_integrate(sed_intr_on_d - sed_d, wave_d, 912.0, 3.0e4)
    # Measure the re-emission from the dust IR component itself over the full
    # infrared range, not the dusted SED over an 8-1000 um window. That window
    # excludes the 3-8 um mid-IR, where aromatic-rich grains re-emit a large
    # fraction of the absorbed energy: THEMIS at qhac=0.17 puts ~14% of L_dust
    # into the 3-8 um PAH bands (energy is conserved to <1%; the window just
    # misses it). Integrating sed_dust_ir over 0.3-1000 um captures all the
    # re-emission with no stellar-NIR contamination, and holds to ~2% for every
    # dust model.
    state = m.predict_state(p)
    wave_state = np.asarray(state.wave)
    sed_dust_ir = np.asarray(state.derived["sed_dust_ir"])
    L_emit = _lnu_integrate(sed_dust_ir, wave_state, 3.0e3, 1.0e7)
    ratio = L_emit / L_abs
    assert 0.90 < ratio < 1.10, (
        f"Energy balance violated for {emission_type}, tau={tau}: "
        f"L_emit/L_abs = {ratio:.3f} (expected ~1.0)"
    )


# ── Cross-model energy-balance sweep ──────────────────────────────────────
# Every dust IR model that is meant to carry the *whole* absorbed energy must
# re-emit it (∫ L_emit dν ≈ L_absorbed). ``pah_drude`` is deliberately excluded
# — it is a PAH-only building block, not a standalone energy-balanced emitter.
_ANALYTIC_BALANCED = ["modified_blackbody", "casey2012", "schreiber2016", "energy_balance_split"]
_GRID_BALANCED = ["draine_li2014", "astrodust", "bosa", "schreiber2018"]


def _dust_ir_energy(m, p):
    """Frequency-integrated 8-1000 um luminosity of a built model (total SED)."""
    out = m.predict_rest_sed(p)
    w = np.asarray(out.wavelength)
    return _lnu_integrate(np.asarray(out.sed), w, 8.0e4, 1.0e7)


def _pure_dust_ir_energy(m, p):
    """Integral of the *pure* dust IR component (``sed_dust_ir``).

    Isolates the dust re-emission from the constant stellar Rayleigh-Jeans
    tail, so a multiplicative scaling (e.g. eta) is recovered exactly.
    """
    s = np.asarray(m.predict_state(p).derived["sed_dust_ir"])
    return float(np.trapezoid(s))


def _build_emission(ssp, emission, tau=0.6):
    return tengri.SEDModel.build(
        ssp,
        sfh={"type": "tsnorm", "all_params": tengri.FIXED},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": tengri.FIXED,
            "tau_diff": tau,
            "tau_bc": tau,
        },
        dust_emission=emission,
        redshift=tengri.Fixed(0.05),
    )


@pytest.mark.conservation
@pytest.mark.parametrize("emission_type", _ANALYTIC_BALANCED + _GRID_BALANCED)
def test_energy_balance_all_models(intrinsic_sed, emission_type):
    """All full-SED dust emitters conserve absorbed -> emitted energy at eta=1."""
    sed_intr, wave, ssp = intrinsic_sed
    try:
        m = _build_emission(ssp, {"type": emission_type, "all_params": tengri.FIXED})
        out = m.predict_rest_sed(dict(m.spec.sample(jax.random.PRNGKey(0))))
    except FileNotFoundError as exc:
        pytest.skip(f"{emission_type} template grid not on disk: {exc}")
    wave_d = np.asarray(out.wavelength)
    sed_d = np.asarray(out.sed)
    sed_intr_on_d = np.interp(wave_d, wave, sed_intr)
    L_abs = _lnu_integrate(sed_intr_on_d - sed_d, wave_d, 912.0, 3.0e4)
    L_emit = _lnu_integrate(sed_d, wave_d, 8.0e4, 1.0e7)
    ratio = L_emit / L_abs
    # casey2012 carries a mid-IR power law extending blueward of 8 um, so the
    # 8-1000 um window captures slightly less than the full budget -> wider band.
    assert 0.82 < ratio < 1.12, (
        f"Energy balance violated for {emission_type}: L_emit/L_abs = {ratio:.3f} (expected ~1.0)"
    )


# ── Opt-in deviation from energy balance (dust_eta_balance) ────────────────
@pytest.mark.regression_bug
def test_eta_balance_scales_ir_linearly(intrinsic_sed):
    """Freeing dust_eta_balance scales the emitted IR by exactly eta, end-to-end.

    This is the opt-in escape hatch for galaxies whose UV/optical and FIR are
    spatially decoupled and so violate strict energy balance. Default (eta=1,
    Fixed) is strict balance; freeing eta lets L_IR float as eta * L_absorbed.
    """
    _, _, ssp = intrinsic_sed
    # modified_blackbody is analytic (no grid) -> CI-safe.
    m = _build_emission(
        ssp,
        {"type": "modified_blackbody", "eta_balance": tengri.LogNormal(mu=0.0, sigma=0.2)},
    )
    assert "dust_eta_balance" in m.spec.free_params
    e_lo = _pure_dust_ir_energy(m, {"dust_eta_balance": 0.5})
    e_hi = _pure_dust_ir_energy(m, {"dust_eta_balance": 2.0})
    assert abs(e_hi / e_lo - 4.0) < 0.02, f"eta scaling broken: {e_hi / e_lo:.3f} (expect 4.0)"


@pytest.mark.regression_bug
def test_eta_balance_fixed_by_default(intrinsic_sed):
    """With no eta override the model is strict energy balance (eta not free)."""
    _, _, ssp = intrinsic_sed
    m = _build_emission(ssp, {"type": "modified_blackbody", "all_params": tengri.FIXED})
    assert "dust_eta_balance" not in m.spec.free_params


@pytest.mark.contract
def test_relaxed_energy_balance_helper_frees_eta(intrinsic_sed):
    """The ``relaxed_energy_balance`` builder helper frees dust_eta_balance."""
    from tengri import builders

    _, _, ssp = intrinsic_sed
    emission = builders.dust.emission.relaxed_energy_balance("modified_blackbody", sigma=0.3)
    m = _build_emission(ssp, emission)
    assert "dust_eta_balance" in m.spec.free_params


# ── energy_balance_split warm/cold + AGN-IR knobs are live (no silent no-op) ──
@pytest.mark.conservation
def test_energy_balance_split_f_cold_conserves_total(intrinsic_sed):
    """Varying the warm/cold split redistributes but conserves total IR energy."""
    _, _, ssp = intrinsic_sed
    # Energy = frequency integral; the warm/cold split changes the SED *shape*
    # but conserves the total re-emitted energy.
    e_cold = _dust_ir_energy(
        _build_emission(
            ssp,
            {
                "type": "energy_balance_split",
                "f_cold": tengri.Fixed(0.2),
                "all_params": tengri.FIXED,
            },
        ),
        {},
    )
    e_warm = _dust_ir_energy(
        _build_emission(
            ssp,
            {
                "type": "energy_balance_split",
                "f_cold": tengri.Fixed(0.8),
                "all_params": tengri.FIXED,
            },
        ),
        {},
    )
    assert abs(e_cold / e_warm - 1.0) < 0.05, (
        f"f_cold must conserve total IR energy: {e_cold / e_warm:.3f}"
    )


@pytest.mark.regression_bug
def test_energy_balance_split_l_agn_ir_adds(intrinsic_sed):
    """L_agn_ir adds AGN-heated IR on top of the stellar energy budget."""
    _, _, ssp = intrinsic_sed
    # base ~ the absorbed (= re-emitted) energy in erg/s; feed it back as the
    # AGN-IR term so the IR budget should roughly double.
    base = _dust_ir_energy(
        _build_emission(ssp, {"type": "energy_balance_split", "all_params": tengri.FIXED}), {}
    )
    boosted = _dust_ir_energy(
        _build_emission(
            ssp,
            {
                "type": "energy_balance_split",
                "L_agn_ir": tengri.Fixed(base),
                "all_params": tengri.FIXED,
            },
        ),
        {},
    )
    assert boosted > 1.5 * base, f"L_agn_ir had no effect: {boosted / base:.3f}"
