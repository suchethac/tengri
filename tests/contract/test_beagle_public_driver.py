"""Contract: the BEAGLE public-ingredient driver reads only committed or user-supplied inputs.

The driver repackages public nebular tables (Gutkin+2016, Feltre+2016) and the CANDELS workshop
photometry, and implements the analytic attenuation, SFH and IGM references. None of it needs
the BEAGLE binary or BEAGLE templates.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from reproduction.beagle._drivers import beagle_driver as bd

pytestmark = pytest.mark.contract


def test_gutkin_axes_shapes():
    axes = bd.gutkin_axes()
    assert len(axes["logZ_abs"]) == 14
    assert len(axes["logU"]) == 7
    assert len(axes["xi_d"]) == 3
    assert len(axes["n_H"]) >= 2
    assert len(axes["C/O"]) == 9
    assert len(axes["m_up"]) == 2


def test_gutkin_lines_are_vacuum():
    wave = dict(bd.GUTKIN_LINES)
    assert wave["Hb"] == pytest.approx(4862.68, abs=0.05)
    assert wave["[OIII]5007"] == pytest.approx(5008.24, abs=0.05)
    assert wave["Ha"] == pytest.approx(6564.6, abs=0.1)


def test_gutkin_selftest_nodes():
    """First-row Hbeta and line ratios of the Z = 0.001 and Z = 0.014 tables."""
    hb_gutkin = bd.gutkin_node(0.001, -1.0, 0.1, 100.0, 1.40, 100.0)["Hb"]
    assert hb_gutkin == pytest.approx(2.398e7, rel=1e-3)
    ratios = bd.gutkin_ratios_to_hbeta(bd.gutkin_node(0.001, -1.0, 0.1, 100.0, 1.40, 100.0))
    assert ratios["[OIII]5007"] == pytest.approx(5.038, rel=1e-3)
    assert ratios["Ha"] == pytest.approx(2.837, rel=1e-3)
    hb_feltre = bd.feltre_node(0.014, -2.5, 0.1, 100.0, -1.2)["Hbeta"]
    assert hb_feltre == pytest.approx(2.4849e42, rel=1e-4)
    f_feltre = bd.feltre_node(0.014, -2.5, 0.1, 100.0, -1.2)
    assert f_feltre["[OIII]5007"] / hb_feltre == pytest.approx(14.07, rel=1e-3)


def test_cf00_tau_at_v_is_total_for_young_and_ism_for_old():
    tau_v, mu = 0.8, 0.3
    young = bd.cf00_tau(np.array([5500.0]), tau_v, mu, young=True)[0]
    old = bd.cf00_tau(np.array([5500.0]), tau_v, mu, young=False)[0]
    assert young == pytest.approx((1.0 - mu) * tau_v + mu * tau_v)
    assert old == pytest.approx(mu * tau_v)


def test_c13_slope_at_v_equals_minus_n_v():
    tau_v = 1.2
    wave = np.array([5500.0 * 0.999, 5500.0 * 1.001])
    tau = bd.c13_tau(wave, tau_v)
    slope = (math.log(tau[1]) - math.log(tau[0])) / (math.log(wave[1]) - math.log(wave[0]))
    n_v = 2.8 / (1.0 + 3.0 * math.sqrt(tau_v))
    assert slope == pytest.approx(-n_v, rel=1e-3)


def test_current_component_round_trip():
    mass_smooth, psi, dt = 3.0e9, 12.0, 1.0e7
    ssfr = psi / (mass_smooth + psi * dt)
    recovered = bd.current_component(mass_smooth, math.log10(ssfr), dt)
    assert recovered == pytest.approx(psi, rel=1e-10)


def test_paper1_ids_are_twenty_unique_ints():
    ids = bd.paper1_ids()
    assert len(ids) == 20
    assert len(set(ids)) == 20
    assert all(isinstance(i, int) for i in ids)


def test_workshop_catalog_rows_and_finite_flux():
    pytest.importorskip("h5py")
    bd.require_available()
    cat_z1 = bd.workshop_catalog(1)
    cat_z3 = bd.workshop_catalog(3)
    assert len(cat_z1) == 371
    assert len(cat_z3) == 161
    row = cat_z1[np.asarray(cat_z1["ID"]) == 79][0]
    _names, flux, sigma = bd.photometry_for_row(row)
    good = ~np.ma.getmaskarray(flux)
    assert good.any()
    assert np.all(np.isfinite(np.asarray(flux.data)[good]))
    assert np.all(np.isfinite(np.asarray(sigma.data)[good]))


def test_beagle_workshop_summary_names_env_when_unset(monkeypatch):
    monkeypatch.delenv("ART_SEDFITTING_DIR", raising=False)
    with pytest.raises(RuntimeError, match="ART_SEDFITTING_DIR"):
        bd.beagle_workshop_summary(1)
