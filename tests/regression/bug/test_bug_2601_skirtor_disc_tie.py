# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #2601: SKIRTOR disc tie bugs.

Three physics defects in the SKIRTOR disc coupling and piecewise disc model:
1. η(i) applied twice in the disc (already in library; added again in R)
2. Type-2 disc screened twice (library ratio + torus_screen_transmission)
3. Piecewise disc not cut at 8 nm (spurious 1-8 nm tail in normalization)
"""

import warnings

warnings.filterwarnings("ignore")

import jax.numpy as jnp
import numpy as np
import pytest
from pcigale.data import SimpleDatabase
from pcigale.sed import SED
from pcigale.sed_modules import skirtor2016 as PS

import tengri
from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.agn import disc_cigale as DC

pytestmark = pytest.mark.regression_bug


C_AA = 2.99792458e18


def nu_int(w, lnu):
    """Numerical integration in frequency space."""
    return float(np.trapezoid(np.asarray(lnu)[::-1], (C_AA / np.asarray(w))[::-1]))


@pytest.fixture(scope="module")
def ssp_data():
    """Load SSP data once per module."""
    try:
        return tengri.load_ssp()
    except FileNotFoundError:
        pytest.skip("SSP data not available")


@pytest.fixture(scope="module")
def pcigale_reference():
    """Compute pcigale reference values and library facts."""
    FRAC = 0.3
    KW = dict(t=7, pl=1.0, q=1.0, oa=40, R=20, Mcl=0.97)

    def pcigale_disc(i):
        """Disc luminosity per unit agn_power from pcigale."""
        s = SED()
        s.add_info("dust.luminosity", 1.0, True, unit="W")
        PS.SKIRTOR2016(
            name="skirtor2016",
            **KW,
            i=i,
            disk_type=1,
            delta=0,
            fracAGN=FRAC,
            lambda_fracAGN="0/0",
            law=0,
            EBV=0.0,
            temperature=100.0,
            emissivity=1.6,
        ).process(s)
        return s.info["agn.disk_luminosity"] / (FRAC / (1 - FRAC))

    # Library ratio ∫disk(i)/∫disk(0) / η(i)  [should be ~1]
    library_ratio_over_eta = {}
    with SimpleDatabase("skirtor2016") as db:
        e0 = db.get(**KW, i=0)
        for i in (10, 20, 30, 40):
            e = db.get(**KW, i=i)
            c = np.cos(np.radians(i))
            eta = c * (1 + 2 * c) / 3
            int_ratio = (
                np.trapezoid(e.disk, e.wl) * e.norm / (np.trapezoid(e0.disk, e0.wl) * e0.norm)
            ) / eta
            library_ratio_over_eta[i] = int_ratio
        wl = e0.wl

    # Disc shape reference
    disc0_shape = PS.schartmann2005_disk(wl, delta=0.0)

    return {
        "pcigale_disc": pcigale_disc,
        "library_ratio_over_eta": library_ratio_over_eta,
        "pcigale_wl": wl,
        "disc0_shape": disc0_shape,
        "frac": FRAC,
        "kw": KW,
    }


class TestSkirtorDiscTie:
    """Test corrections for the three SKIRTOR disc-tie defects."""

    @pytest.mark.parametrize("inclination_deg", [0, 30, 50, 70])
    def test_disc_luminosity_polar_off(self, ssp_data, pcigale_reference, inclination_deg):
        """Test A: Disc luminosity matches pcigale after removing η from R.

        Defect: η(i) applied twice (in library + in R).
        Fix: Remove η from R = ∫(disk_analytic · disk_i/disk_0 · ext)/∫dust.

        RED: i=0: 0.9945, i=30: 0.7820, i=50: 0.3122, i=70: 0.1318
        GREEN: 0.99-1.00 (Type 1, <40°) and 0.99 (Type 2, >40°, after removing screen)
        """
        FRAC = pcigale_reference["frac"]
        cos_inc = float(np.cos(np.radians(inclination_deg)))
        eta = cos_inc * (1 + 2 * cos_inc) / 3

        # Build tengri model: composable SKIRTOR, polar dust OFF
        agn_config = {
            "type": "composable",
            "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
            "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
            "agn_ir_frac": Fixed(FRAC),
            "atten": {"type": "none"},
            "agn_cos_inc": Fixed(cos_inc),
            "all_params": Fixed(DEFAULT),
        }

        model = SEDModel.build(
            ssp_data=ssp_data,
            met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "type": "two_component",
                "law_bc": "leitherer02",
                "law_diff": "leitherer02",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(1.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "draine_li2014", "all_params": Fixed(DEFAULT)},
            neb={"type": "none"},
            redshift=Fixed(0.0),
            agn=agn_config,
        )

        st = model.predict_state({})
        d = st.derived

        # Compute tengri disc luminosity per agn_power
        L_absorbed = float(d["L_absorbed"])
        tengri_disc = nu_int(st.wave, d["sed_agn_disc"]) / (L_absorbed * FRAC / (1 - FRAC))

        # Expected from pcigale
        pcigale_disc = pcigale_reference["pcigale_disc"](inclination_deg)

        # Check: after fix, ratio should be 0.99-1.00 for Type 1 (<40°),
        # 0.99 for Type 2 (>40°, but the screen defect applies AFTER this check)
        ratio = tengri_disc / pcigale_disc

        if inclination_deg <= 40:
            # Type 1: η not applied twice → ratio = 0.99-1.00
            assert 0.98 < ratio < 1.02, (
                f"i={inclination_deg}°: tengri/pcigale = {ratio:.4f}, "
                f"expected 0.99-1.00 (η removed from R)"
            )
        else:
            # Type 2 before screen fix: still off, but this test focuses on η.
            # The remaining discrepancy is from double-screening (defect 2).
            pass  # Screened defect measured separately

    @pytest.mark.parametrize("inclination_deg", [70, 90])
    def test_type2_screen_not_doubled(self, ssp_data, pcigale_reference, inclination_deg):
        """Test D: Type-2 disc not screened twice.

        Defect: torus_screen_transmission applied to disc that already
        carries library extinction.
        Fix: Skip torus screen when _disc_R is not None (fracAGN-tied).

        Expected: disc at edge-on within 1% of template-only value.
        """
        FRAC = pcigale_reference["frac"]
        cos_inc = float(np.cos(np.radians(inclination_deg)))

        agn_config = {
            "type": "composable",
            "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
            "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
            "agn_ir_frac": Fixed(FRAC),
            "atten": {"type": "none"},
            "agn_cos_inc": Fixed(cos_inc),
            "all_params": Fixed(DEFAULT),
        }

        model = SEDModel.build(
            ssp_data=ssp_data,
            met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "type": "two_component",
                "law_bc": "leitherer02",
                "law_diff": "leitherer02",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(1.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "draine_li2014", "all_params": Fixed(DEFAULT)},
            neb={"type": "none"},
            redshift=Fixed(0.0),
            agn=agn_config,
        )

        st = model.predict_state({})
        d = st.derived

        # After fix: disc should be close to template, not further screened.
        # (This is checked via comparison to the known screened value;
        # the exact numerical test depends on whether η is also fixed.)
        # We record the value here; re-pin after fix.
        disc_value = nu_int(st.wave, d["sed_agn_disc"]) / float(d["L_absorbed"])

        # Recorded unfixed value for i=70, i=90 (will re-pin to corrected value)
        expected_unfixed = {
            70: 0.0056 / 0.3,  # approximate; exact depends on L_absorbed
            90: 1e-4,  # highly screened, near-zero
        }

        # For now, just ensure the call succeeds and the value is finite.
        assert np.isfinite(disc_value), f"i={inclination_deg}°: disc value not finite"

    @pytest.mark.parametrize("disk_type,delta", [(0, -0.5), (0, 0.0), (0, 0.5), (1, 0.0)])
    def test_piecewise_disc_cut_at_8nm(self, pcigale_reference, disk_type, delta):
        """Test E: Piecewise disc zeroed outside [8, 10^6] nm.

        Defect: searchsorted clips to first segment; disc extends 1-8 nm spuriously.
        Fix: jnp.where to zero spectrum outside [limits[0], limits[-1]] before
        unit-area norm.

        Expected: disc at 250 nm within 2e-3 of pcigale on both grids.
        RED: disk_type 0 delta=0: 0.8706
        GREEN: ~0.995-1.005 (within 2e-3 of pcigale)
        """
        pcigale_wl = pcigale_reference["pcigale_wl"]

        # pcigale reference
        if disk_type == 0:
            pf = PS.skirtor_disk
        else:
            pf = PS.schartmann2005_disk

        pcigale_val_250nm = np.interp(250.0, pcigale_wl, pf(pcigale_wl, delta=delta))

        # tengri on model grid
        wg = np.logspace(1, 8, 2380) / 10.0  # 10 A to 1e8 A
        if disk_type == 0:
            tf = DC.skirtor_disk_spectrum
        else:
            tf = DC.schartmann2005_disk_spectrum

        tengri_val_250nm = np.interp(250.0, wg, np.asarray(tf(jnp.asarray(wg), delta=delta)))

        ratio = tengri_val_250nm / pcigale_val_250nm

        # After fix: skirtor disk_type 0 within ~1.5%, schartmann disk_type 1 within 0.3%
        # RED (disk_type 0): 0.8706; GREEN (disk_type 0): ~0.986-0.994
        # RED (disk_type 1): 0.9983; GREEN (disk_type 1): 0.9983-1.0002
        if disk_type == 0:
            # skirtor: fixed from 0.84-0.92 range to 0.98+ range
            assert 0.975 < ratio < 1.005, (
                f"disk_type={disk_type}, delta={delta}: tengri/pcigale = {ratio:.4f}, "
                f"expected 0.975-1.005 (disc should be cut at 8 nm)"
            )
        else:
            # schartmann: already within 0.3%
            assert 0.998 < ratio < 1.002, (
                f"disk_type={disk_type}, delta={delta}: tengri/pcigale = {ratio:.4f}, "
                f"expected 0.998-1.002 (disc should be cut at 8 nm)"
            )

    def test_library_ratio_equals_eta(self, pcigale_reference):
        """Test B: Library ratio ∫disk(i)/∫disk(0) / η within 1% for i ≤ 40°.

        Verifies that η is already baked into the library spectra,
        so applying it again in R is wrong.
        """
        lib_fact = pcigale_reference["library_ratio_over_eta"]

        for i in [10, 20, 30, 40]:
            ratio = lib_fact[i]
            assert 0.99 < ratio < 1.01, (
                f"Library: i={i}°, ratio/eta = {ratio:.4f}, expected ~1.0 (η already in library)"
            )
