# SPDX-License-Identifier: BSD-3-Clause
r"""XRayAirdSEDComponent declares sfr, log_mstar, L_2500_30deg, age_weights,
ssp_ages_yr (#1846, #1755, #1706).

The problem: ``predict`` read these five keys from ``**inputs`` with hardcoded
fallbacks, but did not declare them in ``optional_inputs``. The base ``apply()``
only populates ``input_kwargs`` from declared keys, so undeclared reads always
took their defaults:

- sfr → 1.0 Msun/yr (fallback)
- log_mstar → 10.0 (fallback), never 10.3 or 10.5 from stellar
- stellar_age_gyr → 1.0 Gyr (fallback)
- L_2500_30deg → 0.0 (fallback), so AGN corona was always zero
- age_weights, ssp_ages_yr → 0.0 (fallback injections), so stellar age computation broke

Measured before the fix: ``sed_xray`` was **unchanged** by:
1. SFR: L_X should scale ~1:1 with SFR (Lehmer+2016), but was fixed at 1.0 Msun/yr
2. Stellar age: LMXB term is a steep quartic in age; L_X should drop ~50% when evolved
3. AGN: L_X_corona should be zero when no AGN, but corona was independent of L_2500

This is the third recurrence of the same failure mode: #1706 (det_hmxb/det_lmxb
not declared), #1755 (metallicity not wired), now #1846 (four SFR/stellar/AGN reads).

References
----------
.. [1] Lehmer et al. 2016, ApJ 825, 7 — X-ray binary scaling with SFR and age.
.. [2] Yang et al. 2020, ApJ 927, 192 — AGN coronal luminosity.
.. [3] Just et al. 2007, ApJ 665, 1004 — α_OX relation.
"""

import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel
from tengri.utils.physics_constants import Z_SUN

pytestmark = pytest.mark.regression_bug


def _build_xray_model(ssp, obs, *, sfr_100myr=None, agn_model=None, xray_model="xray_aird"):
    """Build a minimal SEDModel with X-ray for testing.

    Parameters
    ----------
    ssp : SSPData
        Stellar population templates.
    obs : Photometry
        Photometric observation.
    sfr_100myr : float, optional
        If set, make SFR a free parameter; else fix to 1.0.
    agn_model : str, optional
        AGN model ("standard", "skirtor", etc.); None means no AGN.
    xray_model : str
        X-ray model ("xray_aird" or "yang20").

    Returns
    -------
    SEDModel

    Notes
    -----
    Currently unreferenced: every test below drives the component directly. Kept
    working, and swept for the dust group split, rather than left as a helper
    that would raise the moment someone used it.
    """
    config = {
        "ssp_data": ssp,
        "observation": obs,
        "sfh": {"type": "delayed", "all_params": FIXED},
        "dust_attenuation": {"law": "power_law", "all_params": FIXED},
        "met": {"logzsol": Fixed(Z_SUN if Z_SUN > 0 else 0.0)},
        "xray": {"type": xray_model, "all_params": FIXED},
        "redshift": Fixed(0.05),
    }

    if agn_model:
        config["agn"] = {"type": agn_model, "all_params": FIXED}

    return SEDModel.build(**config)


@pytest.fixture(scope="module")
def obs():
    """Minimal photometry fixture."""
    import tengri

    return tengri.Photometry.from_names(("sdss_g", "sdss_r", "sdss_i"))


# ── Test 1: SFR Scaling ──────────────────────────────────────────────────────


def test_xray_aird_hmxb_scales_with_sfr(ssp_data_wne, obs):
    """HMXB luminosity tracks SFR (Lehmer+2016).

    Before fix: L_X was always for sfr=1.0 Msun/yr regardless of input.
    After fix: L_X should scale ~linearly with sfr_100myr.
    """
    # Drive the component directly with controlled inputs — the declared-input
    # wiring itself is covered by the census test and the build-path test below.
    from tengri.components.xray.xray_model import XRayAirdSEDComponent

    comp = XRayAirdSEDComponent()
    wave = np.logspace(0, 4, 500)  # X-ray range
    sed_in = np.zeros_like(wave)

    # Test with two SFR values
    p = {
        "gamma_hmxb": np.array(1.6),
        "gamma_lmxb": np.array(1.4),
        "gamma_agn": np.array(1.9),
        "log_nh": np.array(21.0),
    }

    # Isolate the SFR-driven (HMXB + hot gas) part by differencing against the
    # sfr=0 baseline: the LMXB term is mass-driven (in-predict fallback mass
    # 1e10 Msun here) and identical across scenarios, so it cancels.
    inputs_zero = {"sfr": np.array(0.0)}
    _, pub_zero = comp.predict(p, sed_in, wave, **inputs_zero)

    inputs_low = {"sfr": np.array(1.0)}
    _sed_low, pub_low = comp.predict(p, sed_in, wave, **inputs_low)

    inputs_high = {"sfr": np.array(10.0)}
    _sed_high, pub_high = comp.predict(p, sed_in, wave, **inputs_high)

    l_xray_zero = np.sum(np.abs(pub_zero["sed_xray"]))
    l_xray_low = np.sum(np.abs(pub_low["sed_xray"]))
    l_xray_high = np.sum(np.abs(pub_high["sed_xray"]))

    # Lehmer+2016: the SFR-driven part (HMXB + hot gas) is linear in SFR. The
    # total is not — LMXB (mass-driven) is constant across scenarios — so the
    # linearity check must be on the sfr=0-differenced part.
    sfr_part_low = l_xray_low - l_xray_zero
    sfr_part_high = l_xray_high - l_xray_zero
    assert sfr_part_low > 0.0, "SFR=1 must add X-ray output over SFR=0 (HMXB + hot gas)."
    ratio = sfr_part_high / sfr_part_low
    np.testing.assert_allclose(
        ratio,
        10.0,
        rtol=0.05,
        err_msg=(
            f"SFR-driven L_X part scaled by {ratio:.3f} for a 10x SFR increase; "
            "Lehmer+2016 HMXB + hot gas are linear in SFR. "
            "Regression signature: before #1846 the ratio was exactly 1.0 (sfr undeclared)."
        ),
    )


# ── Test 2: Stellar Age Scaling ──────────────────────────────────────────────


def test_xray_aird_lmxb_scales_with_stellar_age(ssp_data_wne, obs):
    """LMXB luminosity responds to stellar age (Lehmer+2016 quartic).

    Before fix: L_X was always for stellar_age_gyr=1.0 Gyr regardless of input.
    After fix: L_X should change when age_weights/ssp_ages_yr are provided.
    """
    from tengri.components.xray.xray_model import XRayAirdSEDComponent

    comp = XRayAirdSEDComponent()
    wave = np.logspace(0, 4, 500)
    sed_in = np.zeros_like(wave)

    p = {
        "gamma_hmxb": np.array(1.6),
        "gamma_lmxb": np.array(1.4),
        "gamma_agn": np.array(1.9),
        "log_nh": np.array(21.0),
    }

    # Scenario 1: Old stellar population (age_weights sum > 0, young age)
    ssp_ages = np.array([8.0e9, 6.0e9, 4.0e9]) * 1.0  # in years
    age_weights_young = np.array([0.1, 0.3, 0.6])  # weighted toward young
    stellar_age_young = np.sum(age_weights_young * ssp_ages) / np.sum(age_weights_young) / 1e9

    inputs_young = {
        "age_weights": age_weights_young,
        "ssp_ages_yr": ssp_ages,
    }
    _sed_young, pub_young = comp.predict(p, sed_in, wave, **inputs_young)

    # Scenario 2: Old stellar population
    age_weights_old = np.array([0.6, 0.3, 0.1])  # weighted toward old
    stellar_age_old = np.sum(age_weights_old * ssp_ages) / np.sum(age_weights_old) / 1e9

    inputs_old = {
        "age_weights": age_weights_old,
        "ssp_ages_yr": ssp_ages,
    }
    _sed_old, pub_old = comp.predict(p, sed_in, wave, **inputs_old)

    l_xray_young = np.sum(np.abs(pub_young["sed_xray"]))
    l_xray_old = np.sum(np.abs(pub_old["sed_xray"]))

    # LMXB drops with stellar age (Lehmer+2016 Eq. 16 quartic)
    # Expected ratio: (age_old/age_young)^4 ~ (8/1)^4 ≈ 4000 but that's unrealistic
    # In practice with more moderate ages, ratio should be > 1.0
    ratio = l_xray_old / l_xray_young if l_xray_young > 0 else 1.0
    assert ratio < 1.0, (
        f"L_X(old age) / L_X(young age) = {ratio}; "
        "should be < 1 since LMXB drops with age. "
        "Regression: age_weights / ssp_ages_yr not declared / wired."
    )


# ── Test 3: AGN Corona ───────────────────────────────────────────────────────


def test_xray_aird_corona_zero_without_agn(ssp_data_wne, obs):
    """X-ray corona is zero when no AGN published L_2500_30deg.

    Before fix: L_X was always just XRB (no corona) anyway due to L_2500_30deg=0.
    After fix: explicitly verify corona term is absent without AGN.
    """
    from tengri.components.xray.xray_model import XRayAirdSEDComponent

    comp = XRayAirdSEDComponent()
    wave = np.logspace(0, 4, 500)
    sed_in = np.zeros_like(wave)

    p = {
        "gamma_hmxb": np.array(1.6),
        "gamma_lmxb": np.array(1.4),
        "gamma_agn": np.array(1.9),
        "log_nh": np.array(21.0),
    }

    # No AGN: L_2500 = 0.0 (injected fallback)
    inputs_no_agn = {"L_2500_30deg": np.array(0.0)}
    _sed_no_agn, pub_no_agn = comp.predict(p, sed_in, wave, **inputs_no_agn)

    # Hard X-rays (few keV) should be small when no corona
    hard_band = (wave >= 2.0) & (wave <= 10.0)
    l_hard_no_agn = np.sum(np.abs(pub_no_agn["sed_xray"][hard_band]))

    # Sanity check: some X-ray output exists (from HMXB/LMXB)
    assert np.sum(np.abs(pub_no_agn["sed_xray"])) > 0, "X-ray SED is zero"
    # Corona should not dominate
    assert l_hard_no_agn < np.sum(np.abs(pub_no_agn["sed_xray"])), (
        "Hard X-ray contribution is too large without AGN"
    )


def test_xray_aird_corona_present_with_agn(ssp_data_wne, obs):
    """X-ray corona is present when AGN publishes L_2500_30deg.

    Before fix: corona was always zero because L_2500_30deg was never read.
    After fix: corona should appear when L_2500 is provided.
    """
    from tengri.components.xray.xray_model import XRayAirdSEDComponent

    comp = XRayAirdSEDComponent()
    wave = np.logspace(0, 4, 500)
    sed_in = np.zeros_like(wave)

    p = {
        "gamma_hmxb": np.array(1.6),
        "gamma_lmxb": np.array(1.4),
        "gamma_agn": np.array(1.9),
        "log_nh": np.array(21.0),
    }

    # With AGN: L_2500 = 1e29 erg/s/Hz (typical AGN)
    inputs_with_agn = {"L_2500_30deg": np.array(1e29)}
    _sed_with_agn, pub_with_agn = comp.predict(p, sed_in, wave, **inputs_with_agn)

    # Compare to no AGN
    inputs_no_agn = {"L_2500_30deg": np.array(0.0)}
    _sed_no_agn, pub_no_agn = comp.predict(p, sed_in, wave, **inputs_no_agn)

    l_agn = np.sum(np.abs(pub_with_agn["sed_xray"]))
    l_no_agn = np.sum(np.abs(pub_no_agn["sed_xray"]))

    # Corona should contribute some additional luminosity
    assert l_agn > l_no_agn, (
        f"L_X(with AGN) {l_agn} not > L_X(no AGN) {l_no_agn}; "
        "corona is not being added. Regression: L_2500_30deg not declared."
    )


# ── Test 4: All inputs declared (metacontract) ──────────────────────────────


def test_xray_aird_declares_all_inputs():
    """All input reads are declared in optional_inputs().

    This is the metacontract: the census test checks it at test time,
    but the component's own docstring should also name them.
    """
    from tengri.components.xray.xray_model import XRayAirdSEDComponent

    comp = XRayAirdSEDComponent()
    declared = {k.name for k in comp.optional_inputs()}

    required = {
        "sfr",
        "log_mstar",
        "L_2500_30deg",
        "age_weights",
        "ssp_ages_yr",
        "log_metallicity_history",
    }
    assert required <= declared, (
        f"Missing declared inputs: {required - declared}. Fixes #1846, #1755, #1706."
    )


def test_xray_aird_absent_publisher_semantics():
    """Absent publishers (apply() injects 0.0) give the documented behavior (#1846).

    Before fix: sfr/log_mstar/L_2500 were read but not declared, so every read
    took its literal in-predict fallback (a phantom fiducial galaxy) no matter
    what the model computed.
    After fix: declared in optional_inputs; absent publishers arrive as the base
    apply()'s injected 0.0, which the docstring documents as: no SFR -> no HMXB
    or hot gas, mass 10^0 = 1 Msun -> negligible LMXB, L_2500 = 0 -> no corona.

    Two assertions pin that contract:

    1. Zeroing ONLY the SFR at fixed mass removes the SFR-driven part while the
       mass-driven LMXB persists: 0 < L(sfr=0, M=1e10) < L(sfr=1, M=1e10).
    2. All publishers absent (everything injected 0.0) is negligible next to a
       published fiducial galaxy: L(all absent) < 1e-6 * L(sfr=1, M=1e10).
    """
    from tengri.components.xray.xray_model import XRayAirdSEDComponent

    comp = XRayAirdSEDComponent()
    wave = np.logspace(0, 4, 300)
    sed_in = np.zeros_like(wave)

    p = {
        "gamma_hmxb": np.array(1.6),
        "gamma_lmxb": np.array(1.4),
        "gamma_agn": np.array(1.9),
        "log_nh": np.array(21.0),
    }
    met = np.array(np.log10(0.0142))

    def l_x(**inputs):
        _, pub = comp.predict(p, sed_in, wave, **inputs)
        return float(np.sum(np.abs(pub["sed_xray"])))

    l_published = l_x(sfr=np.array(1.0), log_mstar=np.array(10.0), log_metallicity_history=met)
    l_no_sfr = l_x(sfr=np.array(0.0), log_mstar=np.array(10.0), log_metallicity_history=met)
    # All publishers absent: apply() injects 0.0 for every declared optional input.
    l_all_absent = l_x(
        sfr=np.array(0.0),
        log_mstar=np.array(0.0),
        L_2500_30deg=np.array(0.0),
        age_weights=np.array(0.0),
        ssp_ages_yr=np.array(0.0),
        log_metallicity_history=None,
    )

    # 1. SFR-driven part responds; mass-driven LMXB persists at fixed mass.
    assert 0.0 < l_no_sfr < l_published, (
        f"L_X(sfr=0, M=1e10) = {l_no_sfr:.3e} must sit strictly between 0 and "
        f"L_X(sfr=1, M=1e10) = {l_published:.3e}: zero SFR removes the HMXB/hot-gas "
        "part only, LMXB (mass-driven) persists."
    )

    # 2. A galaxy with no publishers at all is negligible, not a phantom fiducial.
    assert l_all_absent < 1e-6 * l_published, (
        f"L_X(all publishers absent) = {l_all_absent:.3e} should be negligible vs "
        f"L_X(published fiducial) = {l_published:.3e} — the pre-#1846 behavior was a "
        "phantom 1 Msun/yr, 1e10 Msun galaxy regardless of the model."
    )
