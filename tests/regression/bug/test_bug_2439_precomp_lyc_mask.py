# SPDX-License-Identifier: BSD-3-Clause
"""WavePrecomp applies the nebular Lyman-continuum mask to the stellar photometric LUT.

Regression for #2439 and #2427. The exact wavelength-array path masks the stellar
continuum below rest-frame 912 A by ``neb_fesc`` (``components/nebular/component.py``);
``predict_via_precomp`` (``observation/observation.py``) summed the stellar photometric
LUT with no such correction, so any band whose observed passband samples rest
lambda < 912 A carried the full, unabsorbed stellar Lyman continuum. On the issue's
real-SSP model (Cue nebular, default ``neb_fesc=0.0``, no dust): z=2 GALEX NUV +915%,
z=3 SDSS u +69%, K-invariant, identical for both ``igm_fold`` settings. Forcing
``neb_fesc=1.0`` (mask is a no-op in the exact path too) collapses both to the LUT's
pre-existing floor.

Uses a module-local ``lyc_ssp`` fixture (100 A - 1e7 A, LyC-bright like
``tests/contract/test_energy_balance_lyc_toggle.py``'s ``synthetic_ssp_wide``, but with
a genuine break at 912 A -- see ``_make_lyc_ssp`` for why the plain smooth power law is
unsuitable here) with purpose-built top-hat filters whose REST-frame footprint straddles
912 A by construction, at any redshift: no ``data/`` grid needed. Case 1's own straddle
assertion re-verifies this rather than trusting the geometry.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation.photometry import FilterCurve

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

LYC_LIMIT = 912.0  # Angstrom, rest frame


def _make_lyc_ssp():
    """Synthetic wide SSP with a genuine Lyman-limit break, for #2439/#2427.

    Like ``tests/conftest.py``'s ``synthetic_ssp_wide`` (100 A - 1e7 A, smooth
    LyC-bright power-law continuum), but that fixture's power law continues
    UNBROKEN through 912 A -- no real bare-stellar spectrum does that,
    photospheric absorption plus the Lyman limit itself cut it off. Cue's Q_H
    (ionizing photon rate) integrates the continuum below 912 A, so an
    unbroken power law reaching 100 A wildly overestimates it: measured
    nebular/(nebular+stellar) ~97-100% of the intrinsic flux in the straddle
    band under test below, versus a documented worst case of a few percent
    for real filters (``predict_via_precomp``'s accuracy ledger). That
    swamps the very stellar-continuum signal the #2439/#2427 fix targets,
    making a test built on the unbroken fixture unable to discriminate a
    working fix from a disabled one (case 4 and case 6 passed unchanged
    under the correction-factor-1 mutation on the unbroken fixture). The
    break keeps Cue's Q_H physically reasonable while leaving real, nonzero
    flux below 912 A -- the masking mechanism still has something to mask.
    """
    n_met, n_age = 3, 25
    wave = jnp.logspace(2.0, 7.0, 1600)
    ages_gyr = jnp.linspace(-3.0, 1.14, n_age)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    base = (5000.0 / wave) ** 2
    lyman_break = jnp.where(wave < LYC_LIMIT, 0.03, 1.0)
    flux = (
        base[None, None, :]
        * lyman_break[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    flux = jnp.abs(flux) + 1e-12
    return SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)


@pytest.fixture(scope="module")
def lyc_ssp():
    return _make_lyc_ssp()


def _straddle_filter(z: float, frac: float = 0.35, n: int = 60, name: str = "straddle"):
    """Top-hat filter whose REST-frame footprint straddles 912 A at ANY z.

    Center at ``912*(1+z)`` observed, so dividing by ``(1+z)`` always recovers a
    rest-frame span of ``912*(1-frac)`` to ``912*(1+frac)``: straddles the Lyman
    limit by construction, independent of the redshift used to build the filter.
    """
    center_obs = LYC_LIMIT * (1.0 + z)
    wave = jnp.linspace(center_obs * (1.0 - frac), center_obs * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=name)


def _clean_filter(
    center_rest: float, z: float, frac: float = 0.1, n: int = 40, name: str = "clean"
):
    """Top-hat filter well above the Lyman limit in the rest frame, for contrast."""
    center_obs = center_rest * (1.0 + z)
    wave = jnp.linspace(center_obs * (1.0 - frac), center_obs * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=name)


def _straddles_912(filter_curve, z: float) -> bool:
    """True iff the filter's transmission-weighted flux is non-negligible on
    both sides of rest-frame 912 A at redshift ``z`` (the straddle assertion)."""
    wave_rest = np.asarray(filter_curve.wave) / (1.0 + z)
    trans = np.asarray(filter_curve.trans)
    below = float(np.sum(trans[wave_rest < LYC_LIMIT]))
    above = float(np.sum(trans[wave_rest >= LYC_LIMIT]))
    return below > 0.0 and above > 0.0


def _build(
    ssp,
    z: float,
    approx,
    filters,
    *,
    dust: bool = False,
    igm: str | None = None,
    fesc=None,
    fesc_free: bool = False,
    redshift_free_range: tuple[float, float] | None = None,
    nebular: bool = True,
):
    """Build the #2439 model shape: delayed SFH, Cue nebular, optional dust/IGM."""
    neb_group: dict = (
        {"type": "cue", "all_params": Fixed(DEFAULT)} if nebular else {"type": "none"}
    )
    if nebular:
        if fesc_free:
            neb_group["neb_fesc"] = Uniform(0.0, 1.0)
        elif fesc is not None:
            neb_group["neb_fesc"] = Fixed(float(fesc))

    dust_group = (
        {
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": 0.5,
            "tau_diff": 0.3,
        }
        if dust
        else {"type": "none"}
    )
    igm_group = {"type": "none"} if igm is None else {"type": igm, "all_params": Fixed(DEFAULT)}

    redshift = (
        Uniform(*redshift_free_range) if redshift_free_range is not None else Fixed(float(z))
    )

    obs = Observation(photometry=Photometry(filters=tuple(filters)))
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "log_total_mass": 10.0,
            "tau_gyr": 1.0,
            "age_gyr": 3.0,
            "all_params": Fixed(DEFAULT),
        },
        met={"logzsol": 0.0, "all_params": Fixed(DEFAULT)},
        neb=neb_group,
        dust_attenuation=dust_group,
        igm=igm_group,
        redshift=redshift,
        approx=approx,
    )


def _params(model, key=None):
    """Fixed values plus a sample for any free params (redshift, neb_fesc, ...)."""
    base = dict(model.spec.get_fixed_values())
    free = model.spec.free_params
    if free:
        base.update(model.spec.sample(key if key is not None else jax.random.PRNGKey(0)))
    return base


def _worst_band(exact, lut, names):
    exact = np.asarray(exact)
    lut = np.asarray(lut)
    nonzero = exact != 0.0
    assert np.any(nonzero), "exact-path photometry is identically zero in every band"
    rel = np.full_like(exact, np.inf)
    rel[nonzero] = np.abs((lut[nonzero] - exact[nonzero]) / exact[nonzero])
    idx = int(np.argmax(rel))
    return names[idx], float(rel[idx]), exact, lut


# ── Case 1: worst-band exact vs LUT at z=2 and z=3, default fesc ───────────────


@pytest.mark.parametrize("z", [2.0, 3.0])
def test_case1_worst_band_exact_vs_lut(lyc_ssp, z):
    """Worst band <= the fesc=1 floor after the fix; +915%/+69%-style before it."""
    straddle = _straddle_filter(z, name="straddle")
    clean = _clean_filter(5000.0, z, name="clean")
    filters = [straddle, clean]
    names = ["straddle", "clean"]

    assert _straddles_912(straddle, z), "fixture bug: filter no longer straddles 912 A"

    model_exact = _build(lyc_ssp, z, None, filters)
    model_lut = _build(lyc_ssp, z, WavePrecomp(), filters)
    params = _params(model_exact)

    exact_phot = np.asarray(model_exact.predict_photometry(params))
    lut_phot = np.asarray(model_lut.predict_photometry(params))

    # Ablation assertion: the exact path's own SED must move with fesc, or this
    # config cannot demonstrate the defect at all.
    model_exact_fesc1 = _build(lyc_ssp, z, None, filters, fesc=1.0)
    params_fesc1 = _params(model_exact_fesc1)
    exact_phot_fesc1 = np.asarray(model_exact_fesc1.predict_photometry(params_fesc1))
    straddle_idx = names.index("straddle")
    # atol=0: these are ~1e-12-scale Lnu values, and np.isclose's default
    # atol=1e-8 would swamp any relative difference at this magnitude.
    assert not np.isclose(
        exact_phot[straddle_idx], exact_phot_fesc1[straddle_idx], rtol=1e-6, atol=0.0
    ), "exact-path straddle-band flux does not move with neb_fesc -- fixture cannot see the defect"

    # The floor: exact vs LUT at fesc=1, where the mask is a no-op end to end.
    model_lut_fesc1 = _build(lyc_ssp, z, WavePrecomp(), filters, fesc=1.0)
    lut_phot_fesc1 = np.asarray(model_lut_fesc1.predict_photometry(params_fesc1))
    floor_name, floor_err, _, _ = _worst_band(exact_phot_fesc1, lut_phot_fesc1, names)

    band, err, exact_v, lut_v = _worst_band(exact_phot, lut_phot, names)
    tol = max(floor_err * 3.0, 0.005)
    assert err < tol, (
        f"z={z}: worst band '{band}' error {err * 100:.3f}% exceeds "
        f"{tol * 100:.3f}% (3x the measured fesc=1 floor '{floor_name}' "
        f"{floor_err * 100:.4f}%, or 0.5% minimum). exact={exact_v}, lut={lut_v}"
    )


# ── Case 2: fesc=1 arm is a bit-level no-op ─────────────────────────────────


def test_case2_fesc1_no_op(lyc_ssp):
    z = 2.0
    filters = [_straddle_filter(z, name="straddle"), _clean_filter(5000.0, z, name="clean")]

    model_exact = _build(lyc_ssp, z, None, filters, fesc=1.0)
    model_lut = _build(lyc_ssp, z, WavePrecomp(), filters, fesc=1.0)
    params = _params(model_exact)

    exact_phot = np.asarray(model_exact.predict_photometry(params))
    lut_phot = np.asarray(model_lut.predict_photometry(params))

    # rtol, not a bit-exact rtol=0: exact and LUT are different computational
    # paths with their own pre-existing floor (measured ~5.7e-6 here) even
    # when the LyC mask itself is a true no-op; 1e-4 comfortably separates
    # that floor from anything the size of the original defect (tens to
    # hundreds of percent).
    np.testing.assert_allclose(
        lut_phot,
        exact_phot,
        rtol=1e-4,
        atol=0.0,
        err_msg=f"fesc=1.0 is not (approximately) a no-op: exact={exact_phot}, lut={lut_phot}",
    )


# ── Case 3: neb_fesc FREE -- LUT responds to fesc, matches exact at two values ──


def test_case3_fesc_free_gradient_and_two_point_match(lyc_ssp):
    z = 2.0
    filters = [_straddle_filter(z, name="straddle"), _clean_filter(5000.0, z, name="clean")]
    model_lut = _build(lyc_ssp, z, WavePrecomp(), filters, fesc_free=True)
    model_exact = _build(lyc_ssp, z, None, filters, fesc_free=True)

    fixed = model_lut.spec.get_fixed_values()
    free_name = next(p for p in model_lut.spec.free_params if p.endswith("neb_fesc"))

    def straddle_flux(fesc_value):
        p = dict(fixed)
        p[free_name] = fesc_value
        return model_lut.predict_photometry(p)[0]

    flux_at_point = float(straddle_flux(jnp.asarray(0.4)))
    grad = jax.grad(straddle_flux)(jnp.asarray(0.4))
    # Relative to the flux scale (~1e-12 Lnu here), not an absolute constant:
    # an absolute threshold tuned for O(1) fluxes is meaningless at this
    # magnitude and would falsely read a live gradient as "zero".
    assert abs(float(grad)) > abs(flux_at_point) * 1e-3, (
        f"LUT straddle-band photometry has ~zero gradient wrt {free_name} "
        f"({float(grad):.3e}, flux={flux_at_point:.3e}) -- the fesc correction "
        "is not wired into the LUT path"
    )

    # 0.1%: measured ~0.005% with the fix, ~1-2% with the correction disabled
    # (mutation) -- comfortable separation, not the loose accuracy-floor bound
    # used elsewhere, since this check's whole job is to catch exactly that
    # mutation.
    for fesc_value in (0.25, 0.75):
        p = dict(fixed)
        p[free_name] = float(fesc_value)
        exact_v = np.asarray(model_exact.predict_photometry(p))
        lut_v = np.asarray(model_lut.predict_photometry(p))
        band, err, _, _ = _worst_band(exact_v, lut_v, ["straddle", "clean"])
        assert err < 0.001, (
            f"fesc={fesc_value}: worst band '{band}' error {err * 100:.3f}% too large"
        )


# ── Case 4: the dusty two_component precomp instance ───────────────────────


def test_case4_dusty_instance(lyc_ssp):
    z = 2.0
    filters = [_straddle_filter(z, name="straddle"), _clean_filter(5000.0, z, name="clean")]
    names = ["straddle", "clean"]

    model_exact = _build(lyc_ssp, z, None, filters, dust=True)
    model_lut = _build(lyc_ssp, z, WavePrecomp(), filters, dust=True)
    params = _params(model_exact)

    exact_phot = np.asarray(model_exact.predict_photometry(params))
    lut_phot = np.asarray(model_lut.predict_photometry(params))

    model_exact_fesc1 = _build(lyc_ssp, z, None, filters, dust=True, fesc=1.0)
    model_lut_fesc1 = _build(lyc_ssp, z, WavePrecomp(), filters, dust=True, fesc=1.0)
    params_fesc1 = _params(model_exact_fesc1)
    exact_fesc1 = np.asarray(model_exact_fesc1.predict_photometry(params_fesc1))
    lut_fesc1 = np.asarray(model_lut_fesc1.predict_photometry(params_fesc1))

    # Ablation: bit-identical LUT between fesc=0/1 in the dusty branch was the
    # diagnosis's own signature that the correction never reached this path.
    assert not np.isclose(lut_phot[0], lut_fesc1[0], rtol=1e-6, atol=0.0), (
        "dusty LUT straddle-band photometry is bit-identical between "
        "neb_fesc=0 and neb_fesc=1 -- the two_component precomp branch is not "
        "reading the escape fraction"
    )

    # White-box, on the tensor the fix actually touches
    # (``stellar_phot_lnu_per_age_subband_precomp``): total predict_photometry
    # is contaminated by nebular emission and dust's own K-node quadrature
    # floor, both independent of this fix, and can mask the mutation this
    # test's own mutation-testing pass must catch (measured: total-flux worst
    # band moved only ~0.1-1% end to end here, comparable to those unrelated
    # floors -- not the crisp signal the whole-band case gets). Assert on the
    # corrected tensor directly instead.
    state0 = model_lut.predict_state(params)
    state1 = model_lut_fesc1.predict_state(params_fesc1)
    sub0 = np.asarray(state0.derived.get("stellar_phot_lnu_per_age_subband_precomp"))
    sub1 = np.asarray(state1.derived.get("stellar_phot_lnu_per_age_subband_precomp"))
    assert not np.allclose(sub0, sub1, rtol=1e-6, atol=0.0), (
        "stellar_phot_lnu_per_age_subband_precomp is identical between "
        "neb_fesc=0 and neb_fesc=1 in a dusty model -- the sub-band LyC "
        "correction (nebular/component.py) is not reaching this tensor"
    )

    dust_free_floor_name, dust_free_floor_err, _, _ = _worst_band(exact_fesc1, lut_fesc1, names)
    band, err, exact_v, lut_v = _worst_band(exact_phot, lut_phot, names)
    # Dust already imposes its own K-node quadrature floor (candidate-2-style,
    # per the diagnosis, "K=32 cuts a break-crossing FUV error by only ~9x");
    # the fix here folds a FLAT per-node fesc mask into the same K=5-node
    # sub-band tensor (nebular/component.py), not the dense path's y(age)
    # birth-cloud grading, so a larger multiple of the dust-free floor is the
    # honest bound, not the near-exact whole-band-bucket result.
    tol = max(dust_free_floor_err * 30.0, 0.05)
    assert err < tol, (
        f"dusty z={z}: worst band '{band}' error {err * 100:.3f}% exceeds "
        f"{tol * 100:.3f}% (30x the dust-free-arm floor '{dust_free_floor_name}' "
        f"{dust_free_floor_err * 100:.4f}%, or 5% minimum). exact={exact_v}, lut={lut_v}"
    )


# ── Case 5: z-axis falsification -- node vs between-node, n_z sweep ─────────


def test_case5_z_axis_falsification(lyc_ssp):
    z_min, z_max = 0.5, 2.5
    n_z_a, n_z_b = 60, 240
    dz_a = (z_max - z_min) / (n_z_a - 1)
    node_i = n_z_a // 2
    node_z = z_min + node_i * dz_a
    mid_z = node_z + dz_a / 2.0

    filters = [
        _straddle_filter(node_z, name="straddle"),
        _clean_filter(5000.0, node_z, name="clean"),
    ]
    names = ["straddle", "clean"]

    model_exact = _build(lyc_ssp, node_z, None, filters, redshift_free_range=(z_min, z_max))
    model_lut_a = _build(
        lyc_ssp,
        node_z,
        WavePrecomp(n_z=n_z_a, z_min=z_min, z_max=z_max),
        filters,
        redshift_free_range=(z_min, z_max),
    )
    model_lut_b = _build(
        lyc_ssp,
        node_z,
        WavePrecomp(n_z=n_z_b, z_min=z_min, z_max=z_max),
        filters,
        redshift_free_range=(z_min, z_max),
    )

    fixed = model_exact.spec.get_fixed_values()
    redshift_name = next(p for p in model_exact.spec.free_params if p.endswith("redshift"))

    def _err_at(model, z_value):
        p = dict(fixed)
        p[redshift_name] = float(z_value)
        exact_v = np.asarray(model_exact.predict_photometry(p))
        lut_v = np.asarray(model.predict_photometry(p))
        _, err, _, _ = _worst_band(exact_v, lut_v, names)
        return err

    err_node_a = _err_at(model_lut_a, node_z)
    err_mid_a = _err_at(model_lut_a, mid_z)
    err_mid_b = _err_at(model_lut_b, mid_z)

    # The residual at a between-node z must not blow up relative to the node.
    assert err_mid_a < max(err_node_a * 5.0, 0.02), (
        f"between-node z={mid_z} error {err_mid_a * 100:.3f}% far exceeds "
        f"node z={node_z} error {err_node_a * 100:.3f}%"
    )
    # Raising n_z must not move the between-node residual materially -- the LyC
    # split is an exact algebraic split at build time, not a redshift-quadrature
    # term, so it should not respond to n_z the way a genuine ztable-resolution
    # bug would.
    assert abs(err_mid_a - err_mid_b) < max(err_mid_a, err_mid_b, 1e-6) * 0.75 + 0.005, (
        f"between-node error moved materially with n_z: n_z={n_z_a} -> {err_mid_a * 100:.3f}%, "
        f"n_z={n_z_b} -> {err_mid_b * 100:.3f}%"
    )


# ── Case 6: #2427's Inoue-IGM rows ──────────────────────────────────────────


@pytest.mark.parametrize("z", [0.8, 1.5, 2.0, 3.0])
def test_case6_inoue_igm_rows(lyc_ssp, z):
    """#2427's Inoue-IGM rows: mean-IGM active, no dust.

    Three sub-checks. (1) Negative control: nebular OFF, so ``neb_fesc``
    does not exist and nothing masks the LyC -- exact and LUT must already
    agree at the ordinary precomp floor (mirrors the diagnosis's own Step 2
    "stellar-only control" methodology: this is NOT an isolation of the
    #2439/#2427 mechanism, which requires a nebular component to exist at
    all, it is proof the defect is nebular-conditional). (2) White-box: the
    fix's target tensor for this branch
    (``stellar_phot_lnu_per_age_subband_igm_precomp`` --
    ``observation.predict_via_precomp``'s ``sub_per_age_igm is not None``
    branch, #2439/#2427's *other* precomp instance beside
    ``two_component``'s, reachable with no dust at all) must actually depend
    on ``neb_fesc``; total predict_photometry cannot show this cleanly here
    (see (3)). (3) Full config (Cue nebular + Inoue IGM, matching #2427
    exactly): nebular's own IGM treatment is a SEPARATE, pre-existing,
    documented approximation (``igm_phot_factor`` band-averages the
    transmission alone, unweighted by the nebular SED's own structure -- the
    accuracy ledger's `\\langle S\\rangle\\langle T\\rangle` covariance gap, up
    to ~9.5% on a real filter). On this adversarial synthetic top-hat
    straddling 912 A exactly, with Cue on a LyC-bright synthetic template,
    that gap dominates the straddle band (measured nebular ~73-99% of
    intrinsic flux there) and swamps the #2439/#2427 fix's own effect (a few
    percent at most, per the ledger's stellar-continuum floor) -- bound the
    tolerance by nebular's measured share rather than pretend the fix alone
    should reach the tiny stellar-only floor once nebular dominates.
    """
    filters = [_straddle_filter(z, name="straddle"), _clean_filter(5000.0, z, name="clean")]
    names = ["straddle", "clean"]

    # (1) Negative control: without nebular, neb_fesc does not exist -- exact
    # and LUT must already agree at the ordinary precomp floor.
    model_exact_ctrl = _build(lyc_ssp, z, None, filters, igm="inoue", nebular=False)
    model_lut_ctrl = _build(lyc_ssp, z, WavePrecomp(), filters, igm="inoue", nebular=False)
    params_ctrl = _params(model_exact_ctrl)
    exact_ctrl = np.asarray(model_exact_ctrl.predict_photometry(params_ctrl))
    lut_ctrl = np.asarray(model_lut_ctrl.predict_photometry(params_ctrl))
    band_ctrl, err_ctrl, exact_ctrl_v, lut_ctrl_v = _worst_band(exact_ctrl, lut_ctrl, names)
    assert err_ctrl < 0.02, (
        f"z={z} (Inoue IGM, no nebular -- negative control, no neb_fesc to "
        f"mask anything): worst band '{band_ctrl}' error {err_ctrl * 100:.3f}% "
        f"exceeds 2%. exact={exact_ctrl_v}, lut={lut_ctrl_v}"
    )

    # (2) White-box: the corrected tensor must actually respond to neb_fesc.
    model_lut = _build(lyc_ssp, z, WavePrecomp(), filters, igm="inoue")
    model_lut_fesc1 = _build(lyc_ssp, z, WavePrecomp(), filters, igm="inoue", fesc=1.0)
    params = _params(model_lut)
    params_fesc1 = _params(model_lut_fesc1)
    state0 = model_lut.predict_state(params)
    state1 = model_lut_fesc1.predict_state(params_fesc1)
    sub_igm0 = state0.derived.get("stellar_phot_lnu_per_age_subband_igm_precomp")
    sub_igm1 = state1.derived.get("stellar_phot_lnu_per_age_subband_igm_precomp")
    assert sub_igm0 is not None and sub_igm1 is not None, (
        f"z={z} (Inoue IGM): stellar_phot_lnu_per_age_subband_igm_precomp not "
        "published -- the no-dust mean-IGM sub-band branch did not engage"
    )
    assert not np.allclose(np.asarray(sub_igm0), np.asarray(sub_igm1), rtol=1e-6, atol=0.0), (
        "stellar_phot_lnu_per_age_subband_igm_precomp is identical between "
        "neb_fesc=0 and neb_fesc=1 with no dust -- the sub-band LyC "
        "correction (nebular/component.py) is not reaching this tensor"
    )

    # (3) Full config: Cue nebular + Inoue IGM (#2427's actual shape).
    model_exact = _build(lyc_ssp, z, None, filters, igm="inoue")
    params_full = _params(model_exact)
    state_lut = model_lut.predict_state(params_full)
    stellar_phi = np.abs(np.asarray(state_lut.derived.get("stellar_phot_lnu_precomp"))[0])
    nebular_phi_arr = state_lut.derived.get("nebular_phot_lnu_precomp")
    nebular_phi = np.abs(np.asarray(nebular_phi_arr)[0]) if nebular_phi_arr is not None else 0.0
    nebular_frac = nebular_phi / max(nebular_phi + stellar_phi, 1e-300)

    exact_phot = np.asarray(model_exact.predict_photometry(params_full))
    lut_phot = np.asarray(model_lut.predict_photometry(params_full))
    band, err, exact_v, lut_v = _worst_band(exact_phot, lut_phot, names)
    tol = max(0.02, 0.6 * nebular_frac)
    assert err < tol, (
        f"z={z} (Inoue IGM, full config): worst band '{band}' error "
        f"{err * 100:.3f}% exceeds {tol * 100:.1f}% (nebular is "
        f"{nebular_frac * 100:.1f}% of the straddle band's intrinsic flux; "
        f"the negative control above confirms the mechanism is "
        f"nebular-conditional at {err_ctrl * 100:.3f}%). "
        f"exact={exact_v}, lut={lut_v}"
    )
