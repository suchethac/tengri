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

Fix-round additions (2026-09, on top of the original six cases): R1 forces a
sub-band edge at the physical Lyman limit, gated on a live nebular mask;
R2 has each consumer (dusty ``two_component``, single-screen dust, dust-free
mean-IGM) apply its OWN dense-path rule to a shared
``stellar_subband_lyc_factor_precomp`` factor, with no double-count; R3 adds
four conservation invariants (partition sum, fesc=1 bit-exactness, the
corrected-subband-sum-equals-corrected-whole-band identity, and the
per-age-sums-to-whole-band contract); R4 tolerances are measured-floor times
a stated factor, never sized to an observed residual; R5 notes the
EXACT PATH's own remaining +/- few-percent whole-band residual is its own
SSP-grid-node quantization of the 912 A edge (not the LUT's -- the LUT's own
algebraic split is exact at the true physical edge), filed as #2447 and not
fixed this round. ``TestRealGridIssueRows`` reproduces the issue's own bands
(GALEX NUV at z=2, SDSS u at z=3) on the real, git-tracked
``fsps_prsc_miles_chabrier.h5`` grid with real Cue weights.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    SpectrumPrecomp,
    Uniform,
    WavePrecomp,
)
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation.photometry import FilterCurve
from tengri.observation.spectroscopy import Spectroscopy
from tests._data_skip import requires_cue_weights

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
    lyc_absorb_all: bool = False,
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
            **({"lyc_absorb_all": True} if lyc_absorb_all else {}),
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
    # R4: measured-floor x stated factor, no residual-sized minimum. Measured
    # (2026-09, post fix-round): z=2 err 0.0051%, z=3 err 0.0040%, floor
    # 0.00057% both z -- 20x the floor comfortably covers both with margin
    # to spare, and is ~250x tighter than the old 0.5% hardcoded floor.
    tol = floor_err * 20.0
    assert err < tol, (
        f"z={z}: worst band '{band}' error {err * 100:.4f}% exceeds "
        f"{tol * 100:.4f}% (20x the measured fesc=1 floor '{floor_name}' "
        f"{floor_err * 100:.5f}%). exact={exact_v}, lut={lut_v}"
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

    # R4: 0.02% -- measured ~0.005% with the fix in place, ~1-2% with the
    # correction disabled (mutation, verified separately): a 4x margin over
    # the measured floor that still separates cleanly from the mutation's
    # two-orders-of-magnitude-larger residual, not the accuracy-floor bound
    # used elsewhere (this check's whole job is to catch exactly that
    # mutation, so it must not be loose enough to survive it).
    for fesc_value in (0.25, 0.75):
        p = dict(fixed)
        p[free_name] = float(fesc_value)
        exact_v = np.asarray(model_exact.predict_photometry(p))
        lut_v = np.asarray(model_lut.predict_photometry(p))
        band, err, _, _ = _worst_band(exact_v, lut_v, ["straddle", "clean"])
        assert err < 0.0002, (
            f"fesc={fesc_value}: worst band '{band}' error {err * 100:.4f}% too large"
        )


# ── Case 4: the dusty two_component precomp instance ───────────────────────


@pytest.mark.parametrize("lyc_absorb_all", [False, True])
def test_case4_dusty_instance(lyc_ssp, lyc_absorb_all):
    """The dusty ``two_component`` precomp instance, both R2 sub-rules.

    ``lyc_absorb_all=False`` (default): birth-cloud-graded ``1-y(a)(1-fesc)``.
    ``lyc_absorb_all=True``: flat, all-stellar-LyC-absorbed rule. Each must
    match the SAME dense-path rule under its own flag (R2, no double-count).

    A real bug lived here during this fix round: the sub-band code first read
    ``params.get("neb_fesc", ...)`` directly, but ``DustSEDComponent.apply``'s
    own ``params`` mapping is scoped to ``dust_*`` keys plus the bare
    ``redshift`` (see its docstring) -- "neb_fesc" is never in it, so that read
    silently always took the 0.0 default. It was caught here: the fesc=1
    no-op floor (below) failed to converge with K, staying pinned at the
    fesc=0 answer, and the flat-rule floor diverged from the graded-rule floor
    at the SAME fesc=1 (mathematically must be identical -- both rules reduce
    to a no-op there) rather than matching bit-for-bit. Fixed by reading the
    already-correctly-resolved ``lyc_transmission`` (state.derived, not
    params-scoped) via ``jnp.interp`` onto the sub-band nodes instead of
    re-deriving fesc.
    """
    z = 2.0
    filters = [_straddle_filter(z, name="straddle"), _clean_filter(5000.0, z, name="clean")]
    names = ["straddle", "clean"]

    model_exact = _build(lyc_ssp, z, None, filters, dust=True, lyc_absorb_all=lyc_absorb_all)
    model_lut = _build(
        lyc_ssp, z, WavePrecomp(), filters, dust=True, lyc_absorb_all=lyc_absorb_all
    )
    params = _params(model_exact)

    exact_phot = np.asarray(model_exact.predict_photometry(params))
    lut_phot = np.asarray(model_lut.predict_photometry(params))

    model_exact_fesc1 = _build(
        lyc_ssp, z, None, filters, dust=True, fesc=1.0, lyc_absorb_all=lyc_absorb_all
    )
    model_lut_fesc1 = _build(
        lyc_ssp, z, WavePrecomp(), filters, dust=True, fesc=1.0, lyc_absorb_all=lyc_absorb_all
    )
    params_fesc1 = _params(model_exact_fesc1)
    exact_fesc1 = np.asarray(model_exact_fesc1.predict_photometry(params_fesc1))
    lut_fesc1 = np.asarray(model_lut_fesc1.predict_photometry(params_fesc1))

    # Ablation: bit-identical LUT between fesc=0/1 in the dusty branch was the
    # diagnosis's own signature that the correction never reached this path.
    assert not np.isclose(lut_phot[0], lut_fesc1[0], rtol=1e-6, atol=0.0), (
        f"lyc_absorb_all={lyc_absorb_all}: dusty LUT straddle-band photometry "
        "is bit-identical between neb_fesc=0 and neb_fesc=1 -- the "
        "two_component precomp branch is not reading the escape fraction"
    )

    # White-box, on the tensor the fix actually publishes
    # (``stellar_subband_lyc_factor_precomp`` -- NOT the raw
    # ``stellar_phot_lnu_per_age_subband_precomp``, which R2 deliberately
    # leaves untouched so each consumer applies its own rule to a shared
    # factor). fesc=0 and fesc=1 builds have DIFFERENT sub-band node counts
    # (R1 forces an extra edge at 912 A only when the mask is live, i.e. only
    # at fesc=0), so the two tensors cannot be compared elementwise -- assert
    # on each one's own departure from 1 instead.
    state0 = model_lut.predict_state(params)
    state1 = model_lut_fesc1.predict_state(params_fesc1)
    factor0 = state0.derived.get("stellar_subband_lyc_factor_precomp")
    factor1 = state1.derived.get("stellar_subband_lyc_factor_precomp")
    assert factor0 is not None, "stellar_subband_lyc_factor_precomp not published at fesc=0"
    assert float(np.min(np.asarray(factor0))) < 0.999, (
        f"lyc_absorb_all={lyc_absorb_all}: stellar_subband_lyc_factor_precomp "
        "never departs from 1.0 at neb_fesc=0 -- the sub-band LyC correction "
        "is not reaching this tensor"
    )
    if factor1 is not None:
        np.testing.assert_allclose(
            np.asarray(factor1),
            1.0,
            rtol=0.0,
            atol=1e-9,
            err_msg=f"lyc_absorb_all={lyc_absorb_all}: stellar_subband_lyc_factor_precomp "
            "is not a no-op at neb_fesc=1",
        )

    # R3(b): at fesc=1 the graded and flat rules both reduce to the identity
    # factor, so the two LUTs (which otherwise differ only in
    # ``lyc_absorb_all``) must be BIT-IDENTICAL to each other, not merely
    # close -- this is the exact invariant the params-scoping bug above broke
    # (the flat rule silently used the fesc=0 answer, diverging from graded's
    # correctly-converging fesc=1 floor by a persistent, K-independent ~3%).
    model_lut_fesc1_other = _build(
        lyc_ssp, z, WavePrecomp(), filters, dust=True, fesc=1.0, lyc_absorb_all=not lyc_absorb_all
    )
    lut_fesc1_other = np.asarray(model_lut_fesc1_other.predict_photometry(params_fesc1))
    assert np.array_equal(lut_fesc1, lut_fesc1_other), (
        f"lyc_absorb_all={lyc_absorb_all}: fesc=1 LUT differs between the flat "
        "and graded rules -- both must reduce to the identical no-op factor"
    )

    # Observable (predict_photometry-level, not just white-box): with the fix
    # in place, measured (2026-09, post fix-round) K=5 worst-band error is
    # 0.16-0.18% against a K=5 fesc=1 floor of ~0.29% (the pre-existing K-node
    # dust-quadrature floor, unrelated to this fix) -- both converge together
    # as K rises (mutation-verified separately: disabling the correction
    # collapses this convergence and reopens a large, K-independent gap).
    dust_free_floor_name, dust_free_floor_err, _, _ = _worst_band(exact_fesc1, lut_fesc1, names)
    band, err, exact_v, lut_v = _worst_band(exact_phot, lut_phot, names)
    tol = max(dust_free_floor_err * 3.0, 1e-6)
    assert err < tol, (
        f"lyc_absorb_all={lyc_absorb_all}: dusty z={z}: worst band '{band}' "
        f"error {err * 100:.4f}% exceeds {tol * 100:.4f}% (3x the dust-free-arm "
        f"floor '{dust_free_floor_name}' {dust_free_floor_err * 100:.4f}%). "
        f"exact={exact_v}, lut={lut_v}"
    )


# ── Case 5: z-axis falsification -- node vs between-node, n_z sweep ─────────


def test_case5_z_axis_falsification(lyc_ssp):
    z_min, z_max = 0.5, 2.5
    n_z_a, n_z_b, n_z_c = 60, 240, 960
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
    model_lut_c = _build(
        lyc_ssp,
        node_z,
        WavePrecomp(n_z=n_z_c, z_min=z_min, z_max=z_max),
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
    err_mid_c = _err_at(model_lut_c, mid_z)

    # R4: measured (2026-09) node_err 0.0020-0.0053%, mid_err 0.0069-0.0109%
    # across n_z in {60, 240, 960} -- floor x10 comfortably covers all three
    # with margin, no hardcoded residual-sized minimum.
    assert err_mid_a < max(err_node_a, 1e-6) * 10.0, (
        f"between-node z={mid_z} error {err_mid_a * 100:.4f}% far exceeds "
        f"node z={node_z} error {err_node_a * 100:.4f}%"
    )

    # Convergence statement (replaces the old "did not move materially"
    # pairwise check): the LyC split is an exact algebraic split at BUILD
    # time, not a redshift-quadrature term, so raising n_z must not make the
    # between-node residual diverge -- it may wobble with the ordinary
    # ztable-interpolation floor, but the finest grid's residual must stay
    # bounded by (not grow past) the coarser grids', not run away as a
    # genuine ztable-resolution bug would. Measured: 0.0069% -> 0.0109% ->
    # 0.0078% (n_z=60,240,960) -- bounded, non-diverging.
    errs = {n_z_a: err_mid_a, n_z_b: err_mid_b, n_z_c: err_mid_c}
    assert max(errs.values()) < 0.05, (
        f"between-node residual failed to stay bounded across n_z={list(errs)}: "
        f"{[f'{e * 100:.4f}%' for e in errs.values()]}"
    )
    assert err_mid_c <= max(err_mid_a, err_mid_b) * 2.0 + 0.001, (
        f"between-node residual diverged at the finest n_z ({n_z_c}): "
        f"{err_mid_c * 100:.4f}% vs {max(err_mid_a, err_mid_b) * 100:.4f}% "
        f"at coarser n_z ({n_z_a}, {n_z_b})"
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
    # R4: measured (2026-09) ctrl_err 0.066-0.58% across z in {0.8,1.5,2,3} --
    # 1% comfortably covers all four with margin, ~2x tighter than the old 2%.
    assert err_ctrl < 0.01, (
        f"z={z} (Inoue IGM, no nebular -- negative control, no neb_fesc to "
        f"mask anything): worst band '{band_ctrl}' error {err_ctrl * 100:.4f}% "
        f"exceeds 1%. exact={exact_ctrl_v}, lut={lut_ctrl_v}"
    )

    # (2) White-box: the corrected FACTOR must actually respond to neb_fesc.
    # ``stellar_phot_lnu_per_age_subband_igm_precomp`` (the raw SSP-weighted
    # subband tensor) is deliberately left untouched by R2 -- every consumer
    # reads it raw and applies its own rule via the shared
    # ``stellar_subband_lyc_factor_precomp`` key instead, so that is the
    # tensor to assert on. fesc=0 and fesc=1 builds also have different
    # sub-band node counts (R1's forced 912 A edge is live only at fesc=0),
    # so the two builds' factors cannot be compared elementwise -- assert on
    # each one's own departure from 1 instead (mirrors case 4).
    model_lut = _build(lyc_ssp, z, WavePrecomp(), filters, igm="inoue")
    model_lut_fesc1 = _build(lyc_ssp, z, WavePrecomp(), filters, igm="inoue", fesc=1.0)
    params = _params(model_lut)
    params_fesc1 = _params(model_lut_fesc1)
    state0 = model_lut.predict_state(params)
    state1 = model_lut_fesc1.predict_state(params_fesc1)
    factor0 = state0.derived.get("stellar_subband_lyc_factor_precomp")
    factor1 = state1.derived.get("stellar_subband_lyc_factor_precomp")
    assert factor0 is not None, (
        f"z={z} (Inoue IGM): stellar_subband_lyc_factor_precomp not published "
        "-- the no-dust mean-IGM sub-band branch did not engage"
    )
    assert float(np.min(np.asarray(factor0))) < 0.999, (
        f"z={z}: stellar_subband_lyc_factor_precomp never departs from 1.0 at "
        "neb_fesc=0 with no dust -- the sub-band LyC correction "
        "(nebular/component.py) is not reaching this tensor"
    )
    if factor1 is not None:
        np.testing.assert_allclose(
            np.asarray(factor1),
            1.0,
            rtol=0.0,
            atol=1e-9,
            err_msg=f"z={z}: stellar_subband_lyc_factor_precomp is not a no-op "
            "at neb_fesc=1 with no dust",
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


# ── R3 conservation invariants, as their own cases ──────────────────────────


def test_r3a_subband_partition_sums_to_whole_band(lyc_ssp):
    """R3(a): the raw (pre-LyC-correction) sub-band partition conserves flux.

    ``subband_quadrature`` (#1122) asserts this internally at build time, but
    the contract deserves its own black-box test: summing
    ``stellar_phot_lnu_per_age_subband_precomp`` over K must reconstruct
    ``stellar_phot_lnu_per_age_precomp`` once the LyC correction is undone
    (the published per-age tensor is POST-correction; add back
    ``(1-fesc)*per_age_lyc`` to recover the raw, pre-correction value this
    partition is over). Unrelated to #2439/#2427 itself -- a pre-existing
    quadrature invariant this fix's own sub-band consumers depend on.
    """
    z = 2.0
    fesc = 0.3
    filters = [_straddle_filter(z, name="straddle"), _clean_filter(5000.0, z, name="clean")]
    model = _build(
        lyc_ssp,
        z,
        WavePrecomp(band_integration="quadrature", n_subbands=5),
        filters,
        fesc=fesc,
    )
    p = _params(model)
    state = model.predict_state(p)

    sub_raw = np.asarray(state.derived.get("stellar_phot_lnu_per_age_subband_precomp"))
    per_age_lyc = np.asarray(state.derived.get("stellar_phot_lnu_per_age_precomp_lyc"))
    per_age_corrected = np.asarray(state.derived.get("stellar_phot_lnu_per_age_precomp"))

    partition_sum = sub_raw.sum(axis=-1)
    raw_per_age = per_age_corrected + (1.0 - fesc) * per_age_lyc
    np.testing.assert_allclose(
        partition_sum,
        raw_per_age,
        rtol=1e-9,
        atol=0.0,
        err_msg="R3(a): sub-band partition does not sum to the raw per-age whole band",
    )


def test_r3b_fesc1_bit_exact_no_op(lyc_ssp):
    """R3(b): fesc=1 is bit-for-bit identical to a model with no nebular at all.

    Not merely close: ``neb_fesc=1.0`` means every stellar LyC photon
    escapes, so the mask factor is the identity everywhere and the corrected
    stellar bucket must be EXACTLY what it would be with no nebular component
    (and so no mask) in the chain at all -- ``np.array_equal``, not
    ``assert_allclose``.
    """
    z = 2.0
    filters = [_straddle_filter(z, name="straddle"), _clean_filter(5000.0, z, name="clean")]
    model_fesc1 = _build(lyc_ssp, z, WavePrecomp(), filters, fesc=1.0)
    model_noneb = _build(lyc_ssp, z, WavePrecomp(), filters, nebular=False)
    state1 = model_fesc1.predict_state(_params(model_fesc1))
    state0 = model_noneb.predict_state(_params(model_noneb))
    w1 = np.asarray(state1.derived.get("stellar_phot_lnu_precomp"))
    w0 = np.asarray(state0.derived.get("stellar_phot_lnu_precomp"))
    assert np.array_equal(w1, w0), (
        f"R3(b): fesc=1.0 stellar_phot_lnu_precomp {w1} is not bit-identical "
        f"to the no-nebular control {w0}"
    )


def test_r3c_corrected_subband_sum_matches_corrected_whole_band(lyc_ssp):
    """R3(c): the corrected sub-band sum equals the corrected whole band.

    Applying ``stellar_subband_lyc_factor_precomp`` to each raw chunk and
    summing over (age, K) must reproduce ``stellar_phot_lnu_precomp`` (the
    whole-band bucket, corrected via the exact algebraic split in
    ``nebular/component.py``) to numerical precision: R1's forced edge at the
    physical Lyman limit makes the two mechanisms partition the SAME boundary
    exactly, so a chunk-wise mask and the whole-band's own subtraction must
    agree (this is the invariant #2447's exact-path residual has nothing to
    do with -- that residual lives in the EXACT path's own grid-node
    quantization, not in this LUT-internal identity).
    """
    z = 2.0
    fesc = 0.3
    filters = [_straddle_filter(z, name="straddle"), _clean_filter(5000.0, z, name="clean")]
    model = _build(
        lyc_ssp,
        z,
        WavePrecomp(band_integration="quadrature", n_subbands=5),
        filters,
        fesc=fesc,
    )
    p = _params(model)
    state = model.predict_state(p)

    sub_raw = np.asarray(state.derived.get("stellar_phot_lnu_per_age_subband_precomp"))
    factor = np.asarray(state.derived.get("stellar_subband_lyc_factor_precomp"))
    whole = np.asarray(state.derived.get("stellar_phot_lnu_precomp"))

    corrected_sub_sum = (sub_raw * factor).sum(axis=(0, 2))
    np.testing.assert_allclose(
        corrected_sub_sum,
        whole,
        rtol=1e-9,
        atol=0.0,
        err_msg="R3(c): corrected sub-band sum does not match the corrected whole band",
    )


def test_r3d_per_age_sums_to_whole_band_lyc_twin(lyc_ssp):
    """R3(d): the per-age LyC split sums (over age) to its whole-band twin.

    ``protocols/derived_state.py``'s documented contract: summing
    ``stellar_phot_lnu_per_age_precomp_lyc`` over age must equal
    ``stellar_phot_lnu_precomp_lyc``. Both are built independently inside
    ``preintegrate_grid``/``_compute_photometry_ztable`` (the whole-band
    tensor is not merely the per-age one summed after the fact at this call
    site), so this is a genuine cross-check, not a tautology.
    """
    z = 2.0
    filters = [_straddle_filter(z, name="straddle"), _clean_filter(5000.0, z, name="clean")]
    model = _build(lyc_ssp, z, WavePrecomp(), filters, fesc=0.3)
    p = _params(model)
    state = model.predict_state(p)

    per_age_lyc = np.asarray(state.derived.get("stellar_phot_lnu_per_age_precomp_lyc"))
    whole_lyc = np.asarray(state.derived.get("stellar_phot_lnu_precomp_lyc"))
    np.testing.assert_allclose(
        per_age_lyc.sum(axis=0),
        whole_lyc,
        rtol=1e-9,
        atol=0.0,
        err_msg="R3(d): sum-over-age of the per-age LyC split does not match the whole-band twin",
    )


# ── SpectrumPrecomp: the identical defect, exact per-pixel fix (item 5) ─────


def _spec_build(ssp, z, approx, fesc=0.0, n_pix=40):
    wave_obs = jnp.linspace(LYC_LIMIT * (1.0 + z) * 0.5, LYC_LIMIT * (1.0 + z) * 1.5, n_pix)
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs)),
        sfh={
            "type": "delayed",
            "log_total_mass": 10.0,
            "tau_gyr": 1.0,
            "age_gyr": 3.0,
            "all_params": Fixed(DEFAULT),
        },
        met={"logzsol": 0.0, "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT), "neb_fesc": Fixed(float(fesc))},
        dust_attenuation={"type": "none"},
        igm={"type": "none"},
        redshift=Fixed(float(z)),
        approx=approx,
    )


def test_spectrumprecomp_exact_per_pixel_lyc_mask(lyc_ssp):
    """SpectrumPrecomp gets the identical fix: an exact per-pixel mask.

    A spectrum pixel IS a single rest-frame wavelength (unlike a photometric
    band, which integrates over many), so there is no partition to make
    exact -- the elementwise mask in ``nebular/component.py`` already matches
    the dense path's ``sed_intrinsic`` mask pixel for pixel. Measured
    (2026-09): fesc=0 max relative error ~5e-14%, fesc=1 floor ~5e-14% --
    both at the pre-existing exact-vs-LUT numerical floor, not a residual
    approximation (contrast with the photometric K-node sub-band case, whose
    floor is orders of magnitude looser because it IS a genuine quadrature).
    """
    z = 2.0
    model_exact = _spec_build(lyc_ssp, z, None, fesc=0.0)
    model_lut = _spec_build(lyc_ssp, z, SpectrumPrecomp(), fesc=0.0)
    p = dict(model_exact.spec.get_fixed_values())
    exact_spec = np.asarray(model_exact.predict_spectrum(p))
    lut_spec = np.asarray(model_lut.predict_spectrum(p))

    wave_obs = jnp.linspace(LYC_LIMIT * (1.0 + z) * 0.5, LYC_LIMIT * (1.0 + z) * 1.5, 40)
    below = np.asarray(wave_obs) / (1.0 + z) < LYC_LIMIT
    assert np.any(below) and np.any(~below), "fixture bug: pixel grid no longer straddles 912 A"

    # At fesc=0 every below-912-A pixel is fully absorbed in the exact path;
    # the LUT must match it exactly, not merely closely.
    assert np.array_equal(exact_spec[below], np.zeros_like(exact_spec[below])), (
        "fixture/exact-path bug: fesc=0 below-912 A pixels are not all zero"
    )
    assert np.array_equal(lut_spec[below], exact_spec[below]), (
        "SpectrumPrecomp below-912 A pixels are not bit-identical to the exact "
        "path's fully-absorbed zero at fesc=0"
    )
    nonzero = exact_spec != 0.0
    rel = np.abs((lut_spec[nonzero] - exact_spec[nonzero]) / exact_spec[nonzero])
    assert np.max(rel) < 1e-6, f"SpectrumPrecomp worst relative error {np.max(rel):.3e} too large"

    # fesc=1 floor: same exact-per-pixel mechanism, mask is a no-op.
    model_exact1 = _spec_build(lyc_ssp, z, None, fesc=1.0)
    model_lut1 = _spec_build(lyc_ssp, z, SpectrumPrecomp(), fesc=1.0)
    p1 = dict(model_exact1.spec.get_fixed_values())
    exact1 = np.asarray(model_exact1.predict_spectrum(p1))
    lut1 = np.asarray(model_lut1.predict_spectrum(p1))
    nonzero1 = exact1 != 0.0
    rel1 = np.abs((lut1[nonzero1] - exact1[nonzero1]) / exact1[nonzero1])
    assert np.max(rel1) < 1e-6, f"SpectrumPrecomp fesc=1 floor {np.max(rel1):.3e} too large"


# ── Cache-version defense: a stale entry missing the LyC table is rebuilt ───


def test_ztable_cache_rejects_entry_missing_lyc_table(lyc_ssp, tmp_path, monkeypatch):
    """Item 1 (BLOCKER): a cache entry with no ``ssp_phot_lyc_table`` key is
    never silently served to a ``lyc_gate=True`` request.

    Defense in depth beside the ``_ZTABLE_CACHE_VERSION`` 3->4 bump and
    ``lyc_gate`` joining :class:`ZTableRequest` (both make a stale v3-shaped
    entry hash to a DIFFERENT key, so it could never collide with a
    ``lyc_gate=True`` request in practice): this test manufactures the entry
    the version bump makes unreachable by construction -- a file at the
    EXACT key a ``lyc_gate=True`` request would compute, but missing the key
    -- to verify the reader-side check
    (``has_lyc = "ssp_phot_lyc_table" in d.files``) independently catches it,
    should a future key-computation change ever let one slip through again.
    """
    from tengri.components.stellar.sps.precompute import (
        _ztable_cache_dir,
        _ztable_cache_key,
        precompute_photometry_ztable,
    )
    from tengri.utils.filter_convention import FilterConvention

    monkeypatch.setenv("TENGRI_PRECOMP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TENGRI_DISABLE_PRECOMP_CACHE", raising=False)

    z = 2.0
    straddle = _straddle_filter(z, name="straddle")
    filter_waves = [jnp.asarray(straddle.wave)]
    filter_trans = [jnp.asarray(straddle.trans)]
    z_grid = jnp.linspace(0.5, 2.5, 10)

    table = precompute_photometry_ztable(
        lyc_ssp,
        filter_waves,
        filter_trans,
        z_grid=z_grid,
        n_subbands=5,
        convention=FilterConvention.BESSELL,
        lyc_gate=True,
    )
    assert table.ssp_phot_lyc_table is not None, (
        "a fresh lyc_gate=True build did not publish ssp_phot_lyc_table"
    )

    cache_dir = _ztable_cache_dir()
    key = _ztable_cache_key(
        lyc_ssp,
        filter_waves,
        filter_trans,
        z_grid,
        False,
        False,
        FilterConvention.BESSELL,
        5,
        True,
    )
    cache_path = cache_dir / f"ztable_{key}.npz"
    assert cache_path.is_file(), f"expected cache file not found at {cache_path}"

    # Manufacture the corrupted entry: same file, ``ssp_phot_lyc_table``
    # stripped -- the exact shape the reader-side defense must catch.
    with np.load(cache_path, allow_pickle=False) as d:
        stripped = {k: d[k] for k in d.files if k != "ssp_phot_lyc_table"}
    assert "ssp_phot_lyc_table" not in stripped
    np.savez(cache_path, **stripped)

    table2 = precompute_photometry_ztable(
        lyc_ssp,
        filter_waves,
        filter_trans,
        z_grid=z_grid,
        n_subbands=5,
        convention=FilterConvention.BESSELL,
        lyc_gate=True,
    )
    assert table2.ssp_phot_lyc_table is not None, (
        "precompute_photometry_ztable silently served a cache entry with no "
        "ssp_phot_lyc_table to a lyc_gate=True request instead of rebuilding"
    )
    np.testing.assert_allclose(
        np.asarray(table2.ssp_phot_lyc_table),
        np.asarray(table.ssp_phot_lyc_table),
        rtol=1e-9,
        atol=0.0,
        err_msg="rebuilt ssp_phot_lyc_table does not match the original fresh build",
    )


# ── Real-grid class: the issue's own bands, on real data ────────────────────


@requires_cue_weights
class TestRealGridIssueRows:
    """The issue's own bands/redshifts on the real, git-tracked SSP + Cue.

    ``fsps_prsc_miles_chabrier.h5`` (bare-stellar, suitable for Cue) and
    ``cue_weights.npz`` are both tracked in git, so this class needs no
    optional download -- ``requires_cue_weights`` is defensive (an
    incomplete checkout), matching its own docstring.

    R5: the remaining LUT-vs-exact residual on these real rows (measured
    2026-09, post fix-round: z=2 GALEX NUV ~10.2%, z=3 SDSS u ~4.7%,
    K-invariant -- confirmed by an explicit K=5 vs K=32 sweep below) is the
    EXACT PATH's own SSP-grid-node quantization of the 912 A edge: the dense
    path masks whichever SSP wavelength node sits just below 912 A, not 912 A
    itself, while the LUT's ``preintegrate_grid`` split is exact at the TRUE
    physical edge. Filed as #2447; NOT fixed this round (do not change the
    exact path here). The magnitude is SFH- and filter-dependent (this class
    measures its own SFH's numbers, not a universal constant) -- these
    ratchets bound the residual from getting WORSE (a regression) or
    disappearing without explanation (the correction being silently
    disabled again), not from shrinking further.
    """

    SSP_PATH = "data/fsps_prsc_miles_chabrier.h5"

    @pytest.fixture(scope="class")
    def real_ssp(self):
        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

        return load_ssp_data(self.SSP_PATH)

    @staticmethod
    def _build(real_ssp, z, approx, *, dust=False, igm=None, fesc=0.0):
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
        igm_group = (
            {"type": "none"} if igm is None else {"type": igm, "all_params": Fixed(DEFAULT)}
        )
        return SEDModel.build(
            ssp_data=real_ssp,
            observation=Observation(photometry=Photometry.from_names(["galex_nuv", "sdss_u"])),
            sfh={
                "type": "delayed",
                "log_total_mass": 9.0,
                "tau_gyr": 0.3,
                "age_gyr": 0.5,
                "all_params": Fixed(DEFAULT),
            },
            met={"logzsol": 0.0, "all_params": Fixed(DEFAULT)},
            neb={"type": "cue", "all_params": Fixed(DEFAULT), "neb_fesc": Fixed(float(fesc))},
            dust_attenuation=dust_group,
            igm=igm_group,
            redshift=Fixed(float(z)),
            approx=approx,
        )

    @staticmethod
    def _params(model):
        p = dict(model.spec.get_fixed_values())
        if model.spec.free_params:
            p.update(model.spec.sample(jax.random.PRNGKey(0)))
        return p

    def _rel(self, exact, lut, idx):
        e = float(np.asarray(exact)[idx])
        v = float(np.asarray(lut)[idx])
        if e == 0.0:
            return None
        return abs((v - e) / e)

    @pytest.mark.parametrize("z,band_idx,band_name,ceiling", [(2.0, 0, "galex_nuv", 0.20)])
    def test_dust_free_ratchet(self, real_ssp, z, band_idx, band_name, ceiling):
        """z=2 GALEX NUV, dust-free: #2439's own reported row."""
        m_exact = self._build(real_ssp, z, None, fesc=0.0)
        m_lut = self._build(real_ssp, z, WavePrecomp(), fesc=0.0)
        p = self._params(m_exact)
        e = np.asarray(m_exact.predict_photometry(p))
        lut = np.asarray(m_lut.predict_photometry(p))
        rel = self._rel(e, lut, band_idx)
        assert rel is not None, f"z={z} {band_name}: exact flux is exactly zero"
        assert rel < ceiling, (
            f"z={z} {band_name} dust-free: LUT-vs-exact residual {rel * 100:.2f}% "
            f"exceeds the {ceiling * 100:.0f}% ratchet. A residual around this "
            f"size is expected and accepted this round -- it is the EXACT "
            f"path's own SSP-grid-node quantization of the 912 A edge (#2447), "
            f"not this LUT fix's error; a residual near zero would also be "
            f"suspicious (it would mean the SSP grid happens to have a node "
            f"exactly at 912 A for this z). This ratchet exists to catch a "
            f"REGRESSION (residual growing much worse) or the correction being "
            f"silently disabled again (which reopens the pre-fix +915%/+69% "
            f"defect), not to demand the residual shrink further."
        )

        # fesc=1 floor: must collapse to near-zero regardless of #2447.
        m_exact1 = self._build(real_ssp, z, None, fesc=1.0)
        m_lut1 = self._build(real_ssp, z, WavePrecomp(), fesc=1.0)
        p1 = self._params(m_exact1)
        e1 = np.asarray(m_exact1.predict_photometry(p1))
        lut1 = np.asarray(m_lut1.predict_photometry(p1))
        floor = self._rel(e1, lut1, band_idx)
        if floor is not None:
            assert floor < 0.01, (
                f"z={z} {band_name} fesc=1 floor {floor * 100:.3f}% is not "
                "near-zero -- the mask should be a no-op end to end"
            )

    def test_dust_free_ratchet_z3_u(self, real_ssp):
        """z=3 SDSS u, dust-free: #2439's other reported row."""
        z, band_idx, band_name, ceiling = 3.0, 1, "sdss_u", 0.10
        m_exact = self._build(real_ssp, z, None, fesc=0.0)
        m_lut = self._build(real_ssp, z, WavePrecomp(), fesc=0.0)
        p = self._params(m_exact)
        e = np.asarray(m_exact.predict_photometry(p))
        lut = np.asarray(m_lut.predict_photometry(p))
        rel = self._rel(e, lut, band_idx)
        assert rel is not None, f"z={z} {band_name}: exact flux is exactly zero"
        assert rel < ceiling, (
            f"z={z} {band_name} dust-free: LUT-vs-exact residual {rel * 100:.2f}% "
            f"exceeds the {ceiling * 100:.0f}% ratchet. Expected residual size: "
            f"the EXACT path's own SSP-grid-node quantization of 912 A (#2447), "
            f"not this fix's error -- see test_dust_free_ratchet's docstring."
        )

    @pytest.mark.parametrize("z,band_idx,band_name,ceiling", [(2.0, 0, "galex_nuv", 0.20)])
    def test_2427_mean_igm_ratchet(self, real_ssp, z, band_idx, band_name, ceiling):
        """#2427's own rows: mean-IGM (Inoue) active, no dust."""
        m_exact = self._build(real_ssp, z, None, igm="inoue", fesc=0.0)
        m_lut = self._build(real_ssp, z, WavePrecomp(), igm="inoue", fesc=0.0)
        p = self._params(m_exact)
        e = np.asarray(m_exact.predict_photometry(p))
        lut = np.asarray(m_lut.predict_photometry(p))
        rel = self._rel(e, lut, band_idx)
        assert rel is not None, f"z={z} {band_name} (Inoue IGM): exact flux is exactly zero"
        assert rel < ceiling, (
            f"#2427 z={z} {band_name} (Inoue IGM): LUT-vs-exact residual "
            f"{rel * 100:.2f}% exceeds the {ceiling * 100:.0f}% ratchet -- same "
            f"expected-residual reasoning as the dust-free rows (#2447)."
        )

    def test_2427_mean_igm_ratchet_z3_u(self, real_ssp):
        z, band_idx, band_name, ceiling = 3.0, 1, "sdss_u", 0.10
        m_exact = self._build(real_ssp, z, None, igm="inoue", fesc=0.0)
        m_lut = self._build(real_ssp, z, WavePrecomp(), igm="inoue", fesc=0.0)
        p = self._params(m_exact)
        e = np.asarray(m_exact.predict_photometry(p))
        lut = np.asarray(m_lut.predict_photometry(p))
        rel = self._rel(e, lut, band_idx)
        assert rel is not None, f"z={z} {band_name} (Inoue IGM): exact flux is exactly zero"
        assert rel < ceiling, (
            f"#2427 z={z} {band_name} (Inoue IGM): LUT-vs-exact residual "
            f"{rel * 100:.2f}% exceeds the {ceiling * 100:.0f}% ratchet (#2447)."
        )

    def test_dusty_rows_k5_and_k32(self, real_ssp):
        """Dusty (two_component) rows at K=5 and K=32: convergence, not a cliff.

        z=2 GALEX NUV. K=32 must not be WORSE than K=5 by more than a small
        margin -- the sub-band quadrature must be converging (or at least
        holding steady), never regressing as K rises.
        """
        z, band_idx, band_name = 2.0, 0, "galex_nuv"
        m_exact = self._build(real_ssp, z, None, dust=True, fesc=0.0)
        p = self._params(m_exact)
        e = np.asarray(m_exact.predict_photometry(p))

        errs = {}
        for K in (5, 32):
            m_lut = self._build(
                real_ssp,
                z,
                WavePrecomp(band_integration="quadrature", n_subbands=K),
                dust=True,
                fesc=0.0,
            )
            lut = np.asarray(m_lut.predict_photometry(p))
            rel = self._rel(e, lut, band_idx)
            assert rel is not None, f"K={K} {band_name}: exact flux is exactly zero"
            errs[K] = rel

        assert errs[32] < errs[5] * 2.0 + 0.01, (
            f"dusty z={z} {band_name}: K=32 error {errs[32] * 100:.2f}% is "
            f"much worse than K=5's {errs[5] * 100:.2f}% -- sub-band "
            f"quadrature should converge (or hold steady) as K rises, not "
            f"regress"
        )
        assert errs[5] < 0.20 and errs[32] < 0.20, (
            f"dusty z={z} {band_name}: K=5 {errs[5] * 100:.2f}%, K=32 "
            f"{errs[32] * 100:.2f}% -- both exceed the 20% ratchet"
        )


def test_r2_graded_rule_spares_old_stars_flat_rule_does_not(lyc_ssp):
    """R2: the graded and flat rules are structurally different, not just
    numerically close.

    Total ``predict_photometry`` for this fixture happens to put the two
    rules within a factor of ~1.15 of each other (0.159 % vs 0.182 % at
    K=5), too close for a total-flux tolerance alone to discriminate a
    "graded silently replaced by flat" mutation. The PER-AGE structure does
    discriminate it sharply: at the OLDEST age bin (old/diffuse stars,
    ``y(age)~0``), the default graded rule leaves the factor at ~1
    (unmasked -- old stars are not birth-cloud-embedded), while the flat
    rule (``lyc_absorb_all=True``) masks it down to ``fesc`` regardless of
    age. Measured: graded oldest-age factor 0.99997, flat oldest-age factor
    0.0 at fesc=0 -- not a subtle difference.
    """
    z = 2.0
    fesc = 0.0
    filters = [_straddle_filter(z, name="straddle"), _clean_filter(5000.0, z, name="clean")]
    model_graded = _build(lyc_ssp, z, WavePrecomp(), filters, dust=True, fesc=fesc)
    model_flat = _build(
        lyc_ssp, z, WavePrecomp(), filters, dust=True, fesc=fesc, lyc_absorb_all=True
    )
    state_graded = model_graded.predict_state(_params(model_graded))
    state_flat = model_flat.predict_state(_params(model_flat))
    factor_graded = np.asarray(state_graded.derived.get("stellar_subband_lyc_factor_precomp"))
    factor_flat = np.asarray(state_flat.derived.get("stellar_subband_lyc_factor_precomp"))

    oldest_graded = factor_graded[-1, 0, :]
    oldest_flat = factor_flat[-1, 0, :]
    assert np.all(oldest_graded > 0.99), (
        f"R2: the default graded rule masks the oldest age bin ({oldest_graded}) "
        "-- old/diffuse stars should be ~unmasked (y(age)~0)"
    )
    assert np.all(oldest_flat < 0.5), (
        f"R2: lyc_absorb_all=True does not mask the oldest age bin "
        f"({oldest_flat}) -- the flat rule should mask every age uniformly"
    )


def test_lyc_gate_excludes_baked_in_and_shock_nebular(lyc_ssp):
    """``lyc_gate`` must not fire for a ``NebularSEDComponent`` that never
    reaches the ``neb_fesc`` masking block at all.

    ``neb={'type': 'none'}`` still puts a ``NebularSEDComponent`` in the
    chain (``backend="baked_in"`` -- nebular emission already baked into the
    SSP grid, nothing more to add), and it returns from ``apply`` before
    ever publishing ``lyc_transmission``. A first cut of the
    ``lyc_mask_live`` build-time gate (``forward/sed_model.py``) tested only
    ``isinstance(c, NebularSEDComponent)``, so a baked-in (or ``shock``,
    same early-return shape) nebular model was treated as "live" anyway:
    pure wasted compute for most filters, but for a very wide/red far-IR
    filter whose observed-frame footprint sits nowhere near the forced
    912(1+z) sub-band edge, ``subband_quadrature``'s own partition-
    conservation assertion could raise outright -- caught by a BASE-vs-HEAD
    zero-diff probe across ten real filters including 12um (WISE W3) and
    100um (Herschel PACS green) on the real SSP grid (those two only
    crashed; RED against a mutated ``lyc_mask_live`` that goes back to
    plain ``isinstance``, confirmed below). The direct, always-discriminating
    signal (not dependent on a specific filter/grid numerically tripping the
    assertion) is the sub-band tensor's own last-axis width: K when the gate
    is correctly off, K+1 when R1's forced edge was wrongly inserted.
    """
    z = 1.0
    filters = [_straddle_filter(z, name="straddle"), _clean_filter(5000.0, z, name="clean")]
    K = 5
    model = _build(
        lyc_ssp,
        z,
        WavePrecomp(band_integration="quadrature", n_subbands=K),
        filters,
        nebular=False,
    )
    p = _params(model)
    phot = np.asarray(model.predict_photometry(p))
    assert np.all(np.isfinite(phot)), f"baked-in nebular photometry not finite: {phot}"

    state = model.predict_state(p)
    assert state.derived.get("lyc_transmission") is None, (
        "a baked-in ('none') nebular model published lyc_transmission -- "
        "it should never reach that code path"
    )
    sub_waves = state.derived.get("stellar_subband_waves_rest_precomp")
    assert sub_waves is not None
    assert np.asarray(sub_waves).shape[-1] == K, (
        f"a baked-in ('none') nebular model has a K+1-wide sub-band partition "
        f"(shape {np.asarray(sub_waves).shape}, expected last axis {K}) -- "
        "lyc_mask_live is wrongly treating a non-photoionized backend as a "
        "live Lyman-continuum mask"
    )
