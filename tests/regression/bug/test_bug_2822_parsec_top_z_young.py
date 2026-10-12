# SPDX-License-Identifier: BSD-3-Clause
"""The young Z = 0.06 PARSEC node lacks massive stars; the model refuses it (#2822).

FSPS's ``ISOCHRONES/PARSEC/isoc_z0.0600.dat`` caps the initial mass at 12.01 Msun for
every log age 5.50-7.15, because the PARSEC massive-star tracks reach only Z = 0.04
(Chen et al. 2015, MNRAS 452, 1068, arXiv:1506.01681). Measured on the shipped grid
``fsps_prsc_miles_chabrier``, relative to the Z = 0.04 node at log age 5.5-6.5:

    Q_H (< 912 A)   3.7e-4 / 4.1e-4 / 1.3e-3 at log age 5.5 / 6.0 / 6.5  (~800-2700x low)
    L_nu(1500 A)    ~ 0.06  (about 16x low)

The loader now holds the metallicity at the highest complete node (Z = 0.04) for
log age <= 7.15 and tapers the top node back in linearly in log age over the next
0.05 dex, one grid step (``top_z='clip'``, default); ``top_z='mist'`` makes the clip
target a MIST [Z/H] = +0.5 node (FSPS zlegend Z = 0.0142 x 10^0.5 = 0.0449,
Choi et al. 2016, ApJ 823, 102);
``top_z='raw'`` keeps the file as shipped. The expected values below are read from the
HDF5 files and the published constants, not from the code under test.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.components.stellar.sps.top_z_validity import (
    YoungTopMetallicityWarning,
    top_node_weight,
)
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.regression_bug

REPO = Path(__file__).resolve().parents[3]
BARE = REPO / "data" / "fsps_prsc_miles_chabrier.h5"
WNE = REPO / "data" / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

pytest.importorskip("h5py")
if not BARE.is_file():  # pragma: no cover - data-less checkout
    pytest.skip(f"SSP data not found: {BARE}", allow_module_level=True)

H_PLANCK = 6.62607015e-27  # erg s
LAST_INCOMPLETE = 7.15  # FSPS isoc_z0.0600.dat: Mini = 12.01 Msun up to this log age
TAPER = 0.05  # one grid step
# FSPS isoc_z0.0600.dat, read off the raw isochrones: is the Z = 0.06 node complete (not
# truncated by the 12.01 Msun cap) at these grid ages? 7.15 is capped (main-sequence tip,
# K-band ratio to Z = 0.04 of 0.833); 7.20 is lifetime-limited (core He-burning tip 11.33 Msun)
# with ratios to Z = 0.04 matching 7.25 and 7.30.
ISOCHRONE_COMPLETE = {7.10: False, 7.15: False, 7.20: True, 7.25: True, 7.30: True}
LOGZSOL_Z04 = np.log10(0.04) - LOG10_ZSUN  # the highest complete PARSEC node
LOGZSOL_Z06 = np.log10(0.06) - LOG10_ZSUN  # the truncated node
MIST_Z = 0.0142 * 10**0.5  # Choi et al. 2016: Zsun = 0.0142, [Z/H] = +0.5
LOGZSOL_MIST = np.log10(MIST_Z) - LOG10_ZSUN


def _load(path=BARE, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_ssp_data(str(path), **kw)


@pytest.fixture(scope="module")
def raw_file():
    with h5py.File(BARE, "r") as f:
        return {k: f[k][:] for k in ("ssp_flux", "ssp_wave", "ssp_lg_age_gyr", "ssp_lgmet")}


@pytest.fixture(scope="module")
def ssp_raw():
    return _load(top_z="raw")


@pytest.fixture(scope="module")
def ssp_clip():
    return _load()


@pytest.fixture(scope="module")
def ssp_mist():
    return _load(top_z="mist")


def _q_h(wave, lnu):
    """Photon rate shortward of the Lyman limit [1/s] from L_nu(wave): int L_nu/(h nu) dnu."""
    m = wave < 911.76
    return np.trapezoid(lnu[..., m] / wave[m] / H_PLANCK, np.log(wave[m]), axis=-1)


def _l1500(wave, lnu):
    return lnu[..., int(np.argmin(np.abs(wave - 1500.0)))]


def _model(ssp, *, logzsol_hi=0.7, scatter=0.0, age_gyr=0.002):
    """A young constant-ish burst: stars younger than ~log age 6.3, one metallicity.

    ``age_gyr`` may be a prior, to sweep the population's age with one build.
    """
    return SEDModel.build(
        ssp_data=ssp,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(100.0),
            "age_gyr": age_gyr if not isinstance(age_gyr, float) else Fixed(age_gyr),
            "log_total_mass": Fixed(6.0),
            "all_params": Fixed(DEFAULT),
        },
        met={
            "logzsol": Uniform(-2.0, logzsol_hi),
            "logzsol_scatter": scatter if not isinstance(scatter, float) else Fixed(scatter),
        },
        neb={"type": "none"},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.05),
    )


def _csp(model, logzsol, **extra):
    pred = model.predict({"met_logzsol": logzsol, **extra})
    wave, lnu = np.asarray(pred.wave_rest), np.asarray(pred.rest_sed())
    return _q_h(wave, lnu), _l1500(wave, lnu)


# ---------------------------------------------------------------------------
# (a) the default refuses the truncated node
# ---------------------------------------------------------------------------
def test_raw_node_reproduces_the_bug(ssp_raw):
    """With top_z='raw' the CSP at the top of the prior is >100x low in Q_H (the defect)."""
    model = _model(ssp_raw)
    q_top, l_top = _csp(model, LOGZSOL_Z06)
    q_04, l_04 = _csp(model, LOGZSOL_Z04)
    assert q_top / q_04 < 1e-2
    assert l_top / l_04 < 0.3


def test_clip_makes_top_of_prior_equal_the_z004_prediction(ssp_clip, ssp_raw):
    """Z=0.06 prediction == Z=0.04 prediction to rounding, for a delta metallicity."""
    model = _model(ssp_clip)
    q_top, l_top = _csp(model, LOGZSOL_Z06)
    q_04, l_04 = _csp(model, LOGZSOL_Z04)
    np.testing.assert_allclose(q_top, q_04, rtol=1e-10)
    np.testing.assert_allclose(l_top, l_04, rtol=1e-10)
    # ... and no longer the ~1000x deficit of the raw node.
    q_raw, _ = _csp(_model(ssp_raw), LOGZSOL_Z06)
    assert q_top / q_raw > 100.0


def test_clip_holds_for_any_metallicity_between_the_nodes(ssp_clip):
    """Z between 0.04 and 0.06 sees the same young population as Z = 0.04."""
    model = _model(ssp_clip)
    q_04, _ = _csp(model, LOGZSOL_Z04)
    for lz in np.linspace(LOGZSOL_Z04, LOGZSOL_Z06, 6):
        np.testing.assert_allclose(_csp(model, lz)[0], q_04, rtol=1e-10)


def test_clip_with_scatter_stays_on_the_z_trend(ssp_clip):
    """With the default 0.1 dex MDF the top-of-prior Q_H stays within 2% of the Z=0.04 value."""
    model = _model(ssp_clip, scatter=0.1)
    q_top, _ = _csp(model, LOGZSOL_Z06)
    q_04, _ = _csp(model, LOGZSOL_Z04)
    assert abs(q_top / q_04 - 1.0) < 0.02


def test_ssp_level_deficit_removed_at_log_age_6(raw_file, ssp_clip):
    """The SSP row at log age 6.0: raw Z=0.06 is ~2500x low; the loaded row is the Z=0.04 row."""
    wave = raw_file["ssp_wave"]
    i = int(np.argmin(np.abs(raw_file["ssp_lg_age_gyr"] + 9.0 - 6.0)))
    flux = raw_file["ssp_flux"]
    ratio_file = _q_h(wave, flux[14, i]) / _q_h(wave, flux[13, i])
    assert ratio_file < 1e-3
    scale = 3.839e33 / 3.828e33  # the loader's FSPS -> IAU Lsun factor
    got = np.asarray(ssp_clip.ssp_flux)[14, i]
    np.testing.assert_allclose(got, flux[13, i] * scale, rtol=1e-12)


# ---------------------------------------------------------------------------
# (b) continuity across the age limit
# ---------------------------------------------------------------------------
def test_weight_function_is_zero_at_limit_one_after_taper_and_continuous():
    s = lambda t: float(top_node_weight(t, LAST_INCOMPLETE, TAPER))  # noqa: E731
    assert s(LAST_INCOMPLETE) == 0.0
    assert s(5.5) == 0.0 and s(-np.inf) == 0.0
    assert s(LAST_INCOMPLETE + TAPER) == pytest.approx(1.0, abs=1e-12) and s(10.1) == 1.0
    for eps in (1e-2, 1e-4, 1e-8, 1e-12):
        jump = abs(s(LAST_INCOMPLETE + eps) - s(LAST_INCOMPLETE - eps))
        assert (
            jump <= eps / TAPER + 1e-13
        )  # Lipschitz 1/W (+ float rounding): no step at the limit
    ts = np.linspace(LAST_INCOMPLETE - 0.2, LAST_INCOMPLETE + 0.3, 2001)
    assert np.all(np.diff(top_node_weight(ts, LAST_INCOMPLETE, TAPER)) >= 0.0)


def test_loaded_top_node_follows_the_weights_node_by_node(raw_file, ssp_clip):
    """flux_top'(t) = (1-s) flux_Z0.04(t) + s flux_Z0.06(t), s from the weight function."""
    lg_age = raw_file["ssp_lg_age_gyr"] + 9.0
    s = top_node_weight(lg_age, LAST_INCOMPLETE, TAPER)
    scale = 3.839e33 / 3.828e33
    expect = (
        (1 - s)[:, None] * raw_file["ssp_flux"][13] + s[:, None] * raw_file["ssp_flux"][14]
    ) * scale
    np.testing.assert_allclose(np.asarray(ssp_clip.ssp_flux)[14], expect, rtol=1e-12)
    # the node's weight at each grid age is the isochrone completeness there (0 or 1),
    # so no age is blended: s = [0, 1] at 7.15 and 7.20
    ages = sorted(ISOCHRONE_COMPLETE)
    j = [int(np.argmin(np.abs(lg_age - a))) for a in ages]
    np.testing.assert_allclose(lg_age[j], ages, atol=1e-9)
    np.testing.assert_allclose(s[j], [float(ISOCHRONE_COMPLETE[a]) for a in ages], atol=1e-9)


def test_csp_has_no_jump_where_a_population_crosses_the_limit(ssp_clip):
    """Sweep the age of a young burst through the limit: L1500 changes by small steps."""
    model = _model(ssp_clip, age_gyr=Uniform(1e-4, 0.1))
    ages_gyr = 10.0 ** (np.arange(6.95, 7.45, 0.01) - 9.0)
    out = [_csp(model, LOGZSOL_Z06, sfh_delayed_age_gyr=float(a))[1] for a in ages_gyr]
    steps = np.abs(np.diff(np.log(np.array(out))))
    assert np.all(np.isfinite(steps))
    assert steps.max() < 0.08  # a hard switch would step by ln(1.7) ~ 0.5 at this age


# ---------------------------------------------------------------------------
# (c) the build warns exactly when the prior reaches the region
# ---------------------------------------------------------------------------
def _warnings_for(ssp, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _model(ssp, **kw)
    return [w for w in caught if issubclass(w.category, YoungTopMetallicityWarning)]


def test_warning_fires_exactly_when_prior_reaches_above_z004(ssp_clip):
    assert _warnings_for(ssp_clip, logzsol_hi=LOGZSOL_Z04 - 1e-6) == []
    assert _warnings_for(ssp_clip, logzsol_hi=0.2) == []
    hits = _warnings_for(ssp_clip, logzsol_hi=LOGZSOL_Z04 + 1e-3)
    assert len(hits) == 1
    hits_top = _warnings_for(ssp_clip, logzsol_hi=LOGZSOL_Z06)
    assert len(hits_top) == 1


def test_warning_counts_the_metallicity_scatter_kernel(ssp_clip):
    """hi + 3 x scatter_hi (the triweight kernel's span) decides, fixed or free scatter."""
    top = LOGZSOL_Z04  # prior upper bound sitting exactly on the highest complete node
    below = top - 0.30 - 1e-3
    assert _warnings_for(ssp_clip, logzsol_hi=below, scatter=0.1) == []
    hits = _warnings_for(ssp_clip, logzsol_hi=below + 3e-3, scatter=0.1)
    assert len(hits) == 1
    assert "scatter" in str(hits[0].message)
    # a free scatter prior counts through its upper bound
    assert _warnings_for(ssp_clip, logzsol_hi=top - 0.15 - 1e-3, scatter=Uniform(0.02, 0.05)) == []
    assert (
        len(_warnings_for(ssp_clip, logzsol_hi=top - 0.15 + 1e-3, scatter=Uniform(0.02, 0.05)))
        == 1
    )


def test_warning_text_states_limit_clip_and_exact_opt_in(ssp_clip):
    text = str(_warnings_for(ssp_clip, logzsol_hi=0.7)[0].message)
    assert "7.15" in text and "12 Msun" in text and "Chen et al. 2015" in text
    assert "Z = 0.0400" in text  # the clip
    assert "top_z='mist'" in text and "load_ssp(" in text and "load_ssp_data(" in text


def test_warning_names_raw_and_does_not_offer_mist_when_already_on(ssp_raw, ssp_mist):
    raw_text = str(_warnings_for(ssp_raw, logzsol_hi=0.7)[0].message)
    assert "top_z='raw'" in raw_text and "800-2700x low" in raw_text
    # With the MIST node the highest complete node is Z = 0.0449 ([Z/H] = +0.5).
    assert _warnings_for(ssp_mist, logzsol_hi=LOGZSOL_MIST - 1e-4) == []
    mist_text = str(_warnings_for(ssp_mist, logzsol_hi=0.7)[0].message)
    assert "MIST" in mist_text and "load_ssp(" not in mist_text
    assert "switches isochrone physics" in mist_text


def test_history_ingest_warns_for_a_metallicity_track_above_z004(ssp_clip):
    from tengri.inference.history_ingest import ingest_histories

    t = np.linspace(0.1, 10.0, 12)
    sfr = np.ones((1, 12))
    kw = dict(t_gyr=t, sfr=sfr, ssp_lgmet=np.asarray(ssp_clip.ssp_lgmet), top_z=ssp_clip.top_z)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ingest_histories(met=np.full((1, 12), 0.2), **kw)
        none = [w for w in caught if issubclass(w.category, YoungTopMetallicityWarning)]
        ingest_histories(met=np.full((1, 12), 0.55), **kw)
        some = [w for w in caught if issubclass(w.category, YoungTopMetallicityWarning)]
    assert none == [] and len(some) == 1


# ---------------------------------------------------------------------------
# (d) the MIST opt-in node
# ---------------------------------------------------------------------------
def test_mist_node_sits_at_its_true_metallicity(ssp_mist, ssp_clip):
    lgmet = np.asarray(ssp_mist.ssp_lgmet)
    assert lgmet.shape[0] == np.asarray(ssp_clip.ssp_lgmet).shape[0] + 1
    assert lgmet[-2] == pytest.approx(np.log10(MIST_Z), abs=1e-6)  # 0.04490, not 0.06
    assert lgmet[-1] == pytest.approx(np.log10(0.06), abs=1e-6)
    assert np.all(np.diff(lgmet) > 0)
    assert ssp_mist.top_z.policy == "mist"
    assert ssp_mist.top_z.fallback_log_z_abs == pytest.approx(np.log10(MIST_Z), abs=1e-6)


def test_mist_node_qh_follows_the_z_trend_at_log_age_6(raw_file, ssp_mist):
    wave = raw_file["ssp_wave"]
    i = int(np.argmin(np.abs(raw_file["ssp_lg_age_gyr"] + 9.0 - 6.0)))
    scale = 3.839e33 / 3.828e33
    q = lambda row: _q_h(wave, np.asarray(row))  # noqa: E731
    q03 = q(raw_file["ssp_flux"][12, i] * scale)
    q04 = q(raw_file["ssp_flux"][13, i] * scale)
    q_mist = q(np.asarray(ssp_mist.ssp_flux)[-2, i])
    q_raw06 = q(raw_file["ssp_flux"][14, i] * scale)
    lo, hi = sorted((q03, q04))
    assert 0.5 * lo < q_mist < 2.0 * hi  # same Z trend as the complete nodes (0.9x of Z=0.04)
    assert q_mist / q_raw06 > 500.0  # the 1000x deficit is gone
    # the young part of the truncated node is clipped onto the MIST node
    np.testing.assert_allclose(
        np.asarray(ssp_mist.ssp_flux)[-1, i], np.asarray(ssp_mist.ssp_flux)[-2, i]
    )


def test_mist_surviving_mass_is_its_own_builds(ssp_mist, ssp_clip):
    """Mass comes from the same FSPS-MIST build as the spectra: it is the node file's table."""
    from tengri.components.stellar.sps.top_z_validity import load_mist_node

    node = load_mist_node("fsps_prsc_miles_chabrier")
    mass = np.asarray(ssp_mist.ssp_mass_remaining)
    lg_age = np.asarray(ssp_mist.ssp_lg_age_gyr) + 9.0
    young = lg_age <= LAST_INCOMPLETE
    np.testing.assert_array_equal(mass[-2][young], node.mass_remaining[young])
    # MIST at log age 6: living stars + remnants ~0.99 of the formed mass
    i = int(np.argmin(np.abs(lg_age - 6.0)))
    assert 0.97 < mass[-2, i] < 1.0
    assert abs(mass[-2, i] - float(np.asarray(ssp_clip.ssp_mass_remaining)[13, i])) < 0.02


def test_wne_grid_gets_a_wne_mist_node():
    ssp = _load(WNE, top_z="mist")
    assert np.asarray(ssp.ssp_lgmet).shape[0] == 16
    assert ssp.top_z.policy == "mist"


def test_mist_old_ages_equal_the_parsec_log_z_interpolation(raw_file, ssp_mist):
    """Above the taper the inserted node is what the model interpolated there anyway."""
    lg_age = raw_file["ssp_lg_age_gyr"] + 9.0
    old = lg_age >= LAST_INCOMPLETE + TAPER
    lgm = raw_file["ssp_lgmet"]
    f = (np.log10(MIST_Z) - lgm[13]) / (lgm[14] - lgm[13])
    scale = 3.839e33 / 3.828e33
    expect = ((1 - f) * raw_file["ssp_flux"][13] + f * raw_file["ssp_flux"][14]) * scale
    # FSPS stores the MIST node's Z as a float32-precision 0.04490434: 5e-8 in the weight.
    np.testing.assert_allclose(np.asarray(ssp_mist.ssp_flux)[-2][old], expect[old], rtol=1e-6)


# ---------------------------------------------------------------------------
# (e) old ages unchanged bit for bit
# ---------------------------------------------------------------------------
def test_old_ages_and_other_nodes_are_unchanged_bit_for_bit(ssp_raw, ssp_clip):
    lg_age = np.asarray(ssp_raw.ssp_lg_age_gyr) + 9.0
    complete = lg_age >= LAST_INCOMPLETE + TAPER - 1e-9
    raw, clip = np.asarray(ssp_raw.ssp_flux), np.asarray(ssp_clip.ssp_flux)
    np.testing.assert_array_equal(clip[:14], raw[:14])  # every other node, every age
    np.testing.assert_array_equal(clip[14][complete], raw[14][complete])
    raw_m, clip_m = np.asarray(ssp_raw.ssp_mass_remaining), np.asarray(ssp_clip.ssp_mass_remaining)
    np.testing.assert_array_equal(clip_m[:14], raw_m[:14])
    np.testing.assert_array_equal(clip_m[14][complete], raw_m[14][complete])
    np.testing.assert_array_equal(np.asarray(ssp_clip.ssp_lgmet), np.asarray(ssp_raw.ssp_lgmet))
    assert not np.array_equal(clip[14][~complete], raw[14][~complete])


# ---------------------------------------------------------------------------
# (f) gradients through the clip
# ---------------------------------------------------------------------------
@pytest.mark.gradient
@pytest.mark.parametrize("policy", ["clip", "mist"])
def test_gradients_are_finite_through_the_clip(policy):
    ssp = _load(top_z=policy)
    obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"]))
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(100.0),
            "age_gyr": Fixed(0.002),
            "log_total_mass": Fixed(6.0),
            "all_params": Fixed(DEFAULT),
        },
        met={"logzsol": Uniform(-2.0, 0.7)},
        neb={"type": "none"},
        dust_attenuation={"type": "none"},
        redshift=Fixed(0.05),
    )

    def loss(lz):
        return jnp.sum(jnp.log(model.predict_photometry({"met_logzsol": lz})))

    grads = np.array([float(jax.grad(loss)(jnp.asarray(lz))) for lz in (0.3, 0.46, 0.55, 0.65)])
    gv = np.asarray(jax.vmap(jax.grad(loss))(jnp.asarray([0.3, 0.55, 0.65])))
    for g in (grads, gv):
        # a NaN from the clip would read "not zero" below, so finiteness is asserted first
        assert np.all(np.isfinite(g)), "gradient through the clip is non-finite"
        assert np.any(g != 0.0), "gradient through the clip is identically zero"
    assert grads[0] != 0.0  # below the clip the metallicity still moves the photometry


# ---------------------------------------------------------------------------
# grids without a recorded limit are untouched; bad spellings are refused
# ---------------------------------------------------------------------------
def test_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="top_z"):
        _load(top_z="mist2")


def test_grid_with_no_limit_on_record_is_untouched(synthetic_ssp_wide):
    assert getattr(synthetic_ssp_wide, "top_z", None) is None


# ---------------------------------------------------------------------------
# the limit table and the opt-in refuse what they cannot honor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0.0, -0.05, float("nan"), float("inf")])
def test_top_z_limit_refuses_a_non_positive_taper(bad):
    from tengri.components.stellar.sps.top_z_validity import TopZLimit

    with pytest.raises(ValueError, match="taper_dex"):
        TopZLimit(np.log10(0.06), np.log10(0.04), LAST_INCOMPLETE, bad, "test")
    ok = TopZLimit(np.log10(0.06), np.log10(0.04), LAST_INCOMPLETE, 0.05, "test")
    assert ok.taper_dex == 0.05


def test_registry_taper_is_one_grid_step():
    from tengri.components.stellar.sps.top_z_validity import TOP_Z_REGISTRY

    assert TOP_Z_REGISTRY["prsc"].taper_dex == pytest.approx(0.05)


def test_mist_refuses_a_library_without_a_shipped_node():
    """C3K has no MIST node: FileNotFoundError names the nodes that do ship."""
    from tengri.components.stellar.sps.top_z_validity import load_mist_node

    with pytest.raises(FileNotFoundError) as exc:
        load_mist_node("fsps_prsc_c3k_a_chabrier")
    text = str(exc.value)
    assert "mist_top_node_c3k_chabrier.h5" in text
    assert "mist_top_node_miles_chabrier.h5" in text
    assert "mist_top_node_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5" in text


def test_load_ssp_data_mist_refuses_a_c3k_grid():
    """The same refusal through the public loader, on the shipped C3K grid."""
    c3k = REPO / "data" / "fsps_prsc_c3k_a_chabrier.h5"
    if not c3k.is_file():  # pragma: no cover - data-less checkout
        pytest.skip(f"SSP data not found: {c3k}")
    with pytest.raises(FileNotFoundError, match=r"mist_top_node_c3k_chabrier\.h5"):
        load_ssp_data(str(c3k), top_z="mist")
