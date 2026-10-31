# SPDX-License-Identifier: BSD-3-Clause
"""Conservation: the Balmer lines above Cue's H10 carry exactly R_n x L(H-beta) of flux.

Cue's catalog ends at H10 (3798.98 A vacuum). The added lines H11 to H50 are Case-B
emissivity ratios of Storey & Hummer (1995, MNRAS 272, 41) to H-beta at 1e4 K and
1e2 cm^-3, appended to the catalog after the ionizing-photon escape and dust factors
act on H-beta, so each added line carries the same factors as the native lines.
The luminosity the profile painter puts in the SED rises by exactly the summed added
luminosities and nothing above 3800 A moves.

Wavelength provenance: NIST ASD H I energy levels (Kramida et al.), fetched
2026-10-31 from
https://physics.nist.gov/cgi-bin/ASD/energy1.pl?spectrum=H+I&units=0&format=2&output=0&page_size=15&multiplet_ordered=0&conf_out=on&term_out=on&level_out=on&unc_out=1&j_out=on&g_out=on&submit=Retrieve+Data
as vacuum Ritz wavelengths 1e8 / (E_n - E_2).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular._balmer_series import (
    BALMER_CASE_B_RATIOS_TO_HBETA,
    BALMER_RATIO_SENSITIVITY,
    BALMER_RITZ_VACUUM_AA_N3_TO_N10,
    HBETA_VACUUM_AA,
)
from tengri.components.nebular.cue import CueBackend
from tests._data_skip import CUE_WEIGHTS, requires_cue_weights

pytestmark = [pytest.mark.conservation, requires_cue_weights]

_KW = dict(
    gas_logu=-2.5,
    gas_logqion=50.0,
    ionspec_index1=19.7,
    ionspec_index2=5.3,
    ionspec_index3=1.6,
    ionspec_index4=0.6,
    ionspec_logLratio1=3.9,
    ionspec_logLratio2=0.01,
    ionspec_logLratio3=0.2,
)
_WAVE = jnp.linspace(3300.0, 5200.0, 19000)
_N_ADDED = len(BALMER_CASE_B_RATIOS_TO_HBETA)
_ADDED_WAVE = np.array([row[1] for row in BALMER_CASE_B_RATIOS_TO_HBETA])
_C_AA_PER_S = 2.99792458e18


@pytest.fixture(scope="module")
def on():
    return CueBackend(str(CUE_WEIGHTS))


@pytest.fixture(scope="module")
def off():
    return CueBackend(str(CUE_WEIGHTS), balmer_series_n_max=None)


def _is_added(wav):
    """Boolean mask of the entries of ``wav`` that are the added Balmer lines."""
    return np.isin(np.round(np.asarray(wav), 4), np.round(_ADDED_WAVE, 4))


def _hbeta(wav):
    return int(np.argmin(np.abs(np.asarray(wav) - HBETA_VACUUM_AA)))


def test_shipped_ratios_equal_pyneb():
    """The table is PyNeb's Storey & Hummer 1995 Case-B j(Hn)/j(Hbeta) at 1e4 K, 1e2 cm^-3."""
    pn = pytest.importorskip("pyneb")
    atom = pn.RecAtom("H", 1)
    den = atom.getEmissivity(tem=1e4, den=1e2, lev_i=4, lev_j=2)
    for n, _, ratio in BALMER_CASE_B_RATIOS_TO_HBETA:
        ref = atom.getEmissivity(tem=1e4, den=1e2, lev_i=n, lev_j=2) / den
        assert ratio == pytest.approx(ref, rel=1e-6), n
    levels = [row[0] for row in BALMER_CASE_B_RATIOS_TO_HBETA]
    assert levels == list(range(11, 51))
    assert [row[0] for row in BALMER_RATIO_SENSITIVITY] == levels


def test_series_beyond_n20_is_seven_percent_of_hbeta():
    """The series above n = 20 up to 50 sums to 5.6% of H-beta, so n_max = 20 is not enough."""
    tail = sum(r for n, _, r in BALMER_CASE_B_RATIOS_TO_HBETA if n > 20)
    assert tail == pytest.approx(0.0560, abs=5e-4)


def test_hydrogen_wavelengths_agree_with_cue_catalog(off):
    """H3 to H10 of the NIST Ritz table sit within 0.04 A of Cue's own catalog lines."""
    cat = np.asarray(off.published_line_wavelengths(cloudyfsps_only=False))
    for n, lam in BALMER_RITZ_VACUUM_AA_N3_TO_N10:
        near = cat[np.argmin(np.abs(cat - lam))]
        assert abs(near - lam) < 0.04, (n, lam, near)


def test_added_lines_are_ratio_times_hbeta(on, off):
    """Each added line is R_n L(Hb) to 1e-12; Hb and every native line are unchanged."""
    _, w_on, l_on = on.predict_nebular_split(ssp_wave=_WAVE, **_KW)
    _, w_off, l_off = off.predict_nebular_split(ssp_wave=_WAVE, **_KW)
    w_on, l_on, w_off, l_off = (np.asarray(a) for a in (w_on, l_on, w_off, l_off))
    added = _is_added(w_on)
    assert added.sum() == _N_ADDED
    assert w_off.shape[0] == sum(on.weights.batched_n_lines)
    np.testing.assert_array_equal(w_on[~added], w_off)
    np.testing.assert_array_equal(l_on[~added], l_off)
    hb = l_off[_hbeta(w_off)]
    expect = np.array([r for _, _, r in BALMER_CASE_B_RATIOS_TO_HBETA]) * hb
    np.testing.assert_allclose(l_on[added], expect, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(w_on[added], _ADDED_WAVE)
    assert np.flatnonzero(added)[0] == w_off.shape[0]


def test_added_lines_carry_the_escape_and_dust_factors(on, off):
    """With ionizing photons escaping and dust absorbing, an added line is R_n x that Hb."""
    kw = dict(_KW, neb_fesc=0.3, neb_fdust_frac=0.4)
    _, w_on, l_on = on.predict_nebular_split(ssp_wave=_WAVE, **kw)
    _, w_off, l_off = off.predict_nebular_split(ssp_wave=_WAVE, **kw)
    w_on, l_on, w_off, l_off = (np.asarray(a) for a in (w_on, l_on, w_off, l_off))
    added = _is_added(w_on)
    hb_k = l_off[_hbeta(w_off)]
    _, _, l_free = off.predict_nebular_split(ssp_wave=_WAVE, **_KW)
    assert hb_k < 0.999 * np.asarray(l_free)[_hbeta(w_off)]
    expect = np.array([r for _, _, r in BALMER_CASE_B_RATIOS_TO_HBETA]) * hb_k
    np.testing.assert_allclose(l_on[added], expect, rtol=1e-12, atol=0.0)


def test_sed_gains_exactly_the_added_luminosity_and_nothing_above_3800(on, off):
    """Flux in 3600-3800 A rises by sum(L_added); the SED above 3800 A is bit-identical."""
    sigma_aa = 0.5
    sed_on = np.asarray(on.predict_nebular_sed(ssp_wave=_WAVE, line_sigma_aa=sigma_aa, **_KW))
    sed_off = np.asarray(off.predict_nebular_sed(ssp_wave=_WAVE, line_sigma_aa=sigma_aa, **_KW))
    wave = np.asarray(_WAVE)
    win = (wave >= 3600.0) & (wave <= 3800.0)
    integrand = (sed_on - sed_off)[win] * _C_AA_PER_S / wave[win] ** 2
    gained = np.trapezoid(integrand, wave[win])
    _, w_on, l_on = on.predict_nebular_split(ssp_wave=_WAVE, **_KW)
    added_lsun = np.asarray(l_on)[_is_added(w_on)].sum()
    assert gained == pytest.approx(added_lsun * on.lsun_erg, rel=1e-6)
    _, w_off, l_off = off.predict_nebular_split(ssp_wave=_WAVE, **_KW)
    budget = sum(r for _, _, r in BALMER_CASE_B_RATIOS_TO_HBETA) * float(
        np.asarray(l_off)[_hbeta(w_off)]
    )
    assert gained == pytest.approx(budget * on.lsun_erg, rel=1e-6)
    above = wave > 3800.0
    np.testing.assert_array_equal(sed_on[above], sed_off[above])


def test_disabled_extension_is_the_native_catalog(off):
    """``balmer_series_n_max=None`` publishes exactly the weights' own lines and sigmas."""
    native_wav = np.sort(np.asarray(off.weights.nn_line_wav))
    np.testing.assert_array_equal(
        off.published_line_wavelengths(cloudyfsps_only=False), native_wav
    )
    np.testing.assert_array_equal(
        off.published_line_model_sigma(cloudyfsps_only=False),
        np.asarray(off.weights.line_emul_sigma_frac, dtype=np.float32),
    )
    assert off.balmer_series_line_names == ()


def test_legacy_subset_is_never_extended(on):
    """The 128-line CLOUDY/FSPS subset keeps its own catalog."""
    sub = on.published_line_wavelengths(cloudyfsps_only=True)
    assert sub.shape[0] == np.asarray(on.weights.line_old_idx).shape[0]
    assert not _is_added(sub).any()
    _, wav = on.predict_nebular_line_luminosities(cloudyfsps_only=True, **_KW)
    assert wav.shape[0] == sub.shape[0]


def test_published_catalog_wavelengths_sigma_and_names_align(on, off):
    """Wavelengths, lines, sigmas and names agree line by line, including the added lines."""
    waves = on.published_line_wavelengths(cloudyfsps_only=False)
    sigma = on.published_line_model_sigma(cloudyfsps_only=False)
    _, w_split, _ = on.predict_nebular_split(ssp_wave=_WAVE, **_KW)
    np.testing.assert_array_equal(waves, np.asarray(w_split))
    assert sigma.shape == waves.shape
    added = _is_added(waves)
    assert on.balmer_series_line_names == tuple(f"H{n}" for n in range(11, 51))
    hb_sigma = off.published_line_model_sigma(cloudyfsps_only=False)[_hbeta(waves[~added])]
    s_n = np.array([s for _, s in BALMER_RATIO_SENSITIVITY])
    np.testing.assert_allclose(
        sigma[added], np.sqrt(hb_sigma.astype(np.float64) ** 2 + s_n**2), rtol=1e-6
    )
    assert np.all(sigma[added] > hb_sigma)


def test_n_max_truncates_the_series(off):
    """``balmer_series_n_max=20`` adds H11 to H20 only."""
    b20 = CueBackend(str(CUE_WEIGHTS), balmer_series_n_max=20)
    waves = b20.published_line_wavelengths(cloudyfsps_only=False)
    assert waves.shape[0] == sum(b20.weights.batched_n_lines) + 10
    assert b20.balmer_series_line_names == tuple(f"H{n}" for n in range(11, 21))
    assert b20.cache_key() != off.cache_key()


def test_cache_key_distinguishes_the_extension(on, off):
    """The extension changes the catalog, so it changes the backend's cache key."""
    assert on.cache_key() != off.cache_key()
    assert on.cache_key() == CueBackend(str(CUE_WEIGHTS)).cache_key()


@pytest.mark.parametrize("bad", [9, 51, 20.5, True, "all"])
def test_invalid_n_max_is_refused(bad):
    """Only None or an integer from 10 to 50 is a Balmer upper level this table covers."""
    with pytest.raises(ValueError, match="balmer_series_n_max"):
        CueBackend(str(CUE_WEIGHTS), balmer_series_n_max=bad)


def test_added_luminosity_has_a_live_gradient_in_logu(on):
    """d(sum of added L)/d(neb_logU) is finite and nonzero under jit, and vmaps."""
    probe = np.asarray(on.published_line_wavelengths(cloudyfsps_only=False))
    mask = jnp.asarray(_is_added(probe))

    def added_sum(log_u):
        kw = dict(_KW, gas_logu=log_u)
        _, _, lum = on.predict_nebular_split(ssp_wave=_WAVE, **kw)
        return jnp.sum(jnp.where(mask, lum, 0.0))

    g = jax.jit(jax.grad(added_sum))(jnp.asarray(-2.5))
    assert bool(jnp.isfinite(g)) and abs(float(g)) > 0.0
    vals = jax.vmap(added_sum)(jnp.asarray([-3.0, -2.5, -2.0]))
    assert bool(jnp.all(jnp.isfinite(vals))) and bool(jnp.all(vals > 0.0))


def test_extension_survives_weights_threaded_as_a_jit_argument(on):
    """With the weights passed as a traced jit argument, the added lines are still R_n L(Hb)."""
    eager_wav = np.asarray(on.predict_nebular_split(ssp_wave=_WAVE, **_KW)[1])

    @jax.jit
    def run(weights):
        _, wav, lum = on.predict_nebular_split(ssp_wave=_WAVE, template_data=weights, **_KW)
        return wav, lum

    wav, lum = run(on.weights)
    np.testing.assert_allclose(np.asarray(wav), eager_wav, rtol=0.0, atol=1e-9)
    added = _is_added(eager_wav)
    lum = np.asarray(lum)
    ratios = np.array([r for _, _, r in BALMER_CASE_B_RATIOS_TO_HBETA])
    hb = lum[_hbeta(eager_wav)]
    np.testing.assert_allclose(lum[added], ratios * hb, rtol=1e-9)


def test_grammar_option_reaches_the_published_catalog(synthetic_ssp_wide):
    """``neb={'type': 'cue', 'balmer_series_n_max': 20}`` publishes H11 to H20 in line_lums."""
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel

    def published(neb):
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"])),
            redshift=Fixed(0.1),
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            neb=neb,
        )
        with jax.enable_x64(True):
            derived = model.predict_state({}).derived
        return model, np.asarray(derived["line_waves"]), np.asarray(derived["line_lums"])

    _, w_def, l_def = published({"type": "cue", "all_params": Fixed(DEFAULT)})
    model_20, w_20, l_20 = published(
        {"type": "cue", "all_params": Fixed(DEFAULT), "balmer_series_n_max": 20}
    )
    _, w_off, l_off = published(
        {"type": "cue", "all_params": Fixed(DEFAULT), "balmer_series_n_max": None}
    )
    n_native = sum(model_20._nebular_backend.weights.batched_n_lines)
    assert w_off.shape[0] == l_off.shape[0] == n_native
    assert w_20.shape[0] == l_20.shape[0] == n_native + 10
    assert w_def.shape[0] == l_def.shape[0] == n_native + 40
    hb = l_off[_hbeta(w_off)]
    added = _is_added(w_20)
    r = np.array([row[2] for row in BALMER_CASE_B_RATIOS_TO_HBETA[:10]])
    np.testing.assert_allclose(l_20[added], r * hb, rtol=1e-9)
    with pytest.raises(Exception, match="balmer_series_n_max"):
        published({"type": "cloudy", "balmer_series_n_max": 20})
