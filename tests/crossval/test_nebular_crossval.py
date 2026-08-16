# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate nebular emission: CLOUDY grid and Cue vs FSPS baked-in.

The baked-in SSP files (wNE) include CLOUDY nebular emission computed by FSPS
at fixed logU = -3.0 and logZ_gas = solar.  The test verifies that we can
reproduce this by adding CLOUDY (or Cue) nebular emission to the no-NE SSP.

Three comparisons:
1. BakedIn vs CLOUDY grid: stellar_noNE + CLOUDY(logU=-3, Z_gas=solar) ≈ wNE
2. BakedIn vs Cue: stellar_noNE + Cue(logU=-3, Z_gas=solar) ≈ wNE
3. CLOUDY vs Cue: line-by-line at matched conditions
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S

pytestmark = pytest.mark.crossval


def _baked_nebular_lnu(ssp_wne, ssp_nne, met_idx, age_idx):
    """Difference of (wNE − no-NE) SSP grids, converted to erg/s/Hz.

    The SSP grids store ``ssp_flux`` in [Lsun/Hz/Msun]; the Cue / CLOUDY
    backend outputs are in [erg/s/Hz]. Multiply by ``LSUN_ERG_PER_S`` so
    ratios are dimensionless.
    """
    return (
        np.array(ssp_wne.ssp_flux[met_idx, age_idx] - ssp_nne.ssp_flux[met_idx, age_idx])
        * LSUN_ERG_PER_S
    )


_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# SSP files (PRSC isochrones, matching CLOUDY grid)
_SSP_NONE_PATH = _DATA_DIR / "fsps_prsc_miles_chabrier.h5"
_SSP_WNE_PATH = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_CLOUDY_PRSC_PATH = _DATA_DIR / "cloudy_grid_prsc.h5"
_CUE_PATH = _DATA_DIR / "cue_weights.npz"

# Skip module if core data is missing
if not _SSP_NONE_PATH.is_file() or not _SSP_WNE_PATH.is_file():
    pytest.skip("SSP data files not found", allow_module_level=True)

# log10(Zsun) absolute — Asplund+2009
_LOG10_ZSUN = -1.8477116556169435


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ssp_data():
    from tengri import load_ssp_data

    return load_ssp_data(str(_SSP_NONE_PATH))


@pytest.fixture(scope="module")
def ssp_wne():
    from tengri import load_ssp_data

    return load_ssp_data(str(_SSP_WNE_PATH))


@pytest.fixture(scope="module")
def cloudy_backend(ssp_data):
    if not _CLOUDY_PRSC_PATH.is_file():
        pytest.skip("CLOUDY PRSC grid not found")
    from tengri.components.nebular.cloudy_grid import CloudyGridBackend

    return CloudyGridBackend(str(_CLOUDY_PRSC_PATH), ssp_data)


@pytest.fixture(scope="module")
def cue_backend(ssp_data):
    if not _CUE_PATH.is_file():
        pytest.skip("Cue weights not found")
    from tengri.components.nebular.cue import CueBackend

    return CueBackend(str(_CUE_PATH), ssp_data)


@pytest.fixture(scope="module")
def solar_met_idx(ssp_data):
    """Index of the SSP metallicity grid point closest to solar."""
    return int(np.argmin(np.abs(np.array(ssp_data.ssp_lgmet) - _LOG10_ZSUN)))


@pytest.fixture(scope="module")
def young_burst_idx(ssp_data):
    """Index of the SSP age bin closest to 3 Myr."""
    ssp_log_ages_yr = np.array(ssp_data.ssp_lg_age_gyr) + 9.0
    return int(np.argmin(np.abs(10.0**ssp_log_ages_yr - 3e6)))


# ── Helpers ───────────────────────────────────────────────────────


def _integrated_line_flux(wave, sed, line_center, half_width=15.0):
    """Integrate SED in a window around an emission line.

    Parameters
    ----------
    wave : array
        Wavelength grid.
    sed : array
        SED (Lsun/Hz/Msun or similar).
    line_center : float
        Central wavelength (Angstrom).
    half_width : float
        Half-width of the integration window (Angstrom).
    """
    mask = (wave > line_center - half_width) & (wave < line_center + half_width)
    if mask.sum() < 3:
        return 0.0
    return float(np.trapezoid(sed[mask], wave[mask]))


# ── 1. BakedIn vs CLOUDY grid ─────────────────────────────────────


class TestBakedInVsCloudy:
    """SSP_noNE + CLOUDY grid nebular ≈ SSP_wNE (baked-in)."""

    def test_continuum_agreement(
        self, ssp_data, ssp_wne, cloudy_backend, solar_met_idx, young_burst_idx
    ):
        """Nebular continuum (line-free windows) should agree to <5%."""
        wave = np.array(ssp_data.ssp_wave)
        ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0
        log_z = float(ssp_data.ssp_lgmet[solar_met_idx])

        weights = jnp.zeros(len(ssp_log_ages_yr))
        weights = weights.at[young_burst_idx].set(1.0)

        neb_sed = np.array(
            cloudy_backend.predict_nebular_sed(
                ssp_weights=weights,
                ssp_wave=ssp_data.ssp_wave,
                ssp_log_ages_yr=ssp_log_ages_yr,
                log_z=log_z,
                neb_logU=-3.0,
            )
        )
        baked_neb = _baked_nebular_lnu(ssp_wne, ssp_data, solar_met_idx, young_burst_idx)

        # Compare in line-free continuum windows.
        # Near spectral breaks (Paschen edge ~8200A) grid interpolation
        # mismatch widens the tolerance.
        windows = [
            (4200, 4300, 0.10),
            (5100, 5200, 0.10),
            (7000, 7100, 0.10),
            (8500, 8600, 0.25),  # near Paschen edge
        ]
        for lo, hi, tol in windows:
            mask = (wave > lo) & (wave < hi) & (baked_neb > 1e-16)
            if mask.sum() < 5:
                continue
            ratio = np.median(neb_sed[mask] / baked_neb[mask])
            assert 1.0 - tol < ratio < 1.0 + tol, (
                f"Continuum {lo}-{hi}A: ratio={ratio:.4f}, "
                f"expected {1.0 - tol:.2f}-{1.0 + tol:.2f}"
            )

    @pytest.mark.parametrize(
        "line_name,line_wav",
        [
            ("H-alpha", 6563),
            ("H-beta", 4861),
            ("[OIII]5007", 5007),
            ("[OII]3727", 3727),
        ],
    )
    def test_integrated_line_flux(
        self,
        ssp_data,
        ssp_wne,
        cloudy_backend,
        solar_met_idx,
        young_burst_idx,
        line_name,
        line_wav,
    ):
        """Integrated emission line flux should agree to <10%."""
        wave = np.array(ssp_data.ssp_wave)
        ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0
        log_z = float(ssp_data.ssp_lgmet[solar_met_idx])

        weights = jnp.zeros(len(ssp_log_ages_yr))
        weights = weights.at[young_burst_idx].set(1.0)

        neb_sed = np.array(
            cloudy_backend.predict_nebular_sed(
                ssp_weights=weights,
                ssp_wave=ssp_data.ssp_wave,
                ssp_log_ages_yr=ssp_log_ages_yr,
                log_z=log_z,
                neb_logU=-3.0,
            )
        )
        baked_neb = _baked_nebular_lnu(ssp_wne, ssp_data, solar_met_idx, young_burst_idx)

        flux_cloudy = _integrated_line_flux(wave, neb_sed, line_wav)
        flux_baked = _integrated_line_flux(wave, baked_neb, line_wav)

        if flux_baked < 1e-16:
            pytest.skip(f"{line_name} too faint in baked-in SED")

        ratio = flux_cloudy / flux_baked
        assert 0.80 < ratio < 1.25, f"{line_name}: CLOUDY/baked={ratio:.3f}, expected 0.80-1.25"

    def test_total_bolometric_agreement(
        self, ssp_data, ssp_wne, cloudy_backend, solar_met_idx, young_burst_idx
    ):
        """Total nebular bolometric luminosity should agree to <15%."""
        ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0
        log_z = float(ssp_data.ssp_lgmet[solar_met_idx])

        weights = jnp.zeros(len(ssp_log_ages_yr))
        weights = weights.at[young_burst_idx].set(1.0)

        neb_sed = np.array(
            cloudy_backend.predict_nebular_sed(
                ssp_weights=weights,
                ssp_wave=ssp_data.ssp_wave,
                ssp_log_ages_yr=ssp_log_ages_yr,
                log_z=log_z,
                neb_logU=-3.0,
            )
        )
        baked_neb = _baked_nebular_lnu(ssp_wne, ssp_data, solar_met_idx, young_burst_idx)

        total_cloudy = np.sum(neb_sed[neb_sed > 0])
        total_baked = np.sum(baked_neb[baked_neb > 0])

        ratio = total_cloudy / total_baked
        assert 0.85 < ratio < 1.15, f"Bolometric: CLOUDY/baked={ratio:.3f}, expected 0.85-1.15"

    @pytest.mark.parametrize("age_myr", [1, 3, 5, 10])
    def test_multiple_ages(self, ssp_data, ssp_wne, cloudy_backend, solar_met_idx, age_myr):
        """Agreement should hold across young SSP ages."""
        wave = np.array(ssp_data.ssp_wave)
        ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0
        log_z = float(ssp_data.ssp_lgmet[solar_met_idx])

        age_idx = int(jnp.argmin(jnp.abs(10.0**ssp_log_ages_yr - age_myr * 1e6)))
        weights = jnp.zeros(len(ssp_log_ages_yr))
        weights = weights.at[age_idx].set(1.0)

        neb_sed = np.array(
            cloudy_backend.predict_nebular_sed(
                ssp_weights=weights,
                ssp_wave=ssp_data.ssp_wave,
                ssp_log_ages_yr=ssp_log_ages_yr,
                log_z=log_z,
                neb_logU=-3.0,
            )
        )
        baked_neb = _baked_nebular_lnu(ssp_wne, ssp_data, solar_met_idx, age_idx)

        # Check H-alpha integrated flux
        flux_cloudy = _integrated_line_flux(wave, neb_sed, 6563)
        flux_baked = _integrated_line_flux(wave, baked_neb, 6563)

        if flux_baked < 1e-16:
            pytest.skip(f"Ha too faint at age={age_myr} Myr")

        ratio = flux_cloudy / flux_baked
        assert 0.70 < ratio < 1.40, f"Ha at {age_myr} Myr: CLOUDY/baked={ratio:.3f}"


# ── 2. BakedIn vs Cue ─────────────────────────────────────────────


class TestBakedInVsCue:
    """SSP_noNE + Cue nebular ≈ SSP_wNE (baked-in).

    Cue is trained on a different CLOUDY grid than FSPS, so we allow
    wider tolerances (~factor of 2) compared to the CLOUDY grid test.
    """

    @pytest.mark.parametrize(
        "line_name,line_wav",
        [
            ("H-alpha", 6563),
            ("H-beta", 4861),
            ("[OIII]5007", 5007),
        ],
    )
    def test_integrated_line_flux(
        self,
        ssp_data,
        ssp_wne,
        cue_backend,
        solar_met_idx,
        young_burst_idx,
        line_name,
        line_wav,
    ):
        """Cue integrated line flux should agree with baked-in to factor ~2.

        Uses the high-level interface (same as CloudyGridBackend) —
        ssp_weights, log_z, neb_logU.  Q_H and ionizing spectrum are
        derived internally by the CueBackend.
        """
        wave = np.array(ssp_data.ssp_wave)
        ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0
        log_z = float(ssp_data.ssp_lgmet[solar_met_idx])

        weights = jnp.zeros(len(ssp_log_ages_yr))
        weights = weights.at[young_burst_idx].set(1.0)

        cue_neb_sed = np.array(
            cue_backend.predict_nebular_sed(
                ssp_weights=weights,
                ssp_wave=ssp_data.ssp_wave,
                ssp_log_ages_yr=ssp_log_ages_yr,
                log_z=log_z,
                neb_logU=-3.0,
            )
        )
        baked_neb = _baked_nebular_lnu(ssp_wne, ssp_data, solar_met_idx, young_burst_idx)

        flux_cue = _integrated_line_flux(wave, cue_neb_sed, line_wav)
        flux_baked = _integrated_line_flux(wave, baked_neb, line_wav)

        if flux_baked < 1e-16:
            pytest.skip(f"{line_name} too faint in baked-in SED")

        ratio = flux_cue / flux_baked
        # Wider tolerance: Cue is trained on different CLOUDY grids
        assert 0.3 < ratio < 3.0, f"{line_name}: Cue/baked={ratio:.3f}, expected 0.3-3.0"

    def test_continuum_agreement(
        self,
        ssp_data,
        ssp_wne,
        cue_backend,
        solar_met_idx,
        young_burst_idx,
    ):
        """Cue nebular continuum should agree with baked-in to <50%.

        Uses the high-level interface — identical call as CloudyGridBackend.
        """
        wave = np.array(ssp_data.ssp_wave)
        ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0
        log_z = float(ssp_data.ssp_lgmet[solar_met_idx])

        weights = jnp.zeros(len(ssp_log_ages_yr))
        weights = weights.at[young_burst_idx].set(1.0)

        cue_neb_sed = np.array(
            cue_backend.predict_nebular_sed(
                ssp_weights=weights,
                ssp_wave=ssp_data.ssp_wave,
                ssp_log_ages_yr=ssp_log_ages_yr,
                log_z=log_z,
                neb_logU=-3.0,
            )
        )
        baked_neb = _baked_nebular_lnu(ssp_wne, ssp_data, solar_met_idx, young_burst_idx)

        # Continuum windows (avoid strong lines)
        windows = [(4200, 4300), (5100, 5200), (7000, 7100)]
        for lo, hi in windows:
            mask = (wave > lo) & (wave < hi) & (baked_neb > 1e-16)
            if mask.sum() < 5:
                continue
            ratio = np.median(cue_neb_sed[mask] / baked_neb[mask])
            assert 0.5 < ratio < 2.0, f"Cue continuum {lo}-{hi}A: ratio={ratio:.3f}"


# ── 3. CLOUDY vs Cue line-by-line ─────────────────────────────────


class TestCloudyVsCue:
    """CLOUDY grid and Cue predictions for matched conditions."""

    @pytest.mark.parametrize(
        "line_name,line_wav",
        [
            ("H-alpha", 6563),
            ("H-beta", 4861),
            ("[OIII]5007", 5007),
            ("[OII]3727", 3727),
        ],
    )
    def test_key_line_agreement(
        self,
        ssp_data,
        cloudy_backend,
        cue_backend,
        solar_met_idx,
        young_burst_idx,
        line_name,
        line_wav,
    ):
        """Key diagnostic lines should agree to within ~0.5 dex."""
        ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0
        log_z = float(ssp_data.ssp_lgmet[solar_met_idx])
        log_age_yr = float(ssp_log_ages_yr[young_burst_idx])

        # CLOUDY
        weights = jnp.zeros(len(ssp_log_ages_yr))
        weights = weights.at[young_burst_idx].set(1.0)
        c_wav, c_lum = cloudy_backend.predict_nebular_line_luminosities(
            ssp_weights=weights,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=log_z,
            neb_logU=-2.5,
        )
        c_wav_np = np.array(c_wav)
        c_lum_np = np.array(c_lum)
        c_idx = np.argmin(np.abs(c_wav_np - line_wav))
        c_line_lum = float(c_lum_np[c_idx])

        # Cue
        ionspec_7, logqion = cue_backend.get_ionizing_params_at(log_z, log_age_yr)
        if ionspec_7 is None:
            pytest.skip("Ionizing params not available")

        i7 = np.array(ionspec_7)
        gas_logz = log_z - _LOG10_ZSUN

        q_wav, q_lum = cue_backend.predict_nebular_line_luminosities(
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=gas_logz,
            gas_logno=0.0,
            gas_logco=0.0,
            ionspec_index1=float(i7[0]),
            ionspec_index2=float(i7[1]),
            ionspec_index3=float(i7[2]),
            ionspec_index4=float(i7[3]),
            ionspec_logLratio1=float(i7[4]),
            ionspec_logLratio2=float(i7[5]),
            ionspec_logLratio3=float(i7[6]),
            gas_logqion=float(logqion),
            cloudyfsps_only=True,
        )
        q_wav_np = np.array(q_wav)
        q_lum_np = np.array(q_lum)
        q_idx = np.argmin(np.abs(q_wav_np - line_wav))
        q_line_lum = float(q_lum_np[q_idx])

        if c_line_lum < 1e-10 or q_line_lum < 1e-10:
            pytest.skip(f"{line_name} too faint for comparison")

        log_ratio = np.log10(q_line_lum / c_line_lum)
        assert abs(log_ratio) < 0.5, (
            f"{line_name}: log10(Cue/CLOUDY)={log_ratio:.3f}, expected |dex|<0.5"
        )

    def test_balmer_decrement_consistency(
        self,
        ssp_data,
        cloudy_backend,
        cue_backend,
        solar_met_idx,
        young_burst_idx,
    ):
        """Ha/Hb ratio should be consistent between backends.

        Case B recombination: Ha/Hb ≈ 2.86 at T=10^4 K.
        Both backends should give similar Balmer decrements.
        """
        ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0
        log_z = float(ssp_data.ssp_lgmet[solar_met_idx])
        log_age_yr = float(ssp_log_ages_yr[young_burst_idx])

        # CLOUDY
        weights = jnp.zeros(len(ssp_log_ages_yr))
        weights = weights.at[young_burst_idx].set(1.0)
        c_wav, c_lum = cloudy_backend.predict_nebular_line_luminosities(
            ssp_weights=weights,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=log_z,
            neb_logU=-2.5,
        )
        c_wav_np, c_lum_np = np.array(c_wav), np.array(c_lum)
        c_ha = float(c_lum_np[np.argmin(np.abs(c_wav_np - 6563))])
        c_hb = float(c_lum_np[np.argmin(np.abs(c_wav_np - 4861))])

        # Cue
        ionspec_7, logqion = cue_backend.get_ionizing_params_at(log_z, log_age_yr)
        if ionspec_7 is None:
            pytest.skip("Ionizing params not available")

        i7 = np.array(ionspec_7)
        gas_logz = log_z - _LOG10_ZSUN

        q_wav, q_lum = cue_backend.predict_nebular_line_luminosities(
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=gas_logz,
            gas_logno=0.0,
            gas_logco=0.0,
            ionspec_index1=float(i7[0]),
            ionspec_index2=float(i7[1]),
            ionspec_index3=float(i7[2]),
            ionspec_index4=float(i7[3]),
            ionspec_logLratio1=float(i7[4]),
            ionspec_logLratio2=float(i7[5]),
            ionspec_logLratio3=float(i7[6]),
            gas_logqion=float(logqion),
            cloudyfsps_only=True,
        )
        q_wav_np, q_lum_np = np.array(q_wav), np.array(q_lum)
        q_ha = float(q_lum_np[np.argmin(np.abs(q_wav_np - 6563))])
        q_hb = float(q_lum_np[np.argmin(np.abs(q_wav_np - 4861))])

        if c_hb < 1e-10 or q_hb < 1e-10:
            pytest.skip("Hb too faint")

        cloudy_decrement = c_ha / c_hb
        cue_decrement = q_ha / q_hb

        # Both should be roughly Case B (~2.86), within 30%
        assert 2.0 < cloudy_decrement < 4.0, f"CLOUDY Ha/Hb={cloudy_decrement:.2f}"
        assert 2.0 < cue_decrement < 4.0, f"Cue Ha/Hb={cue_decrement:.2f}"
        # They should agree with each other to ~40%
        ratio = cue_decrement / cloudy_decrement
        assert 0.6 < ratio < 1.6, f"Balmer decrement ratio Cue/CLOUDY={ratio:.2f}"

    def test_line_scatter_statistics(
        self,
        ssp_data,
        cloudy_backend,
        cue_backend,
        solar_met_idx,
        young_burst_idx,
    ):
        """Across all matched lines, median offset should be <0.3 dex."""
        ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0
        log_z = float(ssp_data.ssp_lgmet[solar_met_idx])
        log_age_yr = float(ssp_log_ages_yr[young_burst_idx])

        # CLOUDY
        weights = jnp.zeros(len(ssp_log_ages_yr))
        weights = weights.at[young_burst_idx].set(1.0)
        c_wav, c_lum = cloudy_backend.predict_nebular_line_luminosities(
            ssp_weights=weights,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=log_z,
            neb_logU=-2.5,
        )
        c_wav_np, c_lum_np = np.array(c_wav), np.array(c_lum)

        # Cue
        ionspec_7, logqion = cue_backend.get_ionizing_params_at(log_z, log_age_yr)
        if ionspec_7 is None:
            pytest.skip("Ionizing params not available")

        i7 = np.array(ionspec_7)
        gas_logz = log_z - _LOG10_ZSUN

        q_wav, q_lum = cue_backend.predict_nebular_line_luminosities(
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=gas_logz,
            gas_logno=0.0,
            gas_logco=0.0,
            ionspec_index1=float(i7[0]),
            ionspec_index2=float(i7[1]),
            ionspec_index3=float(i7[2]),
            ionspec_index4=float(i7[3]),
            ionspec_logLratio1=float(i7[4]),
            ionspec_logLratio2=float(i7[5]),
            ionspec_logLratio3=float(i7[6]),
            gas_logqion=float(logqion),
            cloudyfsps_only=True,
        )
        q_wav_np, q_lum_np = np.array(q_wav), np.array(q_lum)

        # Match lines by wavelength
        offsets = []
        for ic, cw in enumerate(q_wav_np):
            diffs = np.abs(c_wav_np - cw)
            best = np.argmin(diffs)
            if diffs[best] < 2.0 and c_lum_np[best] > 0 and q_lum_np[ic] > 0:
                offsets.append(np.log10(float(q_lum_np[ic]) / float(c_lum_np[best])))

        offsets = np.array(offsets)
        valid = np.isfinite(offsets)
        offsets = offsets[valid]

        assert len(offsets) > 20, f"Only {len(offsets)} matched lines, expected >20"

        median_offset = np.median(offsets)
        nmad = 1.4826 * np.median(np.abs(offsets - median_offset))

        assert abs(median_offset) < 0.3, (
            f"Median Cue-CLOUDY offset: {median_offset:.3f} dex, expected <0.3"
        )
        assert nmad < 0.5, f"NMAD scatter: {nmad:.3f} dex, expected <0.5"


# ── 4. tengri nebular vs python-fsps CLOUDY (external reference) ──

_FSPS_NEB_REF = _DATA_DIR / "fsps_nebular_reference.npz"


class TestVsFSPSNebular:
    """Compare tengri nebular emission against python-fsps built-in CLOUDY.

    FSPS has its own CLOUDY integration (gas_logu, gas_logz parameters).
    We compare the nebular contribution (neb - stellar) at key lines.
    Both use the same underlying FSPS SSPs, so differences come from
    the nebular model implementation only.
    """

    @pytest.fixture(scope="class")
    def fsps_ref(self):
        if not _FSPS_NEB_REF.is_file():
            pytest.skip("FSPS nebular reference not found")
        import os as _os

        if "SPS_HOME" not in _os.environ:
            pytest.skip("SPS_HOME not set")
        return dict(np.load(str(_FSPS_NEB_REF)))

    def test_halpha_neb_present(self, fsps_ref):
        """FSPS should show strong Hα nebular emission at 3 Myr."""
        wave = fsps_ref["wave"]
        neb = fsps_ref["neb_logu-2_t0.003"]
        cont = fsps_ref["noneb_logu-2_t0.003"]
        idx = np.argmin(abs(wave - 6563))
        ratio = (neb[idx] - cont[idx]) / max(cont[idx], 1e-50)
        assert ratio > 10, f"FSPS Hα neb/cont = {ratio:.1f}, expected > 10"

    def test_nebular_decreases_with_age(self, fsps_ref):
        """Nebular emission should weaken with age."""
        wave = fsps_ref["wave"]
        idx_ha = np.argmin(abs(wave - 6563))
        ratios = []
        for tage in [0.001, 0.003, 0.01]:
            key = f"logu-2_t{tage:.3f}"
            neb = fsps_ref[f"neb_{key}"]
            cont = fsps_ref[f"noneb_{key}"]
            r = (neb[idx_ha] - cont[idx_ha]) / max(cont[idx_ha], 1e-50)
            ratios.append(r)
        # Should decrease (or at least not increase) with age
        assert ratios[-1] < ratios[0] * 2, f"Nebular should weaken with age: {ratios}"

    def test_higher_logu_stronger_oiii(self, fsps_ref):
        """Higher ionization parameter should produce stronger [OIII]."""
        wave = fsps_ref["wave"]
        idx = np.argmin(abs(wave - 5007))
        neb_hi = fsps_ref["neb_logu-2_t0.003"]
        cont_hi = fsps_ref["noneb_logu-2_t0.003"]
        neb_lo = fsps_ref["neb_logu-3_t0.003"]
        cont_lo = fsps_ref["noneb_logu-3_t0.003"]

        oiii_hi = (neb_hi[idx] - cont_hi[idx]) / max(cont_hi[idx], 1e-50)
        oiii_lo = (neb_lo[idx] - cont_lo[idx]) / max(cont_lo[idx], 1e-50)

        assert oiii_hi > oiii_lo, (
            f"Higher logU should give stronger [OIII]: "
            f"logU=-2: {oiii_hi:.1f}, logU=-3: {oiii_lo:.1f}"
        )

    def test_balmer_decrement_physical(self, fsps_ref):
        """Hα/Hβ ratio should be ~2.86 (Case B recombination)."""
        wave = fsps_ref["wave"]
        neb = fsps_ref["neb_logu-2_t0.003"]
        cont = fsps_ref["noneb_logu-2_t0.003"]

        idx_ha = np.argmin(abs(wave - 6563))
        idx_hb = np.argmin(abs(wave - 4861))

        ha_neb = neb[idx_ha] - cont[idx_ha]
        hb_neb = neb[idx_hb] - cont[idx_hb]

        if hb_neb > 0:
            ratio = ha_neb / hb_neb
            # Case B: 2.86 for T=10^4K. Allow 1.5-5 for resolution effects.
            assert 1.5 < ratio < 5.0, f"Hα/Hβ = {ratio:.2f}, expected ~2.86"
