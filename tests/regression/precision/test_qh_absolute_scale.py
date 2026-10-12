# SPDX-License-Identifier: BSD-3-Clause
"""Absolute scale of the ionizing photon rate and of the Hα luminosity (float64).

The float32 comparison in ``test_mappings_qh_float32.py`` checks one backend against itself
and cannot see a global scale error. These two tests fix the scale absolutely, through the
component that publishes it:

- Case B recombination ties ``L(Hα) = 1.37e-12 erg * Q_H`` (Osterbrock & Ferland 2006,
  Table 4.4), and the band is the one ``tests/contract/test_line_lums_units.py`` uses. A
  ``[Lsun]``-for-``[erg/s]`` mixup moves the ratio by 33.6 dex.
- A 1e10 Msun delayed-SFH population on the FSPS/MILES SSP has ``log10 q_h_total`` near 52.8
  (Cloudy). The table holds ``log10 Q_H`` per Msun shifted by its ``_log_qh_scale`` (~46.9),
  and dropping that shift leaves ~5.9.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.components.nebular import mappings_photo as mp
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tests._data_skip import requires_cloudy_grid

pytestmark = pytest.mark.regression_bug

_ROOT = Path(__file__).resolve().parents[3]
_FSPS = _ROOT / "data" / "fsps_prsc_miles_chabrier.h5"
_BPSS = _ROOT / "data" / "bpss_stars_c3k_a_chabrier.h5"

#: Case B, Osterbrock & Ferland (2006) Table 4.4, T = 1e4 K, n_e = 100 cm^-3 [erg per photon].
_CASE_B_ERG = 1.37e-12
#: Accepted band for L(Hα)/Q_H [erg], as in tests/contract/test_line_lums_units.py.
_RATIO_MIN, _RATIO_MAX = 1.0e-15, 1.0e-10
#: Hα/Hβ in Case B (Osterbrock & Ferland 2006, Table 4.4) and the Hβ value the grid carries.
_HA_OVER_HB_CASE_B = 2.86
_LOG_HB_PER_QH = -12.32

#: Cloudy log10 q_h_total of a 1e10 Msun delayed SFH (tau 1 Gyr, age 5 Gyr) on FSPS/MILES.
_CLOUDY_LOG_QH = 52.8
#: Band on that value [dex]. The measured value is 52.81.
_CLOUDY_LOG_QH_TOL = 0.1

_PARAMS = {"sfh_delayed_log_total_mass": 10.0, "sfh_delayed_tau_gyr": 1.0}
_HALPHA_AA = 6564.61


def _require(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"{path.name} is not present in this checkout")


def _model(ssp, neb_type, **neb_extra):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])),
        redshift=Fixed(0.1),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": Uniform(0.5, 3.0),
            "age_gyr": Fixed(5.0),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": neb_type, "all_params": Fixed(DEFAULT), **neb_extra},
    )


def _case_b_grid():
    """Synthetic stellar grid with Hα/Hβ at its Case B value (Hβ, index 1, has ratio 1)."""
    zo = jnp.array([0.1, 0.5, 1.0])
    shape = (3, 4, 2, 4, 2)
    ratios = jnp.ones((*shape, 5))
    ratios = ratios.at[..., 3].set(_HA_OVER_HB_CASE_B)  # line 3 is Hα (6564.61 A)
    return mp.MappingsStellarGridData(
        line_wavelengths=jnp.array([1215.67, 4862.68, 5008.24, 6564.61, 3728.0]),
        zo_axis=zo,
        logU_axis=jnp.linspace(-4.0, -1.0, 4),
        log_age_yr_axis=jnp.array([6.0, 6.5, 7.0, 7.5]),
        logn_axis=jnp.array([1.0, 3.0]),
        logHB_per_logq=jnp.full(shape, _LOG_HB_PER_QH),
        line_ratios=ratios,
    )


@pytest.mark.parametrize("ssp_path", [_FSPS, _BPSS], ids=["prsc", "bpss"])
def test_mappings_halpha_over_qh_is_case_b_through_the_component(ssp_path):
    """Through ``SEDModel.predict_state``, ``L(Hα)/Q_H`` lies in the repo's Case B band.

    Measured on the Case-B grid: 1.38e-12 erg (FSPS/MILES) and 1.35e-12 erg (BPASS). A
    [Lsun]-as-[erg/s] error would put it at ~5e21 erg.
    """
    _require(ssp_path)
    ssp_kw = {"mass_remaining": "dsps_fit"} if "bpss_" in ssp_path.name else {}
    ssp = load_ssp_data(str(ssp_path), **ssp_kw)
    with patch.object(
        mp, "_load_stellar_grid", return_value=(_case_b_grid(), ["cont", "inst"], 1, 0)
    ):
        model = _model(ssp, "mappings", ionizing_source_warning="suppress")
    derived = model.predict_state(_PARAMS).derived
    waves = np.asarray(derived["line_waves"], dtype=np.float64)
    lums = np.asarray(derived["line_lums"], dtype=np.float64)
    q_h = float(np.asarray(derived["nion"], dtype=np.float64))
    i = int(np.argmin(np.abs(waves - _HALPHA_AA)))
    assert abs(waves[i] - _HALPHA_AA) < 2.0, f"nearest line is {waves[i]:.2f} A, not Hα"
    ratio = lums[i] / q_h
    assert _RATIO_MIN < ratio < _RATIO_MAX, (
        f"mappings L(Hα)/Q_H = {ratio:.4e} erg, outside [{_RATIO_MIN:.0e}, {_RATIO_MAX:.0e}]. "
        f"Case B is {_CASE_B_ERG:.2e} erg: a 33.6-dex excess is the [Lsun]-as-[erg/s] error"
    )


@requires_cloudy_grid
def test_cloudy_q_h_total_carries_the_table_scale():
    """Prediction's ``q_h_total`` for a 1e10 Msun population is near 10**52.8 photons/s.

    ``q_h_total`` has no public accessor: :meth:`Prediction._ensure_lines` publishes it in the
    line cache, and it is read from there. The public ``log_q_h`` (the derived log_nion) is
    checked alongside it.
    """
    _require(_FSPS)
    ssp = load_ssp_data(str(_FSPS))
    model = _model(ssp, "cloudy")
    pred = model.predict(_PARAMS)
    public = float(pred.properties["log_q_h"])
    pred._ensure_lines()
    q_h_total = float(pred._cache["q_h_total"])
    log_total = float(np.log10(q_h_total))
    assert abs(public - _CLOUDY_LOG_QH) < _CLOUDY_LOG_QH_TOL, f"log_q_h = {public:.3f}"
    assert abs(log_total - _CLOUDY_LOG_QH) < _CLOUDY_LOG_QH_TOL, (
        f"Cloudy q_h_total = 10**{log_total:.3f} photons/s, expected 10**{_CLOUDY_LOG_QH} +/- "
        f"{_CLOUDY_LOG_QH_TOL}. A value near 10**5.9 means the table's ~10**46.9 scale is missing"
    )
