# SPDX-License-Identifier: BSD-3-Clause
r"""The current-episode SFH types build through the grammar and publish their SFR (#3177).

``sfh={'type': ['delayed', 'recent']}`` and ``['delayed', 'recent_ssfr']`` were
registry-only: the grammar refused them and the stellar component had no forward
path for them. This file pins the build path end to end on the tracked FSPS SSP
grid: the short parameter names resolve, the episode SFR and the formed and
surviving masses are published with the values the equations give, the
photometry responds to ``log_sfr`` in the expected band order, the gradient is
finite, the sSFR form reproduces the ``recent`` form at the converted psi, and a
prior that reaches the sSFR pole is refused at build.

Bookkeeping (BEAGLE convention, see the registry entry): the smooth
``log_total_mass`` is the formed mass of the smooth history, and the total formed
mass is that plus psi * Delta t.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, FREE, Fixed, SEDModel, Uniform
from tengri.config.exceptions import ParameterError
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_paper

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_SSP_FILE = _DATA_DIR / "fsps_prsc_miles_chabrier.h5"

M_SMOOTH_LOG = 10.0
Z_FIXED = 1.0
PSI_LOG = 1.0  # psi = 10 Msun/yr
DT_MYR = 10.0
FUV_REST = 1500.0  # rest-frame FUV top-hat; observed at 1500 (1 + z) Angstrom
NIR_OBS = 20000.0  # 2 micron, observed frame


@pytest.fixture(scope="module")
def ssp():
    if not _SSP_FILE.is_file():
        pytest.skip(f"bare-stellar SSP not present: {_SSP_FILE}")
    return tengri.load_ssp_data(str(_SSP_FILE))


def _tophat(center, frac=0.12, n=60):
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


@pytest.fixture(scope="module")
def obs():
    curves = (_tophat(FUV_REST * (1.0 + Z_FIXED)), _tophat(NIR_OBS))
    return Observation(photometry=Photometry(filters=curves))


def _delayed(**overrides):
    base = {
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(3.0),
        "log_total_mass": Fixed(M_SMOOTH_LOG),
    }
    return {**base, **overrides}


def _build(ssp, obs, sfh):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh=sfh,
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        dust_emission={"type": "none"},
        redshift=Fixed(Z_FIXED),
    )


def _recent(ssp, obs, log_sfr=None, timescale_myr=None):
    sfh = {
        "type": ["delayed", "recent"],
        **_delayed(),
        "log_sfr": Uniform(-1.0, 2.0) if log_sfr is None else log_sfr,
        "timescale_myr": Fixed(DT_MYR) if timescale_myr is None else timescale_myr,
    }
    return _build(ssp, obs, sfh)


def test_short_names_resolve_to_sfh_recent(ssp, obs):
    spec_model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": ["delayed", "recent"],
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
            "log_sfr": Uniform(-1, 2),
            "timescale_myr": Fixed(10.0),
        },
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        dust_emission={"type": "none"},
        redshift=Fixed(1.0),
    )
    assert "sfh_recent_log_sfr" in spec_model.spec.free_params


def test_sfr_10myr_equals_psi_within_2pct(ssp, obs):
    m = _recent(ssp, obs, log_sfr=Fixed(PSI_LOG))
    sfr10 = float(m.predict_properties({}, names=("sfr_10myr",))["sfr_10myr"])
    ratio = sfr10 / 10.0**PSI_LOG
    print(f"sfr_10myr / psi = {ratio:.6f}")
    assert abs(ratio - 1.0) <= 0.02


def test_sfr_100myr_and_masses_published(ssp, obs):
    m = _recent(ssp, obs, log_sfr=Fixed(PSI_LOG))
    names = ("sfr_10myr", "sfr_100myr", "stellar_mass", "stellar_mass_surviving")
    props = m.predict_properties({}, names=names)
    values = {k: float(props[k]) for k in names}
    assert all(np.isfinite(v) for v in values.values()), values
    expected_total = 10.0**M_SMOOTH_LOG + 10.0**PSI_LOG * DT_MYR * 1e6
    assert abs(values["stellar_mass"] / expected_total - 1.0) <= 0.01, values
    assert values["stellar_mass_surviving"] < values["stellar_mass"]


def test_log_sfr_moves_fuv_more_than_nir(ssp, obs):
    m = _recent(ssp, obs)
    lo = np.asarray(m.predict_photometry({"sfh_recent_log_sfr": 0.0}))
    hi = np.asarray(m.predict_photometry({"sfh_recent_log_sfr": 1.0}))
    d_fuv = abs(hi[0] / lo[0] - 1.0)
    d_nir = abs(hi[1] / lo[1] - 1.0)
    print(f"d_fuv = {d_fuv:.4e}, d_nir = {d_nir:.4e}")
    assert d_fuv > d_nir


def test_grad_wrt_log_sfr_finite_positive_in_fuv(ssp, obs):
    m = _recent(ssp, obs)

    def fuv_flux(x):
        return m.predict_photometry({"sfh_recent_log_sfr": x})[0]

    g = float(jax.grad(fuv_flux)(0.5))
    assert np.isfinite(g) and g > 0.0


def test_timescale_moves_prediction(ssp, obs):
    short = _recent(ssp, obs, log_sfr=Fixed(PSI_LOG), timescale_myr=Fixed(10.0))
    long_ = _recent(ssp, obs, log_sfr=Fixed(PSI_LOG), timescale_myr=Fixed(100.0))
    p_short = np.asarray(short.predict_photometry({}))
    p_long = np.asarray(long_.predict_photometry({}))
    assert not np.allclose(p_short, p_long, rtol=1e-6, atol=0)


def test_ssfr_form_equals_log_sfr_form_at_converted_psi(ssp, obs):
    psi = 10.0
    m_smooth = 10.0**M_SMOOTH_LOG
    dt_yr = DT_MYR * 1e6
    psi_s = psi / (m_smooth + psi * dt_yr)
    recent = _recent(ssp, obs, log_sfr=Fixed(PSI_LOG))
    ssfr = _build(
        ssp,
        obs,
        {
            "type": ["delayed", "recent_ssfr"],
            **_delayed(),
            "log_ssfr": Fixed(float(np.log10(psi_s))),
            "timescale_myr": Fixed(DT_MYR),
        },
    )
    np.testing.assert_allclose(
        np.asarray(ssfr.predict_photometry({})),
        np.asarray(recent.predict_photometry({})),
        rtol=1e-10,
        atol=0,
    )


def test_ssfr_pole_refused_at_build(ssp, obs):
    with pytest.raises(ParameterError, match="pole"):
        _build(
            ssp,
            obs,
            {
                "type": ["delayed", "recent_ssfr"],
                **_delayed(),
                "log_ssfr": Uniform(-12.0, -7.3),
                "timescale_myr": Uniform(10.0, 1000.0),
            },
        )


def _ssfr_sfh(log_ssfr=None, timescale_myr=None):
    sfh = {"type": ["delayed", "recent_ssfr"], **_delayed()}
    if log_ssfr is not None:
        sfh["log_ssfr"] = log_ssfr
    if timescale_myr is not None:
        sfh["timescale_myr"] = timescale_myr
    return sfh


def test_default_ssfr_configuration_builds(ssp, obs):
    """Default sSFR box (Fixed 10 Myr duration, default log_ssfr prior) builds."""
    _build(ssp, obs, _ssfr_sfh(timescale_myr=Fixed(DT_MYR)))


def test_fixed_log_ssfr_with_fixed_timescale_builds(ssp, obs):
    _build(ssp, obs, _ssfr_sfh(log_ssfr=Fixed(-10.0), timescale_myr=Fixed(DT_MYR)))


def test_free_log_ssfr_with_fixed_timescale_builds(ssp, obs):
    """The default log_ssfr prior reaches psi_S * 10 Myr = 0.5, below the pole."""
    _build(ssp, obs, _ssfr_sfh(log_ssfr=FREE, timescale_myr=Fixed(DT_MYR)))


def test_free_timescale_with_free_log_ssfr_refused_naming_both_knobs(ssp, obs):
    """The FREE box reaches psi_S * 100 Myr = 5 >= 1 and is refused, naming both knobs."""
    with pytest.raises(ParameterError) as info:
        _build(ssp, obs, _ssfr_sfh(log_ssfr=FREE, timescale_myr=FREE))
    msg = str(info.value)
    assert "sfh_recent_ssfr_log_ssfr" in msg
    assert "sfh_recent_ssfr_timescale_myr" in msg


def test_free_timescale_with_narrowed_log_ssfr_builds(ssp, obs):
    """log_ssfr up to -8.5 keeps psi_S * 100 Myr = 0.32 < 1 over the whole duration prior."""
    _build(ssp, obs, _ssfr_sfh(log_ssfr=Uniform(-12.0, -8.5), timescale_myr=FREE))


def test_float32_matches_float64(ssp, obs):
    m64 = _recent(ssp, obs, log_sfr=Fixed(PSI_LOG))
    ref = np.asarray(m64.predict_photometry({}))
    with jax.enable_x64(False):
        m32 = _recent(ssp, obs, log_sfr=Fixed(PSI_LOG))
        got = np.asarray(m32.predict_photometry({}))
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=0)


def test_to_groups_round_trip(ssp, obs):
    sfh = {
        "type": ["delayed", "recent"],
        **_delayed(),
        "log_sfr": Uniform(-1.0, 2.0),
        "timescale_myr": Fixed(DT_MYR),
    }
    m = _build(ssp, obs, sfh)
    groups = m.spec.to_groups()
    rebuilt = _build(ssp, obs, groups["sfh"])
    assert sorted(rebuilt.spec.free_params) == sorted(m.spec.free_params)
    np.testing.assert_allclose(
        np.asarray(rebuilt.predict_photometry({"sfh_recent_log_sfr": 0.5})),
        np.asarray(m.predict_photometry({"sfh_recent_log_sfr": 0.5})),
        rtol=1e-12,
        atol=0,
    )


def test_menu_lists_both_types_as_production():
    from tengri.registry import list_sfh_models

    prod = {row["name"] for row in list_sfh_models(status="production")}
    assert {"recent", "recent_ssfr"} <= prod
