# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``bell2003_split`` never double-counts the thermal radio term (ruling R19).

:func:`tengri.radio.radio_sfr_bell2003_split` already allocates 10% of the
Bell (2003) TOTAL L(1.4 GHz) to a thermal component (Dale & Helou 2002;
Condon 1992 slope). Before this fix, the pipeline unconditionally added a
SECOND, independently-normalized :func:`tengri.radio.radio_freefree`
(Murphy+2011) term on top whenever ``radio={'sf': {'type':
'bell2003_split'}}`` reached the component, because ``include_freefree``
(``RadioSEDComponentConfig``, default ``True``) was never threaded from the
public build path and so always took its class default -- measured at
``ff/sf ~ 0.33-0.36`` (33-36% excess thermal radio flux) at
``L_ir = 1e44 erg/s``.

The fix lives in ``RadioSEDComponentConfig.__post_init__``
(``components/radio/component.py``): ``sfr_mode="bell2003_split"`` now
forces ``include_freefree=False`` unless the caller passes an explicit
``True``, which raises :class:`~tengri.config.exceptions.ConfigError` rather
than silently double-counting (explicit-over-silent, ADR-0011).
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.radio.component import RadioSEDComponentConfig
from tengri.config.exceptions import ConfigError
from tengri.observation import Photometry
from tengri.radio import radio_freefree, radio_sfr_bell2003, radio_sfr_bell2003_split

pytestmark = pytest.mark.contract

#: Matches ``tengri.components.radio.radio._RADIO_WAVE_MIN_AA`` (1 mm): the
#: SED is zero shortward of this by construction, so any comparison must be
#: restricted to the radio band or a false "extra term" would show up as a
#: mismatch in the non-radio wavelength range where BOTH sides are already
#: (correctly) zero for an unrelated reason.
_RADIO_WAVE_MIN_AA = 1.0e7


def _build(ssp_data, sf_type: str):
    """A dusty, energy-balanced galaxy with a real (nonzero) L_ir, SF radio only."""
    obs = Photometry.from_names(["sdss_g", "sdss_r"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="BakedInBackend")
        return SEDModel.build(
            ssp_data=ssp_data,
            observation=obs,
            sfh={
                "type": "delayed",
                "sfh_delayed_log_total_mass": Fixed(10.0),
                "sfh_delayed_tau_gyr": Fixed(1.0),
                "sfh_delayed_age_gyr": Fixed(5.0),
            },
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "tau_v": Fixed(0.5),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dh02_ce01", "all_params": Fixed(DEFAULT)},
            radio={
                "sf": {"type": sf_type},
                "agn": {"type": "none"},
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.0),
        )


def test_bell2003_split_component_equals_direct_function_no_extra_term(ssp_data_bc03):
    """End-to-end ``sed_radio`` == ``radio_sfr_bell2003_split(...)``, no free-free added.

    Mutant (see task report): remove the ``include_freefree`` exclusion in
    ``RadioSEDComponentConfig.__post_init__`` -- this assertion then fails
    because the component output picks up the extra ``radio_freefree`` term.
    """
    model = _build(ssp_data_bc03, "bell2003_split")
    params = model.spec.sample(jax.random.PRNGKey(0))
    state = model.predict_state(params)

    L_ir = float(state.derived["L_ir"])
    q_ir = float(params["radio_q_ir"])
    wave = np.asarray(state.wave)
    sed_radio = np.asarray(state.derived["sed_radio"])
    direct = np.asarray(radio_sfr_bell2003_split(wave, L_ir=L_ir, q_ir=q_ir))

    radio_mask = wave > _RADIO_WAVE_MIN_AA
    assert np.any(radio_mask), "no radio-band wavelength points in the model's grid"
    rel_diff = np.abs(sed_radio[radio_mask] / direct[radio_mask] - 1.0)
    assert np.max(rel_diff) < 1e-10, (
        f"bell2003_split component output diverges from the direct function by "
        f"{np.max(rel_diff):.3e} (expected < 1e-10) -- an extra term (e.g. "
        f"free-free) is present"
    )


def test_bell2003_still_adds_freefree(ssp_data_bc03):
    """The non-split ``bell2003`` mode is unchanged: free-free is still added."""
    model = _build(ssp_data_bc03, "bell2003")
    params = model.spec.sample(jax.random.PRNGKey(0))
    state = model.predict_state(params)

    L_ir = float(state.derived["L_ir"])
    q_ir = float(params["radio_q_ir"])
    alpha_sf = float(params["radio_alpha_sf"])
    wave = np.asarray(state.wave)
    sed_radio = np.asarray(state.derived["sed_radio"])

    sf_only = np.asarray(radio_sfr_bell2003(wave, L_ir, q_ir, alpha_sf))
    ff_only = np.asarray(
        radio_freefree(wave, L_ir, float(params["radio_T_e"]), float(params["radio_alpha_ff"]))
    )

    radio_mask = wave > _RADIO_WAVE_MIN_AA
    ff_over_sf = float(np.sum(ff_only[radio_mask]) / np.sum(sf_only[radio_mask]))
    assert ff_over_sf > 0.0, "bell2003 mode must still add a nonzero free-free term"

    np.testing.assert_allclose(
        sed_radio[radio_mask],
        (sf_only + ff_only)[radio_mask],
        rtol=1e-8,
        err_msg="bell2003 component output must equal sf + ff (both terms present)",
    )


def test_explicit_include_freefree_true_with_split_raises():
    """Explicit ``include_freefree=True`` + ``sfr_mode='bell2003_split'`` raises."""
    with pytest.raises(ConfigError, match="bell2003_split"):
        RadioSEDComponentConfig(sfr_mode="bell2003_split", include_freefree=True)

    # Explicit False is fine (redundant with the auto-resolution, not an error).
    cfg_false = RadioSEDComponentConfig(sfr_mode="bell2003_split", include_freefree=False)
    assert cfg_false.include_freefree is False

    # Unset (None, the default) resolves to False automatically -- the fix.
    cfg_auto = RadioSEDComponentConfig(sfr_mode="bell2003_split")
    assert cfg_auto.include_freefree is False

    # Other modes are unaffected: unset resolves to True, as before this fix.
    cfg_bell2003 = RadioSEDComponentConfig(sfr_mode="bell2003")
    assert cfg_bell2003.include_freefree is True
