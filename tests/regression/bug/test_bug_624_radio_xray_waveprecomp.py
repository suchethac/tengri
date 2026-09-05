# SPDX-License-Identifier: BSD-3-Clause
"""Regression #624: radio and X-ray emission must be included in the
WavePrecomp / SpectrumPrecomp photometry projection.

Before the fix, radio (condon92) and X-ray (simple/yang20) were additive
bare-Protocol power-law emitters that added to the full-grid SED but never
published a ``*_phot_lnu_precomp`` family, so ``predict_via_precomp`` omitted
them — the radio band was 100% missing under ``approx=WavePrecomp()`` (LUT=0).
The components now publish ``radio_phot_lnu_precomp`` / ``xray_phot_lnu_precomp``
(the power law evaluated at filter effective wavelengths), summed as unattenuated
additive families.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_BARE_SSP = "data/fsps_prsc_miles_chabrier.h5"


def _ssp_or_skip():
    if not Path(_BARE_SSP).is_file():
        pytest.skip("No bare-stellar SSP grid available under data/.")
    from tengri import load_ssp_data

    return load_ssp_data(_BARE_SSP)


@pytest.mark.parametrize(
    "group,band_wave,fam",
    [
        ("radio", 2.14e9, "radio"),  # 1.4 GHz
        ("xray", 12.4, "xray"),  # ~1 keV
    ],
)
def test_radio_xray_in_waveprecomp(group, band_wave, fam):
    import warnings

    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, WavePrecomp
    from tengri.observation.photometry import FilterCurve
    from tengri.parameters.groups import _valid_radio_types, _valid_xray_types

    ssp = _ssp_or_skip()
    valid = (_valid_radio_types() if group == "radio" else _valid_xray_types()) - {"none"}
    if not valid:
        pytest.skip(f"no valid {group} type")
    # Not sorted(valid)[0]. For xray that is ``agn_xray_corona``, whose corona
    # is anchored to the AGN disc's L_2500 -- with no AGN in ``base`` below it
    # correctly emits nothing, and this test needs the band to carry flux. It
    # passed historically only because that name silently resolved to the
    # shared component and delivered yang20's XRB physics, which is driven by
    # star formation and so emits without an AGN (#1684).
    _PREFERRED = {"xray": "yang20", "radio": "condon92"}
    preferred = _PREFERRED.get(group)
    gtype = preferred if preferred in valid else sorted(valid)[0]

    filt = tuple(
        FilterCurve(
            wave=jnp.linspace(c * 0.85, c * 1.15, 40), trans=jnp.ones(40) * 0.5, name=f"b{c}"
        )
        for c in [6000.0, band_wave]  # optical anchor + the emitter's band
    )
    obs = Observation(photometry=Photometry(filters=filt))
    base = dict(
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
    )
    # #1980: the radio menu's {'type': name} spelling is retired — resolve the
    # preferred name onto the composable sf/agn axes (condon92 -> bell2003 +
    # powerlaw); other groups keep the type key.
    if group == "radio":
        from tengri.parameters.groups import _legacy_radio_type_to_blocks

        sf_variant, agn_variant = _legacy_radio_type_to_blocks(gtype)
        radio_cfg = {
            "sf": {"type": sf_variant},
            "agn": {"type": agn_variant},
            "all_params": Fixed(DEFAULT),
        }
        groups = dict(base, radio=radio_cfg)
    else:
        groups = dict(base, **{group: {"type": gtype, "all_params": Fixed(DEFAULT)}})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_exact = SEDModel.build(
            ssp_data=ssp, observation=obs, redshift=Fixed(0.05), approx=None, **groups
        )
        m_lut = SEDModel.build(
            ssp_data=ssp, observation=obs, redshift=Fixed(0.05), approx=WavePrecomp(), **groups
        )

    fams = [
        k for k in m_lut.predict_state({}).derived.field_names() if k.endswith("_phot_lnu_precomp")
    ]
    assert f"{fam}_phot_lnu_precomp" in fams, f"{fam} LUT family not published"

    pe = np.asarray(m_exact.predict_photometry({}))
    pl = np.asarray(m_lut.predict_photometry({}))
    # The emitter's band (index 1) must carry real flux and match exact (was
    # LUT=0 → rel=1.0 before the fix). Power laws are smooth → sub-% LUT error.
    assert pe[1] > 0, f"{group}: exact band flux should be > 0"
    rel_band = abs(pl[1] - pe[1]) / max(abs(pe[1]), 1e-30)
    assert rel_band < 0.02, f"{group} band WavePrecomp err {rel_band:.3%}"
