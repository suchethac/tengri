# SPDX-License-Identifier: BSD-3-Clause
r"""Under the fracAGN tie the intrinsic anchors do not depend on the disc reddening (#3407).

With ``agn_ir_frac > 0`` the disc is normalized to ``agn_power x R`` (#2603, #2750). The
2500 A and 4400 A anchors (``L_2500_intrinsic``, ``L_4400_intrinsic``) and the X-ray
corona that reads them are the intrinsic, un-reddened disc. Reddening the observed disc
(``agn_ebv_disc``) must reshape only the observed spectrum: at fixed AGN power the anchors
and the X-ray luminosity are invariant in ``agn_ebv_disc``, while the observed tied disc
still dims at 2500 A.

The anchors' unit-area shape is the un-reddened tie disc; before #3407 it was normalized
by the reddened integral, which put :math:`1/T(\mathrm{agn\_ebv\_disc})` into the anchors.
``kubota_done`` takes the disc/corona split path of the tie, ``multicolor`` the plain one.
"""

import jax
import numpy as np
import pytest

from tests._data_skip import DATA_DIR

pytestmark = pytest.mark.regression_bug

_SSP_FILE = DATA_DIR / "fsps_prsc_miles_chabrier.h5"
_EBV_VALUES = (0.0, 0.5, 2.0)
_RTOL = 1e-6


@pytest.fixture(scope="module", params=["multicolor", "kubota_done"])
def tied_model(request):
    """``(model, has_xray)`` for the #3407 report's composable AGN (z = 1, no lines).

    ``kubota_done`` carries its own corona, so the recipe's simple X-ray component is
    dropped for it; its X-ray check is then the anchors alone.
    """
    if not _SSP_FILE.is_file():
        pytest.skip(f"SSP grid not found at {_SSP_FILE}")
    from tengri import Fixed, Observation, Photometry, SEDModel, recipes
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    disc = request.param
    obs = Observation(photometry=Photometry.from_names(["galex_nuv", "sdss_r", "WISE_WISE_W3"]))
    cfg = recipes.composable_agn()
    cfg["redshift"] = Fixed(1.0)
    cfg["agn"]["disc"] = {"type": disc}
    for key in ("nlr", "blr", "feii"):
        cfg["agn"][key] = {"type": "none"}
    has_xray = disc != "kubota_done"
    if not has_xray:
        cfg["xray"] = {"type": "none", "all_params": Fixed(recipes.DEFAULT)}
    model = SEDModel.build(ssp_data=load_ssp_data(str(_SSP_FILE)), observation=obs, **cfg)
    return model, has_xray


@pytest.fixture(scope="module")
def tied_params(tied_model):
    """The sampled parameters with the #3407 AGN inputs and the tie on (``agn_ir_frac`` 0.1)."""
    model, _ = tied_model
    params = dict(model.spec.sample(jax.random.PRNGKey(1)))
    params.update(
        agn_cos_inc=0.5,
        agn_log_mbh=8.0,
        agn_polar_ebv=0.0,
        agn_lum_ratio=1.0,
        agn_ir_frac=0.1,
        agn_log_lbol=12.0,
    )
    return params


def _anchors_and_xray(tied_model, params, ebv):
    """``[L_2500_intrinsic, L_4400_intrinsic(, l_x_agn)]`` at one ``agn_ebv_disc``."""
    from tengri.forward import state_to_xray_quantities

    model, has_xray = tied_model
    state = model.predict_state({**params, "agn_ebv_disc": ebv})
    values = [
        float(np.asarray(state.derived["L_2500_intrinsic"])),
        float(np.asarray(state.derived["L_4400_intrinsic"])),
    ]
    if has_xray:
        values.append(float(np.asarray(state_to_xray_quantities(state).l_x_agn)))
    return np.asarray(values)


@pytest.fixture(scope="module")
def anchors_by_ebv(tied_model, tied_params):
    return {ebv: _anchors_and_xray(tied_model, tied_params, ebv) for ebv in _EBV_VALUES}


@pytest.mark.parametrize("ebv", _EBV_VALUES[1:])
def test_intrinsic_anchors_are_invariant_to_disc_reddening_under_the_tie(anchors_by_ebv, ebv):
    """``L_2500_intrinsic``, ``L_4400_intrinsic`` and ``l_x_agn`` do not depend on agn_ebv_disc."""
    np.testing.assert_allclose(anchors_by_ebv[ebv], anchors_by_ebv[0.0], rtol=_RTOL)


def test_observed_tied_disc_still_dims_at_2500A_with_reddening(tied_model, tied_params):
    """The observed disc is reddened: its 2500 A output falls as agn_ebv_disc rises."""
    model, _ = tied_model
    values = []
    for ebv in _EBV_VALUES:
        state = model.predict_state({**tied_params, "agn_ebv_disc": ebv})
        wave = np.asarray(state.wave)
        disc = np.asarray(state.derived["sed_agn_disc"])
        values.append(float(np.interp(2500.0, wave, disc)))
    assert values[0] > values[1] > values[2] > 0.0, values
