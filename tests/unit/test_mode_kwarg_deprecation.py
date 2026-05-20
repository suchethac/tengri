"""Phase 3e: ``mode=`` kwarg on predict_photometry / predict_spectrum is
deprecated in favour of ``predict_observables``.

The kwarg still functions identically — Phase 3c-3c follow-ups (two-component
dust LUT, free-z+dust) need to land before the explicit ``mode=`` path can
be removed.
"""

from __future__ import annotations

import contextlib
import pathlib
import warnings

import pytest

from tengri import Parameters, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation import Observation, Photometry
from tengri.parameters.priors import Fixed, Uniform

_SSP = pathlib.Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5").resolve()


@pytest.fixture(scope="module")
def model():
    if not _SSP.exists():
        pytest.skip(f"SSP not available at {_SSP}")
    ssp = load_ssp_data(str(_SSP))
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_peak_sfr=Uniform(-1, 3),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    phot = Photometry.from_names(["sdss_r"])
    obs = Observation(photometry=phot)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp, observation=obs)


_PARAMS = {
    "sfh_tsnorm_log_peak_sfr": 1.0,
    "sfh_tsnorm_peak_lbt_gyr": 2.0,
    "sfh_tsnorm_width_gyr": 1.0,
    "sfh_tsnorm_skew": 0.0,
    "sfh_tsnorm_trunc": 3.0,
}


@pytest.mark.parametrize("mode", ["exact", "compositional", "hybrid"])
def test_predict_photometry_mode_emits_deprecation(model, mode):
    """Explicit mode != 'auto' emits a DeprecationWarning that names
    predict_observables as the replacement."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Some modes may fail on this minimal model — we only care
        # that the warning was raised first.
        with contextlib.suppress(Exception):
            model.predict_photometry(_PARAMS, mode=mode)
        deprecation = [x for x in w if issubclass(x.category, DeprecationWarning)]
        msgs = [str(x.message) for x in deprecation]
        assert any("predict_observables" in m for m in msgs), (
            f"expected DeprecationWarning naming predict_observables, got: {msgs}"
        )


def test_predict_photometry_default_mode_no_warning(model):
    """``mode='auto'`` (default) does NOT emit a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.predict_photometry(_PARAMS)
        deprecation = [x for x in w if issubclass(x.category, DeprecationWarning)]
        mode_warnings = [x for x in deprecation if "predict_observables" in str(x.message)]
        assert not mode_warnings, (
            f"default mode should not warn, got: {[str(x.message) for x in mode_warnings]}"
        )
