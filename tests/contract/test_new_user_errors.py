# SPDX-License-Identifier: BSD-3-Clause
"""First-contact error-quality contracts.

Each case here reproduces something a new user actually did in QoL testing
and pins the friendly failure mode that replaced a crash or a misleading
message:

- ``predict_*({})`` with free params → ``MissingParameterError`` naming the
  params (was: bare ``KeyError: 'dust_tau_bc'`` from inside the dust
  component).
- Typo'd filter name → did-you-mean + the real public API name (was: a
  pointer to non-exported ``list_available_filters()``).
- Non-string SFH ``type`` → ``TypeError`` with an example (was:
  ``unhashable type: 'dict'``).
- ``'*': FIXED`` on the delayed SFH → silent (was: three internal-sounding
  midpoint warnings; registry-wide sweep tracked in #1007).
- Missing SSP file → points at ``tengri.download_ssp()`` (was: raw h5py
  OSError).
- wNE SSP detection is two-sided: real wNE grids fail HIGH (log Q_H ≈ 62,
  nebular continuum corrupting the ionizing fit), not just low.
"""

import os
import warnings

import pytest

from tengri import FIXED, Fixed, Photometry, SEDModel, load_ssp_data
from tengri.config.exceptions import MissingParameterError

pytestmark = pytest.mark.contract

_WEIGHTS = os.path.join("data", "cue_weights.npz")


@pytest.fixture(scope="module")
def dustless_model(synthetic_ssp_wide, synthetic_tophat_obs):
    """Minimal build — the auto-filled dust group leaves tau_bc/diff FREE."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "delayed", "*": FIXED},
            redshift=Fixed(0.1),
        )


def test_missing_free_params_raises_helpfully(dustless_model):
    with pytest.raises(MissingParameterError, match="dust_tau_bc"):
        dustless_model.predict_photometry({})


def test_missing_free_params_message_names_the_fix(dustless_model):
    with pytest.raises(MissingParameterError, match=r"spec\.sample"):
        dustless_model.mock({})


def test_full_params_still_predict(dustless_model):
    import jax
    import numpy as np

    params = dustless_model.spec.sample(jax.random.PRNGKey(0))
    flux = np.asarray(dustless_model.predict_photometry(params))
    assert np.isfinite(flux).all()


def test_unknown_filter_did_you_mean():
    with pytest.raises(KeyError, match="sdss_u"):
        Photometry.from_names(["sdss_q"])
    with pytest.raises(KeyError, match="list_filters"):
        Photometry.from_names(["sdss_q"])


def test_sfh_type_must_be_string():
    from tengri.parameters.groups import parse_groups

    with pytest.raises(TypeError, match="must be a string"):
        parse_groups(sfh={"type": {"oops": 1}}, redshift=Fixed(0.1))


def test_delayed_wildcard_is_warning_free(synthetic_ssp_wide, synthetic_tophat_obs):
    from tengri.parameters.groups import parse_groups

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_groups(sfh={"type": "delayed", "*": FIXED}, redshift=Fixed(0.1))
    midpoint_warnings = [w for w in caught if "no curated default" in str(w.message)]
    assert not midpoint_warnings, [str(w.message) for w in midpoint_warnings]


def test_missing_ssp_file_points_at_download():
    with pytest.raises(FileNotFoundError, match="download_ssp"):
        load_ssp_data("data/definitely_not_here.h5")


@pytest.mark.skipif(not os.path.exists(_WEIGHTS), reason="cue weights absent (data-gated)")
@pytest.mark.parametrize(
    ("fake_logqion", "match"),
    [(62.0, "far above"), (30.0, "well below")],
)
def test_wne_guard_is_two_sided(synthetic_ssp_wide, monkeypatch, fake_logqion, match):
    """Real wNE grids fail HIGH (observed log Q_H ≈ 62) — both sides must raise."""
    import numpy as np

    import tengri.components.nebular.ionizing_spectrum as ionspec
    from tengri.components.nebular.cue import CueBackend, CueWNESSPError

    orig = ionspec.precompute_ionizing_params_table

    def rigged(*args, **kwargs):
        result = orig(*args, **kwargs)
        result["logqion_table"] = np.full_like(np.asarray(result["logqion_table"]), fake_logqion)
        return result

    monkeypatch.setattr(ionspec, "precompute_ionizing_params_table", rigged)
    monkeypatch.delenv("TENGRI_ALLOW_WNE_CUE", raising=False)
    with pytest.raises(CueWNESSPError, match=match):
        CueBackend(_WEIGHTS, ssp_data=synthetic_ssp_wide)
