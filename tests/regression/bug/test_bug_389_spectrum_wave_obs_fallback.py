# SPDX-License-Identifier: BSD-3-Clause
"""Regression: ``SEDModel.predict_spectrum`` falls back to
``observation.spectroscopy.wave_obs`` when ``wave_obs`` is omitted
and no precompute has run (issue #389).

Before this fix, Fitter-driven spectroscopy fits raised
``ValueError: No wavelength grid``, because ``Fitter`` calls
``model.predict_spectrum(params)`` with no positional ``wave_obs``
and the docstring's promised "tier 2: observation.spectroscopy.wave_obs"
branch was missing.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def test_predict_spectrum_uses_observation_wave_obs_when_unset():
    """``predict_spectrum(params)`` must not raise when the model was
    constructed with ``Observation(spectroscopy=Spectroscopy(wave_obs=...))``."""
    pytest.importorskip("tengri")
    import tengri
    from tengri.observation import Observation, Spectroscopy

    try:
        ssp = tengri.load_ssp()
    except Exception:
        pytest.skip("SSP fixture not present")

    wave_obs = np.linspace(4000.0, 7000.0, 64)
    spec = Spectroscopy(wave_obs=wave_obs, resolution=2500.0)
    obs = Observation(spectroscopy=spec)

    model = tengri.SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": tengri.FIXED},
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": tengri.FIXED,
        },
        neb={"type": "ssp", "all_params": tengri.FIXED},
        redshift=tengri.Fixed(0.05),
    )

    params = {p: v.value for p, v in model.spec._distributions.items() if hasattr(v, "value")}
    # Must not raise — tier-2 fallback consults observation.spectroscopy.wave_obs.
    _ = model.predict_spectrum(params)


def test_predict_spectrum_raises_without_any_grid():
    """When neither wave_obs, precompute, nor observation.spectroscopy is
    available, predict_spectrum must still raise with a clear message."""
    pytest.importorskip("tengri")
    import tengri

    try:
        ssp = tengri.load_ssp()
    except Exception:
        pytest.skip("SSP fixture not present")

    # No observation at all → no spectroscopy fallback path.
    model = tengri.SEDModel.build(
        ssp_data=ssp,
        sfh={"type": "dpl", "all_params": tengri.FIXED},
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": tengri.FIXED,
        },
        neb={"type": "ssp", "all_params": tengri.FIXED},
        redshift=tengri.Fixed(0.05),
    )

    params = {p: v.value for p, v in model.spec._distributions.items() if hasattr(v, "value")}
    with pytest.raises(ValueError, match="No wavelength grid"):
        model.predict_spectrum(params)
