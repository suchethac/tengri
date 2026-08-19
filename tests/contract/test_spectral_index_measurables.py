# SPDX-License-Identifier: BSD-3-Clause
"""W3: spectral-index measurables through the forward model + precomp path.

Covers the predict_spectral_indices fix (rest-frame SED, no broken custom
grid), multiple simultaneous indices, the new ``slope`` index type (UV slope
β), and exact-vs-SpectrumPrecomp parity for window-based measurables.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import SpectrumPrecomp
from tengri.observation.spectral_indices import (
    STANDARD_INDICES,
    SpectralIndexData,
    SpectralIndexDef,
)

pytestmark = pytest.mark.contract

_SSP_CANDIDATES = [
    "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
    "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
]


def _ssp_path():
    return next((p for p in _SSP_CANDIDATES if Path(p).is_file()), None)


def _build(approx, wave_lo=3500.0):
    import warnings

    from tengri import FIXED, Fixed, Observation, SEDModel, Spectroscopy, load_ssp_data

    ssp_path = _ssp_path()
    if ssp_path is None:
        pytest.skip("No SSP grid available under data/.")
    ssp = load_ssp_data(ssp_path)
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=jnp.linspace(wave_lo, 7500.0, 300)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "*": FIXED},
            dust={
                "law_diff": "calzetti",
                "type": "two_component",
                "law_bc": "calzetti",
                "*": FIXED,
            },
            neb={"type": "none"},
            redshift=Fixed(0.05),
            approx=approx,
        )


class TestSlopeIndexType:
    def test_slope_def_requires_feature(self):
        with pytest.raises(ValueError, match="Slope indices require a feature"):
            SpectralIndexDef(name="x", index_type="slope", continuum=())

    def test_invalid_index_type(self):
        with pytest.raises(ValueError, match=r"EW.*break.*slope"):
            SpectralIndexDef(name="x", index_type="bogus", continuum=())


class TestSpectralIndexThroughModel:
    def test_predict_spectral_indices_fixed(self):
        # The pre-existing custom-grid bug is gone: this returns finite values.
        m = _build(None)
        vals = m.predict_spectral_indices({}, [STANDARD_INDICES["D4000"]])
        assert vals.shape == (1,)
        assert jnp.isfinite(vals[0])

    def test_multiple_indices_at_once(self):
        m = _build(None)
        defs = [STANDARD_INDICES[k] for k in ("D4000", "Hbeta", "Mgb")]
        vals = m.predict_spectral_indices({}, defs)
        assert vals.shape == (3,)
        assert bool(jnp.all(jnp.isfinite(vals)))

    def test_d4000_exact_vs_precomp(self):
        d4000 = [STANDARD_INDICES["D4000"]]
        ve = np.asarray(_build(None).predict_spectral_indices({}, d4000))
        vl = np.asarray(_build(SpectrumPrecomp()).predict_spectral_indices({}, d4000))
        np.testing.assert_allclose(vl, ve, rtol=1e-3)

    def test_uv_slope_matches_sed_property(self):
        m = _build(None, wave_lo=1300.0)
        beta_index = float(m.predict_spectral_indices({}, [STANDARD_INDICES["uv_slope_beta"]])[0])
        beta_ref = float(m.predict({}).sed.uv_slope_beta)
        assert abs(beta_index - beta_ref) < 0.05

    def test_uv_slope_exact_vs_precomp(self):
        uv = [STANDARD_INDICES["uv_slope_beta"]]
        ve = float(np.asarray(_build(None, 1300.0).predict_spectral_indices({}, uv))[0])
        vl = float(
            np.asarray(_build(SpectrumPrecomp(), 1300.0).predict_spectral_indices({}, uv))[0]
        )
        assert abs(vl - ve) < 1e-3


class TestSpectralIndexAsData:
    """Multiple indices (incl. UV slope) compose into the likelihood cohort."""

    def test_index_constraint_in_cohort(self):
        import warnings

        from tengri import FIXED, Fitter, Fixed, Observation, Photometry, SEDModel, load_ssp_data
        from tengri.inference.loss_functions import build_loglikelihood_fn

        ssp_path = _ssp_path()
        if ssp_path is None:
            pytest.skip("No SSP grid available under data/.")
        ssp = load_ssp_data(ssp_path)
        defs = (STANDARD_INDICES["D4000"], STANDARD_INDICES["Hbeta"])
        sid = SpectralIndexData(
            index_defs=defs, values=jnp.array([1.8, 2.0]), errors=jnp.array([0.05, 0.1])
        )
        obs = Observation(
            photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]), spectral_indices=sid
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SEDModel.build(
                ssp_data=ssp,
                observation=obs,
                sfh={"type": "dpl", "*": FIXED},
                dust={
                    "law_diff": "calzetti",
                    "type": "two_component",
                    "law_bc": "calzetti",
                    "*": FIXED,
                },
                neb={"type": "none"},
                redshift=Fixed(0.05),
            )
            fitter = Fitter(
                m,
                data=np.array([1e-28, 1.2e-28, 1.3e-28]),
                noise=np.array([1e-29, 1e-29, 1e-29]),
                data_type="photometry",
            )
        assert "spectral_index_constraint" in fitter._user_likelihood.name
        init = {n: jnp.asarray(0.0) for n in fitter.spec.free_params}
        ll = build_loglikelihood_fn(fitter)(init, fitter._data_args)
        assert jnp.isfinite(ll)
