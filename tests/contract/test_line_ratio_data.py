# SPDX-License-Identifier: BSD-3-Clause
"""W4: measured emission line *ratios* as fittable data.

Covers the LineRatioData container, SEDModel.predict_line_ratios (exact vs
SpectrumPrecomp parity), and the likelihood-cohort wiring.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation import LineRatioData

pytestmark = pytest.mark.contract

_BARE_SSP_CANDIDATES = [
    "data/fsps_prsc_miles_chabrier.h5",
    "data/ssp_prsc_bc03_chabrier.h5",
]


def _bare_ssp_path():
    return next((p for p in _BARE_SSP_CANDIDATES if Path(p).is_file()), None)


class TestLineRatioData:
    def test_from_dict_and_shapes(self):
        lrd = LineRatioData.from_dict(
            {("Halpha", "Hbeta"): (4.2, 0.3), ("NII_6584", "Halpha"): (0.35, 0.05)}
        )
        assert lrd.n_ratios == 2
        assert lrd.numerators == ("Halpha", "NII_6584")
        assert lrd.denominators == ("Hbeta", "Halpha")
        assert lrd.numerator_waves.shape == (2,)

    def test_unknown_line_raises(self):
        with pytest.raises(ValueError, match="Unknown line name"):
            LineRatioData.from_dict({("NotALine", "Hbeta"): (1.0, 0.1)})

    def test_model_ratio_linear_and_log(self):
        lrd = LineRatioData.from_dict({("Halpha", "Hbeta"): (4.0, 0.3)})
        num = jnp.array([8.0])
        den = jnp.array([2.0])
        np.testing.assert_allclose(np.asarray(lrd.model_ratio(num, den)), [4.0])
        lrd_log = LineRatioData.from_dict(
            {("Halpha", "Hbeta"): (np.log10(4.0), 0.1)}, log_space=True
        )
        np.testing.assert_allclose(np.asarray(lrd_log.model_ratio(num, den)), [np.log10(4.0)])

    def test_log_likelihood_peaks_at_match(self):
        lrd = LineRatioData.from_dict({("Halpha", "Hbeta"): (4.0, 0.3)})
        num = jnp.array([8.0])
        ll_match = lrd.log_likelihood(num, jnp.array([2.0]))  # ratio 4.0 == obs
        ll_off = lrd.log_likelihood(num, jnp.array([8.0]))  # ratio 1.0 != obs
        assert float(ll_match) > float(ll_off)


class TestLineRatioPrediction:
    """predict_line_ratios is exact-vs-precomp identical (grid-independent lines)."""

    def _build(self, approx):
        import warnings

        from tengri import FIXED, Fixed, Observation, SEDModel, Spectroscopy, load_ssp_data

        bare = _bare_ssp_path()
        if bare is None or not Path("data/cue_weights.npz").is_file():
            pytest.skip("No bare-stellar SSP / Cue weights available.")
        ssp = load_ssp_data(bare)
        lrd = LineRatioData.from_dict(
            {("Halpha", "Hbeta"): (4.2, 0.3), ("NII_6584", "Halpha"): (0.35, 0.05)}
        )
        obs = Observation(
            spectroscopy=Spectroscopy(wave_obs=jnp.linspace(4500.0, 7500.0, 64)),
            line_ratios=lrd,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SEDModel.build(
                ssp_data=ssp,
                observation=obs,
                sfh={"type": "dpl", "*": FIXED},
                dust={
                    "type": "two_component",
                    "law": "calzetti",
                    "*": FIXED,
                },
                neb={"type": "cue", "*": FIXED},
                redshift=Fixed(0.05),
                approx=approx,
            )
        return m, lrd

    def test_precomp_matches_exact(self):
        from tengri import SpectrumPrecomp

        m_lut, lrd = self._build(SpectrumPrecomp())
        m_exact, _ = self._build(None)
        r_lut = m_lut.predict_line_ratios({}, lrd)
        r_exact = m_exact.predict_line_ratios({}, lrd)
        np.testing.assert_allclose(np.asarray(r_lut), np.asarray(r_exact), rtol=1e-6)


class TestLineRatioLikelihood:
    """The line-ratio term is composed into the likelihood cohort and active."""

    def _loss(self, obs_ratio):
        import warnings

        from tengri import FIXED, Fitter, Fixed, Observation, Photometry, SEDModel, load_ssp_data
        from tengri.inference.loss_functions import build_loglikelihood_fn

        bare = _bare_ssp_path()
        if bare is None or not Path("data/cue_weights.npz").is_file():
            pytest.skip("No bare-stellar SSP / Cue weights available.")
        ssp = load_ssp_data(bare)
        lrd = LineRatioData.from_dict({("Halpha", "Hbeta"): (obs_ratio, 0.3)})
        obs = Observation(
            photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]), line_ratios=lrd
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SEDModel.build(
                ssp_data=ssp,
                observation=obs,
                sfh={"type": "dpl", "*": FIXED},
                dust={
                    "type": "two_component",
                    "law": "calzetti",
                    "*": FIXED,
                },
                neb={"type": "cue", "*": FIXED},
                redshift=Fixed(0.05),
            )
            fitter = Fitter(
                m,
                data=np.array([1e-28, 1.2e-28, 1.3e-28]),
                noise=np.array([1e-29, 1e-29, 1e-29]),
                data_type="photometry",
            )
        assert "line_ratio_obs" in fitter._data_args
        assert "line_ratio_constraint" in fitter._user_likelihood.name
        init = {n: jnp.asarray(0.0) for n in fitter.spec.free_params}
        return float(build_loglikelihood_fn(fitter)(init, fitter._data_args))

    def test_ratio_term_constrains_fit(self):
        # Balmer decrement near the model truth (~2.78) beats an absurd ratio.
        ll_good = self._loss(2.8)
        ll_bad = self._loss(50.0)
        assert ll_good > ll_bad + 100.0
