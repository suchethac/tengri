# SPDX-License-Identifier: BSD-3-Clause
"""Temple+2021 empirical quasar extinction curve (qsogen's own reddening law).

Reproduces qsogen's reddening prescription
``A_lambda = E(B-V) * [E(lambda-V)/E(B-V) + R]`` with ``R = 3.1``
(``qsosed.redden_spectrum``), distinct from the SMC Prevot law AGNfitter uses
for its ``EBVbbb`` (different curve *and* different convention). Covers the
functional form, the domain masking, JIT/grad safety, the atten-block factor,
the convention split from ``smc_prevot``, and the end-to-end build path.

References
----------
- Temple, Hewett & Banerji 2021, MNRAS, 508, 737 (arXiv:2109.04472)
- Prevot et al. 1984, A&A, 132, 389 (the *different* SMC law AGNfitter uses)
"""

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_paper

from tengri.components.agn.blocks.alternates import (
    qsogen_quasar_ext_block,
    smc_prevot_block,
)
from tengri.components.dust.qsogen_ext import QSOGEN_EXT_R, qsogen_quasar_extinction
from tests._grad_parity import assert_grad_matches_fd

# A_lambda/E(B-V) = E(lambda-V)/E(B-V) + R at R=3.1, read off the Temple+2021
# pl_ext_comp_03 curve. The color-excess curve is 0 at V, so A_V/E(B-V) = R.
_EXPECTED_A_OVER_EBV = {
    1500.0: 12.3144,
    2500.0: 7.6762,
    5500.0: 3.1000,  # V band -> exactly R
    6563.0: 2.5299,
}


class TestFunction:
    def test_reproduces_qsogen_formula(self):
        """A_lambda/E(B-V) matches curve + R at reference wavelengths."""
        for lam, expected in _EXPECTED_A_OVER_EBV.items():
            got = float(qsogen_quasar_extinction(jnp.array([lam]))[0])
            np.testing.assert_allclose(got, expected, atol=2e-3)

    def test_v_band_equals_R(self):
        """The color-excess curve is 0 at V, so A_V/E(B-V) = R exactly."""
        got = float(qsogen_quasar_extinction(jnp.array([5500.0]))[0])
        np.testing.assert_allclose(got, QSOGEN_EXT_R, atol=1e-3)

    def test_out_of_domain_is_zero(self):
        """No extinction outside the tabulated 500-60000 A domain."""
        out = qsogen_quasar_extinction(jnp.array([100.0, 1.0e7]))
        np.testing.assert_array_equal(np.asarray(out), np.zeros(2))

    def test_monotonic_uv_to_optical(self):
        """A_lambda/E(B-V) decreases from UV to optical (dust reddens the blue)."""
        wav = jnp.array([1500.0, 2500.0, 4400.0, 5500.0])
        a = np.asarray(qsogen_quasar_extinction(wav))
        assert np.all(np.diff(a) < 0.0)

    def test_R_scales_v_normalization(self):
        """Changing R shifts the whole curve by the R offset (color-excess form)."""
        wav = jnp.array([2500.0])
        a31 = float(qsogen_quasar_extinction(wav, R=3.1)[0])
        a20 = float(qsogen_quasar_extinction(wav, R=2.0)[0])
        np.testing.assert_allclose(a31 - a20, 1.1, atol=1e-6)

    def test_jit_grad_vmap(self):
        wav = jnp.linspace(500.0, 60000.0, 400)
        chex.assert_tree_all_finite(jax.jit(qsogen_quasar_extinction)(wav))

        def loss(w):
            return jnp.sum(qsogen_quasar_extinction(w))

        g = assert_grad_matches_fd(loss, jnp.array([2500.0, 5000.0]))
        chex.assert_tree_all_finite(g)


class TestBlock:
    def test_registered(self):
        from tengri.parameters.groups import _agn_block_types

        assert "qsogen" in _agn_block_types("attenuation")

    def test_zero_ebv_is_noop(self):
        wav = jnp.array([1500.0, 2500.0, 5500.0])
        factor = qsogen_quasar_ext_block(wav, agn_attenuation_ebv=0.0)
        np.testing.assert_allclose(np.asarray(factor), np.ones(3), atol=1e-12)

    def test_factor_matches_formula(self):
        """Block factor = 10^(-0.4 * (curve+R) * E(B-V))."""
        wav = jnp.array(list(_EXPECTED_A_OVER_EBV.keys()))
        ebv = 0.3
        factor = np.asarray(qsogen_quasar_ext_block(wav, agn_attenuation_ebv=ebv))
        expected = 10.0 ** (-0.4 * np.array(list(_EXPECTED_A_OVER_EBV.values())) * ebv)
        np.testing.assert_allclose(factor, expected, rtol=1e-4)

    def test_convention_differs_from_prevot(self):
        """qsogen and smc_prevot are different laws AND conventions.

        At V, qsogen's A_V/E(B-V) = R = 3.1 while Prevot's is R_V = 2.72; and
        the UV shapes differ, so the two blocks must not coincide.
        """
        wav = jnp.array([1500.0, 2500.0, 5500.0])
        ebv = 0.3
        q = np.asarray(qsogen_quasar_ext_block(wav, agn_attenuation_ebv=ebv))
        p = np.asarray(smc_prevot_block(wav, agn_attenuation_ebv=ebv))
        # V-band attenuation: qsogen uses R=3.1, Prevot R_V=2.72 -> distinct.
        assert not np.allclose(q, p, rtol=1e-2)
        av_q = -2.5 * np.log10(q[-1]) / ebv  # A_V/E(B-V)
        np.testing.assert_allclose(av_q, 3.1, atol=1e-3)


@pytest.mark.skipif(
    not (
        Path("data/bc03_pdva_stelib_chabrier.h5").is_file()
        or Path("data/bc03_chabrier.h5").is_file()
    ),
    reason="needs a BC03 SSP grid",
)
def test_end_to_end_through_build(ssp_data_bc03):
    """agn_attenuation_ebv with atten='qsogen' reddens the AGN SED through
    SEDModel.build (not a silent no-op), matching 10^(-0.4*(curve+R)*ebv)."""
    from tengri import DEFAULT, Fixed, SEDModel

    def build(ebv):
        return SEDModel.build(
            ssp_data=ssp_data_bc03,
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(10.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            agn={
                "type": "composable",
                "disc": {"type": "qsogen", "all_params": Fixed(DEFAULT)},
                "torus": {"type": "none"},
                "lines": {"type": "none"},
                "feii": {"type": "none"},
                "atten": {"type": "qsogen", "agn_attenuation_ebv": Fixed(ebv)},
                "agn_log_lbol": Fixed(11.0),
                "agn_polar_ebv": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.0),
        )

    s0 = build(0.0).predict_state({})
    s3 = build(0.3).predict_state({})
    wave = np.asarray(s0.wave)
    sed0 = np.asarray(s0.derived["sed_agn"])
    sed3 = np.asarray(s3.derived["sed_agn"])
    assert not np.allclose(sed0, sed3), "agn_attenuation_ebv had no effect (silent no-op)"

    def at(arr, lam):
        ok = arr > 0
        return float(np.interp(lam, wave[ok], arr[ok]))

    # V-band ratio: 10^(-0.4 * R * 0.3), R = 3.1.
    np.testing.assert_allclose(
        at(sed3, 5500.0) / at(sed0, 5500.0), 10.0 ** (-0.4 * 3.1 * 0.3), rtol=2e-3
    )
    # 2500 A ratio: A/E(B-V) = 7.676.
    np.testing.assert_allclose(
        at(sed3, 2500.0) / at(sed0, 2500.0), 10.0 ** (-0.4 * 7.676 * 0.3), rtol=2e-3
    )
