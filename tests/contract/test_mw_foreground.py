# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for MW foreground extinction (closes #297).

Before this PR, tengri's :class:`MilkyWay` dust component lived in
``mw_model.py`` and was exposed as ``dust={'type': 'mw', ...}``,
sharing the ``dust_`` parameter prefix with the host-galaxy
``two_component`` block. Only one of the two could be active at once;
users with sources at ``b_gal < 30°`` had to pre-correct fluxes
externally or give up host-galaxy dust modeling.

This PR adds a separate ``foreground={'ebmv_mw': ..., 'law': '...',
'rv': ...}`` kwarg on :meth:`SEDModel.build`. The screen is applied at
the observed-frame SED boundary (after IGM and redshifting) and uses
the closed-form ``cardelli()`` law from
``components/dust/attenuation.py``. Independent of the host ``dust``
block — both can be active simultaneously.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri.parameters.groups import Fixed, parse_groups

pytestmark = pytest.mark.contract

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"


@pytest.fixture(scope="module")
def ssp(synthetic_ssp_wide):
    # #613: synthetic SSP — the MW foreground screen multiplies the observed SED,
    # so its attenuation ratios are SSP-independent and these checks run on CI.
    return synthetic_ssp_wide


class TestGrammarPlumbing:
    """``foreground={...}`` reaches ``Parameters`` as flat structural settings."""

    def test_default_is_no_op(self):
        params = parse_groups(redshift=Fixed(0.05))
        assert params.foreground_ebmv_mw == 0.0
        assert params.foreground_law == "cardelli"
        assert params.foreground_rv == 3.1

    def test_grammar_accepts_flat_kwargs(self):
        params = parse_groups(
            foreground={"ebmv_mw": 0.05, "law": "cardelli", "rv": 3.1},
            redshift=Fixed(0.05),
        )
        assert params.foreground_ebmv_mw == 0.05
        assert params.foreground_law == "cardelli"
        assert params.foreground_rv == 3.1

    def test_unknown_law_raises(self):
        with pytest.raises(ValueError, match="Unknown foreground law"):
            parse_groups(
                foreground={"ebmv_mw": 0.05, "law": "not-a-real-law"},
                redshift=Fixed(0.05),
            )

    def test_negative_ebmv_raises(self):
        with pytest.raises(ValueError, match=r"ebmv_mw must be >= 0"):
            parse_groups(foreground={"ebmv_mw": -0.1}, redshift=Fixed(0.05))

    def test_zero_rv_raises(self):
        with pytest.raises(ValueError, match=r"rv must be > 0"):
            parse_groups(foreground={"ebmv_mw": 0.1, "rv": 0.0}, redshift=Fixed(0.05))


class TestForwardPassScreen:
    """The screen actually attenuates the observed SED."""

    @pytest.fixture
    def baseline_model(self, ssp):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return tengri.SEDModel.build(
                ssp,
                sfh={"type": "tsnorm", "all_params": tengri.Fixed(tengri.DEFAULT)},
                dust_attenuation={
                    "law": "power_law",
                    "type": "two_component",
                    "all_params": tengri.Fixed(tengri.DEFAULT),
                },
                redshift=tengri.Fixed(0.05),
            )

    @pytest.fixture
    def fg_model(self, ssp):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return tengri.SEDModel.build(
                ssp,
                sfh={"type": "tsnorm", "all_params": tengri.Fixed(tengri.DEFAULT)},
                dust_attenuation={
                    "law": "power_law",
                    "type": "two_component",
                    "all_params": tengri.Fixed(tengri.DEFAULT),
                },
                foreground={"ebmv_mw": 0.1, "law": "cardelli", "rv": 3.1},
                redshift=tengri.Fixed(0.05),
            )

    @pytest.fixture
    def sample_params(self, baseline_model):
        return dict(baseline_model.spec.sample(jax.random.PRNGKey(0)))

    def test_zero_ebmv_invariant(self, ssp, sample_params, baseline_model):
        """A model with ``ebmv_mw=0`` should match the no-foreground model bit-exactly."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zero_model = tengri.SEDModel.build(
                ssp,
                sfh={"type": "tsnorm", "all_params": tengri.Fixed(tengri.DEFAULT)},
                dust_attenuation={
                    "law": "power_law",
                    "type": "two_component",
                    "all_params": tengri.Fixed(tengri.DEFAULT),
                },
                foreground={"ebmv_mw": 0.0},
                redshift=tengri.Fixed(0.05),
            )
        out_base = baseline_model.predict_obs_sed(sample_params)
        out_zero = zero_model.predict_obs_sed(sample_params)
        np.testing.assert_allclose(out_zero.sed, out_base.sed, rtol=0)

    def test_positive_ebmv_attenuates(self, fg_model, baseline_model, sample_params):
        """A positive ``ebmv_mw`` reduces flux at V band."""
        out_base = baseline_model.predict_obs_sed(sample_params)
        out_fg = fg_model.predict_obs_sed(sample_params)
        # V band ≈ 5500 Å in the observed frame.
        iv = int(jnp.argmin(jnp.abs(out_base.wavelength - 5500.0)))
        ratio = float(out_fg.sed[iv] / out_base.sed[iv])
        # A_V = R_V * E(B-V) = 3.1 * 0.1 = 0.31 mag.
        # Expected transmission ≈ 10^(-0.4 * 0.31) ≈ 0.7499. Cardelli's
        # k(λ) at observed-frame 5500 Å for a z=0.05 source is close to
        # but not exactly 1.0, so allow some tolerance.
        np.testing.assert_allclose(ratio, 0.7499, atol=0.02)

    def test_screen_steeper_in_uv_than_optical(self, fg_model, baseline_model, sample_params):
        """Cardelli's law has steeper UV extinction → larger fractional
        attenuation at FUV vs V."""
        out_base = baseline_model.predict_obs_sed(sample_params)
        out_fg = fg_model.predict_obs_sed(sample_params)
        # FUV ≈ 1500 Å (observed frame), V ≈ 5500 Å.
        i_fuv = int(jnp.argmin(jnp.abs(out_base.wavelength - 1500.0)))
        i_v = int(jnp.argmin(jnp.abs(out_base.wavelength - 5500.0)))
        ratio_fuv = float(out_fg.sed[i_fuv] / out_base.sed[i_fuv])
        ratio_v = float(out_fg.sed[i_v] / out_base.sed[i_v])
        # FUV attenuated more than V (transmission smaller in UV).
        assert ratio_fuv < ratio_v, (
            f"Expected FUV attenuation > V attenuation; got "
            f"T_FUV={ratio_fuv:.3f}, T_V={ratio_v:.3f}"
        )


class TestComposabilityWithHostDust:
    """The whole point of #297 — MW foreground does NOT collide with
    the host-galaxy ``dust`` block."""

    def test_host_dust_and_foreground_both_active(self, ssp):
        """A model can carry both a host two-component dust block AND
        a MW foreground screen at the same time."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = tengri.SEDModel.build(
                ssp,
                sfh={"type": "tsnorm", "all_params": tengri.Fixed(tengri.DEFAULT)},
                dust_attenuation={
                    "law": "power_law",
                    "type": "two_component",
                    "all_params": tengri.Fixed(tengri.DEFAULT),
                    "tau_bc": tengri.Fixed(0.5),
                    "tau_diff": tengri.Fixed(0.3),
                },
                foreground={"ebmv_mw": 0.1},
                redshift=tengri.Fixed(0.05),
            )
        # Host dust is fully wired (two_component) AND foreground is
        # captured on the spec — no collision.
        assert model.spec.dust_model == "two_component"
        assert model.spec.foreground_ebmv_mw == 0.1
        # The forward pass succeeds.
        params = dict(model.spec.sample(jax.random.PRNGKey(0)))
        out = model.predict_obs_sed(params)
        assert out.sed.shape == out.wavelength.shape
