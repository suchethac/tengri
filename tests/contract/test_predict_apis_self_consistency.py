# SPDX-License-Identifier: BSD-3-Clause
"""Cross-checks between public predict_* APIs (issues #436 and #442).

These are not physics tests — they pin contracts between APIs that users
mix in the same workflow and need to agree:

* :meth:`SEDModel.predict_magnitudes` must equal
  ``ab_mag_from_flux(SEDModel.predict_photometry(...))`` to within
  numerical noise. Before #436's fix the two routed through different
  filter-convolution conventions (Bessell & Murphy 2012 eq. A24 vs.
  ``dsps.calc_obs_mag``'s ``dν/ν`` weighting), disagreeing by 5–40 mmag
  in broad blue filters.

* :meth:`SEDModel.build` must emit a ``UserWarning`` whenever a
  ``met_logzsol*`` Fixed value or Uniform prior bound escapes the SSP
  grid. Before #442's fix the forward model silently clipped, producing
  a smooth-but-wrong SED that an optimizer would happily settle on.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.contract


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ssp(synthetic_ssp_wide):
    # #613: synthetic SSP (lgmet range [-2.5,-1.2]) so the self-consistency +
    # metallicity-grid-bounds checks run on CI. logzsol=0 → log10 Z ≈ -1.85 is
    # inside that range (builds cleanly); ±5/-10 fall outside (warn) — assertions
    # preserved.
    return synthetic_ssp_wide


@pytest.fixture(scope="module")
def photometric_model(ssp, synthetic_tophat_obs):
    """SEDModel with synthetic top-hat photometry — minimal setup for both bugs."""
    import tengri

    return tengri.SEDModel.build(
        ssp_data=ssp,
        observation=synthetic_tophat_obs,
        sfh={"type": "dexp", "*": tengri.FIXED},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_diff": tengri.Fixed(0.3),
            "tau_bc": tengri.Fixed(0.2),
        },
        redshift=tengri.Fixed(0.05),
    )


# ── #436: predict_magnitudes ↔ predict_photometry ─────────────────


class TestMagnitudesMatchPhotometry:
    """Two public APIs that share a filter convolution must agree."""

    def test_magnitudes_match_ab_mag_from_flux(self, photometric_model):
        """``predict_magnitudes`` ≡ ``ab_mag_from_flux ∘ predict_photometry`` ."""
        from tengri.observation.photometry import ab_mag_from_flux

        m = photometric_model
        params = dict(m.spec.sample(jax.random.PRNGKey(0)))

        flux = np.asarray(m.predict_photometry(params))
        mag_direct = np.asarray(m.predict_magnitudes(params))
        mag_from_flux = np.asarray(ab_mag_from_flux(flux))

        # Numerical noise from float32 SED + log conversion; well under
        # the 5-40 mmag bifurcation #436 reported.
        np.testing.assert_allclose(
            mag_direct,
            mag_from_flux,
            atol=1e-4,
            err_msg=(
                "predict_magnitudes and predict_photometry disagree by more "
                "than 0.1 mmag — they should share the same filter "
                "convolution (issue #436). Difference per band (mmag): "
                f"{(mag_direct - mag_from_flux) * 1000}"
            ),
        )


# ── #442: SSP grid edges — silent clipping ────────────────────────


class TestMetallicityBoundsValidation:
    """Building a model with an out-of-grid metallicity must warn."""

    def _build_with_met_logzsol(self, ssp, met_dist):
        """Helper — build the smallest model that exposes met_logzsol.

        Uses the short-form ``"logzsol"`` key inside the ``sfh`` group,
        which is how the nested-dict builder routes ``met_logzsol``
        overrides into the spec (the prefix gets stripped via
        ``_extract_short_name``).
        """
        import tengri

        return tengri.SEDModel.build(
            ssp_data=ssp,
            sfh={
                "type": "dexp",
                "*": tengri.FIXED,
                "logzsol": met_dist,
            },
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "*": tengri.FIXED,
                "tau_diff": tengri.Fixed(0.1),
                "tau_bc": tengri.Fixed(0.2),
            },
            redshift=tengri.Fixed(0.0),
        )

    def test_fixed_inside_grid_succeeds(self, ssp):
        """A Fixed value inside the grid range must build cleanly."""
        from tengri import Fixed

        # The FSPS / MILES / Chabrier grid covers ~ -2.3 to +0.6 log10(Z/Zsun)
        # — solar (0.0) is well inside.
        m = self._build_with_met_logzsol(ssp, Fixed(0.0))
        assert m is not None

    def test_fixed_above_grid_warns(self, ssp):
        """A Fixed value above the grid max must warn at build time.

        See the validator docstring for why this is a warning rather
        than a hard raise (some synthetic SSPs use lgmet directly as
        log10(Z/Zsun) rather than absolute log10(Z), and we don't lock
        them out).
        """
        from tengri import Fixed

        with pytest.warns(UserWarning, match=r"met_logzsol=5\.000 is outside"):
            self._build_with_met_logzsol(ssp, Fixed(5.0))

    def test_fixed_below_grid_warns(self, ssp):
        """A Fixed value below the grid min must also warn."""
        from tengri import Fixed

        with pytest.warns(UserWarning, match=r"met_logzsol=-10\.000 is outside"):
            self._build_with_met_logzsol(ssp, Fixed(-10.0))

    def test_uniform_inside_grid_succeeds(self, ssp):
        """A Uniform prior bounded inside the grid must build cleanly."""
        from tengri import Uniform

        m = self._build_with_met_logzsol(ssp, Uniform(-1.0, 0.3))
        assert m is not None

    def test_uniform_extending_above_grid_warns(self, ssp):
        """A Uniform prior whose upper bound escapes the grid must warn.

        Priors are softer than Fixed values — only tail samples land in
        the clamped region — so we warn rather than raise.
        """
        from tengri import Uniform

        with pytest.warns(UserWarning, match=r"met_logzsol prior bounds"):
            self._build_with_met_logzsol(ssp, Uniform(-1.0, 5.0))

    def test_uniform_extending_below_grid_warns(self, ssp):
        """A Uniform prior whose lower bound escapes the grid must warn."""
        from tengri import Uniform

        with pytest.warns(UserWarning, match=r"met_logzsol prior bounds"):
            self._build_with_met_logzsol(ssp, Uniform(-10.0, 0.0))
