# SPDX-License-Identifier: BSD-3-Clause
"""Regression: ``sed.fit`` must accept the ``Data`` record, like ``ForwardModel.fit`` (#1366).

Spec #1320 §3.2 sets one rule -- *models never hold measured values; data enters at
the action* -- and re-blesses ``SEDModel.fit`` as the astronomer one-liner, sugar
over ``ForwardModel.fit``. But only the canonical surface knew how to unpack a
``Data``: ``ForwardModel.fit`` branches on it and calls ``Data.validate_against``,
while ``SEDModel.fit`` had no such branch, so a record fell through to the
positional-argument check and produced

    ValueError: Provide either positional (data, noise) or keyword
                photometry=(flux, noise) / spectrum=(flux, noise).

which is *misleading rather than merely restrictive*: it describes the call shape
as wrong instead of saying this surface does not take this type, and never
mentions ``Data`` at all. Everything the record carries -- censoring, line fluxes,
joint photometry+spectrum -- was unreachable from the entry point we teach.

The fix forwards rather than re-implements. Unpacking a ``Data`` is ~70 lines that
ends by rebuilding the ``Observation`` for line fluxes; a second copy would drift
out of step. One record type, one validation seam, reached from both verbs.

These tests pin equivalence of the two surfaces, not merely that the sugar stopped
raising -- accepting the record but fitting something else would satisfy a weaker
assertion.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Data,
    Fixed,
    ForwardModel,
    NoiseModel,
    Observation,
    Photometry,
    SEDModel,
)

pytestmark = pytest.mark.regression_bug

_BANDS = ["sdss_g", "sdss_r", "sdss_i", "sdss_z", "2mass_ks"]


@pytest.fixture(scope="module")
def sed_and_mock(ssp_data_fsps):
    """A minimal fixed-parameter model plus one mock photometric observation."""
    obs = Observation(
        photometry=Photometry.from_names(_BANDS),
        noise=NoiseModel(calibration_floor=0.01, student_t_dof=None),
    )
    sed = SEDModel.build(
        ssp_data=ssp_data_fsps,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        redshift=Fixed(0.05),
    )
    params = {**sed.spec.get_fixed_values(), **sed.spec.sample(jax.random.PRNGKey(0))}
    mock = sed.mock(params, snr=20.0, key=jax.random.PRNGKey(1))
    return sed, np.asarray(mock.flux_obs), np.asarray(mock.noise)


def _fit(target, data, **kw):
    return target.fit(
        data, method="map", key=jax.random.PRNGKey(2), n_steps=2, verbose=False, **kw
    )


class TestSedFitAcceptsData:
    def test_sed_fit_accepts_a_data_record(self, sed_and_mock):
        """LOAD-BEARING. Neuter: drop the Data branch in ``convenience.fit_model``.

        Without it this raises ``ValueError: Provide either positional
        (data, noise) ...`` -- the misleading message this issue is about.
        """
        sed, flux, err = sed_and_mock
        result = _fit(sed, Data(photometry=(flux, err)))
        assert result is not None
        assert result.params, "sed.fit(Data) returned no parameters"

    def test_both_surfaces_agree_on_the_same_record(self, sed_and_mock):
        """Equivalence, not mere acceptance.

        ``sed.fit`` is documented as sugar for ``ForwardModel.build(sed=...).fit``,
        so the two must land on the same objective. Comparing the fitted parameters
        rather than just 'it did not raise' is what makes this test able to catch a
        sugar that quietly fits something else.
        """
        sed, flux, err = sed_and_mock
        record = Data(photometry=(flux, err))
        via_sugar = _fit(sed, record)
        via_canonical = _fit(ForwardModel.build(sed=sed), record)

        assert set(via_sugar.params) == set(via_canonical.params)
        for name, value in via_sugar.params.items():
            np.testing.assert_allclose(
                np.asarray(value),
                np.asarray(via_canonical.params[name]),
                rtol=1e-10,
                err_msg=f"sed.fit and ForwardModel.fit disagree on '{name}'",
            )

    def test_bare_arrays_still_work(self, sed_and_mock):
        """The sugar's original contract is untouched by the new branch."""
        sed, flux, err = sed_and_mock
        result = sed.fit(
            flux, err, method="map", key=jax.random.PRNGKey(2), n_steps=2, verbose=False
        )
        assert result.params


class TestValidationIsSharedNotDuplicated:
    def test_a_bad_censor_raises_the_same_error_on_both_surfaces(self, sed_and_mock):
        """The point of forwarding: ONE validation seam, so the messages match.

        A boolean censor array is rejected by ``Data.validate_against`` because
        ``True`` would silently mean 'upper limit'. Before the fix the sugar never
        reached that check and complained about ``(flux, noise)`` tuples instead.
        """
        sed, flux, err = sed_and_mock
        bad = Data(photometry=(flux, err), censor=np.array([True, False, True, False, True]))

        with pytest.raises(ValueError) as sugar_err:
            _fit(sed, bad)
        with pytest.raises(ValueError) as canonical_err:
            _fit(ForwardModel.build(sed=sed), bad)

        assert "censor" in str(sugar_err.value), (
            "sed.fit(Data) still reports a call-shape problem instead of the real "
            f"validation failure: {sugar_err.value}"
        )
        assert str(sugar_err.value) == str(canonical_err.value), (
            "the two surfaces validate differently, so the seam is duplicated"
        )


class TestAmbiguousCombinationsAreLoud:
    """A record plus a redundant argument must fail, not silently win or drop."""

    def test_data_plus_positional_noise(self, sed_and_mock):
        sed, flux, err = sed_and_mock
        with pytest.raises(TypeError, match="already carries its uncertainties"):
            sed.fit(
                Data(photometry=(flux, err)),
                err,
                method="map",
                key=jax.random.PRNGKey(2),
                n_steps=2,
            )

    def test_data_plus_channel_keyword(self, sed_and_mock):
        sed, flux, err = sed_and_mock
        with pytest.raises(TypeError, match="already carries every channel"):
            _fit(sed, Data(photometry=(flux, err)), photometry=(flux, err))

    def test_data_plus_data_type(self, sed_and_mock):
        """``data_type`` would be DROPPED, not honored, if forwarded blindly.

        ``ForwardModel.fit`` has no ``data_type`` parameter, so passing it through
        would send it to ``run()`` where it does nothing. It is redundant beside a
        record that already declares its own channels, so it is refused rather than
        silently ignored.
        """
        sed, flux, err = sed_and_mock
        with pytest.raises(TypeError, match="already declares its channels"):
            _fit(sed, Data(photometry=(flux, err)), data_type="photometry")


class TestValidateAgainstCoversBothChannels:
    """#1365: the spectroscopy channel and every noise array were unchecked.

    ``validate_against`` was half a seam. Photometry had its shapes and flux
    finiteness checked since spec 3.3; the spectrum branch checked
    ``spec_flux.shape`` and nothing else, and *neither* channel ever looked at the
    values in its uncertainty array. Measured before the fix, all four of these
    were ACCEPTED:

        spec_noise of the wrong length | spec_noise all NaN
        spec_flux containing NaN       | spec_noise negative

    A non-finite sigma poisons the entire likelihood rather than just its own
    term, and a non-positive one is a division by zero or a negative variance --
    each yields a plausible-looking fit instead of an error, which is why these
    have to be rejected at the seam rather than left to fail downstream.
    """

    @staticmethod
    def _spec_obs(npix=50):
        from tengri import NoiseModel, Observation, Spectroscopy

        return Observation(
            spectroscopy=Spectroscopy(wave_obs=np.linspace(4000.0, 7000.0, npix)),
            noise=NoiseModel(calibration_floor=0.01, student_t_dof=None),
        )

    def test_a_valid_spectrum_is_still_accepted(self):
        """Guards against a fix that simply rejects everything."""
        obs = self._spec_obs()
        Data(spectrum=(np.ones(50), np.full(50, 0.1))).validate_against(obs)

    @pytest.mark.parametrize(
        ("label", "flux", "noise", "match"),
        [
            ("noise wrong length", np.ones(50), np.ones(7), "noise shape"),
            ("noise all NaN", np.ones(50), np.full(50, np.nan), "NaN/inf spectrum uncertainty"),
            (
                "noise negative",
                np.ones(50),
                np.full(50, -0.1),
                "non-positive spectrum uncertainty",
            ),
            ("noise zero", np.ones(50), np.zeros(50), "non-positive spectrum uncertainty"),
            (
                "flux has NaN",
                np.where(np.arange(50) < 3, np.nan, 1.0),
                np.full(50, 0.1),
                "NaN/inf spectrum flux",
            ),
        ],
    )
    def test_bad_spectrum_is_rejected(self, label, flux, noise, match):
        """LOAD-BEARING. Every one of these was ACCEPTED before the fix."""
        obs = self._spec_obs()
        with pytest.raises(ValueError, match=match):
            Data(spectrum=(flux, noise)).validate_against(obs)

    @pytest.mark.parametrize(
        ("label", "noise", "match"),
        [
            ("NaN sigma", np.array([0.1, np.nan, 0.1]), "NaN/inf photometry uncertainty"),
            ("zero sigma", np.array([0.1, 0.0, 0.1]), "non-positive photometry uncertainty"),
            ("negative sigma", np.array([0.1, -0.2, 0.1]), "non-positive photometry uncertainty"),
        ],
    )
    def test_bad_photometry_uncertainty_is_rejected_and_names_the_band(self, label, noise, match):
        """The same rule applies to photometry, and points at the offending band."""
        from tengri import NoiseModel, Observation, Photometry

        obs = Observation(
            photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]),
            noise=NoiseModel(calibration_floor=0.01, student_t_dof=None),
        )
        with pytest.raises(ValueError, match=match) as err:
            Data(photometry=(np.ones(3), noise)).validate_against(obs)
        assert "sdss_r" in str(err.value), (
            f"{label}: the message should name the offending band, got {err.value}"
        )
