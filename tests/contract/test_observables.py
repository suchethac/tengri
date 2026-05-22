# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Observables NamedTuple synthesis and Phase 2 unification.

Unit tests that don't require real SSP data, focusing on:
- NamedTuple field presence/absence
- Magnitude property computation
- Observables type synthesis per model
"""

import math

import chex
import jax.numpy as jnp
import pytest

from tengri.observation import Observation, Photometry, Spectroscopy
from tengri.observation.observables import build_observables_class

pytestmark = pytest.mark.bounds


class TestObservablesBuilding:
    """Test build_observables_class() NamedTuple synthesis."""

    def test_observables_photometry_only(self):
        """Observables with photometry has phot_fnu, phot_rest_fnu, mag properties."""
        obs = Observation(photometry=Photometry.from_names(["sdss_r"]))
        Cls = build_observables_class(obs)

        o = Cls(phot_fnu=jnp.array([1e-26]), phot_rest_fnu=jnp.array([2e-26]))
        assert hasattr(o, "phot_fnu")
        assert hasattr(o, "phot_rest_fnu")
        assert hasattr(o, "mag_apparent")
        assert hasattr(o, "mag_absolute")
        assert not hasattr(o, "spec_fnu")

    def test_observables_spectroscopy_only(self):
        """Observables with spectroscopy has spec_fnu but not photometry fields."""
        obs = Observation(spectroscopy=Spectroscopy(wave_obs=jnp.linspace(3000, 9000, 100)))
        Cls = build_observables_class(obs)

        o = Cls(spec_fnu=jnp.ones(100))
        assert hasattr(o, "spec_fnu")
        assert not hasattr(o, "phot_fnu")
        assert not hasattr(o, "mag_apparent")

    def test_observables_joint(self):
        """Observables with both photometry and spectroscopy has both field sets."""
        obs = Observation(
            photometry=Photometry.from_names(["sdss_r"]),
            spectroscopy=Spectroscopy(wave_obs=jnp.linspace(3000, 9000, 100)),
        )
        Cls = build_observables_class(obs)

        o = Cls(
            phot_fnu=jnp.array([1e-26]),
            phot_rest_fnu=jnp.array([2e-26]),
            spec_fnu=jnp.ones(100),
        )
        assert o.phot_fnu[0] == 1e-26
        assert o.phot_rest_fnu[0] == 2e-26
        chex.assert_shape(o.spec_fnu, (100,))
        assert hasattr(o, "mag_apparent")
        assert hasattr(o, "mag_absolute")

    def test_observables_missing_channel_raises_attribute_error(self):
        """Accessing a missing channel raises AttributeError."""
        obs = Observation(photometry=Photometry.from_names(["sdss_r"]))
        Cls = build_observables_class(obs)

        o = Cls(phot_fnu=jnp.array([1e-26]), phot_rest_fnu=jnp.array([2e-26]))
        with pytest.raises(AttributeError):
            _ = o.spec_fnu

    def test_observables_magnitude_apparent(self):
        """Apparent magnitude computed correctly from phot_fnu."""
        obs = Observation(photometry=Photometry.from_names(["sdss_r"]))
        Cls = build_observables_class(obs)

        fnu = 1e-26  # erg/s/cm²/Hz
        o = Cls(phot_fnu=jnp.array([fnu]), phot_rest_fnu=jnp.array([2e-26]))

        # AB mag = -2.5 * log10(f_nu / f_0), where f_0 = 3.631e-20
        expected_mag = -2.5 * math.log10(fnu / 3.631e-20)
        computed_mag = float(o.mag_apparent[0])
        assert abs(computed_mag - expected_mag) < 1e-6

    def test_observables_magnitude_absolute(self):
        """Absolute magnitude computed correctly from phot_rest_fnu."""
        obs = Observation(photometry=Photometry.from_names(["sdss_r"]))
        Cls = build_observables_class(obs)

        fnu_rest = 2e-26
        o = Cls(phot_fnu=jnp.array([1e-26]), phot_rest_fnu=jnp.array([fnu_rest]))

        expected_mag = -2.5 * math.log10(fnu_rest / 3.631e-20)
        computed_mag = float(o.mag_absolute[0])
        assert abs(computed_mag - expected_mag) < 1e-6

    def test_observables_empty_observation_raises(self):
        """Observation with no sub-blocks raises ValueError at construction."""
        with pytest.raises(ValueError, match="at least one of"):
            Observation()

    def test_observables_fields_order(self):
        """NamedTuple fields appear in canonical order."""
        obs = Observation(
            photometry=Photometry.from_names(["sdss_r"]),
            spectroscopy=Spectroscopy(wave_obs=jnp.linspace(3000, 9000, 100)),
        )
        Cls = build_observables_class(obs)

        # _fields is a tuple of field names in definition order
        assert Cls._fields == ("phot_fnu", "phot_rest_fnu", "spec_fnu")
