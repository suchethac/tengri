"""Unit tests for Galaxy facade and doctor() function."""

import pytest


def test_galaxy_class_exists():
    """Galaxy class should be importable from tengri."""
    from tengri import Galaxy

    assert Galaxy is not None


def test_doctor_returns_string():
    """doctor() should return a string."""
    from tengri import doctor

    out = doctor()
    assert isinstance(out, str)
    assert len(out) > 0


def test_doctor_mentions_jax_and_tengri():
    """doctor() output should mention JAX and tengri."""
    from tengri import doctor

    out = doctor()
    # Check for either case variation
    assert "jax" in out.lower() or "JAX" in out
    assert "tengri" in out.lower() or "Tengri" in out


def test_doctor_includes_python_version():
    """doctor() should report Python version."""
    from tengri import doctor

    out = doctor()
    assert "Python" in out or "python" in out


def test_from_arrays_signature():
    """Galaxy.from_arrays should have the documented signature."""
    from tengri import Galaxy

    # Without SSP, should raise a clear error
    with pytest.raises((ValueError, TypeError, FileNotFoundError, RuntimeError)):
        Galaxy.from_arrays(
            filters=["sdss_u", "sdss_g"],
            flux=[1e-28, 2e-28],
            flux_err=[1e-29, 1e-29],
            redshift=0.1,
        )


def test_from_observation_requires_ssp():
    """Galaxy.from_observation should require SSP data."""
    from tengri import Galaxy, Observation, Photometry

    obs = Observation(photometry=Photometry.from_names(["sdss_r", "sdss_i"]))

    with pytest.raises((ValueError, TypeError, FileNotFoundError)):
        Galaxy.from_observation(obs)


def test_presets_are_importable():
    """Presets should be importable from tengri.presets."""
    from tengri.presets import resolve_preset

    assert resolve_preset is not None


def test_resolve_preset_starforming():
    """resolve_preset('starforming') should return (Parameters, ModelConfig)."""
    from tengri.presets import resolve_preset

    params, config = resolve_preset("starforming", redshift=0.1)
    assert params is not None
    assert config is not None
    assert hasattr(params, "free_params")
    assert hasattr(config, "dust")


def test_resolve_preset_quiescent():
    """resolve_preset('quiescent') should return valid objects."""
    from tengri.presets import resolve_preset

    params, config = resolve_preset("quiescent", redshift=0.5)
    assert params is not None
    assert config is not None


def test_resolve_preset_high_z():
    """resolve_preset('high_z') should return valid objects."""
    from tengri.presets import resolve_preset

    params, config = resolve_preset("high_z", redshift=4.0)
    assert params is not None
    assert config is not None


def test_resolve_preset_invalid():
    """resolve_preset with invalid name should raise ValueError."""
    from tengri.presets import resolve_preset

    with pytest.raises(ValueError):
        resolve_preset("invalid_preset")


def test_galaxy_fit_requires_flux_data():
    """Galaxy.fit() should raise error if flux data not provided."""
    import pytest

    from tengri import Galaxy, Observation, Photometry
    from tengri.presets import resolve_preset

    obs = Observation(photometry=Photometry.from_names(["sdss_r"]))
    params, config = resolve_preset("starforming", redshift=0.1)

    # Mock SSP object (minimal)
    class MockSSP:
        pass

    g = Galaxy(
        ssp=MockSSP(),
        observation=obs,
        parameters=params,
        model_config=config,
    )

    with pytest.raises(AttributeError):
        g.fit(backend="map", verbose=False)


def test_galaxy_summary_requires_fit():
    """Galaxy.summary() should raise error if fit not called."""
    from tengri import Galaxy, Observation, Photometry
    from tengri.presets import resolve_preset

    obs = Observation(photometry=Photometry.from_names(["sdss_r"]))
    params, config = resolve_preset("starforming", redshift=0.1)

    class MockSSP:
        pass

    g = Galaxy(
        ssp=MockSSP(),
        observation=obs,
        parameters=params,
        model_config=config,
    )

    with pytest.raises(RuntimeError):
        g.summary()


def test_galaxy_plot_requires_fit():
    """Galaxy.plot() should raise error if fit not called."""
    from tengri import Galaxy, Observation, Photometry
    from tengri.presets import resolve_preset

    obs = Observation(photometry=Photometry.from_names(["sdss_r"]))
    params, config = resolve_preset("starforming", redshift=0.1)

    class MockSSP:
        pass

    g = Galaxy(
        ssp=MockSSP(),
        observation=obs,
        parameters=params,
        model_config=config,
    )

    with pytest.raises(RuntimeError):
        g.plot()


@pytest.mark.skip(reason="requires SSP data")
def test_galaxy_from_arrays_smoke():
    """Smoke test for Galaxy.from_arrays with real SSP data."""
    import os

    from tengri import Galaxy

    ssp_path = os.environ.get("TENGRI_SSP_PATH")
    if not ssp_path or not os.path.exists(ssp_path):
        pytest.skip("no SSP data available")

    g = Galaxy.from_arrays(
        filters=["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
        flux=[1e-28] * 5,
        flux_err=[1e-29] * 5,
        redshift=0.1,
        ssp_path=ssp_path,
        preset="starforming",
    )
    assert g is not None
    assert g.model is None  # Not built yet
