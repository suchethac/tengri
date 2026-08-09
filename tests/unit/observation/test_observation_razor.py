# SPDX-License-Identifier: BSD-3-Clause
"""#1321: Observation is schema-only. lines= declares WHICH lines;
value-carrying fields (line_fluxes, spectral_indices, line_ratios)
deprecate toward Data with a one-shot warning."""

import pytest


def test_lines_field_accepts_linelist():
    from tengri import LineList
    from tengri.observation import Observation, Photometry

    obs = Observation(
        photometry=Photometry.from_names(["sdss_r"]),
        lines=LineList.from_names(["Halpha"]),
    )
    assert "Halpha" in obs.lines.names


def test_line_fluxes_field_warns_deprecation(monkeypatch):
    """Constructing an Observation with line_fluxes emits a DeprecationWarning."""
    import jax.numpy as jnp

    import tengri.observation.observation as obs_module
    from tengri.observation import LineFluxData, Observation, Photometry

    # Reset the module-level flag for this test
    monkeypatch.setattr(obs_module, "_OBSERVATION_DEPRECATION_WARNED", False)

    # Create minimal LineFluxData
    line_flux_data = LineFluxData(
        names=("Halpha",),
        fluxes=jnp.array([1e-17]),
        errors=jnp.array([0.1e-17]),
        wavelengths=jnp.array([6564.61]),
    )

    with pytest.warns(DeprecationWarning, match="Data"):
        Observation(
            photometry=Photometry.from_names(["sdss_r"]),
            line_fluxes=line_flux_data,
        )


def test_deprecation_is_one_shot(monkeypatch):
    """Constructing multiple Observations with deprecated fields emits only one warning."""
    import jax.numpy as jnp

    import tengri.observation.observation as obs_module
    from tengri.observation import LineFluxData, Observation, Photometry

    # Reset the module-level flag for this isolated test
    monkeypatch.setattr(obs_module, "_OBSERVATION_DEPRECATION_WARNED", False)

    # Create minimal LineFluxData
    line_flux_data = LineFluxData(
        names=("Halpha",),
        fluxes=jnp.array([1e-17]),
        errors=jnp.array([0.1e-17]),
        wavelengths=jnp.array([6564.61]),
    )

    with pytest.warns(DeprecationWarning, match="Data") as record:
        # Create three Observations with the same deprecated field
        Observation(
            photometry=Photometry.from_names(["sdss_r"]),
            line_fluxes=line_flux_data,
        )
        Observation(
            photometry=Photometry.from_names(["sdss_i"]),
            line_fluxes=line_flux_data,
        )
        Observation(
            photometry=Photometry.from_names(["sdss_z"]),
            line_fluxes=line_flux_data,
        )

    # Count the deprecation itself, not every warning raised in the block.
    # `pytest.warns` applies its category/match filters only to decide whether
    # to FAIL; the recorder it returns still collects everything, so a bare
    # `len(record)` asserts "exactly one warning of any kind happened here" —
    # which an unrelated library warning breaks. A JAX compilation-cache
    # UserWarning does exactly that, making this fail order-dependently with
    # `assert 2 == 1` (#1584).
    deprecations = [
        w
        for w in record
        if issubclass(w.category, DeprecationWarning) and "Data" in str(w.message)
    ]
    assert len(deprecations) == 1, (
        f"the deprecation must fire once for three constructions; got "
        f"{len(deprecations)} matching, out of {len(record)} total warnings: "
        f"{[(w.category.__name__, str(w.message)) for w in record]}"
    )


def test_data_lines_resolve_through_schema():
    """Data.lines values are validated against Observation.lines schema."""
    import jax.numpy as jnp

    from tengri import Data, LineList, Observation, Photometry

    obs = Observation(
        photometry=Photometry.from_names(["sdss_r"]),
        lines=LineList.from_names(["Halpha"]),
    )

    d = Data(
        photometry=(jnp.ones(1), jnp.full(1, 0.1)),
        lines={"Halpha": (3.2e-17, 0.4e-17)},
    )
    v = d.validate_against(obs)
    assert v.line_values == {"Halpha": (3.2e-17, 0.4e-17)}
