# SPDX-License-Identifier: BSD-3-Clause
"""Dust attenuation step in profiling pipeline receives shape parameters via shared resolver.

The profiling pipeline's dust step must receive the same law kwargs the forward model
receives, via the shared resolver ``resolve_bc_diff_law_params``, so a shape parameter
of a non-power-law law (kriek_conroy's ``dust_delta``) reaches the dust call.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

from tengri import (
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    SSPData,
)
from tengri.components.dust.laws._registry import law_kwarg_names

pytestmark = pytest.mark.contract


def _build_ssp() -> SSPData:
    """Synthetic SSP: 25 log-ages, 3 metallicities, 1600-point wavelength grid."""
    ages = jnp.linspace(-3.0, 1.14, 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    wave = jnp.logspace(2.0, 7.0, 1600)
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages - ages.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(flux) + 1e-30,
        ssp_lg_age_gyr=ages,
        ssp_lgmet=lgmet,
    )


def _obs() -> Observation:
    return Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g"]))


class TestProfilingPipelineDustShapeParams:
    """Profiling pipeline resolver delivers shape params to dust attenuation call."""

    def test_resolver_delivers_shape_params_to_dust_call(self, monkeypatch):
        """Verify resolver delivers shape params like dust_delta to two_component_dust.

        The profiling pipeline builds bc_params/diff_params dicts via
        resolve_bc_diff_law_params (shared with forward model) and passes them to
        two_component_dust. This test verifies that shape parameters like dust_delta
        for kriek_conroy reach the dust call.
        """
        from tengri.profiling.pipeline import _profile_exact_path

        ssp = _build_ssp()
        obs = _obs()

        # Build model with two-component dust, kriek_conroy BC law with delta parameter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = Parameters(
                mean_sfh_type="dpl",
                dust_model="two_component",
                dust_law_bc="kriek_conroy",
                dust_law_diff="power_law",
                redshift=Fixed(0.1),
                sfh_dpl_alpha=Fixed(1.0),
                sfh_dpl_beta=Fixed(0.6),
                sfh_dpl_tau_gyr=Fixed(3.0),
                sfh_dpl_log_total_mass=Fixed(10.0),
                sfh_dpl_age_gyr=Fixed(5.0),
                met_logzsol=Fixed(0.0),
                dust_tau_bc=Fixed(1.0),
                dust_tau_diff=Fixed(0.3),
                dust_delta=Fixed(-0.3),
            )

        model = SEDModel(spec, ssp, observation=obs)

        # Monkeypatch two_component_dust to record calls
        recorded_calls = []

        def _recorder(wavelength, age_grid, tau_v1, tau_v2, **kwargs):
            """Record call kwargs and return unit attenuation (n_ages, n_wave)."""
            recorded_calls.append(kwargs)
            # two_component_dust returns shape (n_ages, n_wave)
            return jnp.ones((len(age_grid), len(wavelength)))

        monkeypatch.setattr(
            "tengri.components.dust.attenuation.two_component_dust",
            _recorder,
        )
        # Also patch in _apply if attenuation re-exports from there
        monkeypatch.setattr(
            "tengri.components.dust._apply.two_component_dust",
            _recorder,
        )

        # Get params dict: empty dict will trigger defaults from spec
        params = {}

        # Call _profile_exact_path
        report = _profile_exact_path(model, params, n=1)

        # Verify at least one recorded call received the resolver-built kwargs
        assert len(recorded_calls) > 0, "two_component_dust was not called"

        # Find the call that has bc_params and diff_params (the main step, not the timing lambda)
        main_call = None
        for call_kwargs in recorded_calls:
            if "bc_params" in call_kwargs and "diff_params" in call_kwargs:
                main_call = call_kwargs
                break

        assert main_call is not None, (
            "No call to two_component_dust with bc_params/diff_params found. "
            f"Recorded calls: {recorded_calls}"
        )

        # Verify kriek_conroy law and dust_delta in bc_params
        assert main_call.get("law_bc") == "kriek_conroy", (
            f"Expected law_bc='kriek_conroy', got {main_call.get('law_bc')}"
        )

        bc_params = main_call.get("bc_params", {})
        assert "dust_delta" in bc_params, (
            f"dust_delta not in bc_params={bc_params}. "
            "kriek_conroy law should receive dust_delta from resolver."
        )
        assert pytest.approx(bc_params["dust_delta"], rel=1e-10) == -0.3, (
            f"bc_params['dust_delta'] = {bc_params['dust_delta']}, expected -0.3"
        )

        # Verify no garbage in bc_params: every key must be in kriek_conroy's declared kwargs
        kriek_conroy_kwargs = law_kwarg_names("kriek_conroy")
        for key in bc_params:
            assert key in kriek_conroy_kwargs, (
                f"Unexpected key '{key}' in bc_params; kriek_conroy declares: "
                f"{kriek_conroy_kwargs}"
            )

        # Verify diff_params has power_law kwargs only
        diff_params = main_call.get("diff_params", {})
        power_law_kwargs = law_kwarg_names("power_law")
        for key in diff_params:
            assert key in power_law_kwargs, (
                f"Unexpected key '{key}' in diff_params; power_law declares: {power_law_kwargs}"
            )
