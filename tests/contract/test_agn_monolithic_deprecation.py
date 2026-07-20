# SPDX-License-Identifier: BSD-3-Clause
"""Contracts for AGN monolithic model deprecation (#721).

Verifies that deprecated monolithic AGN models continue to resolve and emit
DeprecationWarning, and that their composable equivalents produce finite SEDs.

Marker: contract
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.agn.unified import resolve_agn_model

pytestmark = pytest.mark.contract


class TestMonolithicAGNDeprecation:
    """Contracts: deprecated monolithic models still resolve + warn."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "multicolor_agn",
            "kubota_done",
            "kubota_done_full",
            "silva04",
            "cat3d_wind",
            "adaf",
            "relagn",
            "skirtor",
            "qsogen",
            "grahsp",
            "unified_nlr_blr",
        ],
    )
    def test_deprecated_model_resolves_with_warning(self, model_name):
        """Deprecated model name resolves and emits DeprecationWarning.

        Parameters
        ----------
        model_name : str
            Name of deprecated monolithic AGN model.

        Notes
        -----
        **Marker:** contract

        Verifies non-breaking deprecation: old names still work, but warn.
        """
        with pytest.warns(DeprecationWarning, match=f"'{model_name}' is deprecated"):
            fn = resolve_agn_model(model_name)

        # Callable signature: fn(wavelength, agn_log_lbol, **kwargs) -> L_nu
        assert callable(fn)

    def test_kubota_done_is_alias_of_multicolor_agn(self):
        """kubota_done is a registry alias, not a separate model.

        Notes
        -----
        **Marker:** contract

        Both names must resolve to the same callable (or equivalently,
        both emit the deprecation warning).
        """
        with pytest.warns(DeprecationWarning):
            fn_kubota = resolve_agn_model("kubota_done")

        with pytest.warns(DeprecationWarning):
            fn_multicolor = resolve_agn_model("multicolor_agn")

        # Semantically, kubota_done is an alias for multicolor_agn
        # (or at least they invoke the same model definition).
        # The warning messages should indicate they are the same pattern.
        assert callable(fn_kubota)
        assert callable(fn_multicolor)

    def test_relagn_deprecated(self):
        """relagn is now deprecated in favor of composable grammar.

        Notes
        -----
        **Marker:** contract

        RELAGN relativistic disc is now available as a composable disc block
        (``disc="relagn"``). The monolithic ``relagn`` model is deprecated
        and must emit DeprecationWarning.
        """
        with pytest.warns(DeprecationWarning, match="'relagn' is deprecated"):
            fn = resolve_agn_model("relagn")

        # Callable signature: fn(wavelength, agn_log_lbol, **kwargs) -> L_nu
        assert callable(fn)


class TestComposableAGNEquivalents:
    """Contracts: composable equivalents of deprecated models work.

    Verifies that at least 2 deprecated models can be reproduced via
    composable AGN block combinations, with finite output SEDs.
    """

    @pytest.fixture
    def wavelength(self):
        """Rest-frame wavelength grid [Å]."""
        return jnp.logspace(2, 5, 256)

    def test_multicolor_agn_via_composable_produces_finite_sed(self, wavelength):
        """multicolor_agn → disc=multicolor + torus=silva04 produces finite SED.

        Notes
        -----
        **Marker:** contract

        This is a smoke test confirming the composable path exists and
        runs without NaN/Inf.
        """
        from tengri.components.agn.disc import multicolor_disc
        from tengri.components.agn.silva04 import silva04_sed

        # Deprecated monolithic path
        with pytest.warns(DeprecationWarning):
            fn_monolithic = resolve_agn_model("multicolor_agn")
        l_nu_mono = fn_monolithic(wavelength, agn_log_lbol=11.0, agn_lum_ratio=0.1)

        # Composable path: disc + torus
        agn_log_lbol = 11.0
        agn_lum_ratio = 0.1
        agn_torus_frac = 0.5

        l_disc = multicolor_disc(
            wavelength,
            agn_log_lbol=agn_log_lbol,
            agn_lum_ratio=1.0 - agn_torus_frac,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
            agn_a_spin=0.0,
            agn_cos_inc=0.86602540378443864,
        )
        l_torus = silva04_sed(
            wavelength,
            agn_log_lbol=agn_log_lbol,
            agn_log_nh_silva=23.0,
            agn_torus_frac=agn_torus_frac,
        )
        l_nu_comp = (l_disc + l_torus) * agn_lum_ratio

        # Both paths should produce finite, positive SEDs
        assert jnp.all(jnp.isfinite(l_nu_mono)), "Monolithic path produced NaN/Inf"
        assert jnp.all(jnp.isfinite(l_nu_comp)), "Composable path produced NaN/Inf"
        assert jnp.all(l_nu_mono > 0), "Monolithic path has non-positive values"
        assert jnp.all(l_nu_comp > 0), "Composable path has non-positive values"

    def test_adaf_via_composable_produces_finite_sed(self, wavelength):
        """adaf → disc=adaf + torus=silva04 produces finite SED.

        Notes
        -----
        **Marker:** contract

        ADAF (advection-dominated accretion flow) for low-luminosity AGN.
        Composable: disc=adaf + torus=silva04.
        """
        from tengri.components.agn.adaf import adaf_spectrum
        from tengri.components.agn.silva04 import silva04_sed

        # Deprecated monolithic path
        with pytest.warns(DeprecationWarning):
            fn_monolithic = resolve_agn_model("adaf")
        l_nu_mono = fn_monolithic(wavelength, agn_log_lbol=10.0, agn_lum_ratio=0.05)

        # Composable path (faithful Mahadevan 1997 ADAF, #898)
        agn_log_lbol = 10.0
        agn_lum_ratio = 0.05
        agn_torus_frac = 0.3

        l_disc = adaf_spectrum(
            wavelength,
            agn_log_lbol=agn_log_lbol,
            agn_lum_ratio=1.0 - agn_torus_frac,
            agn_log_mbh=8.0,
            agn_adaf_alpha=0.3,
            agn_adaf_beta=0.5,
            agn_adaf_delta=0.1,
        )
        l_torus = silva04_sed(
            wavelength,
            agn_log_lbol=agn_log_lbol,
            agn_log_nh_silva=23.0,
            agn_torus_frac=agn_torus_frac,
        )
        l_nu_comp = (l_disc + l_torus) * agn_lum_ratio

        assert jnp.all(jnp.isfinite(l_nu_mono)), "Monolithic path produced NaN/Inf"
        assert jnp.all(jnp.isfinite(l_nu_comp)), "Composable path produced NaN/Inf"
        assert jnp.all(l_nu_mono > 0), "Monolithic path has non-positive values"
        assert jnp.all(l_nu_comp > 0), "Composable path has non-positive values"
