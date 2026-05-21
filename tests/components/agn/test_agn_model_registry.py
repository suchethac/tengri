"""Registry-wide tests for all registered AGN models.

Tests every model at default parameters to ensure physical amplitudes and linearity.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds


class TestAGNModelCombinations:
    """Every registered AGN model should produce a physical SED at log(L_bol/L_sun)=11."""

    # All non-template AGN models we expect to evaluate at default parameters
    _ALL_MODELS: ClassVar[list[str]] = [
        "adaf",
        "cat3d_wind",
        "kubota_done",
        "kubota_done_full",
        "multicolor_agn",
        "qsogen",
        "silva04",
        "simple",
        "skirtor",
        "standard",
        "unified_nlr_blr",
    ]

    @pytest.mark.parametrize("name", _ALL_MODELS)
    def test_agn_model_physical_amplitude(self, name):
        """L_ν max should land in 10^22–10^33 erg/s/Hz for log(L_bol/L_sun)=11."""
        from tengri.agn import AGN_MODELS, resolve_agn_model

        if name not in AGN_MODELS:
            pytest.skip(f"AGN model '{name}' not registered")
        fn = resolve_agn_model(name)
        wl = jnp.logspace(np.log10(100), np.log10(5e5), 1000)
        try:
            L = np.array(fn(wl, agn_log_lbol=11.0))
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{name} requires data files not present")
        L_pos = L[L > 0]
        assert len(L_pos) > 0, f"{name}: all-zero/negative output"
        assert 1e22 < L_pos.max() < 1e33, (
            f"{name}: max L_ν = {L_pos.max():.2e} erg/s/Hz (outside 1e22–1e33)"
        )
        assert np.all(np.isfinite(L)), f"{name}: NaN/Inf in output"

    @pytest.mark.parametrize("name", _ALL_MODELS)
    def test_agn_model_linear_in_Lbol(self, name):
        """Each AGN model must scale ~linearly in 10^L_bol (within ~15% — qsogen Baldwin)."""
        from tengri.agn import AGN_MODELS, resolve_agn_model

        if name not in AGN_MODELS:
            pytest.skip(f"AGN model '{name}' not registered")
        fn = resolve_agn_model(name)
        wl = jnp.logspace(np.log10(1e3), np.log10(1e4), 200)
        try:
            a = np.array(fn(wl, agn_log_lbol=10.0))
            b = np.array(fn(wl, agn_log_lbol=11.0))
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{name} requires data files not present")
        # qsogen has mild Baldwin effect; most models linear.
        if a.max() == 0.0:
            pytest.skip(f"{name} returned zeros for log_lbol=10")
        ratio = b.max() / a.max()
        assert 8.0 < ratio < 12.0, f"{name}: 10x L_bol → {ratio:.2f}x L_ν"
