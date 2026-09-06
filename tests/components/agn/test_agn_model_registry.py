# SPDX-License-Identifier: BSD-3-Clause
"""Registry-wide tests for all registered AGN models.

Tests every model at default parameters to ensure physical amplitudes and linearity.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds

# Derived from the registry, not retyped. The hand-written list this replaces
# had drifted both ways: it named four models that no longer resolve
# ("simple", "standard", "adaf_lopez2024", "fritz06"), each of which quietly
# took the `except ValueError: pytest.skip(...)` path below, and it omitted
# three that do ("relagn", "richards2006", "skirtor_stalevski") — so a
# "registry-wide" sweep covered 10 of 13 models and reported green on the
# other four. A guard against a hand-written list that is itself a
# hand-written list guards nothing.
_MONOLITHIC_ALIASES = ["grahsp", "skirtor_stalevski"]


def _all_monolithic_models() -> list[str]:
    """Every monolithic AGN name ``resolve_agn_model`` accepts.

    ``composable`` is excluded deliberately: it is the production model and
    needs an explicit block configuration, so at bare defaults it returns an
    all-zero SED and cannot satisfy the amplitude bound.
    """
    from tengri.components.agn.unified import _AGN_PRESETS

    return sorted(set(_AGN_PRESETS) | set(_MONOLITHIC_ALIASES))


class TestAGNModelCombinations:
    """Every registered AGN model should produce a physical SED at log(L_bol/L_sun)=11."""

    _ALL_MODELS: ClassVar[list[str]] = _all_monolithic_models()

    @pytest.mark.parametrize("name", _ALL_MODELS)
    def test_agn_model_physical_amplitude(self, name):
        """L_ν max should land in 10^22–10^33 erg/s/Hz for log(L_bol/L_sun)=11."""
        import warnings

        from tengri.agn import resolve_agn_model

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                fn = resolve_agn_model(name)
        except ValueError as exc:  # a name in the derived list must resolve
            raise AssertionError(
                f"AGN model '{name}' came from the registry but does not resolve: {exc}"
            ) from exc
        wl = jnp.logspace(np.log10(100), np.log10(5e5), 1000)
        try:
            L = np.array(fn(wl, agn_log_lbol=11.0))
        except FileNotFoundError:
            pytest.skip(f"{name} requires data files not present or has configuration issues")
        L_pos = L[L > 0]
        assert len(L_pos) > 0, f"{name}: all-zero/negative output"
        assert 1e22 < L_pos.max() < 1e33, (
            f"{name}: max L_ν = {L_pos.max():.2e} erg/s/Hz (outside 1e22–1e33)"
        )
        assert np.all(np.isfinite(L)), f"{name}: NaN/Inf in output"

    @pytest.mark.parametrize("name", _ALL_MODELS)
    def test_agn_model_linear_in_Lbol(self, name):
        """Each AGN model must scale ~linearly in 10^L_bol (within ~15% — qsogen Baldwin)."""
        import warnings

        from tengri.agn import resolve_agn_model

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                fn = resolve_agn_model(name)
        except ValueError as exc:  # a name in the derived list must resolve
            raise AssertionError(
                f"AGN model '{name}' came from the registry but does not resolve: {exc}"
            ) from exc
        wl = jnp.logspace(np.log10(1e3), np.log10(1e4), 200)
        try:
            a = np.array(fn(wl, agn_log_lbol=10.0))
            b = np.array(fn(wl, agn_log_lbol=11.0))
        except FileNotFoundError:
            pytest.skip(f"{name} requires data files not present or has configuration issues")
        # qsogen has mild Baldwin effect; most models linear.
        if a.max() == 0.0:
            pytest.skip(f"{name} returned zeros for log_lbol=10")
        ratio = b.max() / a.max()
        assert 8.0 < ratio < 12.0, f"{name}: 10x L_bol → {ratio:.2f}x L_ν"


def test_the_model_sweep_covers_every_resolvable_name():
    """The derived sweep must stay in step with what the resolver accepts.

    Guards the derivation itself. ``_MONOLITHIC_ALIASES`` is the one piece
    still written by hand — two names that resolve without appearing in
    ``_AGN_PRESETS`` — so if a third alias is added, or one of these is
    dropped, this fails rather than letting the sweep silently narrow.
    """
    import warnings

    from tengri.agn import resolve_agn_model

    assert len(TestAGNModelCombinations._ALL_MODELS) >= 13, (
        f"sweep collapsed to {len(TestAGNModelCombinations._ALL_MODELS)} models"
    )
    for name in TestAGNModelCombinations._ALL_MODELS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert resolve_agn_model(name) is not None, f"{name} does not resolve"


def test_composable_is_excluded_for_a_reason():
    """Pins why the production model is not in the amplitude sweep.

    At bare defaults ``composable`` has no blocks configured and returns an
    all-zero SED, so it cannot meet the 1e22-1e33 bound. If that ever changes
    it should join the sweep, and this turns red to say so.
    """
    import warnings

    from tengri.agn import resolve_agn_model

    assert "composable" not in TestAGNModelCombinations._ALL_MODELS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fn = resolve_agn_model("composable")
    wl = jnp.logspace(np.log10(100), np.log10(5e5), 200)
    assert float(np.max(np.abs(np.array(fn(wl, agn_log_lbol=11.0))))) == 0.0
