"""Tests that dust and AGN emission models raise FileNotFoundError when templates
are missing, rather than silently falling back to wrong analytic approximations.

Every template-backed model must fail loudly if its data files are absent.
Silent wrong results are worse than visible errors.
"""

import pytest

import tengri.components.agn.skirtor as _skirtor_mod
import tengri.components.dust.emission as _emission_mod

# ── Helpers: clear module-level caches between tests so each test starts fresh


def _reset_emission_caches():
    """Clear the lazy-loader resolution caches in emission.py."""
    _emission_mod._resolved.clear()
    _emission_mod._dl14_fn = None
    # Clear registry entries back to lazy wrappers
    _emission_mod.DUST_EMISSION_MODELS["draine_li2007"] = _emission_mod._dl07_lazy_wrapper
    _emission_mod.DUST_EMISSION_MODELS["dale2014"] = _emission_mod._make_lazy_loader(
        "dale2014",
        "dale2014_templates.npz",
        "create_dale2014_from_grid",
    )
    _emission_mod.DUST_EMISSION_MODELS["draine_li2014"] = _emission_mod._dl14_lazy_wrapper
    _emission_mod.DUST_EMISSION_MODELS["astrodust"] = _emission_mod._make_lazy_loader(
        "astrodust",
        "astrodust_templates.npz",
        "create_astrodust_from_grid",
    )


def _reset_skirtor_cache():
    _skirtor_mod._skirtor_default = None


# ── DL07 (Draine & Li 2007) ───────────────────────────────────────


class TestDL07NoFallback:
    def test_raises_when_templates_missing(self, monkeypatch):
        """draine_li2007 must raise FileNotFoundError, not silently return wrong SED."""
        monkeypatch.setattr(_emission_mod, "_find_dl07_templates", lambda: None)
        _emission_mod._resolved.discard("draine_li2007")
        _emission_mod.DUST_EMISSION_MODELS["draine_li2007"] = _emission_mod._dl07_lazy_wrapper

        import jax.numpy as jnp

        wave = jnp.linspace(1e3, 1e7, 100)
        with pytest.raises(FileNotFoundError, match="DL07 template files"):
            _emission_mod.draine_li2007(wave, L_absorbed=1e10)

    def test_error_message_contains_script(self, monkeypatch):
        """Error message must tell the user how to fix it."""
        monkeypatch.setattr(_emission_mod, "_find_dl07_templates", lambda: None)
        _emission_mod._resolved.discard("draine_li2007")
        _emission_mod.DUST_EMISSION_MODELS["draine_li2007"] = _emission_mod._dl07_lazy_wrapper

        import jax.numpy as jnp

        wave = jnp.linspace(1e3, 1e7, 100)
        with pytest.raises(FileNotFoundError) as exc_info:
            _emission_mod.draine_li2007(wave, L_absorbed=1e10)
        assert "convert_dl07_templates" in str(exc_info.value)


# ── Dale+2014 ─────────────────────────────────────────────────────


class TestDale2014NoFallback:
    def test_raises_when_templates_missing(self, monkeypatch):
        """dale2014 must raise FileNotFoundError, not silently return wrong SED."""
        monkeypatch.setattr(_emission_mod, "_find_data_file", lambda *a, **k: None)
        _emission_mod._resolved.discard("dale2014")
        _emission_mod.DUST_EMISSION_MODELS["dale2014"] = _emission_mod._make_lazy_loader(
            "dale2014",
            "dale2014_templates.npz",
            "create_dale2014_from_grid",
        )

        import jax.numpy as jnp

        wave = jnp.linspace(1e3, 1e7, 100)
        with pytest.raises(FileNotFoundError, match="dale2014"):
            _emission_mod.dale2014(wave, L_absorbed=1e10)


# ── DL14 (Draine & Li 2014) ───────────────────────────────────────


class TestDL14NoFallback:
    def test_raises_when_templates_missing(self, monkeypatch):
        """draine_li2014 must raise FileNotFoundError, not silently return wrong SED."""
        monkeypatch.setattr(_emission_mod, "_find_data_file", lambda *a, **k: None)
        _emission_mod._dl14_fn = None
        _emission_mod._resolved.discard("draine_li2014")
        _emission_mod.DUST_EMISSION_MODELS["draine_li2014"] = _emission_mod._dl14_lazy_wrapper

        import jax.numpy as jnp

        wave = jnp.linspace(1e3, 1e7, 100)
        with pytest.raises(FileNotFoundError, match="DL14 template files"):
            _emission_mod.draine_li2014(wave, L_absorbed=1e10)

    def test_error_message_contains_script(self, monkeypatch):
        monkeypatch.setattr(_emission_mod, "_find_data_file", lambda *a, **k: None)
        _emission_mod._dl14_fn = None
        _emission_mod._resolved.discard("draine_li2014")
        _emission_mod.DUST_EMISSION_MODELS["draine_li2014"] = _emission_mod._dl14_lazy_wrapper

        import jax.numpy as jnp

        wave = jnp.linspace(1e3, 1e7, 100)
        with pytest.raises(FileNotFoundError) as exc_info:
            _emission_mod.draine_li2014(wave, L_absorbed=1e10)
        assert "download_dl14_templates" in str(exc_info.value)


# ── Astrodust+PAH (Hensley & Draine 2023) ─────────────────────────


class TestAstrodustNoFallback:
    def test_raises_when_templates_missing(self, monkeypatch):
        """astrodust must raise FileNotFoundError, not silently return wrong SED."""
        monkeypatch.setattr(_emission_mod, "_find_data_file", lambda *a, **k: None)
        _emission_mod._resolved.discard("astrodust")
        _emission_mod.DUST_EMISSION_MODELS["astrodust"] = _emission_mod._make_lazy_loader(
            "astrodust",
            "astrodust_templates.npz",
            "create_astrodust_from_grid",
        )

        import jax.numpy as jnp

        wave = jnp.linspace(1e3, 1e7, 100)
        with pytest.raises(FileNotFoundError, match="astrodust"):
            _emission_mod.astrodust(wave, L_absorbed=1e10)


# ── SKIRTOR torus ─────────────────────────────────────────────────


class TestSKIRTORNoFallback:
    def test_raises_when_templates_missing(self, monkeypatch):
        """skirtor_analytic must raise FileNotFoundError when templates are missing."""
        # Patch Path.is_file to always return False so no candidate is found
        monkeypatch.setattr("pathlib.Path.is_file", lambda self: False)
        _skirtor_mod._skirtor_default = None

        import jax.numpy as jnp

        wave = jnp.linspace(1e3, 1e6, 100)
        with pytest.raises(FileNotFoundError, match="SKIRTOR templates not found"):
            _skirtor_mod.skirtor_analytic(wave, L_agn=1e10)

    def test_error_message_contains_download_hint(self, monkeypatch):
        monkeypatch.setattr("pathlib.Path.is_file", lambda self: False)
        _skirtor_mod._skirtor_default = None

        import jax.numpy as jnp

        wave = jnp.linspace(1e3, 1e6, 100)
        with pytest.raises(FileNotFoundError) as exc_info:
            _skirtor_mod.skirtor_analytic(wave, L_agn=1e10)
        assert "download_skirtor_templates" in str(exc_info.value)
