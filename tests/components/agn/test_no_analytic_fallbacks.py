# SPDX-License-Identifier: BSD-3-Clause
"""Tests that dust and AGN emission models raise FileNotFoundError when templates
are missing, rather than silently falling back to wrong analytic approximations.

Every template-backed model must fail loudly if its data files are absent.
Silent wrong results are worse than visible errors.
"""

import pytest

pytestmark = pytest.mark.bounds
import tengri.components.agn.skirtor as _skirtor_mod

# The impl module (emission/emission.py), NOT the package: this test reaches into
# private lazy-loader internals (_load_dl14_fn, _dl07_lazy_wrapper, _make_lazy_loader,
# _resolved, _find_*) and monkeypatches them, which must target where the code runs.
import tengri.components.dust.emission.emission as _emission_mod


# ── Helpers: clear module-level caches between tests so each test starts fresh
def _reset_emission_caches():
    """Clear the lazy-loader resolution caches in emission.py."""
    _emission_mod._resolved.clear()
    _emission_mod._dl14_fn = None
    # Clear functools.cache on the on-demand loader so a subsequent monkeypatch
    # of _find_data_file actually takes effect.
    if hasattr(_emission_mod._load_dl14_fn, "cache_clear"):
        _emission_mod._load_dl14_fn.cache_clear()
    # Clear registry entries back to lazy wrappers
    _emission_mod.DUST_EMISSION_MODELS["draine_li2007"] = _emission_mod._dl07_lazy_wrapper
    _emission_mod.DUST_EMISSION_MODELS["dale2014"] = _emission_mod._make_lazy_loader(
        "dale2014",
        "dale2014_templates.h5",
        "create_dale2014_from_grid",
    )
    _emission_mod.DUST_EMISSION_MODELS["draine_li2014"] = _emission_mod._dl14_lazy_wrapper
    _emission_mod.DUST_EMISSION_MODELS["astrodust"] = _emission_mod._make_lazy_loader(
        "astrodust",
        "astrodust_templates.h5",
        "create_astrodust_from_grid",
    )


def _reset_skirtor_caches():
    """Clear skirtor.py's ``@functools.cache`` grid loaders + the legacy global.

    The #1198/#1199 refactor threads the SKIRTOR grid as arrays through JIT and
    memoizes the loader with ``@functools.cache`` (``_load_skirtor_default_grid``
    and friends). A prior test in the shard that loads the committed grid leaves
    that cache warm, so resetting only ``_skirtor_default`` is no longer enough:
    ``skirtor_analytic`` returns the cached grid and the ``Path.is_file`` patch
    never forces the missing-templates path. Clearing the functools caches (as
    :func:`_reset_emission_caches` already does for emission.py) restores the
    intended isolation. Repopulated lazily on the next real load, so no leakage.
    """
    _skirtor_mod._skirtor_default = None
    for _name in dir(_skirtor_mod):
        _obj = getattr(_skirtor_mod, _name)
        if callable(_obj) and hasattr(_obj, "cache_clear"):
            _obj.cache_clear()


def _reset_skirtor_cache():
    _skirtor_mod._skirtor_default = None
    # Clear functools.cache on default loaders so monkeypatched paths take effect.
    for attr in (
        "_load_skirtor_default",
        "_load_skirtor_default_grid",
        "_load_skirtor_components",
        "_load_raw_disk_dust_grid",
    ):
        fn = getattr(_skirtor_mod, attr, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()


@pytest.fixture(autouse=True)
def _reset_caches_between_tests():
    """Restore the emission-model registry and lazy-resolution caches before
    AND after each test so that monkeypatch-driven ``_find_*_templates = None``
    edits (either from this file or from upstream test modules that cached a
    FileNotFoundError wrapper) don't leak into the current test."""
    _reset_emission_caches()
    _reset_skirtor_cache()
    yield
    _reset_emission_caches()
    _reset_skirtor_cache()


# ── DL07 (Draine & Li 2007) ───────────────────────────────────────
class TestDL07NoFallback:
    def test_raises_when_templates_missing(self, monkeypatch):
        """draine_li2007 must raise FileNotFoundError, not silently return wrong SED."""
        monkeypatch.setattr(_emission_mod, "find_data_str", lambda *a, **k: None)
        _emission_mod._resolved.discard("draine_li2007")
        _emission_mod.DUST_EMISSION_MODELS["draine_li2007"] = _emission_mod._dl07_lazy_wrapper
        import jax.numpy as jnp

        wave = jnp.linspace(1e3, 1e7, 100)
        with pytest.raises(FileNotFoundError, match="DL07 template files"):
            _emission_mod.draine_li2007(wave, L_absorbed=1e10)

    def test_error_message_contains_script(self, monkeypatch):
        """Error message must tell the user how to fix it."""
        monkeypatch.setattr(_emission_mod, "find_data_str", lambda *a, **k: None)
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
        monkeypatch.setattr(_emission_mod, "find_data_str", lambda *a, **k: None)
        _emission_mod._resolved.discard("dale2014")
        _emission_mod.DUST_EMISSION_MODELS["dale2014"] = _emission_mod._make_lazy_loader(
            "dale2014",
            "dale2014_templates.h5",
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
        monkeypatch.setattr(_emission_mod, "find_data_str", lambda *a, **k: None)
        _emission_mod._dl14_fn = None
        _emission_mod._resolved.discard("draine_li2014")
        _emission_mod.DUST_EMISSION_MODELS["draine_li2014"] = _emission_mod._dl14_lazy_wrapper
        import jax.numpy as jnp

        wave = jnp.linspace(1e3, 1e7, 100)
        with pytest.raises(FileNotFoundError, match="DL14 template files"):
            _emission_mod.draine_li2014(wave, L_absorbed=1e10)

    def test_error_message_contains_script(self, monkeypatch):
        monkeypatch.setattr(_emission_mod, "find_data_str", lambda *a, **k: None)
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
        monkeypatch.setattr(_emission_mod, "find_data_str", lambda *a, **k: None)
        _emission_mod._resolved.discard("astrodust")
        _emission_mod.DUST_EMISSION_MODELS["astrodust"] = _emission_mod._make_lazy_loader(
            "astrodust",
            "astrodust_templates.h5",
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
        _reset_skirtor_caches()
        import jax.numpy as jnp

        wave = jnp.linspace(1e3, 1e6, 100)
        with pytest.raises(FileNotFoundError, match="SKIRTOR templates not found"):
            _skirtor_mod.skirtor_analytic(wave, L_agn=1e10)

    def test_error_message_contains_download_hint(self, monkeypatch):
        monkeypatch.setattr("pathlib.Path.is_file", lambda self: False)
        _reset_skirtor_caches()
        import jax.numpy as jnp

        wave = jnp.linspace(1e3, 1e6, 100)
        with pytest.raises(FileNotFoundError) as exc_info:
            _skirtor_mod.skirtor_analytic(wave, L_agn=1e10)
        assert "download_skirtor_templates" in str(exc_info.value)
