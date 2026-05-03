"""Test SEDModelConfig deprecation alias for ModelConfig."""

from __future__ import annotations

import warnings


def test_modelconfig_alias_warns():
    """ModelConfig alias should emit DeprecationWarning and return SEDModelConfig."""
    from tengri.config import settings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = settings.ModelConfig()

        # Check that a warning was raised
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "ModelConfig is deprecated" in str(w[0].message)
        assert "SEDModelConfig" in str(w[0].message)

        # Check that we got the right type
        assert isinstance(cfg, settings.SEDModelConfig)


def test_modelconfig_works_transparently():
    """ModelConfig can still be used despite deprecation."""
    from tengri.config import settings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg1 = settings.ModelConfig()
        cfg2 = settings.SEDModelConfig()

        # Both should be the same type and have identical defaults
        assert type(cfg1) is type(cfg2)
        assert cfg1.sfh.mean_type == cfg2.sfh.mean_type
        assert cfg1.dust.model == cfg2.dust.model
        assert cfg1.nebular.backend == cfg2.nebular.backend


def test_modelconfig_alias_from_import():
    """ModelConfig can still be imported with a warning."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Simulate import-time access via __getattr__
        from tengri.config.settings import ModelConfig

        # The import triggers __getattr__ when accessed, not on the import line itself.
        # Accessing the name in a function should trigger the warning.
        cfg = ModelConfig()

        assert len(w) >= 1
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
