"""Smoke tests for Scope C: model.py split into convenience.py."""


def test_convenience_module_importable():
    from tengri.forward.convenience import (
        fit_catalog,
        fit_population,
        prior_predictive,
    )

    assert callable(prior_predictive)
    assert callable(fit_catalog)
    assert callable(fit_population)


def test_model_still_has_prior_predictive():
    """Model.prior_predictive must still work as a method after extraction."""
    import tengri

    assert hasattr(tengri.Model, "prior_predictive")
    assert callable(tengri.Model.prior_predictive)


def test_model_still_has_fit_catalog():
    import tengri

    assert hasattr(tengri.Model, "fit_catalog")
    assert callable(tengri.Model.fit_catalog)


def test_model_still_has_fit_population():
    import tengri

    assert hasattr(tengri.Model, "fit_population")
    assert callable(tengri.Model.fit_population)
