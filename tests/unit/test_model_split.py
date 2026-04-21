"""Smoke tests for Scope C: model.py split into convenience.py."""


def test_convenience_module_importable():
    from tengri.forward.convenience import (
        fit_batch,
        fit_population,
        prior_predictive,
    )

    assert callable(prior_predictive)
    assert callable(fit_batch)
    assert callable(fit_population)


def test_model_still_has_prior_predictive():
    """SEDModel.prior_predictive must still work as a method after extraction."""
    import tengri

    assert hasattr(tengri.SEDModel, "prior_predictive")
    assert callable(tengri.SEDModel.prior_predictive)


def test_model_still_has_fit_batch():
    import tengri

    assert hasattr(tengri.SEDModel, "fit_batch")
    assert callable(tengri.SEDModel.fit_batch)


def test_model_still_has_fit_population():
    import tengri

    assert hasattr(tengri.SEDModel, "fit_population")
    assert callable(tengri.SEDModel.fit_population)
