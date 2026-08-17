# SPDX-License-Identifier: BSD-3-Clause
"""A saved fit reloaded without its model must be told how to recover (#1777).

``Posterior.save`` deliberately does not persist the model — it is a runtime
object, and the docstring says so. ``Posterior.load(path, model=...)`` takes it
back. Both halves of that design are right.

What was wrong is the error in between. A user who reloads a fit and reaches
for a derived property::

    post = Posterior.load("fit.h5")
    post.stellar_mass

got::

    No model reference on this Posterior — cannot compute properties.
    (It is populated automatically by Fitter.run().)

The only route it names is **re-running the fit**, which is the thing the file
was saved to avoid. The route that works — reloading with ``model=`` — went
unmentioned, and there is no public way to attach a model to a Posterior that
already exists, so a reader who follows the advice literally redoes the
expensive work for nothing.

Measured on a MAP fit: **54** public accessors that work on the original raise
on a model-less reload — ``stellar_mass``, ``sfr_100myr``,
``mass_weighted_age_gyr``, every emission line. That is the whole derived
surface, so this message is the one a user is most likely to meet.

The round-trip itself is a null result and is pinned below as one: loaded with
``model=``, every property matches the original exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import FIXED, Fitter, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.inference.posterior import Posterior

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


@pytest.fixture(scope="module")
def fitted(ssp_data_fsps, tmp_path_factory):
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))
    model = SEDModel.build(
        ssp_data=ssp_data_fsps,
        observation=obs,
        sfh={"type": "delayed", "all_params": FIXED, "log_total_mass": Uniform(9.0, 11.0)},
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "all_params": FIXED,
            "tau_diff": Uniform(0.0, 2.0),
        },
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )
    truth = dict(model.spec.get_fixed_values())
    truth["sfh_delayed_log_total_mass"] = 10.0
    truth["dust_tau_diff"] = 0.5
    flux = np.asarray(model.predict_photometry(truth))
    post = Fitter(model, flux, 0.05 * flux, data_type="photometry").run("map")
    path = tmp_path_factory.mktemp("post") / "fit.h5"
    post.save(str(path))
    return model, post, str(path)


class TestTheErrorNamesTheRouteThatWorks:
    def test_it_mentions_reloading_with_the_model(self, fitted):
        _, _, path = fitted
        bare = Posterior.load(path)
        with pytest.raises(RuntimeError) as exc:
            bare.properties["stellar_mass"]
        message = str(exc.value)
        assert "load" in message and "model=" in message, (
            f"the message names no working recovery route:\n  {message}"
        )

    def test_it_does_not_only_say_rerun_the_fit(self, fitted):
        """The original advice, on its own, costs the user the fit again."""
        _, _, path = fitted
        bare = Posterior.load(path)
        with pytest.raises(RuntimeError) as exc:
            bare.properties["stellar_mass"]
        message = str(exc.value)
        assert not ("Fitter.run()" in message and "load" not in message), (
            "Fitter.run() is named as the only route; reloading is the cheap one"
        )


class TestTheRoundTripItselfIsFine:
    """A null result, pinned so nobody re-investigates it.

    The save/load design is sound: what the file cannot carry is documented,
    and ``load`` takes it back as an argument.
    """

    @pytest.mark.parametrize(
        "prop", ["stellar_mass", "sfr_100myr", "mass_weighted_age_gyr", "l_bol"]
    )
    def test_loading_with_the_model_restores_the_property(self, fitted, prop):
        model, post, path = fitted
        back = Posterior.load(path, model=model)
        a = float(np.asarray(getattr(post, prop)))
        b = float(np.asarray(getattr(back, prop)))
        assert np.isclose(a, b, rtol=1e-12), f"{prop}: {a!r} -> {b!r}"

    def test_the_bare_load_still_returns_a_usable_posterior(self, fitted):
        """Not everything needs the model — params and diagnostics survive."""
        _, post, path = fitted
        bare = Posterior.load(path)
        assert bare.method == post.method
        assert set(bare.params) == set(post.params)
        for k, v in post.params.items():
            assert np.allclose(np.asarray(bare.params[k]), np.asarray(v), rtol=1e-12)
