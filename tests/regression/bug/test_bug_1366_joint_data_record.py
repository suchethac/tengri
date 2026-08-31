# SPDX-License-Identifier: BSD-3-Clause
"""Regression: a *joint* ``Data`` record must fit, not raise (#1366, joint half).

#1366 taught ``SEDModel.fit`` to forward a ``Data`` record to ``ForwardModel.fit``,
and its stated motivation was that "everything ``Data`` carries -- censoring, line
fluxes, joint photometry+spectrum -- was unreachable from the entry point we
teach". The photometry-only record did become reachable. The joint one did not:
``ForwardModel.fit`` unpacked a two-channel record into

    kwargs.setdefault("photometry", (v.flux, v.noise))
    kwargs.setdefault("spectrum", (v.spec_flux, v.spec_noise))
    data, noise = None, None

but ``photometry=``/``spectrum=`` are ``SEDModel.fit`` parameter names, not
``Fitter`` ones. They fell through ``**kwargs`` to ``Fitter.run()`` while
``data=None`` tripped the constructor guard, so *every* joint record raised

    ValueError: Fitter(model, data, noise) requires data and noise for
                non-hierarchical fits. For hierarchical fits, ...

-- a message about hierarchical models, for a single galaxy with two channels.

The ``Fitter`` takes one concatenated vector plus ``data_type="joint"``
(photometry first, then spectrum: the order the joint likelihood splits on).
``data_type`` cannot ride ``**kwargs`` here because it is in
``_FIT_SURFACE_MANAGED``, so ``split_fitter_kwargs`` deliberately keeps it out of
``ctor_kwargs`` for the surface to set.

These tests pin *equivalence* with the explicit concatenated call, not merely that
the record stopped raising: accepting a joint record and then fitting only one of
its channels would satisfy a weaker assertion.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Data,
    Fixed,
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    Spectroscopy,
)
from tengri.inference.fitter import Fitter

pytestmark = pytest.mark.regression_bug

_BANDS = ["sdss_g", "sdss_r", "sdss_i", "sdss_z", "2mass_ks"]
_WAVE = np.linspace(4000.0, 8000.0, 64)


@pytest.fixture(scope="module")
def joint_model_and_mock(ssp_data_fsps):
    """A fixed-parameter joint model plus one photometry+spectrum realization."""
    obs = Observation(
        photometry=Photometry.from_names(_BANDS),
        spectroscopy=Spectroscopy(wave_obs=_WAVE),
    )
    sed = SEDModel.build(
        ssp_data=ssp_data_fsps,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        redshift=Fixed(0.05),
    )
    params = {**sed.spec.get_fixed_values(), **sed.spec.sample(jax.random.PRNGKey(0))}
    flux_p = np.asarray(sed.predict_photometry(params))
    flux_s = np.asarray(sed.predict_spectrum(params, wave_obs=_WAVE))
    err_p, err_s = flux_p / 20.0, flux_s / 30.0
    rng = np.random.default_rng(0)
    return (
        sed,
        flux_p + rng.normal(size=flux_p.shape) * err_p,
        err_p,
        flux_s + rng.normal(size=flux_s.shape) * err_s,
        err_s,
    )


def _map(target, data, **kw):
    return target.fit(
        data, method="map", key=jax.random.PRNGKey(2), n_steps=2, verbose=False, **kw
    )


class TestJointDataRecordFits:
    def test_forward_fit_accepts_a_joint_record(self, joint_model_and_mock):
        """LOAD-BEARING. Neuter: restore the ``kwargs.setdefault("photometry", ...)``
        form in ``ForwardModel.fit``'s joint branch.

        Without the fix this raises the "requires data and noise" ValueError.
        """
        sed, flux_p, err_p, flux_s, err_s = joint_model_and_mock
        record = Data(photometry=(flux_p, err_p), spectrum=(flux_s, err_s))
        result = _map(ForwardModel.build(sed=sed), record)
        assert result.params, "joint Data record returned no parameters"

    def test_matches_the_explicit_concatenated_fit(self, joint_model_and_mock):
        """Equivalence, not mere acceptance.

        The record must land on the same objective as spelling the joint fit out
        by hand. A branch that quietly dropped the spectrum would still return
        parameters, and only this comparison would catch it.
        """
        sed, flux_p, err_p, flux_s, err_s = joint_model_and_mock
        forward = ForwardModel.build(sed=sed)

        via_record = _map(forward, Data(photometry=(flux_p, err_p), spectrum=(flux_s, err_s)))
        explicit = Fitter(
            forward,
            np.concatenate([flux_p, flux_s]),
            np.concatenate([err_p, err_s]),
            data_type="joint",
        ).run("map", key=jax.random.PRNGKey(2), n_steps=2, verbose=False)

        assert set(via_record.params) == set(explicit.params)
        for name, value in via_record.params.items():
            np.testing.assert_allclose(
                np.asarray(value),
                np.asarray(explicit.params[name]),
                rtol=1e-10,
                err_msg=f"joint Data record and explicit joint fit disagree on '{name}'",
            )
        # The objective itself, not just where the optimizer stopped: a few ADAM
        # steps move the parameters very little, so the loss is what actually
        # senses which data went in (see the order test below).
        np.testing.assert_allclose(
            float(via_record.diagnostics["final_loss"]),
            float(explicit.diagnostics["final_loss"]),
            rtol=1e-12,
            err_msg="joint Data record and explicit joint fit have different objectives",
        )

    def test_channel_order_is_photometry_then_spectrum(self, joint_model_and_mock):
        """The concatenation order is a contract, not an implementation detail.

        Reversing it would still fit and still return parameters -- the joint
        likelihood would simply compare photometry against spectral pixels. This
        pins the order by showing the reversed concatenation gives a *different*
        answer, so the assertion above is not order-blind.
        """
        sed, flux_p, err_p, flux_s, err_s = joint_model_and_mock
        forward = ForwardModel.build(sed=sed)

        via_record = _map(forward, Data(photometry=(flux_p, err_p), spectrum=(flux_s, err_s)))
        reversed_fit = Fitter(
            forward,
            np.concatenate([flux_s, flux_p]),
            np.concatenate([err_s, err_p]),
            data_type="joint",
        ).run("map", key=jax.random.PRNGKey(2), n_steps=2, verbose=False)

        assert not np.isclose(
            float(via_record.diagnostics["final_loss"]),
            float(reversed_fit.diagnostics["final_loss"]),
            rtol=1e-6,
        ), (
            "reversing the channel order left the objective unchanged -- the "
            "equivalence test above cannot detect a mis-ordered concatenation"
        )

    def test_sugar_agrees_with_the_canonical_surface(self, joint_model_and_mock):
        """``sed.fit`` forwards joint records to ``ForwardModel.fit`` unchanged."""
        sed, flux_p, err_p, flux_s, err_s = joint_model_and_mock
        record = Data(photometry=(flux_p, err_p), spectrum=(flux_s, err_s))

        via_sugar = _map(sed, record)
        via_canonical = _map(ForwardModel.build(sed=sed), record)

        for name, value in via_sugar.params.items():
            np.testing.assert_allclose(
                np.asarray(value),
                np.asarray(via_canonical.params[name]),
                rtol=1e-10,
                err_msg=f"sed.fit and ForwardModel.fit disagree on '{name}'",
            )
