# SPDX-License-Identifier: BSD-3-Clause
"""``init_from`` takes a mapping of starting values, not only a ``Posterior``.

Every other parameter surface in tengri (``predict``, ``predict_photometry``,
``spec.get_fixed_values``) accepts a plain ``dict[str, float]``. ``init_from``
forwarded straight to ``Fitter._unbounded_from_posterior``, which reads
``.params`` off whatever it is handed, so a dict died on
``AttributeError: 'dict' object has no attribute 'params'`` — a message about
the implementation rather than about the call (issue #1854).

The accept/refuse split follows what the conversion actually consumes: only
free names are read, and anything missing starts at the standardized ``0.0``.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    builders,
    generate_mock,
)
from tengri.config.exceptions import ParameterError

pytestmark = [pytest.mark.regression_bug, pytest.mark.slow]

FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
NUTS = dict(
    method="mcmc_nuts",
    n_warmup=60,
    n_samples=60,
    n_chains=2,
    n_burnin=0,
    dense_mass_matrix=False,
)


def _build(ssp_data):
    sed = SEDModel.build(
        ssp_data=ssp_data,
        observation=Observation(photometry=Photometry.from_names(FILTERS)),
        approx=WavePrecomp(),
        sfh=builders.sfh.tsnorm(all_params=FREE),
        dust_attenuation=builders.dust.two_component(
            all_params=Fixed(DEFAULT), law="calzetti", tau_bc=Uniform(0.0, 1.0)
        ),
        neb=builders.neb.none(),
        redshift=Fixed(0.05),
    )
    return sed, ForwardModel.build(sed=sed)


@pytest.fixture(scope="module")
def fixture(ssp_data_fsps):
    sed, forward = _build(ssp_data_fsps)
    k_truth, k_mock = jax.random.split(jax.random.PRNGKey(9))
    mock = generate_mock(sed, sed.spec.sample(k_truth), key=k_mock, snr=30.0)
    flux = np.asarray(mock["flux_obs"])
    noise = np.asarray(mock["noise"])
    point = forward.fit(
        flux,
        noise,
        method="map",
        key=jax.random.PRNGKey(3),
        n_restarts=4,
        n_steps=300,
        verbose=False,
    )
    return sed, flux, noise, point


def test_mapping_start_matches_the_posterior_it_came_from(ssp_data_fsps, fixture):
    """A dict of a MAP's params must seed exactly as that MAP object does."""
    _sed, flux, noise, point = fixture
    as_dict = {k: float(np.asarray(v)) for k, v in point.params.items()}
    key = jax.random.PRNGKey(3)

    _, a = _build(ssp_data_fsps)
    from_object = a.fit(flux, noise, key=key, init_from=point, **NUTS)
    _, b = _build(ssp_data_fsps)
    from_mapping = b.fit(flux, noise, key=key, init_from=as_dict, **NUTS)

    for name in sorted(from_object.samples):
        np.testing.assert_array_equal(
            np.asarray(from_object.samples[name]),
            np.asarray(from_mapping.samples[name]),
            err_msg=f"{name}: dict and Posterior seeds diverged",
        )


def test_fixed_parameters_in_the_mapping_are_accepted(ssp_data_fsps, fixture):
    """``dict(map_result.params)`` carries fixed values; that must not be refused."""
    sed, flux, noise, point = fixture
    as_dict = {k: float(np.asarray(v)) for k, v in point.params.items()}
    assert set(as_dict) - set(sed.spec.free_params), (
        "fixture no longer exercises fixed keys; the test would be vacuous"
    )

    _, forward = _build(ssp_data_fsps)
    forward.fit(flux, noise, key=jax.random.PRNGKey(3), init_from=as_dict, **NUTS)


def test_unknown_parameter_name_raises(ssp_data_fsps, fixture):
    """A misspelled key must not be silently ignored.

    Ignoring it would start that axis wherever the default lands while the fit
    reported nothing wrong — the shape of the ``n_subbnads`` sentinel defect.
    """
    sed, flux, noise, _point = fixture
    bad = {next(iter(sed.spec.free_params)): 0.5, "dust_tau_bcc": 0.3}

    _, forward = _build(ssp_data_fsps)
    with pytest.raises(ParameterError, match="does not have"):
        forward.fit(flux, noise, key=jax.random.PRNGKey(3), init_from=bad, **NUTS)


def test_mapping_naming_no_free_parameter_raises(ssp_data_fsps, fixture):
    """A start that cannot move any free axis is always a mistake."""
    _, flux, noise, _ = fixture
    _, forward = _build(ssp_data_fsps)
    with pytest.raises(ParameterError, match="no free parameter"):
        forward.fit(flux, noise, key=jax.random.PRNGKey(3), init_from={"redshift": 0.05}, **NUTS)


def test_non_mapping_non_posterior_raises_a_useful_message(ssp_data_fsps, fixture):
    """The old failure was ``AttributeError`` from two frames deep."""
    _, flux, noise, _ = fixture
    _, forward = _build(ssp_data_fsps)
    with pytest.raises(ParameterError, match="must be a mapping"):
        forward.fit(flux, noise, key=jax.random.PRNGKey(3), init_from=3.14, **NUTS)


def test_partial_mapping_warns_that_the_rest_start_at_the_prior_center(ssp_data_fsps, fixture):
    """Partial starts are legal but weak, and must say so.

    Unnamed free parameters start at the standardized 0.0. Measured on this
    fixture, a two-of-seven seed mixes to split R-hat ~1e15 while the fit
    otherwise looks healthy, so silence is the wrong default.
    """
    sed, flux, noise, point = fixture
    free = list(sed.spec.free_params)
    partial = {free[0]: float(np.asarray(point.params[free[0]]))}

    _, forward = _build(ssp_data_fsps)
    with pytest.warns(UserWarning, match="prior center"):
        forward.fit(flux, noise, key=jax.random.PRNGKey(3), init_from=partial, **NUTS)
