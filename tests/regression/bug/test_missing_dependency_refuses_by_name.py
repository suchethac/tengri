# SPDX-License-Identifier: BSD-3-Clause
"""#2900: a missing optional dependency refuses by name instead of substituting a method.

A fallback that changes which algorithm runs is silent substitution: the user asked
for NUTS through BlackJAX and got the JIT CG sampler, or asked for a MAP warm start
and got random starts behind a ``RuntimeWarning``. The precedent is ``BackendError``,
which names the tier and the working alternatives. These tests pin the refusal for the
two optional dependencies whose fallbacks changed the algorithm: BlackJAX (the posterior
draw) and optax (the catalog MAP warm start).
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.config.exceptions import BackendError
from tengri.inference.catalog_fitter import CatalogFitter
from tengri.inference.fitter import Fitter
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug


def _tophat(center, frac=0.16, n=40):
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


@pytest.fixture(scope="module")
def fitter(synthetic_ssp_wide):
    obs = Observation(
        photometry=Photometry(filters=tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0)))
    )
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(8.0, 12.0)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )
    rng = np.random.default_rng(0)
    flux = np.abs(rng.normal(1.0, 0.1, size=3))
    f = Fitter(model, jnp.asarray(flux), jnp.asarray(0.1 * flux), data_type="photometry")
    return f, f._initialize_unbounded(jax.random.PRNGKey(0))


def test_missing_blackjax_refuses_and_names_the_jit_alternative(fitter, monkeypatch):
    """posterior_method='blackjax' without blackjax raises, naming posterior_method='jit'.

    Before #2900 the ImportError was caught and the JIT CG sampler ran instead, so the
    draws came from a different algorithm than the one requested.
    """
    f, init = fitter
    monkeypatch.setitem(sys.modules, "blackjax", None)

    with pytest.raises(BackendError) as exc:
        f._draw_posterior_samples(
            None, init, jax.random.PRNGKey(0), 8, [], method="blackjax", verbose=False
        )
    msg = str(exc.value)
    assert 'posterior_method="jit"' in msg, msg
    assert "pip install" in msg, "the refusal must carry the install hint"


def test_missing_optax_refuses_map_warm_start_and_names_init_from_none(monkeypatch):
    """init_from='map' without optax raises BackendError naming init_from=None.

    The warm start is computed by ``build_catalog_map_init``, which imports optax first,
    so a stub fitter is enough to reach the refusal without building a catalog.
    """
    monkeypatch.setitem(sys.modules, "optax", None)

    class _StubFitter:
        """Never read: optax is imported before the fitter is touched."""

    with pytest.raises(BackendError) as exc:
        CatalogFitter._map_warm_start(
            None,
            _StubFitter(),
            np.zeros((2, 3)),
            (),
            n_steps=1,
            chunk=1,
            thread_redshift=False,
            thread_line_fluxes=False,
            verbose=False,
        )
    msg = str(exc.value)
    assert "init_from=None" in msg, msg
    assert "pip install optax" in msg, msg
