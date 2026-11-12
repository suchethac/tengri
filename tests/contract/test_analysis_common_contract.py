# SPDX-License-Identifier: BSD-3-Clause
"""analysis/common.py builds an SEDModel and its SFH metric runs (#3351).

``analysis/common.py`` imported ``Model`` and ``ParamSpec``, which tengri no
longer exports, so every figure script that imported it failed at import. This
pins the port: ``make_model`` returns an ``SEDModel`` and ``posterior_sfr_full``
evaluates three posterior draws of its SFH.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.contract


def test_make_model_returns_sedmodel_and_sfr_metric_runs():
    from analysis.common import make_model, posterior_sfr_full

    from tengri import SEDModel

    model = make_model("moderate", redshift=0.1, stochastic=True, n_grid=32)
    assert isinstance(model, SEDModel)

    draws = [model.spec.sample(k) for k in jax.random.split(jax.random.PRNGKey(0), 3)]
    samples = {name: np.stack([np.asarray(d[name]) for d in draws]) for name in draws[0]}

    sfr = posterior_sfr_full(model, samples, n_draw=3, chunk_size=2)
    assert sfr.shape == (3, 1000)
    assert np.all(np.isfinite(sfr))
    assert np.all(sfr >= 0.0)
