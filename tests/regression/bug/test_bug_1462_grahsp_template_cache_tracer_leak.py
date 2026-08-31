# SPDX-License-Identifier: BSD-3-Clause
"""The GRAHSP template cache must not hold trace-scoped arrays (#1462).

``load_grahsp_templates`` is ``@lru_cache``-backed and builds its arrays with
``jnp.asarray``. When the first call happened *inside* a jit trace — which is
what ``predict_photometry`` does — the cache stored trace-scoped values, and
the next trace rejected them::

    m.predict_photometry(p)  # OK, populates the cache from inside the trace
    m.predict(p).photometry()  # UnexpectedTracerError

Those are the only two public prediction surfaces ``NAMING_CONTRACT.md`` §4b
defines, and using both on one model is the documented workflow.

Why it stayed hidden: the leak fires only when a model actually compiles its
own kernel. Before #1450/#1463 the AGN cache key omitted the block selectors,
so a second composable-AGN variant silently reused the first one's kernel,
never compiled, and never leaked — a wrong answer instead of a crash. Fixing
the key made every distinct AGN configuration compile fresh, which is why the
audit in #1462 said the two had to land together.

The fix caches **NumPy** arrays. A NumPy array cannot be a tracer under any
trace, so the cache is trace-independent by construction rather than by
discipline. Consumers already wrap with ``jnp.asarray`` at use, and the
dataclass docstrings already documented these as ``ndarray``.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def test_cached_template_arrays_are_numpy():
    """The structural guarantee: NumPy cannot be trace-scoped.

    Asserted on the type rather than by reproducing the leak, so the guard
    holds even if the surrounding jit structure changes.
    """
    from tengri.components.agn.grahsp.templates import load_grahsp_templates

    templates = load_grahsp_templates()
    checked = 0
    for field_name in (
        "feii_wave_nm",
        "feii_lumin",
        "line_wave_nm",
        "line_broad",
        "line_narrow_sy2",
        "line_narrow_liner",
        "torus_wave_nm",
    ):
        value = getattr(templates, field_name, None)
        if value is None:
            continue
        checked += 1
        assert isinstance(value, np.ndarray), (
            f"{field_name} is {type(value).__name__}, not numpy.ndarray — a jax "
            "array created inside a trace and cached here leaks into the next one"
        )
    assert checked >= 5, f"only checked {checked} fields — the guard is near-vacuous"


def test_both_public_predict_surfaces_work_on_one_model(synthetic_ssp_wide, synthetic_tophat_obs):
    """End-to-end: the documented two-surface workflow must not crash.

    The model deliberately uses ``blr='grahsp'``, the block whose template load
    populated the cache from inside the lean path's trace.
    """
    import warnings

    import jax

    from tengri import DEFAULT, Fixed, SEDModel
    from tengri.components.agn.grahsp.templates import load_grahsp_templates

    # Clear the cache first, or this test is a fair-weather guard. The leak
    # fires only when the cache is populated from INSIDE a trace; if any
    # earlier test in the process already loaded the templates eagerly, the
    # lean call below is a cache hit and nothing leaks. Verified: without this
    # line the test passes against the unfixed loader when it runs after its
    # sibling — order-dependent exactly the way the underlying bug is.
    load_grahsp_templates.cache_clear()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            neb={"type": "none"},
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw"},
                "blr": {"type": "grahsp"},
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(2.0),
        )
        params = {
            **model.spec.get_fixed_values(),
            **model.spec.sample(jax.random.PRNGKey(0)),
        }

        # Order matters: the lean path first, so it is the one that populates
        # the template cache from inside its trace. Reversing this hides the bug.
        lean = np.asarray(model.predict_photometry(params))
        rich = np.asarray(model.predict(params).photometry())

    assert np.all(np.isfinite(lean)), "lean path returned non-finite flux"
    assert np.all(np.isfinite(rich)), "rich path returned non-finite flux"
    # Agreement is the point of having two surfaces; a crash-only guard would
    # pass on a fix that made them disagree.
    np.testing.assert_allclose(rich, lean, rtol=1e-10)
