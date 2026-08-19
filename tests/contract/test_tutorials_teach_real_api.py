# SPDX-License-Identifier: BSD-3-Clause
"""The discovery menu must never teach an API that does not exist.

``tengri.tutorial()`` is the first thing a new user reads, and its code blocks
are plain strings — nothing executes them, so a typo or a stale method name
survives indefinitely and is copy-pasted straight into a user's script.

That is not hypothetical. Three defects motivated this guard, all found the day
it was written:

* ``key_classes`` went on teaching ``posterior.derived[...]`` for a full
  deprecation cycle after ``posterior.properties`` replaced it;
* a ``fast_vs_exact`` draft taught ``model.photometry.names()`` — wrong path
  *and* wrong call syntax (the real accessor is
  ``model.observation.photometry.names``);
* ``joint_phot_spec`` taught ``posterior.plot_spectrum_fit()``, which has never
  existed on :class:`Posterior`.

All three are the ``silent-failure`` class: the library is correct, the teaching
surface is wrong, and nothing in CI notices. This module parses every registered
tutorial's code block and resolves each ``receiver.attribute`` against a **live
instance** — not the class. Checking the class would miss every attribute
assigned in ``__init__`` (``model.observation`` is one), which produces false
positives and trains people to ignore the test.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import pytest

import tengri
from tengri import FIXED, Fixed, SEDModel, Uniform
from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.contract

# Public names that still exist but must never be taught: touching one hands the
# user a DeprecationWarning straight out of the tutorial they just copied.
_DEPRECATED = frozenset(
    {
        "derived",
        "predict_rest_sed",
        "predict_obs_sed",
        "predict_derived",
        "predict_magnitudes",
        "predict_sfh_quantities",
        "predict_sed_quantities",
    }
)

# Receiver names as they are spelled in the tutorial code blocks.
_RECEIVERS = ("model", "sed_model", "model_fast", "pred", "prediction", "posterior", "result")


def _taught(code: str) -> set[tuple[str, str]]:
    """Every ``receiver.attribute`` pair a tutorial's code block references."""
    return {
        (recv, attr)
        for recv in (*_RECEIVERS, "tengri")
        for attr in re.findall(rf"\b{re.escape(recv)}\.(\w+)", code)
    }


def _collect():
    from tengri._tutorials import _TUTORIALS

    return sorted(
        (name, recv, attr) for name, tut in _TUTORIALS.items() for recv, attr in _taught(tut.code)
    )


_ALL_TAUGHT = _collect()


@pytest.fixture(scope="module")
def surfaces(synthetic_ssp_wide, synthetic_tophat_obs):
    """Live instances of every class the tutorials teach against.

    Instances, not classes: ``model.observation`` and ``posterior.samples`` are
    set in ``__init__``, so a class-level ``hasattr`` would call them phantom.
    """
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "*": FIXED, "log_total_mass": Uniform(9.0, 11.0)},
        dust={"law_diff": "calzetti", "type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )
    params = {k: jnp.asarray(v) for k, v in model.spec.sample(jax.random.PRNGKey(0)).items()}
    pred = model.predict(params)
    posterior = Posterior(
        samples={k: jnp.repeat(v[None], 4, axis=0) for k, v in params.items()},
        params=params,
        method="vi",
        wall_time_s=1.0,
        diagnostics={},
        _model=model,
    )
    return {
        "model": model,
        "sed_model": model,
        "model_fast": model,
        "pred": pred,
        "prediction": pred,
        "posterior": posterior,
        "result": posterior,
        "tengri": tengri,
    }


@pytest.mark.parametrize("name,recv,attr", _ALL_TAUGHT, ids=lambda v: str(v))
def test_the_tutorial_teaches_an_attribute_that_exists(surfaces, name, recv, attr):
    """Every ``x.y`` a tutorial shows must resolve on a real object."""
    target = surfaces[recv]
    assert hasattr(target, attr), (
        f"tutorial {name!r} teaches {recv}.{attr}, which does not exist on "
        f"{type(target).__name__}. A user who copy-pastes this gets an AttributeError."
    )


@pytest.mark.parametrize("name,recv,attr", _ALL_TAUGHT, ids=lambda v: str(v))
def test_the_tutorial_does_not_teach_a_deprecated_name(name, recv, attr):
    """A tutorial must not hand the user a DeprecationWarning."""
    assert attr not in _DEPRECATED, (
        f"tutorial {name!r} teaches the deprecated {recv}.{attr}. Teach the replacement."
    )


def test_the_sweep_is_not_vacuous(surfaces):
    """Guard the guard: a regex that matches nothing would pass silently."""
    from tengri._tutorials import _TUTORIALS

    assert len(_TUTORIALS) >= 10, "tutorial registry unexpectedly small"
    assert len(_ALL_TAUGHT) >= 50, (
        f"only {len(_ALL_TAUGHT)} taught attributes discovered — the parser has "
        "probably stopped matching the tutorial code blocks, so these tests prove nothing."
    )
    # The parser must genuinely REJECT a phantom method, not merely accept
    # everything. This is the exact defect the module was written for.
    assert ("posterior", "plot_spectrum_fit") in _taught("posterior.plot_spectrum_fit()")
    assert not hasattr(surfaces["posterior"], "plot_spectrum_fit")
    # ...and genuinely ACCEPT an instance attribute a class-level check would miss.
    assert hasattr(surfaces["model"], "observation")
