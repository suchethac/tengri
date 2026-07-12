# SPDX-License-Identifier: BSD-3-Clause
"""``pred.rest_sed`` / ``pred.obs_sed`` are callables, and misuse is loud.

NAMING_CONTRACT §4b.3: every observable on :class:`Prediction` is a *uniform
callable with a default* — ``pred.rest_sed()`` for the model's own grid,
``pred.rest_sed(wave)`` to resample onto the caller's — matching
``photometry()``, ``magnitudes()`` and ``spectrum()``.

Two things this pins down, both of which have already gone wrong:

* **Resampling must not move a number.** The deprecated
  ``model.predict_rest_sed(params, wave=W)`` is exactly ``jnp.interp`` onto
  ``W``; the replacement must be bit-identical, or migrating a call site
  silently changes a published figure.
* **Forgetting the parentheses must raise.** ``pred.rest_sed`` is a method
  object; ``np.asarray`` of one yields a ``dtype=object`` array that matplotlib
  and numpy will happily turn into garbage instead of an error. That is the
  ``silent-failure`` class this package keeps shipping, so the accessor refuses
  every array protocol with an actionable ``TypeError``.
"""

from __future__ import annotations

import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel, Uniform

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def model(synthetic_ssp_wide, synthetic_tophat_obs):
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "*": FIXED, "log_total_mass": Uniform(9.0, 11.0)},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.3),
    )


@pytest.fixture(scope="module")
def params(model):
    return {k: jnp.asarray(v) for k, v in model.spec.sample(jax.random.PRNGKey(0)).items()}


@pytest.fixture(scope="module")
def pred(model, params):
    return model.predict(params)


# ── the callable contract ────────────────────────────────────────────────


def test_calling_with_no_argument_gives_the_model_grid(pred):
    assert np.asarray(pred.rest_sed()).shape == np.asarray(pred.wave_rest).shape
    assert np.asarray(pred.obs_sed()).shape == np.asarray(pred.wave_obs).shape


def test_calling_with_a_grid_resamples_onto_it(pred):
    wave = jnp.linspace(2000.0, 9000.0, 57)
    assert np.asarray(pred.rest_sed(wave)).shape == (57,)
    assert np.asarray(pred.obs_sed(wave)).shape == (57,)


def test_resampling_onto_its_own_axis_is_the_identity(pred):
    """A resample that moves a number when it shouldn't is the whole risk."""
    assert np.array_equal(np.asarray(pred.rest_sed(pred.wave_rest)), np.asarray(pred.rest_sed()))
    assert np.array_equal(np.asarray(pred.obs_sed(pred.wave_obs)), np.asarray(pred.obs_sed()))


def test_rest_sed_is_bit_exact_with_the_deprecated_wave_argument(model, params, pred):
    """Migrating ``predict_rest_sed(p, wave=W)`` must change no number.

    Not a tautology: the two paths are different code. The deprecated method
    interpolates ``state.sed_intrinsic`` onto ``W``; the accessor must do the
    same interpolation, from the same grid, in the same order.
    """
    wave = jnp.logspace(3.0, 4.5, 233)
    with pytest.warns(DeprecationWarning):
        old = np.asarray(model.predict_rest_sed(params, wave=wave).sed)
    new = np.asarray(pred.rest_sed(wave))
    assert np.array_equal(old, new), (
        f"resampling moved the SED: max rel diff {np.abs(old - new).max() / np.abs(old).max():.3e}"
    )


def test_obs_sed_takes_an_observed_frame_grid(model, params, pred):
    """§4b.3: the wavelength argument is in the accessor's OWN frame.

    The deprecated ``predict_obs_sed(params, wave=...)`` took a *rest*-frame
    grid and redshifted it — an observed-frame result with a rest-frame
    argument. That footgun is deliberately not reproduced, so ``obs_sed`` must
    be self-consistent with its own axis and NOT with the rest-frame one.
    """
    z = float(np.asarray(model._get_redshift(params)))
    assert z > 0.0, "fixture must be at non-zero redshift or this test proves nothing"
    # asking for the observed axis returns the observed SED unchanged...
    assert np.array_equal(np.asarray(pred.obs_sed(pred.wave_obs)), np.asarray(pred.obs_sed()))
    # ...and feeding it the REST axis does NOT (it would, if the frames were confused)
    assert not np.array_equal(np.asarray(pred.obs_sed(pred.wave_rest)), np.asarray(pred.obs_sed()))


# ── misuse must be loud ──────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["rest_sed", "obs_sed"])
def test_forgetting_the_parentheses_raises(pred, name):
    """The whole reason `_SEDCallable` exists.

    A plain bound method would coerce to a ``dtype=object`` array here and
    silently plot garbage. Every array protocol must refuse.
    """
    accessor = getattr(pred, name)
    with pytest.raises(TypeError, match="method, not an array"):
        np.asarray(accessor)
    with pytest.raises(TypeError, match="method, not an array"):
        _ = accessor * 2.0
    with pytest.raises(TypeError, match="method, not an array"):
        _ = accessor[0]
    with pytest.raises(TypeError, match="method, not an array"):
        _ = len(accessor)


def test_the_error_message_tells_you_the_fix(pred):
    with pytest.raises(TypeError) as exc:
        np.asarray(pred.rest_sed)
    msg = str(exc.value)
    assert "pred.rest_sed()" in msg, "the message must show the corrected call"
    assert "wave_rest" in msg, "the message must name the matching wavelength axis"


# ── nobody may reintroduce the property form ─────────────────────────────

_REPO = pathlib.Path(__file__).resolve().parents[2]
_NO_PARENS = re.compile(r"\.(rest_sed|obs_sed)\b(?!\s*\()")
_ALLOWED = {
    # the implementation and its own prose
    "src/tengri/forward/prediction.py",
    # this file talks about the misuse on purpose
    "tests/contract/test_sed_accessors_are_callables.py",
}


def test_no_call_site_uses_the_bare_property_form():
    """Static sweep: ``x.rest_sed`` without ``()`` is now always a bug.

    Catches the regression in code that is never executed by CI — the gallery
    skips any example that already has a committed figure, so a bare
    ``pred.rest_sed`` there would ship unnoticed.
    """
    offenders = []
    for root in ("src/tengri", "examples", "notebooks", "reproduction", "docs/api"):
        for path in sorted((_REPO / root).rglob("*")):
            if path.suffix not in {".py", ".md"}:
                continue
            rel = path.relative_to(_REPO).as_posix()
            if rel in _ALLOWED or "archive" in path.parts or "auto_examples" in path.parts:
                continue
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if _NO_PARENS.search(line) and "predict_rest_sed" not in line:
                    offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "bare .rest_sed / .obs_sed (missing call parens):\n" + "\n".join(
        offenders[:20]
    )


def test_the_static_sweep_is_not_vacuous():
    """Guard the guard: the regex must actually match the bad form."""
    assert _NO_PARENS.search("lnu = pred.rest_sed")
    assert _NO_PARENS.search("f = np.asarray(p.obs_sed)")
    assert not _NO_PARENS.search("lnu = pred.rest_sed()")
    assert not _NO_PARENS.search("lnu = pred.rest_sed(wave)")
    # and it must not fire on the wavelength axes, which ARE properties
    assert not _NO_PARENS.search("w = pred.wave_rest")
    assert not _NO_PARENS.search("w = pred.wave_obs")
