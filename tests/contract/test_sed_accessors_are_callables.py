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
        dust={"law_diff": "calzetti", "type": "two_component", "law_bc": "calzetti", "*": FIXED},
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


# Sphinx cross-reference roles name the attribute, not a call: `:meth:`Prediction.rest_sed``
# is correct reStructuredText and must not be flagged.
_SPHINX_ROLE = re.compile(r":(meth|attr|func|obj|class):`")


def _code_lines(path: pathlib.Path):
    """Yield (lineno, line) for CODE only — prose about the misuse is not a misuse.

    In markdown only fenced *code* blocks count. A MyST directive
    (```` ```{note} ````) also opens with a fence but its body is prose, and the
    API guide's note deliberately writes ``pred.rest_sed`` without parens to
    explain why that raises. A guard that fails on the sentence explaining the
    rule is a guard people delete.
    """
    text = path.read_text().splitlines()
    if path.suffix == ".py":
        yield from ((i, ln) for i, ln in enumerate(text, 1) if not _SPHINX_ROLE.search(ln))
        return
    # Track "inside a fence" and "that fence is code" SEPARATELY. Collapsing them
    # into one flag inverts the state at the first MyST directive: ```{note} opens
    # without setting code-mode, and then its CLOSING ``` reads as an opening
    # fence, so every later prose line looks like code and every code block looks
    # like prose. (That bug made this guard silently vacuous once already.)
    in_fence = False
    fence_is_code = False
    for i, line in enumerate(text, 1):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            if in_fence:
                in_fence = fence_is_code = False
            else:
                in_fence = True
                # ```{note} / ```{warning} are MyST directives — prose, not code.
                fence_is_code = not stripped[3:].strip().startswith("{")
            continue
        if in_fence and fence_is_code:
            yield i, line


def test_no_call_site_uses_the_bare_property_form():
    """Static sweep: ``x.rest_sed`` without ``()`` is now always a bug.

    Catches the regression in code that CI never executes — the gallery skips
    any example that already has a committed figure (#1146), so a bare
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
            for i, line in _code_lines(path):
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


# ── units: the distance is applied at PROJECTION, not on the SED ──────────


def test_obs_sed_is_a_luminosity_not_a_flux(model, params, pred):
    """``obs_sed`` returns L_nu [erg/s/Hz]. "Observed" names the FRAME.

    The docstring claimed the opposite for a long time — "Returns F_nu, not
    L_nu ... accounts for the (1+z)/(4 pi d_L^2) cosmological dimming factor" —
    and that claim had already propagated into the naming contract. It is false:
    the dimming lives in the projection layer (``observation/redshift_kernel``),
    and ``obs_sed`` never applies it.

    This matters because the error is silent and enormous: a user who integrates
    ``obs_sed()`` as a flux is wrong by ``1/(4 pi d_L^2)`` — about 57 orders of
    magnitude at z ~ 0.3. Nothing would raise.
    """
    from tengri.utils.cosmology import luminosity_distance

    z = float(np.asarray(model._get_redshift(params)))
    assert z > 0.0, "fixture must be at non-zero redshift or this proves nothing"

    rest = np.asarray(pred.rest_sed())
    obs = np.asarray(pred.obs_sed())

    dl_cm = float(np.asarray(luminosity_distance(z)))
    dimming = (1.0 + z) / (4.0 * np.pi * dl_cm**2)  # ~1e-57 cm^-2
    assert dimming < 1e-50, "sanity: the dimming factor should be astronomically small"

    # Above rest-frame Lyman-alpha the IGM is transparent, so obs_sed must equal
    # rest_sed EXACTLY. If the dimming were applied, it would be ~1e-57x smaller.
    optical = np.asarray(pred.wave_rest) > 2000.0
    assert optical.sum() > 100, "need a decent optical baseline"
    assert np.allclose(obs[optical], rest[optical], rtol=1e-12), (
        "obs_sed differs from rest_sed above Lyman-alpha — is the cosmological "
        "dimming being applied? It must not be; obs_sed is L_nu."
    )
    # ...and explicitly: it is NOT the dimmed version.
    assert not np.allclose(obs[optical], rest[optical] * dimming, rtol=1e-3)


def test_photometry_is_a_flux_and_the_sed_is_not(pred):
    """Only the PROJECTION surfaces carry the 1/(4 pi d_L^2). Orders of magnitude."""
    lnu = float(np.median(np.abs(np.asarray(pred.rest_sed()))))
    fnu = float(np.median(np.abs(np.asarray(pred.photometry()))))
    # L_nu for a galaxy is ~1e28 erg/s/Hz; F_nu is ~1e-27 erg/s/cm^2/Hz.
    # The exact values depend on the fixture, but they cannot be the same scale.
    assert lnu > 1e10, f"rest_sed looks like a flux, not a luminosity: {lnu:.3e}"
    assert fnu < 1e-3, f"photometry looks like a luminosity, not a flux: {fnu:.3e}"
