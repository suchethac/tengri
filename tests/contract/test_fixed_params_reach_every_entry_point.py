# SPDX-License-Identifier: BSD-3-Clause
r"""Contract: a ``Fixed`` parameter must reach **every** public entry point.

Omitting a fixed parameter is legal by design — that is what fixing it *means*.
:meth:`SEDModel._get_redshift` falls back to the spec's value and
``get_internal_params`` merges ``{**fixed_values, **params}``. So a params dict
carrying only the free parameters is valid input everywhere.

Twice now, a public surface has read the redshift out of the **dict** instead of
the **state** and silently answered at :math:`d_L(0)` = 10 pc:

* #1097 shipped it in the exact projectors; #1124 fixed it at ``Prediction``.
* #1127 found it still live in ``SEDModel.measure_line_fluxes``, which never
  routes through ``Prediction`` — 8.5e16 too bright, no warning.

Patching boundaries one at a time does not converge: ``params.get("redshift",
0.0)`` fails **open**, returning a physically meaningful value rather than
raising, so each new entry point re-introduces the bug and nothing complains.

This test closes the *class* instead of the instance. It **auto-discovers** every
public ``SEDModel`` method whose first argument is ``params``, calls each one
twice — once with the redshift omitted (legal) and once with it passed
explicitly — and demands the same answer. A new method that reads the dict is
caught the day it lands, without anyone remembering to add it here.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform

pytestmark = pytest.mark.contract

Z_FIXED = 0.5
FREE_PARAMS = {"sfh_dpl_log_total_mass": jnp.asarray(10.0)}


@pytest.fixture(scope="module")
def model(synthetic_ssp_wide, synthetic_tophat_obs):
    """Redshift FIXED, so a params dict legitimately omits it."""
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(9.0, 11.0)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(Z_FIXED),
    )


def _params_entry_points():
    """Every public SEDModel method whose first argument is ``params``."""
    out = []
    for name, fn in inspect.getmembers(SEDModel, inspect.isfunction):
        if name.startswith("_"):
            continue
        args = list(inspect.signature(fn).parameters)
        if args[1:2] == ["params"]:
            out.append(name)
    return sorted(out)


ENTRY_POINTS = _params_entry_points()

# Methods that cannot be called with a bare params dict (they need a state, a
# line-ratio table, index definitions...). Listing them is a deliberate,
# reviewable exemption rather than a silent skip.
NEEDS_EXTRA_ARGS = {"predict_line_ratios", "predict_state"}

# Entry points that raise when handed a bare params dict *today*, with the
# exception each one actually raises. Same policy as NEEDS_EXTRA_ARGS: named
# and reviewable.
#
# This list replaces a bare ``except Exception: pytest.skip(...)`` around the
# call below, which silently exempted these six from the very contract the
# module exists to enforce — and would have exempted any future entry point
# that started raising, including one that raises *because* it mishandles the
# fixed redshift. The docstring above records that this bug shipped twice and
# that patching boundaries one at a time does not converge; a skip-on-exception
# is how the seventh instance would get in unnoticed.
RAISES_ON_BARE_PARAMS = {
    "mock_spectrum": TypeError,
    "predict_emission_lines": NotImplementedError,
    "predict_line_fluxes": ValueError,
    "predict_spectral_indices": TypeError,
    "predict_spectrum": ValueError,
    "predict_spectrum_components": ValueError,
}

# ``predict`` returns a lazy ``Prediction``, not numbers. Compare the observables
# it exposes instead — that is the surface the bug actually corrupted.
RETURNS_PREDICTION = {"predict"}


def _leaves(x, prefix=""):
    """Flatten any return shape to numeric leaves, dropping what cannot be a float."""
    if hasattr(x, "_asdict"):
        x = x._asdict()
    elif hasattr(x, "sed") and hasattr(x, "wavelength"):  # SEDResult
        x = {"sed": x.sed}

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(_leaves(v, f"{prefix}{k}."))
        return out
    if isinstance(x, (list, tuple)):
        out = {}
        for i, v in enumerate(x):
            out.update(_leaves(v, f"{prefix}{i}."))
        return out
    try:
        return {prefix.rstrip("."): np.asarray(x, dtype=float)}
    except (TypeError, ValueError):
        return {}  # not numeric (a string setting, a callable) — nothing to compare


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_entry_point_honors_a_fixed_redshift(model, name):
    """Omitting a Fixed redshift must give the same answer as passing it."""
    if name in NEEDS_EXTRA_ARGS:
        pytest.skip(f"{name} needs arguments beyond params")

    fn = getattr(model, name)
    if name in RETURNS_PREDICTION:  # compare what the Prediction exposes, not the object
        base = fn

        def fn(p):
            pred = base(p)
            return {"photometry": pred.photometry(), "magnitudes": pred.magnitudes()}

    omitted = dict(FREE_PARAMS)
    explicit = {**FREE_PARAMS, "redshift": jnp.asarray(Z_FIXED)}

    try:
        got = fn(omitted)
    except Exception as exc:
        allowed = RAISES_ON_BARE_PARAMS.get(name)
        assert allowed is not None, (
            f"{name} raised {type(exc).__name__} on a bare params dict, so the "
            f"fixed-redshift contract is unverified for it. If that call really "
            f"cannot take bare params, add it to RAISES_ON_BARE_PARAMS; do not "
            f"let it exempt itself by raising. ({exc})"
        )
        assert isinstance(exc, allowed), (
            f"{name} now raises {type(exc).__name__}, not the recorded "
            f"{allowed.__name__} — the exemption no longer describes reality"
        )
        pytest.skip(f"{name}: listed in RAISES_ON_BARE_PARAMS ({allowed.__name__})")

    expected = fn(explicit)

    got_leaves, exp_leaves = _leaves(got), _leaves(expected)

    # Some entry points (``mock``) echo the parameters they used, so the explicit
    # arm carries a ``redshift`` key the omitted arm does not. That difference is
    # expected — but *only* for fixed-parameter names. Anything else missing from
    # one arm is a real dropped output, so assert the difference is explainable
    # rather than quietly intersecting the keys.
    fixed_names = set(model.spec.fixed_params)
    diff = set(got_leaves) ^ set(exp_leaves)
    unexplained = {k for k in diff if k.split(".")[-1] not in fixed_names}
    assert not unexplained, f"{name} returned different keys per arm: {sorted(unexplained)}"

    for key in set(got_leaves) & set(exp_leaves):
        a, b = got_leaves[key], exp_leaves[key]
        both_nan = np.isnan(a) & np.isnan(b)
        np.testing.assert_allclose(
            np.where(both_nan, 0.0, a),
            np.where(both_nan, 0.0, b),
            rtol=1e-10,
            err_msg=(
                f"{name}{'[' + key + ']' if key else ''} disagrees when the Fixed "
                f"redshift is omitted vs passed explicitly. It is reading the "
                f"redshift out of the params dict instead of resolving it from "
                f"the spec — see #1127."
            ),
        )


def test_the_resolver_cannot_fail_silently(model):
    """A guard against a silent failure must not itself be able to fail silently.

    ``resolve_fixed_params`` used ``getattr(spec, "fixed_params", ())`` and a
    blanket ``except``. Rename the attribute upstream and the resolver quietly
    became a no-op — handing back an unresolved dict and bringing the 1e17 error
    straight back with no error at all (#1127). It must raise instead.
    """
    from tengri.parameters.resolve import resolve_fixed_params

    class SpecWithoutFixedParams:
        """A spec that lost ``fixed_params`` — e.g. to a later rename."""

        fixed_value = model.spec.fixed_value

    class ModelWithBrokenSpec:
        spec = SpecWithoutFixedParams()

    with pytest.raises(AttributeError):
        resolve_fixed_params(ModelWithBrokenSpec(), {})

    # And a model with no spec at all is a programming error, not a no-op.
    with pytest.raises(AttributeError):
        resolve_fixed_params(object(), {})


def test_the_resolver_injects_the_fixed_value_and_never_clobbers_the_user(model):
    """The two halves of the contract, pinned."""
    from tengri.parameters.resolve import resolve_fixed_params

    filled = resolve_fixed_params(model, dict(FREE_PARAMS))
    assert float(filled["redshift"]) == pytest.approx(Z_FIXED)

    override = resolve_fixed_params(model, {**FREE_PARAMS, "redshift": jnp.asarray(2.0)})
    assert float(override["redshift"]) == pytest.approx(2.0), "an explicit value must win"


def test_the_sweep_is_not_vacuous(model):
    """The sweep must actually cover the surfaces that broke, and have power.

    Two ways this suite could pass while proving nothing: it discovers no entry
    points, or it discovers only redshift-insensitive ones. Pin both.
    """
    assert len(ENTRY_POINTS) > 20, f"discovery found only {len(ENTRY_POINTS)} entry points"

    # The three surfaces that actually broke must be in the swept set.
    for known in ("predict_photometry", "measure_line_fluxes", "predict_magnitudes"):
        assert known in ENTRY_POINTS, f"{known} is no longer discovered — the sweep has a hole"

    # And the redshift must genuinely move the number, or the comparison above
    # is satisfied by any implementation at all.
    at_z = np.asarray(model.predict_photometry({**FREE_PARAMS, "redshift": jnp.asarray(Z_FIXED)}))
    at_zero = np.asarray(model.predict_photometry({**FREE_PARAMS, "redshift": jnp.asarray(0.0)}))
    assert np.nanmax(np.abs(at_zero / at_z)) > 1e3, (
        "z=0 and z=0.5 give comparable fluxes on this model, so the sweep cannot "
        "detect a dropped redshift"
    )


def test_every_exemption_names_a_real_entry_point():
    """A stale name in either list exempts nothing and hides that it does."""
    for name in NEEDS_EXTRA_ARGS | set(RAISES_ON_BARE_PARAMS):
        assert name in ENTRY_POINTS, (
            f"{name} is exempted but no longer discovered — drop it from the list"
        )


@pytest.mark.parametrize("name", sorted(RAISES_ON_BARE_PARAMS))
def test_the_bare_params_exemptions_are_still_needed(model, name):
    """The exemption list must shrink on purpose, not drift.

    If one of these starts accepting a bare params dict, it becomes coverable
    and should be covered — this turns red so the name is removed rather than
    sitting there quietly exempting a surface that no longer needs it.
    """
    with pytest.raises(RAISES_ON_BARE_PARAMS[name]):
        getattr(model, name)(dict(FREE_PARAMS))
