# SPDX-License-Identifier: BSD-3-Clause
r"""Contract: a ``Fixed`` parameter is refused, loudly, at every public entry point.

Omitting a fixed parameter is legal by design -- that is what fixing it *means*.
:meth:`SEDModel._get_redshift` falls back to the spec's value and
``get_internal_params`` / ``merge_fixed_params`` fill in ``{**fixed_values,
**params}``. So a params dict carrying only the free parameters is valid input
everywhere.

Before #2296, an *explicit* Fixed key in ``params`` was a silent physics bug
rather than a refusal: twice, a public surface read the redshift out of the
**dict** instead of the **state** and silently answered at :math:`d_L(0)` =
10 pc.

* #1097 shipped it in the exact projectors; #1124 fixed it at ``Prediction``.
* #1127 found it still live in ``SEDModel.measure_line_fluxes``, which never
  routes through ``Prediction`` -- 8.5e16 too bright, no warning.

Patching boundaries one at a time did not converge: ``params.get("redshift",
0.0)`` failed **open**, returning a physically meaningful value rather than
raising, so each new entry point re-introduced the bug and nothing complained.

#2296 closes the *class* structurally instead of patching instances: a params
dict key the spec declared Fixed is now refused outright, at every entry
point, regardless of whether the passed-in value happens to match the pinned
one. There is no longer a "silently disagrees" outcome to detect -- the two
arms below are "omit it" (legal, must work) and "name it explicitly" (refused,
must raise), and nothing in between. This test **auto-discovers** every public
``SEDModel`` method whose first argument is ``params``, and demands exactly
that split from each one. A new method that instead reads the dict and
disagrees silently, or reads it and doesn't raise, is caught the day it lands,
without anyone remembering to add it here.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.config.exceptions import ParameterError

pytestmark = pytest.mark.contract

Z_FIXED = 0.5
FREE_PARAMS = {"sfh_dpl_log_total_mass": jnp.asarray(10.0)}


def _build_model(ssp_data, obs, redshift):
    """Build the fixture model at a given Fixed redshift.

    Factored out so :func:`test_the_sweep_is_not_vacuous` can build a second,
    otherwise-identical model at a different Fixed redshift -- comparing two
    models is now the only way to prove the sweep can detect a dropped
    redshift, since passing an explicit override into ONE model is refused
    (#2296) and can no longer serve as the probe.
    """
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(9.0, 11.0)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(redshift),
    )


@pytest.fixture(scope="module")
def model(synthetic_ssp_wide, synthetic_tophat_obs):
    """Redshift FIXED, so a params dict legitimately omits it."""
    return _build_model(synthetic_ssp_wide, synthetic_tophat_obs, Z_FIXED)


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
# module exists to enforce -- and would have exempted any future entry point
# that started raising, including one that raises *because* it mishandles a
# Fixed key. The docstring above records that the redshift bug shipped twice
# and that patching boundaries one at a time does not converge; a
# skip-on-exception is how the seventh instance would get in unnoticed.
RAISES_ON_BARE_PARAMS = {
    "mock_spectrum": TypeError,
    "predict_emission_lines": NotImplementedError,
    "predict_line_fluxes": ValueError,
    "predict_spectral_indices": TypeError,
    "predict_spectrum": ValueError,
    "predict_spectrum_components": ValueError,
}


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_entry_point_refuses_an_explicit_fixed_redshift(model, name):
    """Omitting a Fixed redshift works; naming it explicitly is refused (#2296).

    Every value the caller could pass for a Fixed key is refused alike --
    including the pinned value itself. There is no "matches, so it's let
    through" carve-out: the rule is on key presence, not value comparison
    (:func:`tengri.parameters.resolve.refuse_fixed_overrides`).
    """
    if name in NEEDS_EXTRA_ARGS:
        pytest.skip(f"{name} needs arguments beyond params")

    fn = getattr(model, name)
    omitted = dict(FREE_PARAMS)
    explicit = {**FREE_PARAMS, "redshift": jnp.asarray(Z_FIXED)}

    try:
        fn(omitted)
    except Exception as exc:
        allowed = RAISES_ON_BARE_PARAMS.get(name)
        assert allowed is not None, (
            f"{name} raised {type(exc).__name__} on a bare params dict, so the "
            f"Fixed-redshift-refusal contract is unverified for it. If that call "
            f"really cannot take bare params, add it to RAISES_ON_BARE_PARAMS; do "
            f"not let it exempt itself by raising. ({exc})"
        )
        assert isinstance(exc, allowed), (
            f"{name} now raises {type(exc).__name__}, not the recorded "
            f"{allowed.__name__} -- the exemption no longer describes reality"
        )
        pytest.skip(f"{name}: listed in RAISES_ON_BARE_PARAMS ({allowed.__name__})")
    except (TypeError, ValueError) as exc:
        # TypeError and ValueError can be real defects. Only allow them if explicitly
        # listed in RAISES_ON_BARE_PARAMS — that's the legitimate signal for "not applicable".
        allowed = RAISES_ON_BARE_PARAMS.get(name)
        assert allowed is not None and isinstance(exc, allowed), (
            f"{name} raised {type(exc).__name__} on a bare params dict, which may be "
            f"a real defect in the entry point. If this is genuinely a 'not applicable' "
            f"case, add it to RAISES_ON_BARE_PARAMS; do not let arbitrary exceptions "
            f"exempt the surface from the fixed-redshift contract. ({exc})"
        )
        pytest.skip(f"{name}: listed in RAISES_ON_BARE_PARAMS ({allowed.__name__})")

    with pytest.raises(ParameterError) as exc_info:
        fn(explicit)
    msg = str(exc_info.value)
    assert "redshift" in msg, (
        f"{name} raised {type(exc_info.value).__name__} for the explicit-redshift "
        f"arm, but the message doesn't name 'redshift': {msg}"
    )


def test_the_resolver_cannot_fail_silently(model):
    """A guard against a silent failure must not itself be able to fail silently.

    ``resolve_fixed_params`` (via ``refuse_fixed_overrides``) reads
    ``spec.fixed_params`` directly, with no ``getattr`` default and no blanket
    ``except``. Rename the attribute upstream and the resolver would quietly
    become a no-op -- handing back an unresolved dict and bringing the 1e17
    error straight back with no error at all (#1127). It must raise instead.
    """
    from tengri.parameters.resolve import resolve_fixed_params

    class SpecWithoutFixedParams:
        """A spec that lost ``fixed_params`` -- e.g. to a later rename."""

        fixed_value = model.spec.fixed_value

    class ModelWithBrokenSpec:
        spec = SpecWithoutFixedParams()

    with pytest.raises(AttributeError):
        resolve_fixed_params(ModelWithBrokenSpec(), {})

    # And a model with no spec at all is a programming error, not a no-op.
    with pytest.raises(AttributeError):
        resolve_fixed_params(object(), {})


def test_the_resolver_injects_omitted_values_and_refuses_an_explicit_one(model):
    """The two halves of the contract, pinned.

    Before #2296 the second half read ``"an explicit value must win"`` --
    ``resolve_fixed_params`` let a caller-supplied Fixed key clobber the
    pinned value. That was the override-wins bug the owner ruling closes:
    presence of a Fixed key is refused now, unconditionally, not merged.
    """
    from tengri.parameters.resolve import resolve_fixed_params

    filled = resolve_fixed_params(model, dict(FREE_PARAMS))
    assert float(filled["redshift"]) == pytest.approx(Z_FIXED)

    with pytest.raises(ParameterError, match="redshift"):
        resolve_fixed_params(model, {**FREE_PARAMS, "redshift": jnp.asarray(2.0)})

    # Even the PINNED value, named explicitly, is refused -- this is a
    # key-presence rule, not a value comparison.
    with pytest.raises(ParameterError, match="redshift"):
        resolve_fixed_params(model, {**FREE_PARAMS, "redshift": jnp.asarray(Z_FIXED)})


def test_the_sweep_is_not_vacuous(model, synthetic_ssp_wide, synthetic_tophat_obs):
    """The sweep must actually cover the surfaces that broke, and have power.

    Two ways this suite could pass while proving nothing: it discovers no entry
    points, or it discovers only redshift-insensitive ones. Pin both.
    """
    assert len(ENTRY_POINTS) > 20, f"discovery found only {len(ENTRY_POINTS)} entry points"

    # The three surfaces that actually broke must be in the swept set.
    for known in ("predict_photometry", "measure_line_fluxes", "predict_magnitudes"):
        assert known in ENTRY_POINTS, f"{known} is no longer discovered -- the sweep has a hole"

    # And the redshift must genuinely move the number, or the refusal above is
    # satisfied vacuously by a model the redshift cannot affect. Compare TWO
    # otherwise-identical models at different Fixed redshifts -- passing an
    # explicit override into one model is refused (#2296) and can no longer
    # serve as this probe.
    model_at_zero = _build_model(synthetic_ssp_wide, synthetic_tophat_obs, 0.0)
    at_z = np.asarray(model.predict_photometry(dict(FREE_PARAMS)))
    at_zero = np.asarray(model_at_zero.predict_photometry(dict(FREE_PARAMS)))
    assert np.nanmax(np.abs(at_zero / at_z)) > 1e3, (
        "z=0 and z=0.5 give comparable fluxes on this model, so the sweep cannot "
        "detect a dropped redshift"
    )


def test_every_exemption_names_a_real_entry_point():
    """A stale name in either list exempts nothing and hides that it does."""
    for name in NEEDS_EXTRA_ARGS | set(RAISES_ON_BARE_PARAMS):
        assert name in ENTRY_POINTS, (
            f"{name} is exempted but no longer discovered -- drop it from the list"
        )


@pytest.mark.parametrize("name", sorted(RAISES_ON_BARE_PARAMS))
def test_the_bare_params_exemptions_are_still_needed(model, name):
    """The exemption list must shrink on purpose, not drift.

    If one of these starts accepting a bare params dict, it becomes coverable
    and should be covered -- this turns red so the name is removed rather than
    sitting there quietly exempting a surface that no longer needs it.
    """
    with pytest.raises(RAISES_ON_BARE_PARAMS[name]):
        getattr(model, name)(dict(FREE_PARAMS))
