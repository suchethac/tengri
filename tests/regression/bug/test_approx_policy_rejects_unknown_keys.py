# SPDX-License-Identifier: BSD-3-Clause
"""A mistyped approximation knob must fail, not silently pick a worse scheme.

The resolved ``approx`` state was a bare ``dict`` read at 43 sites across 9
modules, almost always as ``approx.get("some_key", <default>)``. On a plain
dict that call cannot fail: a typo, a renamed key, or a key a caller forgot
to populate all return the default instead.

That is not a cosmetic risk here, because of what the defaults mean. The
default for ``n_subbands`` at those call sites is ``0`` — the sentinel that
*disables* the dust quadrature. So ``approx.get("n_subbnads", 0)`` silently
turns off the accurate band-integration scheme and falls back to the
effective-wavelength form, which measures up to 42 % wrong in the rest-UV at
z = 1. The failure is invisible: no exception, no warning, just different
numbers.

This exact shape has already shipped once. Before the ``or WavePrecomp()``
fallback existed, ``approx=SpectrumPrecomp()`` on a joint observation reached
the projector with a live photometry LUT and no ``n_subbands`` field; the
``getattr(cfg, "n_subbands", 0)`` beside it returned the disabling sentinel
and the photometry quietly ran the pre-#1122 path.

:class:`ApproxPolicy` closes the class off. It is a frozen dataclass, so
every knob is a typed attribute; it is also a ``Mapping``, so the existing
``[...]`` and ``.get(...)`` call sites keep working — but both validate the
key against the field set and raise on anything unknown. There is no spelling
of "read a key that does not exist" that returns a default any more.
"""

from __future__ import annotations

import dataclasses

import pytest

from tengri.forward.approx_policy import ApproxPolicy

pytestmark = pytest.mark.regression_bug


# ── the defect: a typo used to return a (dangerous) default ──────────────


@pytest.mark.parametrize("typo", ["n_subbnads", "wave_precmop", "tayler_correction", "nsubbands"])
def test_get_raises_on_an_unknown_key_instead_of_returning_the_default(typo):
    """``.get(typo, 0)`` must not hand back ``0``.

    ``0`` is the sentinel that disables the quadrature, so the plain-dict
    behavior converts a typo directly into a silent accuracy regression.
    """
    policy = ApproxPolicy()
    with pytest.raises(KeyError, match=typo):
        policy.get(typo, 0)


@pytest.mark.parametrize("typo", ["n_subbnads", "igmm", ""])
def test_getitem_raises_on_an_unknown_key(typo):
    with pytest.raises(KeyError, match="ApproxPolicy|" + (typo or "empty")):
        ApproxPolicy()[typo]


def test_the_error_names_the_valid_keys():
    """A raise that does not say what IS valid just moves the guesswork."""
    with pytest.raises(KeyError) as exc:
        ApproxPolicy().get("n_subbnads", 0)
    message = str(exc.value)
    assert "n_subbands" in message, "the error should surface the near-miss it did not match"


# ── known keys keep working, through both spellings ──────────────────────


def test_known_keys_read_identically_through_all_three_spellings():
    policy = ApproxPolicy(n_subbands=8, band_integration="quadrature")
    assert policy.n_subbands == 8
    assert policy["n_subbands"] == 8
    assert policy.get("n_subbands") == 8


def test_get_default_is_never_consulted_for_a_known_key():
    """Every field always has a value, so a default would only mask bugs."""
    policy = ApproxPolicy(n_subbands=8)
    assert policy.get("n_subbands", 999) == 8
    assert policy.get("wave_precomp", "sentinel") is False


def test_it_is_a_mapping_so_existing_call_sites_are_untouched():
    from collections.abc import Mapping

    policy = ApproxPolicy()
    assert isinstance(policy, Mapping)
    assert dict(policy)["band_integration"] == "quadrature"
    assert set(policy) == {f.name for f in dataclasses.fields(ApproxPolicy)}
    assert len(policy) == len(dataclasses.fields(ApproxPolicy))


# ── immutability, per the project's immutability rule ────────────────────


def test_it_is_frozen():
    policy = ApproxPolicy()
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.n_subbands = 3


def test_item_assignment_is_refused():
    """The dict it replaces supported ``policy[k] = v``; this must not."""
    policy = ApproxPolicy()
    with pytest.raises(TypeError):
        policy["n_subbands"] = 3


def test_replace_returns_a_new_policy_and_validates_the_field():
    policy = ApproxPolicy()
    updated = policy.replace(wave_precomp=True)
    assert updated.wave_precomp is True
    assert policy.wave_precomp is False, "replace must not mutate the original"
    with pytest.raises(TypeError):
        policy.replace(wave_precmop=True)


# ── the defaults still mean the accurate scheme ──────────────────────────


def test_defaults_select_the_accurate_band_integration():
    """Pinned here as well as at the WavePrecomp end.

    ApproxPolicy is now the single definition of "no preference expressed",
    so a change here silently moves every such model onto another scheme.
    """
    policy = ApproxPolicy()
    assert policy.band_integration == "quadrature"
    assert policy.n_subbands >= 1
    assert policy.taylor_correction is False


def test_sedmodel_default_approx_is_a_policy_not_a_dict():
    """The whole point: no bare dict survives on the resolved state."""
    from tengri import SEDModel

    assert isinstance(SEDModel._DEFAULT_APPROX, ApproxPolicy)


def test_policy_is_hashable_despite_being_a_mapping():
    """``collections.abc.Mapping`` sets ``__hash__ = None``.

    Only the ``@dataclass(frozen=True)`` decorator regenerating ``__hash__``
    on the subclass saves this — subtle enough that a future refactor (say,
    switching to ``eq=False``, or hand-writing ``__eq__``) could silently make
    the policy unhashable. It feeds ``compile_signature``, which is a cache
    key, so that failure would surface far from its cause.
    """
    from collections.abc import Mapping

    assert Mapping.__hash__ is None, "premise changed; this test guards the override"
    assert hash(ApproxPolicy()) == hash(ApproxPolicy())
    assert hash(ApproxPolicy(n_subbands=8)) != hash(ApproxPolicy(n_subbands=5))


def test_every_policy_field_reaches_the_compile_signature():
    """A knob that does not key the cache lets two models share one kernel.

    ``compile_signature`` collected bools generically but singled out
    ``n_subbands`` by name, so ``band_integration`` — a string — was captured
    by neither. It distinguished kernels only *incidentally*, because
    resolving it happens to write n_subbands and taylor_correction to
    per-scheme values. This asserts the property directly, so a future field
    that leaves those two alone cannot silently collide.
    """
    import dataclasses

    import tengri
    from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, WavePrecomp

    # Committed grid, resolved without a working directory (#1486). This test
    # compares compile signatures, so the grid is immaterial to what it pins.
    ssp = tengri.load_ssp()
    obs = Observation(photometry=Photometry.from_names(["galex_fuv", "sdss_r"]))
    common = dict(
        sfh={"type": "dpl", "all_params": FIXED, "log_total_mass": 10.0},
        dust={"law_diff": 'calzetti', "type": "two_component", "law_bc": "calzetti", "all_params": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.05),
    )

    def sig(**precomp):
        model = SEDModel.build(
            ssp_data=ssp, observation=obs, approx=WavePrecomp(**precomp), **common
        )
        return model.compile_signature()

    schemes = ["quadrature", "taylor", "effective_wavelength"]
    sigs = {s: sig(band_integration=s) for s in schemes}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert sigs[schemes[a]] != sigs[schemes[b]], (
            f"{schemes[a]} and {schemes[b]} share a compile signature — the "
            "second silently reuses the first's compiled kernel"
        )

    # The node count must key it too (#1122).
    assert sig(n_subbands=3) != sig(n_subbands=8)

    # And the guard against the next field: every non-bool policy field must
    # be representable in the signature's scalar capture.
    for field in dataclasses.fields(ApproxPolicy):
        value = getattr(ApproxPolicy(), field.name)
        assert isinstance(value, (str, int, float, bool, type(None))), (
            f"{field.name} is a {type(value).__name__}, which compile_signature's "
            "scalar capture will drop. Add it explicitly or widen the capture."
        )
