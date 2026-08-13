# SPDX-License-Identifier: BSD-3-Clause
r"""A precompute is a speed knob: it must not move float64 physics.

``FeaturePrecomp`` is the *line*-channel precompute. Adding it on top of
``WavePrecomp`` changed the *photometry* channel by 11.04 % in the far-IR, in
float64, with the posterior gradient up to 380 % wrong — and #1596 had just made
it the auto-resolved default for every photometry-only Cue fit.

Mechanism: serving photometry from the per-Q_H grid requires zeroing
``sed_nebular``, which was done unconditionally on the stated grounds that its
"only live consumers are the exact spectrum / dust-continuum paths". The dust
energy balance reads it too, to size the absorbed budget, so a model with dust
emission re-emitted the stellar half alone. ``l_dust_absorbed`` under the
feature LUT equalled the **neb=none** model's value to ~1.6e-11 relative.

Two properties are pinned, and they fail for different reasons:

* **Channel isolation** — bit-exact, no tolerance to argue about. Adding a
  line-channel precompute must leave photometry and the energy-balance
  quantities *identical*. This is what the defect violated (0.1104, not 0).
* **Energy-balance fidelity** — no approx may lose the nebular term. Pinned
  against the *nebular contribution itself*, so the assertion states the
  physics rather than quoting a measured constant: recovering none of it is the
  defect, and the bound is derived from the gap between the Cue and neb=none
  models on the same grid.

Deliberately a *physics* test on a real model. The test shipped with #1596
(``test_issue_1596_photometry_feature_default.py``) stubs the model out and
asks only which config the auto policy resolves — a policy question that stays
green no matter what the resolved config then computes.

Every nebular parameter is FIXED in every row, so the Cue feature grid has zero
free ionization axes and ``L = Q_H * l(theta)`` is exact by construction. The
grid-density sweep is part of the guard: an interpolation error shrinks with
``n_grid``, and this one did not move at all between 4 and 32.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.forward.sed_model import FeaturePrecomp, WavePrecomp

pytestmark = pytest.mark.regression_bug

_BASE = dict(
    sfh={
        "type": "delayed",
        "all_params": FIXED,
        "log_total_mass": Uniform(9.0, 11.0),
        "tau_gyr": 1.0,
        "age_gyr": 5.0,
    },
    redshift=Fixed(0.1),
)

#: Dust emission is load-bearing: it is the energy-balance consumer of
#: ``sed_nebular``, and without an IR component nothing reads the absorbed
#: budget and the defect is invisible.
_DUST = {
    "type": "two_component",
    "law_bc": "calzetti",
    "all_params": FIXED,
    "tau_diff": Uniform(0.0, 1.5),
    "tau_bc": 0.0,
    "emission": {"type": "dale2014", "all_params": FIXED},
}

_AGN = {
    "type": "composable",
    "all_params": FIXED,
    "disc": {"type": "multicolor", "all_params": FIXED},
    "torus": {"type": "skirtor", "all_params": FIXED},
    "norm": "cigale_joint",
    "log_lbol": Uniform(9.0, 12.0),
    "fracAGN": 0.1,
}

#: Enumerated by *composition*, not by picking a representative: the defect was
#: identical (0.1104) across all four, which is what proved it was Cue itself
#: rather than an interaction with shock or AGN.
_MODELS = {
    "cue": dict(dust=_DUST, neb={"type": "cue", "all_params": FIXED}),
    "cue_shock": dict(dust=_DUST, neb={"type": "cue", "all_params": FIXED}, shock={"frac": 0.1}),
    "cue_agn": dict(dust=_DUST, neb={"type": "cue", "all_params": FIXED}, agn=_AGN),
}

#: herschel_250 is load-bearing: the nebular light reaches 250 um only by being
#: absorbed and re-emitted, so it is the band the energy-balance term drives.
_BANDS = ["sdss_g", "sdss_r", "wise_w1", "herschel_250"]

_EB_KEYS = ("l_dust_absorbed", "l_tir")


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(_BANDS))


def _build(ssp, obs, groups, approx):
    kw = dict(ssp_data=ssp, observation=obs, **_BASE, **groups)
    if approx is not None:
        kw["approx"] = approx
    return SEDModel.build(**kw)


def _at_prior_center(model):
    return {
        n: float(model.spec._distributions[n].unstandardize(jnp.asarray(0.0)))
        for n in model.spec.free_params
    }


def _photometry_and_eb(ssp, obs, groups, approx):
    model = _build(ssp, obs, groups, approx)
    params = _at_prior_center(model)
    phot = np.asarray(model.predict_photometry(params), dtype=np.float64)
    props = model.predict_properties(params, names=_EB_KEYS)
    eb = {k: float(np.asarray(props[k])) for k in _EB_KEYS}
    return phot, eb


@pytest.mark.parametrize("composition", sorted(_MODELS))
def test_feature_precomp_does_not_move_the_photometry_channel(ssp_data_fsps, obs, composition):
    """Adding the line-channel LUT must leave photometry bit-identical.

    Bit-exact on purpose. ``FeaturePrecomp`` tabulates emission lines; whatever
    it does to the line channel, the photometry channel and the absorbed-energy
    budget are not its business, so there is no residual to allow. Stating it as
    equality also means the guard cannot be satisfied by widening a tolerance.
    """
    groups = _MODELS[composition]
    with jax.enable_x64(True):
        base_phot, base_eb = _photometry_and_eb(ssp_data_fsps, obs, groups, (WavePrecomp(),))
        feat_phot, feat_eb = _photometry_and_eb(
            ssp_data_fsps, obs, groups, (WavePrecomp(), FeaturePrecomp())
        )

    # ``array_equal``, not ``rel.max() == 0.0``: a band whose flux is exactly
    # zero makes ``rel`` nan, and ``nan == 0.0`` is False — the guard would
    # still fail, but reporting a nan instead of the number that explains it.
    # The ratio is computed only to describe the failure.
    identical = np.array_equal(feat_phot, base_phot)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(feat_phot - base_phot) / np.abs(base_phot)
    assert identical, (
        f"FeaturePrecomp moved the photometry channel by {np.nanmax(rel):.4e} on the "
        f"{composition} model (bands={_BANDS}, per-band rel={rel.tolist()}). A "
        "line-channel precompute must not touch photometry: serving photometry "
        "from the per-Q_H grid zeroes sed_nebular, and the dust energy balance "
        "reads it to size the absorbed budget."
    )
    for key in _EB_KEYS:
        assert feat_eb[key] == base_eb[key], (
            f"FeaturePrecomp moved {key} on the {composition} model: "
            f"{feat_eb[key]!r} vs {base_eb[key]!r}. The absorbed-energy budget "
            "must not depend on which precompute serves the line channel."
        )


def test_feature_precomp_keeps_the_nebular_term_in_the_energy_balance(ssp_data_fsps, obs):
    """The absorbed budget must retain the nebular contribution under any approx.

    The bound is derived, not quoted: the nebular contribution is measured as
    the gap between the Cue model and the same model with ``neb='none'`` on this
    grid, and the guard requires the feature path to recover essentially all of
    it. The defect recovered **none** (-0.2 %), landing on the no-nebular value
    to ~1.6e-11 relative -- so a test that merely asserted "not zero" or checked
    finiteness would have passed while the physics was gone.
    """
    cue = _MODELS["cue"]
    no_neb = dict(dust=_DUST, neb={"type": "none"})
    with jax.enable_x64(True):
        _, eb_exact = _photometry_and_eb(ssp_data_fsps, obs, cue, None)
        _, eb_noneb = _photometry_and_eb(ssp_data_fsps, obs, no_neb, None)
        _, eb_feat = _photometry_and_eb(ssp_data_fsps, obs, cue, (WavePrecomp(), FeaturePrecomp()))

    for key in _EB_KEYS:
        contribution = eb_exact[key] - eb_noneb[key]
        assert contribution > 0.0, (
            f"fixture no longer isolates a nebular {key} contribution "
            f"({eb_exact[key]!r} vs {eb_noneb[key]!r}); the guard below would be "
            "vacuous, so fix the fixture rather than the assertion"
        )
        recovered = (eb_feat[key] - eb_noneb[key]) / contribution
        assert recovered > 0.99, (
            f"the feature LUT recovered {recovered:.4f} of the nebular {key} "
            f"contribution ({contribution:.6e} erg/s on this grid). Recovering ~0 "
            "means sed_nebular was zeroed while the dust energy balance still "
            "needed it."
        )


def test_the_grid_still_serves_photometry_when_nothing_consumes_the_continuum(ssp_data_fsps, obs):
    """The fix must be conditional, not a global disable of the fast path.

    Correctness bought by switching the optimization off everywhere would be a
    silent performance regression instead of a silent physics one, and #1596
    exists because that cost is real (~4x on a photometry-only Cue fit). With no
    dust in the chain nothing declares ``sed_nebular``, so the grid may still
    zero the continuum and serve photometry — and must.
    """
    from tengri.components.nebular.component import NebularSEDComponent

    model = _build(
        ssp_data_fsps,
        obs,
        dict(dust={"type": "none"}, neb={"type": "cue", "all_params": FIXED}),
        (WavePrecomp(), FeaturePrecomp()),
    )
    neb = [c for c in model._cached_component_chain if isinstance(c, NebularSEDComponent)]
    assert neb, "no nebular component in the chain — fixture no longer tests anything"
    assert neb[0].must_materialize_sed is False, (
        "the per-Q_H grid was denied the photometry channel on a model with no "
        "sed_nebular consumer at all. The exclusion must key on an actual "
        "declared consumer, not fire whenever a grid is attached."
    )
    # This exact configuration is also where #1673 bites: with the shortcut
    # (correctly) enabled, ``sed_components()`` reports sed_nebular as zero,
    # because that reader takes the published key without declaring an input
    # and so is invisible to the census. Pinned there, not here — this test
    # owns the perf half of the trade.


@pytest.mark.parametrize("dust_type", ["single_component", "two_component", "wg00"])
def test_every_dust_law_is_seen_as_a_nebular_consumer(ssp_data_fsps, obs, dust_type):
    """The derivation must find the consumer for *every* shipped dust law.

    All three declare ``sed_nebular`` as an optional input, and all three drive
    an energy balance. A fix written against the one model that happened to be
    under test would have covered one of three, silently. This asks the
    contract directly, so it is fast, and it fails the moment a dust law is
    added whose declaration is missing.
    """
    from tengri.forward.orchestrator import components_consuming

    model = _build(
        ssp_data_fsps,
        obs,
        dict(
            dust={"type": dust_type, "all_params": FIXED},
            neb={"type": "cue", "all_params": FIXED},
        ),
        None,
    )
    consumers = components_consuming(model._build_component_chain(), "sed_nebular")
    kinds = {type(c).__name__ for c in consumers}
    assert kinds - {"NebularSEDComponent"}, (
        f"no component in the {dust_type} chain declares sed_nebular as an input "
        f"(found {sorted(kinds)}). The per-Q_H grid decides whether it may zero "
        "the continuum by asking this question, so a missing declaration silently "
        "re-enables the dropped-nebular energy-balance defect."
    )


def test_the_feature_error_was_never_interpolation(ssp_data_fsps, obs):
    """Grid density must not be able to explain a photometry shift.

    ``n_grid`` is the knob a reader reaches for first, and the docstring invites
    it ("Denser is tighter"). With every ionization axis FIXED there are no free
    axes to refine, so density is a no-op here -- and when the defect was live
    the error was bit-identical at 4, 8, 16 and 32. Pinning that keeps the next
    reader from spending the session tuning a knob that cannot move the answer.
    """
    cue = _MODELS["cue"]
    with jax.enable_x64(True):
        base, _ = _photometry_and_eb(ssp_data_fsps, obs, cue, (WavePrecomp(),))
        by_n = {
            n: _photometry_and_eb(
                ssp_data_fsps, obs, cue, (WavePrecomp(), FeaturePrecomp(n_grid=n))
            )[0]
            for n in (4, 32)
        }

    for n, phot in by_n.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(phot - base) / np.abs(base)
        assert np.array_equal(phot, base), (
            f"n_grid={n} moved photometry by {np.nanmax(rel):.4e}. With no free "
            "ionization axes the grid has nothing to interpolate, so any shift "
            "here is a dropped term, not resolution."
        )
