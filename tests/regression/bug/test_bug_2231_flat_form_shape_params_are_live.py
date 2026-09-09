# SPDX-License-Identifier: BSD-3-Clause
r"""A flat ``Parameters(...)`` spec must thread dust shape params to the law (#2231).

``SEDModel._requested_law_shape_params`` decides which ``dust_*`` attenuation
shape parameters (``dust_slope``, ``dust_delta``, ``dust_Rv``,
``dust_bump_strength``) are "live" by reading ``spec._group_provenance``, the
richer map ``parse_groups`` attaches after construction. A flat
``Parameters(...)`` call never gets that map, so every name resolved to
``"registry_default"``, was excluded from ``live_shape_params``, and the
attenuation law silently evaluated its own published default no matter what
the flat spec declared. The parameter still showed up in ``spec.free_params``
and got sampled, so a fit reported convergence while exploring a direction the
forward model could not see -- the flat-form sibling of #1808/#2185.

The fix (``Parameters.__init__``) records a narrower ``_flat_provenance`` map
for every parameter the constructor call actually named, tagged
``"user_prior"``/``"user_fixed"`` by presence, never by comparing against the
registry default (an explicit value equal to the default -- e.g.
``dust_bump_strength=Fixed(1.0)``, KC13's own value -- is still a request).
``_requested_law_shape_params`` reads this map only when ``_group_provenance``
is absent, so a grammar-built spec is untouched.

Each parity case below is checked against a group-form control built through
``parse_groups`` with the identical law/value: the flat and group relative
photometry change from moving one shape parameter must agree to machine
precision, and the change must be >1% in at least one band (an in-model
control against a trivially-vacuous 0 == 0 pass).

Note on ``clear_shared_caches()``: two ``SEDModel`` instances with an
identical ``compile_signature()`` share a cached, closure-captured prediction
kernel (see ``SEDModel._get_or_build_predict_observables_jit``).
``compile_signature()`` does not currently key on which shape parameters a
build marked "live" (a gap orthogonal to this fix -- flagged, not addressed,
here), so building a flat and a group model with the same law back-to-back in
one process can silently hand one model's compiled kernel to the other.
Clearing the shared caches before every ``predict_photometry`` call below
forces each model to compile against its own spec.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    SSPData,
    parse_groups,
)
from tengri.inference.jit_engine import clear_shared_caches

pytestmark = pytest.mark.regression_bug

_Z = 0.075
_FILTER_NAMES = ("galex_fuv", "galex_nuv", "sdss_u", "sdss_g", "sdss_r")

#: Every non-dust free parameter pinned to the same numbers on both surfaces,
#: so the ONLY thing that can differ between a flat and a group build is the
#: dust shape parameter under test.
_SFH_FLAT: dict[str, object] = {
    "sfh_dpl_alpha": Fixed(1.0),
    "sfh_dpl_beta": Fixed(0.6),
    "sfh_dpl_tau_gyr": Fixed(3.0),
    "sfh_dpl_log_total_mass": Fixed(10.0),
    "sfh_dpl_age_gyr": Fixed(5.0),
    "met_logzsol": Fixed(0.0),
}
_SFH_SHORT: dict[str, float] = {
    "alpha": 1.0,
    "beta": 0.6,
    "tau_gyr": 3.0,
    "log_total_mass": 10.0,
    "age_gyr": 5.0,
}

#: (law, shape parameter, low value, high value) -- the four cases from the
#: issue, one per registered shape-parameter family (bump, delta, Rv, slope).
_PARITY_CASES = (
    ("kriek_conroy", "dust_bump_strength", 0.0, 3.3),
    ("kriek_conroy", "dust_delta", 0.0, -0.6),
    ("cardelli", "dust_Rv", 3.1, 5.0),
    ("power_law", "dust_slope", -0.7, -1.3),
)

#: Relative-change floor a swept shape parameter must clear in at least one
#: band, so a passing parity assertion is not a vacuous 0.0 == 0.0.
_LIVE_FLOOR = 0.01


def _build_ssp() -> SSPData:
    """Synthetic SSP: 25 log-ages, 3 metallicities, 1600-point log wavelength grid."""
    ages = jnp.linspace(-3.0, 1.14, 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    wave = jnp.logspace(2.0, 7.0, 1600)
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages - ages.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave, ssp_flux=jnp.abs(flux) + 1e-30, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet
    )


def _obs() -> Observation:
    return Observation(photometry=Photometry.from_names(list(_FILTER_NAMES)))


def _flat_model(ssp, obs, law: str, tau_v: float = 0.6, **shape_kwargs) -> SEDModel:
    """Flat ``Parameters(...)`` build: ``single_component`` + ``law``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = Parameters(
            mean_sfh_type="dpl",
            dust_model="single_component",
            dust_law_bc=law,
            redshift=Fixed(_Z),
            dust_tau_v=Fixed(tau_v),
            **_SFH_FLAT,
            **{name: Fixed(value) for name, value in shape_kwargs.items()},
        )
    return SEDModel(spec, ssp, observation=obs)


def _group_model(ssp, obs, law: str, tau_v: float = 0.6, **shape_kwargs) -> SEDModel:
    """Equivalent ``parse_groups`` build: same law/values, nothing else free."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT), **_SFH_SHORT},
            dust_attenuation={
                "type": "single_component",
                "law": law,
                "all_params": Fixed(DEFAULT),
                "tau_v": tau_v,
                **shape_kwargs,
            },
            redshift=Fixed(_Z),
        )
    return SEDModel(spec, ssp, observation=obs)


def _phot(model: SEDModel) -> np.ndarray:
    """Photometry with every free parameter Fixed, guarded against #2231's
    orthogonal ``compile_signature()`` kernel-sharing hazard (see module
    docstring)."""
    clear_shared_caches()
    return np.asarray(model.predict_photometry({}))


def _rel_change(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.abs(high - low) / np.maximum(np.abs(low), 1e-300)


class TestFlatMatchesGroupParity:
    """Sweeping a shape parameter must move flat-form photometry exactly like
    the equivalent group-form build."""

    @pytest.mark.parametrize(("law", "param", "lo", "hi"), _PARITY_CASES)
    def test_flat_and_group_relative_change_agree(self, law, param, lo, hi):
        ssp = _build_ssp()
        obs = _obs()

        flat_lo = _phot(_flat_model(ssp, obs, law, **{param: lo}))
        flat_hi = _phot(_flat_model(ssp, obs, law, **{param: hi}))
        group_lo = _phot(_group_model(ssp, obs, law, **{param: lo}))
        group_hi = _phot(_group_model(ssp, obs, law, **{param: hi}))

        flat_rel = _rel_change(flat_lo, flat_hi)
        group_rel = _rel_change(group_lo, group_hi)

        # In-model control: a null result must be the parameter's, not the
        # fixture's. If this fails, the fixture (SSP/filters/law) cannot
        # detect the parameter at all, and the parity check below would be
        # a vacuous 0.0 == 0.0 pass.
        assert group_rel.max() > _LIVE_FLOOR, (
            f"{law}/{param}: the group-form control moved by only "
            f"{group_rel.max():.4%} across bands; the fixture cannot see "
            "this parameter, so a flat/group agreement here would be vacuous."
        )

        np.testing.assert_allclose(
            flat_rel,
            group_rel,
            rtol=1e-10,
            atol=0.0,
            err_msg=(
                f"{law}/{param}: flat-form relative change {flat_rel} does not "
                f"match the group-form control {group_rel}. A flat "
                f"Parameters(...) spec must thread this shape parameter to "
                f"the law exactly like parse_groups (#2231)."
            ),
        )


class TestExplicitDefaultIsARequest:
    """Presence in the constructor call is the test, never a value comparison."""

    def test_explicit_value_at_the_law_published_default_is_still_live(self):
        """``dust_bump_strength=Fixed(1.0)`` -- KC13's own published default
        (the value ``apply()`` would use anyway on the no-request path) --
        must still mark the curve live: a fix that compared the requested
        value against what the *law* would otherwise produce, rather than
        checking presence in the constructor call, would silently treat this
        exact case as "nobody asked"."""
        ssp = _build_ssp()
        obs = _obs()
        model = _flat_model(ssp, obs, "kriek_conroy", dust_bump_strength=1.0)
        assert "dust_bump_strength" in model._requested_law_shape_params()

    def test_explicit_value_at_the_shared_spec_default_is_still_live(self):
        """``dust_bump_strength=Fixed(0.0)`` equals ``Parameters``'s own
        shared registry default for this parameter (0.0, per the class
        docstring's parameter table -- distinct from KC13's law default of
        1.0 above). An implementation that skips recording provenance
        whenever the requested value matches ``self._defaults[name]`` would
        silently un-request exactly this explicit call."""
        ssp = _build_ssp()
        obs = _obs()
        model = _flat_model(ssp, obs, "kriek_conroy", dust_bump_strength=0.0)
        assert "dust_bump_strength" in model._requested_law_shape_params()

    def test_no_shape_kwarg_is_not_live(self):
        ssp = _build_ssp()
        obs = _obs()
        model = _flat_model(ssp, obs, "kriek_conroy")
        assert "dust_bump_strength" not in model._requested_law_shape_params()
        assert "dust_delta" not in model._requested_law_shape_params()


class TestNoKwargFlatSpecUnchanged:
    """A flat spec that never names a shape parameter must be bit-identical
    to the equivalent group spec at the law's published default (#2231
    changes provenance bookkeeping only, never the no-request path)."""

    @pytest.mark.parametrize("law", ("kriek_conroy", "cardelli", "power_law", "calzetti"))
    def test_default_photometry_matches_group_form(self, law):
        ssp = _build_ssp()
        obs = _obs()

        flat_phot = _phot(_flat_model(ssp, obs, law))
        group_phot = _phot(_group_model(ssp, obs, law))

        np.testing.assert_allclose(flat_phot, group_phot, rtol=1e-12, atol=0.0)


class TestGrammarProvenanceUnchanged:
    """parse_groups' own ``_group_provenance`` must be untouched by #2231.

    The expected dict below was captured from this exact build BEFORE the
    source change in ``Parameters.__init__`` (see the PR report), so any
    regression in ordering -- the flat-form default winning over parse_groups'
    richer map, or vice versa in a way that changes tags -- fails here.
    """

    def test_two_component_provenance_dict_is_unchanged(self):
        from tengri import FREE, Uniform

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = parse_groups(
                sfh={
                    "type": "dpl",
                    "all_params": FREE,
                    "beta": Uniform(1, 3),
                    "alpha": Fixed(2.0),
                },
                dust_attenuation={
                    "type": "two_component",
                    "law": "kriek_conroy",
                    "all_params": Fixed(DEFAULT),
                    "tau_bc": 0.5,
                    "delta": Uniform(-1.0, 0.4),
                },
                redshift=Fixed(0.075),
            )

        expected = {
            "dust_Rv": "wildcard_fixed",
            "dust_bump_strength": "wildcard_fixed",
            "dust_delta": "user_prior",
            "dust_f_obscuration": "wildcard_fixed",
            "dust_slope": "wildcard_fixed",
            "dust_tau_bc": "user_fixed",
            "dust_tau_diff": "wildcard_fixed",
            "met_alpha_fe": "wildcard_fixed",
            "met_logzsol": "wildcard_fixed",
            "met_logzsol_scatter": "wildcard_fixed",
            "noise_dof": "registry_default",
            "noise_frac_cal": "registry_default",
            "redshift": "user_fixed",
            "sfh_dpl_age_gyr": "wildcard_free",
            "sfh_dpl_alpha": "user_fixed",
            "sfh_dpl_beta": "user_prior",
            "sfh_dpl_log_total_mass": "wildcard_free",
            "sfh_dpl_tau_gyr": "wildcard_free",
            "sigma_v_kms": "registry_default",
        }
        assert spec._group_provenance == expected

    def test_flat_construction_still_carries_no_group_provenance(self):
        """The grammar-only attribute must stay absent on a flat spec: several
        consumers (translate.py's ``legacy_flat_spec`` gate, summary()'s
        Source-column toggle) key on its mere presence, not its content."""
        spec = Parameters(mean_sfh_type="dpl", redshift=Fixed(0.1), dust_bump_strength=Fixed(2.0))
        assert not hasattr(spec, "_group_provenance")
        assert spec._flat_provenance.get("dust_bump_strength") == "user_fixed"
