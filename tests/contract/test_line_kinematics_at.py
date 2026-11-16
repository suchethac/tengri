# SPDX-License-Identifier: BSD-3-Clause
"""The one width/offset lookup for lines a model renders: ``line_kinematics_at``.

The observation's kinematic groups are the single source of an emission line's
width and velocity offset (owner ruling 2026-11-16). Lines the model renders (the
nebular backend's catalog, for the forward spectrum and the band tables) take the
group of the spectroscopic catalog line they are, matched by rest wavelength at
build with the one tolerance; every other line, and every line of a
photometry-only observation, takes the default group. These tests pin the
membership rule, its refusals and records, its independence of redshift, its use
under ``jit``, and the declaration of the default pair whenever lines are rendered.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.eline_marginalization import (
    LINE_MATCH_TOL_AA,
    kinematic_membership,
    line_kinematics_at,
)

pytestmark = pytest.mark.contract

#: Rendered lines (rest vacuum, Angstrom): H-beta, [O III] 5008, H-alpha, [N II] 6585.
_RENDERED = np.array([4862.69, 5008.24, 6564.61, 6585.27])
_PARAMS = {
    "eline_sigma_kms": 80.0,
    "eline_delta_v_kms": 10.0,
    "eline_sigma_kms_broad": 1500.0,
    "eline_delta_v_kms_broad": 300.0,
}


def _observation(names, waves, groups):
    """An observation stand-in whose spectroscopy carries a grouped catalog."""
    catalog = SimpleNamespace(
        names=tuple(names),
        wavelengths=np.asarray(waves, dtype=float),
        line_groups=tuple(groups),
        kinematic_groups=tuple(dict.fromkeys(groups)),
    )
    return SimpleNamespace(spectroscopy=SimpleNamespace(effective_catalog=catalog))


def test_photometry_only_puts_every_rendered_line_in_the_default_group():
    for observation in (None, SimpleNamespace(spectroscopy=None)):
        sigma, dv = line_kinematics_at(_PARAMS, {}, observation, _RENDERED)
        np.testing.assert_array_equal(np.asarray(sigma), np.full(4, 80.0))
        np.testing.assert_array_equal(np.asarray(dv), np.full(4, 10.0))


def test_a_rendered_line_takes_its_catalog_lines_group():
    """H-alpha in the 'broad' group reads the broad pair; the other lines read the default."""
    obs = _observation(["Hbeta", "Halpha"], [4862.69, 6564.61], ["", "broad"])
    sigma, dv = line_kinematics_at(_PARAMS, {}, obs, _RENDERED)
    np.testing.assert_array_equal(np.asarray(sigma), [80.0, 80.0, 1500.0, 80.0])
    np.testing.assert_array_equal(np.asarray(dv), [10.0, 10.0, 300.0, 10.0])


def test_fixed_values_are_read_when_not_sampled():
    obs = _observation(["Halpha"], [6564.61], ["broad"])
    sigma, _ = line_kinematics_at({}, {"eline_sigma_kms_broad": 900.0}, obs, _RENDERED)
    np.testing.assert_array_equal(np.asarray(sigma), [0.0, 0.0, 900.0, 0.0])


def test_added_components_are_never_rendered():
    """'<line>__<group>' columns are spectrum-side only: the rendered H-alpha stays default."""
    obs = _observation(["Halpha", "Halpha__broad"], [6564.61, 6564.61], ["", "broad"])
    membership = kinematic_membership(obs, _RENDERED)
    np.testing.assert_array_equal(membership.index, [0, 0, 0, 0])


def test_a_line_matching_two_catalog_lines_is_refused():
    obs = _observation(["A", "B"], [6564.40, 6564.80], ["", "broad"])
    with pytest.raises(ValueError, match="ambiguous"):
        kinematic_membership(obs, _RENDERED)


def test_a_grouped_line_nothing_renders_is_recorded():
    """A group set on a line the model does not render is reported, not dropped silently."""
    obs = _observation(["Halpha", "HeII_1640"], [6564.61, 1640.42], ["broad", "broad"])
    membership = kinematic_membership(obs, _RENDERED)
    assert membership.unrendered == ("HeII_1640",)


def test_the_match_tolerance_is_the_one_constant():
    obs = _observation(["Halpha"], [6564.61 + 0.9 * LINE_MATCH_TOL_AA], ["broad"])
    assert kinematic_membership(obs, _RENDERED).index[2] == 1
    obs = _observation(["Halpha"], [6564.61 + 1.1 * LINE_MATCH_TOL_AA], ["broad"])
    assert kinematic_membership(obs, _RENDERED).index[2] == 0


def test_membership_does_not_depend_on_the_redshift():
    """Rest wavelengths decide membership, so a per-galaxy runtime redshift cannot change it."""
    obs = _observation(["Halpha"], [6564.61], ["broad"])
    at_low = line_kinematics_at({**_PARAMS, "redshift": 0.1}, {}, obs, _RENDERED)
    at_high = line_kinematics_at({**_PARAMS, "redshift": 2.0}, {}, obs, _RENDERED)
    for a, b in zip(at_low, at_high):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_jit_traces_only_parameter_values():
    """Under jit the membership is static; a traced wavelength array is refused."""
    obs = _observation(["Halpha"], [6564.61], ["broad"])
    fn = jax.jit(lambda p: line_kinematics_at(p, {}, obs, _RENDERED))
    sigma, _ = fn({k: jnp.asarray(v) for k, v in _PARAMS.items()})
    np.testing.assert_array_equal(np.asarray(sigma), [80.0, 80.0, 1500.0, 80.0])
    grad = jax.grad(
        lambda s: jnp.sum(
            line_kinematics_at({**_PARAMS, "eline_sigma_kms_broad": s}, {}, obs, _RENDERED)[0]
        )
    )(1500.0)
    assert float(grad) == 1.0
    with pytest.raises(jax.errors.TracerArrayConversionError):
        jax.jit(lambda w: line_kinematics_at(_PARAMS, {}, obs, w))(jnp.asarray(_RENDERED))


def test_the_default_pair_is_declared_whenever_lines_are_rendered(ssp_data_fsps):
    """A photometry-only model with Cue declares the pair, Fixed at the line-fitting defaults."""
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel
    from tengri.components.nebular._params import ELINE_PARAMS

    defaults = {d.name: float(d.prior.value) for d in ELINE_PARAMS}
    with_lines = SEDModel.build(
        ssp_data=ssp_data_fsps,
        observation=Observation(photometry=Photometry.from_names(["sdss_r"])),
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )
    fixed = dict(with_lines.spec.get_fixed_values())
    for name in ("eline_sigma_kms", "eline_delta_v_kms"):
        assert name not in with_lines.spec.free_params, name
        assert float(fixed[name]) == defaults[name], name

    no_lines = SEDModel.build(
        ssp_data=ssp_data_fsps,
        observation=Observation(photometry=Photometry.from_names(["sdss_r"])),
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )
    names = set(no_lines.spec.free_params) | set(no_lines.spec.get_fixed_values())
    assert "eline_sigma_kms" not in names
