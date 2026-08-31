# SPDX-License-Identifier: BSD-3-Clause
"""Regression: ``ForwardModel.build(approx=...)`` must exist — it is the spec's taught
placement for all three LUT families (#1367).

Spec #1320 §5 teaches ``ForwardModel.build(sed=sed, approx=WavePrecomp())`` in five
snippets, with the LUT built against the **authoritative** observation and
reuse-on-match semantics: matching LUT on the sed → reuse; different-filter LUT →
rebuild; no LUT → build. The shipped ``build`` had a closed signature — the taught
call raised ``TypeError`` and approx worked only at ``SEDModel.build``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform, WavePrecomp
from tengri.forward.forward_model import ForwardModel

pytestmark = pytest.mark.regression_bug


def _build_sed(ssp, obs, approx=None):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(9.0, 11.0)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(DEFAULT),
        approx=approx,
    )


def _other_obs():
    """A 3-band observation with different filters than synthetic_tophat_obs."""
    from tengri.observation import Observation, Photometry
    from tengri.observation.photometry import FilterCurve

    def _tophat(center, frac=0.16, n=40):
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"alt{int(center)}")

    curves = tuple(_tophat(c) for c in (4000.0, 5500.0, 8000.0))
    return Observation(photometry=Photometry(filters=curves))


def test_build_accepts_approx_and_builds_the_lut(synthetic_ssp_wide, synthetic_tophat_obs):
    """The spec §5 headline call runs, and the fwd's sed carries the LUT."""
    sed = _build_sed(synthetic_ssp_wide, synthetic_tophat_obs)

    fwd = ForwardModel.build(sed=sed, approx=WavePrecomp())

    assert fwd.populations[0].sed._approx["wave_precomp"] is True
    # The user's model is untouched — with_approx clones (immutability).
    assert sed._approx["wave_precomp"] is False


def test_build_without_approx_is_the_existing_path(synthetic_ssp_wide, synthetic_tophat_obs):
    """approx omitted → byte-identical behavior: the sed is used as-is."""
    sed = _build_sed(synthetic_ssp_wide, synthetic_tophat_obs)

    fwd = ForwardModel.build(sed=sed)

    assert fwd.populations[0].sed is sed


def test_build_reuses_a_matching_lut(synthetic_ssp_wide, synthetic_tophat_obs):
    """sed already carries the requested LUT → reuse, no rebuild (spec §5)."""
    sed = _build_sed(synthetic_ssp_wide, synthetic_tophat_obs, approx=WavePrecomp())

    fwd = ForwardModel.build(sed=sed, approx=WavePrecomp())

    assert fwd.populations[0].sed is sed


def test_build_rebuilds_against_the_authoritative_observation(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """Different-filter LUT + approx= → rebuild against the fwd's observation.

    Without approx= this exact configuration raises the #1315 mismatch guard
    (whose message says "rebuild the sed with this observation" — which is
    precisely what approx= now does). The rebuilt path must agree bit-exactly
    with a directly-built model on the authoritative observation.
    """
    sed = _build_sed(synthetic_ssp_wide, synthetic_tophat_obs, approx=WavePrecomp())
    obs_b = _other_obs()

    fwd = ForwardModel.build(sed=sed, observation=obs_b, approx=WavePrecomp())

    inner = fwd.populations[0].sed
    assert inner is not sed
    assert inner._approx["wave_precomp"] is True
    assert [f.name for f in inner.observation.photometry.filters] == [
        f.name for f in obs_b.photometry.filters
    ]

    import jax

    direct = _build_sed(synthetic_ssp_wide, obs_b, approx=WavePrecomp())
    params = direct.spec.sample(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(
        np.asarray(inner.predict_photometry(params)),
        np.asarray(direct.predict_photometry(params)),
    )


def test_build_approx_without_sed_raises_mode_aware_error(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """approx= applies to the single-sed form — one clear error otherwise (§4 style)."""
    from tengri.forward.forward_model import Population

    sed = _build_sed(synthetic_ssp_wide, synthetic_tophat_obs)
    pops = [Population(name="a", sed=sed)]

    with pytest.raises(ValueError, match="approx"):
        ForwardModel.build(
            populations=pops, observation=synthetic_tophat_obs, approx=WavePrecomp()
        )


def test_build_approx_grammar_errors_surface_canonically(synthetic_ssp_wide, synthetic_tophat_obs):
    """Invalid approx members raise SEDModel's canonical TypeError, not a new one."""
    sed = _build_sed(synthetic_ssp_wide, synthetic_tophat_obs)

    with pytest.raises(TypeError):
        ForwardModel.build(sed=sed, approx="wave_precomp")
