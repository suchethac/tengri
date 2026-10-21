# SPDX-License-Identifier: BSD-3-Clause
"""Batched spectroscopic stack with a per-galaxy non-parametric SFH bin ladder.

Surface protected: ``GalaxySpectrum(sfh_bin_edges_gyr=...)`` through
``build_spectro_batches`` and ``predict_batched_observables``. A catalog fit whose
continuity ladder follows each galaxy's redshift (``make_agebins_from_zred``) must
reproduce, galaxy by galaxy, a model built with that galaxy's own ``bin_edges_gyr``,
in value and in gradient, with every galaxy in one padded bucket and one compile.
A bucket without per-galaxy edges must be unchanged, and a ladder the template
cannot use must be refused rather than ignored.

Runs on the synthetic narrow SSP (no ``data/ssp_*.h5`` needed).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    Observation,
    SEDModel,
    Spectroscopy,
    Uniform,
    make_agebins_from_zred,
)
from tengri.config.exceptions import UnknownParameterError
from tengri.forward.batched_spectrum import make_batched_predict, predict_batched_observables
from tengri.inference.batched_spectro import batched_neg_log_posterior
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.batched import GalaxySpectrum, SpectroBatchSpec, build_spectro_batches

pytestmark = pytest.mark.contract

_R_BANDED = 300.0
_N_DIAG = 21
_Z_PRIOR = Uniform(0.0, 0.5)
# The template's ladder: built at the prior's upper bound, as the grammar does for a
# free redshift. Each galaxy below replaces it with the ladder at its own redshift.
_TEMPLATE_EDGES = make_agebins_from_zred(zred=0.5)
# Ratios far from flat, so where the edges sit changes the spectrum.
_RATIOS = (0.6, -0.4, 0.8, -0.5, 0.3, -0.7)


def _continuity_model(ssp, wave, bm, edges):
    """Free-redshift spectroscopic model with a continuity SFH on ``edges`` [Gyr]."""
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(wave), resample="conserving", resolution_matrix=bm
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=spectroscopy),
        sfh={"type": "continuity", "all_params": FREE, "bin_edges_gyr": [float(e) for e in edges]},
        met={"logzsol": Fixed(-0.5), "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=_Z_PRIOR,
    )


def _params(model) -> dict:
    """Free parameters other than redshift, with non-flat SFR ratios."""
    p = {}
    for name in model.spec.free_params:
        if name == "redshift":
            continue
        if "_ratio_" in name:
            p[name] = jnp.asarray(_RATIOS[int(name.rsplit("_", 1)[1])])
        elif name.endswith("log_total_mass"):
            p[name] = jnp.asarray(10.0)
        else:
            p[name] = jnp.asarray(model.spec.get_distribution(name).default)
    return p


def _stack(params: dict, batch: int) -> dict:
    return {k: jnp.broadcast_to(jnp.asarray(v), (batch,)) for k, v in params.items()}


@pytest.fixture(scope="module")
def galaxies():
    """Three grids of different length and redshift, each with its own z-scaled ladder."""
    out = []
    for n_pix, lo, hi, z in [(300, 4000.0, 7000.0, 0.05), (360, 4200.0, 7400.0, 0.1),
                             (420, 4500.0, 7500.0, 0.2)]:  # fmt: skip
        wave = np.linspace(lo, hi, n_pix)
        bm = gaussian_resolution_bands(jnp.asarray(wave), _R_BANDED, _N_DIAG)
        out.append({"wave": wave, "z": z, "bm": bm, "edges": make_agebins_from_zred(zred=z)})
    return out


def _records(galaxies, *, with_edges=True):
    return [
        GalaxySpectrum(
            wave=g["wave"],
            flux=np.ones(g["wave"].size),
            ivar=np.ones(g["wave"].size),
            resolution=g["bm"],
            z=g["z"],
            sfh_bin_edges_gyr=g["edges"] if with_edges else None,
        )
        for g in galaxies
    ]


@pytest.fixture(scope="module")
def template(synthetic_ssp, galaxies):
    return _continuity_model(
        synthetic_ssp, galaxies[0]["wave"], galaxies[0]["bm"], _TEMPLATE_EDGES
    )


def test_bucket_matches_per_galaxy_models_built_with_own_edges(synthetic_ssp, galaxies, template):
    """Every galaxy in one padded bucket equals a model built on its own ladder.

    The control is the same bucket without per-galaxy edges: it differs from the
    reference by far more than the tolerance, so the match is the ladder taking
    effect, not the edges being irrelevant on this grid.
    """
    buckets = build_spectro_batches(_records(galaxies), quantum=420, model=template)
    assert len(buckets) == 1
    spec, batch, _ = buckets[0]
    assert spec.n_max == 420 and spec.n_sfh_edges == len(_TEMPLATE_EDGES)

    params = _params(template)
    out = np.asarray(make_batched_predict(template, spec)(_stack(params, 3), batch)["spec_fnu"])

    spec0, batch0, _ = build_spectro_batches(
        _records(galaxies, with_edges=False), quantum=420, model=template
    )[0]
    shared = np.asarray(
        make_batched_predict(template, spec0)(_stack(params, 3), batch0)["spec_fnu"]
    )

    for i, g in enumerate(galaxies):
        n = g["wave"].size
        own = _continuity_model(synthetic_ssp, g["wave"], g["bm"], g["edges"])
        ref = np.asarray(own.predict_spectrum({**params, "redshift": g["z"]}))
        np.testing.assert_allclose(out[i, :n], ref, rtol=1e-10, atol=0.0)
        np.testing.assert_array_equal(out[i, n:], 0.0)
        assert np.max(np.abs(shared[i, :n] / ref - 1.0)) > 1e-3


def test_gradient_matches_per_galaxy_models(synthetic_ssp, galaxies, template):
    """d(sum spec)/d(sfh_cont_ratio_2), per galaxy, equals its own model's gradient.

    The ratio sets the SFR step across one interior edge, so its gradient depends on
    where that edge sits.
    """
    spec, batch, _ = build_spectro_batches(_records(galaxies), quantum=420, model=template)[0]
    params = _params(template)
    predict = make_batched_predict(template, spec)
    name = "sfh_cont_ratio_2"

    def batched_loss(r):
        return jnp.sum(predict({**_stack(params, 3), name: r}, batch)["spec_fnu"])

    grad = np.asarray(jax.grad(batched_loss)(jnp.full((3,), float(params[name]))))

    for i, g in enumerate(galaxies):
        own = _continuity_model(synthetic_ssp, g["wave"], g["bm"], g["edges"])

        def ref_loss(r, own=own, z=g["z"]):
            return jnp.sum(own.predict_spectrum({**params, name: r, "redshift": z}))

        ref = float(jax.grad(ref_loss)(params[name]))
        assert np.isfinite(ref) and ref != 0.0
        np.testing.assert_allclose(grad[i], ref, rtol=1e-9, atol=0.0)


def test_template_ladder_on_the_galaxy_is_the_template(galaxies, template):
    """A galaxy carrying the template's own ladder predicts what the template does.

    The per-galaxy channel moves the edges and nothing else: with the template's
    ladder it agrees with the no-edges bucket and with the template's own
    prediction to the batched path's round-off (``rtol=1e-12``, as in
    ``test_batched_spectrum_forward``).
    """
    g = galaxies[0]
    params = _params(template)
    plain = GalaxySpectrum(
        wave=g["wave"], flux=np.ones(300), ivar=np.ones(300), resolution=g["bm"], z=g["z"]
    )
    spec0, sb0, _ = build_spectro_batches([plain], model=template)[0]
    spec1, sb1, _ = build_spectro_batches(
        [plain._replace(sfh_bin_edges_gyr=_TEMPLATE_EDGES)], model=template
    )[0]
    assert spec0.n_sfh_edges == 0 and sb0.sfh_bin_edges_gyr.shape == (1, 0)
    without = np.asarray(
        predict_batched_observables(template, params, sb0.galaxy(0), spec0)["spec_fnu"]
    )
    ref = np.asarray(template.predict_spectrum({**params, "redshift": g["z"]}))
    with_edges = predict_batched_observables(template, params, sb1.galaxy(0), spec1)["spec_fnu"]
    np.testing.assert_allclose(without, ref, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.asarray(with_edges), without, rtol=1e-12, atol=0.0)


def test_vmapped_posterior_matches_per_galaxy_models(synthetic_ssp, galaxies, template):
    """The MAP/MCMC objective under ``vmap(value_and_grad)`` sees each galaxy's ladder.

    Each galaxy's batched negative log-posterior and its gradient equal those of a
    one-galaxy bucket evaluated with a template built on that galaxy's own ladder.
    """
    records = [
        r._replace(flux=1e-15 * (1.0 + 0.1 * np.sin(g["wave"] / 50.0)))
        for r, g in zip(_records(galaxies), galaxies)
    ]
    spec, batch, _ = build_spectro_batches(records, quantum=420, model=template)[0]
    nlp = batched_neg_log_posterior(template, spec)
    latents = {
        name: jnp.full((3,), 0.3) for name in template.spec.free_params if name != "redshift"
    }
    values, grads = jax.jit(jax.vmap(jax.value_and_grad(nlp)))(latents, batch)

    for i, (g, record) in enumerate(zip(galaxies, records)):
        own = _continuity_model(synthetic_ssp, g["wave"], g["bm"], g["edges"])
        spec_i, batch_i, _ = build_spectro_batches(
            [record._replace(sfh_bin_edges_gyr=None)], model=own
        )[0]
        one = {k: v[i] for k, v in latents.items()}
        ref_value, ref_grad = jax.value_and_grad(batched_neg_log_posterior(own, spec_i))(
            one, batch_i.galaxy(0)
        )
        np.testing.assert_allclose(float(values[i]), float(ref_value), rtol=1e-9)
        for name in latents:
            np.testing.assert_allclose(float(grads[name][i]), float(ref_grad[name]), rtol=1e-7)


def test_galaxies_with_and_without_edges_never_share_a_bucket(galaxies, template):
    """The ladder length is part of the bucket key, so the two kinds stack separately."""
    records = _records(galaxies[:2])
    records[1] = records[1]._replace(sfh_bin_edges_gyr=None)
    buckets = build_spectro_batches(records, quantum=420, model=template)
    assert sorted(spec.n_sfh_edges for spec, _, _ in buckets) == [0, len(_TEMPLATE_EDGES)]


def test_ladder_of_the_wrong_length_is_refused(galaxies, template):
    """The template's ratio parameters fix the bin count; a longer ladder is refused."""
    g = galaxies[0]
    edges = np.concatenate([g["edges"][:-1], [g["edges"][-1] * 0.99, g["edges"][-1]]])
    record = _records([g])[0]._replace(sfh_bin_edges_gyr=edges)
    spec, sb, _ = build_spectro_batches([record], model=template)[0]
    with pytest.raises(ValueError, match="bin edges"):
        predict_batched_observables(template, _params(template), sb.galaxy(0), spec)


def test_ladder_for_an_sfh_without_one_is_refused(synthetic_ssp, galaxies):
    """A parametric template has no bins for the ladder to move: refused, not ignored."""
    g = galaxies[0]
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(g["wave"]), resample="conserving", resolution_matrix=g["bm"]
    )
    model = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=Observation(spectroscopy=spectroscopy),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(8.0, 12.0)},
        neb={"type": "none"},
        redshift=_Z_PRIOR,
    )
    spec, sb, _ = build_spectro_batches(_records([g]), model=model)[0]
    with pytest.raises(ValueError, match="per-evaluation sfh_bin_edges_gyr"):
        predict_batched_observables(model, _params(model), sb.galaxy(0), spec)


def test_malformed_ladder_is_refused_at_batch_build(galaxies, template):
    """A descending ladder never reaches the trace; the error names the galaxy."""
    record = _records(galaxies[:1])[0]._replace(sfh_bin_edges_gyr=galaxies[0]["edges"][::-1])
    with pytest.raises(ValueError, match="galaxy 0: sfh_bin_edges_gyr must be strictly"):
        build_spectro_batches([record], model=template)
    with pytest.raises(ValueError, match="n_sfh_edges"):
        SpectroBatchSpec(n_max=10, offsets=None, grid_kind="nonuniform", n_sfh_edges=1)


def test_single_galaxy_predict_refuses_the_runtime_key(galaxies, template):
    """The runtime ladder is a batched-path channel only.

    A single-galaxy ``predict_state`` would apply it to the spectrum but not to
    the derived SFH properties, so the public parameter check refuses the key.
    """
    params = {**_params(template), "redshift": 0.1, "sfh_bin_edges_gyr": galaxies[1]["edges"]}
    with pytest.raises(UnknownParameterError, match="sfh_bin_edges_gyr"):
        template.predict_state(params)
