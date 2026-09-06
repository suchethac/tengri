# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end reachability of the new KD18-agnfitter disc blocks' parameters.

A block can be perfectly implemented and still be practically unreachable
through the public ``SEDModel.build`` grammar if nothing wires its
parameters into a scope a user can actually free. This module checks the
THREE surfaces that reach ``kd18_agnfitter`` / ``kd18_agnfitter_warmindex``
parameters -- explicit per-parameter short keys inside the ``agn.disc``
sub-block, the top-level ``agn={'all_params': FREE}`` wildcard, and (Task 16)
the ``agn.disc`` sub-block's OWN wildcard, ``disc={'all_params': FREE}`` --
and that a free parameter actually moves a UV/optical band's photometry
(``jax.grad`` nonzero).

The sub-block wildcard used to be a structural no-op for every composable
disc type (measured: 13 of the 14 registered ``AGN_BLOCKS["disc"]`` types),
not a defect in this task's two blocks specifically:
``_agn_subblock_declared_params`` (``parameters/groups.py``) filters a disc
block's own signature down to parameters owned by the ``"agn.disc"`` group
in ``_AGN_PARTITION`` -- and ``_AGN_PARTITION`` had **no entries at all**
mapped to ``"agn.disc"`` (every disc-relevant name, including
``agn_log_mbh``/``agn_log_ledd``/``agn_gamma_warm``, was partitioned to the
shared ``"agn"`` group instead). Task 16 (AGNfitter-rX parity plan, item 1
addendum) gave disc physics parameters their own ``"agn.disc"`` entries in
``_AGN_PARTITION``, so the sub-block wildcard now reaches them like any
other composable sub-block's own wildcard.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel
from tengri.config.exceptions import WildcardNoOpWarning
from tengri.observation import Photometry
from tengri.parameters.priors import Uniform

pytestmark = pytest.mark.contract

#: A UV/optical band pair: AGN disc continuum dominates blueward, so a
#: nonzero disc-parameter gradient here is a meaningful, non-vacuous check
#: (not a band where the AGN disc contributes negligibly).
_UV_OPTICAL_BANDS = ("sdss_u", "sdss_g")

#: (disc_type, extra explicit short-key params beyond log_mbh/log_ledd).
_DISC_TYPES = ("kd18_agnfitter", "kd18_agnfitter_warmindex")


@pytest.fixture(scope="module")
def ssp_data():
    import tengri

    return tengri.load_ssp()


def _short_key_priors(disc_type: str) -> dict:
    """The explicit per-parameter short keys this disc type's grid supports."""
    priors = {
        "log_mbh": Uniform(6.5, 9.5),
        "log_ledd": Uniform(-1.3, -0.2),
    }
    if disc_type == "kd18_agnfitter_warmindex":
        priors["gamma_warm"] = Uniform(2.0, 3.0)
    return priors


def _build(ssp_data, disc_type: str, *, disc_all_params=None, disc_explicit=None):
    """One minimal composable-AGN SEDModel with ``disc_type`` selected.

    Parameters
    ----------
    disc_all_params : sentinel or None
        If given, passed as the ``agn.disc`` sub-block's ``'all_params'``.
    disc_explicit : dict or None
        If given, explicit per-parameter short-key priors for the disc
        sub-block (mutually exclusive with ``disc_all_params`` in the tests
        below, though the grammar itself allows combining them).
    """
    disc_dict: dict = {"type": disc_type}
    if disc_all_params is not None:
        disc_dict["all_params"] = disc_all_params
    if disc_explicit is not None:
        disc_dict.update(disc_explicit)

    obs = Photometry.from_names(list(_UV_OPTICAL_BANDS))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp_data,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "none"},
            agn={
                "type": "composable",
                "disc": disc_dict,
                "norm": "independent",
                "log_lbol": Uniform(10.0, 12.0),
            },
            redshift=Fixed(0.1),
        )


@pytest.mark.parametrize("disc_type", _DISC_TYPES)
def test_explicit_short_keys_free_exactly_those_names(ssp_data, disc_type):
    """(a) Explicit per-parameter short keys inside ``agn.disc`` are live.

    ``disc={'type': T, 'log_mbh': Uniform(...), 'log_ledd': Uniform(...)
    [, 'gamma_warm': Uniform(...)]}`` must free exactly ``agn_log_mbh``,
    ``agn_log_ledd`` (and ``agn_gamma_warm`` for the warmindex type) with
    the given bounds -- this surface does not go through the sub-block
    wildcard/partition machinery at all, so it is unaffected by the
    ``_AGN_PARTITION`` gap documented above.
    """
    explicit = _short_key_priors(disc_type)
    model = _build(ssp_data, disc_type, disc_explicit=explicit)
    spec = model.spec

    assert "agn_log_mbh" in spec.free_params
    assert "agn_log_ledd" in spec.free_params
    assert spec._distributions["agn_log_mbh"].bounds == pytest.approx((6.5, 9.5))
    assert spec._distributions["agn_log_ledd"].bounds == pytest.approx((-1.3, -0.2))
    if disc_type == "kd18_agnfitter_warmindex":
        assert "agn_gamma_warm" in spec.free_params
        assert spec._distributions["agn_gamma_warm"].bounds == pytest.approx((2.0, 3.0))


@pytest.mark.parametrize("disc_type", _DISC_TYPES)
def test_top_level_wildcard_frees_them(ssp_data, disc_type):
    """(b) ``agn={'all_params': FREE}`` reaches the new disc's parameters.

    Neither new block is in ``AGN_BLOCK_CONSUMES`` (same as ``slone_netzer``,
    ``cat3d_wind``: documented, safe degradation), so the top-level scope
    falls back to the full ``agn_*`` superset -- which contains
    ``agn_log_mbh``/``agn_log_ledd``/``agn_gamma_warm`` -- rather than the
    narrower per-block declared set. This over-frees (never under-frees).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=ssp_data,
            observation=Photometry.from_names(list(_UV_OPTICAL_BANDS)),
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            # A dust component (not 'none'): the top-level wildcard also frees
            # agn_ir_frac (fracAGN), which requires a dust-absorbed-luminosity
            # sink to normalize against (#944) -- unrelated to this test.
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "other_params": Fixed(DEFAULT),
            },
            agn={
                "type": "composable",
                "disc": {"type": disc_type},
                "norm": "independent",
                "all_params": FREE,
            },
            redshift=Fixed(0.1),
        )
    free = model.spec.free_params
    assert "agn_log_mbh" in free
    assert "agn_log_ledd" in free
    if disc_type == "kd18_agnfitter_warmindex":
        assert "agn_gamma_warm" in free


@pytest.mark.parametrize("disc_type", _DISC_TYPES)
def test_free_parameters_move_a_uv_optical_band(ssp_data, disc_type):
    """(c) jax.grad of a UV/optical band flux is nonzero for every freed param.

    Not merely "declared" -- actually differentiable and non-dead (the
    #1764 failure class: a registered-but-inert knob reports the prior back
    as the posterior with no signal anything is wrong).
    """
    explicit = _short_key_priors(disc_type)
    model = _build(ssp_data, disc_type, disc_explicit=explicit)
    params = model.spec.sample(jax.random.PRNGKey(0))

    grad = jax.grad(lambda p: jnp.sum(model.predict_photometry(p)))(params)

    names = ["agn_log_mbh", "agn_log_ledd"]
    if disc_type == "kd18_agnfitter_warmindex":
        names.append("agn_gamma_warm")
    for name in names:
        g = float(grad[name])
        assert jnp.isfinite(g), f"{name}: non-finite gradient {g}"
        assert g != 0.0, f"{name}: identically-zero gradient on a UV/optical band"


@pytest.mark.parametrize("disc_type", _DISC_TYPES)
def test_subblock_wildcard_frees_disc_physics_params(ssp_data, disc_type):
    """(d) Task 16 (AGNfitter-rX parity plan, item 1 addendum): ``disc={'type':
    T, 'all_params': FREE}`` -- the ``agn.disc`` sub-block's OWN wildcard --
    now frees ``agn_log_mbh``/``agn_log_ledd`` (and ``agn_gamma_warm`` for the
    warmindex type), because ``_AGN_PARTITION`` now maps them to
    ``"agn.disc"`` instead of the shared ``"agn"`` group (see module
    docstring). This flips the PREVIOUSLY pinned no-op behavior (which
    asserted ``pytest.warns(WildcardNoOpWarning)``); no warning should fire
    now that the wildcard covers real parameters, and the freed parameters
    must be live (nonzero gradient), not merely present.

    Since #2187, a covered-zero wildcard raises ``ParameterError`` rather
    than warning, so the ``simplefilter`` below no longer carries the weight
    it did: the load-bearing assertion is that ``SEDModel.build`` returns at
    all, plus the per-parameter liveness checks. The filter is kept as a
    ratchet against the warning form being reintroduced.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=WildcardNoOpWarning)
        model = SEDModel.build(
            ssp_data=ssp_data,
            observation=Photometry.from_names(list(_UV_OPTICAL_BANDS)),
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "none"},
            agn={
                "type": "composable",
                "disc": {"type": disc_type, "all_params": FREE},
                "norm": "independent",
                "log_lbol": Uniform(10.0, 12.0),
            },
            redshift=Fixed(0.1),
        )
    free = model.spec.free_params
    assert "agn_log_mbh" in free, (
        "agn_log_mbh is not free via the agn.disc wildcard -- _AGN_PARTITION "
        "no longer maps it to 'agn.disc'?"
    )
    assert "agn_log_ledd" in free
    if disc_type == "kd18_agnfitter_warmindex":
        assert "agn_gamma_warm" in free

    params = model.spec.sample(jax.random.PRNGKey(0))
    grad = jax.grad(lambda p: jnp.sum(model.predict_photometry(p)))(params)
    names = ["agn_log_mbh", "agn_log_ledd"]
    if disc_type == "kd18_agnfitter_warmindex":
        names.append("agn_gamma_warm")
    for name in names:
        g = float(grad[name])
        assert jnp.isfinite(g), f"{name}: non-finite gradient {g}"
        assert g != 0.0, f"{name}: identically-zero gradient on a UV/optical band (dead free)"
