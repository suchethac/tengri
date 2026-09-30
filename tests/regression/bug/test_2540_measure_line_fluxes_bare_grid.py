# SPDX-License-Identifier: BSD-3-Clause
"""#2540: measure_line_fluxes must refuse on a grid with no nebular emission.

``predict_line_fluxes`` already refuses when the configured nebular backend
publishes no discrete line catalog (BakedIn / no backend). ``measure_line_fluxes``
instead measures the model's own rest-frame SED via continuum-subtraction, so it
silently "worked" on a bare-stellar grid paired with the BakedIn backend
(``neb={'type': 'ssp'}`` or the model's default), returning a small
stellar-continuum/absorption number indistinguishable from ``neb={'type':
'none'}`` -- with no indication the nebular emission it is meant to measure was
never there.

Independent review of the first fix found it gated on the wrong signal: the
guard tested ``backend is not None and not hasattr(backend,
'predict_nebular_line_luminosities')``, true for BOTH ``neb={'type': 'ssp'}``
(or the default) AND an explicit ``neb={'type': 'none'}`` -- both construct a
``BakedInBackend``, so ``hasattr`` alone cannot tell "the user asserted the
grid carries baked-in emission" apart from "the user explicitly asked for no
nebular emission at all". The gate now reads ``spec.nebular_mode`` /
``spec._nebular_explicit`` (the same condition ``_init_nebular`` uses to
decide whether to thread ``ssp_data`` to ``BakedInBackend`` in the first
place), so ``neb={'type': 'none'}`` never reaches the check.

The review also found the ``ssp_status == 'bare'`` raise branch inside this
guard unreachable on the ordinary build path: ``BakedInBackend.__init__``
(``baked_in.py``) already raises ``BakedInNebularBareError`` at
model-construction time for a bare grid whenever ``ssp_data`` is threaded to
it, which it always is for ``neb={'type': 'ssp'}`` and the omitted-``neb=``
default -- the two modes this guard applies to. That branch was removed
rather than kept untested; the build-time refusal is pinned directly below.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

import tengri
from tengri import DEFAULT, Fixed, Observation, Photometry
from tengri.components.nebular.baked_in import BakedInNebularBareError, BakedInNebularWarning
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation.line_flux_data import LineFluxData

pytestmark = pytest.mark.regression_bug

_HALPHA = tengri.observation.line_measurement.DESI_LINES[0]


def _synthetic_ssp(nebular: str) -> SSPData:
    """A minimal SSPData stamped with a known ``nebular`` status.

    Built the same way ``tests/contract/test_ssp_nebular_metadata.py`` does
    (an in-memory ``SSPData``, no file I/O), widened to 3 metallicity nodes:
    DSPS's ``_get_bin_edges`` needs at least 2 metallicity *intervals* to
    compute bin midpoints, so the 2-node grid that file uses for
    construction-only checks cannot run a full forward pass here.
    """
    return SSPData(
        ssp_wave=jnp.linspace(100.0, 20000.0, 100),
        ssp_flux=jnp.full((3, 6, 100), 1e-4),
        ssp_lg_age_gyr=jnp.linspace(-3.0, 1.0, 6),
        ssp_lgmet=jnp.array([-2.5, -1.8, -1.0]),
        ssp_mass_remaining=jnp.full((3, 6), 0.7),
        nebular=nebular,
    )


def _build(ssp, neb):
    lines = LineFluxData.from_dict({"Halpha": (1.2e-16, 0.1e-16)})
    obs = Observation(photometry=Photometry.from_names(["des_g", "des_r"]), line_fluxes=lines)
    return tengri.SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        redshift=Fixed(0.1),
        sfh={
            "type": "const",
            "all_params": Fixed(DEFAULT),
            "start_gyr": Fixed(1.0),
            "end_gyr": Fixed(0.0),
        },
        neb=neb,
    )


def test_measure_line_fluxes_refuses_when_backend_is_bakedin_on_unstamped_grid():
    """``fsps_prsc_miles_chabrier`` is 'unknown' (unstamped) status, not 'included'.

    The default shipped bare-stellar grid carries no nebular metadata stamp,
    so ``BakedInBackend`` (``neb={'type': 'ssp'}``) has no way to supply the
    emission its name implies. This must WARN, not silently succeed and not
    raise (the grid genuinely cannot be proven bare -- #2362).
    """
    ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=False)
    m = _build(ssp, {"type": "ssp"})
    params = dict(m.spec.sample(jax.random.PRNGKey(0)))
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.warns(UserWarning, match="contributes zero nebular flux of its own"):
            m.measure_line_fluxes(params, [_HALPHA])


def test_measure_line_fluxes_still_works_for_cue_backend():
    """A real additive backend (Cue) must keep working, unaffected."""
    ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=False)
    m = _build(ssp, {"type": "cue", "all_params": Fixed(DEFAULT)})
    params = dict(m.spec.sample(jax.random.PRNGKey(0)))
    result = m.measure_line_fluxes(params, [_HALPHA])
    assert result.shape == (1,)


def test_bare_stamped_grid_with_neb_ssp_refuses_at_build_naming_the_remedy():
    """A grid explicitly stamped 'bare' + ``neb={'type': 'ssp'}`` never reaches
    ``measure_line_fluxes`` at all: ``BakedInBackend.__init__`` refuses first."""
    ssp = _synthetic_ssp("bare")
    with pytest.raises(BakedInNebularBareError, match="bare-stellar SSP") as exc_info:
        _build(ssp, {"type": "ssp"})
    msg = str(exc_info.value)
    assert "Fix (one of)" in msg
    assert "neb={'type': 'cue'}" in msg or "neb={'type': 'cloudy_grid'}" in msg


def test_included_stamped_grid_with_neb_ssp_measures_without_a_grid_warning():
    """A grid explicitly stamped 'included' must measure cleanly, no grid-status warning."""
    ssp = _synthetic_ssp("included")
    m = _build(ssp, {"type": "ssp"})
    params = dict(m.spec.sample(jax.random.PRNGKey(0)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = m.measure_line_fluxes(params, [_HALPHA])
    grid_warnings = [w for w in caught if issubclass(w.category, BakedInNebularWarning)]
    assert grid_warnings == []
    assert result.shape == (1,)
    assert bool(jnp.isfinite(result).all())


@pytest.mark.parametrize(
    "ssp_factory",
    [
        pytest.param(lambda: _synthetic_ssp("bare"), id="bare-stamped-synthetic"),
        pytest.param(
            lambda: tengri.load_ssp("fsps_prsc_miles_chabrier", download=False),
            id="shipped-unstamped",
        ),
    ],
)
def test_neb_none_never_warns_or_refuses_regardless_of_grid_status(ssp_factory):
    """``neb={'type': 'none'}`` must never see this guard (#2540 review).

    It also constructs a ``BakedInBackend`` (indistinguishable from
    ``neb={'type': 'ssp'}`` by ``hasattr`` alone), but nothing about ``'none'``
    asserts the grid carries baked-in emission -- measuring the stellar
    continuum/absorption is this configuration's documented, intended use,
    on a grid stamped 'bare' or merely unstamped alike.
    """
    ssp = ssp_factory()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = _build(ssp, {"type": "none"})
        params = dict(m.spec.sample(jax.random.PRNGKey(0)))
        result = m.measure_line_fluxes(params, [_HALPHA])
    nebular_warnings = [w for w in caught if issubclass(w.category, BakedInNebularWarning)]
    assert nebular_warnings == [], f"neb={{'type': 'none'}} must never warn: {nebular_warnings}"
    assert result.shape == (1,)
    assert bool(jnp.isfinite(result).all())
