# SPDX-License-Identifier: BSD-3-Clause
"""#3526: FeaturePrecomp refuses a history whose ionizing shape departs from its stand-in.

The per-Q_H line table is built on one constant-SFR stand-in history. Reconstruction
multiplies nion(actual SFH) by that table, so it assumes the per-photon line
luminosity does not depend on the SFH. It does on the MIST C3K default: old
post-AGB populations harden the ionizing spectrum, and a quickly faded history's
[OIII] came out 3.2x too low, silently.

The guard is a build-time probe that needs no history: it measures the per-photon
lines for the stand-in and for an old-only population (ages > 1 Gyr) at the grid's
corners, and refuses when they differ by more than the tolerance. An explicit
``FeaturePrecomp`` raises :class:`NebularShapeDriftError`. The ``approx="auto"``
policy catches exactly that error, warns with the line and ratio, and falls back to
the exact line path (``WavePrecomp`` alone); photometry is not touched.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytestmark = [pytest.mark.regression_bug]

_N_T = 40
_T = np.linspace(0.0, 12.5, _N_T)


def _history(tau=2.0):
    sfr = (_T / tau) * np.exp(-_T / tau)
    sfr = sfr * (40.0 / sfr.max())
    cum = np.cumsum(sfr)
    cum /= cum[-1]
    return sfr, np.clip(-2.6 + 2.7 * cum, -2.1, 0.4)


@pytest.fixture(scope="module")
def phot_and_line_obs():
    import jax.numpy as jnp

    from tengri.observation import Observation
    from tengri.observation.line_list import LineList
    from tengri.observation.photometry_config import Photometry

    try:
        from tengri.observation.line_flux_data import LineFluxData
    except ImportError:  # pragma: no cover - layout drift
        from tengri.observation.data import LineFluxData

    wanted = ["Hbeta", "OIII_5007", "Halpha", "NII_6584"]
    cat = LineList.default_optical()
    waves = jnp.asarray([float(w) for n, w in zip(cat.names, cat.wavelengths) if n in wanted])
    names = tuple(n for n in cat.names if n in wanted)
    f = jnp.ones(len(names)) * 1e-16
    obs = Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]),
        line_fluxes=LineFluxData(names=names, fluxes=f, errors=f * 0.1, wavelengths=waves),
    )
    return obs, waves


def _build(ssp, obs, *, approx, sfh=None):
    from tengri import DEFAULT, SEDModel
    from tengri.parameters.priors import Fixed, Uniform

    kwargs = {}
    if sfh is not None:
        kwargs["sfh"] = sfh
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp,
            observation=obs,
            met={"logzsol": Uniform(-2.0, 0.4)},
            neb={
                "type": "cue",
                "all_params": Fixed(DEFAULT),
                "logU": Uniform(-4.0, -1.0),
                "logZ_gas": Uniform(-1.5, 0.3),
            },
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.1),
            approx=approx,
            **kwargs,
        )


def _tabulated_params(tau):
    import jax.numpy as jnp

    sfr, _ = _history(tau)
    return {
        "neb_logU": jnp.asarray(-2.5),
        "neb_logZ_gas": jnp.asarray(-0.2),
        "met_logzsol": jnp.asarray(-0.3),
        "sfh_t_gyr": jnp.asarray(_T),
        "sfh_sfr": jnp.asarray(sfr),
    }


# ── MIST C3K default: the guard refuses ─────────────────────────────


def test_explicit_feature_precomp_raises_on_the_mist_default(phot_and_line_obs):
    """Explicit FeaturePrecomp on a tabulated SFH, shape-sensitive grid: refused at build.

    The message must name the line, the measured ratio, the cause and the fix, so the
    refusal reads as a measurement and not as caution.
    """
    import tengri
    from tengri import FeaturePrecomp, WavePrecomp
    from tengri.components.nebular.nebular_grid_precompute import NebularShapeDriftError

    obs, _ = phot_and_line_obs
    with pytest.raises(NebularShapeDriftError) as exc:
        _build(
            tengri.load_ssp(),
            obs,
            approx=(WavePrecomp(), FeaturePrecomp()),
            sfh={"type": "table"},
        )
    message = str(exc.value)
    assert "#3526" in message
    assert "constant-SFR stand-in" in message
    assert "use the exact Cue path" in message
    assert exc.value.departure > 0.05
    assert exc.value.ratio > 1.5
    assert exc.value.line


def test_the_refusal_is_a_value_error_so_existing_handlers_still_catch_it(phot_and_line_obs):
    """A subclass of ValueError: the #1718 refusal's contract, unchanged for callers."""
    from tengri.components.nebular.nebular_grid_precompute import NebularShapeDriftError

    assert issubclass(NebularShapeDriftError, ValueError)


# ── Padova: the same history builds, agrees within 1%, and does not warn ──


def test_padova_tabulated_history_builds_and_agrees_with_no_warning(phot_and_line_obs):
    import jax.numpy as jnp

    import tengri
    from tengri import FeaturePrecomp, WavePrecomp

    obs, waves = phot_and_line_obs
    ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        exact = _build(ssp, obs, approx=(WavePrecomp(),), sfh={"type": "table"})
        lut = _build(
            ssp,
            obs,
            approx=(WavePrecomp(), FeaturePrecomp()),
            sfh={"type": "table"},
        )
    assert not [w for w in seen if "#3526" in str(w.message)], "no shape warning on Padova"

    worst = 0.0
    for tau in (0.5, 2.0, 6.0):
        p = _tabulated_params(tau)
        a = np.asarray(exact.predict_line_fluxes(p, target_wavelengths=waves))
        b = np.asarray(lut.predict_line_fluxes(p, target_wavelengths=waves))
        worst = max(worst, float(np.abs(b / a - 1).max()))
    assert worst < 0.01, f"Padova LUT departs from exact by {worst:.2%}"
    assert jnp.isfinite(worst)


# ── SFH-free model: a parametric SFH has no history to probe ───────


def test_a_parametric_sfh_model_builds_with_feature_precomp_on_the_default_grid(
    phot_and_line_obs,
):
    """No tabulated history means the probe does not run, so the build proceeds.

    Recorded as a limit: the probe needs a tabulated stellar SFH to evaluate its
    quenched reference history, so a parametric SFH on a shape-sensitive grid is not
    refused here. A fixed parametric SFH is built on its own history (no drift); free
    parametric SFHs are the open case, see the report.
    """
    import tengri
    from tengri import FeaturePrecomp, WavePrecomp

    obs, _ = phot_and_line_obs
    _build(tengri.load_ssp(), obs, approx=(WavePrecomp(), FeaturePrecomp()))


# ── approx="auto" through the fitter: fall back with a specific warning ──


def test_auto_policy_falls_back_to_the_exact_line_path_on_the_mist_default():
    """Auto keeps WavePrecomp, warns with the line, the ratio and #3526, and matches exact.

    The auto policy attaches the Cue table to a photometry-only fit (the 2026-11-06
    pair); a line-flux fit on Cue is already served exactly by ``_feature_precomp_can_pay``
    (#2377) and never reaches the shape probe. So this uses a photometry-only observation,
    the path a user gets by default. The fit must not raise.
    """
    import tengri
    from tengri import WavePrecomp
    from tengri.inference.fitter import Fitter
    from tengri.observation import Observation
    from tengri.observation.photometry_config import Photometry

    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))
    ssp = tengri.load_ssp()
    exact = _build(ssp, obs, approx=(WavePrecomp(),), sfh={"type": "table"})
    # Built with no approx=, the default a user gets: the auto policy then tries the
    # (WavePrecomp, FeaturePrecomp) pair on the photometry path (a model already carrying
    # WavePrecomp keeps its raw answer, see _photometry_grid_engages).
    auto_model = _build(ssp, obs, approx=None, sfh={"type": "table"})
    phot = np.ones(3) * 1e-16
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        fitter = Fitter(
            auto_model,
            data=phot,
            noise=phot * 0.1,
            data_type="photometry",
        )
    shape = [w for w in seen if "#3526" in str(w.message)]
    assert shape, "the auto fallback must warn specifically"
    text = str(shape[0].message)
    assert "Line " in text and "x the stand-in" in text, "name the line and the ratio"
    assert not fitter.model.approx.feature_precomp, "auto must not keep the table"

    waves = np.asarray([5006.8, 6562.8, 6583.5, 4861.3])
    p = _tabulated_params(0.5)
    a = np.asarray(exact.predict_line_fluxes(p, target_wavelengths=waves))
    b = np.asarray(fitter.model.predict_line_fluxes(p, target_wavelengths=waves))
    np.testing.assert_allclose(b, a, rtol=1e-6)
