# SPDX-License-Identifier: BSD-3-Clause
"""#2693: Cue's [N/O] can follow metallicity like the grid backends' (opt-in).

Physics. In a photoionized HII region [N II] 6584 / H-beta scales with the
nitrogen abundance, and nitrogen in the ISM tracks oxygen: [N/O] falls with
O/H (primary floor at low O/H, secondary rise above 12+log(O/H) ~ 8.08;
Nicholls et al. 2017). The CLOUDY-grid backends embody that; Cue takes [N/O]
as an input that stays solar unless the user lowers it, so at 0.3 Z_sun its
[N II] 6584 / H-beta is ~2x the grid value. ``neb={'type': 'cue',
'nitrogen': 'nicholls17'}`` makes ``gas_logno`` the offset from the relation,
like ``neb_dno``; the default ``'absolute'`` leaves Cue untouched.

The model-level check compares [N II] 6584 / H-beta from a Cue model and a
shipped CloudyGrid model at the same SFH (hence Q_H) and log U = -2. The
agreement budget, a factor 1.25 either way, covers the measured Cue/grid
[N II]/H-alpha scatter (0.89-1.47 across log Z/Z_sun from -1 to 0; 1.06 in
[N II]/H-beta at Z_sun) and rejects the 2.25 of the default. At 0.1 Z_sun
Cue/grid is 0.78 in [N II]/[O II] but 3.7 in [N II]/H-beta: that residual is
[O II]/H-alpha (Cue 0.361, grid 0.0724), not nitrogen.
"""

from __future__ import annotations

import math
import os

import jax
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, Observation, Photometry, parse_groups
from tengri.observation.line_flux_data import LineFluxData

pytestmark = pytest.mark.regression_bug

_GRID = os.path.join(os.environ.get("TENGRI_DATA_DIR", "data"), "cloudy_grid_mist.h5")
_SCATTER_FACTOR = 1.25


def _build(neb_type: str, log_z_rel: float, **neb_extra):
    ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=False)
    obs = Observation(
        photometry=Photometry.from_names(["des_g"]),
        line_fluxes=LineFluxData.from_dict({"Hbeta": (1e-16, 1e-17), "NII_6584": (1e-16, 1e-17)}),
    )
    neb = {"type": neb_type, "all_params": Fixed(DEFAULT), "logZ_gas": log_z_rel, "logU": -2.0}
    neb.update(neb_extra)
    if neb_type == "cloudy":
        neb["grid"] = _GRID
    spec = parse_groups(
        redshift=Fixed(0.0),
        sfh={
            "type": "const",
            "all_params": Fixed(DEFAULT),
            "start_gyr": Fixed(1.0),
            "end_gyr": Fixed(0.0),
        },
        neb=neb,
    )
    model = tengri.SEDModel(spec, ssp, observation=obs)
    return model, dict(model.spec.sample(jax.random.PRNGKey(0))), obs


def _line_fluxes(neb_type: str, log_z_rel: float, **neb_extra) -> np.ndarray:
    model, params, obs = _build(neb_type, log_z_rel, **neb_extra)
    waves = np.asarray(obs.line_fluxes.wavelengths)
    return np.asarray(model.predict_line_fluxes(params, target_wavelengths=waves))


def _nii_over_hbeta(neb_type: str, log_z_rel: float, **neb_extra) -> float:
    flux = _line_fluxes(neb_type, log_z_rel, **neb_extra)
    return float(flux[1] / flux[0])


@pytest.fixture(scope="module")
def ratios():
    if not os.path.exists(_GRID):
        pytest.skip("shipped CloudyGrid file data/cloudy_grid_mist.h5 not present")
    out = {}
    for z in (0.0, math.log10(0.3)):
        for kind in ("cue", "cloudy"):
            extra = {"nitrogen": "nicholls17"} if kind == "cue" else {}
            out[(kind, z)] = _nii_over_hbeta(kind, z, **extra)
    print("[NII]6584/Hbeta  Cue(Zsun), grid(Zsun), Cue(0.3), grid(0.3):", *out.values())
    return out


def test_cue_nii_matches_grid_at_solar_scatter(ratios):
    """Baseline: the backend-to-backend scatter at Z_sun sits inside the budget."""
    r = ratios[("cue", 0.0)] / ratios[("cloudy", 0.0)]
    assert 1 / _SCATTER_FACTOR < r < _SCATTER_FACTOR, f"Cue/grid at Zsun = {r:.3f}"


def test_cue_nii_matches_grid_at_subsolar_metallicity(ratios):
    """At 0.3 Z_sun, nitrogen=nicholls17, Cue and the grid agree to the scatter budget (#2693)."""
    z = math.log10(0.3)
    r = ratios[("cue", z)] / ratios[("cloudy", z)]
    assert 1 / _SCATTER_FACTOR < r < _SCATTER_FACTOR, (
        f"Cue/grid [N II]/Hbeta at 0.3 Zsun = {r:.3f} (default 'absolute': 2.25)"
    )


def test_default_relation_is_two_regime_and_zero_at_solar():
    """Nicholls+2017 form: exactly 0 at Z_sun, flat primary floor, slope 1 at high O/H."""
    from tengri.components.nebular._default_nitrogen import default_nitrogen_offset

    assert float(default_nitrogen_offset(0.0)) == pytest.approx(0.0, abs=1e-12)
    lo = [float(default_nitrogen_offset(x)) for x in (-6.0, -5.0)]
    assert lo[0] == pytest.approx(lo[1], abs=0.01)  # primary floor
    hi = float(default_nitrogen_offset(1.0)) - float(default_nitrogen_offset(0.5))
    assert hi == pytest.approx(0.5, abs=0.02)  # secondary: [N/O] ∝ O/H


def test_trained_range_warning_uses_the_effective_nitrogen():
    """#2569 under a relation: the trained [N/O] range bounds offset + relation.

    At log Z/Z_sun = -1 the relation is -0.68 dex, so an offset prior
    [-0.5, 0.5] reaches [N/O] = -1.18, below Cue's trained floor of -1; the
    solar case (relation 0) with a tighter prior stays inside, and the
    message prints the effective range.
    """
    from tengri.components.grid_support import check_grid_support

    rel = {"cue_nitrogen": "nicholls17"}
    low_z = {"gas_logno": (-0.5, 0.5), "neb_logZ_gas": (-1.0, -1.0)}
    found = check_grid_support([("neb", "cue")], low_z, rel)
    assert [f[2] for f in found] == ["gas_logno"], found
    assert "effective range" in found[0][3] and "-1.1758" in found[0][3], found[0][3]

    solar = {"gas_logno": (-0.3, 0.3), "neb_logZ_gas": (0.0, 0.0)}
    assert check_grid_support([("neb", "cue")], solar, rel) == []


def test_absolute_mode_compares_gas_logno_directly():
    """Default mode: no shift, the prior is bounded by the absolute [-1, 0.73]."""
    from tengri.components.grid_support import check_grid_support

    cue = [("neb", "cue")]
    solar = {"neb_logZ_gas": (0.0, 0.0)}
    bad = {"gas_logno": (-1.5, -1.2), **solar}
    assert [f[2] for f in check_grid_support(cue, bad, {"cue_nitrogen": "absolute"})] == [
        "gas_logno"
    ]
    assert check_grid_support(cue, {"gas_logno": (-0.3, 0.3), "neb_logZ_gas": (-1.0, -1.0)}) == []


#: [H-beta, N II 6584] fluxes of Cue on origin/main with the default
#: ``gas_logno`` and log U = -2, same SFH: the default must reproduce them.
_MAIN_DEFAULT_FLUXES = {
    0.0: [73.90650561556737, 27.98730662176506],
    math.log10(0.3): [72.79523121746439, 19.117594787480527],
}


@pytest.mark.parametrize("log_z", sorted(_MAIN_DEFAULT_FLUXES))
def test_default_is_bit_identical_to_origin_main(log_z):
    """The default (nitrogen='absolute') leaves Cue exactly as it was."""
    flux = _line_fluxes("cue", log_z)
    np.testing.assert_allclose(flux, _MAIN_DEFAULT_FLUXES[log_z], rtol=1e-12, atol=0)


@pytest.mark.parametrize("log_z", [0.0, math.log10(0.3)])
@pytest.mark.parametrize("mode", ["absolute", "nicholls17"])
def test_log_no_property_is_the_effective_nitrogen(mode, log_z):
    """``log_no`` = gas_logno (absolute) or relation(gas_logz) + gas_logno (relation)."""
    from tengri.components.nebular._default_nitrogen import relation_offset

    model, params, _ = _build("cue", log_z, nitrogen=mode, gas_logno=-0.2)
    got = float(model.predict(params).properties["log_no"])
    want = -0.2 + (0.0 if mode == "absolute" else float(relation_offset("nicholls17", log_z)))
    assert got == pytest.approx(want, abs=1e-6)


def test_nitrogen_key_is_cue_only_and_round_trips():
    from tengri.config.exceptions import ConfigError

    with pytest.raises(ConfigError, match="cue"):
        parse_groups(redshift=Fixed(0.1), neb={"type": "cloudy", "nitrogen": "nicholls17"})
    with pytest.raises(ConfigError, match="nicholls17"):
        parse_groups(redshift=Fixed(0.1), neb={"type": "cue", "nitrogen": "bogus"})

    spec = parse_groups(redshift=Fixed(0.1), neb={"type": "cue", "nitrogen": "nicholls17"})
    assert spec.cue_nitrogen == "nicholls17"
    assert spec.to_groups()["neb"]["nitrogen"] == "nicholls17"
    default = parse_groups(redshift=Fixed(0.1), neb={"type": "cue"})
    assert default.cue_nitrogen == "absolute"
    assert "nitrogen" not in default.to_groups().get("neb", {})
