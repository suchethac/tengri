# SPDX-License-Identifier: BSD-3-Clause
r"""Exact-path vs LUT-path photometry parity, enumerated over the component registries (#2440).

``approx=WavePrecomp()`` is what every fit surface resolves to by default, so a component
the LUT gets wrong is wrong in every fit that uses it. Before this file, each surface had a
parity check for one or two hand-picked components -- ``calzetti`` and ``power_law`` among
22 attenuation laws, five of 19 dust-emission models, ``inoue`` among four IGM models -- and
nothing enumerated a registry. A law or model added tomorrow got no LUT check at all.

The first run of the enumeration found #2553: ``energy_balance_split``'s entire dust emission
was silently dropped on the LUT path, because the band-response homogeneity guard probed a
keyword that emitter does not read and certified the resulting zero spectrum as linear.

Every case below comes from a registry at collection time, and three things are asserted,
each closing a way a parity test passes vacuously:

* **The arms differ.** The LUT arm has ``wave_precomp`` set and the exact arm does not.
  Comparing one configuration with itself agrees to the last digit and proves nothing.
* **The component moves the bands.** Its own effect, against a model without it, must be
  large. A parity check on a band the component barely touches measures the stellar LUT.
* **Worst band, never a mean.** Quadrature error concentrates in the band the feature sits
  in; an average over bands reports "fine" while that band is wrong.

**Attenuation** -- every ``DUST_LAWS`` entry, UV to optical, at the typical optical depth
(``tau_bc`` 0.7, ``tau_diff`` 0.3). The residual is the #1122 sub-band quadrature's
truncation, and it converges roughly as 1/K**2 in the sub-band count, measured on the
steepest UV laws at heavy extinction (``tau_bc = tau_diff = 1``):

====================  =======  =======  =======  =======  ==================
law                   K = 5    K = 10   K = 20   K = 40   band transmission
====================  =======  =======  =======  =======  ==================
``narayanan_z``       2.180 %  0.528 %  0.129 %  0.037 %  0.33 %
``wd01_smcbar``       1.928 %  0.588 %  0.183 %  0.074 %  0.061 %
``calzetti``          0.031 %  0.027 %  0.002 %  0.001 %  2.3 %
====================  =======  =======  =======  =======  ==================

The percent-level cases are relative errors on bands that have lost 99.7-99.9 % of their
light. At the typical optical depth, measured across all 22 registered laws, the worst is
0.2104 % (``wd01_smcbar``, galex_fuv) at the default K = 5 and the median is ~0.04 %;
the tolerance below is set against that worst case.

**IGM** -- every ``IGM_TRANSMISSION_MODELS`` entry with ``igm_fold="exact"``, at z = 7. The
redshift is chosen, not incidental: ``asada25`` is identical to ``inoue14`` at z <= 3 by
construction (it modifies only the reionization era), so a z = 3 check would run the
``inoue14`` code twice. At z = 7 the four models are distinct, which
``test_igm_models_are_distinct`` asserts rather than assumes. The exact fold agrees with
the integrator to ~1e-14. The explicit ``igm_fold="node"`` is deliberately not held to
parity here: it is a documented approximation, off by 85-107 % in these bands at z = 7,
which is why the default became ``"auto"`` (#2445).

**Covered elsewhere, so not repeated here.**

* Dust emission -- every standalone emitter, exact against LUT:
  ``tests/regression/bug/test_bug_2553_band_response_probe_reaches_budget.py``.
* Stellar photometry through the free-redshift z-table:
  ``tests/regression/test_issue_1134_ztable_accuracy.py``.
* Spectra: ``tests/contract/test_spectrum_lut.py``.
* The dusty-nebular channels are the #2387 scope and carry their own parity contract
  with that change.

All fixtures use the bare-stellar ``ssp_data_fsps`` grid with ``neb={'type': 'none'}`` and
every parameter pinned, so ``predict_photometry({})`` needs no parameter values.
"""

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, WavePrecomp
from tengri.components.dust.attenuation import DUST_LAWS
from tengri.components.igm.igm import IGM_TRANSMISSION_MODELS

pytestmark = pytest.mark.contract

ATTENUATION_LAWS = sorted(DUST_LAWS)
IGM_MODELS = sorted(IGM_TRANSMISSION_MODELS)

_UV_OPTICAL = ["galex_fuv", "galex_nuv", "sdss_u", "sdss_g", "sdss_r", "sdss_i"]
_ATTENUATION_Z = 0.1
_TYPICAL_TAU = {"tau_bc": 0.7, "tau_diff": 0.3}

#: Worst-band |LUT / exact - 1| for any attenuation law at the typical optical depth.
#: The measured worst case over all 22 laws is 0.2104 % (``wd01_smcbar``, galex_fuv,
#: K = 5), so this is ~2.4x headroom over it -- and well under the +8.4 % that the pre-#1122
#: effective-wavelength path reached in NUV at z = 1
#: (``tests/regression/bug/test_bug_joint_precomp_dust_quadrature.py``).
_ATTENUATION_TOL = 5e-3

#: A law must move its worst-affected band by at least this much, or the parity check
#: is measuring the stellar LUT. Measured over all 22 laws at the typical optical depth:
#: smallest 80.42 % (``power_law`` / ``vw07_diff``), so the floor is ~8x below it.
_ATTENUATION_MIN_EFFECT = 0.1

#: The Lyman series falls in r, i and z at this redshift.
_IGM_Z = 7.0
_IGM_BANDS = ["sdss_r", "sdss_i", "sdss_z"]

#: Measured worst case 3.1e-14 (``asada25`` and ``inoue14``): the exact fold carries
#: the transmission inside the bandpass integral, so it is exact up to roundoff.
_IGM_EXACT_TOL = 1e-10

#: Measured 100 % for every model at z = 7: the IGM removes nearly all of r and i.
_IGM_MIN_EFFECT = 0.5


def _build(ssp, bands, z, approx, **groups):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(bands)),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Fixed(z),
        approx=approx,
        **groups,
    )


def _photometry(model):
    free = list(model.spec.free_params)
    assert not free, f"every parameter must be pinned for predict_photometry({{}}); free: {free}"
    return np.asarray(model.predict_photometry({}), dtype=float)


def _has_wave_lut(model):
    state = getattr(model, "approx", None)
    return bool(state is not None and getattr(state, "wave_precomp", False))


def _parity(ssp, bands, z, groups, lut_approx):
    """Return (exact photometry, LUT photometry), asserting the two arms differ."""
    exact_model = _build(ssp, bands, z, None, **groups)
    lut_model = _build(ssp, bands, z, lut_approx, **groups)
    assert not _has_wave_lut(exact_model), "the exact arm carries a WavePrecomp LUT"
    assert _has_wave_lut(lut_model), (
        "the LUT arm carries no WavePrecomp LUT, so both arms are the exact path and "
        "their agreement proves nothing"
    )
    exact, lut = _photometry(exact_model), _photometry(lut_model)
    assert np.all(np.isfinite(exact)) and np.all(np.isfinite(lut)), (
        f"non-finite photometry: exact {exact}, LUT {lut}"
    )
    assert np.all(exact > 0.0), f"exact photometry must be positive to take ratios: {exact}"
    return exact, lut


def _worst(values, bands):
    i = int(np.argmax(values))
    return float(values[i]), bands[i]


@pytest.fixture(scope="module")
def unattenuated(ssp_data_fsps):
    return _photometry(
        _build(ssp_data_fsps, _UV_OPTICAL, _ATTENUATION_Z, None, dust_attenuation={"type": "none"})
    )


@pytest.fixture(scope="module")
def igm_free(ssp_data_fsps):
    return _photometry(_build(ssp_data_fsps, _IGM_BANDS, _IGM_Z, None, igm={"type": "none"}))


def test_the_registries_are_populated():
    """An empty registry would parametrize to zero cases and pass without running."""
    assert ATTENUATION_LAWS, "DUST_LAWS enumerated to nothing"
    assert "calzetti" in ATTENUATION_LAWS, f"calzetti missing from {ATTENUATION_LAWS}"
    assert IGM_MODELS, "IGM_TRANSMISSION_MODELS enumerated to nothing"
    assert "inoue14" in IGM_MODELS, f"inoue14 missing from {IGM_MODELS}"


@pytest.mark.parametrize("law", ATTENUATION_LAWS)
def test_attenuation_law_lut_matches_exact(ssp_data_fsps, unattenuated, law):
    """Every attenuation law's sub-band quadrature agrees with the exact screen."""
    groups = {
        "dust_attenuation": {
            "type": "two_component",
            "law": law,
            **_TYPICAL_TAU,
            "all_params": Fixed(DEFAULT),
        }
    }
    exact, lut = _parity(ssp_data_fsps, _UV_OPTICAL, _ATTENUATION_Z, groups, WavePrecomp())

    effect, effect_band = _worst(np.abs(exact / unattenuated - 1.0), _UV_OPTICAL)
    assert effect > _ATTENUATION_MIN_EFFECT, (
        f"{law}: attenuates its worst band by only {effect:.2%} ({effect_band}), so this "
        "comparison would be measuring the stellar LUT rather than the law"
    )

    errors = np.abs(lut / exact - 1.0)
    worst, band = _worst(errors, _UV_OPTICAL)
    assert worst < _ATTENUATION_TOL, (
        f"{law}: LUT photometry is {worst:.4%} off exact in {band} "
        f"(tolerance {_ATTENUATION_TOL:.2%}); per band "
        + ", ".join(f"{b} {e:.4%}" for b, e in zip(_UV_OPTICAL, errors))
    )


@pytest.mark.parametrize("model", IGM_MODELS)
def test_igm_exact_fold_matches_exact(ssp_data_fsps, igm_free, model):
    """Every IGM model's exact fold agrees with the wavelength-grid integrator."""
    groups = {"igm": {"type": model}}
    exact, lut = _parity(ssp_data_fsps, _IGM_BANDS, _IGM_Z, groups, WavePrecomp(igm_fold="exact"))

    effect, effect_band = _worst(np.abs(exact / igm_free - 1.0), _IGM_BANDS)
    assert effect > _IGM_MIN_EFFECT, (
        f"{model}: changes its worst band by only {effect:.2%} ({effect_band}) at z = {_IGM_Z}, "
        "so the Lyman series is not inside these bands"
    )

    errors = np.abs(lut / exact - 1.0)
    worst, band = _worst(errors, _IGM_BANDS)
    assert worst < _IGM_EXACT_TOL, (
        f"{model}: igm_fold='exact' is {worst:.3e} off the integrator in {band} "
        f"(tolerance {_IGM_EXACT_TOL:.0e})"
    )


def test_igm_models_are_distinct(ssp_data_fsps):
    """At the test redshift no two IGM models may give the same photometry.

    Otherwise a per-model parity check runs one code path several times under
    different names -- which is exactly what a z = 3 check does for ``asada25``.
    """
    photometry = {
        model: _photometry(_build(ssp_data_fsps, _IGM_BANDS, _IGM_Z, None, igm={"type": model}))
        for model in IGM_MODELS
    }
    identical = [
        (a, b)
        for i, a in enumerate(IGM_MODELS)
        for b in IGM_MODELS[i + 1 :]
        if np.allclose(photometry[a], photometry[b], rtol=1e-12, atol=0.0)
    ]
    assert not identical, (
        f"IGM models give identical photometry at z = {_IGM_Z}: {identical}. Their parity "
        "checks exercise one code path; raise the test redshift until they differ."
    )
