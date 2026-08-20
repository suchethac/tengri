# SPDX-License-Identifier: BSD-3-Clause
r"""The galaxy's metallicity reaches the Lehmer+2016 HMXB term (#1755).

``xray_model.py`` documented five "opportunistic cross-component reads", one of
them ``metallicity_z``. Nothing in ``src/tengri`` has ever published a key by
that name, it was not declared in ``optional_inputs`` — so the pipeline
validator's unit check could not fire either — and it was not a fittable
parameter. The ``.get(..., 0.02)`` fallback was therefore the *only* value the
HMXB quartic could ever see, and ``component.py`` did not pass metallicity to
``xray_total`` at all.

Measured before the fix: ``sed_xray`` was **bit-identical** between
``met_logzsol = -1.0`` and ``+0.3``, a range over which Lehmer+2016 Eq. 15
implies an 18x change in L_HMXB.

Same failure mode as #1706, one argument over — there the component simply never
passed ``det_hmxb``/``det_lmxb``, so both were free parameters nothing read.

The second half of #1755: the fallback itself was ``0.02``, a second definition
of "solar" inside a project whose canonical value is
:data:`~tengri.utils.physics_constants.Z_SUN` = 0.0142 (Asplund 2009, matching
MIST). At the two values the same relation gives 1.78e39 and 3.22e39 — a factor
1.8 in the stellar baseline against which an AGN corona is measured.

Lehmer et al. 2016, ApJ 825, 7, Eq. 15:

.. math::

    \log(L_X^{\mathrm{HMXB}}(2\text{–}10\,\mathrm{keV})/\mathrm{SFR}) =
        40.28 - 62.12Z + 569.44Z^2 - 1833.80Z^3 + 1968.33Z^4
"""

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel
from tengri.components.xray.xray import (
    metallicity_from_history,
    xray_total_lopez24_terms,
    xray_xrb,
    xray_xrb_terms,
)
from tengri.utils.physics_constants import Z_SUN

pytestmark = pytest.mark.regression_bug

_LEHMER_COEFFS = (40.28, -62.12, 569.44, -1833.80, 1968.33)

#: Wide enough that the Lehmer quartic separates the endpoints by ~18x.
_LOGZSOL_LO = -1.0
_LOGZSOL_HI = 0.3


def _lehmer_l_over_sfr(z: float) -> float:
    """L_X^HMXB(2-10 keV) per unit SFR from Eq. 15. [erg/s/(Msun/yr)]"""
    return 10.0 ** sum(c * z**i for i, c in enumerate(_LEHMER_COEFFS))


# ── 1. The reduction helper ───────────────────────────────────────


def test_absent_history_falls_back_to_the_project_solar():
    """No stellar component published a history -> Z_SUN, not 0.02.

    The bare-Protocol path signals absence with ``None``; ``SEDModelComponent``
    substitutes a 0-d ``jnp.asarray(0.0)`` for a declared-but-unpublished
    optional input. Both must read as absent — ``10**0.0`` is Z = 1, seventy
    times solar, and the quartic would evaluate it without complaint.
    """
    assert metallicity_from_history(None) == pytest.approx(Z_SUN)
    assert metallicity_from_history(jnp.asarray(0.0)) == pytest.approx(Z_SUN)


def test_a_published_history_is_read_at_its_present_day_bin():
    """Index 0, because HMXBs trace the instantaneous SFR.

    A mass-weighted mean over all ages would be the wrong reduction: HMXBs are
    young systems (< 100 Myr), so the metallicity that sets their luminosity is
    the one the currently-forming population was born with.
    """
    history = jnp.array([-1.5, -2.0, -2.5, -3.0])  # absolute log10(Z), present first
    assert float(metallicity_from_history(history)) == pytest.approx(10.0**-1.5)


# ── 2. The default stopped being a second "solar" ─────────────────


@pytest.mark.parametrize(
    "fn", [xray_xrb, xray_xrb_terms, xray_total_lopez24_terms], ids=lambda f: f.__name__
)
def test_the_signature_default_is_the_project_solar(fn):
    """No X-ray entry point may carry its own idea of solar metallicity.

    ``metallicity_z=0.02`` contradicted ``Z_SUN`` = 0.0142 while the docstring
    beside it said "(solar)". Read off the constant so the two cannot drift —
    the same rule ``LOG10_ZSUN`` follows by being computed from ``Z_SUN``.
    """
    default = inspect.signature(fn).parameters["metallicity_z"].default
    assert default == pytest.approx(Z_SUN), (
        f"{fn.__name__} defaults metallicity_z to {default}, not Z_SUN={Z_SUN}. "
        "A second solar convention inside one codebase is #1755."
    )


def test_lopez24_actually_threads_metallicity_rather_than_swallowing_it():
    """``**_kwargs`` on the lopez24 entry points would accept and ignore it.

    Before #1755 ``xray_total_lopez24_terms`` had no ``metallicity_z``
    parameter, only ``**_kwargs``, so passing one raised nothing and changed
    nothing — its HMXB term used the signature default of the ``xray_xrb_terms``
    it called. A silent no-op kwarg is worse than a rejected one.
    """
    wave = jnp.linspace(1.24, 6.2, 200)
    lo = xray_total_lopez24_terms(wave, sfr=1.0, stellar_mass=0.0, metallicity_z=0.004)["hmxb"]
    hi = xray_total_lopez24_terms(wave, sfr=1.0, stellar_mass=0.0, metallicity_z=0.030)["hmxb"]

    assert not np.allclose(np.asarray(lo), np.asarray(hi)), (
        "lopez24 HMXB is unchanged by metallicity — the kwarg is being swallowed by **_kwargs."
    )
    ratio = float(jnp.max(lo) / jnp.max(hi))
    assert ratio == pytest.approx(_lehmer_l_over_sfr(0.004) / _lehmer_l_over_sfr(0.030), rel=1e-3)


def test_the_low_level_default_reproduces_eq_15_at_z_sun():
    """A bare ``xray_xrb`` call integrates to Eq. 15 evaluated at Z_SUN.

    Pins the number a caller gets with no metallicity argument: 3.22e39, not
    the 1.78e39 of the old 0.02 default and not the 2.6e39 the docstring once
    claimed (Grimm+2003/Mineo+2012 genuinely differ by ~30-45% in this band).
    """
    wave = jnp.linspace(1.24, 6.20, 500)  # 2-10 keV
    l_nu = np.asarray(xray_xrb(wave, sfr=1.0, stellar_mass=0.0))
    nu = 2.99792458e18 / np.asarray(wave)
    l_band = abs(np.trapezoid(l_nu[::-1], nu[::-1]))

    np.testing.assert_allclose(l_band, _lehmer_l_over_sfr(Z_SUN), rtol=0.10)


# ── 3. It reaches the forward model ───────────────────────────────
#
# The part that actually failed. A fix that never engages on the model path
# reads exactly like a working one — the #1748 / #1770 lesson.


def _sed_xray(ssp, obs, logzsol, xray_type):
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "delayed", "all_params": FIXED},
        dust_attenuation={"law": "power_law", "all_params": FIXED},
        met={"logzsol": Fixed(logzsol)},
        xray={"type": xray_type, "all_params": FIXED},
        redshift=Fixed(0.05),
    )
    return np.asarray(model.predict_state({}).derived["sed_xray"])


@pytest.fixture(scope="module")
def obs():
    import tengri

    return tengri.Photometry.from_names(("sdss_g", "sdss_r", "sdss_i"))


@pytest.mark.parametrize("xray_type", ["xray_aird", "yang20"])
def test_sed_xray_responds_to_the_fitted_metallicity(ssp_data_wne, obs, xray_type):
    """The regression proper: ``sed_xray`` was bit-identical across 18x of Z.

    Both X-ray components are covered, because they reach the metallicity by
    different routes — ``yang20`` through ``emitter_inputs``, ``xray_aird``
    through the ``SEDModelComponent`` optional-input machinery — and each was
    separately blind to it.
    """
    lo = _sed_xray(ssp_data_wne, obs, _LOGZSOL_LO, xray_type)
    hi = _sed_xray(ssp_data_wne, obs, _LOGZSOL_HI, xray_type)

    band = np.abs(lo) > 0
    assert band.sum() > 0, "no nonzero X-ray bins — the probe is not exercising the component"

    assert not np.array_equal(lo, hi), (
        f"sed_xray is bit-identical for {xray_type} across met_logzsol "
        f"{_LOGZSOL_LO} -> {_LOGZSOL_HI}. The metallicity is not reaching the "
        "Lehmer+2016 HMXB term (#1755)."
    )

    # Direction: metal-poor galaxies host *more* luminous HMXBs, so the low-Z
    # SED must be the brighter one everywhere the X-ray band is populated.
    assert np.all(hi[band] < lo[band]), (
        "raising the metallicity did not lower the X-ray luminosity — the "
        "Lehmer+2016 relation is monotonically decreasing over this range."
    )


def test_the_component_declares_the_key_it_reads(ssp_data_wne, obs):
    """Declared, so the validator can check units on it.

    ``optional_inputs`` is what lets the pipeline validator catch a publisher
    rename or a unit drift. ``metallicity_z`` was read but never declared, so
    the one guard that could have caught this was inert (ADR-0004, Phase B).
    """
    from tengri.components.xray.component import XRaySEDComponent
    from tengri.components.xray.xray_model import XRayAirdSEDComponent

    for cls in (XRaySEDComponent, XRayAirdSEDComponent):
        declared = {k.name for k in cls().optional_inputs()}
        assert "log_metallicity_history" in declared, (
            f"{cls.__name__} reads log_metallicity_history without declaring it; "
            "an undeclared read is invisible to the pipeline validator."
        )
