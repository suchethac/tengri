# SPDX-License-Identifier: BSD-3-Clause
"""Finite-difference gradient tests for the nebular backends.

Each row differentiates a backend's summed rest-frame SED with respect to one
physical parameter and checks JAX autodiff against a central finite difference.

Every test here was skipped on every machine
--------------------------------------------

``_DATA_DIR`` was ``Path(__file__).resolve().parents[2] / "data"``. This file
sits at ``tests/physics/gradients/``, so ``parents[2]`` is ``tests/`` and all
four guards resolved ``tests/data/`` -- a directory that has never existed.
Eight data-gated tests skipped unconditionally, CI included.

Two of them wanted grids that are **tracked in git** and are therefore present
on every runner: ``data/cue_weights.npz`` and ``data/mappings_templates.h5``.
Both are real, working gradient tests. Neither had ever executed. The paths now
come from :mod:`tests._data_skip`, which computes the root once at a fixed
depth.

FD == AD == 0 is not a passing gradient test
--------------------------------------------

``assert_allclose(0.0, 0.0, rtol=5e-2)`` succeeds against any implementation,
including one that returns a constant. That was not hypothetical here -- it is
what the mock SSP this file used to build actually produced. Measured on that
mock, summing ``predict_nebular_sed``:

======= ================ ================================================
backend total            d(total)/d(logU)
======= ================ ================================================
Cue     ``1.382040e-15`` ``0.0`` -- and *also* flat in ``ssp_weights`` and
                         in ``log_z``: the output did not move on any axis
CB19    ``1.555884e+45`` ``NaN``
======= ================ ================================================

Cue's frozen ``1e-15`` is the failure mode ``tests/conftest.py`` documents on
``ssp_data_bc03``: Cue's network needs a *bare-stellar* SSP with
log10(Q_H) >~ 45, and under-predicts by many dex when fed anything softer. The
mock's near-zero ionizing flux is exactly that case, which is also why the old
fixture had to set ``TENGRI_ALLOW_WNE_CUE`` and swallow a ``CueWNESSPWarning``
to construct the backend at all. On the real bare-stellar SSP no warning is
raised, the escape hatch is unnecessary, and the same test has a log
sensitivity of 0.05-0.67 with FD and AD agreeing to five significant figures.

So the mock did not merely weaken these tests -- it removed the signal they
differentiate along, and then the zero it produced was compared against
itself. Every SSP-driven row now builds on the real ``ssp_data_fsps`` fixture.

The non-vacuity floor
---------------------

Every row asserts a log sensitivity ``|x * (dF/dx) / F| > _MIN_LOG_SENS``
*before* comparing FD to AD, so a zeroed, detached or clamped gradient fails
loudly instead of matching a zero FD.

It also catches a degenerate grid, which is not hypothetical either. A
``data/cb19_templates.h5`` built by something other than
``scripts/download_cb19_templates.py`` can be a constant fill: one such file
held 332,640 cells with 9 distinct values, varying *only* along the line axis
-- every physical axis, ``log_U`` included, was bit-flat, and its ``log_OH``
and ``log_CO`` axes were evenly spaced where the real 3MdB grid's are not.
Against that file the CB19 row fails here, naming the possibility, instead of
passing 0 against 0. Neither this grid nor the CLOUDY one is tracked in git or
fetched by CI, so both rows skip there.

Seven ``except Exception: pytest.skip(...)`` handlers are gone
--------------------------------------------------------------

Each wrapped the call under test, so any regression in a backend reported as a
skip -- the #1615 failure mode, and this file held seven of the suite's worst
instances. ``tests/contract/test_broad_except_into_skip_does_not_spread.py``
listed all seven as unjudgeable, on the reasoning that "which exception they
would raise on a machine that has the grids cannot be observed from here."
That was true only because of the path bug above. With the paths fixed the
calls are observable, and none of them raises.

The CLOUDY rows remain unverified
---------------------------------

``data/cloudy_grid_mist.h5`` is not tracked, so those rows skip locally and in
CI alike; they were switched to the real SSP with the rest for consistency and
cannot be exercised here.

Gradient convention: dL/d_logU for the photoionization backends, dL/d_velocity
for the MAPPINGS V shock grid.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._data_skip import (
    CB19_TEMPLATES,
    CLOUDY_GRID_MIST,
    CUE_WEIGHTS,
    requires_cb19,
    requires_cloudy_grid,
    requires_cue_weights,
    requires_mappings,
)

pytestmark = pytest.mark.gradient

#: FD/AD agreement. Central differences on a piecewise-linear interpolant are
#: only this good away from the kinks, hence the interior-of-cell probe points.
_FD_RTOL = 5e-2

#: Non-vacuity floor on |x * dF/dx / F|. The measured values are three to five
#: orders of magnitude above it (Cue 0.30, MAPPINGS 0.83), so this separates
#: "flat" from "weakly curved" without pinning any physics.
_MIN_LOG_SENS = 1e-3


def _fd_grad(fn, x, eps):
    """Central finite difference of a scalar function."""
    return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)


def _check_gradient(fn, x0, eps, label):
    """Assert ``fn`` responds to ``x0`` at all, then that AD matches FD there."""
    x = jnp.array(x0)
    f0 = float(fn(x))
    ad = float(jax.grad(fn)(x))

    assert np.isfinite(f0) and f0 != 0.0, f"{label}: total is {f0}, nothing to differentiate"
    assert np.isfinite(ad), (
        f"{label}: autodiff returned {ad} at x={x0}. A finite forward value with a "
        f"non-finite gradient is the jnp.where gradient leak -- the unsafe branch is "
        f"still evaluated under grad even when the primal selects the safe one."
    )

    sens = abs(x0 * ad / f0)
    assert sens > _MIN_LOG_SENS, (
        f"{label}: log sensitivity |x dF/dx / F| = {sens:.3e} at x={x0} is below "
        f"{_MIN_LOG_SENS:.0e} -- the total does not respond to this parameter, so "
        f"comparing FD to AD compares zero with zero and would pass against any "
        f"implementation. Either the parameter is not wired through, or the "
        f"template grid is constant along this axis (a placeholder build)."
    )

    fd = float(_fd_grad(fn, x, eps))
    np.testing.assert_allclose(ad, fd, rtol=_FD_RTOL, err_msg=f"{label}: FD/AD mismatch at {x0}")


# ── Backends, built on the real bare-stellar SSP ──────────────────


@pytest.fixture(scope="module")
def neb_inputs(ssp_data_fsps):
    """Rest-frame wavelength grid, age grid and a young-burst weighting.

    The mass sits in the youngest ages so log10(Q_H) is high enough for the
    photoionization backends to produce a signal; a flat weighting across 93
    ages dilutes the ionizing population until the logU response vanishes.
    """
    ages_yr = jnp.array(np.asarray(ssp_data_fsps.ssp_lg_age_gyr) + 9.0)
    weights = jnp.where(jnp.arange(ages_yr.shape[0]) < 5, 1e8, 0.0)
    return jnp.array(ssp_data_fsps.ssp_wave), weights, ages_yr


@pytest.fixture(scope="module")
def cloudy_linear_backend(ssp_data_fsps):
    from tengri.components.nebular import CloudyGridBackend

    return CloudyGridBackend(str(CLOUDY_GRID_MIST), ssp_data_fsps)


@pytest.fixture(scope="module")
def cloudy_triweight_backend(ssp_data_fsps):
    from tengri.components.nebular import CloudyGridBackend

    return CloudyGridBackend(str(CLOUDY_GRID_MIST), ssp_data_fsps, grid_interp="triweight")


@pytest.fixture(scope="module")
def cue_backend(ssp_data_fsps):
    from tengri.components.nebular import CueBackend

    return CueBackend(str(CUE_WEIGHTS), ssp_data=ssp_data_fsps)


@pytest.fixture(scope="module")
def cb19_backend(ssp_data_fsps):
    from tengri.components.nebular.cloudy_cb19 import CB19Backend

    return CB19Backend(
        grid_path=str(CB19_TEMPLATES),
        ssp_data=ssp_data_fsps,
        ionizing_source_warning="suppress",
        continuum_warning="suppress",
    )


# ── d(SED)/d(logU) across the photoionization backends ────────────

#: (fixture, logU probe point, FD step).
#:
#: The linear-interpolation backends are probed *inside* a grid cell. CLOUDY
#: nodes sit at -4, -3.5, ... -1, and at a node a piecewise-linear interpolant
#: has a kink: AD returns the right slope while central FD averages both, so
#: FD != AD there even though both are correct. ``triweight`` is the exception
#: -- its C2 kernel has no kink, which is the whole reason it exists, so that
#: row is deliberately probed *on* a node.
_LOGU_ROWS = [
    pytest.param("cloudy_linear_backend", -2.25, 1e-3, marks=requires_cloudy_grid, id="cloudy"),
    pytest.param(
        "cloudy_triweight_backend",
        -2.5,
        1e-3,
        marks=requires_cloudy_grid,
        id="cloudy_triweight_on_grid_node",
    ),
    pytest.param("cue_backend", -2.5, 1e-3, marks=requires_cue_weights, id="cue"),
    pytest.param("cb19_backend", -2.25, 1e-3, marks=requires_cb19, id="cb19"),
]


@pytest.mark.parametrize(("backend_fixture", "logU", "eps"), _LOGU_ROWS)
def test_logu_gradient_matches_finite_difference(request, neb_inputs, backend_fixture, logU, eps):
    """AD == FD for d(total nebular SED)/d(neb_logU), and the response is real."""
    backend = request.getfixturevalue(backend_fixture)
    ssp_wave, ssp_weights, ssp_log_ages = neb_inputs

    def fn(neb_logU):
        return jnp.sum(
            backend.predict_nebular_sed(
                ssp_wave=ssp_wave,
                ssp_weights=ssp_weights,
                ssp_log_ages_yr=ssp_log_ages,
                log_z=-1.848,
                neb_logU=neb_logU,
            )
        )

    _check_gradient(fn, logU, eps, f"{backend_fixture} d/d_logU")


# ── MAPPINGS V shock grid: d(SED)/d(velocity) ─────────────────────


@requires_mappings
def test_mappings_velocity_gradient_matches_finite_difference():
    """AD == FD for d(shock SED)/d(shock_velocity) at 350 km/s.

    The 3MdBs velocity axis spans 100-1000 km/s. 350 km/s is interior and off
    a node. The probe point matters more than usual here: at 150 km/s the
    measured log sensitivity falls to 0.013 and FD and AD disagree by 10%,
    which is the piecewise-linear kink near the low edge, not a defect.
    """
    from tengri.components.nebular.shock import ShockBackend

    backend = ShockBackend(shock_abundance="solar", shock_component="combined")
    wavelength = jnp.linspace(3000.0, 7000.0, 100)

    def fn(shock_velocity):
        return jnp.sum(
            backend.predict_nebular_sed(
                wavelength=wavelength,
                shock_velocity=shock_velocity,
                l_shock_halpha=1e40,  # erg/s, a typical warm-ionized SFG level
                shock_log_density=0.0,
                shock_b_over_sqrt_n=1.0,
            )
        )

    _check_gradient(fn, 350.0, 5.0, "MAPPINGS V d/d_velocity")


# ── CloudyGridBackend physics and construction ────────────────────


@requires_cloudy_grid
def test_logu_ordering(cloudy_linear_backend, neb_inputs):
    """Higher logU -> harder ionization -> higher [OIII]5007/Hbeta.

    Veilleux & Osterbrock (1987, ApJS 63, 295): [OIII]5007/Hbeta is the primary
    BPT diagnostic for the ionization parameter. A harder radiation field
    excites more O++ relative to recombination, raising the ratio. Probed at
    the grid extremes for maximum contrast.

    Line order in CLOUDY_LINE_NAMES: Hbeta at index 4 (4862.68 A), [OIII]5007
    at index 6 (5008.24 A) -- vacuum wavelengths.
    """
    _ssp_wave, ssp_weights, ssp_log_ages = neb_inputs

    def ratio(neb_logU):
        _names, lum = cloudy_linear_backend.predict_nebular_line_luminosities(
            ssp_weights=ssp_weights,
            ssp_log_ages_yr=ssp_log_ages,
            log_z=-1.848,
            neb_logU=neb_logU,
        )
        return float(lum[6]) / float(lum[4])

    ratio_low, ratio_high = ratio(-4.0), ratio(-1.0)

    assert ratio_high > ratio_low, (
        f"logU ordering violated: [OIII]/Hbeta at logU=-1 ({ratio_high:.3f}) "
        f"<= logU=-4 ({ratio_low:.3f}) -- Veilleux & Osterbrock 1987"
    )


@requires_cloudy_grid
def test_cloudy_grid_invalid_interp_mode(ssp_data_fsps):
    """Unknown grid_interp raises ValueError at construction time."""
    from tengri.components.nebular import CloudyGridBackend

    with pytest.raises(ValueError, match="grid_interp"):
        CloudyGridBackend(str(CLOUDY_GRID_MIST), ssp_data_fsps, grid_interp="cubic")


# ── Metallicity conversions ───────────────────────────────────────


@pytest.mark.parametrize("conversion", ["neb_logzsol_to_log_z_abs", "neb_logzsol_to_cloudy_logoh"])
def test_metallicity_conversion_is_a_linear_offset(conversion):
    """Both conversions are pure shifts: differences survive, order survives.

    Asserted as a property rather than against an absolute value, because the
    two anchors differ -- ``log_z_abs`` is pinned to ``_LOG10_ZSUN`` (see the
    test below) while ``cloudy_logoh`` follows the CLOUDY c17.01 solar O/H
    convention, which this file does not have a citation for.
    """
    from tengri.components.nebular import _shared

    fn = getattr(_shared, conversion)
    probes = jnp.array([-2.0, -1.0, -0.3, 0.0, 0.5])
    out = jnp.array([fn(p) for p in probes])

    # A shift preserves every pairwise difference exactly.
    np.testing.assert_allclose(
        np.diff(np.asarray(out)),
        np.diff(np.asarray(probes)),
        atol=1e-12,
        err_msg=f"{conversion} is not a pure offset -- differences are not preserved",
    )
    assert bool(jnp.all(jnp.diff(out) > 0.0)), f"{conversion} is not monotonically increasing"


def test_neb_logzsol_to_log_z_abs_anchors_on_solar():
    """logzsol=0 must map to _LOG10_ZSUN, and the analytic inverse round-trips.

    This is the half the offset property above cannot see: a conversion with
    the wrong constant is still a pure shift.
    """
    from tengri.components.nebular._constants import _LOG10_ZSUN
    from tengri.components.nebular._shared import neb_logzsol_to_log_z_abs

    solar = neb_logzsol_to_log_z_abs(jnp.array(0.0))
    assert abs(float(solar) - _LOG10_ZSUN) < 1e-8, (
        f"solar: expected {_LOG10_ZSUN:.6f}, got {float(solar):.6f}"
    )

    for logzsol_val in (-2.0, -1.0, 0.0, 0.5):
        back = float(neb_logzsol_to_log_z_abs(jnp.array(logzsol_val))) - _LOG10_ZSUN
        assert abs(back - logzsol_val) < 1e-8, (
            f"round-trip failed at logzsol={logzsol_val}: got {back:.6f}"
        )
