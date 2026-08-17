# SPDX-License-Identifier: BSD-3-Clause
r"""The cosmological flux scale, in log space at every seam that carries it (#1859).

Tier A (#1186) moved :math:`(1+z)/(4\pi d_L^2)` into log space at the seams that
*apply* it immediately, and deferred to "Task 6" every seam that **stores** or
**passes** it as a standalone scalar. This module is that task, and the tests
below are the measurement that motivates it.

Nothing in the chain is exotic. The inputs are representable, the answers are
representable, and only the intermediates are not:

===================  ==============  ==============  ============================
quantity             float64         float32         why
===================  ==============  ==============  ============================
``d_L`` (z=0.5)      9.0138e+27      9.0138e+27      fine
``4 pi d_L**2``      1.0210e+57      ``inf``         overflows **at the square**
``(1+z)/4pi d_L^2``  1.4692e-57      ``0.0``         ``(1+z)/inf``
``l_line`` [erg/s]   1.3913e+40      ``inf``         overflows at **every** z
line flux            1.3627e-17      ``nan``         ``inf/inf``
photometry           1.4692e-29      ``0.0``         ``L_nu * 0.0``
===================  ==============  ==============  ============================

Two distinct failure modes, and the *quieter* one is the dangerous one. The line
path yields ``nan``, which a finiteness check catches. The photometry path yields
exactly ``0.0`` — a plausible flux, finite, sign-correct, and wrong by every order
of magnitude at once.

``4 pi d_L**2`` overflows at **10 pc** (1.1965e+40), so there is no safe redshift
and no safe distance; and ``l_line`` overflows independently of distance, so
repairing only the divisor would still leave ``inf/inf``. Both have to leave the
linear domain together.

The fix is the form already proven at
:func:`tengri.observation.redshift_kernel.project_to_observed_frame` — carry the
scale as a ``log10`` offset and hand it to
:func:`tengri.utils.scale.apply_log10_scale`, so the out-of-range factor is never
materialized. What is new here is that the form now lives in *one* named place
rather than being hand-copied: it was spelled out longhand at twelve sites, seven
correct and five not.

**A ``10**(...)`` spelling does not count.** ``10**(log10(1+z) - LOG10_4PI -
2*log10(dl))`` is a linear form wearing a log hat and is ``0.0`` in float32 just
like the original — measured, and it is the trap that makes this fix look done
when it is not. The tests below therefore assert *finiteness of the applied
result*, never of the scale itself.

See #1859; Task 6 of #1186; parent epic #1206.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

#: Redshifts spanning 10 pc (the z=0 convention) to high-z.
ZS = (0.0, 0.01, 0.5, 3.0)

#: A normal galaxy continuum [erg/s/Hz].
L_NU = 1.0e28

#: H-alpha, vacuum [Angstrom], and a representative feature-window width.
LAM_C = 6564.61
WIDTH = 20.0


def _dl_cm(z):
    """Luminosity distance [cm] as a plain Python float, computed in float64."""
    from tengri.cosmology import luminosity_distance

    with jax.enable_x64(True):
        return float(luminosity_distance(jnp.asarray(float(z))))


def _f32_rtol(log10_scale):
    r"""float32 tolerance for a log-domain round trip at ``log10_scale`` [dex].

    Derived, not observed. :func:`~tengri.utils.scale.pow10` evaluates
    :math:`10^x` as :math:`e^{x\ln 10}`, so a relative error :math:`\epsilon` in
    the exponent lands as :math:`|x|\ln 10\,\epsilon` in the result. Across the
    redshifts here :math:`|x|` runs 41-59 dex, giving 1.1e-5 to 1.6e-5 — a floor
    no amount of care in the seam can go below, and one that *grows with
    distance*. A fixed literal would either be too tight at high z or too slack
    at low z; this tracks the exponent. The 4x is margin for the surrounding
    arithmetic (einsum, window means).
    """
    eps = float(np.finfo(np.float32).eps)
    return 4.0 * abs(float(log10_scale)) * float(np.log(10.0)) * eps


@pytest.fixture(scope="module")
def dls():
    return {z: _dl_cm(z) for z in ZS}


# ── the shared spelling ───────────────────────────────────────────


class TestLogScaleHelpers:
    """One named spelling of a formula that was hand-copied at twelve sites."""

    @pytest.mark.parametrize("z", ZS)
    def test_log10_flux_scale_matches_hand_written_form(self, z, dls):
        """Equals the longhand the seven correct sites spell out."""
        from tengri.utils.scale import LOG10_4PI, log10_flux_scale

        with jax.enable_x64(True):
            dl = jnp.asarray(dls[z])
            expected = jnp.log10(1.0 + z) - LOG10_4PI - 2.0 * jnp.log10(dl)
            np.testing.assert_allclose(log10_flux_scale(z, dl), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("z", ZS)
    def test_log10_four_pi_dl2_matches_hand_written_form(self, z, dls):
        """Equals ``log10(4 pi d_L^2)`` — the divisor half of the same formula."""
        from tengri.utils.scale import LOG10_4PI, log10_four_pi_dl2

        with jax.enable_x64(True):
            dl = jnp.asarray(dls[z])
            expected = LOG10_4PI + 2.0 * jnp.log10(dl)
            np.testing.assert_allclose(log10_four_pi_dl2(dl), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("z", ZS)
    def test_the_two_halves_agree(self, z, dls):
        """``log10_flux_scale == log10(1+z) - log10_four_pi_dl2``.

        Two helpers rather than one because the line path needs the divisor
        without the ``(1+z)``; they must not be able to disagree about the
        ``4 pi`` (a dropped ``LOG10_4PI`` is a 12.57x error that reads as a
        modeling choice).
        """
        from tengri.utils.scale import log10_flux_scale, log10_four_pi_dl2

        with jax.enable_x64(True):
            dl = jnp.asarray(dls[z])
            np.testing.assert_allclose(
                log10_flux_scale(z, dl),
                jnp.log10(1.0 + z) - log10_four_pi_dl2(dl),
                rtol=1e-15,
            )

    @pytest.mark.parametrize("z", ZS)
    def test_scale_applies_in_float32(self, z, dls):
        """The applied product is finite in float32 though the factor is not."""
        from tengri.utils.scale import apply_log10_scale, log10_flux_scale

        with jax.enable_x64(True):
            ref = float(apply_log10_scale(jnp.asarray(L_NU), log10_flux_scale(z, dls[z])))

        with jax.enable_x64(False):
            got = float(apply_log10_scale(jnp.asarray(L_NU), log10_flux_scale(z, dls[z])))

        assert np.isfinite(got) and got > 0.0, f"z={z}: got {got!r}, float64 says {ref:.4e}"
        np.testing.assert_allclose(got, ref, rtol=_f32_rtol(log10_flux_scale(z, dls[z])))


class TestLogWeightedSum:
    """A weighted sum of log-domain terms, without exponentiating either."""

    def test_matches_linear_weighted_sum(self):
        """float64 parity with the linear form it replaces, rtol 1e-12."""
        from tengri.utils.scale import log10_weighted_sum

        with jax.enable_x64(True):
            logs = jnp.asarray([-57.0, -56.5, -58.2, -57.7])
            w = jnp.asarray([0.1, 0.6, 0.25, 0.05])
            expected = jnp.sum(w * 10.0**logs)
            got = 10.0 ** log10_weighted_sum(logs, w)
            np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_zero_weight_drops_out_exactly(self):
        """``w=0`` contributes nothing, even against a ``-inf`` log term."""
        from tengri.utils.scale import log10_weighted_sum

        with jax.enable_x64(True):
            logs = jnp.asarray([-57.0, -jnp.inf])
            w = jnp.asarray([1.0, 0.0])
            np.testing.assert_allclose(log10_weighted_sum(logs, w), -57.0, rtol=1e-15)

    def test_finite_in_float32_where_the_linear_sum_is_zero(self):
        """The sum survives float32 at a magnitude the linear form cannot hold."""
        from tengri.utils.scale import log10_weighted_sum

        logs = [-57.0, -56.5]
        w = [0.4, 0.6]
        with jax.enable_x64(True):
            ref = float(log10_weighted_sum(jnp.asarray(logs), jnp.asarray(w)))
        with jax.enable_x64(False):
            got = float(log10_weighted_sum(jnp.asarray(logs), jnp.asarray(w)))
            linear = float(jnp.sum(jnp.asarray(w) * 10.0 ** jnp.asarray(logs)))

        assert linear == 0.0, "fixture no longer exercises the float32 failure"
        assert np.isfinite(got)
        np.testing.assert_allclose(got, ref, rtol=_f32_rtol(1.0))


# ── the line path: inf / inf ──────────────────────────────────────


class TestLineFluxSeam:
    """``_line_flux_from_means`` — the one divisor site for every line surface."""

    @pytest.mark.parametrize("z", ZS)
    def test_finite_in_float32(self, z, dls):
        """Currently ``nan``: ``l_line`` is ``inf`` and so is ``4 pi d_L^2``."""
        from tengri.observation.line_measurement import _line_flux_from_means
        from tengri.utils.scale import log10_four_pi_dl2

        with jax.enable_x64(True):
            ref = float(
                _line_flux_from_means(
                    jnp.asarray(L_NU),
                    jnp.asarray(0.9 * L_NU),
                    LAM_C,
                    WIDTH,
                    log10_four_pi_dl2(dls[z]),
                )
            )

        with jax.enable_x64(False):
            got = float(
                _line_flux_from_means(
                    jnp.asarray(L_NU),
                    jnp.asarray(0.9 * L_NU),
                    LAM_C,
                    WIDTH,
                    log10_four_pi_dl2(dls[z]),
                )
            )

        assert np.isfinite(got), f"z={z}: got {got!r}, float64 says {ref:.4e}"
        np.testing.assert_allclose(got, ref, rtol=_f32_rtol(log10_four_pi_dl2(dls[z])))

    @pytest.mark.parametrize("z", ZS)
    def test_float64_parity_with_the_linear_formula(self, z, dls):
        """No behavioral change in float64: rtol 1e-12 against the old spelling."""
        from tengri.observation.line_measurement import _line_flux_from_means
        from tengri.utils.physics_constants import C_AA
        from tengri.utils.scale import log10_four_pi_dl2

        with jax.enable_x64(True):
            feat, cont = jnp.asarray(L_NU), jnp.asarray(0.9 * L_NU)
            legacy = (feat - cont) * (C_AA / LAM_C**2) * WIDTH / (4.0 * jnp.pi * dls[z] ** 2)
            got = _line_flux_from_means(feat, cont, LAM_C, WIDTH, log10_four_pi_dl2(dls[z]))
            np.testing.assert_allclose(got, legacy, rtol=1e-12)

    def test_absorption_keeps_its_sign(self):
        """A negative feature-minus-continuum stays negative through the log seam."""
        from tengri.observation.line_measurement import _line_flux_from_means
        from tengri.utils.scale import log10_four_pi_dl2

        with jax.enable_x64(True):
            got = float(
                _line_flux_from_means(
                    jnp.asarray(0.9 * L_NU),
                    jnp.asarray(L_NU),
                    LAM_C,
                    WIDTH,
                    log10_four_pi_dl2(_dl_cm(0.5)),
                )
            )
        assert got < 0.0

    def test_zero_line_is_exactly_zero(self):
        """An identically-flat feature gives 0.0, not a denormal or a nan."""
        from tengri.observation.line_measurement import _line_flux_from_means
        from tengri.utils.scale import log10_four_pi_dl2

        with jax.enable_x64(False):
            got = float(
                _line_flux_from_means(
                    jnp.asarray(L_NU),
                    jnp.asarray(L_NU),
                    LAM_C,
                    WIDTH,
                    log10_four_pi_dl2(_dl_cm(0.5)),
                )
            )
        assert got == 0.0


# ── the photometry / spectroscopy path: silent 0.0 ────────────────


class TestFastKernels:
    """``fast_photometry`` / ``fast_spectrum`` take the scale as a log offset."""

    @staticmethod
    def _phot_inputs():
        weights = jnp.asarray([1.0e10, 5.0e9, 2.0e9])
        ssp_phot = jnp.full((3, 4), 1.0e-10)
        dust = jnp.full((3, 4), 0.5)
        return weights, ssp_phot, dust

    @pytest.mark.parametrize("z", ZS)
    def test_fast_photometry_finite_in_float32(self, z, dls):
        """Currently exactly 0.0 — the silent half of the defect."""
        from tengri.components.stellar.sps.precompute import fast_photometry
        from tengri.utils.scale import log10_flux_scale

        args = self._phot_inputs()
        with jax.enable_x64(True):
            ref = np.asarray(fast_photometry(*args, log10_flux_scale(z, dls[z])))
        with jax.enable_x64(False):
            got = np.asarray(fast_photometry(*args, log10_flux_scale(z, dls[z])))

        assert np.all(np.isfinite(got)) and np.all(got > 0.0), (
            f"z={z}: got {got!r}, float64 says {ref!r}"
        )
        np.testing.assert_allclose(got, ref, rtol=_f32_rtol(log10_flux_scale(z, dls[z])))

    @pytest.mark.parametrize("z", ZS)
    def test_fast_spectrum_finite_in_float32(self, z, dls):
        from tengri.components.stellar.sps.precompute import fast_spectrum
        from tengri.utils.scale import log10_flux_scale

        # Sized so the *net* lands in float32's normal range at every z in ZS.
        # ``fast_spectrum`` never applies ``LSUN_ERG_PER_S`` (unlike
        # ``fast_photometry``), so its output carries no compensating +33 dex and
        # a "realistic" Lsun/Hz input would put the true answer at ~1e-57 — below
        # float32's floor, which no arithmetic can rescue. ``apply_log10_scale``
        # promises finiteness only when ``max|arr| * 10**scale`` is representable;
        # asserting more than that would be testing the dtype, not the seam.
        weights = jnp.asarray([1.0e10, 5.0e9, 2.0e9])
        ssp_pix = jnp.full((3, 6), 1.0e12)
        dust = jnp.full((3, 6), 0.5)

        with jax.enable_x64(True):
            ref = np.asarray(fast_spectrum(weights, ssp_pix, dust, log10_flux_scale(z, dls[z])))
        with jax.enable_x64(False):
            got = np.asarray(fast_spectrum(weights, ssp_pix, dust, log10_flux_scale(z, dls[z])))

        assert np.all(np.isfinite(got)) and np.all(got > 0.0)
        np.testing.assert_allclose(got, ref, rtol=_f32_rtol(log10_flux_scale(z, dls[z])))

    @pytest.mark.parametrize("z", ZS)
    def test_fast_photometry_float64_parity(self, z, dls):
        """rtol 1e-12 against the linear kernel it replaces."""
        from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S
        from tengri.components.stellar.sps.precompute import fast_photometry
        from tengri.utils.scale import log10_flux_scale

        weights, ssp_phot, dust = self._phot_inputs()
        with jax.enable_x64(True):
            linear = (1.0 + z) / (4.0 * jnp.pi * dls[z] ** 2)
            legacy = linear * jnp.einsum("i,if,if->f", weights, dust, ssp_phot) * LSUN_ERG_PER_S
            got = fast_photometry(weights, ssp_phot, dust, log10_flux_scale(z, dls[z]))
            np.testing.assert_allclose(got, legacy, rtol=1e-12)


# ── the z-table: a stored scale, interpolated ─────────────────────


class TestZTableInterpolation:
    """The free-redshift table stores the scale; the cast to float32 zeroes it."""

    @staticmethod
    def _table(dls):
        """A log-domain flux-scale table over ``ZS``, plus its linear twin."""
        from tengri.utils.scale import log10_flux_scale

        with jax.enable_x64(True):
            z_grid = jnp.asarray([float(z) for z in ZS])
            logs = jnp.asarray([float(log10_flux_scale(z, dls[z])) for z in ZS])
            linear = jnp.asarray([(1.0 + z) / (4.0 * jnp.pi * dls[z] ** 2) for z in ZS])
        return z_grid, logs, linear

    def test_stored_linear_table_is_zero_in_float32(self, dls):
        """The premise: the table this replaces cannot survive the cast."""
        _, _, linear = self._table(dls)
        with jax.enable_x64(False):
            cast = np.asarray(jnp.asarray(np.asarray(linear, dtype=np.float64)))
        # z=0 lands on a subnormal and survives the *cast*; every other node does not.
        assert np.all(cast[1:] == 0.0), f"fixture no longer exercises the defect: {cast!r}"

    def test_log_interp_is_the_exact_linear_interp(self, dls):
        """float64 parity, rtol 1e-12 — this must be a re-spelling, not a re-model.

        Interpolating ``log10(scale)`` linearly is *not* the same function as
        interpolating ``scale`` linearly. The log form has to reproduce the
        latter exactly, which is what :func:`log10_weighted_sum` buys.
        """
        from tengri.components.stellar.sps.precompute import interpolate_ztable

        z_grid, logs, linear = self._table(dls)
        with jax.enable_x64(True):
            phot = jnp.ones((len(ZS), 1, 1, 1))
            eff = jnp.ones((len(ZS), 1))
            for z_query in (0.005, 0.2, 1.5, 2.9):
                _, _, got_log = interpolate_ztable(phot, eff, logs, z_grid, z_query)
                idx = int(np.searchsorted(np.asarray(z_grid), z_query) - 1)
                frac = (z_query - float(z_grid[idx])) / float(z_grid[idx + 1] - z_grid[idx])
                expected = (1.0 - frac) * float(linear[idx]) + frac * float(linear[idx + 1])
                np.testing.assert_allclose(10.0 ** float(got_log), expected, rtol=1e-12)

    def test_log_interp_finite_in_float32(self, dls):
        from tengri.components.stellar.sps.precompute import interpolate_ztable

        z_grid, logs, _ = self._table(dls)
        phot = jnp.ones((len(ZS), 1, 1, 1))
        eff = jnp.ones((len(ZS), 1))
        with jax.enable_x64(True):
            _, _, ref = interpolate_ztable(phot, eff, logs, z_grid, 0.2)
        with jax.enable_x64(False):
            _, _, got = interpolate_ztable(phot, eff, logs, z_grid, 0.2)

        assert np.isfinite(float(got))
        np.testing.assert_allclose(float(got), float(ref), rtol=_f32_rtol(1.0))

    def test_smooth_log_interp_is_the_exact_weighted_sum(self, dls):
        """Same parity bar for the triweight (C²) interpolator."""
        from tengri.components.stellar.sps.precompute import interpolate_ztable_smooth
        from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

        z_grid, logs, linear = self._table(dls)
        with jax.enable_x64(True):
            phot = jnp.ones((len(ZS), 1, 1, 1))
            eff = jnp.ones((len(ZS), 1))
            _, _, got_log = interpolate_ztable_smooth(phot, eff, logs, z_grid, 0.4, 0.25)
            w = compute_grid_weights(0.4, z_grid, scatter=0.25, edges=edges_for_grid(z_grid))
            expected = float(jnp.dot(w, linear))
            np.testing.assert_allclose(10.0 ** float(got_log), expected, rtol=1e-12)

    def test_smooth_log_interp_finite_in_float32(self, dls):
        from tengri.components.stellar.sps.precompute import interpolate_ztable_smooth

        z_grid, logs, _ = self._table(dls)
        phot = jnp.ones((len(ZS), 1, 1, 1))
        eff = jnp.ones((len(ZS), 1))
        with jax.enable_x64(True):
            _, _, ref = interpolate_ztable_smooth(phot, eff, logs, z_grid, 0.4, 0.25)
        with jax.enable_x64(False):
            _, _, got = interpolate_ztable_smooth(phot, eff, logs, z_grid, 0.4, 0.25)

        assert np.isfinite(float(got))
        np.testing.assert_allclose(float(got), float(ref), rtol=_f32_rtol(1.0))


# ── the public line surface, end to end ───────────────────────────


class TestPublicLineSurface:
    """``measure_line_flux_jax`` is the surface users reach; it must survive too."""

    @pytest.mark.parametrize("z", (0.01, 0.5, 3.0))
    def test_measure_line_flux_jax_finite_in_float32(self, z, dls):
        from tengri.observation.line_measurement import DESI_LINES, measure_line_flux_jax
        from tengri.utils.scale import log10_four_pi_dl2

        line = next(ln for ln in DESI_LINES if "alpha" in ln.name.lower() or "Ha" in ln.name)
        wave = jnp.linspace(4000.0, 9000.0, 2000)

        def sed(dtype_ctx):
            with dtype_ctx:
                w = jnp.linspace(4000.0, 9000.0, 2000)
                cont = jnp.full_like(w, L_NU)
                bump = 0.5 * L_NU * jnp.exp(-0.5 * ((w - line.wavelength) / 8.0) ** 2)
                return measure_line_flux_jax(w, cont + bump, line, log10_four_pi_dl2(dls[z]))

        ref = float(sed(jax.enable_x64(True)))
        got = float(sed(jax.enable_x64(False)))

        assert np.isfinite(got), f"z={z}: got {got!r}, float64 says {ref:.4e}"
        np.testing.assert_allclose(got, ref, rtol=1e-4)
        assert wave.shape == (2000,)
