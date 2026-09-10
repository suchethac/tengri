# SPDX-License-Identifier: BSD-3-Clause
r"""#2195: the cue DIG branch evaluated a population it was never given.

https://github.com/suchethac/tengri/issues/2195

``NebularSEDComponent.apply`` snapshots ``_dig_kwargs`` from ``common_kwargs``
*before* the cue branch fills in ``ssp_weights`` / ``ssp_log_ages_yr`` (or
``gas_logqion``), so the DIG-branch calls to ``predict_nebular_sed`` and
``predict_nebular_line_luminosities`` reached ``CueBackend`` with no ionizing
population at all. Cue then fell back to ``default_gas_logqion = 49.1`` and its
hard-coded young-starburst ``ionspec_*`` values: the DIG component's ionization
parameter was right, its normalization and ionizing-spectrum shape were
decoupled from the galaxy. Line *ratios* survive that (a global normalization
cancels), which is why ``tests/components/nebular/test_dig_mixing.py`` passed
throughout; broadband photometry and absolute line luminosities do not.

Measured on the fixture below (cue backend, bare-stellar SSP, GALEX NUV +
SDSS ugri, delayed SFH, two-component calzetti, dale2014, z = 0.1, one
parameter swept between its own prior 5 %/95 % quantiles with everything else
at its declared default):

=========================================  =============================  =======  =======
condition                                  sweep                          before   after
=========================================  =============================  =======  =======
``neb_dig_frac`` at its default            ``neb_dig_delta_logU``         0.0      0.0
control                                    ``neb_dig_frac`` 0.05 to 0.95  5.83e-2  1.76e-2
``neb_dig_frac`` fixed 0.3                 ``neb_dig_delta_logU``         1.46e-6  2.35e-2
``neb_dig_frac`` fixed 1.0                 ``neb_dig_delta_logU``         4.98e-6  7.44e-2
control                                    ``neb_logU``, same span        7.44e-2  7.44e-2
=========================================  =============================  =======  =======

The exact 0.0 in the first row is an identity, not a defect:
:math:`(1 - 0) L_{\rm HII} + 0 \times L_{\rm DIG} = L_{\rm HII}`, and
``apply`` short-circuits the second backend call away. It is pinned here so it
is never mistaken for the bug the other rows report.

The last two rows agreeing to every digit after the fix is the identity that
identifies the mechanism: at :math:`f_{\rm DIG} = 1` the whole nebular
emission is the DIG branch, so sweeping the offset must reproduce sweeping
:math:`\log U` itself over the same span. It did not, by four orders of
magnitude, while the DIG branch was evaluating a different population. The
``neb_dig_frac`` control moves less after the fix (1.76e-2, from 5.83e-2)
because the HII and DIG branches now differ only by the ionization parameter,
no longer also by Q_H and the ionizing-spectrum shape.

The second gap #2195 covers is structural: the per-Q_H nebular grid
(``enable_fast_nebular`` / ``approx=FeaturePrecomp()``) tabulates only
``met_logzsol`` / ``neb_logU`` / ``neb_logZ_gas``, so DIG mixing cannot be
reconstructed from it at all. That now raises instead of silently dropping
both DIG parameters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_platforms", "cpu")

from tengri import DEFAULT, FREE, FeaturePrecomp, Fixed, Observation, Photometry, SEDModel
from tengri.config.exceptions import DIGNotOnNebularGridError
from tests._data_skip import CUE_WEIGHTS, DATA_DIR, requires_cue_weights

pytestmark = pytest.mark.regression_bug

_SSP_PATH = DATA_DIR / "fsps_prsc_miles_chabrier.h5"

#: Bands and redshift of the #2195 reproduction.
_FILTERS = ("galex_nuv", "sdss_u", "sdss_g", "sdss_r", "sdss_i")
_REDSHIFT = 0.1

#: The in-model control, a ``neb_dig_frac`` sweep over the same fixture, moves
#: the photometry by 1.76e-2 on the fixed tree (5.83e-2 before the fix). A DIG
#: parameter that moves it by less than a hundredth of the control is inert for
#: any practical purpose, so that is the floor. The measurements this separates:
#: 1.46e-6 / 4.98e-6 before the fix against 2.35e-2 / 7.44e-2 after.
_CONTROL_REL = 1.76e-2
_LIVE_FLOOR = _CONTROL_REL / 100.0


def _build(neb_extra, *, approx=None):
    """The #2195 fixture with ``neb_extra`` merged into the nebular group."""
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
    from tengri.observation.filters import load_filter

    obs = Observation(photometry=Photometry(filters=tuple(load_filter(n) for n in _FILTERS)))
    kwargs = dict(
        ssp_data=load_ssp_data(str(_SSP_PATH)),
        observation=obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT), **neb_extra},
        redshift=Fixed(_REDSHIFT),
    )
    if approx is not None:
        kwargs["approx"] = approx
    # No warning filter here, deliberately. Every build this file makes was
    # measured with ``simplefilter("always")`` and raises exactly zero warnings,
    # so a blanket ignore silenced nothing and would only have hidden the next
    # one. The degenerate pair this fixture exercises on purpose,
    # ``dig_delta_logU`` free at ``dig_frac = 0``, is a plausible future
    # advisory; it should be visible here, not swallowed.
    return SEDModel.build(**kwargs)


def _prior_quantiles(model, name, q=0.05):
    """The ``q`` / ``1 - q`` quantiles of ``name``'s declared uniform prior."""
    lo, hi = (float(v) for v in model.spec.get_distribution(name).bounds)
    span = hi - lo
    return lo + q * span, hi - q * span


def _max_rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.max(np.abs(b - a) / np.abs(a)))


def _sweep_photometry(model, name):
    """Largest relative photometry change across ``name``'s prior quantiles."""
    lo, hi = _prior_quantiles(model, name)
    return _max_rel(
        model.predict_photometry({name: jnp.asarray(lo)}),
        model.predict_photometry({name: jnp.asarray(hi)}),
    )


@pytest.fixture(scope="module")
def _cue_fixture_available():
    if not _SSP_PATH.is_file() or not CUE_WEIGHTS.is_file():
        pytest.skip(f"needs {_SSP_PATH} and {CUE_WEIGHTS}")


@requires_cue_weights
def test_delta_logu_is_inert_at_zero_dig_fraction(_cue_fixture_available):
    r"""At ``neb_dig_frac = 0`` the offset is inert **by construction**.

    :math:`L = (1 - f_{\rm DIG}) L_{\rm HII}(\log U) +
    f_{\rm DIG} L_{\rm DIG}(\log U + \Delta \log U)` has no
    :math:`\Delta \log U` dependence at :math:`f_{\rm DIG} = 0`. Expected
    behavior, pinned so the exact zero is never read as the #2195 defect.
    """
    model = _build({"dig_delta_logU": FREE})
    lo, hi = _prior_quantiles(model, "neb_dig_delta_logU")
    at_lo = np.asarray(model.predict_photometry({"neb_dig_delta_logU": jnp.asarray(lo)}))
    at_hi = np.asarray(model.predict_photometry({"neb_dig_delta_logU": jnp.asarray(hi)}))
    assert np.array_equal(at_lo, at_hi), (
        "photometry moved across the neb_dig_delta_logU prior at neb_dig_frac = 0; "
        f"the mixing identity says it cannot: {at_lo} vs {at_hi}"
    )


@requires_cue_weights
def test_delta_logu_moves_photometry_at_nonzero_dig_fraction(_cue_fixture_available):
    """The offset must move broadband photometry once DIG mixing is on.

    Before the fix: 1.46e-6 at ``neb_dig_frac = 0.3`` and 4.98e-6 at 1.0.
    After: 2.35e-2 and 7.44e-2, against a 1.76e-2 in-model control.
    """
    for frac in (0.3, 1.0):
        model = _build({"dig_delta_logU": FREE, "dig_frac": Fixed(frac)})
        rel = _sweep_photometry(model, "neb_dig_delta_logU")
        assert rel > _LIVE_FLOOR, (
            f"neb_dig_delta_logU moved the photometry by {rel:.3e} at "
            f"neb_dig_frac = {frac}, below the {_LIVE_FLOOR:.3e} floor set from "
            f"the {_CONTROL_REL:.3e} neb_dig_frac control: the DIG branch is not "
            "seeing the ionizing population (#2195)"
        )


@requires_cue_weights
def test_delta_logu_moves_absolute_line_luminosities(_cue_fixture_available):
    """Absolute mixed line luminosities [erg/s], not ratios, must respond.

    Ratios cancel the normalization the stale snapshot destroyed, so
    ``test_dig_mixing.py`` passed throughout #2195. Before the fix H-alpha
    moved by 1.22e-5 across this sweep; after, by 2.09e-1.
    """
    model = _build({"dig_delta_logU": FREE, "dig_frac": Fixed(0.3)})
    lo, hi = _prior_quantiles(model, "neb_dig_delta_logU")
    at_lo = model.predict({"neb_dig_delta_logU": jnp.asarray(lo)}).lines
    at_hi = model.predict({"neb_dig_delta_logU": jnp.asarray(hi)}).lines
    moved = {
        name: abs(float(getattr(at_hi, name)) - float(getattr(at_lo, name)))
        / abs(float(getattr(at_lo, name)))
        for name in ("halpha", "hbeta", "oiii_5007", "nii_6584")
    }
    assert min(moved.values()) > _LIVE_FLOOR, (
        f"absolute line luminosities barely moved across the neb_dig_delta_logU "
        f"prior at neb_dig_frac = 0.3: {moved} (floor {_LIVE_FLOOR:.3e}); the DIG "
        "branch is not seeing the ionizing population (#2195)"
    )


@requires_cue_weights
def test_pure_dig_with_no_offset_is_the_hii_solution(_cue_fixture_available):
    r"""``f_DIG = 1`` with :math:`\Delta \log U = 0` must reproduce ``f_DIG = 0``.

    Both branches then evaluate the same backend at the same ionization
    parameter with the same population, so the mix is an identity. Before the
    fix the two differed by 6.46e-2, because only the HII branch carried the
    SSP-derived Q_H and ionizing-spectrum shape.
    """
    pure_dig = _build({"dig_delta_logU": FREE, "dig_frac": Fixed(1.0)})
    pure_hii = _build({"dig_delta_logU": FREE, "dig_frac": Fixed(0.0)})
    at_dig = np.asarray(pure_dig.predict_photometry({"neb_dig_delta_logU": jnp.asarray(0.0)}))
    at_hii = np.asarray(pure_hii.predict_photometry({"neb_dig_delta_logU": jnp.asarray(0.0)}))
    np.testing.assert_allclose(
        at_dig,
        at_hii,
        rtol=1e-8,
        err_msg=(
            "pure DIG with a zero ionization-parameter offset did not reproduce "
            "pure HII: the two branches are not seeing the same population (#2195)"
        ),
    )


@requires_cue_weights
def test_nebular_grid_refuses_active_dig_mixing(_cue_fixture_available):
    """The per-Q_H grid has no DIG axis, so arming it with DIG on must raise.

    ``neb_dig_frac`` fixed at 0, its declared default, is the one disposition
    the grid can represent, and that build still succeeds.
    """
    approx = FeaturePrecomp(lines=jnp.asarray([4862.68, 5008.24, 6564.61]), n_grid=4)

    with pytest.raises(DIGNotOnNebularGridError, match=r"neb_dig_frac is fixed at 0\.3"):
        _build({"dig_frac": Fixed(0.3)}, approx=approx)

    with pytest.raises(DIGNotOnNebularGridError, match="neb_dig_frac is free"):
        _build({"dig_frac": FREE}, approx=approx)

    model = _build({"dig_delta_logU": FREE}, approx=approx)
    assert model.spec.fixed_value("neb_dig_frac") == 0.0


@pytest.mark.skip(
    reason="cloudy_grid/cb19 are structurally immune to #2195: both DIG call "
    "sites pass ssp_weights/ssp_log_ages_yr explicitly rather than fishing them "
    "out of the _dig_kwargs snapshot (component.py, the non-cue branch). A "
    "measured mirror of the photometry sweep needs data/cloudy_grid_mist.h5, "
    "which is not tracked in git; cb19's own log_U axis is flat (#2181/#924), "
    "so the sweep it would run cannot move anything either."
)
def test_cloudy_grid_mirror_of_the_photometry_sweep():
    """Placeholder for the cloudy_grid mirror; see the skip reason."""
