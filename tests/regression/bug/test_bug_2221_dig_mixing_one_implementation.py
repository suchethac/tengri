# SPDX-License-Identifier: BSD-3-Clause
r"""#2221: the nebular component carried its own copy of the DIG mixing arithmetic.

https://github.com/suchethac/tengri/issues/2221

``mix_dig_emission`` in ``dig.py`` was the tested implementation of
:math:`L = (1 - f) L_{\rm HII}(\log U) + f L_{\rm DIG}(\log U + \Delta \log U)`,
but nothing in production called it: ``NebularSEDComponent.apply`` carried the
same arithmetic inline at four sites (cue SED, cloudy/cb19 SED, cue lines,
cloudy/cb19 lines), each with its own snapshot of ``common_kwargs``
(``_dig_kwargs`` / ``_dig_frac_is_zero``) taken *before* the cue branch
resolved the ionizing population into it. That snapshot-before-mutation
hazard is how #2195 shipped behind twenty green unit tests of the unused
``mix_dig_emission`` copy: the inline copy diverged from the tested one and
no test exercised the production path with a live DIG parameter.

This test pins two things the refactor (routing all four sites through
``dig.py``) must preserve going forward:

1. **The mixing identity itself**, on the production path: predicted
   photometry and line luminosities at ``f = 0.3`` must equal
   :math:`(1-f)` times the ``f=0`` prediction at ``neb_logU`` plus :math:`f`
   times the ``f=0`` prediction at ``neb_logU + \Delta\log U`` -- to
   ``rtol=1e-12, atol=0.0``, since this is an arithmetic identity, not an
   approximation.
2. **Delta-logU liveness**: mutating ``neb_dig_delta_logU`` must move both
   the photometry and the line-luminosity channels once ``neb_dig_frac`` is
   nonzero -- the #2195 regression, restated at the production-path level.

**Why every comparison below reuses ONE model instance, varied only via
override dicts (never three separately-built models with different Fixed
values)**: measured both ways, and they agree to the last measured digit.
Three separately-built ``Fixed``-valued models (``dig_frac=0`` at
``neb_logU=-2.5``, ``dig_frac=0`` at ``-3.5``, and ``dig_frac=0.3`` at
``-2.5``) satisfy the mixing identity to ``5.902e-15`` relative
(photometry) and ``1.829e-14`` relative (line luminosities); one ``FREE``
model varied via ``jnp.asarray(...)`` override dicts gives the identical
``5.902e-15`` / ``1.829e-14``. A ``FREE`` model evaluated at a given
``(neb_dig_frac, neb_logU)`` is also bit-identical (``rel = 0.0``) to a
separately-built ``Fixed`` model at the same values: there is no
Fixed-versus-FREE seam in the cue backend or the grammar. The one-model
construction below is kept only because it is one ``SEDModel.build`` call
instead of three, not because separate builds disagree.

Mutation-check for this test (recorded in the #2221 commit body, not
committed as code): dropping ``cue_population`` from the DIG evaluation on
the cue branch reproduces #2195 and turns this test (or
``test_bug_2195_dig_delta_logu_reaches_cue.py``) red.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_platforms", "cpu")

from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel
from tests._data_skip import CUE_WEIGHTS, DATA_DIR

pytestmark = pytest.mark.regression_bug

_SSP_PATH = DATA_DIR / "fsps_prsc_miles_chabrier.h5"
_FILTERS = ("galex_nuv", "sdss_u", "sdss_g", "sdss_r", "sdss_i")
_REDSHIFT = 0.1
_LOGU = -2.5
_DELTA_LOGU = -1.0
_DIG_FRAC = 0.3


@pytest.fixture(scope="module")
def _model():
    """One cue-backend model, DIG params all FREE, varied only via override dicts.

    Building once and overriding at call time (rather than building separate
    ``Fixed``-valued models per condition) keeps every prediction below inside
    the same compiled ``predict_photometry`` graph -- see the module
    docstring for why that matters for a neural-network backend.
    """
    if not _SSP_PATH.is_file() or not CUE_WEIGHTS.is_file():
        pytest.skip(f"needs {_SSP_PATH} and {CUE_WEIGHTS}")

    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
    from tengri.observation.filters import load_filter

    obs = Observation(photometry=Photometry(filters=tuple(load_filter(n) for n in _FILTERS)))
    return SEDModel.build(
        ssp_data=load_ssp_data(str(_SSP_PATH)),
        observation=obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={
            "type": "cue",
            "all_params": Fixed(DEFAULT),
            "dig_frac": FREE,
            "logU": FREE,
            "dig_delta_logU": FREE,
        },
        redshift=Fixed(_REDSHIFT),
    )


def _override(dig_frac, logU=_LOGU, delta=_DELTA_LOGU):
    return {
        "neb_dig_frac": jnp.asarray(dig_frac),
        "neb_logU": jnp.asarray(logU),
        "neb_dig_delta_logU": jnp.asarray(delta),
    }


def test_dig_mixing_is_the_arithmetic_identity_for_photometry(_model):
    r"""Photometry at ``f=0.3`` must equal the closed-form mix, to machine precision.

    :math:`L(f{=}0.3) = 0.7\,L(f{=}0,\log U) + 0.3\,L(f{=}0,\log U+\Delta)`.
    """
    hii_phot = np.asarray(_model.predict_photometry(_override(dig_frac=0.0, logU=_LOGU)))
    dig_phot = np.asarray(
        _model.predict_photometry(_override(dig_frac=0.0, logU=_LOGU + _DELTA_LOGU))
    )
    mixed_phot = np.asarray(_model.predict_photometry(_override(dig_frac=_DIG_FRAC, logU=_LOGU)))

    expected = (1.0 - _DIG_FRAC) * hii_phot + _DIG_FRAC * dig_phot
    np.testing.assert_allclose(
        mixed_phot,
        expected,
        rtol=1e-12,
        atol=0.0,
        err_msg=(
            "production-path photometry at neb_dig_frac=0.3 does not equal the "
            "closed-form (1-f)*HII + f*DIG mix of the two single-regime "
            "predictions: the cue SED call site in component.py is not "
            "computing the mix_dig_emission identity (#2221)"
        ),
    )


def test_dig_mixing_is_the_arithmetic_identity_for_line_luminosities(_model):
    r"""Line luminosities at ``f=0.3`` must equal the closed-form mix."""
    hii_lums = np.asarray(_model.predict(_override(dig_frac=0.0, logU=_LOGU)).lines.all_lums)
    dig_lums = np.asarray(
        _model.predict(_override(dig_frac=0.0, logU=_LOGU + _DELTA_LOGU)).lines.all_lums
    )
    mixed_lums = np.asarray(
        _model.predict(_override(dig_frac=_DIG_FRAC, logU=_LOGU)).lines.all_lums
    )

    expected = (1.0 - _DIG_FRAC) * hii_lums + _DIG_FRAC * dig_lums
    np.testing.assert_allclose(
        mixed_lums,
        expected,
        rtol=1e-12,
        atol=0.0,
        err_msg=(
            "production-path line luminosities at neb_dig_frac=0.3 do not equal "
            "the closed-form (1-f)*HII + f*DIG mix: the cue line-luminosity call "
            "site in component.py is not computing the mix_dig_line_luminosities "
            "identity (#2221)"
        ),
    )


def test_delta_logu_moves_both_channels_at_nonzero_dig_frac(_model):
    """#2195, restated: with DIG mixing on, the offset must move both channels.

    Evaluating ``neb_dig_delta_logU`` at two different values (``dig_frac``
    pinned nonzero throughout) must disagree on both photometry and line
    luminosities. If a call site silently drops the offset (or the ionizing
    population, #2195's actual mechanism), both differences degrade to
    numerical noise.
    """
    base_phot = np.asarray(
        _model.predict_photometry(_override(dig_frac=_DIG_FRAC, delta=_DELTA_LOGU))
    )
    moved_phot = np.asarray(
        _model.predict_photometry(_override(dig_frac=_DIG_FRAC, delta=2.0 * _DELTA_LOGU))
    )
    assert not np.allclose(base_phot, moved_phot, rtol=1e-6, atol=0.0), (
        "photometry did not move when neb_dig_delta_logU changed at "
        "neb_dig_frac=0.3: the DIG branch is not seeing the offset (#2195/#2221)"
    )

    base_lums = np.asarray(
        _model.predict(_override(dig_frac=_DIG_FRAC, delta=_DELTA_LOGU)).lines.all_lums
    )
    moved_lums = np.asarray(
        _model.predict(_override(dig_frac=_DIG_FRAC, delta=2.0 * _DELTA_LOGU)).lines.all_lums
    )
    assert not np.allclose(base_lums, moved_lums, rtol=1e-6, atol=0.0), (
        "line luminosities did not move when neb_dig_delta_logU changed at "
        "neb_dig_frac=0.3: the DIG branch is not seeing the offset (#2195/#2221)"
    )
