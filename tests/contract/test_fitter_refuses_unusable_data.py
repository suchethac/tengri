# SPDX-License-Identifier: BSD-3-Clause
"""A fit on unusable data must not return a number (#2155).

One NaN flux in a five-band photometry fit produced this::

    clean       log_M = 9.999993   n_steps = 350   final_loss = 2.49e-06
    one NaN     log_M = 9.942578   n_steps = 100   final_loss = nan

``final_loss`` is ``nan`` — the objective is undefined everywhere, the
optimizer early-stops, and ``params`` comes back carrying a finite,
plausible-looking stellar mass. No warning, no error. It is 0.057 dex (14% in
mass) from the truth, and **not** the answer you get by genuinely dropping that
band (9.999996), so it was not masking either — just an undefined optimization
reported as a measurement.

Zero uncertainty measured identically (``final_loss = nan``, same 9.942578,
same early stop). Negative uncertainty is a different case: chi2 squares it, so
it silently gives the *right* answer, which is exactly how a sign error
upstream survives forever. A sigma is not negative under any reading, so it is
refused too.

**The library already knew the answer.** ``inference/catalog_ingest.py`` has
had the right semantics all along: a non-finite flux means *absent*, chosen
with ``missing='mask'``, and a non-finite error beside a finite flux is always
an error because "an unknown uncertainty is not an absent band". None of it
reached the single-object path, so the same array meant "this band is missing"
through ``CatalogFitter`` and "silently corrupt the fit" through ``Fitter``.
One sibling right, one wrong, which is the giveaway that nothing enforced the
rule.

The second half of this file is the part that nearly shipped broken. The new
error tells the user to mark the band absent with ``presence=`` — and that did
not work, because ``0 * nan`` is ``nan``, so a *correctly masked* NaN poisoned
the likelihood exactly as an unmasked one did. Advice that does not work is
worse than no advice. Masked data is now neutralized the same way
``catalog_ingest`` does it, and the test below pins the property that makes the
advice true: masking a band must equal dropping it.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fitter,
    Fixed,
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
)

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

_BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
_DROP = 1


def _build(ssp, bands):
    obs = Observation(photometry=Photometry.from_names(bands))
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )
    return model, obs


@pytest.fixture(scope="module")
def clean(ssp_data_fsps):
    model, obs = _build(ssp_data_fsps, _BANDS)
    truth = dict(model.spec.get_fixed_values())
    truth["sfh_delayed_log_total_mass"] = 10.0
    flux = np.asarray(model.predict_photometry(truth))
    return model, obs, flux, 0.05 * flux


def _fit(model, obs, flux, noise, **kw):
    f = Fitter(ForwardModel.build(sed=model, observation=obs), flux, noise, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        return f.run("map")


class TestUnusableDataIsRefused:
    @pytest.mark.parametrize(
        ("label", "kind", "value"),
        [
            ("nan flux", "flux", np.nan),
            ("inf flux", "flux", np.inf),
            ("nan noise", "noise", np.nan),
            ("zero noise", "noise", 0.0),
            ("negative noise", "noise", -1.0),
        ],
    )
    def test_it_raises_and_names_the_index(self, clean, label, kind, value):
        model, obs, flux, noise = clean
        f, n = flux.copy(), noise.copy()
        if kind == "flux":
            f[_DROP] = value
        else:
            n[_DROP] = value if value != -1.0 else -n[_DROP]
        with pytest.raises(ValueError) as exc:
            _fit(model, obs, f, n)
        message = str(exc.value)
        assert str(_DROP) in message, f"{label}: message does not name the index:\n  {message}"
        assert "presence=" in message, f"{label}: message offers no way forward:\n  {message}"

    def test_clean_data_still_fits(self, clean):
        """The guard must not cost the working case."""
        model, obs, flux, noise = clean
        post = _fit(model, obs, flux, noise)
        got = float(np.asarray(post.params["sfh_delayed_log_total_mass"]))
        assert np.isclose(got, 10.0, atol=1e-3), got
        assert np.isfinite(post.diagnostics["final_loss"])


class TestTheAdviceInTheMessageWorks:
    """The half that nearly shipped broken.

    ``0 * nan`` is ``nan``, so telling a user to mark the band absent was
    telling them to do something that left the objective just as undefined.
    """

    def test_masking_a_nan_band_equals_dropping_it(self, clean, ssp_data_fsps):
        model, obs, flux, noise = clean
        nan_flux = flux.copy()
        nan_flux[_DROP] = np.nan
        presence = np.ones(len(_BANDS))
        presence[_DROP] = 0.0
        masked = _fit(model, obs, nan_flux, noise, presence=presence)

        kept = [b for i, b in enumerate(_BANDS) if i != _DROP]
        sub_model, sub_obs = _build(ssp_data_fsps, kept)
        keep = np.arange(len(_BANDS)) != _DROP
        dropped = _fit(sub_model, sub_obs, flux[keep], noise[keep])

        a = float(np.asarray(masked.params["sfh_delayed_log_total_mass"]))
        b = float(np.asarray(dropped.params["sfh_delayed_log_total_mass"]))
        assert np.isfinite(masked.diagnostics["final_loss"]), (
            "a masked NaN still produced a non-finite objective — the advice in "
            "the error message does not work"
        )
        assert np.isclose(a, b, atol=1e-4), (
            f"masking band {_DROP} gave log_M={a:.6f} but dropping it gave "
            f"{b:.6f}; masking must mean absent, not something else"
        )

    def test_the_catalog_shape_of_a_masked_band_is_accepted(self, clean):
        """Guard every consumer, and check which one runs first.

        ``missing='mask'`` ingestion hands an absent band **flux 0.0 and noise
        0.0** — both through ``np.nan_to_num``. So the guard's ``noise <= 0``
        arm sees a zero uncertainty on every masked catalog band, and any
        consumer that forgets to thread ``presence`` is refused.

        ``CatalogFitter._get_dummy_fitter`` was exactly that consumer: the
        per-galaxy fitter passed ``presence=`` and the dummy one did not, so
        adding the guard alone would have broken every catalog with a masked
        band in galaxy 0. This pins the shape it must accept.
        """
        model, obs, flux, noise = clean
        ingested_flux, ingested_noise = flux.copy(), noise.copy()
        ingested_flux[_DROP] = 0.0
        ingested_noise[_DROP] = 0.0
        presence = np.ones(len(_BANDS))
        presence[_DROP] = 0.0
        Fitter(
            ForwardModel.build(sed=model, observation=obs),
            ingested_flux,
            ingested_noise,
            presence=presence,
        )

    def test_the_dummy_fitter_threads_presence(self):
        """The consumer above, pinned by construction rather than by a fit.

        Reading the source is the cheap way to keep the two Fitter call sites
        in ``catalog_fitter`` in step without building a catalog fixture.
        """
        import inspect

        from tengri.inference.catalog_fitter import _CatalogFitterOriginal

        src = inspect.getsource(_CatalogFitterOriginal._get_dummy_fitter)
        assert "presence=" in src, (
            "_get_dummy_fitter no longer threads presence; a catalog with a "
            "masked band in galaxy 0 will be refused by the data guard"
        )

    def test_a_present_band_is_still_checked(self, clean):
        """Neutralization must not become a blanket nan-scrub.

        A presence mask that marks a *different* band absent leaves the NaN
        band present, and it must still be refused.
        """
        model, obs, flux, noise = clean
        nan_flux = flux.copy()
        nan_flux[_DROP] = np.nan
        presence = np.ones(len(_BANDS))
        presence[_DROP + 1] = 0.0  # some other band absent
        with pytest.raises(ValueError):
            _fit(model, obs, nan_flux, noise, presence=presence)


class TestOrderDependence:
    """Verify the guard runs on EVERY fit call, not cached from the first (#2155).

    The root cause was that multiple Fitter instances with the same model and
    data shape share a cached loss function. A fresh Fitter.__init__ was the
    only place the guard ran per-fit call. If a code path reused a Fitter or
    skipped __init__ for subsequent fits, bad data in fit N would not be
    caught even if fit 1 was clean.

    These tests pin the cache-skip regime on the direct Fitter path: a
    warm loss cache must never stand in for validation of a later
    Fitter's own data.
    """

    def test_bad_data_in_second_fit_is_refused(self, clean):
        """Fit 1 good, Fit 2 bad: bad data must still raise.

        Simulates the order-dependent case: first Fitter compiles and caches
        the loss function, second Fitter with different (bad) data reuses the
        cache. Without the __init__ guard on EVERY Fitter, fit 2 would silently
        return a wrong answer.
        """
        model, obs, flux, noise = clean

        # Fit 1: clean data (warms the cache)
        post1 = _fit(model, obs, flux, noise)
        assert np.isfinite(post1.diagnostics["final_loss"])

        # Fit 2: same model, same shape, but different bad data
        bad_flux = flux.copy()
        bad_flux[_DROP] = np.nan
        with pytest.raises(ValueError) as exc:
            _fit(model, obs, bad_flux, noise)
        message = str(exc.value)
        assert str(_DROP) in message, f"bad fit 2 did not name the bad index: {message}"
        assert "presence=" in message

    def test_bad_noise_in_second_fit_is_refused(self, clean):
        """Fit 1 good, Fit 2 bad noise: bad noise must still raise.

        Same as above but with noise, testing both zero and negative.
        """
        model, obs, flux, noise = clean

        # Fit 1: clean
        post1 = _fit(model, obs, flux, noise)
        assert np.isfinite(post1.diagnostics["final_loss"])

        # Fit 2: zero noise
        bad_noise = noise.copy()
        bad_noise[_DROP] = 0.0
        with pytest.raises(ValueError):
            _fit(model, obs, flux, bad_noise)

        # Fit 3: negative noise (separate Fitter, same cache state)
        bad_noise2 = noise.copy()
        bad_noise2[_DROP] = -1.0
        with pytest.raises(ValueError):
            _fit(model, obs, flux, bad_noise2)
