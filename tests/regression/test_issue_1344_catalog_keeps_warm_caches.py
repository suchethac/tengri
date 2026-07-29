# SPDX-License-Identifier: BSD-3-Clause
"""Catalog fits keep their warm caches; they must not sweep (#1344).

The retired-``lean=`` warning promised "Catalog sweeps automatically". Nothing in
``src/`` ever passed ``_cache_policy='sweep'``, and #1344 left open whether to
wire it or withdraw the promise. Measurement settles it: **wiring it would be a
regression.**

``Fitter._lean_keep_sig`` is ``compile_signature()`` — data *shape*, never data
values — so two galaxies of the same model and shape produce the same key:

* ``iterate`` keeps the entry they share  -> one inference-body compile per catalog
* ``sweep`` passes ``keep_sig=None``      -> drops it too -> one compile per galaxy

That second case is the #1316 cliff the catalog path exists to remove, so the
promise was withdrawn rather than implemented. These tests pin the two facts the
decision rests on, because the decision is only correct while they hold.

``sweep`` itself is *not* dead and is not removed: ``tengri.lean()`` and the
deprecated ``lean=True`` both still select it, for runs that would rather pay the
recompile than hold the scan body in RAM.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

BANDS = ["sdss_g", "sdss_r", "sdss_i"]


def _fitter(model, flux):
    from tengri.inference.fitter import Fitter

    return Fitter(model, np.asarray(flux), np.asarray(flux) * 0.1)


@pytest.fixture
def model(ssp_data_wne):
    from tengri import Fixed, Observation, Photometry, SEDModel

    obs = Observation(photometry=Photometry.from_names(BANDS))
    return SEDModel.build(
        ssp_data=ssp_data_wne, observation=obs, sfh={"type": "dpl"}, redshift=Fixed(0.1)
    )


class TestTheFactTheDecisionRestsOn:
    """If these stop holding, `iterate` stops being right for catalogs."""

    def test_keep_sig_is_shared_across_different_galaxies(self, model):
        """Data *values* must not enter the key, or every galaxy recompiles."""
        a = _fitter(model, [1e-29, 2e-29, 3e-29])
        b = _fitter(model, [5e-29, 1e-29, 9e-29])

        assert a._lean_keep_sig == b._lean_keep_sig, (
            "two galaxies of the same model and shape must share the L3 keep "
            "signature; if they do not, the iterate policy recompiles per "
            "galaxy and the #1344 decision is wrong"
        )

    def test_engine_cache_key_is_shared_across_different_galaxies(self, model):
        a = _fitter(model, [1e-29, 2e-29, 3e-29])
        b = _fitter(model, [5e-29, 1e-29, 9e-29])
        assert a._engine_cache_key() == b._engine_cache_key()

    def test_keep_sig_is_the_compile_signature(self, model):
        """Pin the identity the reasoning quotes, not just today's value."""
        f = _fitter(model, [1e-29, 2e-29, 3e-29])
        assert f._lean_keep_sig == f.compile_signature()


class TestThePromiseIsGone:
    def test_no_source_file_passes_cache_policy_sweep(self):
        """The withdrawn promise must not quietly come back as a wiring."""
        import pathlib

        import tengri

        src = pathlib.Path(tengri.__file__).parent
        offenders = [
            str(p.relative_to(src))
            for p in src.rglob("*.py")
            if "_cache_policy" in p.read_text()
            and "sweep" in p.read_text().split("_cache_policy")[1][:400]
            and p.name != "fitter.py"
        ]
        assert not offenders, (
            f"{offenders} appear to pass _cache_policy near 'sweep'. Wiring sweep "
            "into the catalog path recompiles per galaxy (#1344, #1316)."
        )

    def test_retirement_warning_no_longer_promises_a_catalog_sweep(self, model):
        """The warning is the user-facing artifact; it must not claim the sweep."""
        import contextlib
        import warnings

        f = _fitter(model, [1e-29, 2e-29, 3e-29])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Only the warning text is under test; the fit itself may fail for
            # unrelated reasons and must not mask it.
            with contextlib.suppress(Exception):
                f.run("map", lean=True, n_steps=1)

        texts = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        lean_msgs = [t for t in texts if "lean=" in t]
        assert lean_msgs, "the lean= retirement warning should still fire"
        assert not any("sweeps automatically" in t for t in lean_msgs), (
            "the warning still promises an automatic catalog sweep (#1344)"
        )


def test_sweep_remains_selectable_and_is_not_dead_code():
    """Withdrawing the promise must not delete the policy it named."""
    import inspect

    from tengri.inference import fitter as fitter_mod

    src = inspect.getsource(fitter_mod.Fitter.run)
    assert '"sweep"' in src or "'sweep'" in src, (
        "the sweep policy branch was removed; tengri.lean() still selects it"
    )
