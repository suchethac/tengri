#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Rank samplers on a published notebook's own model, by seconds per effective sample.

Generalizes ``benchmark_quickstart_sampler.py`` to the other two notebooks that
never followed nb06/nb07 from NUTS to fixed-length HMC. That earlier migration
bought 6.3x and 3.4x; ``bench/reports/2026-08-17_quickstart_nuts_vs_hmc.md``
measured the same swap on ``00_quickstart`` and found it **worse**, and closed
with the warning that ``01_why_jax`` and ``05_fitting_photometry`` were *not*
measured and must not be assumed to follow. This script measures them.

Two columns exist because both of the obvious ones lie:

* **Wall time** rewards a sampler for drawing correlated samples quickly. On
  the quickstart, HMC at L=20 was 8.8x faster than NUTS and returned 18.8
  effective samples.
* **Mean ESS across parameters** hides the failure mode, which is a single
  weakly-identified direction dragging while the rest look healthy. The
  worst-mixing parameter is therefore named per row.

Wall time is the one number this machine cannot measure reliably under load;
ESS, R-hat and the divergence count are deterministic given the seed.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py --notebook 05
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py \\
        --notebook 01 --quick

    # six seeds per row, one fit per subprocess (the 2026-08-21 campaign protocol)
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py \\
        --notebook 05 --only "nuts (shipped),mclmc" --seeds 6

THE DUST LAW, AND WHY THERE ARE TWO OF SEVERAL FIXTURES
=======================================================

This file could not build its own nb05 model on ``main``: it passed the retired
``dust=`` peer group (since split into ``dust_attenuation`` / ``dust_emission``)
and named ``law_bc`` without ``law_diff``, which now raises. Every row raised
``ValueError`` at model build, so **neither this file nor
``bench/scripts/benchmark_quickstart_sampler.py`` could be rerun at all** until
it was repaired. ``benchmark_quickstart_sampler.py`` has since been repaired too
(#2096) by deleting its model and importing :data:`NOTEBOOKS` from here, so the
registry below is now the single definition every consumer shares --
``diagnose_ghmc_meads.py`` already worked this way.

Repairing it forces a choice that is **physics, not spelling**, and the two
repairs that look identical are not:

* ``law="calzetti"`` sets **both** screens to Calzetti.
* ``law_bc="calzetti"`` alone -- the retired spelling -- left the diffuse screen
  at ``TwoComponentDustConfig.law_diff``'s own default of ``"power_law"``, i.e.
  Charlot & Fall's Calzetti birth cloud over a power-law diffuse screen. Written
  out in full today that is ``law_bc="calzetti", law_diff="power_law"``.

PR #1989 (``176f8fd9d``, "fix(dust,api): attenuation laws are explicit and
required -- law, or law_bc+law_diff", 2026-08-20) rewrote ``05_fitting_photometry``
from the first spelling to the second. Presented and reviewed as an
API-explicitness change, **it moved the physics.** Measured on nb05 at seed 7,
same shipped NUTS call, everything else held::

    law_bc=calzetti + law_diff=power_law   R-hat 1.0043    4 div   ESS 88.1
    law=calzetti (both screens)            R-hat 1.1426  166 div   ESS  3.0

That fixture is far more sensitive to the dust law than to the sampler. So
``bench/reports/2026-08-17_nb01_nb05_nuts_vs_hmc.md``'s published row (R-hat
1.0033, 0 divergences, ESS 144.2) is a correct measurement of the **pre-#1989**
model, and the notebook's surviving convergence narrative is a claim about a
fixture it no longer builds.

Both models are legitimately wanted, so both are first-class named fixtures and
neither is the "real" one:

===========  ==================================================================
``05``       what ``notebooks/05_fitting_photometry.py`` ships **today**
             (``law="calzetti"``). The primary gate fixture -- this is what a
             reader who runs the notebook actually gets.
``05pre``    the **pre-#1989** model (``law_bc="calzetti", law_diff="power_law"``).
             Kept so the published 2026-08-17 table stays reproducible and so
             the dust-law sensitivity is measurable rather than inferred.
``00now``    what ``notebooks/00_quickstart.py`` ships **today** (``dpl`` SFH,
             ONE Calzetti screen, nebular baked into the wNE grid, D=6). The
             live nb00 fixture, added in #2096. See below.
``00``       the pre-#2044 quickstart, ``law="calzetti"``. See below.
``00pre``    the same fixture in the pre-#1989 dust spelling, which is what
             ``benchmark_quickstart_sampler.build_model`` used to literally
             contain. See below.
===========  ==================================================================

The ``00`` family's names are historically inverted with respect to the ``05``
family's: ``05`` is today's notebook and ``05pre`` is the old one, whereas
``00`` is an OLD one and ``00now`` is today's. That is not a preference. ``00``
and ``00pre`` name rows already published in three 2026-08 reports, and
reassigning either name would silently repoint those tables at a different
model -- the exact failure this file exists to prevent.

**A reader comparing against ``bench/reports/2026-08-17_nb01_nb05_nuts_vs_hmc.md``
wants ``05pre``, not ``05``.** The 2026-08-17 report predates #1989.

NB00 IS NOT TODAY'S QUICKSTART
==============================

``_build_nb00`` mirrors the **pre-#2044** quickstart -- ``tsnorm`` SFH,
two-component dust, free metallicity, D=7, 12 bands -- because that is the model
``bench/reports/2026-08-17_quickstart_nuts_vs_hmc.md`` measured and the only
reason to carry nb00 in this harness is to stay comparable with it.
``notebooks/00_quickstart.py`` **today** ships a ``dpl`` SFH with
``single_component`` dust at D=6 against the nebular-baked ``prsc_miles_chabrier_wNE``
grid, which is a different model. Do not read the ``00``/``00pre`` rows as
measuring today's quickstart; they do not. ``00now`` (:func:`_build_nb00_today`)
does, and exists because before #2096 **nothing in the tree tracked the live
quickstart at all** -- which is how #2044 moved it on 2026-08-23 with no row
noticing. Note the SSP grid: #2096 enumerated four differences between
``_build_nb00`` and today's quickstart (SFH family, dust component count,
nebular treatment, dimension) and the grid is a fifth. Fifteen spec keys differ
in total; ``tools/check_harness_parity.py --fixture 00`` prints them.

``benchmark_quickstart_sampler.build_model`` itself built **D = 6** -- it had no
``met`` group -- while its own published table states **D = 7** and names
``met_logzsol`` as the worst-mixing parameter of its HMC L=160 row, so it could
not reproduce its own table. The report's numbers are a correct measurement of a
D = 7 model and stand unchanged; it was the committed builder that drifted away
from them. The free metallicity is restored in both nb00 fixtures here. They
differ in the two things the two independent repairs of this
file disagreed on, and the pair is kept for the same reason ``05``/``05pre`` is:

* ``00`` uses ``law="calzetti"`` on ``00_quickstart``'s own ``met`` range
  ``U(-2.0, 0.2)``. This is the fixture whose rows are published in
  ``bench/reports/2026-08-30_chees_hmc.md`` and
  ``bench/reports/2026-08-30_ghmc_meads_adaptation.md``, and it reproduces the
  2026-08-17 published nb00 row (R-hat 1.0060 against 1.0087, min ESS 229.9
  against 231.5).
* ``00pre`` uses ``law_bc="calzetti", law_diff="power_law"`` on
  ``05_fitting_photometry``'s ``met`` range ``U(-1.5, 0.3)`` -- the literal
  restoration of what ``benchmark_quickstart_sampler.build_model`` used to spell,
  nb05 being described as this model plus ``met_logzsol`` and ``dust_tau_diff``.
  Its rows appear in ``bench/reports/2026-08-30_mclmc_tuning.md``.

Note that the *forward model* moves by about the same amount in both notebooks:
evaluated at one shared parameter point, the two spellings differ by a median
5.1% (max 8.2%) across nb00's 12 bands and a median 4.1% (max 7.3%) across
nb05's 14. What differs is what that change does to the **posterior geometry**.
nb00 pins ``dust_tau_diff`` (FIXED at its declared default of 0.3), so the
diffuse law only shifts a fixed screen; nb05 frees it over ``U(0, 1)``, so the
law reshapes a fitted direction and its degeneracy with ``met_logzsol`` and the
SFH -- which is what turns a 4% flux change into two orders of magnitude of ESS.
The sampling consequence on nb00 has **not** been measured here; only its
forward-model consequence has. It is a real difference either way, not a
spelling one.

HOW A FIXTURE IS HELD TO ITS NOTEBOOK (#2096)
=============================================

Every entry in :data:`NOTEBOOKS` carries a ``parity=`` block naming what it is a
copy of, and ``tools/check_harness_parity.py`` -- gated by
``tests/contract/test_harness_notebook_parity.py`` -- holds it to that claim by
building the notebook's own model and comparing both the canonical parameter
spec and the predicted photometry at a fixed parameter vector. Adding a fixture
without a ``parity=`` block fails the test.

The block also says which fixtures are old **on purpose**. ``05pre``, ``00`` and
``00pre`` are ``kind="historical"``: not checked against today's notebook,
because they are not supposed to match it, but anchored to a live sibling and
required to differ from it in exactly the spec keys they declare. So
``05pre`` -> ``05`` -> ``05_fitting_photometry.py`` and
``00pre`` -> ``00`` -> ``00now`` -> ``00_quickstart.py`` both end at a real
notebook, and "historical" cannot be used to opt out. ``bench/README.md`` has
the failure playbook; the short version is that the harness follows the notebook
and a published measurement is never edited to make a fixture agree with it.

Current state of every fixture, measured rather than asserted: ``05``, ``01``,
``00now`` and ``ctl-jwst`` each match their notebook with a maximum relative
difference of **0.0** in predicted photometry. **nb01 was the last fixture of
unknown status and it is clean** -- and structurally so, since the notebook and
:func:`_build_nb01` both say ``**recipes.mock_recovery_minimal()`` rather than
each spelling a model out. Duplication is what drifts.

THE CONTROLS
============

Two controls, and they are **not** interchangeable. Both were previously called
``ctl`` by different campaigns, so the bare name is deliberately no longer
accepted: a stale ``--notebook ctl`` now fails loudly instead of silently
measuring the other campaign's model.

* ``ctl-dpl`` -- swaps **only** the SFH family against nb05 (DPL instead of
  tsnorm; same 14 bands, mock seed, SNR, chain count, dust). A controlled A/B
  for the tsnorm degeneracy. This is the ``ctl`` of
  ``bench/reports/2026-08-30_chees_hmc.md``.
* ``ctl-jwst`` -- the **healthy** control: a non-tsnorm 9-D ``continuity`` SFH
  against 19 JWST bands at z = 1.5, mirroring
  ``notebooks/jwst_nonparametric_fits.py``. This is the ``ctl`` of
  ``bench/reports/2026-08-30_mclmc_tuning.md``.

**The ``div`` column is not defined for every sampler.** MCLMC is unadjusted --
there is no Metropolis step that could reject, so there is no divergence to
count -- and it prints ``n/a`` with the energy-error variance per dimension
(EEVPD) in its own column instead. A zero in ``div`` would be a claim about a
mechanism the sampler does not have; ``bench/reports/2026-08-17_nb01_nb05_nuts_vs_hmc.md``
already warns that zero divergences is not evidence of convergence for a
fixed-trajectory sampler, and that warning applies with more force here.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import warnings

warnings.filterwarnings("ignore")

import jax
import numpy as np

import tengri
from tengri import (
    FIXED,
    FREE,
    Data,
    Fixed,
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    builders,
    generate_mock,
    recipes,
)
from tengri.analysis.diagnostics.autocorrelation import effective_sample_size
from tengri.config.exceptions import DeadFitError
from tengri.inference.backends.mcmc._shared import total_draws

#: Each notebook's own convergence claim, and so the bar a replacement clears.
#:
#: PRIMARY criterion. This is an *absolute self-assessment* -- what a shipped
#: notebook fit asserts about itself -- and it is reported unchanged.
MAX_RHAT = 1.01
MAX_DIVERGENCES = 0

#: SECONDARY, comparative criterion, applied identically to every sampler
#: including the NUTS baseline.
#:
#: The primary bar was written by the notebooks to describe one fit, and it does
#: not discriminate when it is borrowed as a *ranking* criterion between
#: samplers: NUTS on the healthy DPL control is split-R-hat 1.0002 at min ESS
#: 223 with **17 divergences**, a plainly good fit that "zero divergences" calls
#: a miss, and ChEES rows miss on 1 to 3 divergences out of 1200 draws. A
#: criterion that fails the incumbent and the challenger alike cannot separate
#: them, which is the one job a comparative gate has.
#:
#: 0.5% of TOTAL draws is the replacement threshold for that clause alone. It is
#: stated here rather than tuned: Stan and BlackJAX both treat a handful of
#: divergences in a long chain as worth investigating rather than disqualifying,
#: and 0.5% is comfortably below the 1.4% (17/1200) the NUTS control shows while
#: being orders of magnitude below the 100% of a genuinely dead fit. The R-hat
#: and ESS clauses are unchanged -- only the divergence clause moves, and it
#: moves from a count to a rate because a count is not comparable across
#: configurations with different draw budgets.
#:
#: For an *unadjusted* sampler the clause is vacuous rather than satisfied:
#: there is no accept step, so no divergence can be counted. Those rows report
#: EEVPD beside them instead of folding a missing mechanism into a pass.
MAX_DIVERGENCE_RATE = 0.005

_NB05_FILTERS = (
    "galex_fuv",
    "galex_nuv",
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "2mass_j",
    "2mass_h",
    "2mass_ks",
    "wise_w1",
    "wise_w2",
    "wise_w3",
    "wise_w4",
)
_NB01_FILTERS = ("sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1")
_NB00_FILTERS = (
    "galex_fuv",
    "galex_nuv",
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "2mass_j",
    "2mass_h",
    "2mass_ks",
    "wise_w1",
    "wise_w2",
)

#: The healthy control's bands. ``jwst_nonparametric_fits`` -- a non-tsnorm, 9-D
#: ``continuity`` SFH against 19 JWST bands at z = 1.5 -- re-measured across six
#: seeds in #2014, where NUTS on a diagonal metric returns a median min ESS of
#: 119 in 85 s. Every tsnorm row in this file is degenerate: Finding 15 of
#: ``bench/reports/2026-08-20_cuda_device_matrix.md`` puts their min ESS at 1-4
#: out of 600 draws. A sampler comparison run only on degenerate fixtures cannot
#: separate "this sampler is slow" from "this posterior is hard", because NUTS
#: answers bad geometry by doubling its trajectory -- up to 2^10 leapfrog steps
#: per draw where a healthy posterior needs 2^3-2^5 -- so the wall clock and the
#: non-convergence are one phenomenon, not two.
_CTL_BROAD = (
    "jwst_f090w",
    "jwst_f115w",
    "jwst_f150w",
    "jwst_f200w",
    "jwst_f277w",
    "jwst_f356w",
    "jwst_f444w",
)
_CTL_MEDIUM = (
    "jwst_f140m",
    "jwst_f162m",
    "jwst_f182m",
    "jwst_f210m",
    "jwst_f250m",
    "jwst_f300m",
    "jwst_f335m",
    "jwst_f360m",
    "jwst_f410m",
    "jwst_f430m",
    "jwst_f460m",
    "jwst_f480m",
)
_CTL_Z_GAL = 1.5


def _build_ctl_jwst(ssp):
    """The HEALTHY control: 9-D ``continuity`` SFH, 19 JWST bands, z = 1.5.

    Mirrors ``notebooks/jwst_nonparametric_fits.py`` exactly, including the
    ``tau_bc = 0.0`` pin that #2014 measured as the difference between 176 and 30
    effective samples at the same seed. Its ``ssp_data`` differs from the tsnorm
    rows' (``prsc_miles_chabrier_wNE``, nebular baked in) because the page turns
    nebular emission on.

    This is the fixture ``bench/reports/2026-08-30_mclmc_tuning.md`` calls
    ``ctl``. It is NOT :func:`_build_ctl_dpl`, which a different campaign also
    called ``ctl``; see the module docstring.
    """
    from tengri.cosmology import age_at_z

    t_univ = float(age_at_z(_CTL_Z_GAL))
    bin_edges = np.concatenate([[0.0, 0.03], np.logspace(np.log10(0.1), np.log10(t_univ), 6)])
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(_CTL_BROAD + _CTL_MEDIUM))),
        redshift=Fixed(_CTL_Z_GAL),
        sfh={"type": "continuity", "all_params": FREE, "bin_edges_gyr": bin_edges},
        met={"logzsol": Uniform(-1.5, 0.3), "all_params": FIXED},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
            "tau_bc": 0.0,
            "tau_diff": Uniform(0.0, 2.0),
        },
        neb={"type": "ssp"},
        approx=WavePrecomp(),
    )


def _build_ctl_dpl(ssp):
    """Non-tsnorm control: the same 14 bands as nb05 over a **DPL** SFH. D=8.

    (``bench/reports/2026-08-30_chees_hmc.md`` tabulated this fixture as D=7 and
    ``bench/results/2026-08-30_chees_control.json`` recorded a seven-item
    ``free_params`` list; both omitted ``sfh_dpl_age_gyr``, and both have been
    corrected to D=8 -- the same count as nb05, which is what "otherwise
    identical to nb05" should give. The fixture is unchanged.)

    Notebooks 00, 01 and 05 all run a ``tsnorm`` SFH, and
    ``bench/reports/2026-08-20_cuda_device_matrix.md`` Finding 15 measured that
    family's ``skew``/``trunc``/``width_gyr`` as strongly degenerate: ESS_min
    stays 1.7-4.3 *even with 260 spectral pixels*, so 52x the data moves it only
    1.7 to 4.3 and it is not a data-volume problem.

    A sampler that misses the bar on all three notebooks is therefore
    uninterpretable on its own: "this sampler is wrong" and "this posterior is
    degenerate for everything" predict the same table. This row separates them.
    Dual power law is the family ``recipes.star_forming_photometry`` ships and
    the one the 2026-05-22 backend validation used, so a failure here is about
    the sampler.

    This is the fixture ``bench/reports/2026-08-30_chees_hmc.md`` calls ``ctl``.
    It is NOT :func:`_build_ctl_jwst`, which a different campaign also called
    ``ctl``; see the module docstring.
    """
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(_NB05_FILTERS))),
        approx=WavePrecomp(),
        sfh=builders.sfh.dpl(all_params=FREE),
        dust_attenuation=builders.dust.two_component(
            all_params=FIXED,
            law="calzetti",
            tau_bc=Uniform(0.0, 1.0),
            tau_diff=Uniform(0.0, 1.0),
        ),
        dust_emission=builders.dust.emission.modified_blackbody(all_params=FIXED),
        neb=builders.neb.none(),
        met={"logzsol": Uniform(-1.5, 0.3)},
        redshift=Fixed(0.05),
    )


def _build_nb00(ssp):
    """The **pre-#2044** ``00_quickstart``: tsnorm SFH, two-component Calzetti. D=7.

    NOT what ``notebooks/00_quickstart.py`` ships today -- that is a ``dpl`` SFH
    with ``single_component`` dust at D=6. This fixture exists to stay
    comparable with ``bench/reports/2026-08-17_quickstart_nuts_vs_hmc.md``, which
    measured the older model, and it reproduces that report's published row
    (R-hat 1.0060 against 1.0087, min ESS 229.9 against 231.5). Realigning nb00
    with today's quickstart is deliberately out of scope; see the module
    docstring.

    Free metallicity is what makes this D=7 rather than D=6, and the 2026-08-17
    report's L=160 row names ``met_logzsol`` as its worst-mixing parameter -- so
    it was free there too, even though
    ``benchmark_quickstart_sampler.build_model`` as committed has no ``met``
    group at all. The range is ``00_quickstart``'s own ``U(-2.0, 0.2)``.

    Uses ``law="calzetti"`` -- both screens. :func:`_build_nb00_prelaw` is the
    same fixture in the pre-#1989 spelling. At one shared parameter point the
    two differ by a median 5.1% in predicted flux, comparable to the 4.1% the
    same choice costs nb05 -- but ``dust_tau_diff`` is pinned here and free
    there, so on nb00 the law shifts a fixed screen while on nb05 it reshapes a
    fitted direction. The sampling consequence on nb00 is unmeasured.
    """
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(_NB00_FILTERS))),
        approx=WavePrecomp(),
        sfh=builders.sfh.tsnorm(all_params=FREE),
        dust_attenuation=builders.dust.two_component(
            all_params=FIXED, law="calzetti", tau_bc=Uniform(0.0, 1.0)
        ),
        neb=builders.neb.none(),
        met={"logzsol": Uniform(-2.0, 0.2)},
        redshift=Fixed(0.05),
    )


def _build_nb00_prelaw(ssp):
    """The pre-#2044 quickstart in the **pre-#1989 dust spelling**. D=7.

    The literal restoration of what ``benchmark_quickstart_sampler.build_model``
    still contains: ``law_bc="calzetti"`` alone, which resolved to
    ``TwoComponentDustConfig.law_diff``'s own default of ``"power_law"``. That
    script still carries the retired spelling and so still cannot build; writing
    it out in full is what makes the model runnable again without changing it.

    Differs from :func:`_build_nb00` in exactly two things -- the diffuse
    attenuation law, and ``05_fitting_photometry``'s ``met`` range ``U(-1.5,
    0.3)`` instead of the quickstart's ``U(-2.0, 0.2)``, nb05 being described as
    this model plus ``met_logzsol`` and ``dust_tau_diff``. Both fixtures are
    kept because two independent repairs of this file chose differently and both
    have published rows; see the module docstring.
    """
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(_NB00_FILTERS))),
        approx=WavePrecomp(),
        sfh=builders.sfh.tsnorm(all_params=FREE),
        dust_attenuation=builders.dust.two_component(
            all_params=FIXED, law_bc="calzetti", law_diff="power_law", tau_bc=Uniform(0.0, 1.0)
        ),
        neb=builders.neb.none(),
        met={"logzsol": Uniform(-1.5, 0.3)},
        redshift=Fixed(0.05),
    )


def _build_nb00_today(ssp):
    """``00_quickstart`` **as shipped today**: dpl SFH, ONE Calzetti screen, nebular on. D=6.

    The live nb00 fixture, added in #2096 so the quickstart has something that
    tracks it. ``00`` and ``00pre`` are both pre-#2044 models kept for published
    rows; before this function existed, *nothing in the tree checked
    ``notebooks/00_quickstart.py`` at all*, which is how #2044 moved the
    quickstart on 2026-08-23 without a single row noticing.

    Note the SSP grid: ``prsc_miles_chabrier_wNE``, not the bare-stellar
    ``fsps_prsc_miles_chabrier`` the ``00``/``00pre`` fixtures use. #2096
    enumerated four differences between ``_build_nb00`` and today's quickstart
    (SFH family, dust component count, nebular treatment, dimension); the grid
    is a fifth, and it is the one that makes ``neb={"type": "ssp"}`` mean
    anything -- the nebular contribution is baked into this file at logU = -3.0.

    Requires ``ssp="prsc_miles_chabrier_wNE"`` in its :data:`NOTEBOOKS` entry;
    building it against the bare-stellar grid would silently drop the nebular
    emission and is what ``tools/check_harness_parity.py`` would catch.

    The mock differs from the notebook's on purpose: ``run_one`` draws its truth
    from the prior at a named seed, while the notebook hand-picks a truth
    (alpha 0.5, beta 2.0, tau 5.8 Gyr, logM 10.0, tau_v 0.3, logzsol -0.3)
    chosen to sit on a star-forming plateau. Parity is a claim about the
    *model*, not about which galaxy is fed to it; a harness that copied the
    hand-picked truth could not run the seed sweeps this file exists for.
    """
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(_NB00_FILTERS))),
        approx=WavePrecomp(),
        sfh={"type": "dpl", "all_params": FREE, "age_gyr": 13.1},
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "tau_v": Uniform(0.0, 4.0),
        },
        neb={"type": "ssp"},
        met={"logzsol": Uniform(-2.0, 0.2)},
        redshift=Fixed(0.05),
    )


def _build_nb05(ssp):
    """``05_fitting_photometry`` **as shipped today**: quickstart + logzsol + tau_diff. D=8.

    ``law="calzetti"`` -- both screens Calzetti -- which is what the notebook
    builds on this HEAD, after PR #1989 (``176f8fd9d``) rewrote its dust line.
    This is the primary gate fixture: it is what a reader who runs the notebook
    actually gets.

    It is **not** the model ``bench/reports/2026-08-17_nb01_nb05_nuts_vs_hmc.md``
    measured, and it does not reproduce that report's row. Use ``05pre``
    (:func:`_build_nb05_prelaw`) for that; the module docstring has the
    measured before/after and the history.
    """
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(_NB05_FILTERS))),
        approx=WavePrecomp(),
        sfh=builders.sfh.tsnorm(all_params=FREE),
        dust_attenuation=builders.dust.two_component(
            all_params=FIXED,
            law="calzetti",
            tau_bc=Uniform(0.0, 1.0),
            tau_diff=Uniform(0.0, 1.0),
        ),
        dust_emission=builders.dust.emission.modified_blackbody(all_params=FIXED),
        neb=builders.neb.none(),
        met={"logzsol": Uniform(-1.5, 0.3)},
        redshift=Fixed(0.05),
    )


def _build_nb05_prelaw(ssp):
    """``05_fitting_photometry`` as it stood BEFORE PR #1989. D=8.

    Identical to :func:`_build_nb05` in every parameter and prior, and different
    in exactly one thing: the **diffuse** screen is a power law rather than
    Calzetti. Both are real configurations of the same notebook, three days
    apart, and the pair is the reason this function exists.

    The history, from the notebook's own git log rather than from anyone's
    reconstruction. Until 176f8fd9d ("fix(dust,api): attenuation laws are
    explicit and required -- law, or law_bc+law_diff", #1989, 2026-08-20) nb05
    read::

        dust=builders.dust.two_component(defaults=FIXED, law_bc="calzetti", ...)

    ``law_bc`` alone left the diffuse screen at its declared default of
    ``power_law`` (``components/dust/two_component.py``, ``law_diff: str =
    "power_law"``), i.e. Charlot & Fall's Calzetti birth cloud over a power-law
    diffuse screen. #1989 rewrote it to ``law="calzetti"``, which sets **both**
    screens. Presented and reviewed as an API-explicitness change, it moved the
    physics. That both spellings also had to be rewritten because the ``dust=``
    peer group split into ``dust_attenuation`` / ``dust_emission`` is what made
    the two changes easy to confuse for one.

    The measured consequence, same seed, same shipped NUTS call, everything else
    held::

        law_bc=calzetti + law_diff=power_law   R-hat 1.0043    4 div   ESS 88.1
        law=calzetti (both screens)            R-hat 1.1426  166 div   ESS  3.0

    So ``bench/reports/2026-08-17_nb01_nb05_nuts_vs_hmc.md``'s published row
    (R-hat 1.0033, 0 divergences, ESS 144.2) is a correct measurement of THIS
    model, and the notebook's surviving convergence narrative is a claim about a
    fixture it no longer builds. :func:`_build_nb05` is what a reader runs today
    and is the primary gate fixture; this one exists so the published table stays
    reproducible and so the dust-law sensitivity is measurable rather than
    inferred. **A reader comparing against the 2026-08-17 report wants this
    fixture.**
    """
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(_NB05_FILTERS))),
        approx=WavePrecomp(),
        sfh=builders.sfh.tsnorm(all_params=FREE),
        dust_attenuation=builders.dust.two_component(
            all_params=FIXED,
            law_bc="calzetti",
            law_diff="power_law",
            tau_bc=Uniform(0.0, 1.0),
            tau_diff=Uniform(0.0, 1.0),
        ),
        dust_emission=builders.dust.emission.modified_blackbody(all_params=FIXED),
        neb=builders.neb.none(),
        met={"logzsol": Uniform(-1.5, 0.3)},
        redshift=Fixed(0.05),
    )


def _build_nb01(ssp):
    """``01_why_jax``: the minimal mock-recovery recipe, six bands.

    Dimensionality is whatever ``recipes.mock_recovery_minimal`` currently
    declares -- the recipe owns it, this file does not restate it -- and the
    header line prints the measured count per run.

    **Settled (#2096): this fixture matches the notebook, and structurally
    cannot drift from it.** ``tools/check_harness_parity.py`` measures the two
    models as spec-identical with a maximum relative difference of 0.0 in
    predicted photometry across all six bands. That is not luck: nb05 and nb00
    drifted because each *spelled its model out* in two places, whereas
    ``01_why_jax.py`` and this function both say
    ``**recipes.mock_recovery_minimal()`` -- one definition, in ``src/``, that
    changes for both at once. The six filters are the only thing duplicated, and
    ``notebooks/01_why_jax.py``'s ``SEDModel.build`` line has not changed since
    ``27ffb8d0d`` ("docs(nb01): rewrite why-JAX for astronomers") apart from
    formatting and the ``load_ssp`` path resolver (#1486). #2096 listed nb01 as
    "unchecked", which was accurate, and as the last fixture of unknown status,
    which it no longer is.

    The mock differs from the notebook's: the notebook draws its truth at
    ``PRNGKey(0)`` and its noise at ``PRNGKey(1)``, while ``run_one`` splits one
    seed three ways. Parity is a claim about the model, not the galaxy.
    """
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(_NB01_FILTERS))),
        **recipes.mock_recovery_minimal(),
    )


#: Per-notebook setup. ``shipped`` mirrors the notebook's committed fit call, so
#: the baseline row is what a reader actually runs -- not a tuned stand-in.
#:
#: There is deliberately no ``"ctl"`` key. Two campaigns used that name for two
#: different models (``ctl-dpl`` and ``ctl-jwst``), so a stale ``--notebook ctl``
#: must fail loudly rather than silently measure the other campaign's fixture.
#:
#: **Every entry must carry a ``parity=`` block** (#2096). It declares what the
#: fixture is a copy of, and ``tools/check_harness_parity.py`` --- run by
#: ``tests/contract/test_harness_notebook_parity.py`` --- holds it to that claim:
#:
#: * ``kind="mirrors"`` with ``notebook=`` --- must build the same model as that
#:   notebook, checked against the notebook's own executed code.
#: * ``kind="historical"`` with ``anchor=``, ``differs_in=`` and
#:   ``superseded_by=`` --- reproduces a superseded model on purpose, so it must
#:   differ from its anchor in exactly the declared spec keys and no others. The
#:   anchor chain must end at a ``mirrors`` fixture, which is what keeps
#:   "historical" from becoming an exemption.
#: * ``kind="standalone"`` with ``why=`` --- not a copy of anything.
#:
#: There is no default. A fixture added without a ``parity=`` block fails the
#: contract test, because a fixture that never said what it mirrors is exactly
#: the defect #2096 reports.
NOTEBOOKS = {
    "00": dict(
        build=_build_nb00,
        parity=dict(
            kind="historical",
            anchor="00now",
            superseded_by="#2044 (36d7189cf, 2026-08-23)",
            differs_in=(
                "dust_attenuation.Rv",
                "dust_attenuation.all_params",
                "dust_attenuation.bump_strength",
                "dust_attenuation.delta",
                "dust_attenuation.f_obscuration",
                "dust_attenuation.slope",
                "dust_attenuation.tau_bc",
                "dust_attenuation.tau_v",
                "dust_attenuation.type",
                "free_params",
                "neb.type",
                "sfh.age_gyr",
                "sfh.type",
                "ssp.file",
                "ssp.nebular",
            ),
            why=(
                "The pre-#2044 quickstart. Fifteen spec keys differ from today's "
                "because #2044 replaced the SFH family, the dust component, the "
                "nebular treatment, the SSP grid and the dimension in one commit. "
                "Kept because 2026-08-17_quickstart_nuts_vs_hmc.md and the two "
                "2026-08-30 reports measured THIS model."
            ),
        ),
        # PRNGKey(9) at SNR 30, not this file's usual (1, 20): these are
        # ``benchmark_quickstart_sampler.py``'s own values, and the whole reason
        # to carry nb00 here is that its rows stay comparable with
        # 2026-08-17_quickstart_nuts_vs_hmc.md. Changing them silently would
        # produce a different mock and a baseline that looks like a regression.
        seed=9,
        snr=30.0,
        n_chains=4,
        dense_hmc=True,
        shipped=dict(
            method="mcmc_nuts",
            n_warmup=1500,
            n_samples=250,
            n_burnin=0,
            dense_mass_matrix=False,
            target_accept_rate=0.9,
        ),
        note=(
            "The PRE-#2044 quickstart, not the one the notebook ships today "
            "(which is dpl + single_component at D=6). Mirrors "
            "benchmark_quickstart_sampler.py's own model, seed and NUTS row so "
            "the table stays comparable with 2026-08-17_quickstart_nuts_vs_hmc.md."
        ),
    ),
    "00now": dict(
        build=_build_nb00_today,
        parity=dict(kind="mirrors", notebook="notebooks/00_quickstart.py"),
        ssp="prsc_miles_chabrier_wNE",
        # The notebook's own PRNGKey(6), SNR and 4 chains. Its truth is
        # hand-picked rather than a prior draw (see the builder's docstring), so
        # this row's mock is the harness's, not the notebook's.
        seed=6,
        snr=30.0,
        n_chains=4,
        shipped=dict(
            method="mcmc_hmc",
            n_warmup=200,
            n_samples=300,
            n_burnin=0,
            n_leapfrog_steps=50,
            dense_mass_matrix=False,
            target_accept_rate=0.85,
            precondition=True,
        ),
        note=(
            "00_quickstart AS SHIPPED TODAY (dpl SFH, single_component "
            "Calzetti, nebular baked into the wNE grid, D=6). The only fixture "
            "here that tracks the live quickstart; '00' and '00pre' are both "
            "pre-#2044 and do NOT. No published row measures this fixture yet "
            "-- it exists so #2044 cannot happen again unnoticed. Its shipped "
            "row is fixed-length HMC, not NUTS, because that is the notebook's "
            "committed fit."
        ),
    ),
    "00pre": dict(
        build=_build_nb00_prelaw,
        parity=dict(
            kind="historical",
            anchor="00",
            superseded_by="#1989 (176f8fd9d, 2026-08-20)",
            differs_in=(
                "dust_attenuation.law",
                "dust_attenuation.law_bc",
                "dust_attenuation.law_diff",
                "sfh.logzsol",
            ),
            why=(
                "The pre-#1989 dust spelling of the pre-#2044 quickstart, on "
                "05_fitting_photometry's met range. Anchored to '00' rather than "
                "to '00now' because the one change it isolates is the dust law; "
                "the chain 00pre -> 00 -> 00now -> 00_quickstart.py is what "
                "grounds it in a live notebook."
            ),
        ),
        # nb00's seed, SNR and chain count exactly: this row differs from "00"
        # in the diffuse dust law and the met range and in nothing else.
        seed=9,
        snr=30.0,
        n_chains=4,
        dense_hmc=True,
        shipped=dict(
            method="mcmc_nuts",
            n_warmup=1500,
            n_samples=250,
            n_burnin=0,
            dense_mass_matrix=False,
            target_accept_rate=0.9,
        ),
        note=(
            "The pre-#2044 quickstart in the PRE-#1989 dust spelling "
            "(law_bc='calzetti' + law_diff='power_law'), which is what "
            "benchmark_quickstart_sampler.build_model still literally contains. "
            "Use '00' for the both-screens-Calzetti reading."
        ),
    ),
    "01": dict(
        build=_build_nb01,
        parity=dict(kind="mirrors", notebook="notebooks/01_why_jax.py"),
        seed=1,
        snr=20.0,
        n_chains=4,
        shipped=dict(method="mcmc_nuts", n_warmup=100, n_samples=100),
        note=(
            "The committed fit is labelled in the notebook as a timing "
            "demonstration, NOT a converged posterior -- it produces one bar in "
            "a chart against an emcee literature baseline. Read the R-hat column "
            "accordingly; 100 warmup is not meant to clear any bar."
        ),
    ),
    "05": dict(
        build=_build_nb05,
        parity=dict(kind="mirrors", notebook="notebooks/05_fitting_photometry.py"),
        seed=7,
        snr=20.0,
        n_chains=2,
        shipped=dict(method="mcmc_nuts", n_warmup=600, n_samples=600),
        note=(
            "05_fitting_photometry AS SHIPPED TODAY (law='calzetti', both "
            "screens, post-#1989). Does NOT reproduce the published "
            "2026-08-17 report -- use '05pre' for that. D=8 with "
            "dense_mass_matrix=True is the configuration CLAUDE.md records "
            "peaking at 20+ GB in NUTS warmup, so HMC rows here run "
            "dense_mass_matrix=False unless --dense is passed."
        ),
    ),
    "05pre": dict(
        build=_build_nb05_prelaw,
        parity=dict(
            kind="historical",
            anchor="05",
            superseded_by="#1989 (176f8fd9d, 2026-08-20)",
            differs_in=(
                "dust_attenuation.law",
                "dust_attenuation.law_bc",
                "dust_attenuation.law_diff",
            ),
            why=(
                "nb05 before #1989 rewrote law_bc='calzetti' to law='calzetti'. "
                "Exactly one physical change -- the diffuse screen -- and the "
                "three keys are the two spellings of it. Kept so "
                "2026-08-17_nb01_nb05_nuts_vs_hmc.md stays reproducible."
            ),
        ),
        # nb05's own seed, SNR and chain count exactly. This row differs from
        # "05" in the diffuse dust law and in nothing else, so the pair
        # isolates PR #1989's physics change.
        seed=7,
        snr=20.0,
        n_chains=2,
        shipped=dict(method="mcmc_nuts", n_warmup=600, n_samples=600),
        note=(
            "05_fitting_photometry as it stood BEFORE PR #1989 changed "
            "law_bc='calzetti' to law='calzetti'. The model the published "
            "2026-08-17 report measured, kept so that table stays reproducible. "
            "NOT what the notebook builds today -- use '05' for that."
        ),
    ),
    "ctl-dpl": dict(
        build=_build_ctl_dpl,
        parity=dict(
            kind="standalone",
            why=(
                "not a notebook: nb05's bands, mock and dust over a DPL SFH, so "
                "an SFH-family effect can be told apart from a sampler effect. "
                "Nothing upstream to mirror."
            ),
        ),
        # nb05's seed, SNR and chain count exactly: this row is a CONTROL for
        # the SFH family, so everything else must be held fixed or it controls
        # for nothing.
        seed=7,
        snr=20.0,
        n_chains=2,
        shipped=dict(method="mcmc_nuts", n_warmup=600, n_samples=600),
        note=(
            "NOT a notebook. The non-tsnorm control: nb05's mock, bands, seed, "
            "SNR and chain count over a DPL SFH instead of tsnorm. Exists so a "
            "sampler failure on 00/01/05 can be told apart from the tsnorm "
            "family's own degeneracy (2026-08-20_cuda_device_matrix.md, "
            "Finding 15). Called 'ctl' in 2026-08-30_chees_hmc.md."
        ),
    ),
    "ctl-jwst": dict(
        build=_build_ctl_jwst,
        # Its docstring claims to mirror the page "exactly"; kind="mirrors"
        # turns that sentence into something a test can fail on. It does mirror
        # it: spec-identical, max relative flux difference 0.0.
        parity=dict(kind="mirrors", notebook="notebooks/jwst_nonparametric_fits.py"),
        ssp="prsc_miles_chabrier_wNE",
        seed=4,
        snr=20.0,
        n_chains=2,
        shipped=dict(
            method="mcmc_nuts",
            n_warmup=1000,
            n_samples=400,
            dense_mass_matrix=False,
        ),
        note=(
            "NOT a notebook in this series. The HEALTHY control, and the only "
            "non-tsnorm, non-photometry-degenerate row here: read every other "
            "row against it, because a sampler comparison run only on "
            "degenerate posteriors measures the fixture, not the sampler. "
            "Called 'ctl' in 2026-08-30_mclmc_tuning.md."
        ),
    ),
}


#: Sampler families a row can belong to. ``--methods`` selects a subset.
FAMILIES = ("nuts", "hmc", "ghmc", "chees", "mclmc")


def shipped_family(cfg: dict) -> str:
    """The sampler family a fixture's committed notebook fit belongs to.

    Every fixture here used to ship NUTS, so the baseline row could be labelled
    ``"nuts (shipped)"`` unconditionally. ``00now`` does not -- today's
    ``00_quickstart`` commits fixed-length HMC at L=50 with the analytic
    preconditioner -- and a row labelled ``nuts`` that ran HMC is the same class
    of quiet mislabeling as a fixture named for a notebook it no longer
    mirrors. Derived rather than stored, so it cannot disagree with ``shipped``.
    """
    return {"mcmc_nuts": "nuts", "mcmc_hmc": "hmc", "mcmc_chees": "chees"}.get(
        cfg["shipped"]["method"], cfg["shipped"]["method"].removeprefix("mcmc_")
    )


def shipped_label(cfg: dict) -> str:
    """The baseline row's label, e.g. ``"nuts (shipped)"`` or ``"hmc (shipped)"``."""
    return f"{shipped_family(cfg)} (shipped)"


def configurations(nb: str, quick: bool, dense: bool, families=FAMILIES) -> dict[str, dict]:
    """Sampler recipes to compare, keyed by label."""
    cfg = NOTEBOOKS[nb]
    shipped = dict(cfg["shipped"])
    if quick:
        shipped["n_warmup"] = min(shipped["n_warmup"], 300)
        shipped["n_samples"] = min(shipped["n_samples"], 150)

    configs = {}
    if shipped_family(cfg) in families:
        configs[shipped_label(cfg)] = shipped

    draws = 150 if quick else max(600, shipped["n_samples"])
    warmup = 300 if quick else 1000
    if "hmc" in families:
        leapfrogs = (20, 40) if quick else (20, 40, 80, 160)
        for leapfrog in leapfrogs:
            configs[f"hmc L={leapfrog}"] = dict(
                method="mcmc_hmc",
                n_warmup=warmup,
                n_samples=draws,
                n_leapfrog_steps=leapfrog,
                dense_mass_matrix=dense or cfg.get("dense_hmc", False),
                target_accept_rate=0.9,
            )
    if "ghmc" in families:
        # GHMC is one leapfrog per step, so a step is ~L times cheaper than an
        # HMC row's and the draw budget is scaled up to match: comparing a
        # 600-draw GHMC against a 600-draw L=160 HMC would be comparing samplers
        # given 160x different gradient budgets. Warmup is the MEADS ensemble's,
        # priced at n_warmup * n_ensemble gradients.
        #
        # ``allow_unvalidated`` is required while mcmc_ghmc is tier="broken" --
        # which is exactly the claim these rows exist to settle. Remove it if
        # and only if the tier moves.
        for ensemble in (32,) if quick else (32, 64):
            configs[f"ghmc meads E={ensemble}"] = dict(
                method="mcmc_ghmc",
                n_warmup=warmup,
                n_burnin=200 if quick else 500,
                n_samples=draws * 4,
                n_ensemble=ensemble,
                allow_unvalidated=True,
            )
    if "chees" in families:
        # Two rows, and the pair IS the experiment. ChEES with
        # `mass_matrix_estimation=None` has no metric of its own -- the geometry
        # comes from tengri's analytic `J^T N^-1 J + I` or from nowhere -- so
        # "did preconditioning help?" cannot be answered by one row.
        #
        # The draw budget matches the HMC rows rather than GHMC's x4: a ChEES
        # step is a full L-leapfrog HMC proposal, not GHMC's single leapfrog, so
        # equal draws is already roughly equal gradient budget.
        # Three arms, and the third is not redundant with the second.
        # ``precondition=True`` resolves to DEFAULT_WHITENING_STRENGTH = 0.5, so
        # a metric of condition 1e6 is whitened only to 1e3. ``1.0`` is the full
        # whitening that actually drives the condition number to 1 at the
        # expansion point -- and the one #1442 warns amplifies a *misspecified*
        # metric without bound. Which of those two effects dominates on these
        # posteriors is a measurement, not a preference.
        for label, precondition in (
            ("chees", None),
            ("chees+precond", True),
            ("chees+full", 1.0),
        ):
            configs[label] = dict(
                method="mcmc_chees",
                n_warmup=warmup,
                n_burnin=200 if quick else 500,
                n_samples=draws,
                n_ensemble=32,
                precondition=precondition,
            )
    if "mclmc" in families:
        # MCLMC draws are single integrator steps, not trajectories, so its
        # n_samples is an order of magnitude larger than NUTS's *by construction*
        # and not by generosity: successive draws sit ~L/step_size ~ 40-50 steps
        # apart. Two budgets, so the report can show the dependence rather than
        # assert a number. `allow_unvalidated` because the backend is quarantined
        # until a campaign like this one clears it.
        for label, (mclmc_warmup, mclmc_draws) in {
            "mclmc": (5000, 20000),
            "mclmc 2x": (5000, 40000),
        }.items():
            configs[label] = dict(
                method="mcmc_mclmc",
                n_warmup=300 if quick else mclmc_warmup,
                n_samples=1000 if quick else mclmc_draws,
                allow_unvalidated=True,
            )
    return configs


def _gradients_per_draw(diag: dict) -> float | None:
    """Gradient evaluations one draw cost, or None when the sampler cannot say.

    The load-independent half of the comparison, and the one that survives a
    shared box. NUTS reports ``tree_depth_mean``, and a tree of mean depth d is
    2**d leapfrog steps, each one gradient: that is how it answers bad geometry,
    by spending more of them, up to 2**10. MCLMC cannot answer geometry at all --
    the McLachlan integrator is two gradients per step and one step per draw,
    always -- so this column is the whole structural difference between them in
    one number.
    """
    if "tree_depth_mean" in diag:
        return float(2.0 ** diag["tree_depth_mean"])
    if "energy_var_per_dim" in diag:  # MCLMC: one isokinetic McLachlan step
        return 2.0
    if diag.get("n_leapfrog_steps") is not None:
        return float(diag["n_leapfrog_steps"])
    return None


def _unique_draw_fraction(posterior) -> float:
    """Fraction of the joint draws that are distinct positions.

    Zero divergences is not evidence of health. ``mcmc_nuts`` returned a
    *completely frozen* chain on 3.1% of galaxies with zero divergences reported
    (#1999): every proposal rejected, R-hat near 1.0 because within- and
    between-chain variance are both zero, and nothing in the divergence column
    to see. Split R-hat cannot detect that and neither can ESS on its own, so
    the count of distinct rows is carried as its own column. Healthy chains sit
    near 1.0; anything far below it is a chain that stopped moving.
    """
    keys = sorted(posterior.samples)
    if not keys:
        return float("nan")
    matrix = np.column_stack([np.asarray(posterior.samples[k]).ravel() for k in keys])
    return float(len(np.unique(matrix, axis=0)) / matrix.shape[0])


def _fmt_rhat(value: float) -> str:
    """Four decimals near the bar, scientific once a chain has actually diverged.

    A fixed ``.4f`` was fine while every row was near 1.0. It is not: a NUTS row
    on nb05's seed-0 mock reported R-hat 1.4e13, which printed 19 characters wide
    and ran into the neighboring column, so the whole line became unreadable
    exactly when it had the most to say.
    """
    if not np.isfinite(value):
        return f"{'n/a':>10}"
    return f"{value:>10.4f}" if abs(value) < 1e4 else f"{value:>10.3e}"


def score(posterior, wall: float) -> dict:
    """Diagnostics that decide adoption, plus the parameter that mixes worst.

    ``divergences`` is ``None``, not ``0``, for a sampler whose diagnostics
    carry no ``n_divergent`` key. An unadjusted sampler has no accept step, so
    the count does not exist; reporting a zero would read as "no divergences
    were found" when the truth is that none could be. Those runs report
    ``eevpd`` instead -- the achieved energy-error variance per dimension,
    against the target the tuner aimed at.
    """
    diag = posterior.diagnostics or {}
    rhats = posterior.rhat()
    grad_per_draw = _gradients_per_draw(diag)
    ess = effective_sample_size({k: np.asarray(v) for k, v in posterior.samples.items()})
    finite = [(k, v["ess"]) for k, v in ess.items() if np.isfinite(v["ess"])]
    worst_name, worst_ess = (
        min(finite, key=lambda pair: pair[1]) if finite else ("?", float("nan"))
    )
    return {
        "wall": wall,
        "rhat": max(float(v) for v in rhats.values()) if rhats else float("nan"),
        "divergences": (None if "n_divergent" not in diag else int(diag["n_divergent"] or 0)),
        "eevpd": diag.get("energy_var_per_dim"),
        "eevpd_target": diag.get("energy_var_per_dim_target"),
        "nonfinite_steps": diag.get("n_nonfinite_steps"),
        "min_ess": worst_ess,
        "worst": worst_name,
        "sec_per_ess": wall / max(worst_ess, 1e-9),
        "unique_frac": _unique_draw_fraction(posterior),
        # The denominator a divergence RATE needs, and the reason it is recorded
        # rather than recomputed: every backend stores ``n_samples`` PER CHAIN
        # while ``n_divergent`` is summed over the flattened
        # ``(n_chains * n_samples,)`` record, so ``n_divergent / n_samples`` is
        # n_chains times too large (#2087). ``total_draws`` is the shared helper
        # that fixes it.
        "n_draws_total": total_draws(diag),
        # Leapfrog steps per proposal actually in effect. Deliberately NOT named
        # "learned": mcmc_hmc reports its hand-set L under the same diagnostics
        # key, and the whole point of the ChEES rows is that theirs was not set.
        # Which it is comes from the config label; this column says what it was.
        # None for samplers with no trajectory at all (NUTS reports a tree depth,
        # not a length).
        "n_leapfrog": diag.get("n_leapfrog_steps"),
        "step_size": diag.get("step_size"),
        # NUTS only. The gradient cost of a NUTS draw is ~2**tree_depth - 1
        # leapfrog steps, so this is the column that says whether a slow fit is
        # a slow sampler or a posterior forcing the tree deeper. A healthy
        # geometry sits at depth 3-5 (7-31 leapfrogs); saturation at
        # max_num_doublings means every draw paid the cap.
        "tree_depth_mean": diag.get("tree_depth_mean"),
        "tree_depth_max": diag.get("tree_depth_max"),
        "frac_max_depth": diag.get("frac_max_depth"),
        "grad_per_draw": grad_per_draw,
        "grad_per_ess": (
            None
            if grad_per_draw is None
            else grad_per_draw * total_draws(diag) / max(worst_ess, 1e-9)
        ),
        "leapfrogs_per_draw": (
            2.0 ** diag["tree_depth_mean"] - 1.0
            if diag.get("tree_depth_mean") is not None
            else diag.get("n_leapfrog_steps")
        ),
    }


def divergence_rate(row: dict) -> float | None:
    """Divergences as a fraction of TOTAL draws, or None when undefined.

    ``None`` for an unadjusted sampler, which has no accept step and therefore
    no divergence to count -- not zero. See :data:`MAX_DIVERGENCE_RATE`.
    """
    if row.get("divergences") is None:
        return None
    return row["divergences"] / max(row.get("n_draws_total") or 0, 1)


def clears_bar(row: dict) -> bool:
    """PRIMARY criterion: the notebooks' own bar, with the divergence clause honest.

    ``divergences is None`` means the sampler cannot report divergences, so the
    clause is vacuous rather than satisfied; the energy diagnostics are what
    substitute for it and they are printed beside the row, not folded into a
    pass/fail.
    """
    if row.get("dead_fit"):
        return False
    if not (row["rhat"] < MAX_RHAT):
        return False
    return row["divergences"] is None or row["divergences"] <= MAX_DIVERGENCES


def clears_bar_comparative(row: dict, baseline_ess: float) -> bool:
    """SECONDARY criterion: R-hat, divergence RATE, and min ESS against the baseline.

    Applied identically to the NUTS baseline row. See
    :data:`MAX_DIVERGENCE_RATE` for why the divergence clause is a rate.
    """
    if row.get("dead_fit"):
        return False
    if not (row["rhat"] < MAX_RHAT):
        return False
    if row["min_ess"] < baseline_ess:
        return False
    rate = divergence_rate(row)
    return rate is None or rate < MAX_DIVERGENCE_RATE


def run_one(nb: str, label: str, kwargs: dict, seed: int) -> dict:
    """Build the notebook's mock at ``seed``, MAP-seed it, run one fit, score it."""
    cfg = NOTEBOOKS[nb]
    ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)
    sed = cfg["build"](ssp)
    key_truth, key_mock, key_fit = jax.random.split(jax.random.PRNGKey(seed), 3)
    mock = generate_mock(sed, sed.spec.sample(key_truth), key=key_mock, snr=cfg["snr"])
    data = Data(photometry=(np.asarray(mock["flux_obs"]), np.asarray(mock["noise"])))

    # A fresh model per row: adaptation caches are keyed on tuning settings
    # (#1853), and a fresh build also keeps the MAP seed identical per row.
    forward = ForwardModel.build(sed=cfg["build"](ssp))
    map_seed = forward.fit(
        data, method="map", key=key_fit, n_restarts=8, n_steps=800, verbose=False
    )
    started = time.perf_counter()
    try:
        posterior = forward.fit(
            data,
            key=key_fit,
            init_from=map_seed,
            n_chains=cfg["n_chains"],
            verbose=False,
            **kwargs,
        )
    except DeadFitError as exc:
        # A refusal is an OUTCOME, not a missing value and not a harness
        # failure (#2088). Since PR #2090 the window-adaptation backends refuse
        # to sample when >= 90% of the final warmup window diverged, where they
        # previously returned a frozen posterior and a warning. Recording it as
        # a row is what keeps the baseline column honest: "the library refused
        # to hand this back" is a stronger statement about a sampler than any
        # R-hat it could have printed, and folding it into a blank cell would
        # quietly improve the baseline -- and it keeps the seed in the
        # denominator of a seed sweep.
        return {
            "wall": time.perf_counter() - started,
            "rhat": float("inf"),
            "divergences": None,
            "dead_fit": True,
            "dead_fit_reason": str(exc)[:300],
            "warmup_divergence_frac": getattr(exc, "warmup_divergence_frac", None),
            "eevpd": None,
            "eevpd_target": None,
            "nonfinite_steps": None,
            "min_ess": 0.0,
            "worst": "REFUSED (DeadFitError)",
            "sec_per_ess": float("inf"),
            "unique_frac": 0.0,
            "n_draws_total": None,
            "n_leapfrog": None,
            "step_size": getattr(exc, "step_size", None),
            "tree_depth_mean": None,
            "tree_depth_max": None,
            "frac_max_depth": None,
            "grad_per_draw": None,
            "grad_per_ess": None,
            "leapfrogs_per_draw": None,
        }
    return score(posterior, time.perf_counter() - started)


def format_row(label: str, row: dict) -> str:
    """One table line, with ``n/a`` where a column does not apply to the sampler."""
    if row.get("dead_fit"):
        frac = row.get("warmup_divergence_frac")
        tail = "" if frac is None else f" ({frac} of the final warmup window divergent)"
        return f"{label:<20}{row['wall']:>9.1f}{'REFUSED (DeadFitError)':>44}{tail}"
    div = "n/a" if row["divergences"] is None else str(row["divergences"])
    gpd = "" if row.get("grad_per_draw") is None else f"{row['grad_per_draw']:>8.1f}"
    gpe = "" if row.get("grad_per_ess") is None else f"{row['grad_per_ess']:>10.0f}"
    eevpd = "" if row.get("eevpd") is None else f"  EEVPD {row['eevpd']:.2e}"
    return (
        f"{label:<20}{row['wall']:>9.1f}{_fmt_rhat(row['rhat'])}{div:>5}"
        f"{row['min_ess']:>9.1f}{row['sec_per_ess']:>9.3f}{row['unique_frac']:>7.3f}"
        f"{gpd}{gpe}  {row['worst']}{eevpd}"
    )


def _append_json(path: str, notebook: str, seed: int, label: str, row: dict) -> None:
    """One JSON object per line, appended.

    ``score_chees_campaign.py`` consumes this format and keys on
    ``(notebook, config, seed)``, so those three fields are not optional.
    Append-only, so a re-run of one cell supersedes the earlier line rather than
    corrupting the file.
    """
    with open(path, "a") as fh:
        fh.write(json.dumps({"notebook": notebook, "seed": seed, "config": label, **row}) + "\n")


def _sweep(args, configs: dict[str, dict]) -> None:
    """Driver: one fit per subprocess, ``args.seeds`` seeds per row.

    A subprocess per fit is not fastidiousness. Adaptation and MAP caches live
    on the Model and are content-keyed, so two rows in one process can share an
    entry that the second row's settings should have invalidated (#1853), and a
    seed sweep is precisely the shape that trips it. A fresh interpreter is the
    only guarantee that the row measured is the row requested.
    """
    seeds = [NOTEBOOKS[args.notebook]["seed"] + i for i in range(args.seeds)]
    results: dict[str, list[dict]] = {}
    for label in configs:
        results[label] = []
        for seed in seeds:
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--notebook",
                    args.notebook,
                    "--only",
                    label,
                    "--seed",
                    str(seed),
                    "--emit-json",
                    *(["--quick"] if args.quick else []),
                    *(["--dense"] if args.dense else []),
                ],
                capture_output=True,
                text=True,
                env={**os.environ, "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "cpu")},
            )
            if proc.returncode != 0:
                print(f"{label:<20} seed {seed}: FAILED\n{proc.stderr[-2000:]}", flush=True)
                continue
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            results[label].append(row)
            print(f"  seed {seed}: " + format_row(label, row), flush=True)
            if args.json:
                _append_json(args.json, args.notebook, seed, label, row)

    print("\nper-row summary over seeds (worst seed decides the bar):")
    header = (
        f"{'config':<20}{'seeds':>6}{'maxRhat':>10}{'div':>5}"
        f"{'minESS':>9}{'medWall':>9}  worst param"
    )
    print(header)
    print("-" * len(header))
    for label, rows in results.items():
        if not rows:
            print(f"{label:<20}  no successful seeds")
            continue
        dead = [r for r in rows if r.get("dead_fit")]
        live = [r for r in rows if not r.get("dead_fit")]
        if not live:
            print(f"{label:<20}{len(rows):>6}  every seed REFUSED (DeadFitError)")
            continue
        worst = max(live, key=lambda r: r["rhat"])
        div_vals = [r["divergences"] for r in live if r["divergences"] is not None]
        div = "n/a" if not div_vals else str(max(div_vals))
        min_ess = min(r["min_ess"] for r in live)
        med_wall = float(np.median([r["wall"] for r in rows]))
        eevpd = [r["eevpd"] for r in rows if r.get("eevpd") is not None]
        tail = f"  max EEVPD {max(eevpd):.2e}" if eevpd else ""
        if dead:
            tail += f"  [{len(dead)}/{len(rows)} seeds REFUSED]"
        print(
            f"{label:<20}{len(rows):>6}{_fmt_rhat(worst['rhat'])}{div:>5}"
            f"{min_ess:>9.1f}{med_wall:>9.1f}  {worst['worst']}{tail}"
        )
        passes = sum(clears_bar(r) for r in rows)
        print(f"{'':20}  clears bar on {passes}/{len(rows)} seeds")

    if args.sweep_json:
        with open(args.sweep_json, "w") as fh:
            json.dump({"notebook": args.notebook, "seeds": seeds, "rows": results}, fh, indent=2)
        print(f"\nwrote {args.sweep_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=sorted(NOTEBOOKS), required=True)
    parser.add_argument("--quick", action="store_true", help="shorter chains for a smoke run")
    parser.add_argument(
        "--dense", action="store_true", help="dense mass matrix on the HMC rows (memory-hungry)"
    )
    parser.add_argument(
        "--methods",
        default=",".join(FAMILIES),
        help=(
            f"comma-separated subset of sampler FAMILIES {FAMILIES}. Applied "
            "before --only, which filters the resulting labels."
        ),
    )
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "comma-separated exact config LABELS to run, e.g. "
            "'nuts (shipped),mclmc'. Applied after --methods; use --methods to "
            "select whole families instead."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "override the notebook's own seed. The campaign protocol is six seeds "
            "per row, ONE FIT PER SUBPROCESS -- a shared process reuses the "
            "adaptation cache and the compile cache across seeds, so the second "
            "seed onward is not an independent measurement. --seeds does this."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=0,
        help="run each row across this many consecutive seeds, one fit per subprocess",
    )
    parser.add_argument(
        "--json",
        default=None,
        help=(
            "append one JSON row per (notebook, config, seed) to this file, one "
            "object per line. This is the format score_chees_campaign.py and "
            "run_ghmc_meads_campaign.py consume; it is written in both "
            "single-run and --seeds mode."
        ),
    )
    parser.add_argument(
        "--sweep-json",
        default=None,
        help=(
            "--seeds only: write the whole sweep as one nested JSON document "
            "here. Spelled --json on fix/mclmc-tuning; renamed because --json "
            "is the append-only JSONL that the campaign scorers parse."
        ),
    )
    parser.add_argument(
        "--emit-json", action="store_true", help="internal: print one JSON row and exit"
    )
    args = parser.parse_args()

    families = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    unknown = set(families) - set(FAMILIES)
    if unknown:
        parser.error(f"unknown --methods entries {sorted(unknown)}; choose from {FAMILIES}")

    cfg = NOTEBOOKS[args.notebook]
    seed = cfg["seed"] if args.seed is None else args.seed
    configs = configurations(args.notebook, args.quick, args.dense, families)
    if args.only:
        wanted = [label.strip() for label in args.only.split(",")]
        missing = [label for label in wanted if label not in configs]
        if missing:
            parser.error(
                f"unknown config(s) {missing}; available under --methods "
                f"{','.join(families)}: {sorted(configs)}"
            )
        configs = {label: configs[label] for label in wanted}

    if args.emit_json:
        label, kwargs = next(iter(configs.items()))
        print(json.dumps(run_one(args.notebook, label, kwargs, seed)))
        return

    ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)
    sed = cfg["build"](ssp)
    print(
        f"notebook {args.notebook}: D = {len(sed.spec.free_params)} free parameters, "
        f"{cfg['n_chains']} chains, seed {seed}"
    )
    print(f"adoption bar: max split R-hat < {MAX_RHAT} and {MAX_DIVERGENCES} divergences")
    print(f"note: {cfg['note']}\n")

    if args.seeds:
        _sweep(args, configs)
        return

    header = (
        f"{'config':<20}{'wall s':>9}{'maxRhat':>10}{'div':>5}"
        f"{'minESS':>9}{'s/ESS':>9}{'uniq':>7}{'grad/draw':>8}{'grad/ESS':>10}"
        "  worst-mixing parameter"
    )
    print(header)
    print("-" * len(header), flush=True)

    rows = {}
    for label, kwargs in configs.items():
        rows[label] = run_one(args.notebook, label, kwargs, seed)
        print(format_row(label, rows[label]), flush=True)
        if args.json:
            _append_json(args.json, args.notebook, seed, label, rows[label])

    print("\nverdict (ranked on seconds per effective sample):")
    print("  primary   = the notebooks' own bar: R-hat < 1.01, ZERO divergences, ESS >= nuts")
    print("  secondary = comparative: R-hat < 1.01, divergence RATE < 0.5% of total draws,")
    baseline_name = shipped_label(NOTEBOOKS[args.notebook])
    print(f"              ESS >= {baseline_name} -- applied identically to that baseline row")
    print("  an unadjusted sampler has no divergences to count: that clause reads n/a,")
    print("              and is vacuous rather than satisfied. Read EEVPD instead.")
    baseline = rows.get(baseline_name)
    baseline_ess = baseline["min_ess"] if baseline and not baseline.get("dead_fit") else 0.0
    for label, row in sorted(rows.items(), key=lambda kv: kv[1]["sec_per_ess"]):
        if row.get("dead_fit"):
            # A refusal is never a pass under either criterion.
            print(f"  {label:<20} {'REFUSED':<12}{'REFUSED':<12} DeadFitError, no posterior")
            continue
        primary = clears_bar(row) and row["min_ess"] >= baseline_ess
        secondary = clears_bar_comparative(row, baseline_ess)
        rate = divergence_rate(row)
        div_col = (
            f"div {'n/a':>5}/{row.get('n_draws_total') or '?':<6}(  n/a )"
            if rate is None
            else f"div {row['divergences']:>5}/{row.get('n_draws_total') or '?':<6}"
            f"({100 * rate:5.2f}%)"
        )
        versus = (
            f"{baseline['sec_per_ess'] / max(row['sec_per_ess'], 1e-9):5.2f}x"
            f" vs {baseline_name.split()[0]}"
            if baseline
            else ""
        )
        print(
            f"  {label:<20} {'clears' if primary else 'MISSES':<12}"
            f"{'clears' if secondary else 'MISSES':<12}{div_col}  {versus}"
        )


if __name__ == "__main__":
    main()
