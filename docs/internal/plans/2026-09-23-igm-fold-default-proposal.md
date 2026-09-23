# Proposal: make `igm_fold="auto"` the default

**Status: proposal. Nothing is flipped.** The default is `"node"` today
(`forward/sed_model.py`, `igm_fold: str = "node"`) and stays there until the
owner rules. Around twenty peer sessions build against this code and a silent
default change would move their numbers without their asking.

This is the last open piece of the exact-IGM-fold handoff: the opt-in path is
implemented, the A/B comparison exists, and the validation has run. What was
missing is the proposal the handoff asked for, with its evidence and its cost.

## Recommendation

Default to **`"auto"`**, not `"exact"`.

`"exact"` cannot be a default. It *raises* on two legitimate configurations:
a transmission that is not a fixed function of `(wavelength, redshift)` —
patchy reionization, discrete DLAs — and a model with a free redshift, whose
z-table the exact fold does not implement. Making it the default would turn
those into build failures for people who changed nothing.

`"auto"` takes the exact fold where it can be built and falls back to the node
fold where it cannot, so no configuration breaks. The fall-back and the two
refusals already read one shared predicate (`_exact_fold_blocker`), so they
cannot drift apart.

## The validation

`analysis/paper1/igm_fold_error_budget.py`, four arms, each measured against
`approx=None` — the exact wavelength-grid integrator. Probe band `galex_nuv` at
z = 1.0, chosen because that is where the Lyman break is sweeping, so the IGM
term is live rather than negligible.

| arm | dust | nebular | node fold | exact fold |
|---|---|---|---|---|
| bare stellar | — | — | +0.3646% | **+0.0000%** |
| + dust | yes | — | +0.4769% | +0.1018% |
| + nebular | — | yes | +0.8549% | +0.5637% |
| + both (production) | yes | yes | **−0.3982%** | −0.6934% |

The **bare arm is the validation**. There the IGM is the only approximation in
the model, so a correct exact fold must read zero, and it does to the printed
precision while the node fold reads +0.36%. That is the claim "the exact fold
is correct" resting on a case where the right answer is known independently.

The other three arms do not measure the IGM fold. They measure what is *left*
once it is removed: the sub-band node approximation applied to the dust screen
and to the nebular continuum. Worth noting because the appendix used to say the
exact fold "leaves only the dust-node error" — it does not, and the nebular
continuum is the larger of the two, with no dust in the model at all.

## The objection this will meet, and why it does not hold

On the production row the node fold's error is *smaller in magnitude*:
−0.3982% against −0.6934%. Someone will read that as node being more accurate
for real models and stop there.

It is cancellation, not accuracy. The bare arm shows the IGM node error is
about +0.36% and positive; the exact-fold residual in the production arm is
−0.69% and negative. Adding the IGM error to a residual of the opposite sign
moves the total toward zero. Node is not describing the IGM better — it is
carrying a second error that happens to point the other way.

That cancellation is a coincidence of this model, this band and this redshift.
Nothing holds it in place: change the dust law, the nebular backend, the band
or z and the two terms stop opposing each other, with no warning, because
nothing in the output says a cancellation was ever happening. An error budget
whose terms are individually correct is the robust one; a smaller total built
from two wrongs is not a property anyone can rely on.

The repository already has the sharper version of this argument. A constant
forward bias is not noise: it enters the posterior **gradient multiplied by
SNR** (#1671, measured at ~5% gradient error at SNR 30 and ~50% at SNR 300 for
a 0.13% forward bias). Removing a real term is therefore worth more than the
forward percentage suggests, and worth most exactly where the data are best.

## Blast radius, measured

Of the ten shipped recipes, **six declare a free redshift** and so keep the
node fold unchanged under `"auto"`:

    star_forming_photometry, high_z, photoz,
    agn_panchromatic, composable_agn, stochastic_sfh_jwst

**Four pin the redshift** and would switch to the exact fold:

    quiescent_z0, mock_recovery_minimal, dust_demo, unified_agn

No shipped recipe uses patchy reionization or a DLA, so that refusal is rare in
practice. Neither do the paper's six CANDELS configurations, but all six fit at
a fixed catalog redshift, so all six would switch.

Two honest readings of that split, and they point opposite ways:

- The change is *contained*. The majority of recipes are structurally excluded,
  so most users see nothing.
- The change is *narrow*. The exact fold reaches a minority of recipe users,
  and the free-redshift case — the one where a photo-z walks the break across a
  band and the fold matters most — is precisely the one it cannot serve yet.

The second is the stronger argument against flipping now, and the strongest
argument for implementing the z-table exact fold before flipping anything.

## Preconditions before the default moves

1. Owner's ruling. This is a numbers-changing default on a shared library.
2. A CHANGELOG entry under a version bump, naming the four recipes whose
   photometry moves and the size of the move in the affected bands.
3. A sweep of the paper's own figures and tables for anything computed with the
   node fold whose caption would become false.
4. Ideally: the free-redshift z-table path, so the change reaches the case that
   motivates it instead of excluding it.

Until then `"auto"` remains opt-in and `"node"` remains the default, which is
what the handoff asked for.
