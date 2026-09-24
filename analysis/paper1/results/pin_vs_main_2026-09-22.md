# Is it safe to move the pin forward? (2026-09-22)

`paper1/pin-2026-09-14` is 15 commits behind `origin/main`. Several of those
touch components the kitchen-sink mock uses by name — the Lyman-continuum mask
on the photometric LUT (#2439, PR #2474), the composable torus at 1 mm (#1512,
PR #2452), the photoionized age-0 SSP anchor (#2418, PR #2459) — so "does main
change the figure?" is not answerable by reading the shortlog.

Measured instead, with `paper1.mock_forward_digest`: the mock's forward
prediction at its own truth, both channels, on each tree.

| channel | bit-identical | worst shift | chi2 contribution |
|---|---|---|---|
| photometry | 15/16 | 0.0039 sigma (`galex_nuv`, -0.053%) | 0.0000 |
| spectrum | 1500/1500 | 0.0000 sigma | 0.0000 |

`galex_nuv` is the one band whose passband reaches below 912 A rest at z=1
(blue edge 1693 A observed = 846 A rest), so it is the band the LyC fix moves,
and it moves by four thousandths of its own noise. The torus and age-0 fixes do
not fire on this configuration at all.

**So moving the pin forward is numerically free for this figure.** That is
evidence for a decision, not the decision: it measures one mock's forward
prediction, not the 20x6 CANDELS grid, not any posterior, and not wall clock.

## Reproducing

```sh
SNAP=$(mktemp -d)
git archive origin/main src | tar -x -C "$SNAP"

PYTHONPATH=<pin>/src:<pin>/analysis TENGRI_DISABLE_PRECOMP_CACHE=1 \
  python -m paper1.mock_forward_digest pin.npz
PYTHONPATH="$SNAP"/src:<pin>/analysis TENGRI_DISABLE_PRECOMP_CACHE=1 \
  TENGRI_DATA_DIR=<repo>/data python -m paper1.mock_forward_digest main.npz
python -m paper1.mock_forward_digest --compare pin.npz main.npz
```

`TENGRI_DATA_DIR` is required for the snapshot: the data locator walks ancestor
directories from the package, and a tree outside the repo has none to walk.
`TENGRI_DISABLE_PRECOMP_CACHE=1` on **both** arms — the cache is content-hashed,
but a stale entry is the one failure mode that would make two trees look
identical when they are not.
