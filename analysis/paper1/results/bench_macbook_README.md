# MacBook shape-C benchmark — why these files are not in `fig:gpu`

`bench_macbook_shapeC_f32.json` and `bench_macbook_shapeC_f64.json` are a
batch sweep of the shape-C photometry model on an Apple Silicon laptop CPU
(`JAX_PLATFORMS=cpu`), measured 2026-09-20 on JAX 0.10.0.

Nothing reads them. That is deliberate, and this file exists so the next
session does not "fix" it.

## Why they are not an arm of `fig:gpu`

Two independent reasons, either sufficient.

**The figure's claim is a crossover, and a crossover needs one machine.**
§6 states it directly: the H100 is measured against an Intel Xeon Platinum
8462Y+ "both measured on the same node under the same allocation --- a
crossover taken against a different machine's CPU would not be one." A laptop
CPU plotted against a datacenter GPU is precisely the comparison that sentence
disclaims. Adding it would contradict the text on the same page.

**Three campaigns, three JAX versions.** The H100 sweep ran on JAX 0.7.0 (the
Sherlock glibc 2.17 ceiling), the RTX 3060 appendix campaign on 0.11.0, and
this laptop on 0.10.0. §6 already flags the first pair as "indicative rather
than controlled". A third version on a third machine compounds that rather
than resolving it.

## What they do show

Per-galaxy cost, shape C, `wave_precomp` photometry, 7 free parameters:

| batch | f64 forward | f64 gradient | f32 forward | f32 gradient |
|------:|------------:|-------------:|------------:|-------------:|
| 1     | 100.633     | 214.446      | 112.892     | 235.271      |
| 8     | 42.595      | 120.195      | 41.895      | 106.080      |
| 32    | 25.937      | 58.351       | 17.649      | 50.116       |
| 128   | 12.899      | 37.497       | 9.465       | 32.035       |
| 512   | **10.895**  | 38.544       | **7.174**   | **28.577**   |
| 2048  | 13.716      | 55.404       | 8.412       | 32.221       |

All values are microseconds per galaxy.

Two things here are worth someone's attention.

The curve has a **minimum and then rises**. Forward cost bottoms at batch 512
in both precisions and is worse at 2048. That is the CPU saturating, which is
the behavior §6 asserts of a CPU arm and which the H100 arm never shows --- the
H100's per-call time is flat from batch 1 to 2048, so its per-galaxy curve is
`1/n` by construction and has not begun to level off.

The laptop reaches **7.2 µs per galaxy** on the single-precision forward pass.
§6 quotes 6.7 µs for the H100 at batch 2048. Those two numbers being within
7% of each other is not a claim that a laptop matches an H100; it is the same
observation §6 already makes, that the H100 figure is a property of the batch
size rather than of the card. It is tempting to put that sentence in the
paper. Do not, until the provenance gap below is closed.

## The provenance gap

These files record `_machine` and `_measured` and the JAX version, and nothing
else. There is **no commit SHA**, so the code they measured is not identified.
`sherlock_h100_README.md` is the standard to match: it names the issue, the
hardware, the stack, and the caveats, and it exists precisely because a caption
naming hardware the arrays did not come from is the drift a committed file
prevents.

A number without a commit cannot go in a pinned paper. To quote anything above,
re-run the sweep and record the SHA alongside it; until then these are a
measurement someone took, not a result the paper can stand on.
