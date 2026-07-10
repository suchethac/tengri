# Adding a reproduction comparison for a new code

How to stand up a component-by-component comparison against a reference code
that does not have one yet (BEAGLE, MAGPHYS, and friends).

Mechanics — folder layout, section numbering, figure naming, the capstone,
rendering and publishing — are fixed by
[`reproduction/CONTRACT.md`](../../reproduction/CONTRACT.md) §1–9. Follow it.
This document covers the parts that are specific to *bringing a new engine in*,
and it ends where [`reproduction-parity-audit.md`](reproduction-parity-audit.md)
begins.

## 1. Find the engine's real invocation surface first

This decision shapes the whole driver, so make it before writing anything.
Codes expose their forward model very differently:

- **Python API** — `python-fsps` (Prospector), the CIGALE modules. Call it live
  in the driver, one real call per panel.
- **R package** — ProSpect. Drive it through `rpy2`; every left-hand panel is
  the R code's own output.
- **Binary plus configuration file** — BEAGLE, MAGPHYS. There is no Python
  forward model to import. The driver writes a parameter file, invokes the
  binary (often containerized), and parses the native output (FITS for BEAGLE).
  Cache the run: these are slow, and the notebook should not re-run the fitter
  on every render.
- **Template library only** — when the fitter cannot be driven headlessly, read
  the code's own template libraries directly and state plainly that the
  comparison is against its templates, not its fitter (AGNFITTER-RX does this).

Whatever the surface, drive the reference's own engine. Do not port its physics
into the driver — a comparison of your port against tengri tests nothing.

## 2. Match the templates, or document the mismatch

Match the isochrone × spectral library × IMF the reference uses.
`tengri.list_known_ssps()` lists the available combinations and
`tengri.download_ssp()` fetches them. BEAGLE, for instance, is built on BC03
with its own nebular grid, so the stellar comparison should read a BC03 grid on
both sides.

If no shipped grid matches, say so loudly in the notebook and README rather
than quietly comparing different stellar physics.

Verify the unit contract rather than assuming it: ported grids may store flux in
the source code's native solar luminosity. Bit-match the raw array values
against the reference's own output for one SSP before trusting any absolute
comparison — a silent 0.3 % scale error hides easily behind percent-level
astrophysics.

## 3. Write the drivers

Copy `_drivers/units.py` from an existing comparison unchanged (CONTRACT §3) and
add a code-specific driver module that returns the reference's SSPs, composite
SEDs, SFHs, and per-block curves in tengri's units (erg/s/Hz, rest-frame Å).

Ship a unit round-trip check that runs at Setup and fails the notebook if the
converter drifts. A factor error in the converter silently misshapes every
panel downstream.

Guard Setup on the reference's binaries, tables, or environment variables being
present, with a message saying exactly what to install or export.

## 4. Pin the conventions

Work through the convention table in
[`reproduction-parity-audit.md`](reproduction-parity-audit.md) §5 — solar
luminosity, IMF, IGM default, filter convolution, metallicity, dust energy
balance, nebular provenance — and state in the README which side each notebook
uses. Where the two codes' defaults differ, give readers the checklist of
toggles that make the comparison faithful.

## 5. Then audit it

The first render will surface real disagreements. That is the point of building
it. Take them through [`reproduction-parity-audit.md`](reproduction-parity-audit.md):
attribute each to a mechanism, fix genuine bugs with regression tests, document
deliberate conventions, and file issues for what you do not fix in this pass.

A new comparison that reports nothing but agreement has not been read closely
enough.
