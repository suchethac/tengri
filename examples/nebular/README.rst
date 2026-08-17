Nebular Emission
================

Emission lines are vacuum throughout: Hα is 6564.61 Å, not the 6562.8 Å air
value. Mixing the two shifts every line centroid.

``neb={'type': ...}`` takes ``ssp``, ``cue``, ``cb19``, ``cloudy`` or ``none``.
The default, ``ssp``, uses the emission already baked into a with-nebular (wNE)
SSP grid. The live backends instead compute it, and expect a bare stellar grid.
Feed a bare grid to the baked-in path and both continuum and line fluxes come
out low, with no error raised.

Gas-phase metallicity is its own knob and does not follow the stellar one.
