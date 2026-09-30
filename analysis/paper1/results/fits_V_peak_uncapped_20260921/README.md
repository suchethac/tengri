# Row V cells run under the uncapped `peak_gyr` prior

Configuration V's log-normal SFH carried `peak_gyr ~ Uniform(0.1, 13.0)` until
2026-09-21, while `age_gyr` was bounded by `age_at_z(z)` (5.1-5.9 Gyr for the
sample). More than half of each cell's rung-1 draws put the SFH peak after the
epoch of observation, where only the rising limb of the log-normal lies inside
the galaxy's life and many (peak, width) pairs share one slope: a flat ridge,
the same direction that row II's `tau_gyr > age(z)` produced. On that geometry
every retune rung adapted to a step of 0.0002-0.001 and returned a frozen chain.

The owner capped `peak_gyr` at `age_at_z(z)` on 2026-09-21 and added a minimum
effective-sample-size leg to the adoption bar (`fit_one.ESS_FLOOR`). Row V is
rerun under the capped prior; these three cells are the pre-cap record and are
not part of the grid:

| cell    | outcome under the uncapped prior                                              |
|---------|--------------------------------------------------------------------------------|
| 9884/V  | adopted on rung 2 (0 divergences, rhat 1.0037, ESS 381) -- the one that mixed  |
| 14099/V | `adoption_pass: true` on ESS 3 -- the case that motivated the ESS floor         |
| 7837/V  | best attempt rung 1 (5 divergences); rungs 2-3 froze (ESS 1); rung 3 lost      |
