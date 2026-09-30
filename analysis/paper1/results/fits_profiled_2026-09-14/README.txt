These 8 cells were computed on 2026-09-14 with profile_mass ENGAGED, before
tengri#2356 forced the workaround. The production grid runs with
profile_mass=False, where the stellar mass is sampled rather than marginalized
and reinserted. Those are different inference paths and must not share a table:
the posteriors are equivalent, but a paper's methods section can describe only one.
