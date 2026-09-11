# compile_signature() Benchmark Results

**Date:** 2026-09-11 14:11:10
**JAX Version:** 0.11.1
**Platform:** macOS-26.6.2-arm64-arm-64bit
**CPU:** Apple M4 Pro
**Config:** 2000 signature repeats, 200 predict repeats

| Build | Status | n_fields | t_sig_first [µs] | t_sig_repeat median [µs] | p90 [µs] | t_predict [µs] | sig_share [%] |
|-------|--------|----------|------------------|--------------------------|----------|----------------|---------------|
| A. Photometry (star-forming) | ✓ | 68 | 102501.2 | 44.67 | 46.50 | 430.3 | 10.38 |
| B. Spectroscopy (simple) | ✓ | 68 | 103373.9 | 33.33 | 34.88 | 4544.3 | 0.73 |
| C. AGN + dust emission | ✓ | 68 | 103256.8 | 49.58 | 50.38 | 5041.5 | 0.98 |
| D. Nebular + shock | ✓ | 68 | 105404.5 | 44.17 | 48.21 | 6178.9 | 0.71 |
| E. Per-screen laws + THEMIS (build) | ✓ | 68 | 104642.4 | 43.46 | 44.63 | 474.3 | 9.16 |
| F. from_config | ✓ | 68 | 102188.9 | 41.58 | 42.50 | 436.6 | 9.53 |
