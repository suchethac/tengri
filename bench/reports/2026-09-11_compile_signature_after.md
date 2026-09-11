# compile_signature() Benchmark Results

**Date:** 2026-09-11 15:47:44
**JAX Version:** 0.11.1
**Platform:** macOS-26.6.2-arm64-arm-64bit
**CPU:** Apple M4 Pro
**Config:** 2000 signature repeats, 200 predict repeats

| Build | Status | n_fields | t_sig_first [µs] | t_sig_repeat median [µs] | p90 [µs] | t_predict [µs] | sig_share [%] |
|-------|--------|----------|------------------|--------------------------|----------|----------------|---------------|
| A. Photometry (star-forming) | ✓ | 4 | 3635.5 | 0.08 | 0.08 | 328.0 | 0.03 |
| B. Spectroscopy (simple) | ✓ | 4 | 3461.7 | 0.08 | 0.08 | 408.4 | 0.02 |
| C. AGN + dust emission | ✓ | 4 | 2891.5 | 0.08 | 0.08 | 2090.7 | 0.00 |
| D. Nebular + shock | ✓ | 4 | 2444.7 | 0.08 | 0.08 | 1428.2 | 0.01 |
| E. Per-screen laws + THEMIS (build) | ✓ | 4 | 2397.8 | 0.04 | 0.08 | 371.7 | 0.01 |
| F. from_config | ✓ | 4 | 2320.1 | 0.08 | 0.08 | 358.2 | 0.02 |
