# Heavy-tailed unitary shift — tail-shape evaluation (BNBUSDT)

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory. Closes the §7 gap: the heavy-tail mechanism is now an exactly-unitary Lévy shift, not a classical Bernoulli/Pareto sampler.

- Feature file: `bnb_combined.parquet` (1,000,000 rows), tick size 0.01
- Horizon: 50 ticks · lattice 16001 positions · 200,000 empirical displacements
- Git commit: `a2d16f1fd9d00f4bd475e243e71a6cc85f593492` · Python 3.14.5

Only tail *shape* is compared, via quantile ratios of |x − median| (scale-free). Variance and kurtosis are deliberately avoided: for a Lévy walk with α < 2 the second moment does not exist, so on a finite lattice any σ or kurtosis measured is a function of the lattice size, not of the distribution.

## Empirical reference

| q99/q75 | q999/q75 |
|---:|---:|
| 2.69 | 4.00 |

## Lévy unitary walks (α = 1 is the ordinary nearest-neighbour walk)

| α | q99/q75 | q999/q75 | shape distance | wraparound | valid |
|---:|---:|---:|---:|---:|:--:|
| 0.3 | 11.40 | 75.35 | 4.379 | 5.8e-05 | ✘ wrap |
| 0.5 | 5.85 | 25.55 | 2.630 | 5.4e-07 | ✔ |
| 0.7 | 3.62 | 7.74 | 0.957 | 6.9e-06 | ✔ |
| 0.9 | 1.62 | 3.10 | 0.763 | 4.7e-06 | ✔ |
| 1.0 *(ordinary)* | 1.12 | 1.15 | 2.126 | 2.2e-29 | ✔ |
| 1.3 | 1.17 | 1.23 | 2.012 | 2.7e-07 | ✔ |
| 1.6 | 1.56 | 3.15 | 0.784 | 1.3e-05 | ✘ wrap |
| 2.0 | 1.81 | 1.90 | 1.145 | 7.1e-07 | ✔ |

## Verdict

The heavy-tailed unitary (alpha=0.9) matches the empirical tail shape better than the ordinary walk (shape distance 0.763 vs 2.126). The ordinary walk's marginal is bimodal/ballistic with q999/q75 = 1.15, against an empirical 4.00; the Lévy shift reaches 3.10.
