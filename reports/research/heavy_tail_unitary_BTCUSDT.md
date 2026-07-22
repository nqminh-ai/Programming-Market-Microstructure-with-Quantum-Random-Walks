# Heavy-tailed unitary shift — tail-shape evaluation (BTCUSDT)

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory. Closes the §7 gap: the heavy-tail mechanism is now an exactly-unitary Lévy shift, not a classical Bernoulli/Pareto sampler.

- Feature file: `features_BTCUSDT_recent_subset.parquet` (1,000,000 rows), tick size 0.03
- Horizon: 50 ticks · lattice 16001 positions · 200,000 empirical displacements
- Git commit: `a2d16f1fd9d00f4bd475e243e71a6cc85f593492` · Python 3.14.5

Only tail *shape* is compared, via quantile ratios of |x − median| (scale-free). Variance and kurtosis are deliberately avoided: for a Lévy walk with α < 2 the second moment does not exist, so on a finite lattice any σ or kurtosis measured is a function of the lattice size, not of the distribution.

## Empirical reference

| q99/q75 | q999/q75 |
|---:|---:|
| 2.92 | 5.07 |

## Lévy unitary walks (α = 1 is the ordinary nearest-neighbour walk)

| α | q99/q75 | q999/q75 | shape distance | wraparound | valid |
|---:|---:|---:|---:|---:|:--:|
| 0.3 | 11.40 | 75.35 | 4.062 | 5.8e-05 | ✘ wrap |
| 0.5 | 5.85 | 25.55 | 2.313 | 5.4e-07 | ✔ |
| 0.7 | 3.62 | 7.74 | 0.640 | 6.9e-06 | ✔ |
| 0.9 | 1.62 | 3.10 | 1.080 | 4.7e-06 | ✔ |
| 1.0 *(ordinary)* | 1.12 | 1.15 | 2.443 | 2.2e-29 | ✔ |
| 1.3 | 1.17 | 1.23 | 2.329 | 2.7e-07 | ✔ |
| 1.6 | 1.56 | 3.15 | 1.101 | 1.3e-05 | ✘ wrap |
| 2.0 | 1.81 | 1.90 | 1.463 | 7.1e-07 | ✔ |

## Verdict

The heavy-tailed unitary (alpha=0.7) matches the empirical tail shape better than the ordinary walk (shape distance 0.640 vs 2.443). The ordinary walk's marginal is bimodal/ballistic with q999/q75 = 1.15, against an empirical 5.07; the Lévy shift reaches 7.74.
