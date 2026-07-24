# Marginal-CRPS comparison — BNBUSDT_69d_daycluster

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory. Registered PRIMARY endpoint: mean fixed-origin marginal CRPS.

- Protocol: `fixed_origin_marginal_density_matrix_ar1_obi_v4`
- Feature file: `features_BNBUSDT_69d.parquet` (54,096,780 rows)
- Windows: 20 non-overlapping chronological splits · n_steps=200 · n_paths=2000
- Git commit: `81737c3dddf94218d4e096f659241ae3c80ef880` · Python 3.14.5

## Mean marginal CRPS across windows (lower = better)

| Rank | Model | mean CRPS | windows best | per-window CRPS |
|---:|---|---:|---:|---|
| 1 | GBM | 0.0873 | 4/20 | 0.032, 0.061, 0.127, 0.166, 0.046, 0.122, 0.068, 0.153, 0.034, 0.105, 0.200, 0.036, 0.043, 0.075, 0.182, 0.053, 0.122, 0.041, 0.021, 0.058 |
| 2 | GARCH(1,1) | 0.0884 | 4/20 | 0.033, 0.055, 0.118, 0.183, 0.044, 0.135, 0.087, 0.148, 0.031, 0.091, 0.205, 0.036, 0.050, 0.066, 0.200, 0.053, 0.112, 0.043, 0.018, 0.062 |
| 3 | CRW Correlated | 0.0889 | 2/20 | 0.032, 0.065, 0.126, 0.168, 0.042, 0.126, 0.069, 0.143, 0.035, 0.113, 0.206, 0.036, 0.043, 0.088, 0.185, 0.053, 0.118, 0.045, 0.021, 0.065 |
| 4 | QRW Adaptive **(QRW)** | 0.0889 | 8/20 | 0.051, 0.062, 0.086, 0.116, 0.054, 0.121, 0.074, 0.112, 0.067, 0.085, 0.150, 0.080, 0.073, 0.110, 0.118, 0.083, 0.097, 0.084, 0.068, 0.089 |
| 5 | CRW Simple | 0.0944 | 0/20 | 0.034, 0.068, 0.134, 0.176, 0.047, 0.141, 0.077, 0.170, 0.034, 0.115, 0.213, 0.037, 0.046, 0.090, 0.194, 0.055, 0.127, 0.044, 0.023, 0.063 |
| 6 | CRW Biased | 0.0951 | 2/20 | 0.034, 0.069, 0.134, 0.178, 0.048, 0.138, 0.076, 0.182, 0.031, 0.117, 0.213, 0.036, 0.047, 0.093, 0.200, 0.055, 0.126, 0.042, 0.020, 0.064 |

## Verdict

The QRW marginal wins the CRPS endpoint in 8/20 windows; overall best is 'GBM'. The volatility story is NOT supported at this sample size: Spearman between realised volatility and the QRW's relative CRPS gap is -0.08 (p=0.738, 20 windows).
