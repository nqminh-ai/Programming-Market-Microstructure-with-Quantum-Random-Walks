# Does the volatility claim survive measurement?

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory.

**Claim under test:** The QRW falls further behind the best alternative as realised volatility rises, i.e. it does not model volatility dynamics.

**Gap definition:** (QRW CRPS - best rival CRPS) / best rival CRPS, per window

- Git commit: `b6ce0f291e54ed631f5cc1845c9d5bf9303b09c7` · Python 3.14.5

## Per asset (40 non-overlapping windows each)

| Asset | Windows | Spearman | p (two-sided) | Supports claim | QRW CRPS rank | QRW wins |
|---|---:|---:|---:|:--:|:--:|:--:|
| BNBUSDT_69d | 40 | +0.222 | 0.169 | no | 1/6 | 22/40 |
| ETHUSDT_69d | 40 | +0.047 | 0.773 | no | 1/6 | 31/40 |
| BTCUSDT_69d | 40 | +0.213 | 0.187 | no | 3/6 | 3/40 |

## Pooled across assets (one-sided, in the asserted direction)

- One-sided p-values: [0.084338, 0.386462, 0.093738]
- All in the claimed direction: **True**
- Fisher: chi2=11.58, p=**0.0720**
- Stouffer: z=1.72, p=**0.0425**

## Verdict

NOT established at alpha=0.05. The relationship runs in the asserted direction on every asset but no asset reaches significance on its own (BNBUSDT rho=+0.22 (p=0.169), ETHUSDT rho=+0.05 (p=0.773), BTCUSDT rho=+0.21 (p=0.187)), and the two pooling methods straddle alpha -- Fisher p=0.0720, Stouffer p=0.0425 -- so the pooled answer depends on which test is quoted. The claim should be stated as a direction the data leans towards, not a finding.

### Sources

- `reports/research/marginal_crps_vol_BNBUSDT.json` — sha256 `19db3938f3237ad3…`
- `reports/research/marginal_crps_vol_ETHUSDT.json` — sha256 `52f940f07231413d…`
- `reports/research/marginal_crps_vol_BTCUSDT.json` — sha256 `2d394e00be8d0c15…`
