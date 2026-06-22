# Quantum Random Walks for Market Microstructure

Phase 6 final presentation

---

# Research Question

Can a causally calibrated QRW reproduce short-horizon market properties better than classical baselines?

---

# Why QRW?

Interference, ballistic spreading, and a tunable coherent-to-diffusive transition.

---

# QRW vs CRW

Coherent QRW variance grows near t^2; symmetric CRW variance grows as t.

---

# Active Data

June 12, 2026 BTCUSDT: 1,869,214 train rows and 1,246,144 later holdout rows.

---

# Causal Pipeline

Trailing cleaner, chronological split, disjoint validation, fixed seed, and matched samples.

---

# Market Mapping

OBI and direction enter a unitary phase-adaptive coin; intensity controls dephasing.

---

# Benchmark Models

QRW Adaptive, three CRWs, GARCH(1,1), and GBM.

---

# Evaluation

Distribution, variance scaling, autocorrelation, tails, and an eight-metric rank scorecard.

---

# Probability Evolution

../figures/prob_evolution.gif

---

# Variance Scaling

../figures/variance_scaling.png

---

# Distribution Shape

../figures/return_distributions.png

---

# Dependence and Paths

../figures/acf_comparison.png

---

# Scorecard

CRW Correlated ranks first; QRW ranks 7.

---

# What We Can Claim

The software pipeline passes. Current evidence does not establish QRW superiority.

---

# Next Study

Freeze protocol, collect 20+ fresh UTC days with synchronized LOB, then run one confirmatory evaluation.