"""Compile Phase 5 statistical outputs into a model scorecard."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


class ResultsCompiler:
    """Merge statistical test outputs and rank models against empirical data."""

    REQUIRED_KEYS = {
        "distribution",
        "variance_scaling",
        "autocorrelation",
        "tail",
    }

    @staticmethod
    def _load(value: pd.DataFrame | str | Path) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value.copy()
        return pd.read_csv(value)

    @staticmethod
    def _write(output: str | Path | None, frame: pd.DataFrame) -> None:
        if output is None:
            return
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)

    def compile(
        self,
        results: Mapping[str, pd.DataFrame | str | Path],
        *,
        comparison_output: str | Path | None = None,
        scorecard_output: str | Path | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return a merged comparison table and empirical-distance scorecard."""
        missing = sorted(self.REQUIRED_KEYS.difference(results))
        if missing:
            raise ValueError(f"missing statistical result groups: {missing}")
        distribution = self._load(results["distribution"])
        scaling = self._load(results["variance_scaling"])
        autocorrelation = self._load(results["autocorrelation"])
        tail = self._load(results["tail"])

        distribution_h1 = distribution.loc[
            distribution["horizon"] == 1,
            [
                "model",
                "ks_statistic",
                "ks_pvalue",
                "ks_pvalue_bh",
                "wasserstein_distance",
            ],
        ]
        model_names = distribution_h1["model"].drop_duplicates().tolist()
        comparison = pd.DataFrame({"model": model_names})
        comparison = comparison.merge(
            distribution_h1,
            on="model",
            how="left",
            validate="one_to_one",
        )
        comparison = comparison.merge(
            scaling.loc[
                scaling["model"] != "Empirical",
                [
                    "model",
                    "beta",
                    "beta_ci_low",
                    "beta_ci_high",
                    "pvalue_beta_gt_1",
                    "pvalue_beta_gt_1_bh",
                ],
            ].rename(columns={"beta": "variance_scaling_beta"}),
            on="model",
            how="left",
            validate="one_to_one",
        )
        comparison = comparison.merge(
            autocorrelation.loc[
                autocorrelation["model"] != "Empirical",
                ["model", "acf_mse"],
            ],
            on="model",
            how="left",
            validate="one_to_one",
        )
        comparison = comparison.merge(
            tail.loc[
                tail["model"] != "Empirical",
                [
                    "model",
                    "kurtosis",
                    "tail_index",
                    "var_99",
                    "cvar_99",
                ],
            ],
            on="model",
            how="left",
            validate="one_to_one",
        )
        numeric = comparison.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("final comparison contains non-finite values")

        empirical_scaling = float(
            scaling.loc[scaling["model"] == "Empirical", "beta"].iloc[0]
        )
        empirical_tail = tail.loc[tail["model"] == "Empirical"].iloc[0]
        scoring = pd.DataFrame({"model": comparison["model"]})
        scoring["ks_statistic_rank"] = comparison["ks_statistic"].rank(
            method="min",
            ascending=True,
        )
        scoring["wasserstein_rank"] = comparison[
            "wasserstein_distance"
        ].rank(method="min", ascending=True)
        scoring["variance_scaling_rank"] = (
            comparison["variance_scaling_beta"] - empirical_scaling
        ).abs().rank(method="min", ascending=True)
        scoring["acf_mse_rank"] = comparison["acf_mse"].rank(
            method="min",
            ascending=True,
        )
        scoring["kurtosis_rank"] = (
            comparison["kurtosis"] - float(empirical_tail["kurtosis"])
        ).abs().rank(method="min", ascending=True)
        scoring["var_99_rank"] = (
            comparison["var_99"] - float(empirical_tail["var_99"])
        ).abs().rank(method="min", ascending=True)
        scoring["cvar_99_rank"] = (
            comparison["cvar_99"] - float(empirical_tail["cvar_99"])
        ).abs().rank(method="min", ascending=True)
        rank_columns = [
            column for column in scoring.columns if column.endswith("_rank")
        ]
        scoring["mean_rank"] = scoring[rank_columns].mean(axis=1)
        scoring["overall_rank"] = scoring["mean_rank"].rank(
            method="min",
            ascending=True,
        )
        scoring = scoring.sort_values(
            ["overall_rank", "mean_rank", "model"],
            kind="stable",
        ).reset_index(drop=True)

        self._write(comparison_output, comparison)
        self._write(scorecard_output, scoring)
        return comparison, scoring
