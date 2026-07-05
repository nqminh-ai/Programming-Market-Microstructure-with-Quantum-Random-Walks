"""Compile marginal forecast evidence using pre-registered endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


class ResultsCompiler:
    """Build a scorecard without averaging unrelated metric ranks."""

    REQUIRED_KEYS = {"marginal_scores", "variance_scaling"}
    SELECTION_RULE = (
        "primary=mean_marginal_crps; "
        "tiebreak=mean_direction_log_loss; diagnostics_not_ranked"
    )

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
        """Return diagnostic comparison and primary-endpoint scorecard."""
        missing = sorted(self.REQUIRED_KEYS.difference(results))
        if missing:
            raise ValueError(f"missing statistical result groups: {missing}")

        marginal = self._load(results["marginal_scores"])
        scaling = self._load(results["variance_scaling"])
        required_columns = {
            "model",
            "horizon",
            "crps",
            "mean_absolute_error",
            "direction_brier_score",
            "direction_log_loss",
            "interval_90_covered",
        }
        missing_columns = sorted(required_columns.difference(marginal.columns))
        if missing_columns:
            raise ValueError(
                f"marginal scores are missing columns: {missing_columns}"
            )

        comparison = (
            marginal.groupby("model", sort=False)
            .agg(
                evaluated_horizons=("horizon", "nunique"),
                mean_marginal_crps=("crps", "mean"),
                mean_marginal_absolute_error=("mean_absolute_error", "mean"),
                mean_direction_brier_score=("direction_brier_score", "mean"),
                mean_direction_log_loss=("direction_log_loss", "mean"),
                coverage_90=("interval_90_covered", "mean"),
            )
            .reset_index()
        )
        comparison = comparison.merge(
            scaling.loc[
                scaling["model"] != "Empirical",
                ["model", "beta", "beta_ci_low", "beta_ci_high"],
            ].rename(columns={"beta": "marginal_variance_scaling_beta"}),
            on="model",
            how="left",
            validate="one_to_one",
        )
        comparison["coverage_90_absolute_error"] = (
            comparison["coverage_90"] - 0.90
        ).abs()
        comparison["selection_rule"] = self.SELECTION_RULE

        numeric = comparison.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("final comparison contains non-finite values")

        scorecard = comparison[
            [
                "model",
                "mean_marginal_crps",
                "mean_direction_log_loss",
                "mean_direction_brier_score",
                "coverage_90",
            ]
        ].copy()
        scorecard["overall_rank"] = scorecard[
            "mean_marginal_crps"
        ].rank(method="min", ascending=True)
        scorecard["direction_tiebreak_rank"] = scorecard[
            "mean_direction_log_loss"
        ].rank(method="min", ascending=True)
        scorecard["selection_rule"] = self.SELECTION_RULE
        scorecard = scorecard.sort_values(
            [
                "mean_marginal_crps",
                "mean_direction_log_loss",
                "model",
            ],
            kind="stable",
        ).reset_index(drop=True)

        self._write(comparison_output, comparison)
        self._write(scorecard_output, scorecard)
        return comparison, scorecard
