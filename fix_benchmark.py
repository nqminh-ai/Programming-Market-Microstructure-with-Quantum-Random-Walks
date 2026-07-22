import re

with open("src/evaluation/benchmark_suite.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Imports
code = code.replace(
    "from src.models.adaptive_market_qrw import AdaptiveDecoherenceQRW",
    "from src.models.qrw_market_sim import MarketQRW\nfrom src.models.coin_operators import su2_market_coin"
)

# 2. REQUIRED_COLUMNS
code = code.replace(
    "REQUIRED_COLUMNS = AdaptiveDecoherenceQRW.REQUIRED_COLUMNS",
    "REQUIRED_COLUMNS = MarketQRW.REQUIRED_COLUMNS"
)

# 3. _fit_qrw
old_fit = """    def _fit_qrw(self) -> tuple[AdaptiveDecoherenceQRW, dict[str, Any]]:
        calibration = AdaptiveDecoherenceQRW(
            self.train,
            {"n_positions": 2 * self.n_steps + 3},
        )
        parameters = calibration.calibrate_two_stage(None)
        return calibration, parameters"""

new_fit = """    def _fit_qrw(self) -> tuple[MarketQRW, dict[str, Any]]:
        calibration = MarketQRW(
            self.train,
            {
                "n_positions": 2 * self.n_steps + 3,
                "coin_type": "obi_adaptive",
                "quantum_calibration_max_events": 5000,
                "quantum_window_size": 5,
            },
        )
        parameters = calibration.calibrate("results/benchmark_params.json")
        return calibration, parameters"""
code = code.replace(old_fit, new_fit)

# 4. _forecast_qrw_features
old_forecast = """    def _forecast_qrw_features(
        self,
        model: AdaptiveDecoherenceQRW,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        \"\"\"Create a causal fixed-origin feature path for the QRW horizon.\"\"\"
        ar1 = self._fit_obi_ar1(self.train)
        features = np.empty((self.n_steps, 5), dtype=np.float64)
        raw = model._raw_features()
        features[0] = raw[-1]
        obi = float(features[0, 0])
        direction = float(features[0, 1])
        log_intensity = float(features[0, 4])
        for index in range(1, self.n_steps):
            forecast_obi = float(
                np.clip(ar1["intercept"] + ar1["phi"] * obi, -1.0, 1.0)
            )
            features[index] = (
                forecast_obi,
                direction,
                forecast_obi - obi,
                abs(forecast_obi),
                log_intensity,
            )
            obi = forecast_obi
        return features, ar1"""

new_forecast = """    def _forecast_qrw_features(
        self,
        model: MarketQRW,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        \"\"\"Create a causal fixed-origin feature path for the QRW horizon.\"\"\"
        ar1 = self._fit_obi_ar1(self.train)
        features = np.empty((self.n_steps, 2), dtype=np.float64)
        obi_history = model.tick_data["obi"].to_numpy(dtype=np.float64)
        direction_history = model.tick_data["tick_direction"].to_numpy(dtype=np.float64)
        features[0] = (obi_history[-1], direction_history[-1])
        obi = float(features[0, 0])
        direction = float(features[0, 1])
        for index in range(1, self.n_steps):
            forecast_obi = float(
                np.clip(ar1["intercept"] + ar1["phi"] * obi, -1.0, 1.0)
            )
            features[index] = (
                forecast_obi,
                direction,
            )
            obi = forecast_obi
        return features, ar1"""
code = code.replace(old_forecast, new_forecast)

# 5. _simulate_qrw
old_simulate = """    def _simulate_qrw(
        self,
        model: AdaptiveDecoherenceQRW,
        *,
        seed: int,
    ) -> np.ndarray:
        \"\"\"Evolve a density matrix and sample each fixed-origin marginal.

        The samples are forecast-horizon marginals, not repeatedly observed
        classical trajectories.  Keeping that distinction avoids destroying
        interference through an implicit position measurement after every step.
        \"\"\"
        rng = np.random.default_rng(seed)
        features, ar1 = self._forecast_qrw_features(model)
        engine = DensityMatrixQRW(2 * self.n_steps + 3)
        movement_probability = float(model.movement_probability)
        marginals = np.empty(
            (self.n_paths, self.n_steps + 1),
            dtype=np.float64,
        )
        marginals[:, 0] = self.initial_price
        quantum_variance = np.empty(self.n_steps, dtype=np.float64)
        for index, feature in enumerate(features):
            previous_rho = engine.rho.copy()
            event_gamma, coin = model._event_kernel(feature)
            engine.step_with_decoherence(event_gamma, coin_matrix=coin)"""

new_simulate = """    def _simulate_qrw(
        self,
        model: MarketQRW,
        *,
        seed: int,
    ) -> np.ndarray:
        \"\"\"Evolve a density matrix and sample each fixed-origin marginal.

        The samples are forecast-horizon marginals, not repeatedly observed
        classical trajectories.  Keeping that distinction avoids destroying
        interference through an implicit position measurement after every step.
        \"\"\"
        rng = np.random.default_rng(seed)
        features, ar1 = self._forecast_qrw_features(model)
        engine = DensityMatrixQRW(2 * self.n_steps + 3)
        movement_probability = float(model.movement_probability)
        marginals = np.empty(
            (self.n_paths, self.n_steps + 1),
            dtype=np.float64,
        )
        marginals[:, 0] = self.initial_price
        quantum_variance = np.empty(self.n_steps, dtype=np.float64)
        for index, feature in enumerate(features):
            previous_rho = engine.rho.copy()
            event_gamma = model.gamma
            coin = su2_market_coin(
                float(feature[0]),
                float(feature[1]),
                bias=model.obi_bias,
                alpha_obi=model.alpha_obi,
                alpha_direction=model.alpha_direction,
                alpha_phase=getattr(model, "alpha_phase", 0.0),
                window=getattr(model, "quantum_window_size", 1),
            )
            engine.step_with_decoherence(event_gamma, coin_matrix=coin)"""
code = code.replace(old_simulate, new_simulate)

# 6. _rolling_one_step_absolute_losses
old_rolling = """        qrw_predictor = AdaptiveDecoherenceQRW(
            predictor_frame,
            {
                "n_positions": 101,
                "gamma_base": qrw_parameters["gamma"],
                "obi_bias": qrw_parameters["obi_bias"],
                "alpha_obi": qrw_parameters["alpha_obi"],
                "alpha_direction": qrw_parameters["alpha_direction"],
                "alpha_obi_change": qrw_parameters["alpha_obi_change"],
                "alpha_abs_obi": qrw_parameters["alpha_abs_obi"],
                "gamma_intensity": qrw_parameters["gamma_intensity"],
                "feature_mean": qrw_parameters["feature_mean"],
                "feature_scale": qrw_parameters["feature_scale"],
                "movement_probability": qrw_parameters[
                    "movement_probability"
                ],
            },
        )
        probability_up = qrw_predictor.predict_probability()[1:]"""

new_rolling = """        qrw_predictor = MarketQRW(
            predictor_frame,
            {
                "n_positions": 101,
                "gamma_base": qrw_parameters.get("gamma", 0.0),
                "obi_bias": qrw_parameters.get("obi_bias", 0.0),
                "alpha_obi": qrw_parameters.get("alpha_obi", 0.0),
                "alpha_direction": qrw_parameters.get("alpha_direction", 0.0),
                "coin_type": "obi_adaptive",
            },
        )
        probability_up = np.clip(
            qrw_predictor.quantum_probabilities(
                predictor_frame["obi"].to_numpy(dtype=np.float64)[:-1],
                predictor_frame["tick_direction"].to_numpy(dtype=np.float64)[:-1]
            ),
            1e-12,
            1.0 - 1e-12,
        )[1:]"""
code = code.replace(old_rolling, new_rolling)

# 7. run (AdaptiveDecoherenceQRW init)
old_run = """        train_qrw_probability = np.clip(
            AdaptiveDecoherenceQRW(
                self.train.iloc[:-1].copy(),
                {
                    "n_positions": 101,
                    "gamma_base": qrw_parameters["gamma"],
                    "obi_bias": qrw_parameters["obi_bias"],
                    "alpha_obi": qrw_parameters["alpha_obi"],
                    "alpha_direction": qrw_parameters["alpha_direction"],
                    "alpha_obi_change": qrw_parameters["alpha_obi_change"],
                    "alpha_abs_obi": qrw_parameters["alpha_abs_obi"],
                    "gamma_intensity": qrw_parameters["gamma_intensity"],
                    "feature_mean": qrw_parameters["feature_mean"],
                    "feature_scale": qrw_parameters["feature_scale"],
                },
            ).predict_probability()[train_valid],
            1e-12,
            1.0 - 1e-12,
        )"""

new_run = """        train_qrw_probability = np.clip(
            MarketQRW(
                self.train.iloc[:-1].copy(),
                {
                    "n_positions": 101,
                    "gamma_base": qrw_parameters.get("gamma", 0.0),
                    "obi_bias": qrw_parameters.get("obi_bias", 0.0),
                    "alpha_obi": qrw_parameters.get("alpha_obi", 0.0),
                    "alpha_direction": qrw_parameters.get("alpha_direction", 0.0),
                    "coin_type": "obi_adaptive",
                },
            ).quantum_probabilities(
                self.train["obi"].to_numpy(dtype=np.float64)[:-1],
                self.train["tick_direction"].to_numpy(dtype=np.float64)[:-1]
            )[train_valid],
            1e-12,
            1.0 - 1e-12,
        )"""
code = code.replace(old_run, new_run)

# 8. diagnostics
old_diag = """            "qrw_forecast": {
                "state_evolution": getattr(
                    AdaptiveDecoherenceQRW(
                        self.train,
                        {"n_positions": 2},
                    ),
                    "SIMULATION_ENGINE",
                    "density_matrix"
                ),
                "position_variance_by_horizon": (
                    self.forecast_samples["QRW Adaptive"].var(axis=0)[1:].tolist()
                ),
                "alpha_obi": qrw_parameters["alpha_obi"],
                "alpha_direction": qrw_parameters["alpha_direction"],
                "alpha_obi_change": qrw_parameters["alpha_obi_change"],
                "alpha_abs_obi": qrw_parameters["alpha_abs_obi"],
                "obi_bias": qrw_parameters["obi_bias"],
                "gamma_intensity": qrw_parameters["gamma_intensity"],
            },"""

new_diag = """            "qrw_forecast": {
                "state_evolution": getattr(
                    MarketQRW,
                    "SIMULATION_ENGINE",
                    "density_matrix"
                ),
                "position_variance_by_horizon": (
                    self.forecast_samples["QRW Adaptive"].var(axis=0)[1:].tolist()
                ),
                "alpha_obi": qrw_parameters.get("alpha_obi", 0.0),
                "alpha_direction": qrw_parameters.get("alpha_direction", 0.0),
                "obi_bias": qrw_parameters.get("obi_bias", 0.0),
                "gamma": qrw_parameters.get("gamma", 0.0),
                "alpha_phase": qrw_parameters.get("alpha_phase", 0.0),
            },"""
code = code.replace(old_diag, new_diag)

with open("src/evaluation/benchmark_suite.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Migration completed.")
