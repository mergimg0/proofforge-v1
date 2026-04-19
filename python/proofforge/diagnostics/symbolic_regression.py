"""Idea Reactor: Symbolic Regression for Scaling Law Prediction (App 7).

Fits parametric models to the GRPO pass rate curve to predict:
  - Steps needed to reach a target pass rate
  - Ceiling of the current training run
  - Inflection point (breakout step)

The pass rate curve has three regimes:
  1. Initial rapid rise (steps 0-5): formatting improvement
  2. Plateau (steps 5-30): infrastructure accumulation
  3. Sigmoid rise (steps 30+): infrastructure-driven improvement

A three-parameter sigmoid fits regime 3:
  pass_rate(step) = L / (1 + exp(-k * (step - step_0)))

Where L = ceiling, k = growth rate, step_0 = inflection point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SigmoidModel:
    """Three-parameter sigmoid model for pass rate curves.

    pass_rate(step) = L / (1 + exp(-k * (step - step_0)))
    """
    ceiling: float      # L: asymptotic ceiling
    growth_rate: float  # k: steepness of the sigmoid
    inflection: float   # step_0: inflection point (50% of ceiling)

    def predict(self, step: float) -> float:
        """Predict pass rate at a given step."""
        return self.ceiling / (1.0 + np.exp(-self.growth_rate * (step - self.inflection)))

    def predict_steps(self, steps: np.ndarray) -> np.ndarray:
        """Predict pass rates for an array of steps."""
        return self.ceiling / (1.0 + np.exp(-self.growth_rate * (steps - self.inflection)))

    def steps_to_target(self, target: float) -> Optional[float]:
        """Predict steps needed to reach target pass rate.

        Returns None if target exceeds ceiling.
        """
        if target >= self.ceiling:
            return None
        if target <= 0:
            return 0.0

        ratio = self.ceiling / target - 1.0
        if ratio <= 0:
            return None

        return self.inflection - np.log(ratio) / self.growth_rate

    def summary(self) -> dict:
        return {
            "ceiling": round(self.ceiling, 4),
            "growth_rate": round(self.growth_rate, 6),
            "inflection_step": round(self.inflection, 1),
            "step_to_50pct": round(s, 1) if self.ceiling > 0.5 and (s := self.steps_to_target(0.5)) is not None else None,
            "step_to_80pct": round(s, 1) if self.ceiling > 0.8 and (s := self.steps_to_target(0.8)) is not None else None,
        }


@dataclass
class ScalingLawFitter:
    """Fit scaling laws to GRPO pass rate data.

    Implements least-squares fitting of the three-parameter sigmoid
    to observed (step, pass_rate) data. Supports:
    - Fitting from partial data (predict future from early steps)
    - Leave-one-out cross-validation for confidence estimation
    - Transfer testing (fit on dataset A, predict dataset B)
    """

    steps: list[float] = field(default_factory=list)
    pass_rates: list[float] = field(default_factory=list)

    def add_observation(self, step: float, pass_rate: float) -> None:
        """Record an observation."""
        self.steps.append(step)
        self.pass_rates.append(pass_rate)

    def add_observations(self, steps: list[float], pass_rates: list[float]) -> None:
        """Record multiple observations."""
        self.steps.extend(steps)
        self.pass_rates.extend(pass_rates)

    def fit(
        self,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> Optional[SigmoidModel]:
        """Fit sigmoid model to observed data.

        Args:
            start_step: Only use observations from this step onward
                (skip formatting phase)
            end_step: Only use observations up to this step
                (for holdout validation)

        Returns SigmoidModel or None if fitting fails.
        """
        from scipy.optimize import curve_fit

        x = np.array(self.steps)
        y = np.array(self.pass_rates)

        mask = np.ones(len(x), dtype=bool)
        if start_step is not None:
            mask &= x >= start_step
        if end_step is not None:
            mask &= x <= end_step

        x_fit = x[mask]
        y_fit = y[mask]

        if len(x_fit) < 4:
            return None

        def sigmoid(step, L, k, s0):
            return L / (1.0 + np.exp(-k * (step - s0)))

        try:
            # Initial guesses from data
            L_guess = min(max(y_fit) * 1.2, 1.0)
            s0_guess = x_fit[len(x_fit) // 2]
            k_guess = 0.05

            popt, _ = curve_fit(
                sigmoid,
                x_fit,
                y_fit,
                p0=[L_guess, k_guess, s0_guess],
                bounds=([0.01, 0.001, 0], [1.0, 1.0, max(x_fit) * 2]),
                maxfev=5000,
            )

            return SigmoidModel(
                ceiling=float(popt[0]),
                growth_rate=float(popt[1]),
                inflection=float(popt[2]),
            )
        except (RuntimeError, ValueError):
            return None

    def validate_holdout(
        self, train_end_step: int
    ) -> Optional[dict]:
        """Fit on data up to train_end_step, validate on remaining.

        Returns validation metrics or None if insufficient data.
        """
        model = self.fit(end_step=train_end_step)
        if model is None:
            return None

        x = np.array(self.steps)
        y = np.array(self.pass_rates)

        holdout_mask = x > train_end_step
        if not holdout_mask.any():
            return None

        x_hold = x[holdout_mask]
        y_hold = y[holdout_mask]
        y_pred = model.predict_steps(x_hold)

        residuals = y_hold - y_pred
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        max_error = float(np.max(np.abs(residuals)))

        return {
            "model": model.summary(),
            "train_steps": int(train_end_step),
            "holdout_size": len(x_hold),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "max_error": round(max_error, 4),
            "holdout_range": [float(x_hold.min()), float(x_hold.max())],
        }

    def predict_training_time(self, target_pass_rate: float) -> Optional[dict]:
        """Predict steps needed to reach target pass rate.

        Fits the full dataset and extrapolates.
        """
        model = self.fit()
        if model is None:
            return None

        steps_needed = model.steps_to_target(target_pass_rate)
        if steps_needed is None:
            return {
                "reachable": False,
                "target": target_pass_rate,
                "ceiling": model.ceiling,
                "reason": f"Target {target_pass_rate:.0%} exceeds predicted ceiling {model.ceiling:.0%}",
            }

        return {
            "reachable": True,
            "target": target_pass_rate,
            "predicted_steps": round(steps_needed, 0),
            "ceiling": model.ceiling,
            "current_steps": max(self.steps) if self.steps else 0,
            "remaining_steps": max(0, round(steps_needed - max(self.steps, default=0), 0)),
        }
