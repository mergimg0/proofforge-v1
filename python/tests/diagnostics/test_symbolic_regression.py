"""Tests for proofforge.diagnostics.symbolic_regression."""

import numpy as np
import pytest

from proofforge.diagnostics.symbolic_regression import ScalingLawFitter, SigmoidModel


# ---------------------------------------------------------------------------
# SigmoidModel tests
# ---------------------------------------------------------------------------


class TestSigmoidModelPredict:
    """Tests for SigmoidModel.predict()."""

    def _make_model(self, ceiling=0.8, growth_rate=0.1, inflection=50.0) -> SigmoidModel:
        return SigmoidModel(ceiling=ceiling, growth_rate=growth_rate, inflection=inflection)

    def test_predict_at_inflection_point_equals_half_ceiling(self):
        """At the inflection step the sigmoid equals exactly ceiling / 2."""
        model = self._make_model(ceiling=0.8, growth_rate=0.1, inflection=50.0)
        result = model.predict(50.0)
        assert result == pytest.approx(0.4, abs=1e-10)

    def test_predict_far_left_approaches_zero(self):
        """Far to the left of inflection the prediction is effectively 0."""
        model = self._make_model(ceiling=0.8, growth_rate=0.1, inflection=50.0)
        result = model.predict(-1000.0)
        assert result < 1e-30

    def test_predict_far_right_approaches_ceiling(self):
        """Far to the right of inflection the prediction is effectively the ceiling."""
        model = self._make_model(ceiling=0.8, growth_rate=0.1, inflection=50.0)
        result = model.predict(1000.0)
        assert result == pytest.approx(0.8, abs=1e-10)

    def test_predict_monotone_increasing(self):
        """Predictions should be strictly increasing with step."""
        model = self._make_model()
        steps = [10.0, 30.0, 50.0, 70.0, 90.0]
        preds = [model.predict(s) for s in steps]
        for a, b in zip(preds, preds[1:]):
            assert a < b

    def test_predict_ceiling_respected(self):
        """Predictions never exceed the ceiling."""
        model = self._make_model(ceiling=0.75)
        for step in np.linspace(-100, 1000, 200):
            assert model.predict(float(step)) <= model.ceiling + 1e-12

    def test_predict_symmetry_around_inflection(self):
        """Points equidistant from inflection are symmetric around ceiling/2."""
        model = self._make_model(ceiling=0.6, growth_rate=0.2, inflection=30.0)
        delta = 20.0
        low = model.predict(30.0 - delta)
        high = model.predict(30.0 + delta)
        assert low + high == pytest.approx(0.6, abs=1e-10)


class TestSigmoidModelPredictSteps:
    """Tests for SigmoidModel.predict_steps()."""

    def test_predict_steps_returns_array(self):
        """predict_steps returns an ndarray of the same length."""
        model = SigmoidModel(ceiling=0.8, growth_rate=0.1, inflection=50.0)
        steps = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
        result = model.predict_steps(steps)
        assert isinstance(result, np.ndarray)
        assert result.shape == steps.shape

    def test_predict_steps_matches_predict(self):
        """predict_steps matches element-wise calls to predict."""
        model = SigmoidModel(ceiling=0.8, growth_rate=0.1, inflection=50.0)
        steps = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        result = model.predict_steps(steps)
        expected = np.array([model.predict(s) for s in steps])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_predict_steps_at_inflection(self):
        """Inflection entry in array equals ceiling / 2."""
        model = SigmoidModel(ceiling=1.0, growth_rate=0.05, inflection=40.0)
        steps = np.array([40.0])
        result = model.predict_steps(steps)
        assert result[0] == pytest.approx(0.5, abs=1e-10)


class TestSigmoidModelStepsToTarget:
    """Tests for SigmoidModel.steps_to_target()."""

    def setup_method(self):
        self.model = SigmoidModel(ceiling=0.8, growth_rate=0.1, inflection=50.0)

    def test_steps_to_target_at_half_ceiling_is_inflection(self):
        """steps_to_target(ceiling/2) returns the inflection step."""
        result = self.model.steps_to_target(0.4)
        assert result == pytest.approx(50.0, abs=1e-8)

    def test_steps_to_target_reachable_target_correct_step(self):
        """For a reachable target the predicted step satisfies predict(step) == target."""
        target = 0.6
        step = self.model.steps_to_target(target)
        assert step is not None
        assert self.model.predict(step) == pytest.approx(target, abs=1e-6)

    def test_steps_to_target_above_ceiling_returns_none(self):
        """When target >= ceiling, returns None."""
        assert self.model.steps_to_target(0.8) is None
        assert self.model.steps_to_target(0.9) is None
        assert self.model.steps_to_target(1.0) is None

    def test_steps_to_target_zero_or_negative_returns_zero(self):
        """When target <= 0, returns 0 (already achieved)."""
        assert self.model.steps_to_target(0.0) == 0.0
        assert self.model.steps_to_target(-0.5) == 0.0

    def test_steps_to_target_increases_monotonically(self):
        """Higher targets require more steps."""
        targets = [0.1, 0.3, 0.5, 0.7]
        steps = [self.model.steps_to_target(t) for t in targets]
        assert all(s is not None for s in steps)
        concrete: list[float] = [s for s in steps if s is not None]
        for a, b in zip(concrete, concrete[1:]):
            assert a < b


class TestSigmoidModelSummary:
    """Tests for SigmoidModel.summary()."""

    def test_summary_returns_dict(self):
        model = SigmoidModel(ceiling=0.8, growth_rate=0.1, inflection=50.0)
        result = model.summary()
        assert isinstance(result, dict)

    def test_summary_contains_required_keys(self):
        model = SigmoidModel(ceiling=0.8, growth_rate=0.1, inflection=50.0)
        result = model.summary()
        expected_keys = {"ceiling", "growth_rate", "inflection_step", "step_to_50pct", "step_to_80pct"}
        assert expected_keys == set(result.keys())

    def test_summary_ceiling_matches(self):
        model = SigmoidModel(ceiling=0.75, growth_rate=0.2, inflection=30.0)
        assert model.summary()["ceiling"] == pytest.approx(0.75, abs=1e-4)

    def test_summary_step_to_50pct_none_when_ceiling_too_low(self):
        """step_to_50pct is None when ceiling <= 0.5."""
        model = SigmoidModel(ceiling=0.4, growth_rate=0.1, inflection=50.0)
        result = model.summary()
        assert result["step_to_50pct"] is None

    def test_summary_step_to_80pct_none_when_ceiling_too_low(self):
        """step_to_80pct is None when ceiling <= 0.8."""
        model = SigmoidModel(ceiling=0.75, growth_rate=0.1, inflection=50.0)
        result = model.summary()
        assert result["step_to_80pct"] is None

    def test_summary_step_to_50pct_consistent_with_predict(self):
        """The reported step_to_50pct should satisfy predict(step) ≈ 0.5."""
        model = SigmoidModel(ceiling=0.9, growth_rate=0.1, inflection=50.0)
        summary = model.summary()
        step = summary["step_to_50pct"]
        assert step is not None
        assert model.predict(step) == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------------------
# ScalingLawFitter tests
# ---------------------------------------------------------------------------


class TestScalingLawFitterObservations:
    """Tests for add_observation and add_observations."""

    def test_add_observation_accumulates_single(self):
        fitter = ScalingLawFitter()
        fitter.add_observation(10.0, 0.2)
        assert fitter.steps == [10.0]
        assert fitter.pass_rates == [0.2]

    def test_add_observation_accumulates_multiple(self):
        fitter = ScalingLawFitter()
        fitter.add_observation(10.0, 0.1)
        fitter.add_observation(20.0, 0.3)
        fitter.add_observation(30.0, 0.5)
        assert len(fitter.steps) == 3
        assert fitter.steps == [10.0, 20.0, 30.0]

    def test_add_observations_batch(self):
        fitter = ScalingLawFitter()
        steps = [10.0, 20.0, 30.0, 40.0]
        rates = [0.1, 0.2, 0.4, 0.6]
        fitter.add_observations(steps, rates)
        assert fitter.steps == steps
        assert fitter.pass_rates == rates

    def test_add_observations_extends_existing(self):
        fitter = ScalingLawFitter()
        fitter.add_observation(5.0, 0.05)
        fitter.add_observations([10.0, 20.0], [0.1, 0.3])
        assert len(fitter.steps) == 3
        assert fitter.steps[0] == 5.0


class TestScalingLawFitterFit:
    """Tests for ScalingLawFitter.fit()."""

    @pytest.fixture(autouse=True)
    def _skip_without_scipy(self):
        pytest.importorskip("scipy")

    def _make_synthetic_data(
        self,
        ceiling=0.75,
        growth_rate=0.08,
        inflection=40.0,
        step_start=10,
        step_end=101,
        step_size=5,
        noise_std=0.005,
        rng_seed=42,
    ):
        """Generate clean synthetic data from a known SigmoidModel."""
        rng = np.random.default_rng(rng_seed)
        true_model = SigmoidModel(ceiling=ceiling, growth_rate=growth_rate, inflection=inflection)
        steps: list[float] = [float(s) for s in range(step_start, step_end, step_size)]
        rates = [
            float(np.clip(true_model.predict(s) + rng.normal(0, noise_std), 0.0, 1.0))
            for s in steps
        ]
        return steps, rates, true_model

    def test_fit_recovers_ceiling_approximately(self):
        """Fit on clean sigmoid data recovers ceiling within 10%."""
        steps, rates, true_model = self._make_synthetic_data()
        fitter = ScalingLawFitter()
        fitter.add_observations(steps, rates)
        fitted = fitter.fit()
        assert fitted is not None
        assert fitted.ceiling == pytest.approx(true_model.ceiling, rel=0.10)

    def test_fit_recovers_inflection_approximately(self):
        """Fit on clean sigmoid data recovers inflection within 20%."""
        steps, rates, true_model = self._make_synthetic_data()
        fitter = ScalingLawFitter()
        fitter.add_observations(steps, rates)
        fitted = fitter.fit()
        assert fitted is not None
        assert fitted.inflection == pytest.approx(true_model.inflection, rel=0.20)

    def test_fit_returns_sigmoid_model(self):
        """fit() returns a SigmoidModel instance on success."""
        steps, rates, _ = self._make_synthetic_data()
        fitter = ScalingLawFitter()
        fitter.add_observations(steps, rates)
        fitted = fitter.fit()
        assert isinstance(fitted, SigmoidModel)

    def test_fit_with_too_few_points_returns_none(self):
        """fit() returns None when fewer than 4 points pass the mask."""
        fitter = ScalingLawFitter()
        fitter.add_observations([10.0, 20.0, 30.0], [0.1, 0.2, 0.4])
        assert fitter.fit() is None

    def test_fit_with_exactly_three_filtered_points_returns_none(self):
        """When start_step/end_step leaves fewer than 4 points, returns None."""
        steps, rates, _ = self._make_synthetic_data()
        fitter = ScalingLawFitter()
        fitter.add_observations(steps, rates)
        # Restrict to only the first 3 steps
        result = fitter.fit(end_step=int(steps[2]))
        assert result is None

    def test_fit_start_step_filters_early_data(self):
        """start_step excludes points before it from the fit."""
        steps, rates, true_model = self._make_synthetic_data()
        fitter = ScalingLawFitter()
        fitter.add_observations(steps, rates)
        # With or without start filtering, a fit should succeed on the right range
        fitted = fitter.fit(start_step=30)
        assert fitted is not None

    def test_fit_end_step_filters_late_data(self):
        """end_step excludes points after it from the fit."""
        steps, rates, _ = self._make_synthetic_data(step_end=201, step_size=5)
        fitter = ScalingLawFitter()
        fitter.add_observations(steps, rates)
        # Fit only up to step 60 — should still have >=4 points
        fitted = fitter.fit(end_step=60)
        assert fitted is not None

    def test_fit_combined_start_end_step_filtering(self):
        """Combining start_step and end_step applies both masks."""
        steps, rates, _ = self._make_synthetic_data(step_end=201, step_size=5)
        fitter = ScalingLawFitter()
        fitter.add_observations(steps, rates)
        fitted = fitter.fit(start_step=20, end_step=80)
        assert fitted is not None


class TestScalingLawFitterValidateHoldout:
    """Tests for ScalingLawFitter.validate_holdout()."""

    @pytest.fixture(autouse=True)
    def _skip_without_scipy(self):
        pytest.importorskip("scipy")

    def _make_fitter_with_data(self, rng_seed=0) -> tuple[ScalingLawFitter, SigmoidModel]:
        rng = np.random.default_rng(rng_seed)
        true_model = SigmoidModel(ceiling=0.75, growth_rate=0.08, inflection=40.0)
        steps: list[float] = [float(s) for s in range(10, 101, 5)]
        rates = [
            float(np.clip(true_model.predict(s) + rng.normal(0, 0.005), 0.0, 1.0))
            for s in steps
        ]
        fitter = ScalingLawFitter()
        fitter.add_observations(steps, rates)
        return fitter, true_model

    def test_validate_holdout_returns_dict(self):
        fitter, _ = self._make_fitter_with_data()
        result = fitter.validate_holdout(train_end_step=50)
        assert isinstance(result, dict)

    def test_validate_holdout_contains_required_keys(self):
        fitter, _ = self._make_fitter_with_data()
        result = fitter.validate_holdout(train_end_step=50)
        assert result is not None
        for key in ("model", "train_steps", "holdout_size", "mae", "rmse", "max_error", "holdout_range"):
            assert key in result, f"Missing key: {key}"

    def test_validate_holdout_mae_is_float(self):
        fitter, _ = self._make_fitter_with_data()
        result = fitter.validate_holdout(train_end_step=50)
        assert result is not None
        assert isinstance(result["mae"], float)

    def test_validate_holdout_low_mae_on_clean_data(self):
        """On near-noiseless data the holdout MAE should be small."""
        rng = np.random.default_rng(7)
        true_model = SigmoidModel(ceiling=0.75, growth_rate=0.08, inflection=40.0)
        steps: list[float] = [float(s) for s in range(10, 121, 5)]
        rates = [float(true_model.predict(s) + rng.normal(0, 0.001)) for s in steps]
        fitter = ScalingLawFitter()
        fitter.add_observations(steps, rates)
        result = fitter.validate_holdout(train_end_step=60)
        assert result is not None
        assert result["mae"] < 0.10

    def test_validate_holdout_train_steps_matches_input(self):
        fitter, _ = self._make_fitter_with_data()
        result = fitter.validate_holdout(train_end_step=55)
        assert result is not None
        assert result["train_steps"] == 55

    def test_validate_holdout_returns_none_when_no_holdout_data(self):
        """Returns None when no observations fall after train_end_step."""
        true_model = SigmoidModel(ceiling=0.75, growth_rate=0.08, inflection=40.0)
        steps: list[float] = [10.0, 20.0, 30.0, 40.0, 50.0]
        rates = [true_model.predict(s) for s in steps]
        fitter = ScalingLawFitter()
        fitter.add_observations(steps, rates)
        # train_end_step beyond all data
        result = fitter.validate_holdout(train_end_step=1000)
        assert result is None

    def test_validate_holdout_fit_on_first_half_validates_on_second(self):
        """The holdout range starts after train_end_step."""
        fitter, _ = self._make_fitter_with_data()
        result = fitter.validate_holdout(train_end_step=50)
        assert result is not None
        holdout_min = result["holdout_range"][0]
        assert holdout_min > 50


class TestScalingLawFitterPredictTrainingTime:
    """Tests for ScalingLawFitter.predict_training_time()."""

    @pytest.fixture(autouse=True)
    def _skip_without_scipy(self):
        pytest.importorskip("scipy")

    def _make_fitted_fitter(self, ceiling=0.75, rng_seed=0) -> ScalingLawFitter:
        rng = np.random.default_rng(rng_seed)
        true_model = SigmoidModel(ceiling=ceiling, growth_rate=0.08, inflection=40.0)
        steps: list[float] = [float(s) for s in range(10, 101, 5)]
        rates = [
            float(np.clip(true_model.predict(s) + rng.normal(0, 0.005), 0.0, 1.0))
            for s in steps
        ]
        fitter = ScalingLawFitter()
        fitter.add_observations(steps, rates)
        return fitter

    def test_predict_training_time_reachable_target(self):
        """A target below the ceiling should be reachable."""
        fitter = self._make_fitted_fitter(ceiling=0.75)
        result = fitter.predict_training_time(0.5)
        assert result is not None
        assert result["reachable"] is True
        assert "predicted_steps" in result
        assert result["target"] == pytest.approx(0.5)

    def test_predict_training_time_reachable_contains_ceiling(self):
        """Reachable result includes the predicted ceiling."""
        fitter = self._make_fitted_fitter(ceiling=0.75)
        result = fitter.predict_training_time(0.5)
        assert result is not None
        assert "ceiling" in result

    def test_predict_training_time_unreachable_target_above_ceiling(self):
        """Target above predicted ceiling returns reachable=False."""
        fitter = self._make_fitted_fitter(ceiling=0.75)
        result = fitter.predict_training_time(0.99)
        assert result is not None
        assert result["reachable"] is False
        assert "reason" in result

    def test_predict_training_time_unreachable_has_ceiling(self):
        fitter = self._make_fitted_fitter(ceiling=0.75)
        result = fitter.predict_training_time(0.99)
        assert result is not None
        assert "ceiling" in result

    def test_predict_training_time_returns_none_with_too_few_points(self):
        """Returns None when fit fails due to insufficient data."""
        fitter = ScalingLawFitter()
        fitter.add_observations([10.0, 20.0], [0.1, 0.2])
        assert fitter.predict_training_time(0.5) is None
