"""Tests for proofforge.diagnostics.training_dynamics."""

import pytest

from proofforge.controller.phase_detector import TrainingPhase
from proofforge.diagnostics.training_dynamics import PhaseAnnotation, ThreePhaseAnalyzer


# ---------------------------------------------------------------------------
# PhaseAnnotation tests
# ---------------------------------------------------------------------------


class TestPhaseAnnotation:
    """Tests for PhaseAnnotation dataclass creation and field access."""

    def test_creation_with_all_fields(self):
        """PhaseAnnotation can be constructed with all required fields."""
        ann = PhaseAnnotation(
            step=42,
            phase=TrainingPhase.ACCUMULATION,
            reward_mean=0.15,
            pass_rate=0.04,
            reward_trend=0.005,
            leading_indicator=True,
        )
        assert ann.step == 42
        assert ann.phase == TrainingPhase.ACCUMULATION
        assert ann.reward_mean == 0.15
        assert ann.pass_rate == 0.04
        assert ann.reward_trend == 0.005
        assert ann.leading_indicator is True

    def test_is_frozen(self):
        """PhaseAnnotation is immutable (frozen dataclass)."""
        ann = PhaseAnnotation(
            step=1,
            phase=TrainingPhase.DISRUPTION,
            reward_mean=0.05,
            pass_rate=0.02,
            reward_trend=0.0,
            leading_indicator=False,
        )
        with pytest.raises((AttributeError, TypeError)):
            ann.step = 99  # type: ignore[misc]

    def test_all_training_phases_accepted(self):
        """PhaseAnnotation accepts any TrainingPhase value."""
        for phase in TrainingPhase:
            ann = PhaseAnnotation(
                step=0,
                phase=phase,
                reward_mean=0.0,
                pass_rate=0.0,
                reward_trend=0.0,
                leading_indicator=False,
            )
            assert ann.phase == phase


# ---------------------------------------------------------------------------
# ThreePhaseAnalyzer.add_step tests
# ---------------------------------------------------------------------------


class TestThreePhaseAnalyzerAddStep:
    """Tests for ThreePhaseAnalyzer.add_step()."""

    def test_add_step_accumulates_data(self):
        """Each add_step call appends to all three internal lists."""
        analyzer = ThreePhaseAnalyzer()
        analyzer.add_step(0, reward=0.05, pass_rate=0.01)
        analyzer.add_step(1, reward=0.06, pass_rate=0.02)
        assert analyzer.steps == [0, 1]
        assert analyzer.rewards == [0.05, 0.06]
        assert analyzer.pass_rates == [0.01, 0.02]

    def test_add_step_preserves_order(self):
        """Steps are appended in insertion order."""
        analyzer = ThreePhaseAnalyzer()
        for i in range(5):
            analyzer.add_step(i * 10, reward=float(i) * 0.1, pass_rate=float(i) * 0.05)
        assert analyzer.steps == [0, 10, 20, 30, 40]


# ---------------------------------------------------------------------------
# ThreePhaseAnalyzer.annotate tests
# ---------------------------------------------------------------------------


class TestThreePhaseAnalyzerAnnotate:
    """Tests for ThreePhaseAnalyzer.annotate()."""

    def _add_constant_steps(
        self,
        analyzer: ThreePhaseAnalyzer,
        n: int,
        reward: float = 0.05,
        pass_rate: float = 0.02,
        start: int = 0,
    ) -> None:
        for i in range(n):
            analyzer.add_step(start + i, reward=reward, pass_rate=pass_rate)

    def test_annotate_returns_list(self):
        analyzer = ThreePhaseAnalyzer()
        self._add_constant_steps(analyzer, 20)
        result = analyzer.annotate()
        assert isinstance(result, list)

    def test_annotate_insufficient_data_returns_empty_list(self):
        """With fewer steps than reward_window, annotate returns []."""
        analyzer = ThreePhaseAnalyzer(reward_window=10)
        self._add_constant_steps(analyzer, 9)
        assert analyzer.annotate() == []

    def test_annotate_exactly_at_window_size_returns_empty(self):
        """The loop starts at index reward_window, so == window yields 0 annotations."""
        analyzer = ThreePhaseAnalyzer(reward_window=10)
        self._add_constant_steps(analyzer, 10)
        assert analyzer.annotate() == []

    def test_annotate_returns_phase_annotations(self):
        """Each element of annotate() is a PhaseAnnotation."""
        analyzer = ThreePhaseAnalyzer()
        self._add_constant_steps(analyzer, 20)
        for ann in analyzer.annotate():
            assert isinstance(ann, PhaseAnnotation)

    def test_annotate_count_equals_steps_minus_window(self):
        """Number of annotations equals len(steps) - reward_window."""
        n = 25
        window = 10
        analyzer = ThreePhaseAnalyzer(reward_window=window)
        self._add_constant_steps(analyzer, n)
        result = analyzer.annotate()
        assert len(result) == n - window

    def test_annotate_disruption_data_labels_as_disruption(self):
        """Low reward and low pass rate throughout -> DISRUPTION phase."""
        analyzer = ThreePhaseAnalyzer()
        # reward_mean < 0.1 and pass_rate < disruption_threshold (0.05)
        self._add_constant_steps(analyzer, 25, reward=0.03, pass_rate=0.01)
        phases = {a.phase for a in analyzer.annotate()}
        assert TrainingPhase.DISRUPTION in phases
        # All should be disruption given constant low values
        assert phases == {TrainingPhase.DISRUPTION}

    def test_annotate_breakout_data_labels_as_breakout(self):
        """Rising reward_mean > 0.25 with rising pass rate -> BREAKOUT phase."""
        analyzer = ThreePhaseAnalyzer()
        # Build data where reward is high and pass rate is climbing
        for i in range(25):
            reward = 0.3 + i * 0.01   # well above breakout_reward_threshold
            pr = 0.05 + i * 0.02      # rising above breakout_pass_rate_slope
            analyzer.add_step(i, reward=reward, pass_rate=pr)
        phases = {a.phase for a in analyzer.annotate()}
        assert TrainingPhase.BREAKOUT in phases

    def test_annotate_each_annotation_has_correct_step(self):
        """Each annotation's step corresponds to the correct position in steps list."""
        analyzer = ThreePhaseAnalyzer(reward_window=5)
        for i in range(15):
            analyzer.add_step(i * 2, reward=0.05, pass_rate=0.01)
        annotations = analyzer.annotate()
        # Annotations start at index reward_window in the steps list
        for idx, ann in enumerate(annotations):
            expected_step = analyzer.steps[analyzer.reward_window + idx]
            assert ann.step == expected_step

    def test_annotate_reward_mean_is_rounded(self):
        """reward_mean in annotations is rounded to 4 decimal places."""
        analyzer = ThreePhaseAnalyzer(reward_window=5)
        for i in range(10):
            analyzer.add_step(i, reward=0.123456789, pass_rate=0.01)
        for ann in analyzer.annotate():
            # Check that rounding has been applied (at most 4 decimal digits)
            s = str(ann.reward_mean)
            if "." in s:
                decimals = len(s.split(".")[1])
                assert decimals <= 4


# ---------------------------------------------------------------------------
# ThreePhaseAnalyzer.find_transitions tests
# ---------------------------------------------------------------------------


class TestThreePhaseAnalyzerFindTransitions:
    """Tests for ThreePhaseAnalyzer.find_transitions()."""

    def test_find_transitions_returns_list(self):
        analyzer = ThreePhaseAnalyzer()
        for i in range(20):
            analyzer.add_step(i, reward=0.05, pass_rate=0.01)
        result = analyzer.find_transitions()
        assert isinstance(result, list)

    def test_find_transitions_constant_phase_yields_empty(self):
        """No transitions when every step stays in the same phase."""
        analyzer = ThreePhaseAnalyzer()
        for i in range(30):
            analyzer.add_step(i, reward=0.03, pass_rate=0.01)
        result = analyzer.find_transitions()
        assert result == []

    def test_find_transitions_detects_phase_change(self):
        """A clear disruption-to-accumulation shift produces at least one transition."""
        analyzer = ThreePhaseAnalyzer()
        # Disruption phase: low reward, low pass rate
        for i in range(20):
            analyzer.add_step(i, reward=0.03, pass_rate=0.01)
        # Accumulation phase: reward climbing, pass rate still flat
        for i in range(20, 50):
            reward = 0.03 + (i - 20) * 0.006   # rises above accumulation threshold
            analyzer.add_step(i, reward=reward, pass_rate=0.04)
        transitions = analyzer.find_transitions()
        assert len(transitions) >= 1

    def test_find_transitions_each_entry_has_required_keys(self):
        """Each transition dict contains step, from, to, reward_mean, pass_rate."""
        analyzer = ThreePhaseAnalyzer()
        for i in range(20):
            analyzer.add_step(i, reward=0.03, pass_rate=0.01)
        for i in range(20, 50):
            reward = 0.03 + (i - 20) * 0.006
            analyzer.add_step(i, reward=reward, pass_rate=0.04)
        transitions = analyzer.find_transitions()
        for t in transitions:
            for key in ("step", "from", "to", "reward_mean", "pass_rate", "leading_indicator"):
                assert key in t, f"Missing key '{key}' in transition: {t}"

    def test_find_transitions_phase_values_are_strings(self):
        """'from' and 'to' fields are the string .value of TrainingPhase."""
        analyzer = ThreePhaseAnalyzer()
        for i in range(20):
            analyzer.add_step(i, reward=0.03, pass_rate=0.01)
        for i in range(20, 50):
            reward = 0.03 + (i - 20) * 0.006
            analyzer.add_step(i, reward=reward, pass_rate=0.04)
        for t in analyzer.find_transitions():
            assert isinstance(t["from"], str)
            assert isinstance(t["to"], str)

    def test_find_transitions_insufficient_data_yields_empty(self):
        """With fewer than reward_window steps, annotate returns [] so transitions are []."""
        analyzer = ThreePhaseAnalyzer(reward_window=10)
        for i in range(5):
            analyzer.add_step(i, reward=0.05, pass_rate=0.01)
        assert analyzer.find_transitions() == []


# ---------------------------------------------------------------------------
# ThreePhaseAnalyzer.leading_indicator_report tests
# ---------------------------------------------------------------------------


class TestLeadingIndicatorReport:
    """Tests for ThreePhaseAnalyzer.leading_indicator_report()."""

    def _build_three_phase_analyzer(self) -> ThreePhaseAnalyzer:
        """Build an analyzer with a clear leading-indicator pattern.

        Phase 1: disruption (steps 0-9)
        Phase 2: accumulation - reward climbs first (steps 10-29)
        Phase 3: breakout - pass rate climbs after reward (steps 30-49)
        """
        analyzer = ThreePhaseAnalyzer()
        # Disruption
        for i in range(10):
            analyzer.add_step(i, reward=0.03, pass_rate=0.01)
        # Accumulation: reward climbing, pass rate flat
        for i in range(10, 30):
            reward = 0.03 + (i - 10) * 0.01
            analyzer.add_step(i, reward=reward, pass_rate=0.04)
        # Breakout: pass rate now climbing with high reward
        for i in range(30, 50):
            reward = 0.30 + (i - 30) * 0.01
            pr = 0.04 + (i - 30) * 0.015
            analyzer.add_step(i, reward=reward, pass_rate=pr)
        return analyzer

    def test_leading_indicator_report_sufficient_data_true(self):
        analyzer = self._build_three_phase_analyzer()
        report = analyzer.leading_indicator_report()
        assert report["sufficient_data"] is True

    def test_leading_indicator_report_lead_confirmed_true_when_reward_leads(self):
        """lead_confirmed is True when reward starts climbing before pass rate breakout."""
        analyzer = self._build_three_phase_analyzer()
        report = analyzer.leading_indicator_report()
        assert report["lead_confirmed"] is True

    def test_leading_indicator_report_lead_steps_positive(self):
        """lead_steps > 0 confirms reward climbs before pass rate."""
        analyzer = self._build_three_phase_analyzer()
        report = analyzer.leading_indicator_report()
        assert report["lead_steps"] is not None
        assert report["lead_steps"] > 0

    def test_leading_indicator_report_insufficient_data_returns_flag(self):
        """With fewer than reward_window steps, returns sufficient_data=False."""
        analyzer = ThreePhaseAnalyzer(reward_window=10)
        for i in range(5):
            analyzer.add_step(i, reward=0.05, pass_rate=0.01)
        report = analyzer.leading_indicator_report()
        assert report["sufficient_data"] is False

    def test_leading_indicator_report_contains_expected_keys(self):
        analyzer = self._build_three_phase_analyzer()
        report = analyzer.leading_indicator_report()
        expected_keys = {
            "sufficient_data",
            "reward_climb_step",
            "pass_rate_breakout_step",
            "lead_steps",
            "lead_confirmed",
            "total_steps_analyzed",
        }
        assert expected_keys == set(report.keys())

    def test_leading_indicator_report_reward_climb_step_before_breakout_step(self):
        """reward_climb_step is earlier than pass_rate_breakout_step."""
        analyzer = self._build_three_phase_analyzer()
        report = analyzer.leading_indicator_report()
        assert report["reward_climb_step"] is not None
        assert report["pass_rate_breakout_step"] is not None
        assert report["reward_climb_step"] < report["pass_rate_breakout_step"]

    def test_leading_indicator_report_total_steps_analyzed(self):
        """total_steps_analyzed equals the number of annotations produced."""
        analyzer = self._build_three_phase_analyzer()
        report = analyzer.leading_indicator_report()
        annotations = analyzer.annotate()
        assert report["total_steps_analyzed"] == len(annotations)

    def test_leading_indicator_report_no_breakout_lead_not_confirmed(self):
        """If there is no breakout phase, lead_confirmed is False."""
        analyzer = ThreePhaseAnalyzer()
        # Only disruption data, never reaches breakout
        for i in range(30):
            analyzer.add_step(i, reward=0.03, pass_rate=0.01)
        report = analyzer.leading_indicator_report()
        assert report["lead_confirmed"] is False


# ---------------------------------------------------------------------------
# ThreePhaseAnalyzer.methodology_report tests
# ---------------------------------------------------------------------------


class TestMethodologyReport:
    """Tests for ThreePhaseAnalyzer.methodology_report()."""

    REQUIRED_KEYS = {
        "title",
        "total_steps",
        "annotated_steps",
        "phase_distribution",
        "transitions",
        "leading_indicator",
        "final_pass_rate",
        "final_reward",
        "key_claim",
    }

    def _make_analyzer_with_data(self, n: int = 30) -> ThreePhaseAnalyzer:
        analyzer = ThreePhaseAnalyzer()
        for i in range(n):
            analyzer.add_step(i, reward=0.05, pass_rate=0.02)
        return analyzer

    def test_methodology_report_returns_dict(self):
        analyzer = self._make_analyzer_with_data()
        assert isinstance(analyzer.methodology_report(), dict)

    def test_methodology_report_contains_all_required_keys(self):
        analyzer = self._make_analyzer_with_data()
        report = analyzer.methodology_report()
        assert self.REQUIRED_KEYS == set(report.keys())

    def test_methodology_report_total_steps_matches(self):
        n = 25
        analyzer = self._make_analyzer_with_data(n)
        report = analyzer.methodology_report()
        assert report["total_steps"] == n

    def test_methodology_report_annotated_steps_consistent(self):
        """annotated_steps equals len(annotate())."""
        analyzer = self._make_analyzer_with_data(30)
        report = analyzer.methodology_report()
        assert report["annotated_steps"] == len(analyzer.annotate())

    def test_methodology_report_final_pass_rate_matches_last_observation(self):
        analyzer = self._make_analyzer_with_data(20)
        report = analyzer.methodology_report()
        assert report["final_pass_rate"] == pytest.approx(analyzer.pass_rates[-1])

    def test_methodology_report_final_reward_matches_last_observation(self):
        analyzer = self._make_analyzer_with_data(20)
        report = analyzer.methodology_report()
        assert report["final_reward"] == pytest.approx(analyzer.rewards[-1])

    def test_methodology_report_phase_distribution_is_dict(self):
        analyzer = self._make_analyzer_with_data(25)
        report = analyzer.methodology_report()
        assert isinstance(report["phase_distribution"], dict)

    def test_methodology_report_phase_distribution_sums_to_annotated_steps(self):
        """Phase distribution counts should sum to annotated_steps."""
        analyzer = self._make_analyzer_with_data(30)
        report = analyzer.methodology_report()
        total = sum(report["phase_distribution"].values())
        assert total == report["annotated_steps"]

    def test_methodology_report_transitions_is_list(self):
        analyzer = self._make_analyzer_with_data()
        report = analyzer.methodology_report()
        assert isinstance(report["transitions"], list)

    def test_methodology_report_leading_indicator_is_dict(self):
        analyzer = self._make_analyzer_with_data()
        report = analyzer.methodology_report()
        assert isinstance(report["leading_indicator"], dict)

    def test_methodology_report_key_claim_is_string(self):
        analyzer = self._make_analyzer_with_data()
        report = analyzer.methodology_report()
        assert isinstance(report["key_claim"], str)
        assert len(report["key_claim"]) > 0

    def test_methodology_report_title_is_string(self):
        analyzer = self._make_analyzer_with_data()
        report = analyzer.methodology_report()
        assert isinstance(report["title"], str)

    def test_methodology_report_empty_analyzer_returns_none_finals(self):
        """With no data, final_pass_rate and final_reward are None."""
        analyzer = ThreePhaseAnalyzer()
        report = analyzer.methodology_report()
        assert report["final_pass_rate"] is None
        assert report["final_reward"] is None
        assert report["total_steps"] == 0
