"""Expanding ring curriculum for preventing saturation.

When the controller detects saturation (App 8), the expanding ring
activates: harder theorems/tasks from the next difficulty tier are
added to the training set, keeping theorems in the frontier zone.

The ring expands based on pass rate: as the model masters the current
set, harder problems are included. This prevents gradient saturation
without manual intervention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic

T = TypeVar("T")


@dataclass
class RingConfig:
    """Configuration for the expanding ring curriculum."""

    tiers: int = 4
    """Number of difficulty tiers."""

    expansion_threshold: float = 0.7
    """Pass rate above this triggers ring expansion."""

    contraction_threshold: float = 0.1
    """Pass rate below this at current tier triggers contraction."""

    min_tier_steps: int = 20
    """Minimum steps at a tier before expansion."""

    warmup_fraction: float = 0.3
    """Fraction of new-tier tasks to mix in during expansion (ramp up)."""


@dataclass
class ExpandingRing(Generic[T]):
    """Expanding ring curriculum manager.

    Manages a tiered set of items (theorems, code tasks) and controls
    which tiers are active based on model performance.

    Type parameter T: the item type (theorem statement, CodeTask, etc.)
    """

    config: RingConfig = field(default_factory=RingConfig)

    tiers: dict[int, list[T]] = field(default_factory=dict)
    """tier_number -> list of items at that tier."""

    active_tier: int = 0
    """Current highest active tier."""

    tier_pass_rates: dict[int, list[float]] = field(default_factory=dict)
    """Rolling pass rates per tier."""

    tier_steps: dict[int, int] = field(default_factory=lambda: {0: 0})
    """Steps spent at each tier."""

    expansion_history: list[dict] = field(default_factory=list)

    def add_tier(self, tier: int, items: list[T]) -> None:
        """Register items for a difficulty tier."""
        self.tiers[tier] = items

    def get_active_items(self) -> list[T]:
        """Get all items from active tiers."""
        items = []
        for tier in range(self.active_tier + 1):
            items.extend(self.tiers.get(tier, []))
        return items

    def get_frontier_items(self) -> list[T]:
        """Get items from the current frontier tier only.

        These are the newest/hardest active items — where gradient
        signal is richest because the model hasn't mastered them yet.
        """
        return list(self.tiers.get(self.active_tier, []))

    def record_pass_rate(self, tier: int, pass_rate: float) -> None:
        """Record pass rate for a tier."""
        if tier not in self.tier_pass_rates:
            self.tier_pass_rates[tier] = []
        self.tier_pass_rates[tier].append(pass_rate)
        self.tier_steps[tier] = self.tier_steps.get(tier, 0) + 1

    def should_expand(self) -> bool:
        """Check if the ring should expand to include the next tier."""
        rates = self.tier_pass_rates.get(self.active_tier, [])
        steps = self.tier_steps.get(self.active_tier, 0)

        if steps < self.config.min_tier_steps:
            return False

        if len(rates) < 5:
            return False

        recent_rate = sum(rates[-5:]) / 5
        return recent_rate >= self.config.expansion_threshold

    def should_contract(self) -> bool:
        """Check if the ring should contract (frontier tier too hard)."""
        if self.active_tier == 0:
            return False

        rates = self.tier_pass_rates.get(self.active_tier, [])
        if len(rates) < 10:
            return False

        recent_rate = sum(rates[-10:]) / 10
        return recent_rate <= self.config.contraction_threshold

    def expand(self) -> bool:
        """Expand to next tier. Returns True if expansion occurred."""
        next_tier = self.active_tier + 1
        if next_tier not in self.tiers:
            return False

        self.expansion_history.append({
            "action": "expand",
            "from_tier": self.active_tier,
            "to_tier": next_tier,
            "trigger_pass_rate": self.tier_pass_rates.get(self.active_tier, [0])[-1] if self.tier_pass_rates.get(self.active_tier) else 0,
        })
        self.active_tier = next_tier
        return True

    def contract(self) -> bool:
        """Contract to previous tier. Returns True if contraction occurred."""
        if self.active_tier == 0:
            return False

        self.expansion_history.append({
            "action": "contract",
            "from_tier": self.active_tier,
            "to_tier": self.active_tier - 1,
            "trigger_pass_rate": self.tier_pass_rates.get(self.active_tier, [0])[-1] if self.tier_pass_rates.get(self.active_tier) else 0,
        })
        self.active_tier -= 1
        return True

    def step(self, tier_pass_rates: dict[int, float]) -> str:
        """Process one training step: record rates, auto-expand/contract.

        Returns action taken: "expand", "contract", or "hold".
        """
        for tier, rate in tier_pass_rates.items():
            self.record_pass_rate(tier, rate)

        if self.should_expand():
            if self.expand():
                return "expand"
        elif self.should_contract():
            if self.contract():
                return "contract"

        return "hold"

    def summary(self) -> dict:
        """Curriculum state summary."""
        return {
            "active_tier": self.active_tier,
            "total_tiers": len(self.tiers),
            "items_per_tier": {t: len(items) for t, items in self.tiers.items()},
            "active_items": len(self.get_active_items()),
            "tier_pass_rates": {
                t: round(sum(rates[-5:]) / len(rates[-5:]), 3) if rates else 0.0
                for t, rates in self.tier_pass_rates.items()
            },
            "expansion_history": self.expansion_history,
        }
