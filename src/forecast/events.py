"""Dated cost events that adjust a baseline forecast.

Each event has a shape (step, ramp, pulse, multiplier, cliff), optional
confidence weighting, and metadata for the dashboard ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional


class Effect(str, Enum):
    STEP = "step"
    RAMP = "ramp"
    PULSE = "pulse"
    MULTIPLIER = "multiplier"
    CLIFF = "cliff"


CATEGORIES = {
    "demand": "Customer / demand",
    "release": "Release or migration",
    "optimization": "Cost action",
    "commitment": "Commitment / discount",
    "pricing": "Pricing / credits",
}


@dataclass
class CostEvent:
    """A dated thing that changes future cost.

    ``amount_daily`` is the $/day effect at full strength; negative means
    savings. For MULTIPLIER events, ``multiplier_pct`` applies to the
    baseline instead.
    """
    name: str
    start_date: date
    effect: Effect
    category: str = "demand"
    amount_daily: float = 0.0
    end_date: Optional[date] = None
    ramp_days: int = 0
    multiplier_pct: float = 0.0
    confidence: float = 80.0
    enabled: bool = True
    source: str = "manual"
    external_id: str = ""
    note: str = ""

    def strength_on(self, day: date) -> float:
        """Fraction of full effect active on ``day``, 0.0–1.0."""
        if not self.enabled or day < self.start_date:
            return 0.0
        if self.effect is Effect.PULSE:
            if self.end_date and day > self.end_date:
                return 0.0
            return 1.0
        if self.effect is Effect.RAMP and self.ramp_days > 0:
            days_in = (day - self.start_date).days
            return min(1.0, (days_in + 1) / float(self.ramp_days))
        return 1.0

    def additive_on(self, day: date) -> float:
        if self.effect is Effect.MULTIPLIER:
            return 0.0
        return self.amount_daily * self.strength_on(day)

    def multiplier_on(self, day: date) -> float:
        if self.effect is not Effect.MULTIPLIER:
            return 1.0
        return 1.0 + (self.multiplier_pct / 100.0) * self.strength_on(day)

    @property
    def is_saving(self) -> bool:
        if self.effect is Effect.MULTIPLIER:
            return self.multiplier_pct < 0
        return self.amount_daily < 0

    @property
    def shape_label(self) -> str:
        if self.effect is Effect.RAMP:
            return f"Ramp · {self.ramp_days} days"
        if self.effect is Effect.PULSE and self.end_date:
            return (
                f"Pulse · {self.start_date:%d %b} – {self.end_date:%d %b}"
            )
        if self.effect is Effect.MULTIPLIER:
            return f"Multiplier · {self.multiplier_pct:+.0f}%"
        if self.effect is Effect.CLIFF:
            return f"Cliff · {self.start_date:%d %b}"
        return f"Step · {self.start_date:%d %b}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start_date": self.start_date.isoformat(),
            "effect": self.effect.value,
            "category": self.category,
            "amount_daily": self.amount_daily,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "ramp_days": self.ramp_days,
            "multiplier_pct": self.multiplier_pct,
            "confidence": self.confidence,
            "enabled": self.enabled,
            "source": self.source,
            "external_id": self.external_id,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CostEvent:
        end_raw = data.get("end_date")
        effect_raw = data.get("effect", Effect.STEP.value)
        return cls(
            name=data["name"],
            start_date=date.fromisoformat(data["start_date"]),
            effect=Effect(effect_raw),
            category=data.get("category", "demand"),
            amount_daily=float(data.get("amount_daily", 0.0)),
            end_date=date.fromisoformat(end_raw) if end_raw else None,
            ramp_days=int(data.get("ramp_days", 0)),
            multiplier_pct=float(data.get("multiplier_pct", 0.0)),
            confidence=float(data.get("confidence", 80.0)),
            enabled=bool(data.get("enabled", True)),
            source=data.get("source", "manual"),
            external_id=data.get("external_id", ""),
            note=data.get("note", ""),
        )
