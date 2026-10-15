"""Phase 1.1 — time-ordered train/validation/test trace splits.

The legacy env reads ``trajectories[:, slot_index % full_length]`` which
conflates train and test data: the wrap-around means a test episode starting
at slot 0 will, after one full pass, replay training-era positions. This
module replaces that with strict 60/20/20 time-ordered splits separated by
guard bands sized to the longest feedback delay plus broadcast period, so no
slot used at evaluation time has its exogenous state (user positions, sampled
requests) revealed during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class TraceSplit:
    dataset: str
    split: SplitName
    start_slot: int
    end_slot: int       # exclusive
    n_slots: int
    guard_before: int
    seed_namespace: str

    def contains(self, slot: int) -> bool:
        return self.start_slot <= slot < self.end_slot

    def __post_init__(self) -> None:
        if self.start_slot < 0:
            raise ValueError(f"start_slot must be >= 0, got {self.start_slot}")
        if self.end_slot <= self.start_slot:
            raise ValueError(
                f"end_slot ({self.end_slot}) must exceed start_slot ({self.start_slot})"
            )
        if self.n_slots != self.end_slot - self.start_slot:
            raise ValueError(
                f"n_slots ({self.n_slots}) != end_slot - start_slot "
                f"({self.end_slot - self.start_slot})"
            )


@dataclass(frozen=True)
class SplitPlan:
    """All three splits for a given trace length and delay/broadcast settings."""

    dataset: str
    n_total_slots: int
    guard_before: int
    train: TraceSplit
    validation: TraceSplit
    test: TraceSplit

    def by_name(self, name: SplitName) -> TraceSplit:
        if name == "train":
            return self.train
        if name == "validation":
            return self.validation
        if name == "test":
            return self.test
        raise ValueError(f"unknown split name: {name}")


def compute_splits(
    n_total_slots: int,
    *,
    dataset: str,
    broadcast_period: int,
    feedback_delay_max: int,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
) -> SplitPlan:
    """60/20/20 time-ordered split with guards >= feedback_delay_max + broadcast_period.

    Fractions are normalized. Splits are kept in original time order
    (train → validation → test) with guard bands in between so that no slot
    used for test evaluation has its state leaked into training. If the total
    trace length is too short to accommodate the three splits plus guards,
    ``ValueError`` is raised.
    """
    if n_total_slots <= 0:
        raise ValueError(f"n_total_slots must be > 0, got {n_total_slots}")
    if broadcast_period < 1:
        raise ValueError(f"broadcast_period must be >= 1, got {broadcast_period}")
    if feedback_delay_max < 0:
        raise ValueError(f"feedback_delay_max must be >= 0, got {feedback_delay_max}")
    total_fraction = train_fraction + validation_fraction + test_fraction
    if total_fraction <= 0:
        raise ValueError(f"split fractions must sum > 0, got {total_fraction}")
    train_frac = train_fraction / total_fraction
    val_frac = validation_fraction / total_fraction
    test_frac = test_fraction / total_fraction

    guard = int(feedback_delay_max + broadcast_period)
    train_size = int(train_frac * n_total_slots)
    val_size = int(val_frac * n_total_slots)
    test_size = int(test_frac * n_total_slots)
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError(
            f"Trace too short ({n_total_slots} slots) for the requested splits: "
            f"train={train_size}, val={val_size}, test={test_size}."
        )

    train_start = 0
    train_end = train_start + train_size
    val_start = train_end + guard
    val_end = val_start + val_size
    test_start = val_end + guard
    test_end = test_start + test_size
    if test_end > n_total_slots:
        # Shrink test, then validation, never wrap.
        overshoot = test_end - n_total_slots
        if overshoot < test_size:
            test_size -= overshoot
            test_end = test_start + test_size
        else:
            raise ValueError(
                f"Trace too short to fit guards + splits: need {test_end} slots, "
                f"have {n_total_slots}."
            )

    return SplitPlan(
        dataset=dataset,
        n_total_slots=n_total_slots,
        guard_before=guard,
        train=TraceSplit(
            dataset=dataset,
            split="train",
            start_slot=train_start,
            end_slot=train_end,
            n_slots=train_size,
            guard_before=0,
            seed_namespace=f"{dataset}/train",
        ),
        validation=TraceSplit(
            dataset=dataset,
            split="validation",
            start_slot=val_start,
            end_slot=val_end,
            n_slots=val_size,
            guard_before=guard,
            seed_namespace=f"{dataset}/validation",
        ),
        test=TraceSplit(
            dataset=dataset,
            split="test",
            start_slot=test_start,
            end_slot=test_end,
            n_slots=test_size,
            guard_before=guard,
            seed_namespace=f"{dataset}/test",
        ),
    )


def slice_trajectories(trajectories: np.ndarray, split: TraceSplit) -> np.ndarray:
    """Return ``trajectories[:, split.start_slot:split.end_slot, :]``.

    Validates shapes; never wraps.
    """
    if trajectories.ndim != 3 or trajectories.shape[2] < 2:
        raise ValueError(
            f"trajectories must have shape (users, slots, >=2), got {trajectories.shape}"
        )
    if split.end_slot > trajectories.shape[1]:
        raise ValueError(
            f"split end_slot {split.end_slot} exceeds trajectory length "
            f"{trajectories.shape[1]}"
        )
    return trajectories[:, split.start_slot:split.end_slot, :]


def episode_length(plan: SplitPlan, split: SplitName) -> int:
    """Number of slots the env should run for a single episode in this split."""
    return plan.by_name(split).n_slots


__all__ = [
    "SplitName",
    "TraceSplit",
    "SplitPlan",
    "compute_splits",
    "slice_trajectories",
    "episode_length",
]
