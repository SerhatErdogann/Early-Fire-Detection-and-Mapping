"""
Adaptive frame sampling strategy for video inference throughput.
"""
from __future__ import annotations


class AdaptiveFrameSampler:
    """
    Dynamically adjusts frame step based on motion and alarm risk.
    """

    def __init__(
        self,
        base_step: int = 5,
        min_step: int = 1,
        max_step: int = 12,
        low_motion_threshold: float = 0.03,
        high_risk_threshold: float = 0.65,
    ):
        self.base_step = max(1, int(base_step))
        self.min_step = max(1, int(min_step))
        self.max_step = max(self.min_step, int(max_step))
        self.low_motion_threshold = float(low_motion_threshold)
        self.high_risk_threshold = float(high_risk_threshold)
        self.current_step = self.base_step

    def update(self, motion_mae: float, decision_prob: float, alarm_state: str) -> int:
        risk_high = (decision_prob >= self.high_risk_threshold) or (alarm_state in {"suspected", "confirmed"})
        low_motion = motion_mae <= self.low_motion_threshold

        if risk_high:
            self.current_step = self.min_step
        elif low_motion and decision_prob < 0.25:
            self.current_step = min(self.max_step, max(self.base_step + 1, self.base_step * 2))
        else:
            self.current_step = self.base_step
        return self.current_step

    def skip_count(self) -> int:
        return max(0, int(self.current_step) - 1)
