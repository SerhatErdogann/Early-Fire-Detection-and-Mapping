"""
Alarm state machine for robust video fire alerting.
"""
from __future__ import annotations

from dataclasses import dataclass


ALARM_IDLE = "idle"
ALARM_SUSPECTED = "suspected"
ALARM_CONFIRMED = "confirmed"
ALARM_COOLDOWN = "cooldown"


@dataclass
class AlarmConfig:
    high_threshold: float = 0.7
    low_threshold: float = 0.4
    suspect_threshold: float = 0.55
    confirm_frames: int = 5
    cooldown_frames: int = 6


class AlarmStateMachine:
    """
    Four-state alarm machine with hysteresis and persistence.
    Designed to reduce single-frame spikes and improve explainability.
    """

    def __init__(self, cfg: AlarmConfig):
        self.cfg = cfg
        self.state = ALARM_IDLE
        self._high_run = 0
        self._cooldown_left = 0
        self._suspect_score = 0.0

    def reset(self) -> None:
        self.state = ALARM_IDLE
        self._high_run = 0
        self._cooldown_left = 0
        self._suspect_score = 0.0

    def update(
        self,
        decision_prob: float,
        top10_intensity: float = 0.0,
        largest_component_area: float = 0.0,
        scene_changed: bool = False,
    ) -> tuple[str, int, float, str]:
        """
        Simplified hysteresis + persistence:
        - suspected: decision_prob >= low_threshold
        - confirmed: decision_prob >= high_threshold for `confirm_frames` consecutive updates
        """
        reasons: list[str] = []
        p = float(decision_prob)

        if scene_changed and self.state != ALARM_CONFIRMED:
            self.reset()
            reasons.append("scene_reset")

        spatial_support = (top10_intensity >= 0.65) or (largest_component_area >= 0.01)
        if spatial_support:
            reasons.append("spatial_support")

        if self.state == ALARM_COOLDOWN:
            if p >= self.cfg.high_threshold:
                self._cooldown_left = 0
            else:
                self._cooldown_left = max(0, self._cooldown_left - 1)
                self.state = ALARM_COOLDOWN if self._cooldown_left > 0 else ALARM_IDLE
                reasons.append("cooldown")
                reason = "|".join(reasons) if reasons else "cooldown"
                return self.state, 0, float(p), reason

        if self.state == ALARM_CONFIRMED and p < self.cfg.low_threshold:
            self._high_run = 0
            self._cooldown_left = max(1, int(self.cfg.cooldown_frames))
            self.state = ALARM_COOLDOWN
            reasons.append("cooldown")
            return self.state, 0, float(p), "|".join(reasons)

        # persistence counter (strict consecutive high)
        if p >= self.cfg.high_threshold:
            self._high_run += 1
            reasons.append("high_probability")
        else:
            self._high_run = 0

        if self._high_run >= int(self.cfg.confirm_frames):
            self.state = ALARM_CONFIRMED
            reasons.append("persistence_confirmed")
        else:
            # not confirmed
            if p >= self.cfg.low_threshold:
                self.state = ALARM_SUSPECTED
                reasons.append("suspected")
            else:
                self.state = ALARM_IDLE

        confidence = p
        event = 1 if self.state == ALARM_CONFIRMED else 0
        reason = "|".join(reasons) if reasons else "no_trigger"
        return self.state, event, float(confidence), reason
