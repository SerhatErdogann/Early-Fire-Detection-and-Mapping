"""
Gerçek zamanlı drone / akış görüntüsü için seçici model çıkarımı.

- Kalp atımı: en fazla ``max_infer_gap_sec`` çıkarımsız kalmaz.
- Sert zorlamalar yalnızca sahne/parlaklık/sıcaklık sıçraması ve güçlü hareket
  (heartbeat bu grupta kalır).
- Alarm / yükselen olasılık: her dekode edilen karede model çalıştırmaz; örnekleme
  hızını 5–10 Hz bandına çıkarır.
- Grafit korelasyonu (SSIM-benzeri hafif sinyal) benzer-atlama filtresinde kullanılır.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _norm_hist_counts(h: np.ndarray) -> np.ndarray:
    x = np.asarray(h, dtype=np.float64).ravel().copy()
    s = float(x.sum())
    if s <= 0:
        x[:] = 1.0 / max(1, x.size)
    else:
        x /= s
    return x


def _gray_hist_vec(gray_01: np.ndarray, bins: int = 24) -> np.ndarray:
    g = np.clip(np.asarray(gray_01, dtype=np.float32).ravel(), 0.0, 1.0)
    h, _ = np.histogram(g, bins=bins, range=(0.0, 1.0), density=False)
    return _norm_hist_counts(h)


@dataclass
class RTSampleDecision:
    run_model: bool
    inferred: bool
    skipped_similar: bool
    skipped_budget: bool
    reason: str


@dataclass
class RTStreamCounters:
    decoded: int = 0
    inferred: int = 0
    skipped_similar: int = 0
    skipped_budget: int = 0


@dataclass
class RTStreamRuntime:
    mono_last_infer: float | None = None
    gray_hist_prev: np.ndarray | None = None
    mean_gray_prev: float | None = None
    hot_frac_prev: float | None = None
    boost_until_mono: float = -1e30
    counters: RTStreamCounters = field(default_factory=RTStreamCounters)


class DroneRTStreamPolicy:
    """Durum saklayan seçici çıkarım kararı (frame başına tek çağrı)."""

    def __init__(
        self,
        *,
        target_infer_hz: float = 1.0,
        max_infer_gap_sec: float = 1.0,
        hist_bins: int = 24,
        similarity_hist_l1_max: float = 0.10,
        similarity_pix_mae_max: float = 0.022,
        similarity_bright_jump_max: float = 0.028,
        similarity_gray_corr_min: float = 0.992,
        motion_spike_mae_mult: float = 1.45,
        motion_spike_mae_floor: float = 0.055,
        motion_soft_mae_mult: float = 1.15,
        bright_spike_abs: float = 0.05,
        hot_spike_abs: float = 0.035,
        review_prob_frac_of_thr: float = 0.42,
        elevated_prob_frac_of_thr: float = 0.34,
        boost_suspected_sec: float = 2.2,
        boost_confirmed_sec: float = 4.0,
        boost_min_interval_sec: float = 0.12,
        suspected_min_interval_sec: float = 0.35,
        normal_min_hz_cap: float = 4.0,
        elevated_infer_hz: float = 5.0,
        alarm_infer_hz: float = 10.0,
        alarm_similarity_tighten: float = 0.72,
    ):
        self.target_infer_hz = max(0.2, float(target_infer_hz))
        self.max_infer_gap_sec = max(0.2, float(max_infer_gap_sec))
        self.hist_bins = max(8, int(hist_bins))
        self.similarity_hist_l1_max = float(similarity_hist_l1_max)
        self.similarity_pix_mae_max = float(similarity_pix_mae_max)
        self.similarity_bright_jump_max = float(similarity_bright_jump_max)
        self.similarity_gray_corr_min = float(similarity_gray_corr_min)
        self.motion_spike_mae_mult = float(motion_spike_mae_mult)
        self.motion_spike_mae_floor = float(motion_spike_mae_floor)
        self.motion_soft_mae_mult = float(motion_soft_mae_mult)
        self.bright_spike_abs = float(bright_spike_abs)
        self.hot_spike_abs = float(hot_spike_abs)
        self.review_prob_frac_of_thr = float(review_prob_frac_of_thr)
        self.elevated_prob_frac_of_thr = float(elevated_prob_frac_of_thr)
        self.boost_suspected_sec = float(boost_suspected_sec)
        self.boost_confirmed_sec = float(boost_confirmed_sec)
        self.boost_min_interval_sec = float(boost_min_interval_sec)
        self.suspected_min_interval_sec = float(suspected_min_interval_sec)
        self.normal_min_hz_cap = float(normal_min_hz_cap)
        self.elevated_infer_hz = max(1.0, float(elevated_infer_hz))
        self.alarm_infer_hz = max(self.elevated_infer_hz, float(alarm_infer_hz))
        self.alarm_similarity_tighten = float(np.clip(alarm_similarity_tighten, 0.3, 1.0))
        self.rt = RTStreamRuntime()

    def _min_interval_seconds(
        self,
        *,
        mono_now: float,
        fps: float,
        internal_alarm: str,
        last_smoothed_prob: float | None,
        operating_thr_proxy: float,
        pix_mae_vs_prev_gray: float,
        mae_motion_baseline_hint: float,
    ) -> float:
        """Hedef örnekleme aralığı (saniye)."""
        if mono_now < float(self.rt.boost_until_mono):
            return max(1e-3, 1.0 / self.alarm_infer_hz)

        ial = str(internal_alarm or "").lower().strip()
        if ial == "confirmed":
            return max(1e-3, 1.0 / self.alarm_infer_hz)
        if ial == "suspected":
            return max(1e-3, float(self.suspected_min_interval_sec))

        lt = float(operating_thr_proxy or 0.5)
        review_hi = float(self.review_prob_frac_of_thr) * lt
        review_lo = float(self.elevated_prob_frac_of_thr) * lt
        lsp = last_smoothed_prob
        if lsp is not None and float(lsp) >= review_hi:
            return max(1e-3, 1.0 / self.alarm_infer_hz)
        if lsp is not None and float(lsp) >= review_lo:
            return max(1e-3, 1.0 / self.elevated_infer_hz)

        spike_base = float(max(float(mae_motion_baseline_hint), self.motion_spike_mae_floor))
        spike_hi = float(self.motion_spike_mae_mult) * spike_base
        spike_lo = float(self.motion_soft_mae_mult) * spike_base
        if spike_lo <= float(pix_mae_vs_prev_gray) < spike_hi:
            return max(1e-3, 1.0 / max(3.0, self.target_infer_hz * 3.0))

        # Normal — yaklaşık target_infer_hz; FPS uyumu
        base = 1.0 / self.target_infer_hz
        fps_eff = float(fps or 25.0)
        cap_hz = float(self.normal_min_hz_cap)
        floor_sec = max(1.0 / cap_hz, base)
        hi = np.clip(base * (25.0 / max(fps_eff, 1.0)), base, floor_sec * 4.0)
        return float(hi)

    def pulse_alarm_burst(self, mono_now: float, old_state: str | None, new_state: str | None) -> None:
        """Suspected / confirmed’a geçişte sık çıkarım penceresi."""
        nw = str(new_state or "").lower().strip()
        ow = str(old_state or "").lower().strip()
        mono_now = float(mono_now)
        if nw == "confirmed" and ow != "confirmed":
            until = mono_now + float(self.boost_confirmed_sec)
            self.rt.boost_until_mono = max(float(self.rt.boost_until_mono), until)
            return
        if nw == "suspected" and ow not in ("suspected", "confirmed"):
            until = mono_now + float(self.boost_suspected_sec)
            self.rt.boost_until_mono = max(float(self.rt.boost_until_mono), until)

    def record_infer_started(self, mono_now: float) -> None:
        self.rt.mono_last_infer = float(mono_now)

    def finalize_frame_observer(
        self,
        *,
        gray_01: np.ndarray,
        mean_gray: float,
        hot_frac: float,
        hist_vec: np.ndarray | None,
    ) -> None:
        if hist_vec is None:
            hist_vec = _gray_hist_vec(gray_01, bins=self.hist_bins)
        self.rt.gray_hist_prev = np.asarray(hist_vec, dtype=np.float64).copy()
        self.rt.mean_gray_prev = float(mean_gray)
        self.rt.hot_frac_prev = float(hot_frac)

    def similarity_bundle_ok(
        self,
        *,
        hist_curr: np.ndarray,
        pix_mae_vs_prev_gray: float,
        d_brightness: float,
        gray_corr_prev: float | None,
        internal_alarm_prior: str,
    ) -> bool:
        """ Histogram + parlaklık + MAE + (varsa) gri korelasyon — alarmda daha sıkı."""
        ial = str(internal_alarm_prior or "").lower().strip()
        tight = float(self.alarm_similarity_tighten) if ial in ("suspected", "confirmed") else 1.0

        hp = self.rt.gray_hist_prev
        if hp is None:
            return False
        hc = np.asarray(hist_curr, dtype=np.float64).ravel()
        hp = np.asarray(hp, dtype=np.float64).ravel()
        hist_l1 = float(np.abs(hp - hc).sum())
        if hist_l1 > float(self.similarity_hist_l1_max) * tight:
            return False
        if float(pix_mae_vs_prev_gray) > float(self.similarity_pix_mae_max) * tight:
            return False
        if float(abs(d_brightness)) > float(self.similarity_bright_jump_max) * tight:
            return False
        gcc = gray_corr_prev
        if gcc is not None and gcc == gcc:
            if float(gcc) < float(self.similarity_gray_corr_min):
                return False
        return True

    # Back-compat name for tests
    def similarity_triple_ok(
        self,
        *,
        hist_curr: np.ndarray,
        pix_mae_vs_prev_gray: float,
        d_brightness: float,
    ) -> bool:
        return self.similarity_bundle_ok(
            hist_curr=hist_curr,
            pix_mae_vs_prev_gray=pix_mae_vs_prev_gray,
            d_brightness=d_brightness,
            gray_corr_prev=None,
            internal_alarm_prior="idle",
        )

    def decide(
        self,
        *,
        mono_now: float,
        fps: float,
        internal_alarm_prior: str,
        scene_changed: bool,
        pix_mae_vs_prev_gray: float,
        mae_motion_baseline_hint: float,
        hist_curr: np.ndarray,
        mean_gray: float,
        hot_frac: float,
        last_smoothed_prob: float | None,
        last_raw_prob: float | None,
        operating_thr_proxy: float,
        cam_hotspot_delta: float | None,
        gray_corr_prev: float | None = None,
    ) -> RTSampleDecision:
        counters = self.rt.counters
        counters.decoded += 1
        prev_mean = self.rt.mean_gray_prev
        dbrightness = (
            abs(float(mean_gray) - float(prev_mean)) if prev_mean is not None else float("nan")
        )
        hp = self.rt.hot_frac_prev
        dhfrac = float(hot_frac) - float(hp) if hp is not None else float("nan")

        tail = internal_alarm_prior or "idle"

        heartbeat = self.rt.mono_last_infer is None or (
            mono_now - float(self.rt.mono_last_infer) >= self.max_infer_gap_sec
        )

        spike_base = float(max(float(mae_motion_baseline_hint), self.motion_spike_mae_floor))
        motion_hard = float(pix_mae_vs_prev_gray) >= float(self.motion_spike_mae_mult) * spike_base

        hard_force = heartbeat
        rs: list[str] = []
        if heartbeat:
            rs.append("heartbeat")
        if scene_changed:
            hard_force = True
            rs.append("scene")
        if motion_hard:
            hard_force = True
            rs.append("motion_spike")
        if not np.isnan(dbrightness) and dbrightness >= self.bright_spike_abs:
            hard_force = True
            rs.append("bright_spike")
        if cam_hotspot_delta is not None and float(cam_hotspot_delta) >= self.hot_spike_abs:
            hard_force = True
            rs.append("cam_hot_grow")
        if not np.isnan(dhfrac) and dhfrac >= self.hot_spike_abs:
            hard_force = True
            rs.append("gray_hot_grow")

        if hard_force:
            self.record_infer_started(mono_now)
            counters.inferred += 1
            joined = "|".join(rs) if rs else "heartbeat"
            return RTSampleDecision(True, True, False, False, joined)

        min_int = self._min_interval_seconds(
            mono_now=mono_now,
            fps=fps,
            internal_alarm=tail,
            last_smoothed_prob=last_smoothed_prob,
            operating_thr_proxy=float(operating_thr_proxy),
            pix_mae_vs_prev_gray=float(pix_mae_vs_prev_gray),
            mae_motion_baseline_hint=float(mae_motion_baseline_hint),
        )
        elapsed = (
            mono_now - float(self.rt.mono_last_infer)
            if self.rt.mono_last_infer is not None
            else min_int + 1.0
        )
        if elapsed + 1e-6 < min_int:
            counters.skipped_budget += 1
            return RTSampleDecision(False, False, False, True, f"budget_hz>{min_int:.3f}s")

        _ = last_raw_prob
        if self.similarity_bundle_ok(
            hist_curr=hist_curr,
            pix_mae_vs_prev_gray=float(pix_mae_vs_prev_gray),
            d_brightness=dbrightness if prev_mean is not None else float("inf"),
            gray_corr_prev=gray_corr_prev,
            internal_alarm_prior=tail,
        ):
            counters.skipped_similar += 1
            return RTSampleDecision(False, False, True, False, "similar")

        self.record_infer_started(mono_now)
        counters.inferred += 1
        return RTSampleDecision(True, True, False, False, "sample")


__all__ = [
    "DroneRTStreamPolicy",
    "RTSampleDecision",
    "RTStreamCounters",
    "_gray_hist_vec",
]
