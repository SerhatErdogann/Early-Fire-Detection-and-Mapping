"""
Gerçek zamanlı drone / akış görüntüsü için seçici model çıkarımı.

Amaç: varsayılan ~1 Hz model çağrısı, kalp atımı ile en fazla 1 sn çıkarımsız
kalmama, güvenlik zorlamalarında (hareket/sahne yükseliş/alarm süitleri)
sıklaştırma, benzer kare atlama ile CPU/GPU tasarrufu.
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
    last_public_alarm_prior: str = "ok"
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
        motion_spike_mae_mult: float = 1.45,
        motion_spike_mae_floor: float = 0.055,
        bright_spike_abs: float = 0.05,
        hot_spike_abs: float = 0.035,
        review_prob_frac_of_thr: float = 0.42,
        boost_suspected_sec: float = 2.2,
        boost_confirmed_sec: float = 4.0,
        boost_min_interval_sec: float = 0.12,
        suspected_min_interval_sec: float = 0.35,
        normal_min_hz_cap: float = 4.0,
    ):
        self.target_infer_hz = max(0.2, float(target_infer_hz))
        self.max_infer_gap_sec = max(0.2, float(max_infer_gap_sec))
        self.hist_bins = max(8, int(hist_bins))
        self.similarity_hist_l1_max = float(similarity_hist_l1_max)
        self.similarity_pix_mae_max = float(similarity_pix_mae_max)
        self.similarity_bright_jump_max = float(similarity_bright_jump_max)
        self.motion_spike_mae_mult = float(motion_spike_mae_mult)
        self.motion_spike_mae_floor = float(motion_spike_mae_floor)
        self.bright_spike_abs = float(bright_spike_abs)
        self.hot_spike_abs = float(hot_spike_abs)
        self.review_prob_frac_of_thr = float(review_prob_frac_of_thr)
        self.boost_suspected_sec = float(boost_suspected_sec)
        self.boost_confirmed_sec = float(boost_confirmed_sec)
        self.boost_min_interval_sec = float(boost_min_interval_sec)
        self.suspected_min_interval_sec = float(suspected_min_interval_sec)
        self.normal_min_hz_cap = float(normal_min_hz_cap)
        self.rt = RTStreamRuntime()

    def _min_interval_seconds(self, *, mono_now: float, fps: float, internal_alarm: str) -> float:
        if mono_now < float(self.rt.boost_until_mono):
            return max(1e-3, float(self.boost_min_interval_sec))
        ial = str(internal_alarm or "").lower().strip()
        if ial == "confirmed":
            return max(1e-3, float(self.boost_min_interval_sec))
        if ial == "suspected":
            return max(1e-3, float(self.suspected_min_interval_sec))
        # Normal: yaklaşık target_infer_hz, FPS çok düşükse biraz sıklaş
        base = 1.0 / self.target_infer_hz
        fps_eff = float(fps or 25.0)
        cap_hz = float(self.normal_min_hz_cap)
        floor_sec = max(1.0 / cap_hz, base)
        hi = np.clip(base * (25.0 / max(fps_eff, 1.0)), base, floor_sec * 4.0)
        return float(hi)

    @staticmethod
    def _alarm_force(internal_alarm: str) -> bool:
        s = (internal_alarm or "").lower().strip()
        return s in ("suspected", "confirmed")

    def pulse_alarm_burst(self, mono_now: float, old_state: str | None, new_state: str | None) -> None:
        """Suspected / confirmed’a geçişte kısa süre sık çıkarım penceresi aç."""
        nw = str(new_state or "").lower().strip()
        ow = str(old_state or "").lower().strip()
        mono_now = float(mono_now)
        if nw == "confirmed" and ow != "confirmed":
            until = mono_now + float(self.boost_confirmed_sec)
            self.rt.boost_until_mono = max(float(self.rt.boost_until_mono), until)
            return
        if nw == "suspected" and ow != "suspected" and ow != "confirmed":
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

    def similarity_triple_ok(
        self,
        *,
        hist_curr: np.ndarray,
        pix_mae_vs_prev_gray: float,
        d_brightness: float,
    ) -> bool:
        hp = self.rt.gray_hist_prev
        if hp is None:
            return False
        hc = np.asarray(hist_curr, dtype=np.float64).ravel()
        hp = np.asarray(hp, dtype=np.float64).ravel()
        hist_l1 = float(np.abs(hp - hc).sum())
        if hist_l1 > self.similarity_hist_l1_max:
            return False
        if float(pix_mae_vs_prev_gray) > self.similarity_pix_mae_max:
            return False
        if float(abs(d_brightness)) > self.similarity_bright_jump_max:
            return False
        return True

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
    ) -> RTSampleDecision:
        """
        internal_alarm_prior: bir önceki kare sonunda alarm makinesinin iç durumu
        (`idle`|`suspected`|`confirmed`).
        """
        counters = self.rt.counters
        counters.decoded += 1

        prev_mean = self.rt.mean_gray_prev
        dbrightness = (
            abs(float(mean_gray) - float(prev_mean)) if prev_mean is not None else float("nan")
        )
        hp = self.rt.hot_frac_prev
        dhfrac = (
            float(hot_frac) - float(hp) if hp is not None else float("nan")
        )

        tail = internal_alarm_prior or "idle"

        heartbeat = False
        if self.rt.mono_last_infer is None:
            heartbeat = True
        elif (mono_now - float(self.rt.mono_last_infer)) >= self.max_infer_gap_sec:
            heartbeat = True

        force = heartbeat
        rs = ""

        # Risk / alarm zorlamaları kalp atımından bağımsız da infer sebebidir — ama heartbeat
        # çok sık gereksiz çift infer üretmez; ilk kare için zaten heartbeat.
        if scene_changed:
            force = True
            rs += "scene|"
        if self._alarm_force(tail):
            force = True
            rs += "alarm|"
        lt = float(operating_thr_proxy or 0.5)
        review_floor = float(self.review_prob_frac_of_thr) * lt
        lsp = last_smoothed_prob
        if lsp is not None and float(lsp) >= review_floor:
            force = True
            rs += "prob_tail|"

        spike_base = float(max(float(mae_motion_baseline_hint), self.motion_spike_mae_floor))
        if float(pix_mae_vs_prev_gray) >= float(self.motion_spike_mae_mult) * spike_base:
            force = True
            rs += "motion_spike|"

        if not np.isnan(dbrightness) and dbrightness >= self.bright_spike_abs:
            force = True
            rs += "bright_spike|"

        if cam_hotspot_delta is not None and float(cam_hotspot_delta) >= self.hot_spike_abs:
            force = True
            rs += "cam_hot_grow|"

        if not np.isnan(dhfrac) and dhfrac >= self.hot_spike_abs:
            force = True
            rs += "gray_hot_grow|"

        if force:
            self.record_infer_started(mono_now)
            counters.inferred += 1
            rs = rs.rstrip("|") or "heartbeat"
            return RTSampleDecision(True, True, False, False, rs)

        # Bütçe (hedef Hz)
        min_int = self._min_interval_seconds(mono_now=mono_now, fps=fps, internal_alarm=tail)
        elapsed = (
            mono_now - float(self.rt.mono_last_infer) if self.rt.mono_last_infer is not None else min_int + 1.0
        )
        if elapsed + 1e-6 < min_int:
            counters.skipped_budget += 1
            return RTSampleDecision(
                False, False, False, True, f"budget_hz>{min_int:.3f}s"
            )

        # Benzerlik (muhafazakâr: üç kontrolün hepsi)
        if self.similarity_triple_ok(
            hist_curr=hist_curr,
            pix_mae_vs_prev_gray=float(pix_mae_vs_prev_gray),
            d_brightness=dbrightness if prev_mean is not None else float("inf"),
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
