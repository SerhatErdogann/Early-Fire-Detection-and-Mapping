from __future__ import annotations

import time

import numpy as np

import pytest

from src.inference.rt_stream_policy import DroneRTStreamPolicy, _gray_hist_vec


@pytest.fixture()
def gray_pair():
    rng = np.random.RandomState(0)
    base = rng.rand(160, 120).astype(np.float32) * 0.4 + 0.2
    near = np.clip(base + rng.randn(160, 120).astype(np.float32) * 0.008, 0.0, 1.0)
    far = np.clip(base + 0.55, 0.0, 1.0)
    return near, far


def test_first_frame_forces_infer():
    pol = DroneRTStreamPolicy(target_infer_hz=1.0)
    mono = time.monotonic()
    g = np.full((160, 120), 0.5, dtype=np.float32)
    h = _gray_hist_vec(g)
    d = pol.decide(
        mono_now=mono,
        fps=30.0,
        internal_alarm_prior="idle",
        scene_changed=False,
        pix_mae_vs_prev_gray=0.08,
        mae_motion_baseline_hint=0.10,
        hist_curr=h,
        mean_gray=0.5,
        hot_frac=0.03,
        last_smoothed_prob=None,
        last_raw_prob=None,
        operating_thr_proxy=0.5,
        cam_hotspot_delta=None,
    )
    assert d.run_model and d.reason == "heartbeat"


def test_similar_skip_when_budget_elapsed(gray_pair):
    near, _far = gray_pair
    pol = DroneRTStreamPolicy(target_infer_hz=12.0, max_infer_gap_sec=5.0)
    mono = time.monotonic()
    h = _gray_hist_vec(near)
    assert (
        pol.decide(
            mono_now=mono,
            fps=30.0,
            internal_alarm_prior="idle",
            scene_changed=False,
            pix_mae_vs_prev_gray=0.5,
            mae_motion_baseline_hint=0.10,
            hist_curr=h,
            mean_gray=float(near.mean()),
            hot_frac=float((near > 0.85).mean()),
            last_smoothed_prob=None,
            last_raw_prob=None,
            operating_thr_proxy=0.5,
            cam_hotspot_delta=None,
        ).run_model
    )
    pol.finalize_frame_observer(
        gray_01=near,
        mean_gray=float(near.mean()),
        hot_frac=float((near > 0.85).mean()),
        hist_vec=h,
    )

    mono2 = mono + 0.25  # hz=12 ⇒ ~0.08s nominal aralıktan daha uzun bekledik (sabit aralıkta da geçecek kadar)
    near_next = np.clip(near + np.float32(-0.0005), 0.0, 1.0)
    pix_mae = float(np.mean(np.abs(near_next.astype(np.float32) - near.astype(np.float32))))
    hn = _gray_hist_vec(near_next)
    d = pol.decide(
        mono_now=mono2,
        fps=30.0,
        internal_alarm_prior="idle",
        scene_changed=False,
        pix_mae_vs_prev_gray=pix_mae,
        mae_motion_baseline_hint=0.10,
        hist_curr=hn,
        mean_gray=float(near_next.mean()),
        hot_frac=float((near_next > 0.85).mean()),
        last_smoothed_prob=None,
        last_raw_prob=None,
        operating_thr_proxy=0.5,
        cam_hotspot_delta=None,
    )
    assert d.skipped_similar


def test_scene_change_always_runs(monkeypatch, gray_pair):
    near, _far = gray_pair
    pol = DroneRTStreamPolicy(target_infer_hz=0.000001)
    pol.rt.mono_last_infer = time.monotonic()
    monkeypatch.setattr(pol, "max_infer_gap_sec", 1e9)
    h = _gray_hist_vec(near)
    assert pol.decide(
        mono_now=time.monotonic(),
        fps=30.0,
        internal_alarm_prior="idle",
        scene_changed=True,
        pix_mae_vs_prev_gray=0.001,
        mae_motion_baseline_hint=0.10,
        hist_curr=h,
        mean_gray=0.3,
        hot_frac=0.01,
        last_smoothed_prob=None,
        last_raw_prob=None,
        operating_thr_proxy=0.5,
        cam_hotspot_delta=None,
    ).run_model
