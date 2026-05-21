"""Inference orchestration for the UI: CSV → risk table → merged operational events."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from src.eval.event_extractor import extract_events
from src.inference.downstream_alarm_feed import alarm_feed_paths_for_csv
from src.inference.model_loader import route_checkpoint_for_video
from src.inference.video import run_video_inference
from src.risk.scoring import build_risk_table

try:
    from config import INFERENCE_DEFAULT, RISK_SCORE_WEIGHTS
except Exception:  # pragma: no cover
    INFERENCE_DEFAULT = {}
    RISK_SCORE_WEIGHTS = {}


def run_analysis_pipeline(
    rgb_path: str,
    th_path: str | None,
    preset_args: dict[str, Any],
    ckpt_path: str,
    out_dir: Path,
    *,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / "video_predictions.csv"
    bench_json = out_dir / "video_predictions.benchmark.json"

    a = preset_args

    ckpt_resolved, _, _, vid_mode = route_checkpoint_for_video(
        str(ckpt_path),
        has_thermal_video=bool(th_path and str(th_path).strip()),
    )

    run_video_inference(
        rgb_video_path=rgb_path,
        th_video_path=th_path,
        ckpt_path=ckpt_resolved or str(ckpt_path),
        mode=vid_mode,
        size=int(a.get("size", 224)),
        step_frames=int(a.get("step", 6)),
        smooth_window=int(a.get("smooth_win", 7)),
        ema_alpha=float(a.get("ema_alpha", 0.30)),
        use_tta=bool(a.get("tta", False)),
        out_csv=str(pred_csv),
        use_fp16=bool(a.get("fp16", False)),
        temporal_guard=bool(a.get("temporal_guard", True)),
        adaptive_step=False,
        auto_step_long_video=False,
        min_component_area=float(a.get("min_component_area", 0.01)),
        texture_prob_max=float(a.get("texture_prob_max", INFERENCE_DEFAULT.get("texture_prob_max", 0.2))),
        small_fire_boost=float(a.get("small_fire_boost", INFERENCE_DEFAULT.get("small_fire_boost", 1.3))),
        growth_upscale=float(a.get("growth_upscale", INFERENCE_DEFAULT.get("growth_upscale", 1.2))),
        benchmark=True,
        benchmark_out=str(bench_json),
        prob_temporal_blend=float(a.get("prob_temporal_blend", 0.0)),
        burst_min_frames=int(a.get("burst_min_frames", 3)),
        burst_threshold_frac=float(a.get("burst_threshold_frac", 1.0)),
        stream_buffer_reduce=bool(a.get("stream_buffer_reduce", True)),
        progress_callback=progress_callback,
        target_infer_hz=float(a.get("target_infer_hz", 1.0)),
        max_infer_gap_sec=float(a.get("max_infer_gap_sec", 1.0)),
    )
    try:
        df_pred = pd.read_csv(pred_csv)
    except Exception as e:
        raise RuntimeError(
            "Analiz çıktısı okunamadı. Dosya biçimi hatalı veya video çözümlenemedi."
        ) from e
    if df_pred.empty:
        raise RuntimeError(
            "Hiç kare işlenemedi. Codec uyumsuzluğu veya bozuk dosya olabilir; MP4 (H.264) deneyin."
        )
    thr_used = (
        float(pd.to_numeric(df_pred.get("threshold_used", 0.5), errors="coerce").dropna().median())
        if len(df_pred)
        else 0.5
    )
    hh = float(pd.to_numeric(df_pred.get("hyst_high_used"), errors="coerce").dropna().median()) if "hyst_high_used" in df_pred.columns else float(thr_used)
    hl = float(pd.to_numeric(df_pred.get("hyst_low_used"), errors="coerce").dropna().median()) if "hyst_low_used" in df_pred.columns else float(thr_used) * 0.6

    scored, _meta = build_risk_table(
        df_pred.sort_values("frame_idx").reset_index(drop=True),
        risk_weights={k: float(v) for k, v in dict(RISK_SCORE_WEIGHTS).items()},
        persistence_win=7,
        persistence_thr=thr_used,
    )
    merge_gap_sec = float(a.get("event_merge_gap_sec", 2.0))
    events_df = extract_events(scored, merge_gap_sec=merge_gap_sec)
    scored_csv = out_dir / "video_predictions_scored.csv"
    events_csv = out_dir / "events.csv"
    event_summary_csv = out_dir / "event_summary.csv"
    mapping_export_json = out_dir / "mapping_export.json"
    scored.to_csv(scored_csv, index=False)
    events_df.to_csv(events_csv, index=False)
    events_df.to_csv(event_summary_csv, index=False)

    mapping_records: list[dict[str, Any]] = []
    if events_df.empty:
        mapping_records.append(
            {
                "fire_detected": False,
                "risk_level": "ok",
                "probability": 0.0,
                "timestamp": None,
            }
        )
    else:
        for _, r in events_df.iterrows():
            rl = str(r.get("risk_level", "ok"))
            mapping_records.append(
                {
                    "fire_detected": rl in ("suspected", "confirmed"),
                    "risk_level": rl,
                    "probability": round(float(r.get("max_prob", 0.0)), 4),
                    "timestamp": round(float(r.get("start_sec", 0.0)), 3),
                    "event_id": str(r.get("event_id", "")),
                    "event_duration": round(float(r.get("duration_sec", 0.0)), 3),
                    "max_probability": round(float(r.get("max_prob", 0.0)), 4),
                }
            )
    with mapping_export_json.open("w", encoding="utf-8") as fmap:
        json.dump(mapping_records, fmap, indent=2, ensure_ascii=False)

    af_csv, af_jsonl, af_schema = alarm_feed_paths_for_csv(pred_csv)
    return {
        "pred_csv": str(pred_csv),
        "scored_csv": str(scored_csv),
        "events_csv": str(events_csv),
        "event_summary_csv": str(event_summary_csv),
        "mapping_export_json": str(mapping_export_json),
        "benchmark_json": str(bench_json),
        "alarm_feed_csv": str(af_csv),
        "alarm_feed_jsonl": str(af_jsonl),
        "alarm_feed_schema": str(af_schema),
        "df_scored": scored,
        "df_events": events_df,
        "threshold_used": thr_used,
        "hyst_high_used": hh,
        "hyst_low_used": hl,
    }
