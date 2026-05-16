"""
Final video inference entrypoint (auto fusion -> rgb -> thermal fallback).

Examples:
  python src/05_video_infer.py --rgb_video path/to/rgb.mp4 --th_video path/to/th.mp4
  python src/05_video_infer.py --rgb_video path/to/rgb.mp4
  python src/05_video_infer.py --th_video path/to/th.mp4
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.video import run_video_inference

try:
    from config import CKPT_FUSION, CKPT_RGB, CKPT_THERMAL, INFERENCE_DEFAULT, OUTPUTS_DIR
except ImportError:
    CKPT_FUSION = Path("models/fusion.pt")
    CKPT_RGB = Path("models/rgb.pt")
    CKPT_THERMAL = Path("models/thermal.pt")
    INFERENCE_DEFAULT = {
        "smooth_window": 7,
        "ema_alpha": 0.30,
        "use_tta": False,
        "step_frames": 12,
        "scene_thresh": 0.10,
        "scene_conf_scale": 0.7,
        "hyst_high": 0.60,
        "hyst_low": 0.40,
        "persist_n": 4,
        "min_component_area": 0.01,
        "growth_downscale": 0.85,
        "kl_hist_thresh": 0.35,
    }
    OUTPUTS_DIR = Path("outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_video", default=None)
    ap.add_argument("--th_video", default=None)
    ap.add_argument("--step", type=int, default=INFERENCE_DEFAULT.get("step_frames", 12))
    ap.add_argument("--smooth_win", type=int, default=INFERENCE_DEFAULT.get("smooth_window", 5))
    ap.add_argument("--ema_alpha", type=float, default=INFERENCE_DEFAULT.get("ema_alpha", 0.3))
    ap.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable test-time augmentation (horizontal flip)",
    )
    ap.add_argument("--override_thr", type=float, default=None)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--model_fusion", default=str(CKPT_FUSION))
    ap.add_argument("--model_rgb", default=str(CKPT_RGB))
    ap.add_argument("--model_thermal", default=str(CKPT_THERMAL))
    ap.add_argument("--out", default=str(OUTPUTS_DIR / "video_predictions.csv"))
    ap.add_argument("--save_heatmaps", action="store_true")
    ap.add_argument("--save_masks", action="store_true", help="Save EMA-smoothed soft mask PNGs")
    ap.add_argument("--save_polygons", action="store_true", help="Save per-frame centroid JSON (mask-based)")
    ap.add_argument(
        "--cam-stats",
        action="store_true",
        help="Run Grad-CAM each frame for area/growth (no PNG); enables spatial filters. Slower than plain infer.",
    )
    ap.add_argument(
        "--fp16",
        action="store_true",
        help="CUDA half-precision forward (faster; no effect without GPU). Skipped when saving heatmaps/masks.",
    )
    ap.add_argument("--mode", choices=["auto", "rgb", "thermal", "fusion"], default="auto")
    idf = INFERENCE_DEFAULT
    ap.add_argument(
        "--no-temporal-guard",
        action="store_true",
        help="Disable scene change / hysteresis / N-frame fire_event (legacy).",
    )
    ap.add_argument("--scene-thresh", type=float, default=idf.get("scene_thresh", 0.10))
    ap.add_argument("--scene-confidence-scale", type=float, default=idf.get("scene_conf_scale", 0.7))
    ap.add_argument("--hyst-high", type=float, default=idf.get("hyst_high", 0.55))
    ap.add_argument("--hyst-low", type=float, default=idf.get("hyst_low", 0.35))
    ap.add_argument("--persist-n", type=int, default=idf.get("persist_n", 2))
    ap.add_argument("--min-area", type=float, default=idf.get("min_component_area", 0.01))
    ap.add_argument("--growth-downscale", type=float, default=idf.get("growth_downscale", 0.85))
    ap.add_argument("--kl-scene", action="store_true", help="Also trigger scene reset on RGB histogram KL jump")
    ap.add_argument("--kl-hist-thresh", type=float, default=idf.get("kl_hist_thresh", 0.35))
    ap.add_argument("--early-detection", action="store_true", default=idf.get("early_detection", False))
    ap.add_argument("--early-thr-shift", type=float, default=idf.get("early_threshold_shift", 0.15))
    ap.add_argument("--early-min-thr", type=float, default=idf.get("early_min_threshold", 0.25))
    ap.add_argument("--early-persist-n", type=int, default=idf.get("early_persist_n", 2))
    ap.add_argument("--small-fire-boost", type=float, default=idf.get("small_fire_boost", 1.3))
    ap.add_argument("--small-fire-area-max", type=float, default=idf.get("small_fire_area_max", 0.02))
    ap.add_argument("--growth-upscale", type=float, default=idf.get("growth_upscale", 1.2))
    ap.add_argument("--texture-prob-max", type=float, default=idf.get("texture_prob_max", 0.2))
    ap.add_argument("--texture-top10-min", type=float, default=idf.get("texture_top10_min", 0.7))
    ap.add_argument("--modal-agreement", action="store_true", default=idf.get("enable_modal_agreement", False))
    ap.add_argument("--modal-min-corr", type=float, default=idf.get("modal_agreement_min_corr", 0.2))
    ap.add_argument("--modal-penalty", type=float, default=idf.get("modal_agreement_penalty", 0.6))
    # adaptive-step can be enabled/disabled explicitly
    ap.add_argument(
        "--adaptive-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable adaptive frame step based on motion/risk",
    )
    ap.add_argument("--adaptive-min-step", type=int, default=idf.get("adaptive_min_step", 1))
    ap.add_argument("--adaptive-max-step", type=int, default=idf.get("adaptive_max_step", 12))
    ap.add_argument("--adaptive-low-motion", type=float, default=idf.get("adaptive_low_motion", 0.03))
    ap.add_argument("--adaptive-high-risk", type=float, default=idf.get("adaptive_high_risk", 0.65))
    ap.add_argument("--benchmark", action="store_true", help="Write performance benchmark JSON")
    ap.add_argument("--benchmark-out", default=None, help="Benchmark JSON output path")
    ap.add_argument(
        "--prob-temporal-blend",
        type=float,
        default=0.0,
        help="Blend EMA with moving-average of last smooth_window probs (0=EMA-only, 1=MA-only).",
    )
    ap.add_argument(
        "--burst-min-frames",
        type=int,
        default=3,
        help="Consecutive frames with MA prob >= frac*thr to set pred_fire_burst_consec.",
    )
    ap.add_argument(
        "--burst-threshold-frac",
        type=float,
        default=1.0,
        help="Burst threshold multiplier vs saved operating threshold.",
    )
    ap.add_argument(
        "--auto-step-long-video",
        action="store_true",
        help="If duration exceeds --long-video-seconds, grow step_frames (capped).",
    )
    ap.add_argument("--long-video-seconds", type=float, default=600.0)
    ap.add_argument("--long-video-step-scale", type=float, default=2.0)
    ap.add_argument("--max-step-cap", type=int, default=64)
    ap.add_argument(
        "--stream-buffer-reduce/--no-stream-buffer-reduce",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For RTSP/http streams, hint OpenCV buffer size=1.",
    )
    ap.add_argument(
        "--infer-batch-size",
        type=int,
        default=1,
        help="Placeholder (only 1 supported; temporal/CAM-safe path).",
    )
    ap.add_argument(
        "--no-alarm-feed-export",
        action="store_true",
        help="Mapping/GIS downstream: alarm_feed CSV+JSONL yazılmasın.",
    )
    ap.add_argument(
        "--target-infer-hz",
        type=float,
        default=1.0,
        help="Seçici çıkarım: hedef yaklaşık model çağrı sıklığı (Hz). Drone/canlı için ~1 önerilir.",
    )
    ap.add_argument(
        "--max-infer-gap-sec",
        type=float,
        default=1.0,
        help="Kalp atımı: güvenlik için bu süreyi geçmeden çıkarımsız kalmayı engelle.",
    )
    args = ap.parse_args()

    if not args.rgb_video and not args.th_video:
        raise SystemExit("Provide at least one of: --rgb_video, --th_video")

    out_csv = run_video_inference(
        rgb_video_path=args.rgb_video,
        th_video_path=args.th_video,
        ckpt_fusion=args.model_fusion,
        ckpt_rgb=args.model_rgb,
        ckpt_thermal=args.model_thermal,
        mode=args.mode,
        size=args.size,
        step_frames=args.step,
        smooth_window=args.smooth_win,
        ema_alpha=args.ema_alpha,
        use_tta=args.tta,
        override_thr=args.override_thr,
        save_heatmaps=args.save_heatmaps,
        save_masks=args.save_masks,
        save_polygons=args.save_polygons,
        out_csv=args.out,
        use_fp16=args.fp16
        and not (args.save_heatmaps or args.save_masks or args.save_polygons or args.cam_stats),
        cam_stats_only=args.cam_stats,
        temporal_guard=not args.no_temporal_guard,
        scene_thresh=args.scene_thresh,
        scene_conf_scale=args.scene_confidence_scale,
        hyst_high=args.hyst_high,
        hyst_low=args.hyst_low,
        persist_n=args.persist_n,
        min_component_area=args.min_area,
        growth_downscale=args.growth_downscale,
        use_kl_scene=args.kl_scene,
        kl_hist_thresh=args.kl_hist_thresh,
        early_detection=args.early_detection,
        early_threshold_shift=args.early_thr_shift,
        early_min_threshold=args.early_min_thr,
        early_persist_n=args.early_persist_n,
        small_fire_boost=args.small_fire_boost,
        small_fire_area_max=args.small_fire_area_max,
        growth_upscale=args.growth_upscale,
        texture_prob_max=args.texture_prob_max,
        texture_top10_min=args.texture_top10_min,
        enable_modal_agreement=args.modal_agreement,
        modal_agreement_min_corr=args.modal_min_corr,
        modal_agreement_penalty=args.modal_penalty,
        adaptive_step=args.adaptive_step,
        adaptive_min_step=args.adaptive_min_step,
        adaptive_max_step=args.adaptive_max_step,
        adaptive_low_motion=args.adaptive_low_motion,
        adaptive_high_risk=args.adaptive_high_risk,
        benchmark=args.benchmark,
        benchmark_out=args.benchmark_out,
        prob_temporal_blend=float(args.prob_temporal_blend),
        burst_min_frames=int(args.burst_min_frames),
        burst_threshold_frac=float(args.burst_threshold_frac),
        auto_step_long_video=bool(args.auto_step_long_video),
        long_video_seconds=float(args.long_video_seconds),
        long_video_step_scale=float(args.long_video_step_scale),
        max_step_cap=int(args.max_step_cap),
        stream_buffer_reduce=bool(args.stream_buffer_reduce),
        infer_batch_size=int(args.infer_batch_size),
        export_alarm_feed=not bool(args.no_alarm_feed_export),
        target_infer_hz=float(args.target_infer_hz),
        max_infer_gap_sec=float(args.max_infer_gap_sec),
    )
    print("Output written:", out_csv)
    if not args.no_alarm_feed_export:
        from src.inference.downstream_alarm_feed import alarm_feed_paths_for_csv

        c, jl, scm = alarm_feed_paths_for_csv(Path(out_csv))
        print("Alarm feed (mapping):", c)
        print("Alarm feed JSONL:", jl)
        print("Alarm feed schema:", scm)


if __name__ == "__main__":
    main()
