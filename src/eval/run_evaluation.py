"""
Run full video evaluation pipeline on a folder of videos.

Pipeline per video:
1) video inference
2) risk scoring
3) event extraction
4) event-level metric aggregation

Usage:
    python src/eval/run_evaluation.py --videos_dir data/flame3/videos --profile balanced
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.event_metrics import compute_event_metrics


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

PROFILE_ARGS = {
    "fast": [
        "--size",
        "224",
        "--step",
        "8",
        "--adaptive-step",
        "--adaptive-max-step",
        "16",
    ],
    "balanced": [
        "--size",
        "224",
        "--step",
        "6",
        "--adaptive-step",
        "--adaptive-max-step",
        "12",
        "--smooth_win",
        "7",
        "--ema_alpha",
        "0.3",
        "--tta",
    ],
    "safe": [
        "--size",
        "384",
        "--step",
        "4",
        "--adaptive-step",
        "--adaptive-high-risk",
        "0.75",
        "--hyst-high",
        "0.78",
        "--hyst-low",
        "0.50",
        "--persist-n",
        "6",
        "--smooth_win",
        "9",
        "--ema_alpha",
        "0.35",
        "--tta",
    ],
}


def _collect_videos(videos_dir: Path, recursive: bool = True) -> list[Path]:
    if recursive:
        paths = [p for p in videos_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    else:
        paths = [p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(paths)


def _safe_id(video_path: Path, root: Path) -> str:
    try:
        rel = video_path.relative_to(root)
    except ValueError:
        rel = video_path.name
    else:
        rel = rel.with_suffix("").as_posix()
    return str(rel).replace("/", "__").replace("\\", "__").replace(" ", "_")


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _compute_event_metrics(scored_csv: Path, events_csv: Path) -> dict:
    scored = pd.read_csv(scored_csv)
    events = pd.read_csv(events_csv) if events_csv.exists() and events_csv.stat().st_size > 0 else pd.DataFrame()

    out = {
        "frames_total": int(len(scored)),
        "confirmed_frames": int((scored.get("alarm_state", pd.Series(dtype=str)) == "confirmed").sum()),
        "fire_event_frames": int(pd.to_numeric(scored.get("fire_event", 0), errors="coerce").fillna(0).astype(int).sum()),
        "num_events": int(len(events)),
        "total_event_duration": 0,
        "max_event_duration": 0,
        "mean_event_duration": 0.0,
        "max_event_prob": 0.0,
        "mean_event_prob": 0.0,
    }
    if len(events) > 0:
        durations = pd.to_numeric(events["duration"], errors="coerce").fillna(0)
        max_probs = pd.to_numeric(events["max_prob"], errors="coerce").fillna(0.0)
        avg_probs = pd.to_numeric(events["avg_prob"], errors="coerce").fillna(0.0)
        out.update(
            {
                "total_event_duration": int(durations.sum()),
                "max_event_duration": int(durations.max()),
                "mean_event_duration": float(durations.mean()),
                "max_event_prob": float(max_probs.max()),
                "mean_event_prob": float(avg_probs.mean()),
            }
        )
    return out


def _video_duration_seconds(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps <= 1e-9:
        return 0.0
    return max(0.0, frames / fps)


def _read_metrics_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_metrics_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _profiles_to_run(profile: str) -> list[str]:
    if profile == "all":
        return ["fast", "balanced", "safe"]
    return [profile]


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch evaluation runner for video fire detection.")
    ap.add_argument("--videos_dir", required=True, help="Folder containing videos.")
    ap.add_argument("--profile", choices=["fast", "balanced", "safe", "all"], default="balanced")
    ap.add_argument("--output", default="outputs/eval_summary.csv", help="Aggregated summary CSV path.")
    ap.add_argument("--no_recursive", action="store_true", help="Disable recursive video search.")
    ap.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Optional limit: process only first N discovered videos.",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip per video/profile run if output files already exist.",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    videos_dir = Path(args.videos_dir)
    if not videos_dir.is_absolute():
        videos_dir = (project_root / videos_dir).resolve()
    if not videos_dir.exists():
        raise SystemExit(f"Videos directory not found: {videos_dir}")

    videos = _collect_videos(videos_dir, recursive=not args.no_recursive)
    if not videos:
        raise SystemExit(f"No video files found in: {videos_dir}")
    if args.max_videos is not None:
        max_n = max(0, int(args.max_videos))
        videos = videos[:max_n]
    if not videos:
        raise SystemExit("No videos selected after applying --max_videos.")

    profiles = _profiles_to_run(args.profile)
    out_summary = Path(args.output)
    if not out_summary.is_absolute():
        out_summary = project_root / out_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    eval_root = project_root / "outputs" / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    videos_considered = len(videos)
    runs_attempted = 0
    succeeded = 0
    failed = 0
    skipped = 0
    total_jobs = len(videos) * len(profiles)
    job_i = 0

    for profile in profiles:
        profile_args = PROFILE_ARGS[profile]
        profile_dir = eval_root / profile
        profile_dir.mkdir(parents=True, exist_ok=True)

        for video in videos:
            job_i += 1
            vid = _safe_id(video, videos_dir)
            pred_csv = profile_dir / f"{vid}__pred.csv"
            scored_csv = profile_dir / f"{vid}__scored.csv"
            events_csv = profile_dir / f"{vid}__events.csv"
            bench_json = profile_dir / f"{vid}__bench.json"
            event_metrics_json = profile_dir / f"{vid}__event_metrics.json"
            expected_outputs = [pred_csv, scored_csv, events_csv, bench_json]

            print(f"[{job_i}/{total_jobs}] profile={profile} video={video.name}")
            if args.skip_existing and all(p.exists() for p in expected_outputs):
                print("  - skipped (existing outputs)")
                duration_sec = _video_duration_seconds(video)
                metrics_source = "computed"
                try:
                    metrics = _compute_event_metrics(scored_csv=scored_csv, events_csv=events_csv)
                    event_metrics = _read_metrics_json(event_metrics_json)
                    if event_metrics is None:
                        event_metrics = compute_event_metrics(events_csv=events_csv, duration_sec=duration_sec)
                        _write_metrics_json(event_metrics_json, event_metrics)
                        metrics_source = "computed"
                    else:
                        metrics_source = "reused"
                    metrics.update(event_metrics)
                except Exception:
                    metrics = {}
                rows.append(
                    {
                        "status": "skipped",
                        "profile": profile,
                        "video_name": video.name,
                        "video_path": str(video),
                        "pred_csv": str(pred_csv),
                        "scored_csv": str(scored_csv),
                        "events_csv": str(events_csv),
                        "benchmark_json": str(bench_json),
                        "event_metrics_json": str(event_metrics_json),
                        "event_metrics_source": metrics_source,
                        "video_duration_sec": float(duration_sec),
                        "error": "",
                        "error_message": "",
                        **metrics,
                    }
                )
                skipped += 1
                continue

            try:
                runs_attempted += 1
                infer_cmd = [
                    sys.executable,
                    "src/05_video_infer.py",
                    "--rgb_video",
                    str(video),
                    "--out",
                    str(pred_csv),
                    "--benchmark",
                    "--benchmark-out",
                    str(bench_json),
                    *profile_args,
                ]
                _run_cmd(infer_cmd, cwd=project_root)

                risk_cmd = [
                    sys.executable,
                    "src/06_add_risk_score.py",
                    "--inp",
                    str(pred_csv),
                    "--out",
                    str(scored_csv),
                ]
                _run_cmd(risk_cmd, cwd=project_root)

                event_cmd = [
                    sys.executable,
                    "src/eval/event_extractor.py",
                    "--input",
                    str(scored_csv),
                    "--output",
                    str(events_csv),
                ]
                _run_cmd(event_cmd, cwd=project_root)

                duration_sec = _video_duration_seconds(video)
                metrics = _compute_event_metrics(scored_csv=scored_csv, events_csv=events_csv)
                event_metrics = compute_event_metrics(events_csv=events_csv, duration_sec=duration_sec)
                _write_metrics_json(event_metrics_json, event_metrics)
                metrics.update(event_metrics)
                rows.append(
                    {
                        "status": "ok",
                        "profile": profile,
                        "video_name": video.name,
                        "video_path": str(video),
                        "pred_csv": str(pred_csv),
                        "scored_csv": str(scored_csv),
                        "events_csv": str(events_csv),
                        "benchmark_json": str(bench_json),
                        "event_metrics_json": str(event_metrics_json),
                        "event_metrics_source": "computed",
                        "video_duration_sec": float(duration_sec),
                        "error": "",
                        "error_message": "",
                        **metrics,
                    }
                )
                succeeded += 1
            except subprocess.CalledProcessError as e:
                print(f"  ! failed: {e}")
                rows.append(
                    {
                        "status": "failed",
                        "profile": profile,
                        "video_name": video.name,
                        "video_path": str(video),
                        "pred_csv": str(pred_csv),
                        "scored_csv": str(scored_csv),
                        "events_csv": str(events_csv),
                        "benchmark_json": str(bench_json),
                        "event_metrics_json": str(event_metrics_json),
                        "event_metrics_source": "",
                        "error": f"command_failed: {e}",
                        "error_message": str(e),
                    }
                )
                failed += 1
            except Exception as e:  # broad on purpose for resilient batch processing
                print(f"  ! unexpected error: {e}")
                rows.append(
                    {
                        "status": "failed",
                        "profile": profile,
                        "video_name": video.name,
                        "video_path": str(video),
                        "pred_csv": str(pred_csv),
                        "scored_csv": str(scored_csv),
                        "events_csv": str(events_csv),
                        "benchmark_json": str(bench_json),
                        "event_metrics_json": str(event_metrics_json),
                        "event_metrics_source": "",
                        "error": str(e),
                        "error_message": str(e),
                    }
                )
                failed += 1

    summary = pd.DataFrame(rows)
    summary.to_csv(out_summary, index=False)
    print("\nEvaluation Summary")
    print(f"- videos considered: {videos_considered}")
    print(f"- runs attempted: {runs_attempted}")
    print(f"- succeeded: {succeeded}")
    print(f"- failed: {failed}")
    print(f"- skipped: {skipped}")
    print(f"- summary csv: {out_summary}")


if __name__ == "__main__":
    main()
