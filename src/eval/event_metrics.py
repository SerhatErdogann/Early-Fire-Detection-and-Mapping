"""
Compute event-level metrics from an events CSV.

Usage:
    python src/eval/event_metrics.py --events outputs/events.csv --duration_sec 120 --output outputs/event_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["event_id", "start_frame", "end_frame", "duration", "max_prob", "avg_prob"]


def _read_events(events_csv: Path) -> pd.DataFrame:
    if not events_csv.exists():
        raise FileNotFoundError(f"Events CSV not found: {events_csv}")
    try:
        df = pd.read_csv(events_csv)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    return df


def compute_event_metrics_df(events_df: pd.DataFrame, duration_sec: float) -> dict[str, float | int]:
    """
    Compute deterministic event metrics from event segments.

    Note:
    - ``confirmed_frames_total`` and ``confirmed_coverage_ratio`` are estimated from event durations
      as sum(duration + 1), i.e. in sampled-frame units.
    """
    dur_sec = max(float(duration_sec), 1e-9)
    event_count = int(len(events_df))

    if event_count == 0:
        return {
            "event_count": 0,
            "avg_event_duration": 0.0,
            "max_event_duration": 0.0,
            "min_event_duration": 0.0,
            "false_alarms_per_hour": 0.0,
            "events_per_minute": 0.0,
            "confirmed_frames_total": 0,
            "confirmed_coverage_ratio": 0.0,
        }

    durations = pd.to_numeric(events_df["duration"], errors="coerce").fillna(0.0)
    confirmed_frames_total = int((durations + 1).clip(lower=0).sum())
    duration_sum = float(durations.sum())

    return {
        "event_count": event_count,
        "avg_event_duration": float(durations.mean()),
        "max_event_duration": float(durations.max()),
        "min_event_duration": float(durations.min()),
        "false_alarms_per_hour": float(event_count * 3600.0 / dur_sec),
        "events_per_minute": float(event_count * 60.0 / dur_sec),
        "confirmed_frames_total": confirmed_frames_total,
        "confirmed_coverage_ratio": float(duration_sum / dur_sec),
    }


def compute_event_metrics(events_csv: Path, duration_sec: float) -> dict[str, float | int]:
    df = _read_events(events_csv)
    return compute_event_metrics_df(df, duration_sec=duration_sec)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute event-level metrics from events CSV.")
    ap.add_argument("--events", required=True, help="Path to events CSV.")
    ap.add_argument("--duration_sec", required=True, type=float, help="Video duration in seconds.")
    ap.add_argument("--output", required=True, help="Path to output JSON.")
    args = ap.parse_args()

    events_csv = Path(args.events)
    output_json = Path(args.output)

    metrics = compute_event_metrics(events_csv=events_csv, duration_sec=float(args.duration_sec))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Written: {output_json}")


if __name__ == "__main__":
    main()
