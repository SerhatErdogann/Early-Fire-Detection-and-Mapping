"""
Extract event-level segments from frame-level fire predictions.

Usage:
    python src/eval/event_extractor.py --input outputs/video_predictions_scored.csv --output outputs/events.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EVENT_COLUMNS = [
    "event_id",
    "start_frame",
    "end_frame",
    "duration",
    "max_prob",
    "avg_prob",
]


def _pick_prob_column(df: pd.DataFrame) -> str:
    if "decision_prob" in df.columns:
        return "decision_prob"
    if "prob_fire" in df.columns:
        return "prob_fire"
    raise ValueError("Input CSV must contain 'decision_prob' or 'prob_fire' column.")


def extract_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert frame-level predictions into event segments.

    An event is a contiguous run (row-wise) where alarm_state == "confirmed".
    Event starts when state enters confirmed and ends when it leaves confirmed.
    """
    if "frame_idx" not in df.columns:
        raise ValueError("Input CSV must contain 'frame_idx' column.")
    if "alarm_state" not in df.columns:
        raise ValueError("Input CSV must contain 'alarm_state' column.")

    if df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    data = df.copy()
    data["frame_idx"] = pd.to_numeric(data["frame_idx"], errors="coerce")
    data = data.dropna(subset=["frame_idx"]).sort_values("frame_idx").reset_index(drop=True)
    if data.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    prob_col = _pick_prob_column(data)
    data[prob_col] = pd.to_numeric(data[prob_col], errors="coerce").fillna(0.0)

    events: list[dict] = []
    in_event = False
    start_frame = 0
    probs: list[float] = []
    event_no = 0

    for row in data.itertuples(index=False):
        frame = int(row.frame_idx)
        state = str(row.alarm_state)
        prob = float(getattr(row, prob_col))

        if state == "confirmed":
            if not in_event:
                in_event = True
                start_frame = frame
                probs = [prob]
            else:
                probs.append(prob)
            end_frame = frame
        elif in_event:
            event_no += 1
            events.append(
                {
                    "event_id": f"event_{event_no:04d}",
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "duration": int(end_frame - start_frame),
                    "max_prob": float(max(probs)) if probs else 0.0,
                    "avg_prob": float(sum(probs) / len(probs)) if probs else 0.0,
                }
            )
            in_event = False
            probs = []

    # Edge case: file ends while still in confirmed state.
    if in_event:
        event_no += 1
        events.append(
            {
                "event_id": f"event_{event_no:04d}",
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "duration": int(end_frame - start_frame),
                "max_prob": float(max(probs)) if probs else 0.0,
                "avg_prob": float(sum(probs) / len(probs)) if probs else 0.0,
            }
        )

    return pd.DataFrame(events, columns=EVENT_COLUMNS)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert frame-level alarms into event-level segments.")
    ap.add_argument(
        "--input",
        required=True,
        help="Input CSV path (typically outputs/video_predictions_scored.csv because alarm_state is required).",
    )
    ap.add_argument("--output", default="outputs/events.csv", help="Output CSV path for extracted events.")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    df = pd.read_csv(inp)
    events = extract_events(df)

    out.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(out, index=False)
    print(f"Written: {out}")
    print(f"Events: {len(events)}")


if __name__ == "__main__":
    main()
