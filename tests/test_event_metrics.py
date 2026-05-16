import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.eval.event_metrics import compute_event_metrics, compute_event_metrics_df


def test_compute_event_metrics_basic():
    events = pd.DataFrame(
        {
            "event_id": ["event_0001", "event_0002"],
            "start_frame": [10, 40],
            "end_frame": [20, 50],
            "duration": [10, 10],
            "max_prob": [0.9, 0.8],
            "avg_prob": [0.7, 0.6],
        }
    )
    m = compute_event_metrics_df(events, duration_sec=120.0)
    assert m["event_count"] == 2
    assert m["avg_event_duration"] == pytest.approx(10.0)
    assert m["max_event_duration"] == pytest.approx(10.0)
    assert m["min_event_duration"] == pytest.approx(10.0)
    assert m["false_alarms_per_hour"] == pytest.approx(60.0)
    assert m["events_per_minute"] == pytest.approx(1.0)
    assert m["confirmed_frames_total"] == 22


def test_compute_event_metrics_empty_events(tmp_path: Path):
    p = tmp_path / "events.csv"
    pd.DataFrame(columns=["event_id", "start_frame", "end_frame", "duration", "max_prob", "avg_prob"]).to_csv(
        p, index=False
    )
    m = compute_event_metrics(p, duration_sec=100.0)
    assert m["event_count"] == 0
    assert m["false_alarms_per_hour"] == 0.0
    assert m["confirmed_coverage_ratio"] == 0.0


def test_compute_event_metrics_missing_required_column(tmp_path: Path):
    p = tmp_path / "bad_events.csv"
    pd.DataFrame({"event_id": ["event_0001"], "duration": [3]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Missing required column"):
        compute_event_metrics(p, duration_sec=100.0)


def test_cli_writes_json(tmp_path: Path):
    events_csv = tmp_path / "events.csv"
    out_json = tmp_path / "metrics.json"
    pd.DataFrame(
        {
            "event_id": ["event_0001"],
            "start_frame": [1],
            "end_frame": [2],
            "duration": [1],
            "max_prob": [0.8],
            "avg_prob": [0.7],
        }
    ).to_csv(events_csv, index=False)

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "event_metrics.py",
            "--events",
            str(events_csv),
            "--duration_sec",
            "120",
            "--output",
            str(out_json),
        ]
        runpy.run_path(str(Path("src") / "eval" / "event_metrics.py"), run_name="__main__")
    finally:
        sys.argv = old_argv

    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["event_count"] == 1
    assert "false_alarms_per_hour" in payload
