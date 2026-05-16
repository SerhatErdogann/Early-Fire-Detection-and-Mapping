import json
import runpy
import sys
from pathlib import Path

import pandas as pd

from src.eval.profile_comparison import build_profile_comparison


def test_profile_comparison_ranking_ignores_failed(tmp_path: Path):
    inp = tmp_path / "eval_summary.csv"
    pd.DataFrame(
        [
            {
                "video_name": "v1.mp4",
                "profile": "fast",
                "status": "ok",
                "video_duration_sec": 100.0,
                "event_count": 3,
                "false_alarms_per_hour": 20.0,
                "avg_event_duration": 2.0,
                "confirmed_coverage_ratio": 0.2,
                "pipeline_fps_processed": 12.0,
                "error_message": "",
            },
            {
                "video_name": "v1.mp4",
                "profile": "balanced",
                "status": "ok",
                "video_duration_sec": 100.0,
                "event_count": 2,
                "false_alarms_per_hour": 5.0,
                "avg_event_duration": 3.0,
                "confirmed_coverage_ratio": 0.1,
                "pipeline_fps_processed": 8.0,
                "error_message": "",
            },
            {
                "video_name": "v1.mp4",
                "profile": "safe",
                "status": "ok",
                "video_duration_sec": 100.0,
                "event_count": 1,
                "false_alarms_per_hour": 2.0,
                "avg_event_duration": 4.0,
                "confirmed_coverage_ratio": 0.05,
                "pipeline_fps_processed": 6.0,
                "error_message": "",
            },
            {
                "video_name": "v2.mp4",
                "profile": "fast",
                "status": "failed",
                "video_duration_sec": 100.0,
                "event_count": 0,
                "false_alarms_per_hour": 0.0,
                "avg_event_duration": 0.0,
                "confirmed_coverage_ratio": 0.0,
                "pipeline_fps_processed": 0.0,
                "error_message": "boom",
            },
        ]
    ).to_csv(inp, index=False)

    result = build_profile_comparison(inp)
    picks = result["picks"]
    assert picks["fastest"] == "fast"
    assert picks["safest"] == "safe"
    assert picks["balanced"] == "balanced"
    assert picks["recommended_default"] == "balanced"
    assert result["failed_count"] == 1


def test_profile_comparison_cli_writes_md_and_json(tmp_path: Path):
    inp = tmp_path / "eval_summary.csv"
    out_md = tmp_path / "profile_comparison.md"
    out_json = tmp_path / "profile_comparison.json"

    pd.DataFrame(
        [
            {
                "video_name": "v1.mp4",
                "profile": "balanced",
                "status": "ok",
                "video_duration_sec": 120.0,
                "event_count": 1,
                "false_alarms_per_hour": 3.0,
                "avg_event_duration": 2.0,
                "confirmed_coverage_ratio": 0.02,
                "pipeline_fps_processed": 7.0,
                "error_message": "",
            }
        ]
    ).to_csv(inp, index=False)

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "profile_comparison.py",
            "--input",
            str(inp),
            "--output",
            str(out_md),
            "--json_output",
            str(out_json),
        ]
        runpy.run_path(str(Path("src") / "eval" / "profile_comparison.py"), run_name="__main__")
    finally:
        sys.argv = old_argv

    assert out_md.exists()
    text = out_md.read_text(encoding="utf-8")
    assert "Profile Comparison Report" in text
    assert "Recommended default profile" in text

    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["picks"]["recommended_default"] == "balanced"
