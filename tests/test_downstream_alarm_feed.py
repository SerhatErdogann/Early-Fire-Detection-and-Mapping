from __future__ import annotations

from pathlib import Path

from src.inference.downstream_alarm_feed import (
    SCHEMA_VERSION,
    alarm_feed_row_dict,
    confidence_level_from_state,
    export_alarm_feed_bundle,
    fire_detected_from_state,
    load_alarm_feed_csv,
    public_alarm_state,
    risk_level_from_state,
    write_alarm_feed_jsonl,
)


def test_public_alarm_state_mapping():
    assert public_alarm_state("idle", temporal_guard=True) == "ok"
    assert public_alarm_state("suspected", temporal_guard=True) == "suspected"
    assert public_alarm_state("confirmed", temporal_guard=True) == "confirmed"
    assert public_alarm_state("confirmed", temporal_guard=False) == "ok"


def test_fire_risk_confidence():
    assert fire_detected_from_state("ok") is False
    assert fire_detected_from_state("suspected") is True
    assert fire_detected_from_state("confirmed") is True
    assert risk_level_from_state("ok") == "none"
    assert risk_level_from_state("suspected") == "medium"
    assert risk_level_from_state("confirmed") == "high"
    assert confidence_level_from_state("ok") == "low"


def test_alarm_feed_row_roundtrip(tmp_path: Path):
    row = alarm_feed_row_dict(
        frame_idx=10,
        timestamp_sec=1.5,
        fire_probability=0.9,
        smoothed_probability=0.85,
        internal_alarm_state="confirmed",
        temporal_guard=True,
        pred_fire_burst=1,
        episode_start_ts=0.5,
    )
    assert row["schema_version"] == SCHEMA_VERSION
    assert row["sampled_frame"] == 10
    assert row["inferred"] == 1
    assert row["skipped_similar"] == 0
    assert row["skipped_budget"] == 0
    assert row["alarm_state"] == "confirmed"
    assert row["fire_detected"] is True
    assert row["risk_level"] == "high"
    assert row["confidence_level"] == "high"
    assert row["burst_gate_active"] is True
    assert isinstance(row["alarm_duration"], float)

    out = tmp_path / "video_predictions.csv"
    out.write_text("")
    paths = export_alarm_feed_bundle(out, [row])
    assert Path(paths["alarm_feed_csv"]).exists()
    assert Path(paths["alarm_feed_jsonl"]).exists()
    assert Path(paths["alarm_feed_schema"]).exists()

    df = load_alarm_feed_csv(paths["alarm_feed_csv"])
    assert len(df) == 1
    assert df.iloc[0]["alarm_state"] == "confirmed"

    jl = tmp_path / "x.jsonl"
    write_alarm_feed_jsonl([row], jl)
    text = jl.read_text(encoding="utf-8").strip()
    assert '"alarm_state": "confirmed"' in text
