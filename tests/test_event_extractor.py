import pytest
import pandas as pd

from src.eval.event_extractor import extract_events


def test_single_confirmed_event():
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2, 3, 4],
            "alarm_state": ["idle", "confirmed", "confirmed", "confirmed", "idle"],
            "decision_prob": [0.1, 0.7, 0.8, 0.9, 0.2],
        }
    )
    out = extract_events(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["event_id"] == "event_0001"
    assert int(row["start_frame"]) == 1
    assert int(row["end_frame"]) == 3
    assert int(row["duration"]) == 2
    assert float(row["max_prob"]) == pytest.approx(0.9)
    assert float(row["avg_prob"]) == pytest.approx((0.7 + 0.8 + 0.9) / 3.0)


def test_multiple_separated_confirmed_events():
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2, 3, 4, 5, 6],
            "alarm_state": ["confirmed", "confirmed", "idle", "idle", "confirmed", "confirmed", "idle"],
            "decision_prob": [0.8, 0.6, 0.2, 0.1, 0.75, 0.65, 0.1],
        }
    )
    out = extract_events(df)
    assert len(out) == 2

    e1 = out.iloc[0]
    assert e1["event_id"] == "event_0001"
    assert int(e1["start_frame"]) == 0
    assert int(e1["end_frame"]) == 1
    assert int(e1["duration"]) == 1
    assert float(e1["max_prob"]) == pytest.approx(0.8)
    assert float(e1["avg_prob"]) == pytest.approx(0.7)

    e2 = out.iloc[1]
    assert e2["event_id"] == "event_0002"
    assert int(e2["start_frame"]) == 4
    assert int(e2["end_frame"]) == 5
    assert int(e2["duration"]) == 1
    assert float(e2["max_prob"]) == pytest.approx(0.75)
    assert float(e2["avg_prob"]) == pytest.approx(0.7)


def test_event_closes_when_file_ends_confirmed():
    df = pd.DataFrame(
        {
            "frame_idx": [10, 11, 12],
            "alarm_state": ["idle", "confirmed", "confirmed"],
            "decision_prob": [0.1, 0.55, 0.95],
        }
    )
    out = extract_events(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert int(row["start_frame"]) == 11
    assert int(row["end_frame"]) == 12
    assert int(row["duration"]) == 1
    assert float(row["max_prob"]) == pytest.approx(0.95)
    assert float(row["avg_prob"]) == pytest.approx((0.55 + 0.95) / 2.0)


def test_no_confirmed_frames_returns_empty():
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2],
            "alarm_state": ["idle", "suspected", "cooldown"],
            "decision_prob": [0.2, 0.3, 0.1],
        }
    )
    out = extract_events(df)
    assert out.empty
    assert list(out.columns) == ["event_id", "start_frame", "end_frame", "duration", "max_prob", "avg_prob"]


def test_fallback_from_decision_prob_to_prob_fire():
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2],
            "alarm_state": ["confirmed", "confirmed", "idle"],
            "prob_fire": [0.33, 0.77, 0.1],
        }
    )
    out = extract_events(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert float(row["max_prob"]) == pytest.approx(0.77)
    assert float(row["avg_prob"]) == pytest.approx((0.33 + 0.77) / 2.0)


def test_empty_input_and_missing_columns_handled_cleanly():
    empty_df = pd.DataFrame(columns=["frame_idx", "alarm_state", "decision_prob"])
    out = extract_events(empty_df)
    assert out.empty

    with pytest.raises(ValueError, match="frame_idx"):
        extract_events(pd.DataFrame({"alarm_state": ["confirmed"], "decision_prob": [0.9]}))

    with pytest.raises(ValueError, match="alarm_state"):
        extract_events(pd.DataFrame({"frame_idx": [1], "decision_prob": [0.9]}))

    with pytest.raises(ValueError, match="decision_prob|prob_fire"):
        extract_events(pd.DataFrame({"frame_idx": [1], "alarm_state": ["confirmed"]}))
