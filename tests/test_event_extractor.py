import pytest
import pandas as pd

from src.eval.event_extractor import EVENT_SUMMARY_COLUMNS, extract_events


def test_single_confirmed_event():
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2, 3, 4],
            "timestamp_sec": [0.0, 1 / 30, 2 / 30, 3 / 30, 4 / 30],
            "alarm_state": ["idle", "confirmed", "confirmed", "confirmed", "idle"],
            "decision_prob": [0.1, 0.7, 0.8, 0.9, 0.2],
        }
    )
    out = extract_events(df, merge_gap_sec=0.0, fps_fallback=30.0)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["event_id"] == "event_0001"
    assert float(row["start_sec"]) == pytest.approx(1 / 30)
    assert float(row["end_sec"]) == pytest.approx(3 / 30)
    assert float(row["duration_sec"]) == pytest.approx(2 / 30)
    assert float(row["max_prob"]) == pytest.approx(0.9)
    assert float(row["avg_prob"]) == pytest.approx((0.7 + 0.8 + 0.9) / 3.0)
    assert int(row["peak_frame"]) == 3


def test_two_events_merge_gap_zero():
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2, 3, 4, 5, 6],
            "timestamp_sec": [0.0, 1.0, 2.0, 50.0, 51.0, 52.0, 53.0],
            "alarm_state": ["confirmed", "confirmed", "idle", "idle", "confirmed", "confirmed", "idle"],
            "decision_prob": [0.8, 0.6, 0.2, 0.1, 0.75, 0.65, 0.1],
        }
    )
    out = extract_events(df, merge_gap_sec=0.0)
    assert len(out) == 2

    assert out.iloc[0]["event_id"] == "event_0001"
    assert float(out.iloc[0]["start_sec"]) == pytest.approx(0.0)
    assert float(out.iloc[1]["start_sec"]) == pytest.approx(51.0)


def test_gap_merge_below_two_seconds():
    df = pd.DataFrame(
        {
            "frame_idx": [10, 11, 12],
            "timestamp_sec": [100.0, 100.1, 101.7],
            "alarm_state": ["confirmed", "idle", "confirmed"],
            "decision_prob": [0.9, 0.1, 0.8],
        }
    )
    out = extract_events(df, merge_gap_sec=2.0)
    assert len(out) == 1
    assert float(out.iloc[0]["start_sec"]) == pytest.approx(100.0)
    assert float(out.iloc[0]["end_sec"]) == pytest.approx(101.7)



def test_suspected_raises_event_not_empty():
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2],
            "timestamp_sec": [0.0, 1.0, 2.0],
            "alarm_state": ["idle", "suspected", "idle"],
            "decision_prob": [0.2, 0.45, 0.1],
        }
    )
    out = extract_events(df, merge_gap_sec=2.0)
    assert len(out) == 1
    assert out.iloc[0]["risk_level"] == "suspected"


def test_no_events_when_only_idle_and_cooldown():
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2],
            "alarm_state": ["idle", "cooldown", "idle"],
            "decision_prob": [0.2, 0.3, 0.1],
        }
    )
    out = extract_events(df)
    assert out.empty
    assert list(out.columns) == list(EVENT_SUMMARY_COLUMNS)


def test_fallback_from_decision_prob_to_prob_fire():
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2],
            "timestamp_sec": [0.0, 10.0, 20.0],
            "alarm_state": ["confirmed", "confirmed", "idle"],
            "prob_fire": [0.33, 0.77, 0.1],
        }
    )
    out = extract_events(df, merge_gap_sec=0.0)
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
