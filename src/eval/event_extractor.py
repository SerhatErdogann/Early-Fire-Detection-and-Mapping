"""
Frame-level çıktılardan olay (event) segmentleri çıkarımı.

Şüpheli/onaylı alarm süreleri zaman ekseninde birleştirilir; küçük boşluklar
(merge_gap_sec) iki olayı tek olay olarak birleştirir.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Geriye dönük: eski araç çıktıları
EVENT_COLUMNS = [
    "event_id",
    "start_frame",
    "end_frame",
    "duration",
    "max_prob",
    "avg_prob",
]

EVENT_SUMMARY_COLUMNS = [
    "event_id",
    "start_sec",
    "end_sec",
    "duration_sec",
    "max_prob",
    "avg_prob",
    "peak_frame",
    "risk_level",
]


def _pick_prob_column(df: pd.DataFrame) -> str:
    if "decision_prob" in df.columns:
        return "decision_prob"
    if "prob_fire" in df.columns:
        return "prob_fire"
    raise ValueError("Input CSV must contain 'decision_prob' or 'prob_fire' column.")


def _row_timestamp_sec(row: object, fps_fallback: float) -> float:
    ts = getattr(row, "timestamp_sec", None)
    if ts is not None:
        try:
            if pd.notna(ts):
                return float(ts)
        except Exception:
            pass
    fi = int(getattr(row, "frame_idx", 0))
    return float(fi) / max(fps_fallback, 1e-6)


def _state_elevated(state: object) -> bool:
    return str(state or "").strip().lower() in ("suspected", "confirmed")


def _risk_level_segment(states: list[str]) -> str:
    lows = {str(s).lower() for s in states}
    if "confirmed" in lows:
        return "confirmed"
    if "suspected" in lows:
        return "suspected"
    return "ok"


def merge_events_by_gap(
    raw: pd.DataFrame,
    *,
    merge_gap_sec: float = 2.0,
) -> pd.DataFrame:
    """Ham olay bloklarını zaman boşluğuna göre birleştir."""
    if raw.empty:
        return pd.DataFrame(columns=EVENT_SUMMARY_COLUMNS)
    gap = max(0.0, float(merge_gap_sec))
    chunks: list[list[int]] = []  # row indices into raw
    cur_chunk: list[int] = []

    rs = raw.reset_index(drop=True)
    prev_end = None
    for i in range(len(rs)):
        row = rs.iloc[i]
        s_sec = float(row["start_sec"])
        e_sec = float(row["end_sec"])
        if not cur_chunk:
            cur_chunk = [int(i)]
            prev_end = e_sec
            continue
        if s_sec - float(prev_end) <= gap:
            cur_chunk.append(int(i))
            prev_end = max(float(prev_end), e_sec)
        else:
            chunks.append(cur_chunk)
            cur_chunk = [int(i)]
            prev_end = e_sec
    if cur_chunk:
        chunks.append(cur_chunk)

    out_rows: list[dict] = []
    for nth, ix_list in enumerate(chunks, start=1):
        sub = rs.iloc[ix_list]
        pk = sub.loc[sub["max_prob"].idxmax()]
        rl = _risk_level_segment([str(x) for x in sub["risk_level"].tolist()])
        out_rows.append(
            {
                "event_id": f"event_{nth:04d}",
                "start_sec": float(sub["start_sec"].min()),
                "end_sec": float(sub["end_sec"].max()),
                "duration_sec": float(sub["end_sec"].max() - sub["start_sec"].min()),
                "max_prob": float(sub["max_prob"].max()),
                "avg_prob": float(pd.to_numeric(sub["avg_prob"], errors="coerce").mean()),
                "peak_frame": int(pk["peak_frame"]),
                "risk_level": rl,
            }
        )
    return pd.DataFrame(out_rows, columns=list(EVENT_SUMMARY_COLUMNS))


def extract_operational_events(
    df: pd.DataFrame,
    *,
    merge_gap_sec: float = 2.0,
    fps_fallback: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    İç alarm durumu şüpheli/onaylı iken oluşan segmentler + birleştirilmiş özet.

    Returns:
        (merged_event_summary_df, raw_before_merge_rows_as_df_optional)
        İkinci değer yalın ham segmentler için `events_raw` olarak saklanır.
    """
    if "frame_idx" not in df.columns:
        raise ValueError("Input CSV must contain 'frame_idx' column.")
    if df.empty:
        empt = pd.DataFrame(columns=list(EVENT_SUMMARY_COLUMNS))
        return empt, empt.copy()

    data = df.copy()
    data["frame_idx"] = pd.to_numeric(data["frame_idx"], errors="coerce")
    data = data.dropna(subset=["frame_idx"]).sort_values("frame_idx").reset_index(drop=True)
    if data.empty:
        empt = pd.DataFrame(columns=list(EVENT_SUMMARY_COLUMNS))
        return empt, empt.copy()

    if "alarm_state" not in data.columns:
        raise ValueError("Input CSV must contain 'alarm_state' column.")

    prob_col = _pick_prob_column(data)
    data[prob_col] = pd.to_numeric(data[prob_col], errors="coerce").fillna(0.0)

    segments: list[dict] = []
    in_evt = False
    start_ts = end_ts = 0.0
    start_fr = end_fr = 0
    probs: list[float] = []
    states: list[str] = []
    frames_data: list[tuple[int, float]] = []

    for row in data.itertuples(index=False):
        frame = int(row.frame_idx)
        state = str(getattr(row, "alarm_state", "idle"))
        prob = float(getattr(row, prob_col))
        ts = _row_timestamp_sec(row, fps_fallback)

        elev = _state_elevated(state)
        if elev:
            if not in_evt:
                in_evt = True
                start_ts = end_ts = ts
                start_fr = end_fr = frame
                probs = [prob]
                states = [state]
                frames_data = [(frame, prob)]
            else:
                end_ts = ts
                end_fr = frame
                probs.append(prob)
                states.append(state)
                frames_data.append((frame, prob))
        elif in_evt:
            peak_fr, mx = max(frames_data, key=lambda x: x[1])
            segments.append(
                {
                    "start_sec": float(start_ts),
                    "end_sec": float(end_ts),
                    "duration_sec": max(0.0, float(end_ts - start_ts)),
                    "max_prob": float(max(probs)),
                    "avg_prob": float(sum(probs) / len(probs)),
                    "peak_frame": int(peak_fr),
                    "risk_level": _risk_level_segment(states),
                }
            )
            in_evt = False
            probs = []
            states = []
            frames_data = []

    if in_evt:
        peak_fr, _mx = max(frames_data, key=lambda x: x[1])
        segments.append(
            {
                "start_sec": float(start_ts),
                "end_sec": float(end_ts),
                "duration_sec": max(0.0, float(end_ts - start_ts)),
                "max_prob": float(max(probs)),
                "avg_prob": float(sum(probs) / len(probs)),
                "peak_frame": int(peak_fr),
                "risk_level": _risk_level_segment(states),
            }
        )

    raw_df = pd.DataFrame(segments)
    if raw_df.empty:
        empt = pd.DataFrame(columns=list(EVENT_SUMMARY_COLUMNS))
        return empt, empt.copy()

    raw_df.insert(0, "event_id", [f"raw_{i+1:04d}" for i in range(len(raw_df))])
    raw_out = raw_df.rename(
        columns={
            "risk_level": "risk_level",
        }
    )
    # Ham tablo için sütun hizası
    raw_aligned = pd.DataFrame(
        {
            "event_id": raw_out["event_id"],
            "start_sec": raw_out["start_sec"],
            "end_sec": raw_out["end_sec"],
            "duration_sec": raw_out["duration_sec"],
            "max_prob": raw_out["max_prob"],
            "avg_prob": raw_out["avg_prob"],
            "peak_frame": raw_out["peak_frame"],
            "risk_level": raw_out["risk_level"],
        }
    )

    merged = merge_events_by_gap(raw_aligned.drop(columns=["event_id"]), merge_gap_sec=merge_gap_sec)
    return merged, raw_aligned


def extract_events_legacy_confirmed_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Yalnızca `confirmed` sürekesenleri (legacy CLI uyumu)."""
    if "frame_idx" not in df.columns or "alarm_state" not in df.columns:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    data = df.copy()
    data["frame_idx"] = pd.to_numeric(data["frame_idx"], errors="coerce")
    data = data.dropna(subset=["frame_idx"]).sort_values("frame_idx").reset_index(drop=True)
    prob_col = _pick_prob_column(data)
    data[prob_col] = pd.to_numeric(data[prob_col], errors="coerce").fillna(0.0)

    events: list[dict] = []
    in_event = False
    start_frame = end_frame = 0
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


def extract_events(
    df: pd.DataFrame,
    *,
    merge_gap_sec: float = 2.0,
    fps_fallback: float = 30.0,
    legacy_confirmed_only: bool = False,
) -> pd.DataFrame:
    """
    Varsayılan: operasyon olayları (suspected+confirmed), ``merge_gap_sec`` ile birleştirilmiş.
    legacy_confirmed_only=True: eski davranış.
    """
    if legacy_confirmed_only:
        return extract_events_legacy_confirmed_frames(df)
    merged, _raw = extract_operational_events(df, merge_gap_sec=merge_gap_sec, fps_fallback=fps_fallback)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Frame-level çıktılardan olay segmentleri.")
    ap.add_argument(
        "--input",
        required=True,
        help="CSV (typically video_predictions_scored.csv).",
    )
    ap.add_argument("--output", default="outputs/events.csv", help="Çıktı CSV.")
    ap.add_argument(
        "--legacy-confirmed-only",
        action="store_true",
        help="Yalnızca confirmed süreleri (legacy).",
    )
    ap.add_argument("--merge-gap-sec", type=float, default=2.0, help="Olayları birleştirmek için maksimum boşluk (saniye).")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    df = pd.read_csv(inp)
    events = extract_events(
        df,
        merge_gap_sec=float(args.merge_gap_sec),
        legacy_confirmed_only=bool(args.legacy_confirmed_only),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(out, index=False)
    print(f"Written: {out}")
    print(f"Events: {len(events)}")


if __name__ == "__main__":
    main()
