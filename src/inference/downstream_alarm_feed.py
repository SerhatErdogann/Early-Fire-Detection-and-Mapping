"""Downstream mapping / GIS alarm bileşeni için stabilized çıktı sözleşmesi.

Ana CSV (``video_predictions.csv``) tüm ara alanları tutar; bu modül ise
sırasıyla **temporal smoothing, EMA, burst metriği ve histerezis / persistence**
işlendikten sonra harici sisteme iletilecek sütunları tek biçimde üretir.

Şema sürümü: ``schema_version`` alanıyla birlikte değişir; downstream bu alanı
doğrulamalıdır.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd

SCHEMA_VERSION = "1.0"

AlarmStatePublic = Literal["ok", "suspected", "confirmed"]
ConfidenceLevel = Literal["low", "medium", "high"]
RiskLevel = Literal["none", "medium", "high"]

# Harici sisteme yazılan sütun sırası (CSV / JSON Lines ortak).
ALARM_FEED_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "frame_idx",
    "timestamp",
    "fire_probability",
    "smoothed_probability",
    "alarm_state",
    "fire_detected",
    "risk_level",
    "confidence_level",
    "first_alarm_ts",
    "last_alarm_ts",
    "alarm_duration",
    "burst_gate_active",
)

ALARM_FEED_DOC: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "columns": ALARM_FEED_COLUMNS,
    "semantics": {
        "timestamp": "Clip başlangıcından bu örnek kareye kadar süre (saniye, kaynak FPS).",
        "fire_probability": "Ham model çıkışı (softmax/TTA öncesi ham kare ortalaması).",
        "smoothed_probability": "Temporal + (varsa CAM/uzamsal) düzeltmeler sonrası "
        "histerezise giren karar olasılığı.",
        "alarm_state": {"ok": "Normal; harici fire=false.", "suspected": "Orta risk.", "confirmed": "Yüksek güven uyarısı."},
        "fire_detected": "ok dışında true (suspected | confirmed için true).",
        "risk_level": {"none": "ok", "medium": "suspected", "high": "confirmed"},
        "confidence_level": "low | medium | high — alarm_state ile uyumlu.",
        "burst_gate_active": "Ardışık kare burst eşiği (filtre) için bayrak.",
        "first_alarm_ts": "Bu satır için aktif uyarı süitinin başlangıç zamanı (clip saniye); uyarı yoksa boş.",
        "last_alarm_ts": "Uyarı süitinin son zamanı; uyarı yoksa boş.",
        "alarm_duration": "Uyarı süitinde geçen süre (saniye); uyarı yoksa 0.",
    },
}


def public_alarm_state(
    internal_alarm_state: str,
    *,
    temporal_guard: bool,
) -> AlarmStatePublic:
    """Harici yüz yüz görünüm: idle/cooldown -> ok."""
    if not temporal_guard:
        return "ok"
    s = (internal_alarm_state or "").lower().strip()
    if s in ("confirmed",):
        return "confirmed"
    if s in ("suspected",):
        return "suspected"
    # idle, cooldown veya beklenmedik → ok
    return "ok"


def fire_detected_from_state(pub: AlarmStatePublic) -> bool:
    return pub in ("suspected", "confirmed")


def risk_level_from_state(pub: AlarmStatePublic) -> RiskLevel:
    if pub == "confirmed":
        return "high"
    if pub == "suspected":
        return "medium"
    return "none"


def confidence_level_from_state(pub: AlarmStatePublic) -> ConfidenceLevel:
    if pub == "confirmed":
        return "high"
    if pub == "suspected":
        return "medium"
    return "low"


def alarm_feed_row_dict(
    *,
    frame_idx: int,
    timestamp_sec: float,
    fire_probability: float,
    smoothed_probability: float,
    internal_alarm_state: str,
    temporal_guard: bool,
    pred_fire_burst: int,
    episode_start_ts: float | None,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    """Tek kare/satır için alarm feed kaydı (dict)."""
    pub = public_alarm_state(internal_alarm_state, temporal_guard=temporal_guard)

    ts = float(timestamp_sec)
    if episode_start_ts is not None:
        first_ts = float(episode_start_ts)
        last_ts = ts
        duration = max(0.0, last_ts - first_ts)
        first_cell: float | str = round(first_ts, 6)
        last_cell: float | str = round(last_ts, 6)
    else:
        first_cell = ""
        last_cell = ""
        duration = 0.0

    return {
        "schema_version": schema_version,
        "frame_idx": int(frame_idx),
        "timestamp": round(ts, 6),
        "fire_probability": round(float(fire_probability), 8),
        "smoothed_probability": round(float(smoothed_probability), 8),
        "alarm_state": pub,
        "fire_detected": bool(fire_detected_from_state(pub)),
        "risk_level": risk_level_from_state(pub),
        "confidence_level": confidence_level_from_state(pub),
        "first_alarm_ts": first_cell,
        "last_alarm_ts": last_cell,
        "alarm_duration": round(float(duration), 6),
        "burst_gate_active": bool(int(pred_fire_burst)),
    }


def write_alarm_feed_manifest(path: Path) -> None:
    """Şema özeti JSON (bir kez yazılır)."""
    doc = dict(ALARM_FEED_DOC)
    doc["schema_version"] = SCHEMA_VERSION
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


def alarm_feed_paths_for_csv(out_csv: Path) -> tuple[Path, Path, Path]:
    """video_predictions.csv yanına stabilize dosya adları."""
    stem = Path(out_csv).stem
    parent = Path(out_csv).parent
    return (
        parent / f"{stem}_alarm_feed.csv",
        parent / f"{stem}_alarm_feed.jsonl",
        parent / f"{stem}_alarm_feed.schema.json",
    )


def write_alarm_feed_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        df = pd.DataFrame(columns=list(ALARM_FEED_COLUMNS))
    else:
        df = pd.DataFrame(rows)
        df = df.reindex(columns=list(ALARM_FEED_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_alarm_feed_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            rec = {k: row[k] for k in ALARM_FEED_COLUMNS if k in row}
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def load_alarm_feed_csv(path: str | Path) -> pd.DataFrame:
    """Downstream doğrulama / okuma için ince sar."""
    df = pd.read_csv(path)
    cols = list(ALARM_FEED_COLUMNS)
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"alarm_feed CSV eksik kolonlar: {sorted(missing)}")
    return df.reindex(columns=cols)


def export_alarm_feed_bundle(
    out_csv: Path | str,
    alarm_feed_rows: list[dict[str, Any]],
) -> dict[str, str]:
    """Haritalama bileşeni için üç dosyayı tek seferde yazar."""
    outp = Path(out_csv)
    c_path, jl_path, sc_path = alarm_feed_paths_for_csv(outp)
    write_alarm_feed_csv(alarm_feed_rows, c_path)
    write_alarm_feed_jsonl(alarm_feed_rows, jl_path)
    write_alarm_feed_manifest(sc_path)
    return {
        "alarm_feed_csv": str(c_path),
        "alarm_feed_jsonl": str(jl_path),
        "alarm_feed_schema": str(sc_path),
    }


__all__ = [
    "SCHEMA_VERSION",
    "ALARM_FEED_COLUMNS",
    "ALARM_FEED_DOC",
    "AlarmStatePublic",
    "ConfidenceLevel",
    "RiskLevel",
    "public_alarm_state",
    "fire_detected_from_state",
    "risk_level_from_state",
    "confidence_level_from_state",
    "alarm_feed_row_dict",
    "alarm_feed_paths_for_csv",
    "write_alarm_feed_csv",
    "write_alarm_feed_jsonl",
    "write_alarm_feed_manifest",
    "load_alarm_feed_csv",
    "export_alarm_feed_bundle",
]
