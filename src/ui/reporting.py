"""Export summaries: Markdown, CSV, ZIP of suspicious frames."""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


@dataclass
class FinalReport:
    verdict_tr: str
    verdict_key: str
    max_prob: float
    mean_prob: float
    frames_analyzed: int
    alarm_esigi: float
    inceleme_esigi: float
    alarm_zaman_araliklari: list[tuple[float, float]]
    guvenilirlik_notu: str
    raw_summary: dict[str, Any]


def _pick_prob_col(df: pd.DataFrame) -> str:
    return "decision_prob" if "decision_prob" in df.columns else "prob_fire"


def alarm_time_segments(
    df: pd.DataFrame,
    *,
    alarm_esigi: float,
    inceleme_esigi: float,
    use_column: str | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return (yüksek_uyarı_aralıkları, inceleme_üzeri_aralıkları) in seconds."""
    if df.empty or "timestamp_sec" not in df.columns:
        return [], []
    df = df.sort_values("frame_idx").reset_index(drop=True)
    col = use_column or _pick_prob_col(df)
    if col not in df.columns:
        return [], []
    t = pd.to_numeric(df["timestamp_sec"], errors="coerce").fillna(0.0).values
    p = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values

    def merge_segments(mask: np.ndarray) -> list[tuple[float, float]]:
        segs: list[tuple[float, float]] = []
        i = 0
        n = len(mask)
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            segs.append((float(t[i]), float(t[j])))
            i = j + 1
        return segs

    high = p >= float(alarm_esigi)
    review = (p >= float(inceleme_esigi)) & (~high)
    return merge_segments(high), merge_segments(review)


def reliability_note(df: pd.DataFrame, max_prob: float) -> str:
    n = len(df)
    parts: list[str] = []
    if n < 8:
        parts.append("Çok az kare analiz edildi; sonuçlar özet niteliğindedir.")
    if "scene_changed" in df.columns:
        sc = pd.to_numeric(df["scene_changed"], errors="coerce").fillna(0)
        frac = float(sc.mean()) if len(sc) else 0.0
        if frac > 0.25:
            parts.append("Görüntü sık sık değiştiği için olasılık dalgalanması normaldir.")
    if max_prob >= 0.85:
        parts.append("En yüksek olasılık çok yüksek; görüntüyü mutlaka insan gözüyle doğrulayın.")
    elif max_prob >= 0.55:
        parts.append("Olasılıklar orta düzeyde; çevresel parlamalar yanlış uyarı üretebilir.")
    else:
        parts.append("Olasılıklar genelde düşük kaldı; güçlü bir yangın paterni raporlanmadı.")
    if not parts:
        parts.append("Model çıktıları tek başına kesin teşhis yerine geçmez.")
    return " ".join(parts)


def build_final_report(
    df_scored: pd.DataFrame,
    df_events: pd.DataFrame,
    threshold_used: float,
    hyst_high: float,
    hyst_low: float,
) -> FinalReport:
    prob_col = _pick_prob_col(df_scored)
    probs = pd.to_numeric(df_scored.get(prob_col, 0.0), errors="coerce").fillna(0.0)
    mx = float(probs.max()) if len(probs) else 0.0
    mu = float(probs.mean()) if len(probs) else 0.0
    thr = float(threshold_used)

    fire_mask = (
        df_scored["pred_fire"].astype(int) == 1
        if "pred_fire" in df_scored.columns
        else probs >= thr
    )
    fire_frames = int(fire_mask.sum())
    alarm_events = int(len(df_events)) if df_events is not None else 0

    confirmed = False
    if "alarm_state" in df_scored.columns:
        confirmed = bool((df_scored["alarm_state"].astype(str) == "confirmed").any())

    has_fire_event = bool((df_scored["fire_event"].astype(int) == 1).any()) if "fire_event" in df_scored.columns else False

    # Final özet tek kare ile kırmızıya sıçramasın — teyit / olay süreleri öncelikli.
    if confirmed or alarm_events > 0 or has_fire_event:
        verdict_key = "fire"
        verdict_tr = "Yangın riski yüksek"
    elif mx >= float(hyst_low) or fire_frames > 0:
        verdict_key = "review"
        verdict_tr = "İnceleme gerekli"
    else:
        verdict_key = "safe"
        verdict_tr = "Yangın yok"

    high_segs, _ = alarm_time_segments(
        df_scored,
        alarm_esigi=float(hyst_high),
        inceleme_esigi=float(hyst_low),
        use_column=prob_col,
    )
    # Prefer merged high segments; if empty but events exist, leave empty (event table separate)
    note = reliability_note(df_scored, mx)

    raw = {
        "yangin_frame_sayisi": int(fire_frames),
        "olay_sayisi": int(alarm_events),
        "max_prob": mx,
        "mean_prob": mu,
    }
    return FinalReport(
        verdict_tr=verdict_tr,
        verdict_key=verdict_key,
        max_prob=mx,
        mean_prob=mu,
        frames_analyzed=int(len(df_scored)),
        alarm_esigi=float(hyst_high),
        inceleme_esigi=float(hyst_low),
        alarm_zaman_araliklari=high_segs,
        guvenilirlik_notu=note,
        raw_summary=raw,
    )


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8-sig")


def build_markdown_report(
    report: FinalReport,
    *,
    model_path: str,
    video_name: str,
    analiz_modu: str,
    prob_col: str,
) -> str:
    lines = [
        "# Yangın video analiz özeti",
        "",
        f"- **Video:** `{video_name}`",
        f"- **Analiz modu:** {analiz_modu}",
        f"- **Model dosyası:** `{model_path}`",
        "",
        "## Genel karar",
        "",
        f"**{report.verdict_tr}**",
        "",
        "## Sayılar",
        "",
        f"- En yüksek yangın olasılığı: **{report.max_prob:.1%}**",
        f"- Ortalama yangın olasılığı: **{report.mean_prob:.1%}**",
        f"- Analiz edilen örnek kare sayısı: **{report.frames_analyzed}**",
        f"- Alarm eşiği (yüksek uyarı): **{report.alarm_esigi:.3f}**",
        f"- İnceleme eşiği: **{report.inceleme_esigi:.3f}**",
        f"- Kullanılan olasılık sütunu: `{prob_col}`",
        "",
        "## Yüksek uyarı zaman aralıkları (yaklaşık)",
        "",
    ]
    if report.alarm_zaman_araliklari:
        for a, b in report.alarm_zaman_araliklari:
            lines.append(f"- {a:.2f} s – {b:.2f} s")
    else:
        lines.append("- Kayıt yok (eşik üstü sürekli segment bulunamadı).")
    lines.extend(["", "## Güvenilirlik notu", "", report.guvenilirlik_notu, ""])
    return "\n".join(lines)


def zip_suspicious_frames(
    rgb_video_path: str,
    df: pd.DataFrame,
    *,
    prob_col: str,
    review_thr: float,
    max_frames: int = 120,
) -> bytes | None:
    """Encode JPGs for frames with prob >= review_thr. Returns zip bytes."""
    if not rgb_video_path or not Path(rgb_video_path).exists():
        return None
    if df.empty or prob_col not in df.columns or "frame_idx" not in df.columns:
        return None
    sub = df.copy()
    sub["_p"] = pd.to_numeric(sub[prob_col], errors="coerce").fillna(0.0)
    sub = sub[sub["_p"] >= float(review_thr)].sort_values("_p", ascending=False)
    if sub.empty:
        return None
    sub = sub.head(int(max_frames))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for _, row in sub.iterrows():
            fi = int(row["frame_idx"])
            cap = cv2.VideoCapture(rgb_video_path)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            cap.release()
            if not ok or fr is None:
                continue
            ok2, enc = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok2:
                continue
            name = f"frame_{fi:06d}_p{float(row['_p']):.3f}.jpg"
            zf.writestr(name, enc.tobytes())
    data = buf.getvalue()
    return data if len(data) > 100 else None
