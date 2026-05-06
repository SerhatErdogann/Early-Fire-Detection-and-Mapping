"""Streamlit review UI for the wildfire detection pipeline.

Hızlı Test sekmesi gerçek tahmin akışını sade tutar (verdict card + özet);
diagnostics, raw CSV, frame paneli gibi geliştirici bilgileri ``st.expander``
içine kapatılır. Diğer sekmeler (İnceleme, Metrikler, Batch Eval) salt-okunur
analiz amacıyla korunmuştur. Noise / robustness testleri ayrı bir CLI
modülünde (``src/eval/robustness_eval.py``); UI'da görünmez.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.event_extractor import extract_events  # noqa: E402
from src.inference.video import run_video_inference  # noqa: E402
from src.risk.scoring import build_risk_table  # noqa: E402

try:  # noqa: E402
    from config import CKPT_FUSION, CKPT_RGB, INFERENCE_DEFAULT, OUTPUTS_DIR, RISK_SCORE_WEIGHTS
except Exception:  # pragma: no cover
    CKPT_FUSION = Path("models/fusion.pt")
    CKPT_RGB = Path("models/rgb.pt")
    INFERENCE_DEFAULT = {}
    OUTPUTS_DIR = Path("outputs")
    RISK_SCORE_WEIGHTS = {}


st.set_page_config(page_title="Fire Risk Review", layout="wide")
st.title("Fire Risk Review")


@dataclass(frozen=True)
class InferPreset:
    key: str
    title: str
    description: str
    args: dict[str, Any]


PRESETS: list[InferPreset] = [
    InferPreset(
        key="fast",
        title="Hızlı test",
        description="Küçük/orta videoda hızlı sonuç (FP16 + seyrek örnekleme).",
        args={
            "size": 224,
            "step": 8,
            "smooth_win": 5,
            "ema_alpha": 0.25,
            "tta": False,
            "fp16": True,
            "adaptive_step": True,
            "temporal_guard": True,
            "min_component_area": 0.0,
            "texture_prob_max": 0.0,
            "small_fire_boost": 1.0,
            "growth_upscale": 1.0,
        },
    ),
    InferPreset(
        key="balanced",
        title="Dengeli",
        description="Günlük kullanım (EMA + TTA).",
        args={
            "size": 224,
            "step": 6,
            "smooth_win": 7,
            "ema_alpha": 0.30,
            "tta": True,
            "fp16": True,
            "adaptive_step": True,
            "temporal_guard": True,
            "min_component_area": float(INFERENCE_DEFAULT.get("min_component_area", 0.01) or 0.01),
        },
    ),
    InferPreset(
        key="safe",
        title="Kaliteli analiz",
        description="Daha büyük giriş + daha sık örnekleme (daha yavaş).",
        args={
            "size": 384,
            "step": 4,
            "smooth_win": 9,
            "ema_alpha": 0.35,
            "tta": True,
            "fp16": False,
            "adaptive_step": True,
            "temporal_guard": True,
            "min_component_area": float(INFERENCE_DEFAULT.get("min_component_area", 0.01) or 0.01),
        },
    ),
]


@st.cache_data(show_spinner=False)
def _video_info(video_path: str) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "error": "Video açılamadı."}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {"ok": True, "fps": fps, "frame_count": frame_count}


@st.cache_data(show_spinner=False)
def _read_frame_rgb(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, frame_idx)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _nearest_row_by_frame(df_: pd.DataFrame, frame_idx: int):
    if df_.empty or "frame_idx" not in df_.columns:
        return None
    ff = pd.to_numeric(df_["frame_idx"], errors="coerce")
    if ff.isna().all():
        return None
    idx_near = (ff - float(frame_idx)).abs().idxmin()
    return df_.loc[idx_near]


def _format_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _save_upload(upload) -> str:
    suffix = Path(upload.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name


def _run_inference(
    rgb_path: str,
    th_path: str | None,
    preset: InferPreset,
    ckpt_path: str,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / "video_predictions.csv"
    bench_json = out_dir / "video_predictions.benchmark.json"

    a = preset.args
    # Fusion checkpoint requires thermal input; auto-fallback to RGB checkpoint
    # when the user did not upload a thermal video.
    ckpt_eff = ckpt_path
    if not th_path:
        try:
            from config import CKPT_RGB as _CKPT_RGB

            ckpt_eff = str(_CKPT_RGB)
        except Exception:
            ckpt_eff = "models/rgb.pt"

    out_csv = run_video_inference(
        rgb_path,
        th_video_path=th_path,
        ckpt_path=ckpt_eff,
        mode="fusion" if th_path else "rgb",
        size=int(a.get("size", 224)),
        step_frames=int(a.get("step", 6)),
        smooth_window=int(a.get("smooth_win", 7)),
        ema_alpha=float(a.get("ema_alpha", 0.30)),
        use_tta=bool(a.get("tta", False)),
        out_csv=str(pred_csv),
        use_fp16=bool(a.get("fp16", False)),
        temporal_guard=bool(a.get("temporal_guard", True)),
        adaptive_step=bool(a.get("adaptive_step", True)),
        min_component_area=float(a.get("min_component_area", 0.01)),
        texture_prob_max=float(a.get("texture_prob_max", INFERENCE_DEFAULT.get("texture_prob_max", 0.2))),
        small_fire_boost=float(a.get("small_fire_boost", INFERENCE_DEFAULT.get("small_fire_boost", 1.3))),
        growth_upscale=float(a.get("growth_upscale", INFERENCE_DEFAULT.get("growth_upscale", 1.2))),
        benchmark=True,
        benchmark_out=str(bench_json),
    )
    try:
        df_pred = pd.read_csv(out_csv)
    except Exception as e:
        raise RuntimeError(
            "Inference çıktısı okunamadı (CSV boş veya parse edilemedi). "
            "Video decode başarısız olabilir. Farklı bir MP4 (H.264) ile tekrar deneyin."
        ) from e
    if df_pred.empty:
        raise RuntimeError(
            "Inference hiçbir frame işleyemedi (çıktı CSV boş). "
            "Muhtemel nedenler: codec uyumsuzluğu, bozuk video, thermal video ile senkron/fps uyuşmazlığı. "
            "Öneri: videoyu MP4 (H.264) olarak yeniden encode edip tekrar deneyin."
        )
    thr_used = float(pd.to_numeric(df_pred.get("threshold_used", 0.5), errors="coerce").dropna().median()) if len(df_pred) else 0.5
    scored, _meta = build_risk_table(
        df_pred.sort_values("frame_idx").reset_index(drop=True),
        risk_weights={k: float(v) for k, v in dict(RISK_SCORE_WEIGHTS).items()},
        persistence_win=7,
        persistence_thr=thr_used,
    )
    events_df = extract_events(scored)
    scored_csv = out_dir / "video_predictions_scored.csv"
    events_csv = out_dir / "events.csv"
    scored.to_csv(scored_csv, index=False)
    events_df.to_csv(events_csv, index=False)
    return {
        "pred_csv": str(pred_csv),
        "scored_csv": str(scored_csv),
        "events_csv": str(events_csv),
        "benchmark_json": str(bench_json),
        "df_scored": scored,
        "df_events": events_df,
        "threshold_used": thr_used,
    }


def _summarize_result(df_scored: pd.DataFrame, df_events: pd.DataFrame, threshold_used: float) -> dict[str, Any]:
    """Compact summary used by the verdict card."""
    prob_col = "decision_prob" if "decision_prob" in df_scored.columns else "prob_fire"
    probs = pd.to_numeric(df_scored.get(prob_col, 0.0), errors="coerce").fillna(0.0)
    fire_mask = (
        df_scored["pred_fire"].astype(int) == 1
        if "pred_fire" in df_scored.columns
        else probs >= float(threshold_used)
    )
    fire_frames = int(fire_mask.sum())
    n_total = int(len(df_scored))

    confirmed = 0
    if "alarm_state" in df_scored.columns:
        confirmed = int((df_scored["alarm_state"].astype(str) == "confirmed").sum())

    first_detect_sec = None
    first_detect_frame = None
    if fire_frames > 0 and "frame_idx" in df_scored.columns:
        first_row = df_scored[fire_mask].sort_values("frame_idx").iloc[0]
        first_detect_frame = int(first_row["frame_idx"])
        if "timestamp_sec" in df_scored.columns:
            ts = pd.to_numeric(first_row.get("timestamp_sec"), errors="coerce")
            if pd.notna(ts):
                first_detect_sec = float(ts)

    risk_band = None
    if "confidence_band" in df_scored.columns:
        bands = df_scored["confidence_band"].astype(str)
        nonempty = bands[bands.str.len() > 0]
        if len(nonempty):
            risk_band = nonempty.mode().iloc[0] if not nonempty.mode().empty else nonempty.iloc[-1]

    return {
        "fire_detected": fire_frames > 0 or len(df_events) > 0,
        "fire_frames": fire_frames,
        "total_frames": n_total,
        "fire_ratio": (fire_frames / n_total) if n_total else 0.0,
        "max_prob": float(probs.max()) if len(probs) else 0.0,
        "mean_prob": float(probs.mean()) if len(probs) else 0.0,
        "confirmed_frames": confirmed,
        "n_events": int(len(df_events)),
        "first_detect_frame": first_detect_frame,
        "first_detect_sec": first_detect_sec,
        "threshold": float(threshold_used),
        "risk_band": risk_band,
        "prob_col": prob_col,
    }


def _show_row_metrics(r: pd.Series) -> None:
    """Compact, human-readable metric block for a single CSV row."""
    cols = st.columns(4)
    cols[0].metric("frame_idx", int(r.get("frame_idx", -1)))
    if "timestamp_sec" in r.index and pd.notna(r.get("timestamp_sec")):
        cols[1].metric("zaman (s)", f"{float(r['timestamp_sec']):.2f}")
    prob_key = "decision_prob" if "decision_prob" in r.index else "prob_fire"
    if prob_key in r.index and pd.notna(r.get(prob_key)):
        cols[2].metric("prob_fire", f"{float(r[prob_key]):.3f}")
    if "alarm_state" in r.index:
        cols[3].metric("alarm", str(r.get("alarm_state", "—")))
    extras = {k: r[k] for k in ("risk_score_norm", "confidence_band", "scene_changed") if k in r.index}
    if extras:
        st.caption(", ".join(f"{k}={v}" for k, v in extras.items()))


def _render_verdict(summary: dict[str, Any]) -> None:
    fire = bool(summary.get("fire_detected"))
    max_prob = float(summary.get("max_prob", 0.0))
    n_events = int(summary.get("n_events", 0))
    fire_frames = int(summary.get("fire_frames", 0))
    thr = float(summary.get("threshold", 0.5))
    band = summary.get("risk_band") or ("yüksek" if max_prob >= 0.85 else "orta" if max_prob >= 0.55 else "düşük")

    if fire:
        st.error(
            f"### Yangın tespit edildi\n"
            f"- En yüksek güven: **{max_prob:.2%}** (eşik {thr:.2f})\n"
            f"- Yangın olarak işaretlenen kare sayısı: **{fire_frames}**\n"
            f"- Tespit edilen olay (event) sayısı: **{n_events}**\n"
            f"- Risk bandı: **{band}**"
        )
    else:
        st.success(
            f"### Yangın tespit edilmedi\n"
            f"- En yüksek güven: **{max_prob:.2%}** (eşik {thr:.2f}, altında kaldı)\n"
            f"- Risk bandı: **{band}**"
        )


tab_infer, tab_review, tab_metrics, tab_eval = st.tabs(
    ["Hızlı Test", "İnceleme (CSV)", "Model Metrikleri", "Video Eval (batch)"]
)

with tab_infer:
    st.subheader("Video yükle → tahmin → özet")
    c1, c2 = st.columns([1, 1])
    with c1:
        up_rgb = st.file_uploader("RGB video", type=["mp4", "avi", "mov", "mkv", "webm"], key="up_rgb")
        up_th = st.file_uploader("Thermal video (opsiyonel)", type=["mp4", "avi", "mov", "mkv", "webm"], key="up_th")
        preset_key = st.radio("Mod", [p.key for p in PRESETS], index=1, horizontal=True)
        preset = next(p for p in PRESETS if p.key == preset_key)
        st.caption(f"**{preset.title}** — {preset.description}")
        ckpt_choice = st.selectbox("Checkpoint", options=[str(CKPT_FUSION), str(CKPT_RGB)], index=0)
        out_base = st.text_input("Çıktı klasörü", str(OUTPUTS_DIR / "ui_runs"))
        run_btn = st.button("▶ Çalıştır", type="primary", disabled=(up_rgb is None))

    with c2:
        st.caption(
            "Sonuç: yangın var/yok kartı + özet metrikler. "
            "Detaylı tablolar, ham JSON ve frame panel aşağıdaki "
            "açılır panellerin içinde gizlidir."
        )

    if run_btn and up_rgb is not None:
        rgb_path = _save_upload(up_rgb)
        th_path = _save_upload(up_th) if up_th is not None else None
        out_dir = Path(out_base) / _format_run_id()

        with st.status("Çalışıyor…", expanded=False) as status:
            t0 = time.perf_counter()
            try:
                result = _run_inference(rgb_path, th_path, preset, ckpt_choice, out_dir)
                dt = time.perf_counter() - t0
                status.update(label=f"Tamamlandı ({dt:.1f}s)", state="complete")
            except Exception as e:
                status.update(label="Hata", state="error")
                st.exception(e)
                result = None

        if result is not None:
            df_scored = result["df_scored"]
            df_events = result["df_events"]
            summary = _summarize_result(df_scored, df_events, result["threshold_used"])

            _render_verdict(summary)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Max güven", f"{summary['max_prob']:.2%}")
            m2.metric("Yangın frame", f"{summary['fire_frames']}/{summary['total_frames']}")
            m3.metric("Event", f"{summary['n_events']}")
            if summary["first_detect_sec"] is not None:
                m4.metric("İlk tespit", f"{summary['first_detect_sec']:.1f}s")
            elif summary["first_detect_frame"] is not None:
                m4.metric("İlk tespit", f"frame {summary['first_detect_frame']}")
            else:
                m4.metric("İlk tespit", "—")

            if "frame_idx" in df_scored.columns:
                st.caption("Olasılık ve risk skorunun zaman serisi:")
                plot_df = df_scored.sort_values("frame_idx").copy()
                plot_df["prob"] = pd.to_numeric(plot_df.get(summary["prob_col"], 0.0), errors="coerce").fillna(0.0)
                plot_df["risk"] = pd.to_numeric(plot_df.get("risk_score_norm", 0.0), errors="coerce").fillna(0.0)
                st.line_chart(plot_df.set_index("frame_idx")[["prob", "risk"]], height=200)

            with st.expander("📊 Detaylı analiz (event listesi, fire frame tablosu, frame panel)", expanded=False):
                if df_events.empty:
                    st.caption("Event yok.")
                else:
                    st.markdown("**Event listesi**")
                    st.dataframe(df_events, use_container_width=True)

                if "pred_fire" in df_scored.columns and "frame_idx" in df_scored.columns:
                    st.markdown("**🔥 Yangın olarak işaretlenen kareler**")
                    df_fire = df_scored[df_scored["pred_fire"].astype(int) == 1].copy()
                    if df_fire.empty:
                        st.caption("Bu videoda `pred_fire=1` olan kare yok.")
                    else:
                        df_fire = df_fire.sort_values(summary["prob_col"], ascending=False)
                        show_n = st.slider("Gösterilecek fire frame sayısı", 5, 200, 30, 5)
                        cols = [
                            c
                            for c in [
                                "frame_idx",
                                "timestamp_sec",
                                summary["prob_col"],
                                "threshold_used",
                                "alarm_state",
                                "scene_changed",
                            ]
                            if c in df_fire.columns
                        ]
                        st.dataframe(df_fire[cols].head(int(show_n)), use_container_width=True)

                        top_frames = df_fire["frame_idx"].astype(int).head(int(show_n)).tolist()
                        if top_frames:
                            pick = st.selectbox("Göster (frame_idx)", options=top_frames, index=0, key="pick_fire_frame")
                            fr2 = _read_frame_rgb(rgb_path, int(pick))
                            if fr2 is not None:
                                st.image(fr2, use_container_width=True)
                            rr = _nearest_row_by_frame(df_scored, int(pick))
                            if rr is not None:
                                _show_row_metrics(rr)

                st.markdown("**Manuel frame tarayıcı**")
                info = _video_info(rgb_path)
                max_frame = max(0, int(info.get("frame_count") or 0) - 1)
                frame_sel = st.slider("Frame", 0, max_frame, 0, 1, key="fr_browse")
                a, b = st.columns([1, 1])
                with a:
                    fr = _read_frame_rgb(rgb_path, frame_sel)
                    if fr is not None:
                        st.image(fr, use_container_width=True)
                    else:
                        st.warning("Frame okunamadı.")
                with b:
                    r = _nearest_row_by_frame(df_scored, frame_sel)
                    if r is not None:
                        _show_row_metrics(r)
                    else:
                        st.info("Satır bulunamadı.")

            with st.expander("🛠️ Geliştirici / Debug bilgileri (yüklenen video, CSV yolları, ham JSON)", expanded=False):
                st.markdown("**Yüklenen video diagnostics**")
                st.json(
                    {
                        "rgb": {"path": rgb_path, **_video_info(rgb_path)},
                        "thermal": ({"path": th_path, **_video_info(th_path)} if th_path else None),
                        "checkpoint": ckpt_choice,
                        "preset": preset.key,
                        "preset_args": preset.args,
                    }
                )
                st.markdown("**Çıktı dosyaları**")
                st.write({k: result[k] for k in ["pred_csv", "scored_csv", "events_csv", "benchmark_json"]})
                st.markdown("**Özet (raw)**")
                st.json(summary)


with tab_review:
    st.subheader("Var olan CSV ile inceleme")
    csv_path = st.text_input("CSV path", "outputs/video_predictions_scored.csv", key="review_csv")
    if not os.path.exists(csv_path):
        st.warning("CSV bulunamadı. Üstteki 'Hızlı Test' sekmesinden üretebilir veya CLI ile üretebilirsin.")
    else:
        df = pd.read_csv(csv_path)
        st.caption(f"Satır sayısı: **{len(df)}**")

        rgb_for_preview = st.text_input(
            "RGB video path (opsiyonel: frame görüntülemek için)",
            "",
            key="review_rgb_path",
        )

        prob_col2 = "decision_prob" if "decision_prob" in df.columns else ("prob_fire" if "prob_fire" in df.columns else "prob_fire")

        filt = st.selectbox("Filtre", ["all", "only fire (pred_fire=1)", "only no_fire (pred_fire=0)"], index=0)
        df_view = df.copy()
        if "pred_fire" in df_view.columns and filt != "all":
            want = 1 if "pred_fire=1" in filt else 0
            df_view = df_view[df_view["pred_fire"].astype(int) == int(want)]

        sort_key = st.selectbox("Sırala", ["frame_idx", prob_col2], index=0)
        asc = sort_key == "frame_idx"
        if sort_key in df_view.columns:
            df_view = df_view.sort_values(sort_key, ascending=asc)

        page_size = st.selectbox("Sayfa boyutu", [50, 100, 200, 500], index=2)
        n_pages = max(1, int((len(df_view) + int(page_size) - 1) / int(page_size)))
        page = st.number_input("Sayfa", min_value=1, max_value=n_pages, value=1, step=1)
        start = (int(page) - 1) * int(page_size)
        end = min(len(df_view), start + int(page_size))
        show_cols = [c for c in ["frame_idx", prob_col2, "pred_fire", "threshold_used", "alarm_state", "scene_changed"] if c in df_view.columns]
        if not show_cols:
            show_cols = list(df_view.columns[:12])
        st.dataframe(df_view.iloc[start:end][show_cols], use_container_width=True)

        if "frame_idx" in df_view.columns and len(df_view):
            st.subheader("Frame önizleme")
            pick_idx = st.number_input(
                "Gösterilecek frame_idx",
                min_value=0,
                value=int(df_view["frame_idx"].iloc[start]),
                step=1,
            )
            rr2 = _nearest_row_by_frame(df_view, int(pick_idx))
            if rr2 is not None:
                _show_row_metrics(rr2)
            if rgb_for_preview and os.path.exists(rgb_for_preview):
                frp = _read_frame_rgb(rgb_for_preview, int(pick_idx))
                if frp is not None:
                    st.image(frp, use_container_width=True)
                else:
                    st.warning("Frame okunamadı.")


with tab_metrics:
    st.subheader("Eğitim/Test metrikleri (outputs/metrics_*.json)")
    outputs_dir = st.text_input("Outputs klasörü", "outputs", key="metrics_dir")
    p = Path(outputs_dir)
    metric_files = sorted([x for x in p.glob("metrics_*.json")]) if p.exists() else []
    if not metric_files:
        st.warning("`outputs/metrics_*.json` bulunamadı. Eğitim sonrası oluşur (python src/02_train.py ...).")
    else:
        selected = st.selectbox("Metrik dosyası seç", [str(x) for x in metric_files], key="metrics_file")
        payload = json.loads(Path(selected).read_text(encoding="utf-8"))
        rows = []
        for split in ("val", "test"):
            d = payload.get(split, {})
            if isinstance(d, dict):
                rows.append(
                    {
                        "split": split,
                        "acc": d.get("acc"),
                        "auc": d.get("auc"),
                        "ap": d.get("ap"),
                        "precision": d.get("precision"),
                        "recall": d.get("recall"),
                        "f1": d.get("f1"),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        with st.expander("Ham metrik JSON", expanded=False):
            st.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json")


with tab_eval:
    st.subheader("Batch video evaluation (eval_summary.csv)")
    eval_csv = st.text_input("Eval summary CSV", "outputs/eval_summary.csv", key="eval_summary")
    if not os.path.exists(eval_csv):
        st.warning("`outputs/eval_summary.csv` yok. Üretmek için: python src/eval/run_evaluation.py --videos_dir <klasör> --profile balanced")
    else:
        edf = pd.read_csv(eval_csv)
        st.dataframe(edf, use_container_width=True)
