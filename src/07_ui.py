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
            # keep need_cam false to allow fp16 (video.py logic):
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
    # Guard: fusion checkpoint requires thermal input. If thermal is not provided,
    # auto-switch to RGB checkpoint to avoid 0-frame processing.
    ckpt_eff = ckpt_path
    if not th_path:
        try:
            from config import CKPT_RGB as _CKPT_RGB  # type: ignore

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
    }


tab_infer, tab_review, tab_metrics, tab_eval = st.tabs(
    ["Hızlı Test (Video yükle → çalıştır)", "İnceleme (CSV ile)", "Model Metrikleri", "Video Eval (batch)"]
)

with tab_infer:
    st.subheader("Video yükle → inference + risk + event → sonuç")
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
        st.caption("Çalıştırınca özet + timeline + event listesi + frame panel burada çıkacak.")

    if run_btn and up_rgb is not None:
        rgb_path = _save_upload(up_rgb)
        th_path = _save_upload(up_th) if up_th is not None else None
        out_dir = Path(out_base) / _format_run_id()

        # Quick diagnostics for uploaded temp videos
        st.caption("Video diagnostics (uploaded temp files)")
        st.json(
            {
                "rgb": {"path": rgb_path, **_video_info(rgb_path)},
                "thermal": ({"path": th_path, **_video_info(th_path)} if th_path else None),
            }
        )

        with st.status("Çalışıyor…", expanded=True) as status:
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

            st.subheader("Özet")
            m1, m2, m3, m4 = st.columns(4)
            prob_col = "decision_prob" if "decision_prob" in df_scored.columns else "prob_fire"
            m1.metric("Max prob", f"{float(df_scored[prob_col].max()):.3f}")
            m2.metric("Max risk(norm)", f"{float(pd.to_numeric(df_scored.get('risk_score_norm', 0.0), errors='coerce').max() or 0.0):.3f}")
            m3.metric("Confirmed frames", f"{int((df_scored.get('alarm_state','').astype(str)=='confirmed').sum()) if 'alarm_state' in df_scored.columns else 0}")
            m4.metric("Events", f"{len(df_events)}")

            if "frame_idx" in df_scored.columns:
                plot_df = df_scored.sort_values("frame_idx").copy()
                plot_df["prob"] = pd.to_numeric(plot_df.get(prob_col, 0.0), errors="coerce").fillna(0.0)
                plot_df["risk"] = pd.to_numeric(plot_df.get("risk_score_norm", 0.0), errors="coerce").fillna(0.0)
                st.subheader("Timeline")
                st.line_chart(plot_df.set_index("frame_idx")[["prob", "risk"]], height=220)

            st.subheader("🔥 Fire dediği frameler")
            if "pred_fire" in df_scored.columns and "frame_idx" in df_scored.columns:
                df_fire = df_scored[df_scored["pred_fire"].astype(int) == 1].copy()
                if df_fire.empty:
                    st.info("Bu videoda `pred_fire=1` olan frame yok.")
                else:
                    df_fire = df_fire.sort_values(prob_col, ascending=False)
                    show_n = st.slider("Gösterilecek fire frame sayısı", 5, 200, 30, 5)
                    cols = [c for c in ["frame_idx", prob_col, "threshold_used", "alarm_state", "scene_changed"] if c in df_fire.columns]
                    st.dataframe(df_fire[cols].head(int(show_n)), use_container_width=True)
                    top_frames = df_fire["frame_idx"].astype(int).head(int(show_n)).tolist()
                    pick = st.selectbox("Göster (frame_idx)", options=top_frames, index=0, key="pick_fire_frame")
                    fr2 = _read_frame_rgb(rgb_path, int(pick))
                    if fr2 is not None:
                        st.image(fr2, use_container_width=True)
                    rr = _nearest_row_by_frame(df_scored, int(pick))
                    st.json(rr.to_dict()) if rr is not None else None
            else:
                st.info("CSV içinde `pred_fire` veya `frame_idx` yok.")

            st.subheader("Event listesi")
            if df_events.empty:
                st.info("Event yok.")
            else:
                st.dataframe(df_events, use_container_width=True)

            st.subheader("Frame panel")
            info = _video_info(rgb_path)
            max_frame = int(info.get("frame_count") or 0) - 1
            max_frame = max(0, max_frame)
            frame_sel = st.slider("Frame", 0, max_frame, 0, 1)
            a, b = st.columns([1, 1])
            with a:
                fr = _read_frame_rgb(rgb_path, frame_sel)
                st.image(fr, use_container_width=True) if fr is not None else st.warning("Frame okunamadı.")
            with b:
                r = _nearest_row_by_frame(df_scored, frame_sel)
                st.json(r.to_dict()) if r is not None else st.info("Satır bulunamadı.")

            with st.expander("Çıktı dosyaları"):
                st.write({k: result[k] for k in ["pred_csv", "scored_csv", "events_csv", "benchmark_json"]})


with tab_review:
    st.subheader("Var olan CSV ile inceleme")
    csv_path = st.text_input("CSV path", "outputs/video_predictions_scored.csv", key="review_csv")
    if not os.path.exists(csv_path):
        st.warning("CSV bulunamadı. Üstteki 'Hızlı Test' sekmesinden üretebilir veya CLI ile üretebilirsin.")
    else:
        df = pd.read_csv(csv_path)
        st.caption(f"Satır sayısı: **{len(df)}**")

        st.subheader("Tüm frame listesi (sayfalı)")
        rgb_for_preview = st.text_input(
            "RGB video path (opsiyonel: frame görüntülemek için)",
            "",
            key="review_rgb_path",
        )

        prob_col2 = "decision_prob" if "decision_prob" in df.columns else ("prob_fire" if "prob_fire" in df.columns else None)
        if prob_col2 is None:
            prob_col2 = "prob_fire"

        # Filters
        filt = st.selectbox("Filtre", ["all", "only fire (pred_fire=1)", "only no_fire (pred_fire=0)"], index=0)
        df_view = df.copy()
        if "pred_fire" in df_view.columns and filt != "all":
            want = 1 if "pred_fire=1" in filt else 0
            df_view = df_view[df_view["pred_fire"].astype(int) == int(want)]

        # Sorting
        sort_key = st.selectbox("Sırala", ["frame_idx", prob_col2], index=0)
        asc = sort_key == "frame_idx"
        if sort_key in df_view.columns:
            df_view = df_view.sort_values(sort_key, ascending=asc)

        # Pagination
        page_size = st.selectbox("Sayfa boyutu", [50, 100, 200, 500], index=2)
        n_pages = max(1, int((len(df_view) + int(page_size) - 1) / int(page_size)))
        page = st.number_input("Sayfa", min_value=1, max_value=n_pages, value=1, step=1)
        start = (int(page) - 1) * int(page_size)
        end = min(len(df_view), start + int(page_size))
        show_cols = [c for c in ["frame_idx", prob_col2, "pred_fire", "threshold_used", "alarm_state", "scene_changed"] if c in df_view.columns]
        if not show_cols:
            show_cols = list(df_view.columns[:12])
        st.dataframe(df_view.iloc[start:end][show_cols], use_container_width=True)

        # Preview a selected frame_idx
        if "frame_idx" in df_view.columns:
            st.subheader("Frame önizleme")
            pick_idx = st.number_input("Gösterilecek frame_idx", min_value=0, value=int(df_view["frame_idx"].iloc[start]) if len(df_view) else 0, step=1)
            rr2 = _nearest_row_by_frame(df_view, int(pick_idx))
            if rr2 is not None:
                st.json(rr2.to_dict())
            if rgb_for_preview and os.path.exists(rgb_for_preview):
                frp = _read_frame_rgb(rgb_for_preview, int(pick_idx))
                st.image(frp, use_container_width=True) if frp is not None else st.warning("Frame okunamadı.")

        if "pred_fire" in df.columns and "frame_idx" in df.columns:
            st.subheader("🔥 Fire dediği frameler (CSV)")
            df_fire2 = df[df["pred_fire"].astype(int) == 1].copy()
            if df_fire2.empty:
                st.info("Bu CSV’de `pred_fire=1` yok.")
            else:
                df_fire2 = df_fire2.sort_values(prob_col2, ascending=False)
                cols2 = [c for c in ["frame_idx", prob_col2, "threshold_used", "alarm_state"] if c in df_fire2.columns]
                st.dataframe(df_fire2[cols2].head(50), use_container_width=True)


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
        st.dataframe(pd.DataFrame(rows), use_container_width=True) if rows else None
        st.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json")


with tab_eval:
    st.subheader("Batch video evaluation (eval_summary.csv)")
    eval_csv = st.text_input("Eval summary CSV", "outputs/eval_summary.csv", key="eval_summary")
    if not os.path.exists(eval_csv):
        st.warning("`outputs/eval_summary.csv` yok. Üretmek için: python src/eval/run_evaluation.py --videos_dir <klasör> --profile balanced")
    else:
        edf = pd.read_csv(eval_csv)
        st.dataframe(edf, use_container_width=True)
