"""Main Streamlit entry: professional Turkish fire video analysis dashboard."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.components import render_metric_row
from src.ui.constants import DEFAULT_INFER_UI_ARGS
from src.ui.display_format import cap_probability_for_chart
from src.ui.inference_runner import run_analysis_pipeline
from src.ui.reporting import (
    FinalReport,
    build_final_report,
    build_markdown_report,
    dataframe_to_csv_bytes,
    zip_suspicious_frames,
)
from src.ui.result_panel import render_live_panel, turkish_caption_for_row
from src.ui.styles import inject_global_styles
from src.ui.video_helpers import nearest_row_by_frame
from src.ui.video_panel import render_frame_cards, render_preview_with_frame_arrows

try:
    from config import (
        CKPT_DUAL_BRANCH,
        CKPT_FUSION,
        CKPT_RGB,
        MODELS_DIR,
        OUTPUTS_DIR,
    )
except Exception:
    CKPT_DUAL_BRANCH = Path("models/dual_branch.pt")
    CKPT_FUSION = Path("models/fusion.pt")
    CKPT_RGB = Path("models/rgb.pt")
    MODELS_DIR = Path("models")
    OUTPUTS_DIR = Path("outputs")


def _checkpoint_options() -> list[str]:
    seen: list[str] = []

    def _add(p: Path) -> None:
        s = str(p)
        if s not in seen:
            seen.append(s)

    for cand in (CKPT_DUAL_BRANCH, CKPT_FUSION, CKPT_RGB):
        if Path(cand).is_file():
            _add(cand)
    try:
        for p in sorted(Path(MODELS_DIR).glob("*.pt")):
            if p.is_file():
                _add(p)
    except Exception:
        pass
    if not seen:
        for cand in (CKPT_DUAL_BRANCH, CKPT_FUSION, CKPT_RGB):
            _add(cand)
    return seen


def _format_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _save_upload(upload) -> str:
    suffix = Path(upload.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name


def _pick_prob_col(df: pd.DataFrame) -> str:
    return "decision_prob" if "decision_prob" in df.columns else "prob_fire"


def _default_sample_dir() -> Path:
    return PROJECT_ROOT / "samples"


def _sample_video_options() -> list[str]:
    d = _default_sample_dir()
    if not d.is_dir():
        return []
    return sorted([str(p) for p in d.glob("*.mp4")] + [str(p) for p in d.glob("*.avi")])


def _render_home(
    *,
    ckpt_options: list[str],
    sample_videos: list[str],
) -> tuple[bool, dict[str, Any] | None]:
    """Draw landing inputs. Returns (should_run, run_config dict or None)."""
    st.markdown('<div class="fire-hero">', unsafe_allow_html=True)
    st.markdown(
        '<h1 class="fire-main-title">Yangın Erken Tespit Sistemi</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Video veya kamera görüntüsünden **yangın olasılığını** analiz eder. "
        "Akışlar için sunucuda **RTSP** veya **Yerel dosya yolu** kullanılır; "
        "tarayıcıdan **dosya yükleme** de desteklenir."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Giriş")

    up_rgb = st.file_uploader(
        "Video dosyası yükle",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Kısa videolar için uygun. Çok büyük dosyalarda yerel dosya yolu önerilir.",
    )
    rgb_path_input = st.text_input(
        "Yerel video dosya yolu veya RTSP/HTTP adresi",
        "",
        placeholder=r"C:\Videos\fire.mp4 veya rtsp://...",
        help="Büyük dosyalarda tüm videoyu belleğe almadan okuma için önerilir.",
    )
    use_sample = None
    if sample_videos:
        use_sample = st.selectbox(
            "Örnek video (isteğe bağlı)",
            options=["(kullanma)"] + sample_videos,
            index=0,
        )

    up_th = st.file_uploader("Termal video (isteğe bağlı)", type=["mp4", "avi", "mov", "mkv", "webm"])
    th_path_input = st.text_input("Termal: yerel path veya URI (isteğe bağlı)", "")

    ckpt_choice = st.selectbox("Hazır kalibre model seçin", options=ckpt_options, index=0)
    if not Path(ckpt_choice).is_file():
        st.warning(
            f"Seçilen dosya bulunamadı: `{ckpt_choice}`. "
            f"Eğitim çıktısını `{MODELS_DIR}` içine koyun."
        )

    out_base = st.text_input("Çalışma çıktı klasörü", str(OUTPUTS_DIR / "ui_runs"))

    col_a, col_b = st.columns(2)
    with col_a:
        run_btn = st.button("Analizi başlat", type="primary", use_container_width=True)
    with col_b:
        if st.button("Formu temizle", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k.startswith("fire_"):
                    del st.session_state[k]
            st.rerun()

    if run_btn:
        if use_sample and use_sample != "(kullanma)":
            rgb_path = use_sample
        elif rgb_path_input.strip():
            rgb_path = rgb_path_input.strip()
        elif up_rgb is not None:
            rgb_path = _save_upload(up_rgb)
        else:
            st.error("RGB video gerekli: yükleyin, yerel path/URL girin veya örnek seçin.")
            return False, None

        if th_path_input.strip():
            th_path = th_path_input.strip()
        elif up_th is not None:
            th_path = _save_upload(up_th)
        else:
            th_path = None
        cfg = {
            "rgb_path": rgb_path,
            "th_path": th_path,
            "infer_args": dict(DEFAULT_INFER_UI_ARGS),
            "ckpt": ckpt_choice,
            "out_base": out_base,
        }
        return True, cfg
    return False, None


def _median_frame_spacing_sec(ts: pd.Series) -> float:
    t = pd.to_numeric(ts, errors="coerce").dropna()
    if len(t) < 2:
        return 1.0 / 30.0
    d = t.diff().dropna()
    if d.empty:
        return 1.0 / 30.0
    m = float(d.median())
    return m if m > 1e-9 else 1.0 / 30.0


def _alarm_active_duration_sec(df: pd.DataFrame) -> float:
    """Şüpheli/onaylı süreler için kaba süre (satırların timestamp aralığı ile)."""
    if df.empty or "timestamp_sec" not in df.columns or "alarm_state" not in df.columns:
        return 0.0
    d = df.sort_values("frame_idx").reset_index(drop=True)
    ts = pd.to_numeric(d["timestamp_sec"], errors="coerce")
    dt = _median_frame_spacing_sec(ts)
    active = d["alarm_state"].astype(str).str.lower().isin(["suspected", "confirmed"])
    total = 0.0
    in_alarm = False
    seg_start = 0.0
    last_t = 0.0
    for i in range(len(d)):
        ti = ts.iloc[i]
        if pd.isna(ti):
            continue
        t = float(ti)
        last_t = t
        a = bool(active.iloc[i])
        if a and not in_alarm:
            in_alarm = True
            seg_start = t
        elif (not a) and in_alarm:
            total += max(dt, t - seg_start)
            in_alarm = False
    if in_alarm:
        total += max(dt, last_t - seg_start + dt)
    return float(total)


def _load_benchmark_dict(result: dict[str, Any]) -> dict[str, Any]:
    p = result.get("benchmark_json")
    if not p:
        return {}
    bp = Path(str(p))
    if not bp.is_file():
        return {}
    try:
        return json.loads(bp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _default_preview_frame(events_df: pd.DataFrame | None, df_scored: pd.DataFrame | None) -> int:
    if df_scored is None or df_scored.empty:
        return 0
    try:
        if events_df is not None and not events_df.empty and "peak_frame" in events_df.columns:
            imx = pd.to_numeric(events_df["max_prob"], errors="coerce").fillna(-1.0).idxmax()
            return int(events_df.loc[imx, "peak_frame"])
    except Exception:
        pass
    return int(df_scored.sort_values("frame_idx")["frame_idx"].iloc[-1])


def _risk_level_tr(lv: object) -> str:
    s = str(lv or "").strip().lower()
    return {"confirmed": "Yüksek risk", "suspected": "İnceleme gerekli", "ok": "Güvenli"}.get(s, str(lv))


def _run_with_progress(
    cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    progress = st.progress(0.0, text="Hazırlanıyor…")
    t_started = time.perf_counter()

    def cb(done: int, est: int | None) -> None:
        elapsed_s = int(time.perf_counter() - t_started)
        em = f"{elapsed_s // 60} dk {elapsed_s % 60} sn"
        if est and est > 0:
            denom = max(int(est), int(done))
            r = min(1.0, float(done) / float(denom))
            over = f" (tahmin aşıldı — devam)" if done > est else ""
            txt = f"Tarama: {done} / ~{est} kare işlendi · {em}{over}"
        else:
            r = min(1.0, float(done) / max(5000.0, float(done)))
            txt = f"Tarama ilerliyor: {done} işlenen zaman adımı · geçen {em}"
        progress.progress(min(0.99, r), text=txt)

    res = run_analysis_pipeline(
        cfg["rgb_path"],
        cfg["th_path"],
        cfg["infer_args"],
        cfg["ckpt"],
        out_dir,
        progress_callback=cb,
    )
    progress.progress(1.0, text="Tamamlandı")
    return res


def _render_analysis_dashboard(
    cfg: dict[str, Any],
    result: dict[str, Any],
) -> None:
    df_scored: pd.DataFrame = result["df_scored"]
    df_events: pd.DataFrame = result["df_events"]
    thr_u = float(result["threshold_used"])
    alarm_e = float(result.get("hyst_high_used") or thr_u)
    inceleme_e = float(result.get("hyst_low_used") or thr_u * 0.6)
    prob_col = _pick_prob_col(df_scored)
    report: FinalReport = build_final_report(
        df_scored,
        df_events,
        thr_u,
        alarm_e,
        inceleme_e,
    )

    rgb_path = cfg["rgb_path"]
    exists_or_stream = (
        (rgb_path and os.path.exists(rgb_path))
        or str(rgb_path).lower().startswith("rtsp://")
        or str(rgb_path).lower().startswith("http://")
        or str(rgb_path).lower().startswith("https://")
    )

    st.subheader("Operasyon paneli")
    verdict_line = {"fire": "Yüksek risk modu aktif.", "review": "İnceleme gerekli görünüm.", "safe": "Güvenlik seviyesi normal."}[report.verdict_key]
    st.caption(f"{report.verdict_tr} — {verdict_line}")

    sel_key = "fire_preview_frame_idx"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = int(_default_preview_frame(df_events, df_scored))

    col_vid, col_stat = st.columns([1.2, 1.0])
    with col_vid:
        render_preview_with_frame_arrows(
            rgb_path if exists_or_stream else None,
            df_scored,
            session_key_selected=sel_key,
        )

    raw_fi = int(st.session_state[sel_key])
    row_live = nearest_row_by_frame(df_scored, raw_fi)
    if row_live is None:
        row_live = df_scored.iloc[-1]
    prob_raw = float(row_live[prob_col]) if prob_col in row_live.index else 0.0
    st_al = str(row_live.get("alarm_state", "")) if hasattr(row_live, "get") else ""

    tail = df_scored.sort_values("frame_idx").tail(5)
    slope = None
    if len(tail) > 1 and prob_col in tail.columns:
        tprob = pd.to_numeric(tail[prob_col], errors="coerce")
        slope = float(tprob.diff().mean()) if len(tprob) else None

    cap = turkish_caption_for_row(prob_raw, alarm_e, inceleme_e, slope, st_al or None)

    with col_stat:
        render_live_panel(prob_raw, alarm_e, inceleme_e, st_al or None, cap)

    st.markdown("### Risk — zaman içinde")
    chart_df = df_scored.sort_values("frame_idx").copy()
    ts_col = pd.to_numeric(chart_df["timestamp_sec"], errors="coerce").fillna(0.0)
    chart_df["Zaman_sn"] = ts_col
    raw_probs = pd.to_numeric(chart_df[prob_col], errors="coerce").fillna(0.0)
    chart_df["Gösterilen risk"] = raw_probs.map(cap_probability_for_chart)
    chart_df["Alarm eşiği"] = min(float(alarm_e), cap_probability_for_chart(float(alarm_e)))
    chart_df["İnceleme eşiği"] = min(float(inceleme_e), cap_probability_for_chart(float(inceleme_e)))
    try:
        st.line_chart(
            chart_df.set_index("Zaman_sn")[["Gösterilen risk", "Alarm eşiği", "İnceleme eşiği"]],
            height=300,
        )
    except Exception:
        st.line_chart(chart_df[["Gösterilen risk"]], height=300)

    st.markdown("### Olay özeti")
    if df_events is None or df_events.empty:
        st.info("Operasyon sırasında teyit gerektiren ardışık olay kümesi tespit edilmedi.")
    else:
        ev_disp = pd.DataFrame(
            {
                "Olay no": pd.RangeIndex(start=1, stop=len(df_events) + 1),
                "Başlangıç (sn)": pd.to_numeric(df_events["start_sec"], errors="coerce").map(lambda x: f"{float(x):.1f}" if pd.notna(x) else "—"),
                "Bitiş (sn)": pd.to_numeric(df_events["end_sec"], errors="coerce").map(lambda x: f"{float(x):.1f}" if pd.notna(x) else "—"),
                "Süre (sn)": pd.to_numeric(df_events["duration_sec"], errors="coerce").map(lambda x: f"{float(x):.1f}" if pd.notna(x) else "—"),
                "Maks risk": pd.to_numeric(df_events["max_prob"], errors="coerce").map(
                    lambda p: ">97%"
                    if (pd.notna(p) and float(p) > 0.97)
                    else (f"{100.0 * float(p):.0f}%")
                    if pd.notna(p)
                    else "—"
                ),
                "Ort risk": pd.to_numeric(df_events["avg_prob"], errors="coerce").map(
                    lambda p: (
                        ">97%"
                        if (pd.notna(p) and float(p) > 0.97)
                        else (f"{100.0 * float(p):.0f}%")
                        if pd.notna(p)
                        else "—"
                    )
                ),
                "Durum": df_events["risk_level"].map(_risk_level_tr),
                "Tepe kare": df_events["peak_frame"].astype(str),
            }
        )
        st.dataframe(ev_disp, use_container_width=True)

    st.markdown("### Dışa aktarım")
    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        st.download_button(
            "Skorlu ölçümler (CSV)",
            data=dataframe_to_csv_bytes(df_scored),
            file_name="video_predictions_scored.csv",
            mime="text/csv",
        )
    with d2:
        md = build_markdown_report(
            report,
            model_path=str(cfg["ckpt"]),
            video_name=Path(rgb_path).name,
            prob_col=prob_col,
        )
        st.download_button(
            "Operasyon özet raporu (MD)",
            data=md.encode("utf-8"),
            file_name="yangin_operasyon_ozeti.md",
            mime="text/markdown",
        )
    with d3:
        zip_b = zip_suspicious_frames(rgb_path, df_scored, prob_col=prob_col, review_thr=inceleme_e)
        if zip_b:
            st.download_button(
                "Şüpheli kareler (ZIP)",
                data=zip_b,
                file_name="supheli_kareler.zip",
                mime="application/zip",
            )
        else:
            st.caption("ZIP yok")
    with d4:
        st.download_button(
            "Olay özeti (CSV)",
            data=dataframe_to_csv_bytes(df_events if df_events is not None else pd.DataFrame()),
            file_name="event_summary.csv",
            mime="text/csv",
        )
    with d5:
        mp = Path(str(result.get("mapping_export_json", "")))
        if mp.is_file():
            st.download_button(
                "Haritalama (JSON)",
                data=mp.read_bytes(),
                file_name="mapping_export.json",
                mime="application/json",
            )
        else:
            st.caption("JSON yok")

    with st.expander("Teknik detaylar", expanded=False):
        bench = _load_benchmark_dict(result)
        n_infer = int(bench.get("infer_calls") or 0)
        n_sim = int(bench.get("rt_skipped_similar") or 0)
        n_budget = int(bench.get("rt_skipped_budget") or 0)
        n_decoded = int(bench.get("decode_frames") or 0)
        if not bench:
            if "inferred" in df_scored.columns:
                n_infer = int(pd.to_numeric(df_scored["inferred"], errors="coerce").fillna(0).sum())
            if "skipped_similar" in df_scored.columns:
                n_sim = int(pd.to_numeric(df_scored["skipped_similar"], errors="coerce").fillna(0).sum())
            n_decoded = len(df_scored)
        elif n_decoded <= 0:
            n_decoded = len(df_scored)
        fps_proc = bench.get("pipeline_fps_processed")

        perf_rows = [
            ("İşlenen kare", str(max(n_decoded, len(df_scored)))),
            ("Çıkarım (infer)", str(n_infer)),
            ("Benzer kare atlanan", str(n_sim)),
            ("Hedef hızından ıskalanan", str(n_budget)),
        ]
        sf = bench.get("skipped_frames")
        if sf is not None:
            perf_rows.insert(1, ("Kayıptan atlanan kare", str(int(sf))))
        if fps_proc is not None:
            perf_rows.append(("Pipeline FPS (~)", f"{float(fps_proc):.2f}"))
        render_metric_row(perf_rows)
        alarm_sec = _alarm_active_duration_sec(df_scored)
        st.caption(f"Tahmini alarm aktif süresi (iş mantığı zamanı): {alarm_sec:.1f} sn")

        st.markdown("###### Kare örneklemeönizleme ve ham tablo")
        render_frame_cards(
            df_scored,
            rgb_path if exists_or_stream else "",
            prob_col,
            session_key_selected=sel_key,
        )

        st.markdown("###### Ham çıktılar (ilk kayıtlar)")
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(
                df_scored[
                    [c for c in ["frame_idx", "timestamp_sec", prob_col, "prob_fire_raw"] if c in df_scored.columns]
                ].head(50),
                use_container_width=True,
            )
        with c2:
            ptail = [c for c in ["decision_prob", "prob_fire", "alarm_state", "pred_fire"] if c in df_scored.columns]
            if ptail:
                st.dataframe(df_scored[ptail].head(50), use_container_width=True)

        infer_profile = cfg.get("infer_args") if isinstance(cfg.get("infer_args"), dict) else {}
        paths_block = {
            k: result.get(k)
            for k in (
                "pred_csv",
                "scored_csv",
                "events_csv",
                "event_summary_csv",
                "mapping_export_json",
                "benchmark_json",
                "alarm_feed_csv",
            )
            if k in result
        }

        st.markdown("###### Yapılandırma ve dosya yolları")
        st.json(
            {
                "kalibrasyon": str(cfg.get("ckpt")),
                "termal": cfg.get("th_path"),
                "isletme_esigi": result.get("threshold_used"),
                "alarm_histerezisi_ust": result.get("hyst_high_used"),
                "alarm_histerezisi_alt": result.get("hyst_low_used"),
                "cikti_dosyalari": paths_block,
                "cikarim_profili": infer_profile or {},
            }
        )
        bp = Path(str(result.get("benchmark_json", "")))
        if bp.is_file():
            st.markdown("###### Performans (benchmark)")
            txt = bp.read_text(encoding="utf-8")
            st.code(txt[:8000], language="json")


def _infer_video_output_csv(filename: str) -> bool:
    """Video çıkarımı / GIS çıktıları — metrik raporu seçicisinden çıkar."""
    n = filename.lower()
    return (
        n.startswith("video_predictions")
        or "predictions_scored" in n
        or "_scored.csv" in n
        or "alarm_feed" in n
        or n.endswith("events.csv")
        or n.endswith("event_summary.csv")
    )


def _collect_metric_report_paths(outputs_dir: Path) -> list[tuple[str, Path]]:
    """Eğitim JSON + bilinen CSV raporları (+ diğer küçük csv'ler)."""
    items: list[tuple[str, Path]] = []
    if not outputs_dir.is_dir():
        return items
    seen: set[str] = set()

    def add(label: str, path: Path) -> None:
        sp = path.resolve()
        key = str(sp)
        if key not in seen:
            seen.add(key)
            items.append((label, path))

    for x in sorted(outputs_dir.glob("metrics_*.json")):
        add(f"[JSON] {x.name}", x)

    csv_priority_patterns = ("robustness*.csv", "ablation*.csv", "improve*.csv")
    for pat in csv_priority_patterns:
        for x in sorted(outputs_dir.glob(pat)):
            if x.is_file():
                add(f"[CSV] {x.name}", x)

    for x in sorted(outputs_dir.glob("*.csv")):
        if x.is_file() and str(x.resolve()) not in seen and not _infer_video_output_csv(x.name):
            add(f"[CSV] {x.name}", x)

    items.sort(key=lambda t: (t[0].split("]", 1)[0], str(t[1]).lower()))
    return items


def _render_review_tab() -> None:
    csv_path = st.text_input("CSV dosya yolu", "outputs/video_predictions_scored.csv", key="rev_csv")
    if not os.path.exists(csv_path):
        st.warning("Dosya yok. Önce ana analizden CSV üretin.")
        return
    df = pd.read_csv(csv_path)
    st.caption(f"Satır: **{len(df)}**")
    st.dataframe(df.head(500), use_container_width=True)


def _render_metrics_tab() -> None:
    st.caption(
        "`final_candidates.zip` içindeki dosyaları bir klasöre çıkarın ve o klasörün yolunu girin; "
        "veya **CSV / JSON** doğrudan yükleyin."
    )
    outputs_dir_str = st.text_input(
        "Çıktı klasörü (örn. `outputs` veya çıkarılmış `final_candidates`)",
        "outputs",
        key="met_dir",
    )
    p = Path(outputs_dir_str.strip())
    uploads = st.file_uploader(
        "Dosya yükle (tablo olarak)",
        type=["csv", "json"],
        key="metrics_file_upload",
        accept_multiple_files=False,
    )

    if uploads is not None:
        up_name = uploads.name or "upload"
        try:
            if up_name.lower().endswith(".json"):
                payload = json.loads(uploads.getvalue().decode("utf-8", errors="replace"))
                st.markdown(f"**{up_name}**")
                rows = []
                for split in ("val", "test"):
                    d = payload.get(split, {})
                    if isinstance(d, dict) and any(v is not None for v in d.values()):
                        rows.append(
                            {
                                "split": split,
                                "accuracy": d.get("acc"),
                                "auc": d.get("auc"),
                                "precision": d.get("precision"),
                                "recall": d.get("recall"),
                                "f1": d.get("f1"),
                            }
                        )
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.json(dict(list(payload.items())[:100]))
                return

            df = pd.read_csv(uploads)
            st.markdown(f"**{up_name}** · {len(df)} satır × {df.shape[1]} sütun")
            _h = min(620, max(360, min(len(df), 25) * 28))
            try:
                st.dataframe(df, use_container_width=True, height=_h)
            except TypeError:
                st.dataframe(df, use_container_width=True)
            return
        except Exception as e:
            st.error(f"Dosya okunamadı: {type(e).__name__}: {e}")
            return

    choices = _collect_metric_report_paths(p)
    if not choices:
        st.warning(
            f"`{p}` içinde rapor bulunamadı. **Zip’i çıkarıp** doğru klasör yolunu yazın veya üstten dosya yükleyin."
        )
        return

    labels = [c[0] for c in choices]
    selected_label = st.selectbox("Dosya seç", labels, index=0)
    sel_path = next(path for lbl, path in choices if lbl == selected_label)

    st.markdown(f"**{sel_path.name}** • `{sel_path.resolve()}`")

    if sel_path.suffix.lower() == ".json":
        try:
            payload = json.loads(sel_path.read_text(encoding="utf-8"))
        except Exception as e:
            st.error(str(e))
            return
        rows = []
        for split in ("val", "test"):
            d = payload.get(split, {})
            if isinstance(d, dict) and len(d):
                rows.append(
                    {
                        "split": split,
                        "accuracy": d.get("acc"),
                        "auc": d.get("auc"),
                        "precision": d.get("precision"),
                        "recall": d.get("recall"),
                        "f1": d.get("f1"),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        extras = [{"field": k, "value": payload[k]} for k in ("epoch", "mode", "model_family", "threshold") if k in payload]
        if extras:
            with st.expander("Extra fields"):
                st.dataframe(pd.DataFrame(extras), use_container_width=True)
        if not rows:
            st.caption("`val`/`test` blokları yok; ham JSON (kısaltılmış).")
            st.json(dict(list(payload.items())[:100]))
    else:
        try:
            df = pd.read_csv(sel_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(sel_path, encoding="latin-1")
        if len(df) > 8000:
            st.warning(f"Çok fazla satır ({len(df)}); ilk **8000** gösteriliyor.")
            df = df.iloc[:8000]
        _h = min(620, max(360, min(len(df), 25) * 28))
        try:
            st.dataframe(df, use_container_width=True, height=_h)
        except TypeError:
            st.dataframe(df, use_container_width=True)

    st.caption(
        "Summary columns: **accuracy**, **precision**, **recall** (sensitivity), **f1**. "
        "Video inference CSV filenames are intentionally excluded from the file list."
    )


def _render_eval_tab() -> None:
    eval_csv = st.text_input("Toplu değerlendirme CSV", "outputs/eval_summary.csv")
    if not os.path.exists(eval_csv):
        st.warning("Dosya yok.")
        return
    st.dataframe(pd.read_csv(eval_csv), use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Yangın Erken Tespit",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_global_styles()

    ckpt_opts = _checkpoint_options()
    samples = _sample_video_options()

    tab_main, tab_rev, tab_met, tab_eval = st.tabs(
        ["Ana analiz", "CSV ile inceleme", "Model metrikleri", "Toplu video değerlendirme"]
    )

    with tab_main:
        if "fire_analysis_cfg" not in st.session_state:
            st.session_state["fire_analysis_cfg"] = None
        if "fire_analysis_result" not in st.session_state:
            st.session_state["fire_analysis_result"] = None

        if st.session_state["fire_analysis_result"] is None:
            triggered, cfg = _render_home(ckpt_options=ckpt_opts, sample_videos=samples)
            if triggered and cfg:
                out_dir = Path(cfg["out_base"]) / _format_run_id()
                with st.spinner("Analiz çalışıyor…"):
                    try:
                        res = _run_with_progress(cfg, out_dir)
                        st.session_state["fire_analysis_cfg"] = cfg
                        st.session_state["fire_analysis_result"] = res
                        st.rerun()
                    except Exception as e:
                        st.error("Analiz başarısız.")
                        st.exception(e)
        else:
            cfg = st.session_state["fire_analysis_cfg"]
            res = st.session_state["fire_analysis_result"]
            if cfg and res:
                if st.button("← Yeni analiz", type="secondary"):
                    st.session_state["fire_analysis_cfg"] = None
                    st.session_state["fire_analysis_result"] = None
                    if "fire_preview_frame_idx" in st.session_state:
                        del st.session_state["fire_preview_frame_idx"]
                    st.rerun()
                else:
                    _render_analysis_dashboard(cfg, res)

        st.caption(
            "Kurumsal kullanımda daha zengin arayüz için ileride **FastAPI + React** ile "
            "canlı panel ayrılabilir; şu an hızlı dağıtım için **Streamlit** kullanılmaktadır."
        )

    with tab_rev:
        _render_review_tab()
    with tab_met:
        _render_metrics_tab()
    with tab_eval:
        _render_eval_tab()


if __name__ == "__main__":
    main()
