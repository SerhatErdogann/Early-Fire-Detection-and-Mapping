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
from src.ui.constants import PRESETS
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
from src.ui.video_panel import render_dual_preview, render_frame_cards

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

    st.subheader("Analiz hassasiyeti")
    preset_map = {p.key: p for p in PRESETS}
    preset_key = st.radio(
        "",
        options=[p.key for p in PRESETS],
        format_func=lambda k: preset_map[k].title,
        horizontal=True,
        label_visibility="collapsed",
    )
    preset = preset_map[preset_key]
    st.caption(f"_{preset.description}_")

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
            "preset": preset,
            "ckpt": ckpt_choice,
            "out_base": out_base,
        }
        return True, cfg
    return False, None


def _run_with_progress(
    cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    progress = st.progress(0.0, text="Hazırlanıyor…")

    def cb(done: int, est: int | None) -> None:
        if est and est > 0:
            r = min(1.0, float(done) / float(est))
            txt = f"Video analiz ediliyor: {done} / ~{est} örnek kare"
        else:
            r = min(1.0, float(done) / 5000.0)
            txt = f"Video analiz ediliyor: {done} örnek kare (süre bilinmiyor veya canlı akış)"
        progress.progress(min(0.99, r), text=txt)

    res = run_analysis_pipeline(
        cfg["rgb_path"],
        cfg["th_path"],
        cfg["preset"].args,
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

    st.subheader("Video analiz ekranı")
    left, right = st.columns([1.1, 1.0])

    sel_key = "fire_sel_frame"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = int(df_scored["frame_idx"].iloc[0])

    row_live = nearest_row_by_frame(df_scored, int(st.session_state[sel_key]))
    if row_live is None:
        row_live = df_scored.iloc[0]
    prob = float(row_live[prob_col]) if prob_col in row_live.index else 0.0
    st_al = str(row_live.get("alarm_state", "")) if hasattr(row_live, "get") else ""

    tail = df_scored.sort_values("frame_idx").tail(5)
    slope = None
    if len(tail) > 1 and prob_col in tail.columns:
        tprob = pd.to_numeric(tail[prob_col], errors="coerce")
        slope = float(tprob.diff().mean()) if len(tprob) else None

    cap = turkish_caption_for_row(prob, alarm_e, inceleme_e, slope, st_al or None)

    with left:
        render_dual_preview(
            rgb_path if exists_or_stream else None,
            int(st.session_state[sel_key]),
            df_scored,
            prob_col,
        )
        st.markdown("---")
        render_frame_cards(
            df_scored,
            rgb_path,
            prob_col,
            session_key_selected=sel_key,
        )

    with right:
        render_live_panel(
            prob * 100.0,
            prob,
            alarm_e,
            inceleme_e,
            st_al or None,
            cap,
        )
        st.markdown("---")
        st.markdown("##### Olasılığın zaman içindeki seyri")
        chart_df = df_scored.sort_values("frame_idx").copy()
        ts_col = pd.to_numeric(chart_df["timestamp_sec"], errors="coerce").fillna(0.0)
        chart_df["Zaman_sn"] = ts_col
        chart_df["Yangın olasılığı"] = pd.to_numeric(chart_df[prob_col], errors="coerce").fillna(0.0)
        chart_df["Alarm eşiği"] = float(alarm_e)
        chart_df["İnceleme eşiği"] = float(inceleme_e)
        try:
            st.line_chart(
                chart_df.set_index("Zaman_sn")[
                    ["Yangın olasılığı", "Alarm eşiği", "İnceleme eşiği"]
                ],
                height=280,
            )
        except Exception:
            st.line_chart(chart_df[["Yangın olasılığı"]], height=280)
        st.caption(
            "_Eşik çizgileri: alarm (yüksek uyarı) ve inceleme (orta) düzeyidir._"
        )

    st.markdown("---")
    st.subheader("Özet rapor")
    v_kind = report.verdict_key
    badge = (
        "🔴 Yangın riski yüksek"
        if v_kind == "fire"
        else "🟡 İnceleme gerekli"
        if v_kind == "review"
        else "🟢 Yangın yok"
    )
    st.markdown(f"**Genel sonuç:** {report.verdict_tr} &nbsp; {badge}")
    render_metric_row(
        [
            ("En yüksek olasılık", f"{report.max_prob:.1%}"),
            ("Ortalama olasılık", f"{report.mean_prob:.1%}"),
            ("Analiz edilen örnek", str(report.frames_analyzed)),
            ("Alarm eşiği", f"{report.alarm_esigi:.3f}"),
        ]
    )
    st.markdown(f"**İnceleme eşiği:** {report.inceleme_esigi:.3f}")
    if report.alarm_zaman_araliklari:
        st.markdown("**Alarm için öne çıkan zaman aralıkları:**")
        for a, b in report.alarm_zaman_araliklari:
            st.write(f"- {a:.2f} s – {b:.2f} s")
    else:
        st.info("Sürekli yüksek uyarı segmenti raporlanmadı (eşik üstü kısa süreler olabilir).")

    st.markdown(f"**Model güvenilirlik notu:** {report.guvenilirlik_notu}")

    st.markdown("---")
    st.subheader("Çıktıları indir")
    csv_b = dataframe_to_csv_bytes(df_scored)
    st.download_button(
        "Örnek kare tablosunu indir (CSV)",
        data=csv_b,
        file_name="yangin_analiz_olcumleri.csv",
        mime="text/csv",
    )
    md = build_markdown_report(
        report,
        model_path=str(cfg["ckpt"]),
        video_name=Path(rgb_path).name,
        analiz_modu=str(cfg["preset"].title),
        prob_col=prob_col,
    )
    st.download_button(
        "Özet raporu indir (Markdown)",
        data=md.encode("utf-8"),
        file_name="yangin_analiz_ozeti.md",
        mime="text/markdown",
    )
    zip_b = zip_suspicious_frames(rgb_path, df_scored, prob_col=prob_col, review_thr=inceleme_e)
    if zip_b:
        st.download_button(
            "Şüpheli kareleri indir (ZIP, JPG)",
            data=zip_b,
            file_name="supheli_kareler.zip",
            mime="application/zip",
        )
    else:
        st.caption("İnceleme eşiği üstü kare bulunamadı veya video okunamadı — ZIP oluşturulmadı.")

    st.caption(
        "_PDF: Markdown dosyasını bir metin düzenleyicide açıp «Yazdır → PDF olarak kaydet» kullanabilirsiniz._"
    )


def _render_debug(cfg: dict[str, Any], result: dict[str, Any]) -> None:
    df_scored: pd.DataFrame = result["df_scored"]
    prob_col = _pick_prob_col(df_scored)
    with st.expander("Gelişmiş — teknik ayrıntılar", expanded=False):
        st.markdown("##### Ham olasılık ve zaman bilgisi")
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(
                df_scored[
                    [c for c in ["frame_idx", "timestamp_sec", prob_col, "prob_fire_raw"] if c in df_scored.columns]
                ].head(40),
                use_container_width=True,
            )
        with c2:
            ptail = [c for c in ["decision_prob", "prob_fire", "prob_fire_ema", "prob_fire_ma"] if c in df_scored.columns]
            if ptail:
                st.dataframe(df_scored[ptail].tail(40), use_container_width=True)
        st.markdown("##### Model ve eşikler")
        st.json(
            {
                "model_path": cfg.get("ckpt"),
                "thermal_path": cfg.get("th_path"),
                "threshold_used": result.get("threshold_used"),
                "hyst_high_used": result.get("hyst_high_used"),
                "hyst_low_used": result.get("hyst_low_used"),
                "out_files": {k: result[k] for k in ("pred_csv", "scored_csv", "events_csv", "benchmark_json") if k in result},
                "preset": getattr(cfg.get("preset"), "key", ""),
            }
        )
        if Path(str(result.get("benchmark_json", ""))).is_file():
            st.markdown("##### Performans (benchmark JSON)")
            st.code(Path(result["benchmark_json"]).read_text(encoding="utf-8")[:4000], language="json")


def _render_review_tab() -> None:
    csv_path = st.text_input("CSV dosya yolu", "outputs/video_predictions_scored.csv", key="rev_csv")
    if not os.path.exists(csv_path):
        st.warning("Dosya yok. Önce ana analizden CSV üretin.")
        return
    df = pd.read_csv(csv_path)
    st.caption(f"Satır: **{len(df)}**")
    st.dataframe(df.head(500), use_container_width=True)


def _render_metrics_tab() -> None:
    outputs_dir = st.text_input("Outputs klasörü", "outputs", key="met_dir")
    p = Path(outputs_dir)
    metric_files = sorted([x for x in p.glob("metrics_*.json")]) if p.exists() else []
    if not metric_files:
        st.warning("`metrics_*.json` bulunamadı.")
        return
    selected = st.selectbox("Metrik dosyası", [str(x) for x in metric_files])
    payload = json.loads(Path(selected).read_text(encoding="utf-8"))
    rows = []
    for split in ("val", "test"):
        d = payload.get(split, {})
        if isinstance(d, dict):
            rows.append(
                {
                    "Bölüm": split,
                    "Doğruluk": d.get("acc"),
                    "AUC": d.get("auc"),
                    "Kesinlik": d.get("precision"),
                    "Yangını kaçırmama": d.get("recall"),
                    "F1": d.get("f1"),
                }
            )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption(
        "_Yangını kaçırmama oranı: modelin gerçek yangınları ne kadar yakaladığıdır "
        "(teknik adıyla duyarlılık / recall)._"
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
                    if "fire_sel_frame" in st.session_state:
                        del st.session_state["fire_sel_frame"]
                    st.rerun()
                _render_analysis_dashboard(cfg, res)
                _render_debug(cfg, res)

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
