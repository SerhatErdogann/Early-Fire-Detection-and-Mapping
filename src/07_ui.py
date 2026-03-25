import os
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Fire Risk Review", layout="wide")

st.title("Fire Risk Review (RGB+Thermal Fusion)")

csv_path = st.text_input("CSV path", "outputs/video_predictions_scored.csv")

if not os.path.exists(csv_path):
    st.warning("CSV bulunamadı. Önce 05_video_infer + 06_add_risk_score çalıştır.")
    st.stop()

df = pd.read_csv(csv_path)


def _col(df_, name, default=0.0):
    if name in df_.columns:
        return pd.to_numeric(df_[name], errors="coerce").fillna(default)
    return pd.Series([default] * len(df_), index=df_.index, dtype=float)

st.sidebar.header("Filtre / Sıralama")
sort_by = st.sidebar.selectbox(
    "Sırala",
    [
        "risk_score",
        "prob_fire",
        "intensity_top10",
        "area_heat_gt_0_6",
        "largest_component_area",
        "peak_intensity",
    ],
    index=0,
)
desc = st.sidebar.checkbox("Azalan sırala", value=True)

min_prob = st.sidebar.slider("min prob_fire", 0.0, 1.0, 0.0, 0.01)
min_risk = st.sidebar.slider("min risk_score_norm", 0.0, 1.0, 0.0, 0.01)

if "risk_score_norm" not in df.columns:
    mx = df["risk_score"].max()
    df["risk_score_norm"] = df["risk_score"] / (mx + 1e-9)

# Genel veri kalitesi / çalışma modu uyarıları
high_prob_ratio = float((_col(df, "prob_fire") >= 0.7).mean())
fire_event_ratio = float((_col(df, "fire_event") > 0.5).mean()) if "fire_event" in df.columns else 0.0
if "early_detection" in df.columns and int(_col(df, "early_detection").max()) == 1:
    st.sidebar.warning("Early detection aktif: recall artar, false positive artabilir.")
if high_prob_ratio > 0.35:
    st.warning(
        "Birçok frame yüksek olasılığa çıktı. Threshold düşük veya sahne sıcak/noisy olabilir."
    )
if "fire_event" in df.columns and fire_event_ratio > 0.25:
    st.info("Fire_event oranı yüksek. Persist/hysteresis ayarlarını gözden geçirmen faydalı olabilir.")

view = df[(df["prob_fire"] >= min_prob) & (df["risk_score_norm"] >= min_risk)].copy()
view = view.sort_values(sort_by, ascending=not desc).reset_index(drop=True)

st.write(f"Toplam satır: {len(df)} | Filtre sonrası: {len(view)}")

# En riskli ilk N
N = st.sidebar.number_input("Listelenecek satır", min_value=10, max_value=500, value=100, step=10)
viewN = view.head(N)

# seçim
idx = st.number_input("Seçili satır index (0..)", min_value=0, max_value=max(0, len(viewN)-1), value=0, step=1)

col1, col2 = st.columns([1, 1])

row = viewN.iloc[idx]

with col1:
    st.subheader("Seçili Frame Bilgisi")
    st.json({
        "frame_idx": int(row["frame_idx"]),
        "prob_fire": float(row["prob_fire"]),
        "intensity_mean": float(row["intensity_mean"]),
        "intensity_top10": float(row["intensity_top10"]),
        "area_heat_gt_0_6": float(row["area_heat_gt_0_6"]),
        "risk_score": float(row["risk_score"]),
        "risk_score_norm": float(row["risk_score_norm"]),
        "heatmap_path": str(row.get("heatmap_path", "")),
    })

    st.subheader("Manuel Etiketleme")
    label = st.radio(
        "Sınıf",
        ["unknown", "yes_fire", "no_fire", "smoke_only", "hot_nonfire", "uncertain"],
        horizontal=True,
    )
    note = st.text_input("Not (opsiyonel)", "")

    if st.button("✅ Kaydet / Güncelle"):
        out = "outputs/manual_review.csv"
        os.makedirs("outputs", exist_ok=True)

        rec = {
            "frame_idx": int(row["frame_idx"]),
            "prob_fire": float(row["prob_fire"]),
            "risk_score": float(row.get("risk_score", 0.0)),
            "label": label,
            "note": note,
            "heatmap_path": str(row.get("heatmap_path", "")),
            "mask_path": str(row.get("mask_path", "")),
        }

        if os.path.exists(out):
            old = pd.read_csv(out)
            # aynı frame varsa güncelle
            old = old[old["frame_idx"] != rec["frame_idx"]]
            new = pd.concat([old, pd.DataFrame([rec])], ignore_index=True)
        else:
            new = pd.DataFrame([rec])

        new.to_csv(out, index=False)
        st.success(f"Kaydedildi: {out}")

    st.subheader("Uyarılar")
    warnings = []
    prob_fire = float(row.get("prob_fire", 0.0))
    decision_prob = float(row.get("decision_prob", prob_fire))
    risk_norm = float(row.get("risk_score_norm", 0.0))
    area = float(row.get("largest_component_area", 0.0))
    growth = float(row.get("growth_rate", 0.0))
    top10 = float(row.get("intensity_top10", 0.0))
    modal = row.get("modal_agreement", "")
    modal_val = None
    try:
        modal_val = float(modal)
    except Exception:
        modal_val = None

    if decision_prob >= 0.85 or risk_norm >= 0.85:
        warnings.append(("error", "Yuksek risk: bu frame alarm adayi. Oncelikli manuel kontrol et."))
    elif decision_prob >= 0.65 or risk_norm >= 0.65:
        warnings.append(("warning", "Orta-yuksek risk: devam eden karelerle birlikte degerlendir."))

    if 0.0 < area < 0.02 and prob_fire >= 0.4:
        warnings.append(("warning", "Kucuk ama anlamli sicak bolge: erken yangin olasiligi olabilir."))
    if growth > 0.0 and area > 0.01:
        warnings.append(("warning", "Alan buyume egiliminde: gercek yangin ihtimali artiyor."))
    if prob_fire < 0.25 and top10 > 0.7:
        warnings.append(("info", "Parlak/sicak tepe var ama genel olasilik dusuk: false positive olabilir."))
    if modal_val is not None and modal_val < 0.2:
        warnings.append(("info", "RGB-termal uyumu dusuk: termal kaynakli yalanci alarm olabilir."))
    if int(row.get("early_detection", 0)) == 1:
        warnings.append(("info", "Early detection acik: sistem daha erken ama daha hassas alarm verir."))

    if warnings:
        for level, text in warnings:
            if level == "error":
                st.error(text)
            elif level == "warning":
                st.warning(text)
            else:
                st.info(text)
    else:
        st.success("Belirgin bir uyari yok. Frame genel olarak dengeli gorunuyor.")

with col2:
    st.subheader("Heatmap / Maske")
    hp = str(row.get("heatmap_path", ""))
    mp = str(row.get("mask_path", ""))
    if mp and os.path.exists(mp):
        st.image(Image.open(mp), caption="Mask (soft)", use_container_width=True)
    elif hp and os.path.exists(hp):
        st.image(Image.open(hp), caption=f"frame_idx={int(row['frame_idx'])}", use_container_width=True)
    else:
        st.warning("heatmap_path / mask_path yok. 05_video_infer --save_heatmaps veya --save_masks kullan.")

st.subheader("Liste (ilk N)")
cols = [
    "frame_idx",
    "prob_fire",
    "intensity_top10",
    "area_heat_gt_0_6",
    "largest_component_area",
    "risk_score",
    "risk_score_norm",
    "heatmap_path",
    "mask_path",
]
cols = [c for c in cols if c in viewN.columns]
st.dataframe(viewN[cols])