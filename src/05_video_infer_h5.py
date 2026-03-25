# src/05_video_infer_h5.py — .h5 (Keras/TF) model ile video inference
# Çıktı formatı 05_video_infer.py ile aynı; 06_add_risk_score ve 07_ui aynen kullanılır.

import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    raise SystemExit("TensorFlow gerekli: pip install tensorflow")

# ------------------ Preprocess ------------------
def prep_rgb(frame_bgr, size=224):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    # Keras genelde (H,W,3) veya (N,H,W,3); normalize 0-1
    arr = (rgb.astype(np.float32) / 255.0)
    return arr, rgb


def main():
    import argparse
    ap = argparse.ArgumentParser(description=".h5 (Keras) model ile video inference")
    ap.add_argument("--rgb_video", required=True)
    ap.add_argument("--model_h5", required=True, help=".h5 model dosyası yolu")
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--smooth_win", type=int, default=1)
    ap.add_argument("--override_thr", type=float, default=0.5, help="Eşik (0-1)")
    ap.add_argument("--size", type=int, default=224, help="Model giriş boyutu (224 veya 384)")
    ap.add_argument("--out", default="outputs/video_predictions.csv")
    args = ap.parse_args()

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Model yükle
    model_path = Path(args.model_h5)
    if not model_path.exists():
        raise SystemExit(f"Model bulunamadı: {model_path}")
    model = keras.models.load_model(str(model_path))
    # Giriş boyutunu modelden al (opsiyonel)
    try:
        in_shape = model.input_shape
        if in_shape[1] is not None and in_shape[2] is not None:
            args.size = int(in_shape[1])
    except Exception:
        pass

    cap = cv2.VideoCapture(args.rgb_video)
    if not cap.isOpened():
        raise SystemExit(f"Video açılamadı: {args.rgb_video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    rows = []
    prob_buffer = []
    win = max(1, int(args.smooth_win))
    idx = 0
    thr = float(args.override_thr)

    pbar = tqdm(total=total if total > 0 else None, desc="Infer (h5)")
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % args.step != 0:
            idx += 1
            pbar.update(1)
            continue

        arr, _ = prep_rgb(fr, size=args.size)
        # (1, H, W, 3) veya (1, H, W, 3) - Keras channel-last
        x = np.expand_dims(arr, axis=0).astype(np.float32)
        out = model.predict(x, verbose=0)

        # Çıktı (N,2) logit/softmax veya (N,1)
        out = np.asarray(out)
        if out.shape[-1] >= 2:
            logits = out[0]
            if logits.max() > 1.0 or logits.min() < 0.0:
                exp = np.exp(logits - logits.max())
                prob_raw = float((exp / exp.sum())[1])
            else:
                prob_raw = float(logits[1])
        else:
            prob_raw = float(out[0, 0])
        prob_raw = max(0.0, min(1.0, prob_raw))

        prob_buffer.append(prob_raw)
        if len(prob_buffer) > win:
            prob_buffer = prob_buffer[-win:]
        prob = float(np.mean(prob_buffer))
        pred_fire = 1 if prob >= thr else 0

        # .h5 ile Grad-CAM yok; intensity alanları 0, heatmap boş
        rows.append({
            "frame_idx": idx,
            "prob_fire_raw": prob_raw,
            "prob_fire": prob,
            "pred_fire": pred_fire,
            "threshold_used": thr,
            "intensity_mean": 0.0,
            "intensity_top10": 0.0,
            "area_heat_gt_0_6": 0.0,
            "heatmap_path": ""
        })
        idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("✅ yazıldı:", out_csv)


if __name__ == "__main__":
    main()
