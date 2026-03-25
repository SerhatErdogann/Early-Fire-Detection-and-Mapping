import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchcam.methods import SmoothGradCAMpp

from src.inference.model_loader import load_checkpoint
from src.inference.postprocess import stats_from_soft_map
from src.data.dataset import read_rgb_pil, read_thermal_raw, thermal_to_norm01

try:
    from config import DEFAULT_INPUT_SIZE, OUTPUTS_DIR, CKPT_RGB, CKPT_THERMAL, CKPT_FUSION, FLAME_INDEX_CSV, MASTER_INDEX_PARQUET
except ImportError:
    DEFAULT_INPUT_SIZE = 384
    OUTPUTS_DIR = Path("outputs")
    CKPT_RGB = Path("models/rgb.pt")
    CKPT_THERMAL = Path("models/thermal.pt")
    CKPT_FUSION = Path("models/fusion.pt")
    FLAME_INDEX_CSV = Path("outputs/flame_index.csv")
    MASTER_INDEX_PARQUET = Path("data/master_index.parquet")


def read_rgb(path, size=None):
    size = size or DEFAULT_INPUT_SIZE
    img = read_rgb_pil(path).resize((size, size))
    arr = (np.array(img).astype(np.float32) / 255.0).transpose(2, 0, 1)
    return arr, img


def read_thermal(path, size=None):
    size = size or DEFAULT_INPUT_SIZE
    th_raw, kind = read_thermal_raw(path)
    img = cv2.resize(th_raw, (size, size), interpolation=cv2.INTER_AREA)
    img = thermal_to_norm01(img, kind)
    pil = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
    return img[None, ...], pil


def _load_mask_prob(path: str, size: int) -> np.ndarray | None:
    if not path or not Path(path).exists():
        return None
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = cv2.resize(m.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    if m.max() > 1.5:
        m = m / 255.0
    return np.clip(m, 0, 1)


def infer_one(model, mode, device, rgb_path, th_path, size=None, temperature=1.0, path_mask=None):
    size = size or DEFAULT_INPUT_SIZE
    rgb_arr, rgb_pil = read_rgb(rgb_path, size)
    th_arr, th_pil = read_thermal(th_path, size)

    if mode == "rgb":
        x = torch.tensor(rgb_arr[None, ...], dtype=torch.float32).to(device)
    elif mode == "thermal":
        x = torch.tensor(th_arr[None, ...], dtype=torch.float32).to(device)
    else:
        x = torch.tensor(np.concatenate([rgb_arr, th_arr], axis=0)[None, ...], dtype=torch.float32).to(device)

    mask_prob = _load_mask_prob(path_mask or "", size)
    if mask_prob is not None:
        with torch.no_grad():
            scores = model(x)
            scores_cal = scores / max(1e-6, temperature)
            prob = torch.softmax(scores_cal, dim=1)[0, 1].item()
        st = stats_from_soft_map(mask_prob)
        st["prob_fire"] = prob
        st["intensity_mean"] = float(mask_prob.mean())
        st["intensity_top10"] = float(np.mean(np.sort(mask_prob.ravel())[-max(1, int(0.1 * mask_prob.size)) :]))
        st["area_heat_gt_0_6"] = float((mask_prob > 0.6).mean())
        return st

    scores = model(x)
    scores_cal = scores / max(1e-6, temperature)
    prob = torch.softmax(scores_cal, dim=1)[0, 1].item()

    try:
        cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")
        cam = cam_extractor(class_idx=1, scores=scores)[0].squeeze().detach().cpu().numpy()
    except Exception:
        try:
            cam_extractor = SmoothGradCAMpp(model, target_layer=model.rgb_branch.layer4)
            cam = cam_extractor(class_idx=1, scores=scores)[0].squeeze().detach().cpu().numpy()
        except Exception:
            st = stats_from_soft_map(np.full((size, size), prob, dtype=np.float32))
            st["prob_fire"] = prob
            st["intensity_mean"] = prob
            st["intensity_top10"] = prob
            st["area_heat_gt_0_6"] = float(prob > 0.6)
            return st

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    st = stats_from_soft_map(cam)
    st["prob_fire"] = prob
    st["intensity_mean"] = float(cam.mean())
    st["intensity_top10"] = float(np.mean(np.sort(cam.ravel())[-max(1, int(0.1 * cam.size)) :]))
    st["area_heat_gt_0_6"] = float((cam > 0.6).mean())
    return st


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Export per-sample spatial stats (mask if available else CAM fallback)")
    ap.add_argument("--model", choices=["rgb", "thermal", "fusion"], default="fusion")
    ap.add_argument("--index", default=None, help="CSV or Parquet index")
    args = ap.parse_args()

    CKPT = {"rgb": str(CKPT_RGB), "thermal": str(CKPT_THERMAL), "fusion": str(CKPT_FUSION)}[args.model]
    index_csv = args.index
    if index_csv is None:
        index_csv = str(MASTER_INDEX_PARQUET) if Path(MASTER_INDEX_PARQUET).exists() else str(FLAME_INDEX_CSV)

    out_csv = str(OUTPUTS_DIR / f"intensity_{args.model}.csv")

    df = pd.read_parquet(index_csv) if str(index_csv).lower().endswith(".parquet") else pd.read_csv(index_csv)
    model, mode, device, _thr, temperature = load_checkpoint(CKPT)

    out_rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Export {args.model}"):
        mask_col = r.get("path_mask") if "path_mask" in r.index else None
        if mask_col is not None and pd.isna(mask_col):
            mask_col = None
        st = infer_one(
            model,
            mode,
            device,
            str(r["path_rgb"]),
            str(r["path_th"] if "path_th" in r.index and pd.notna(r["path_th"]) else r["path_thermal"]),
            temperature=temperature,
            path_mask=str(mask_col) if mask_col else None,
        )
        row = {
            "path_rgb": r["path_rgb"],
            "path_th": r.get("path_th", r.get("path_thermal")),
            "label": int(r["label"]),
            "prob_fire": st.get("prob_fire", 0.0),
            "fire_mass": st.get("fire_mass", 0.0),
            "fire_area_soft": st.get("fire_area_soft", 0.0),
            "fire_area_hard": st.get("fire_area_hard", 0.0),
            "peak_intensity": st.get("peak_intensity", 0.0),
            "num_components": st.get("num_components", 0),
            "largest_component_area": st.get("largest_component_area", 0.0),
            "centroid_x_norm": st.get("centroid_x_norm", 0.0),
            "centroid_y_norm": st.get("centroid_y_norm", 0.0),
            "edge_density": st.get("edge_density", 0.0),
            "intensity_mean": st.get("intensity_mean", 0.0),
            "intensity_top10": st.get("intensity_top10", 0.0),
            "area_heat_gt_0_6": st.get("area_heat_gt_0_6", 0.0),
        }
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)
    print("✅ yazıldı:", out_csv)


if __name__ == "__main__":
    main()
