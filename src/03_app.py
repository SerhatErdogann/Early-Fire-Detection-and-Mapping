# src/03_app.py — Gradio demo: detection metrics + optional CAM explainability tab
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image
import torch
import gradio as gr
from torchcam.methods import SmoothGradCAMpp

from src.inference.model_loader import load_checkpoint
from src.inference.postprocess import stats_from_soft_map
from src.data.dataset import read_rgb_pil, read_thermal_raw, thermal_to_norm01
from src.models.cls.dual_branch_fusion import DualBranchFusion

try:
    from config import CKPT_RGB, CKPT_THERMAL, CKPT_FUSION, DEFAULT_INPUT_SIZE
except ImportError:
    CKPT_RGB = Path("models/rgb.pt")
    CKPT_THERMAL = Path("models/thermal.pt")
    CKPT_FUSION = Path("models/fusion.pt")
    DEFAULT_INPUT_SIZE = 384

MODELS = {"rgb": str(CKPT_RGB), "thermal": str(CKPT_THERMAL), "fusion": str(CKPT_FUSION)}

loaded = {}


def read_rgb(path, size=None):
    size = size or DEFAULT_INPUT_SIZE
    img = read_rgb_pil(path)
    img = img.resize((size, size))
    arr = (np.array(img).astype(np.float32) / 255.0).transpose(2, 0, 1)
    return arr, img


def read_thermal(path, size=None):
    size = size or DEFAULT_INPUT_SIZE
    th_raw, kind = read_thermal_raw(path)
    img = cv2.resize(th_raw, (size, size), interpolation=cv2.INTER_AREA)
    img = thermal_to_norm01(img, kind)
    pil = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
    return img[None, ...], pil


def _forward_batch(model_name, rgb_path, th_path):
    if model_name not in loaded:
        loaded[model_name] = load_checkpoint(MODELS[model_name])
    model, mode, device, thr, temperature = loaded[model_name]

    rgb_arr, rgb_pil = read_rgb(rgb_path)
    th_arr, th_pil = read_thermal(th_path)

    if mode == "rgb":
        x = torch.tensor(rgb_arr[None, ...], dtype=torch.float32).to(device)
        base = np.array(rgb_pil)
    elif mode == "thermal":
        x = torch.tensor(th_arr[None, ...], dtype=torch.float32).to(device)
        base = np.array(th_pil)
    else:
        x = torch.tensor(np.concatenate([rgb_arr, th_arr], axis=0)[None, ...], dtype=torch.float32).to(device)
        base = np.array(rgb_pil)

    scores = model(x)
    scores_cal = scores / max(1e-6, temperature)
    prob = torch.softmax(scores_cal, dim=1)[0, 1].item()
    return model, mode, device, base, x, scores, prob, thr, temperature


def infer_detection(model_name, rgb_path, th_path):
    model, mode, device, base, x, scores, prob, thr, temperature = _forward_batch(model_name, rgb_path, th_path)
    cam = None
    try:
        if isinstance(model, DualBranchFusion):
            cam_extractor = SmoothGradCAMpp(model, target_layer=model.rgb_branch.layer4)
        else:
            cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")
        cam = cam_extractor(class_idx=1, scores=scores)[0].squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    except Exception:
        cam = np.full((base.shape[0], base.shape[1]), prob, dtype=np.float32)

    st = stats_from_soft_map(cam)
    txt = (
        f"prob_fire={prob:.3f}\n"
        f"threshold_ckpt={thr:.3f}  temperature={temperature:.3f}\n"
        f"mask_area_soft={st['fire_area_soft']:.4f}  largest_component={st['largest_component_area']:.4f}\n"
        f"peak_intensity={st['peak_intensity']:.4f}  num_components={st['num_components']}\n"
    )
    blank = np.zeros_like(base)
    return txt, blank


def infer_explain(model_name, rgb_path, th_path):
    model, mode, device, base, x, scores, prob, thr, temperature = _forward_batch(model_name, rgb_path, th_path)
    try:
        if isinstance(model, DualBranchFusion):
            cam_extractor = SmoothGradCAMpp(model, target_layer=model.rgb_branch.layer4)
        else:
            cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")
        cam = cam_extractor(class_idx=1, scores=scores)[0].squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    except Exception as e:
        return None, f"CAM üretilemedi: {e}"

    heat = (cam * 255).astype(np.uint8)
    heat = cv2.resize(heat, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
    overlay = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    out = (0.55 * base + 0.45 * overlay).astype(np.uint8)
    st = stats_from_soft_map(cam)
    txt = f"p(fire)={prob:.3f}\nCAM mean={cam.mean():.3f} peak={st['peak_intensity']:.3f}"
    return out, txt


with gr.Blocks(title="Yangın — tespit + açıklanabilirlik") as demo:
    gr.Markdown("### RGB + termal — ana metrikler ayrı; CAM sadece **Açıklanabilirlik** sekmesinde.")
    model_dd = gr.Dropdown(["rgb", "thermal", "fusion"], value="fusion", label="Model")
    rgb_in = gr.Image(type="filepath", label="RGB")
    th_in = gr.Image(type="filepath", label="Thermal")

    with gr.Tab("Tespit"):
        det_txt = gr.Textbox(label="Özet", lines=6)
        det_placeholder = gr.Image(label="(CAM burada gösterilmez)")
        go_det = gr.Button("Çalıştır")
        go_det.click(infer_detection, [model_dd, rgb_in, th_in], [det_txt, det_placeholder])

    with gr.Tab("Açıklanabilirlik (CAM)"):
        cam_img = gr.Image(label="Heatmap")
        cam_txt = gr.Textbox(label="CAM notları")
        go_cam = gr.Button("CAM üret")
        go_cam.click(infer_explain, [model_dd, rgb_in, th_in], [cam_img, cam_txt])

if __name__ == "__main__":
    demo.launch()
