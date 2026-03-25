"""
Video inference with optional EMA smoothing and TTA for more stable predictions on drone footage.
"""
import json
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from .model_loader import load_checkpoint
from .postprocess import stats_from_soft_map
from .preprocess import prep_rgb, prep_thermal

try:
    from config import INFERENCE_DEFAULT, OUTPUTS_DIR
except ImportError:
    INFERENCE_DEFAULT = {"smooth_window": 5, "ema_alpha": 0.3, "use_tta": False, "step_frames": 5}
    OUTPUTS_DIR = Path("outputs")


def _smooth_ema(prob_raw: float, prev_ema: float, alpha: float):
    return alpha * prev_ema + (1.0 - alpha) * prob_raw


def _safe_corr2d(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float32).ravel()
    bv = np.asarray(b, dtype=np.float32).ravel()
    if av.size == 0 or bv.size == 0 or av.size != bv.size:
        return float("nan")
    sa = float(av.std())
    sb = float(bv.std())
    if sa < 1e-6 or sb < 1e-6:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def _sym_kl_hist(p: np.ndarray, q: np.ndarray, bins: int = 32) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    hp, _ = np.histogram(p, bins=bins, range=(0.0, 1.0), density=True)
    hq, _ = np.histogram(q, bins=bins, range=(0.0, 1.0), density=True)
    eps = 1e-8
    hp = hp + eps
    hq = hq + eps
    hp = hp / hp.sum()
    hq = hq / hq.sum()
    kl_pq = float(np.sum(hp * np.log(hp / hq)))
    kl_qp = float(np.sum(hq * np.log(hq / hp)))
    return 0.5 * (kl_pq + kl_qp)


def run_video_inference(
    rgb_video_path,
    th_video_path=None,
    ckpt_path=None,
    mode="auto",
    size=384,
    step_frames=5,
    smooth_window=5,
    ema_alpha=0.3,
    use_tta=False,
    override_thr=None,
    save_heatmaps=False,
    save_masks=False,
    save_polygons=False,
    out_csv=None,
    heatmap_dir=None,
    mask_dir=None,
    polygon_dir=None,
    cam_layer="layer4",
    use_fp16=False,
    cam_stats_only: bool = False,
    temporal_guard: bool = True,
    scene_thresh: float = 0.10,
    scene_conf_scale: float = 0.7,
    hyst_high: float = 0.7,
    hyst_low: float = 0.4,
    persist_n: int = 5,
    min_component_area: float = 0.01,
    growth_downscale: float = 0.85,
    use_kl_scene: bool = False,
    kl_hist_thresh: float = 0.35,
    early_detection: bool = False,
    early_threshold_shift: float = 0.15,
    early_min_threshold: float = 0.25,
    early_persist_n: int = 2,
    small_fire_boost: float = 1.3,
    small_fire_area_max: float = 0.02,
    growth_upscale: float = 1.2,
    texture_prob_max: float = 0.2,
    texture_top10_min: float = 0.7,
    enable_modal_agreement: bool = False,
    modal_agreement_min_corr: float = 0.2,
    modal_agreement_penalty: float = 0.6,
):
    """
    Run fire classification on video (RGB only or RGB+thermal).
    Uses moving average or EMA for probability smoothing (more stable on drone videos).
    """
    out_csv = Path(out_csv or OUTPUTS_DIR / "video_predictions.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if save_heatmaps:
        heatmap_dir = Path(heatmap_dir or OUTPUTS_DIR / "heatmaps")
        heatmap_dir.mkdir(parents=True, exist_ok=True)
    if save_masks:
        mask_dir = Path(mask_dir or OUTPUTS_DIR / "masks")
        mask_dir.mkdir(parents=True, exist_ok=True)
    if save_polygons:
        polygon_dir = Path(polygon_dir or OUTPUTS_DIR / "polygons")
        polygon_dir.mkdir(parents=True, exist_ok=True)

    cap_rgb = cv2.VideoCapture(str(rgb_video_path))
    if not cap_rgb.isOpened():
        raise SystemExit(f"Cannot open RGB video: {rgb_video_path}")
    cap_th = None
    if th_video_path:
        cap_th = cv2.VideoCapture(str(th_video_path))
        if not cap_th.isOpened():
            raise SystemExit(f"Cannot open thermal video: {th_video_path}")

    use_fusion = (mode == "fusion" or (mode == "auto" and cap_th is not None))
    if mode == "thermal" and cap_th is None:
        raise SystemExit("--mode thermal requires --th_video")
    if ckpt_path is None:
        ckpt_path = "models/fusion.pt" if use_fusion else "models/rgb.pt"
    model, run_mode, device, thr, temperature = load_checkpoint(ckpt_path)
    if override_thr is not None:
        thr = float(override_thr)
    if early_detection:
        thr = max(float(early_min_threshold), float(thr) - float(early_threshold_shift))
    persist_target = max(1, int(early_persist_n if early_detection else persist_n))
    hyst_high_eff = float(thr) if early_detection else float(hyst_high)
    hyst_low_eff = float(thr) * 0.6 if early_detection else float(hyst_low)
    hyst_low_eff = max(0.01, min(hyst_low_eff, hyst_high_eff - 1e-3))

    need_cam = bool(
        save_heatmaps
        or save_masks
        or save_polygons
        or cam_stats_only
        or (
            temporal_guard
            and (
                float(small_fire_boost) != 1.0
                or float(growth_upscale) != 1.0
                or float(texture_prob_max) > 0.0
                or float(min_component_area) > 0.0
            )
        )
    )
    cam_extractor = None
    if need_cam:
        try:
            from torchcam.methods import SmoothGradCAMpp

            cam_extractor = SmoothGradCAMpp(model, target_layer=cam_layer)
        except Exception:
            cam_extractor = None
    amp_cuda = bool(
        use_fp16
        and device == "cuda"
        and not need_cam
    )

    total = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    rows = []
    prob_buffer = []
    ema_prob = 0.5
    mask_ema = None
    prev_largest_area = 0.0
    track_id = 0
    win = max(1, int(smooth_window))
    idx = 0
    prev_gray = None
    hyst_fire = False
    persist_run = 0
    n_processed = 0
    modal_agreement = float("nan")

    pbar = tqdm(total=total if total > 0 else None, desc=f"Infer ({run_mode})")
    while True:
        ok, fr = cap_rgb.read()
        if not ok:
            break
        modal_agreement = float("nan")
        th_fr = None
        if cap_th is not None:
            ok2, th_fr = cap_th.read()
            if not ok2:
                th_fr = None

        if idx % step_frames != 0:
            idx += 1
            pbar.update(1)
            continue

        small = cv2.resize(fr, (160, 120))
        small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        scene_changed = False
        mae_scene = 0.0
        kl_scene = 0.0
        if temporal_guard and prev_gray is not None:
            mae_scene = float(np.mean(np.abs(small_gray - prev_gray)))
            if mae_scene > float(scene_thresh):
                scene_changed = True
            if use_kl_scene:
                kl_scene = _sym_kl_hist(small_gray, prev_gray)
                if kl_scene > float(kl_hist_thresh):
                    scene_changed = True
        if temporal_guard and scene_changed:
            prob_buffer.clear()
            mask_ema = None
            prev_largest_area = 0.0
            persist_run = 0
            hyst_fire = False
            track_id = 0

        rgb_arr, rgb_base = prep_rgb(fr, size=size)
        if run_mode == "fusion":
            if th_fr is None:
                idx += 1
                pbar.update(1)
                continue
            th_arr, _ = prep_thermal(th_fr, size=size)
            if enable_modal_agreement:
                modal_agreement = _safe_corr2d(rgb_arr.mean(axis=0), th_arr[0])
            x = torch.tensor(np.concatenate([rgb_arr, th_arr], axis=0)[None, ...], dtype=torch.float32).to(device)
            base_for_overlay = rgb_base
        elif run_mode == "thermal":
            if th_fr is None:
                idx += 1
                pbar.update(1)
                continue
            th_arr, _ = prep_thermal(th_fr, size=size)
            x = torch.tensor(th_arr[None, ...], dtype=torch.float32).to(device)
            base_for_overlay = rgb_base
        else:
            x = torch.tensor(rgb_arr[None, ...], dtype=torch.float32).to(device)
            base_for_overlay = rgb_base

        probs_tta = []
        if cam_extractor is not None:
            # Grad-CAM hooks need activations with grad; no_grad breaks register_hook.
            x_in = x.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                logits = model(x_in)
                scores_cal = logits / max(1e-6, temperature)
                prob_raw = torch.softmax(scores_cal, dim=1)[0, 1].item()
                probs_tta.append(prob_raw)
                if use_tta:
                    x_flip = torch.flip(x.detach().clone().requires_grad_(True), dims=[3])
                    logits_f = model(x_flip)
                    scores_cal_f = logits_f / max(1e-6, temperature)
                    prob_f = torch.softmax(scores_cal_f, dim=1)[0, 1].item()
                    probs_tta.append(prob_f)
        else:
            with torch.no_grad():
                if amp_cuda:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        logits = model(x)
                        scores_cal = logits / max(1e-6, temperature)
                        prob_raw = torch.softmax(scores_cal, dim=1)[0, 1].item()
                        probs_tta.append(prob_raw)
                        if use_tta:
                            x_flip = torch.flip(x, dims=[3])
                            logits_f = model(x_flip)
                            scores_cal_f = logits_f / max(1e-6, temperature)
                            prob_f = torch.softmax(scores_cal_f, dim=1)[0, 1].item()
                            probs_tta.append(prob_f)
                else:
                    logits = model(x)
                    scores_cal = logits / max(1e-6, temperature)
                    prob_raw = torch.softmax(scores_cal, dim=1)[0, 1].item()
                    probs_tta.append(prob_raw)
                    if use_tta:
                        x_flip = torch.flip(x, dims=[3])
                        logits_f = model(x_flip)
                        scores_cal_f = logits_f / max(1e-6, temperature)
                        prob_f = torch.softmax(scores_cal_f, dim=1)[0, 1].item()
                        probs_tta.append(prob_f)
        prob_model = float(np.mean(probs_tta))
        prob_post = (
            prob_model * float(scene_conf_scale)
            if (temporal_guard and scene_changed)
            else prob_model
        )
        if (
            run_mode == "fusion"
            and enable_modal_agreement
            and modal_agreement == modal_agreement
            and modal_agreement < float(modal_agreement_min_corr)
        ):
            prob_post *= float(modal_agreement_penalty)
        if temporal_guard and scene_changed:
            ema_prob = prob_post
            prob_buffer.clear()
            prob_buffer.append(prob_post)
        else:
            prob_buffer.append(prob_post)
            if len(prob_buffer) > win:
                prob_buffer = prob_buffer[-win:]
            ema_prob = _smooth_ema(prob_post, ema_prob, ema_alpha)
        prob_ma = float(np.mean(prob_buffer))
        prob = ema_prob

        mean_intensity = 0.0
        top10 = 0.0
        area60 = 0.0
        heat_path = ""
        mask_path = ""
        polygon_path = ""
        largest_component_area = 0.0
        num_components = 0
        centroid_x = 0.0
        centroid_y = 0.0
        growth_rate = 0.0
        cam_for_stats = None
        cam_ok_frame = False
        if cam_extractor is not None and logits is not None:
            try:
                cam = cam_extractor(class_idx=1, scores=logits)[0].squeeze().detach().cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
                if mask_ema is None:
                    mask_ema = cam.astype(np.float32)
                else:
                    mask_ema = float(ema_alpha) * mask_ema + (1.0 - float(ema_alpha)) * cam.astype(np.float32)
                cam_for_stats = mask_ema
                mean_intensity = float(cam_for_stats.mean())
                top10 = float(
                    np.mean(np.sort(cam_for_stats.ravel())[-max(1, int(0.1 * cam_for_stats.size)):])
                )
                area60 = float((cam_for_stats > 0.6).mean())
                st = stats_from_soft_map(cam_for_stats)
                largest_component_area = float(st["largest_component_area"])
                num_components = int(st["num_components"])
                centroid_x = float(st["centroid_x_norm"])
                centroid_y = float(st["centroid_y_norm"])
                growth_rate = float(largest_component_area - prev_largest_area)
                prev_largest_area = largest_component_area
                track_id = 1 if largest_component_area > 0.002 else 0
                if save_heatmaps:
                    H, W = base_for_overlay.shape[:2]
                    cam_resized = cv2.resize(cam_for_stats, (W, H), interpolation=cv2.INTER_CUBIC)
                    cam_u8 = (np.clip(cam_resized, 0, 1) * 255).astype(np.uint8)
                    heat_colored = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
                    heat_colored = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)
                    overlay = (0.6 * base_for_overlay + 0.4 * heat_colored).astype(np.uint8)
                    heat_path = str(heatmap_dir / f"frame_{idx:06d}.png")
                    Image.fromarray(overlay).save(heat_path)
                if save_masks:
                    H, W = base_for_overlay.shape[:2]
                    m_u8 = (np.clip(cv2.resize(cam_for_stats, (W, H)), 0, 1) * 255).astype(np.uint8)
                    mask_path = str(mask_dir / f"frame_{idx:06d}.png")
                    Image.fromarray(m_u8).save(mask_path)
                if save_polygons and num_components > 0:
                    poly = {
                        "frame_idx": int(idx),
                        "centroid_x_norm": centroid_x,
                        "centroid_y_norm": centroid_y,
                        "largest_component_area": largest_component_area,
                        "num_components": num_components,
                    }
                    polygon_path = str(polygon_dir / f"frame_{idx:06d}.json")
                    with open(polygon_path, "w", encoding="utf-8") as f:
                        json.dump(poly, f)
                cam_ok_frame = True
            except Exception:
                pass

        decision_prob = float(prob)
        if temporal_guard:
            if (
                cam_ok_frame
                and 0.0 < float(largest_component_area) < float(small_fire_area_max)
            ):
                decision_prob *= float(small_fire_boost)
            if (
                cam_ok_frame
                and float(min_component_area) > 0.0
                and largest_component_area < float(min_component_area)
            ):
                decision_prob = min(decision_prob, float(hyst_low_eff) - 0.01)
            if cam_ok_frame and n_processed > 0:
                if growth_rate > 0:
                    decision_prob *= float(growth_upscale)
                else:
                    decision_prob *= float(growth_downscale)
            if (
                decision_prob < float(texture_prob_max)
                and top10 > float(texture_top10_min)
            ):
                decision_prob = 0.0
            decision_prob = float(np.clip(decision_prob, 0.0, 1.0))

            if decision_prob > float(hyst_high_eff):
                hyst_fire = True
            elif decision_prob < float(hyst_low_eff):
                hyst_fire = False

            if hyst_fire:
                persist_run += 1
            else:
                persist_run = 0
            fire_event = 1 if persist_run >= int(persist_target) else 0
            pred_fire = int(hyst_fire)
        else:
            pred_fire = 1 if prob >= thr else 0
            fire_event = 0
            persist_run = 0

        prev_gray = small_gray.copy()
        n_processed += 1

        rows.append({
            "frame_idx": idx,
            "prob_fire_raw": prob_model,
            "prob_fire": prob,
            "decision_prob": decision_prob,
            "pred_fire": pred_fire,
            "threshold_used": thr,
            "hyst_high_used": float(hyst_high_eff),
            "hyst_low_used": float(hyst_low_eff),
            "persist_n_used": int(persist_target),
            "early_detection": int(bool(early_detection)),
            "infer_temporal_applied": int(bool(temporal_guard)),
            "scene_changed": int(bool(scene_changed)),
            "modal_agreement": float(modal_agreement) if modal_agreement == modal_agreement else "",
            "mae_scene": mae_scene,
            "kl_scene": kl_scene,
            "hyst_fire": int(bool(hyst_fire)),
            "fire_run_len": int(persist_run),
            "fire_event": int(fire_event),
            "intensity_mean": mean_intensity,
            "intensity_top10": top10,
            "area_heat_gt_0_6": area60,
            "heatmap_path": heat_path,
            "mask_path": mask_path,
            "polygon_path": polygon_path,
            "largest_component_area": largest_component_area,
            "num_components": num_components,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "growth_rate": growth_rate,
            "track_id": int(track_id),
        })
        idx += 1
        pbar.update(1)

    pbar.close()
    cap_rgb.release()
    if cap_th is not None:
        cap_th.release()

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv
