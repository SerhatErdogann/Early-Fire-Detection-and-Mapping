"""
Video inference with optional EMA smoothing and TTA for more stable predictions on drone footage.
"""
from __future__ import annotations

import json
import time
import cv2
import numpy as np
import pandas as pd
import torch
from collections.abc import Callable
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from .model_loader import load_checkpoint
from .postprocess import stats_from_soft_map
from .preprocess import prep_rgb, prep_thermal
from .alarm import AlarmConfig, AlarmStateMachine
from .capture_utils import open_video_capture
from .frame_sampling import AdaptiveFrameSampler

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


def _to_device_batch(x_np: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(x_np)).unsqueeze(0).float()
    if device == "cuda":
        x = x.pin_memory()
        return x.to(device, non_blocking=True)
    return x.to(device)


def run_video_inference(
    rgb_video_path=None,
    th_video_path=None,
    ckpt_fusion: str | None = None,
    ckpt_rgb: str | None = None,
    ckpt_thermal: str | None = None,
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
    adaptive_step: bool = False,
    adaptive_min_step: int = 1,
    adaptive_max_step: int = 12,
    adaptive_low_motion: float = 0.03,
    adaptive_high_risk: float = 0.65,
    benchmark: bool = False,
    benchmark_out: str | None = None,
    prob_temporal_blend: float = 0.0,
    burst_min_frames: int = 3,
    burst_threshold_frac: float = 1.0,
    auto_step_long_video: bool = False,
    long_video_seconds: float = 600.0,
    long_video_step_scale: float = 2.0,
    max_step_cap: int = 64,
    stream_buffer_reduce: bool = True,
    infer_batch_size: int = 1,
    progress_callback: Callable[[int, int | None], None] | None = None,
):
    """
    Run fire classification on video (RGB only or RGB+thermal).
    Uses optional moving-average + EMA blend for probability smoothing, plus a
    consecutive-frame rule independent of hysteresis alarms.
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

    if int(infer_batch_size) != 1:
        print(
            f"[video] infer_batch_size={infer_batch_size} not supported "
            "(temporal_guard/CAM/TTA need per-frame); using 1."
        )
    burst_min_frames_eff = max(1, int(burst_min_frames))
    burst_threshold_frac_eff = float(np.clip(float(burst_threshold_frac), 0.05, 1.5))

    cap_rgb = None
    if rgb_video_path:
        try:
            cap_rgb = open_video_capture(str(rgb_video_path).strip(), buffer_reduce=bool(stream_buffer_reduce))
        except Exception as exc:
            raise SystemExit(f"Cannot open RGB video/source: {rgb_video_path} ({exc})")

    cap_th = None
    if th_video_path:
        try:
            cap_th = open_video_capture(str(th_video_path).strip(), buffer_reduce=bool(stream_buffer_reduce))
        except Exception as exc:
            raise SystemExit(f"Cannot open thermal video/source: {th_video_path} ({exc})")

    if cap_rgb is None and cap_th is None:
        raise SystemExit("No input video provided. Provide rgb_video_path and/or th_video_path.")

    def _exists(p: str | None) -> bool:
        if not p:
            return False
        try:
            return Path(p).exists()
        except Exception:
            return False

    # Load available checkpoints (fusion preferred). Fusion can also act as fallback by zero-filling missing modality.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models: dict[str, dict] = {}
    if _exists(ckpt_fusion):
        m, _m_mode, _dev, thr_f, temp_f = load_checkpoint(str(ckpt_fusion))
        models["fusion"] = {"model": m, "thr": float(thr_f), "temp": float(temp_f), "ckpt": str(ckpt_fusion), "in_ch": 4}
    if _exists(ckpt_rgb):
        m, _m_mode, _dev, thr_r, temp_r = load_checkpoint(str(ckpt_rgb))
        models["rgb"] = {"model": m, "thr": float(thr_r), "temp": float(temp_r), "ckpt": str(ckpt_rgb), "in_ch": 3}
    if _exists(ckpt_thermal):
        m, _m_mode, _dev, thr_t, temp_t = load_checkpoint(str(ckpt_thermal))
        models["thermal"] = {"model": m, "thr": float(thr_t), "temp": float(temp_t), "ckpt": str(ckpt_thermal), "in_ch": 1}

    if not models:
        raise SystemExit("No checkpoints found. Provide at least one of ckpt_fusion/ckpt_rgb/ckpt_thermal.")

    # default override threshold (applies to all modes if set)
    override_thr = float(override_thr) if override_thr is not None else None
    # Seed `thr` from the first available checkpoint so early-detection threshold
    # shifting has a defined starting point even before we know which per-frame
    # model will run. Re-set per frame inside the loop using the selected model.
    _first_model_pack = next(iter(models.values()))
    thr = float(_first_model_pack["thr"])
    if early_detection:
        thr = max(float(early_min_threshold), float(thr) - float(early_threshold_shift))
    persist_target = max(1, int(early_persist_n if early_detection else persist_n))
    hyst_high_eff = float(thr) if early_detection else float(hyst_high)
    hyst_low_eff = float(thr) * 0.6 if early_detection else float(hyst_low)
    hyst_low_eff = max(0.01, min(hyst_low_eff, hyst_high_eff - 1e-3))

    # CAM / spatial filtering is opt-in: only switched on when the caller
    # explicitly asks for heatmap/mask/polygon export or cam-stats. Spatial
    # rules (small_fire_boost, min_component_area, ...) only fire when
    # cam_ok_frame=True inside the loop, so keeping CAM off by default disables
    # them safely and also lets FP16 inference run.
    need_cam = bool(save_heatmaps or save_masks or save_polygons or cam_stats_only)
    cam_extractor = None
    if need_cam:
        try:
            from torchcam.methods import SmoothGradCAMpp

            # Attach CAM to whichever model we loaded first (best-effort
            # analytics; CAM target layer varies per backbone).
            _cam_model = next(iter(models.values()))["model"]
            cam_extractor = SmoothGradCAMpp(_cam_model, target_layer=cam_layer)
        except Exception as e:
            print(f"[video] CAM setup failed ({type(e).__name__}): {e}; continuing without CAM.")
            cam_extractor = None
    # Determine starting preference (auto: fusion -> rgb -> thermal)
    mode = (mode or "auto").lower().strip()
    if mode not in ("auto", "fusion", "rgb", "thermal"):
        mode = "auto"

    def _pref_list() -> list[str]:
        if mode == "fusion":
            return ["fusion", "rgb", "thermal"]
        if mode == "rgb":
            return ["rgb", "fusion", "thermal"]
        if mode == "thermal":
            return ["thermal", "fusion", "rgb"]
        return ["fusion", "rgb", "thermal"]

    pref = _pref_list()

    amp_cuda = bool(
        use_fp16
        and device == "cuda"
        and not need_cam
    )

    primary = cap_rgb if cap_rgb is not None else cap_th
    total = int(primary.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(primary.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = fps if fps > 1e-6 else 30.0
    rows = []
    prob_buffer = []
    ema_prob = 0.5
    mask_ema = None
    prev_largest_area = 0.0
    track_id = 0
    win = max(1, int(smooth_window))
    step = max(1, int(step_frames))
    duration_est_s = float(total) / float(fps) if total > 0 else 0.0
    step_pre_auto = step
    if (
        auto_step_long_video
        and duration_est_s >= float(long_video_seconds)
        and step_pre_auto < int(max_step_cap)
    ):
        scaled = max(step_pre_auto, int(round(step_pre_auto * float(long_video_step_scale))))
        step = min(int(max_step_cap), scaled)
        print(
            f"[video] long video (~{duration_est_s:.0f}s): step_frames {step_pre_auto} -> {step} "
            f"(<= max_step_cap={max_step_cap})"
        )
    alarm_machine = AlarmStateMachine(
        AlarmConfig(
            high_threshold=float(hyst_high_eff),
            low_threshold=float(hyst_low_eff),
            suspect_threshold=float(hyst_low_eff),
            confirm_frames=int(persist_target),
            cooldown_frames=max(2, int(persist_target)),
        )
    )
    sampler = AdaptiveFrameSampler(
        base_step=step,
        min_step=max(1, int(adaptive_min_step)),
        max_step=max(int(adaptive_max_step), step),
        low_motion_threshold=float(adaptive_low_motion),
        high_risk_threshold=float(adaptive_high_risk),
    )
    perf = {
        "decode_s": 0.0,
        "preprocess_s": 0.0,
        "infer_s": 0.0,
        "postprocess_s": 0.0,
        "processed_frames": 0,
        "skipped_frames": 0,
    }
    idx = 0
    prev_gray = None
    hyst_fire = False
    persist_run = 0
    burst_run = 0
    n_processed = 0
    modal_agreement = float("nan")

    pbar = tqdm(total=total if total > 0 else None, desc="Infer (auto)")
    while True:
        t_decode0 = time.perf_counter()
        fr = None
        th_fr = None
        if cap_rgb is not None:
            ok, fr = cap_rgb.read()
            if not ok:
                fr = None
        if cap_th is not None:
            ok2, th_fr = cap_th.read()
            if not ok2:
                th_fr = None

        # Stop when primary stream ends
        if primary is cap_rgb and fr is None:
            break
        if primary is cap_th and th_fr is None:
            break
        modal_agreement = float("nan")
        # For RGB-only runs, fr must exist; for thermal-only runs, th_fr must exist.
        if fr is None and th_fr is None:
            break

        # Scene-change uses RGB if available, else thermal converted to gray
        if fr is not None:
            small = cv2.resize(fr, (160, 120))
            small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            small = cv2.resize(th_fr, (160, 120))
            if len(small.shape) == 3:
                small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            else:
                small_gray = small.astype(np.float32) / 255.0
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
            burst_run = 0
            hyst_fire = False
            track_id = 0

        perf["decode_s"] += time.perf_counter() - t_decode0

        t_pre0 = time.perf_counter()
        rgb_arr = None
        rgb_base = None
        th_arr = None
        if fr is not None:
            rgb_arr, rgb_base = prep_rgb(fr, size=size)
        if th_fr is not None:
            th_arr, _ = prep_thermal(th_fr, size=size)

        # Pick mode_used for this frame, with fallback if one stream missing mid-video.
        have_rgb = rgb_arr is not None
        have_th = th_arr is not None

        mode_used = None
        for m in pref:
            if m == "fusion" and have_rgb and have_th and ("fusion" in models):
                mode_used = "fusion"
                break
            if m == "rgb" and have_rgb and (("rgb" in models) or ("fusion" in models)):
                mode_used = "rgb"
                break
            if m == "thermal" and have_th and (("thermal" in models) or ("fusion" in models)):
                mode_used = "thermal"
                break
        if mode_used is None:
            idx += 1
            pbar.update(1)
            continue

        # Build input tensor for selected model
        if mode_used == "fusion":
            if enable_modal_agreement and have_rgb and have_th:
                modal_agreement = _safe_corr2d(rgb_arr.mean(axis=0), th_arr[0])
            x_np = np.concatenate([rgb_arr, th_arr], axis=0)
            model_pack = models["fusion"]
        elif mode_used == "rgb":
            if "rgb" in models:
                x_np = rgb_arr
                model_pack = models["rgb"]
            else:
                # fusion fallback with missing thermal = zeros
                z = np.zeros((1, rgb_arr.shape[1], rgb_arr.shape[2]), dtype=np.float32)
                x_np = np.concatenate([rgb_arr, z], axis=0)
                model_pack = models["fusion"]
        else:  # thermal
            if "thermal" in models:
                x_np = th_arr
                model_pack = models["thermal"]
            else:
                # fusion fallback with missing rgb = zeros
                z = np.zeros((3, th_arr.shape[1], th_arr.shape[2]), dtype=np.float32)
                x_np = np.concatenate([z, th_arr], axis=0)
                model_pack = models["fusion"]

        model = model_pack["model"]
        temperature = float(model_pack["temp"])
        thr = float(model_pack["thr"])
        if override_thr is not None:
            thr = float(override_thr)
        base_for_overlay = rgb_base
        perf["preprocess_s"] += time.perf_counter() - t_pre0

        t_inf0 = time.perf_counter()
        probs_tta = []
        if cam_extractor is not None:
            # Grad-CAM hooks need activations with grad; no_grad breaks register_hook.
            x = _to_device_batch(x_np, device=device)
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
            with torch.inference_mode():
                x = _to_device_batch(x_np, device=device)
                if amp_cuda:
                    try:
                        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                    except (TypeError, AttributeError):
                        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
                    with autocast_ctx:
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
        perf["infer_s"] += time.perf_counter() - t_inf0
        prob_model = float(np.mean(probs_tta))
        prob_post = (
            prob_model * float(scene_conf_scale)
            if (temporal_guard and scene_changed)
            else prob_model
        )
        if (
            mode_used == "fusion"
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
        prob_ma = float(np.mean(prob_buffer)) if prob_buffer else float(prob_post)
        blend_w = float(np.clip(prob_temporal_blend, 0.0, 1.0))
        prob_smooth = (1.0 - blend_w) * float(ema_prob) + blend_w * prob_ma
        burst_thr = float(thr) * burst_threshold_frac_eff
        if prob_ma >= burst_thr:
            burst_run += 1
        else:
            burst_run = 0
        pred_fire_burst = int(burst_run >= burst_min_frames_eff)
        prob_fire_ema_val = float(ema_prob)
        prob_fire_ma_val = float(prob_ma)
        prob = float(prob_smooth)

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
                # Faster than full sort: top 10% mean via partition.
                flat = cam_for_stats.ravel()
                k10 = max(1, int(0.1 * flat.size))
                top10 = float(np.mean(np.partition(flat, flat.size - k10)[-k10:]))
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
        t_post0 = time.perf_counter()
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

            state, fire_event, alarm_conf, alarm_reason = alarm_machine.update(
                decision_prob=decision_prob,
                top10_intensity=top10,
                largest_component_area=largest_component_area,
                scene_changed=scene_changed,
            )
            hyst_fire = state in ("suspected", "confirmed")
            persist_run = int(alarm_machine._high_run)
            # frame-level prediction uses the model's own threshold
            pred_fire = int(float(prob) >= float(thr))
        else:
            # Still write compatible fields
            pred_fire = int(float(prob) >= float(thr))
            fire_event = 0
            persist_run = 0
            state = "idle"
            alarm_conf = float(prob)
            alarm_reason = "temporal_guard_disabled"

        prev_gray = small_gray.copy()
        n_processed += 1
        perf["processed_frames"] += 1
        perf["postprocess_s"] += time.perf_counter() - t_post0

        rows.append({
            "frame_idx": idx,
            "timestamp_sec": float(idx) / float(fps),
            "prob_fire_raw": prob_model,
            "prob_fire_ma": prob_fire_ma_val,
            "prob_fire_ema": prob_fire_ema_val,
            "prob_fire": prob,
            "burst_run_len": int(burst_run),
            "pred_fire_burst_consec": pred_fire_burst,
            "decision_prob": decision_prob,
            "pred_fire": pred_fire,
            "mode_used": str(mode_used),
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
            "alarm_state": state,
            "alarm_confidence": float(alarm_conf),
            "alarm_reason": alarm_reason,
        })
        if progress_callback is not None:
            try:
                est: int | None
                if total > 0:
                    est = max(1, int((total + max(1, int(step)) - 1) // max(1, int(step))))
                else:
                    est = None
                progress_callback(int(len(rows)), est)
            except Exception:
                pass
        idx += 1
        pbar.update(1)

        step_now = step
        if adaptive_step:
            step_now = sampler.update(
                motion_mae=float(mae_scene),
                decision_prob=float(decision_prob),
                alarm_state=state,
            )

        if step_now > 1:
            reached_end = False
            for _ in range(step_now - 1):
                t_dec_skip0 = time.perf_counter()
                ok_skip = True
                if cap_rgb is not None:
                    ok_skip = cap_rgb.grab()
                if cap_th is not None:
                    cap_th.grab()
                if not ok_skip:
                    reached_end = True
                    break
                perf["decode_s"] += time.perf_counter() - t_dec_skip0
                idx += 1
                pbar.update(1)
                perf["skipped_frames"] += 1
            if reached_end:
                break

    pbar.close()
    if cap_rgb is not None:
        cap_rgb.release()
    if cap_th is not None:
        cap_th.release()

    if int(perf.get("processed_frames", 0)) == 0:
        raise RuntimeError(
            "No frames processed during inference. "
            f"mode={mode} rgb_opened={bool(cap_rgb is not None)} th_provided={bool(th_video_path)} "
            f"th_opened={bool(cap_th is not None)} processed={perf.get('processed_frames')} skipped={perf.get('skipped_frames')} "
            "If fusion/thermal mode was used, the thermal stream may be shorter/unsynced."
        )

    # Summary (counts per mode_used)
    try:
        df_tmp = pd.DataFrame(rows) if rows else pd.DataFrame()
        n_pred_fire = int(df_tmp["pred_fire"].sum()) if (not df_tmp.empty and "pred_fire" in df_tmp.columns) else 0
        n_fire_event = int(df_tmp["fire_event"].sum()) if (not df_tmp.empty and "fire_event" in df_tmp.columns) else 0
        mode_counts = df_tmp["mode_used"].value_counts().to_dict() if (not df_tmp.empty and "mode_used" in df_tmp.columns) else {}
        print(
            "[summary] processed_frames="
            + str(int(perf.get("processed_frames", 0)))
            + " skipped_frames="
            + str(int(perf.get("skipped_frames", 0)))
            + " pred_fire="
            + str(n_pred_fire)
            + " fire_event="
            + str(n_fire_event)
            + f" mode_counts={mode_counts} temporal_guard={bool(temporal_guard)} "
            + f"hyst_high={float(hyst_high_eff):.3f} hyst_low={float(hyst_low_eff):.3f} persist_n={int(persist_target)} "
            + f"step={int(step_frames)} inferred_step_now={step} adaptive_step={bool(adaptive_step)} "
            + f"prob_temporal_blend={prob_temporal_blend}"
        )
    except Exception:
        pass

    # If no frames were processed, still write a parseable CSV with expected columns.
    if rows:
        df_out = pd.DataFrame(rows)
    else:
        df_out = pd.DataFrame(
            columns=[
                "frame_idx",
                "timestamp_sec",
                "prob_fire_raw",
                "prob_fire_ma",
                "prob_fire_ema",
                "prob_fire",
                "burst_run_len",
                "pred_fire_burst_consec",
                "decision_prob",
                "pred_fire",
                "mode_used",
                "threshold_used",
                "hyst_high_used",
                "hyst_low_used",
                "persist_n_used",
                "early_detection",
                "infer_temporal_applied",
                "scene_changed",
                "modal_agreement",
                "mae_scene",
                "kl_scene",
                "hyst_fire",
                "fire_run_len",
                "fire_event",
                "intensity_mean",
                "intensity_top10",
                "area_heat_gt_0_6",
                "heatmap_path",
                "mask_path",
                "polygon_path",
                "largest_component_area",
                "num_components",
                "centroid_x",
                "centroid_y",
                "growth_rate",
                "track_id",
                "alarm_state",
                "alarm_confidence",
                "alarm_reason",
            ]
        )
    df_out.to_csv(out_csv, index=False)
    if benchmark:
        total_s = max(1e-9, perf["decode_s"] + perf["preprocess_s"] + perf["infer_s"] + perf["postprocess_s"])
        bench = {
            "video": str(rgb_video_path),
            "mode": str(mode),
            "device": str(device),
            "processed_frames": int(perf["processed_frames"]),
            "skipped_frames": int(perf["skipped_frames"]),
            "decode_s": float(perf["decode_s"]),
            "preprocess_s": float(perf["preprocess_s"]),
            "infer_s": float(perf["infer_s"]),
            "postprocess_s": float(perf["postprocess_s"]),
            "pipeline_fps_processed": float(perf["processed_frames"] / total_s),
        }
        bench_path = Path(benchmark_out) if benchmark_out else out_csv.with_suffix(".benchmark.json")
        with open(bench_path, "w", encoding="utf-8") as f:
            json.dump(bench, f, indent=2, ensure_ascii=False)
    return out_csv
