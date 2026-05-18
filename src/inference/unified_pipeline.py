import os
from pathlib import Path
import cv2
import pandas as pd
import json

from src.model_loader import load_dual_branch_model, predict_fire_probability
from src.frame_preprocess import build_fusion_tensor
from src.geospatial.pixel_projection import pixel_to_geo_point, estimate_area_from_pixel_area
from src.geospatial.fire_tracker import FireTracker
from src.postgis_writer import PostgisWriter
from src.segmentation.thermal_threshold import create_fire_mask_from_thermal
from src.segmentation.contours import extract_fire_regions
from src.segmentation.temporal_smoothing import TemporalMaskSmoother
from src.segmentation.mask_filter import filter_fire_regions
from src.segmentation.visualization import overlay_mask_on_frame, draw_fire_regions_on_frame
from src.telemetry_provider import DjiCsvTelemetryProvider
from src.risk.fuel_scorer import FuelScorer

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def run_unified_pipeline(
    rgb_video_path: str,
    thermal_video_path: str,
    checkpoint_path: str,
    output_dir: Path,
    telemetry_csv: str = None,
    fuel_model_path: str = None,
    frame_step: int = 10,
    max_frames: int = None,
    percentile: float = 97.0,
    min_area: int = 300,
    threshold_mode: str = "hybrid",
    threshold_value: float = 210.0,
    write_postgis: bool = False,
    db_config: dict = None,
    progress_callback=None
):
    """
    Unified pipeline integrating Dual-Branch Model, Thermal Thresholding, 
    Telemetry, PostGIS, and GEE Fuel Scorer.
    """
    out_dir = Path(output_dir)
    fire_frames_dir = out_dir / "fire_frames"
    overlays_dir = out_dir / "overlays"
    masks_dir = out_dir / "masks"

    ensure_dir(out_dir)
    ensure_dir(fire_frames_dir)
    ensure_dir(overlays_dir)
    ensure_dir(masks_dir)

    print("[UnifiedPipeline] Yükleniyor: Dual Branch Model...")
    model, meta = load_dual_branch_model(checkpoint_path)
    base_threshold = meta["threshold"]
    
    print("[UnifiedPipeline] Yükleniyor: Fuel Scorer (GEE)...")
    fuel_model_path = fuel_model_path or str(Path(__file__).resolve().parents[1] / "models" / "fuel_scorer_model.pkl")
    fuel_scorer = FuelScorer(model_path=fuel_model_path, use_gee=True)

    rgb_cap = cv2.VideoCapture(rgb_video_path)
    thermal_cap = cv2.VideoCapture(thermal_video_path)

    if not rgb_cap.isOpened():
        raise RuntimeError(f"RGB video açılamadı: {rgb_video_path}")
    if not thermal_cap.isOpened():
        raise RuntimeError(f"Thermal video açılamadı: {thermal_video_path}")

    rgb_total = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    thermal_total = int(thermal_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(rgb_total, thermal_total)

    fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    video_duration_s = total_frames / fps

    telemetry_provider = None
    if telemetry_csv and os.path.exists(telemetry_csv):
        telemetry_provider = DjiCsvTelemetryProvider(csv_path=telemetry_csv, video_duration_s=video_duration_s)
        print(f"[UnifiedPipeline] Telemetri aktif: {telemetry_csv}")

    postgis_writer = None
    if write_postgis and db_config:
        postgis_writer = PostgisWriter(**db_config)
        print("[UnifiedPipeline] PostGIS bağlantısı aktif.")

    mask_smoother = TemporalMaskSmoother(history_size=3, vote_threshold=0.34)
    fire_tracker = FireTracker(match_distance_m=80.0, max_missing_frames=60)

    results = []
    events = [] # For UI summary
    processed_count = 0
    frame_idx = 0

    while True:
        rgb_ret, rgb_frame = rgb_cap.read()
        thermal_ret, thermal_frame = thermal_cap.read()

        if not rgb_ret or not thermal_ret or frame_idx >= total_frames:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        video_time_s = frame_idx / fps
        
        # 1. Telemetri
        telemetry_info = {}
        if telemetry_provider:
            telemetry_info = telemetry_provider.get_current(video_time_s)
            
        lat = telemetry_info.get("latitude")
        lon = telemetry_info.get("longitude")

        # 2. Dual Branch Model Inference
        input_tensor = build_fusion_tensor(rgb_frame, thermal_frame, meta["input_size"], meta["device"])
        kamera_skoru = predict_fire_probability(model, input_tensor, meta["temperature"])

        # 3. Bitki (Fuel) Scorer Entegrasyonu
        fuel_score = fuel_scorer.get_score(lat, lon)
        MAX_ETKI = 0.05
        modifiye_deger = ((fuel_score - 0.50) / 0.50) * MAX_ETKI
        
        # Final probability
        final_prob = max(0.0, min(1.0, kamera_skoru + modifiye_deger))
        is_fire = final_prob >= base_threshold

        base_result = {
            "frame_idx": frame_idx,
            "video_time_s": video_time_s,
            "timestamp_sec": video_time_s,
            "kamera_prob": kamera_skoru,
            "fuel_score": fuel_score,
            "fire_probability": final_prob,
            "decision_prob": final_prob,
            "threshold": base_threshold,
            "prediction": "fire" if is_fire else "no_fire",
            "pred_fire": 1 if is_fire else 0,
            "alarm_state": "confirmed" if is_fire else "ok"
        }
        base_result.update(telemetry_info)

        if postgis_writer:
            postgis_writer.insert_drone_frame_point(base_result)

        # 4. Termal Thresholding & Geolocation
        if is_fire:
            raw_mask, thermal_norm = create_fire_mask_from_thermal(
                thermal_frame,
                threshold_mode=threshold_mode,
                threshold_value=threshold_value,
                percentile_value=percentile,
                min_area=min_area,
                kernel_size=9,
                use_strong_closing=False,
                dilate_iterations=0,
            )
            mask = mask_smoother.update(raw_mask)
            fire_regions = extract_fire_regions(mask, min_contour_area=min_area)
            fire_regions = filter_fire_regions(fire_regions, mask.shape, max_region_area_ratio=0.35)

            thermal_bgr = cv2.cvtColor(thermal_norm, cv2.COLOR_GRAY2BGR)
            thermal_overlay = overlay_mask_on_frame(thermal_bgr, mask)
            thermal_contours = draw_fire_regions_on_frame(thermal_overlay, fire_regions)

            frame_name = f"frame_{frame_idx:06d}"
            fire_frame_path = fire_frames_dir / f"{frame_name}.jpg"
            mask_path = masks_dir / f"{frame_name}.png"
            overlay_path = overlays_dir / f"{frame_name}.jpg"
            cv2.imwrite(str(fire_frame_path), thermal_bgr)
            cv2.imwrite(str(mask_path), mask)
            cv2.imwrite(str(overlay_path), thermal_contours)

            if len(fire_regions) == 0:
                base_result.update({
                    "region_id": None,
                    "pixel_area": None,
                    "approx_area_m2": None,
                    "fire_latitude": None,
                    "fire_longitude": None,
                    "fire_track_id": None,
                    "fire_frame_path": str(fire_frame_path),
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                })
                results.append(base_result)
            else:
                image_height, image_width = mask.shape[:2]
                for region in fire_regions:
                    fire_lat, fire_lon, approx_area_m2, track_id = None, None, None, None
                    if lat is not None and lon is not None:
                        fire_lat, fire_lon = pixel_to_geo_point(
                            region["centroid_x"], region["centroid_y"], image_width, image_height,
                            lat, lon, telemetry_info.get("altitude_m"), 73.7, 53.1, telemetry_info.get("drone_yaw") or 0.0
                        )
                        approx_area_m2 = estimate_area_from_pixel_area(
                            region["pixel_area"], image_width, image_height, telemetry_info.get("altitude_m"), 73.7, 53.1
                        )
                        track_id = fire_tracker.update(fire_lat, fire_lon, frame_idx, approx_area_m2)

                    result = base_result.copy()
                    result.update({
                        "region_id": region["region_id"],
                        "pixel_area": region.get("pixel_area"),
                        "approx_area_m2": approx_area_m2,
                        "fire_latitude": fire_lat,
                        "fire_longitude": fire_lon,
                        "fire_track_id": track_id,
                        "fire_frame_path": str(fire_frame_path),
                        "mask_path": str(mask_path),
                        "overlay_path": str(overlay_path),
                    })

                    if postgis_writer:
                        postgis_writer.insert_fire_observation(result)
                        postgis_writer.upsert_active_fire_track(result)

                    results.append(result)
                    
                    # Store event for UI mapping export
                    if fire_lat is not None:
                        events.append({
                            "event_id": str(track_id),
                            "timestamp": video_time_s,
                            "fire_detected": True,
                            "risk_level": "confirmed",
                            "probability": final_prob,
                            "latitude": fire_lat,
                            "longitude": fire_lon,
                            "area_m2": approx_area_m2
                        })

        else:
            mask_smoother.reset()
            results.append(base_result)

        processed_count += 1
        if progress_callback:
            progress_callback(processed_count, (max_frames if max_frames else (total_frames // frame_step)))

        if max_frames and processed_count >= max_frames:
            break

        frame_idx += 1

    rgb_cap.release()
    thermal_cap.release()
    if postgis_writer: postgis_writer.close()

    df_scored = pd.DataFrame(results)
    
    # UI için uyumluluk sütunları ekle
    if 'prob_fire' not in df_scored.columns and 'decision_prob' in df_scored.columns:
        df_scored['prob_fire'] = df_scored['decision_prob']

    scored_csv = out_dir / "video_predictions_scored.csv"
    mapping_export_json = out_dir / "mapping_export.json"
    
    df_scored.to_csv(scored_csv, index=False)
    with open(mapping_export_json, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)

    return {
        "df_scored": df_scored,
        "df_events": pd.DataFrame(events), # Mock events DF for UI compatibility
        "threshold_used": base_threshold,
        "hyst_high_used": base_threshold,
        "hyst_low_used": base_threshold * 0.6,
        "scored_csv": str(scored_csv),
        "mapping_export_json": str(mapping_export_json),
    }
