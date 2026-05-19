# src/live_video_fire_pipeline.py

import argparse
from pathlib import Path

import cv2
import pandas as pd

from model_loader import (
    load_dual_branch_model,
    predict_fire_probability
)

from frame_preprocess import build_fusion_tensor

from geospatial.pixel_projection import (
    pixel_to_geo_point,
    estimate_area_from_pixel_area
)

from geospatial.fire_tracker import FireTracker, haversine_distance_m
from postgis_writer import PostgisWriter

from segmentation.thermal_threshold import create_fire_mask_from_thermal
from segmentation.contours import extract_fire_regions
from segmentation.temporal_smoothing import TemporalMaskSmoother
from segmentation.mask_filter import filter_fire_regions
from segmentation.region_merge import merge_close_fire_regions
from segmentation.visualization import (
    overlay_mask_on_frame,
    draw_fire_regions_on_frame
)

from telemetry_provider import DjiCsvTelemetryProvider
from risk.fuel_scorer import FuelScorer


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Live-like video fire detection pipeline"
    )

    parser.add_argument("--rgb_video", required=True, help="RGB video path")
    parser.add_argument("--thermal_video", required=True, help="Thermal video path")

    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/dual_branch.pt",
        help="Model checkpoint path"
    )

    parser.add_argument(
        "--telemetry_csv",
        default=None,
        help="Optional DJI telemetry CSV used as simulated live telemetry"
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/live_video_results",
        help="Output directory"
    )

    parser.add_argument(
        "--frame_step",
        type=int,
        default=10,
        help="Process every N frames"
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum processed frame count for testing"
    )

    parser.add_argument(
        "--percentile",
        type=float,
        default=97,
        help="Thermal mask percentile threshold"
    )

    parser.add_argument(
        "--threshold_mode",
        choices=["fixed", "absolute", "percentile", "hybrid"],
        default="hybrid",
        help="Thermal mask thresholding mode"
    )

    parser.add_argument(
        "--threshold_value",
        type=float,
        default=180,
        help="Raw thermal/grayscale threshold used by fixed/absolute/hybrid modes"
    )

    parser.add_argument(
        "--min_area",
        type=int,
        default=130,
        help="Minimum fire region area in pixels"
    )

    parser.add_argument("--write_postgis", action="store_true")
    parser.add_argument("--db_host", default="localhost")
    parser.add_argument("--db_port", type=int, default=5432)
    parser.add_argument("--db_name", default="fire_mapping")
    parser.add_argument("--db_user", default="postgres")
    parser.add_argument("--db_password", default="postgres")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fire_frames_dir = output_dir / "fire_frames"
    overlays_dir = output_dir / "overlays"
    masks_dir = output_dir / "masks"

    ensure_dir(output_dir)
    ensure_dir(fire_frames_dir)
    ensure_dir(overlays_dir)
    ensure_dir(masks_dir)
    ensure_dir(overlays_dir)

    print("Loading Dual Branch Model...")
    model, meta = load_dual_branch_model(args.checkpoint)

    print("Loading Fuel Scorer (Plant Model)...")
    fuel_model_path = str(Path(__file__).parent / "models" / "fuel_scorer_model.pkl")
    fuel_scorer = FuelScorer(model_path=fuel_model_path, use_gee=True)

    print("Model loaded.")
    print("Device:", meta["device"])
    print("Threshold:", meta["threshold"])
    print("Input size:", meta["input_size"])
    print("Temperature:", meta["temperature"])

    rgb_cap = cv2.VideoCapture(args.rgb_video)
    thermal_cap = cv2.VideoCapture(args.thermal_video)

    if not rgb_cap.isOpened():
        raise RuntimeError(f"RGB video could not be opened: {args.rgb_video}")

    if not thermal_cap.isOpened():
        raise RuntimeError(f"Thermal video could not be opened: {args.thermal_video}")

    rgb_total = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    thermal_total = int(thermal_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(rgb_total, thermal_total)

    fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    print("RGB total frames:", rgb_total)
    print("Thermal total frames:", thermal_total)
    print("Using total frames:", total_frames)
    print("FPS:", fps)

    video_duration_s = total_frames / fps

    telemetry_provider = None

    if args.telemetry_csv:
        telemetry_provider = DjiCsvTelemetryProvider(
            csv_path=args.telemetry_csv,
            video_duration_s=video_duration_s
        )

        print("Telemetry provider started.")
        print("Telemetry source:", args.telemetry_csv)
    else:
        print("Telemetry provider not used.")

    postgis_writer = None

    if args.write_postgis:
        postgis_writer = PostgisWriter(
            host=args.db_host,
            port=args.db_port,
            database=args.db_name,
            user=args.db_user,
            password=args.db_password
        )

        print("PostGIS writer started.")
    else:
        print("PostGIS writer not used.")

    results = []

    mask_smoother = TemporalMaskSmoother(
        history_size=3,
        vote_threshold=0.34
    )

    fire_tracker = FireTracker(
        match_distance_m=40.0,
        max_missing_frames=60
    )

    track_alert_cooldown = {}
    ALERT_LOCATION_HISTORY = []
    ALERT_COOLDOWN_FRAMES = 15
    ALERT_MIN_DISTANCE_M = 40

    processed_count = 0
    frame_idx = 0

    while True:
        rgb_ret, rgb_frame = rgb_cap.read()
        thermal_ret, thermal_frame = thermal_cap.read()

        if not rgb_ret or not thermal_ret:
            break

        if frame_idx >= total_frames:
            break

        if frame_idx % args.frame_step != 0:
            frame_idx += 1
            continue

        input_tensor = build_fusion_tensor(
            rgb_frame=rgb_frame,
            thermal_frame=thermal_frame,
            input_size=meta["input_size"],
            device=meta["device"]
        )

        fire_prob_raw = predict_fire_probability(
            model=model,
            input_tensor=input_tensor,
            temperature=meta["temperature"]
        )

        video_time_s = frame_idx / fps
        telemetry_info = {}
        lat = None
        lon = None
        if telemetry_provider is not None:
            telemetry_info = telemetry_provider.get_current(video_time_s)
            lat = telemetry_info.get("latitude")
            lon = telemetry_info.get("longitude")

        fuel_score = 0.50
        if lat is not None and lon is not None:
            fuel_score = fuel_scorer.get_score(lat, lon)
        
        MAX_ETKI = 0.05
        modifiye_deger = ((fuel_score - 0.50) / 0.50) * MAX_ETKI
        
        # Final birleşik olasılık
        fire_prob = max(0.0, min(1.0, fire_prob_raw + modifiye_deger))

        is_fire = fire_prob >= meta["threshold"]
        video_time_s = frame_idx / fps

        base_result = {
            "frame_idx": frame_idx,
            "video_time_s": video_time_s,
            "kamera_prob": fire_prob_raw,
            "fuel_score": fuel_score,
            "fire_probability": fire_prob,
            "decision_prob": fire_prob,
            "threshold": meta["threshold"],
            "prediction": "fire" if is_fire else "no_fire",
            "pred_fire": 1 if is_fire else 0,
        }

        if telemetry_provider is not None:
            base_result.update(telemetry_info)

        if postgis_writer is not None:
            postgis_writer.insert_drone_frame_point(base_result)

        print(
            f"Frame {frame_idx} | time={video_time_s:.2f}s | "
            f"Kamera={fire_prob_raw:.4f} | Bitki={fuel_score:.4f} | "
            f"Final={fire_prob:.4f} | pred={base_result['prediction']}"
        )

        if is_fire:
            raw_mask, thermal_norm = create_fire_mask_from_thermal(
                thermal_frame,
                threshold_mode=args.threshold_mode,
                threshold_value=args.threshold_value,
                percentile_value=args.percentile,
                min_area=args.min_area,
                kernel_size=5,
                use_strong_closing=False,
                dilate_iterations=0
            )

            raw_mask_pixels = int((raw_mask > 0).sum())

            mask = mask_smoother.update(raw_mask)

            stable_mask_pixels = int((mask > 0).sum())

            fire_regions = extract_fire_regions(
                mask,
                min_contour_area=args.min_area
            )

            raw_region_count = len(fire_regions)

            # Yakın yangın parçalarını birleştir.
            # Küçük yangınları silmez; sadece yakın olanları tek bölge yapar.
            fire_regions = merge_close_fire_regions(
                fire_regions,
                image_shape=mask.shape,
                merge_distance_px=50
            )

            merged_region_count = len(fire_regions)

            fire_regions = filter_fire_regions(
                fire_regions,
                image_shape=mask.shape,
                max_region_area_ratio=0.35,
                max_bbox_width_ratio=0.95,
                max_bbox_height_ratio=0.75,
                max_aspect_ratio=15.0
            )

            print(
                f"  Raw mask pixels: {raw_mask_pixels} | "
                f"Stable mask pixels: {stable_mask_pixels} | "
                f"Raw regions: {raw_region_count} | "
                f"Merged regions: {merged_region_count} | "
                f"Filtered regions: {len(fire_regions)}"
            )

            if len(fire_regions) == 0:
                print(f"  SKIPPED: Model said fire but no valid regions found")
                results.append(base_result)
            else:
                thermal_bgr = cv2.cvtColor(thermal_norm, cv2.COLOR_GRAY2BGR)

                thermal_overlay = overlay_mask_on_frame(thermal_bgr, mask)
                thermal_contours = draw_fire_regions_on_frame(
                    thermal_overlay,
                    fire_regions
                )

                frame_name = f"frame_{frame_idx:06d}"

                fire_frame_path = fire_frames_dir / f"{frame_name}.jpg"
                mask_path = masks_dir / f"{frame_name}_mask.png"
                overlay_path = overlays_dir / f"{frame_name}_thermal_overlay.png"

                cv2.imwrite(str(fire_frame_path), thermal_bgr)
                cv2.imwrite(str(mask_path), mask)
                cv2.imwrite(str(overlay_path), thermal_contours)

                image_height, image_width = mask.shape[:2]

                for region in fire_regions:
                    bbox = region["bbox"]

                    fire_lat = None
                    fire_lon = None
                    approx_area_m2 = None
                    fire_track_id = None

                    if "latitude" in base_result and "longitude" in base_result:
                        fire_lat, fire_lon = pixel_to_geo_point(
                            pixel_x=region["centroid_x"],
                            pixel_y=region["centroid_y"],
                            image_width=image_width,
                            image_height=image_height,
                            drone_lat=base_result["latitude"],
                            drone_lon=base_result["longitude"],
                            altitude_m=base_result.get("altitude_m"),
                            horizontal_fov_deg=73.7,
                            vertical_fov_deg=53.1,
                            drone_yaw_deg=base_result.get("drone_yaw") or 0.0
                        )

                        approx_area_m2 = estimate_area_from_pixel_area(
                            pixel_area=region["pixel_area"],
                            image_width=image_width,
                            image_height=image_height,
                            altitude_m=base_result.get("altitude_m"),
                            horizontal_fov_deg=73.7,
                            vertical_fov_deg=53.1
                        )

                        fire_track_id, best_lat, best_lon = fire_tracker.update(
                            fire_lat=fire_lat,
                            fire_lon=fire_lon,
                            frame_idx=frame_idx,
                            approx_area_m2=approx_area_m2,
                            fire_probability=fire_prob
                        )

                    should_alert = True
                    if fire_lat is not None and fire_lon is not None:
                        for prev in ALERT_LOCATION_HISTORY:
                            dist = haversine_distance_m(fire_lat, fire_lon, prev["lat"], prev["lon"])
                            if dist < ALERT_MIN_DISTANCE_M and (frame_idx - prev["frame"]) < ALERT_COOLDOWN_FRAMES:
                                should_alert = False
                                break
                    if should_alert:
                        ALERT_LOCATION_HISTORY.append({"lat": fire_lat, "lon": fire_lon, "frame": frame_idx})
                        ALERT_LOCATION_HISTORY[:] = [
                            x for x in ALERT_LOCATION_HISTORY
                            if (frame_idx - x["frame"]) < ALERT_COOLDOWN_FRAMES
                        ]

                    result = base_result.copy()
                    result.update({
                        "region_id": region["region_id"],
                        "pixel_area": region["pixel_area"],
                        "centroid_x": region["centroid_x"],
                        "centroid_y": region["centroid_y"],
                        "bbox_x": bbox["x"],
                        "bbox_y": bbox["y"],
                        "bbox_width": bbox["width"],
                        "bbox_height": bbox["height"],
                        "fire_frame_path": str(fire_frame_path),
                        "mask_path": str(mask_path),
                        "overlay_path": str(overlay_path),
                        "fire_latitude": fire_lat,
                        "fire_longitude": fire_lon,
                        "track_best_latitude": best_lat,
                        "track_best_longitude": best_lon,
                        "approx_area_m2": approx_area_m2,
                        "area_method": "centroid_gsd_fov_altitude",
                        "fire_track_id": fire_track_id,
                        "alerted": should_alert
                    })

                    if postgis_writer is not None and should_alert:
                        postgis_writer.insert_fire_observation(result)
                        postgis_writer.upsert_active_fire_track(result)

                    results.append(result)

                print(f"  Fire regions: {len(fire_regions)}")

        else:
            mask_smoother.reset()
            results.append(base_result)

        processed_count += 1

        if args.max_frames is not None and processed_count >= args.max_frames:
            break

        frame_idx += 1

    rgb_cap.release()
    thermal_cap.release()

    results_df = pd.DataFrame(results)
    output_csv = output_dir / "live_fire_results.csv"
    results_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    if postgis_writer is not None:
        postgis_writer.close()

    print("\nDone.")
    print("Saved CSV:", output_csv)
    print("Total rows:", len(results_df))


if __name__ == "__main__":
    main()
