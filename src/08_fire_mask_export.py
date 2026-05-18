# src/08_fire_mask_export.py

import argparse
import os
from pathlib import Path

import cv2
import pandas as pd

from segmentation.thermal_threshold import create_fire_mask_from_thermal
from segmentation.contours import extract_fire_regions
from segmentation.visualization import (
    draw_fire_regions_on_frame,
    overlay_mask_on_frame
)


def find_frame_column(df):
    """
    video_predictions.csv içinde frame kolon adını bulur.
    Repo çıktısına göre frame_idx veya frame_id olabilir.
    """
    possible_columns = ["frame_idx", "frame_id", "frame", "idx"]

    for col in possible_columns:
        if col in df.columns:
            return col

    raise ValueError(
        f"Frame column not found. Available columns: {list(df.columns)}"
    )


def find_fire_column(df):
    """
    Fire/no-fire karar kolonunu bulur.
    Öncelik stabilize edilmiş karar kolonlarında.
    """
    possible_columns = [
        "fire_detected",
        "pred_fire_burst_consec",
        "pred_fire",
        "fire",
        "is_fire"
    ]

    for col in possible_columns:
        if col in df.columns:
            return col

    raise ValueError(
        f"Fire decision column not found. Available columns: {list(df.columns)}"
    )


def read_frame_from_video(video_capture, frame_idx):
    """
    Videodan istenen frame index'ini okur.
    """
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = video_capture.read()

    if not ret:
        return None

    return frame


def is_fire_value(value):
    """
    CSV içindeki fire değerini boolean'a çevirir.
    True / 1 / 'true' / 'fire' gibi değerleri destekler.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value == 1

    value_str = str(value).strip().lower()

    return value_str in ["true", "1", "fire", "yes", "y"]


def main():
    parser = argparse.ArgumentParser(
        description="Export fire masks from thermal video using video_predictions.csv"
    )

    parser.add_argument(
        "--pred_csv",
        required=True,
        help="Path to video_predictions.csv"
    )

    parser.add_argument(
        "--thermal_video",
        required=True,
        help="Path to thermal video"
    )

    parser.add_argument(
        "--rgb_video",
        default=None,
        help="Optional RGB video path for RGB overlay output"
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/fire_mask_results",
        help="Output directory"
    )

    parser.add_argument(
        "--percentile",
        type=float,
        default=97,
        help="Percentile threshold value. Example: 97 means hottest 3 percent."
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
        default=210,
        help="Raw thermal/grayscale threshold used by fixed/absolute/hybrid modes"
    )

    parser.add_argument(
        "--min_area",
        type=int,
        default=300,
        help="Minimum region area in pixels"
    )

    parser.add_argument(
        "--strong_closing",
        action="store_true",
        help="Use stronger morphology closing to merge nearby fire regions"
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional limit for debugging"
    )

    args = parser.parse_args()

    pred_csv_path = Path(args.pred_csv)
    thermal_video_path = Path(args.thermal_video)

    if not pred_csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv_path}")

    if not thermal_video_path.exists():
        raise FileNotFoundError(f"Thermal video not found: {thermal_video_path}")

    output_dir = Path(args.output_dir)
    masks_dir = output_dir / "masks"
    thermal_overlays_dir = output_dir / "thermal_overlays"
    rgb_overlays_dir = output_dir / "rgb_overlays"

    masks_dir.mkdir(parents=True, exist_ok=True)
    thermal_overlays_dir.mkdir(parents=True, exist_ok=True)
    rgb_overlays_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(pred_csv_path)

    frame_col = find_frame_column(predictions)
    fire_col = find_fire_column(predictions)

    print(f"Using frame column: {frame_col}")
    print(f"Using fire column: {fire_col}")

    fire_predictions = predictions[
        predictions[fire_col].apply(is_fire_value)
    ].copy()

    if args.max_frames is not None:
        fire_predictions = fire_predictions.head(args.max_frames)

    print(f"Fire frame count: {len(fire_predictions)}")

    thermal_cap = cv2.VideoCapture(str(thermal_video_path))

    if not thermal_cap.isOpened():
        raise RuntimeError(f"Could not open thermal video: {thermal_video_path}")

    rgb_cap = None

    if args.rgb_video:
        rgb_video_path = Path(args.rgb_video)

        if not rgb_video_path.exists():
            raise FileNotFoundError(f"RGB video not found: {rgb_video_path}")

        rgb_cap = cv2.VideoCapture(str(rgb_video_path))

        if not rgb_cap.isOpened():
            raise RuntimeError(f"Could not open RGB video: {rgb_video_path}")

    results = []

    for _, row in fire_predictions.iterrows():
        frame_idx = int(row[frame_col])

        thermal_frame = read_frame_from_video(thermal_cap, frame_idx)

        if thermal_frame is None:
            print(f"Warning: could not read thermal frame {frame_idx}")
            continue

        mask, thermal_norm = create_fire_mask_from_thermal(
            thermal_frame,
            threshold_mode=args.threshold_mode,
            threshold_value=args.threshold_value,
            percentile_value=args.percentile,
            min_area=args.min_area,
            kernel_size=9,
            use_strong_closing=args.strong_closing,
            dilate_iterations=0,
        )

        fire_regions = extract_fire_regions(
            mask,
            min_contour_area=args.min_area
        )

        if len(fire_regions) == 0:
            continue

        thermal_bgr = cv2.cvtColor(thermal_norm, cv2.COLOR_GRAY2BGR)

        thermal_overlay = overlay_mask_on_frame(thermal_bgr, mask)
        thermal_contours = draw_fire_regions_on_frame(
            thermal_overlay,
            fire_regions
        )

        mask_filename = f"frame_{frame_idx:06d}_mask.png"
        thermal_overlay_filename = f"frame_{frame_idx:06d}_thermal_overlay.png"

        mask_path = masks_dir / mask_filename
        thermal_overlay_path = thermal_overlays_dir / thermal_overlay_filename

        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(thermal_overlay_path), thermal_contours)

        rgb_overlay_path = None

        if rgb_cap is not None:
            rgb_frame = read_frame_from_video(rgb_cap, frame_idx)

            if rgb_frame is not None:
                if rgb_frame.shape[:2] != thermal_frame.shape[:2]:
                    rgb_frame = cv2.resize(
                        rgb_frame,
                        (thermal_frame.shape[1], thermal_frame.shape[0])
                    )

                rgb_overlay = overlay_mask_on_frame(rgb_frame, mask)
                rgb_contours = draw_fire_regions_on_frame(
                    rgb_overlay,
                    fire_regions
                )

                rgb_overlay_filename = f"frame_{frame_idx:06d}_rgb_overlay.png"
                rgb_overlay_path = rgb_overlays_dir / rgb_overlay_filename

                cv2.imwrite(str(rgb_overlay_path), rgb_contours)

        for region in fire_regions:
            bbox = region["bbox"]

            result = {
                "frame_idx": frame_idx,
                "region_id": region["region_id"],
                "pixel_area": region["pixel_area"],
                "centroid_x": region["centroid_x"],
                "centroid_y": region["centroid_y"],
                "bbox_x": bbox["x"],
                "bbox_y": bbox["y"],
                "bbox_width": bbox["width"],
                "bbox_height": bbox["height"],
                "mask_path": str(mask_path),
                "thermal_overlay_path": str(thermal_overlay_path),
                "rgb_overlay_path": str(rgb_overlay_path) if rgb_overlay_path else None,
                "threshold_mode": args.threshold_mode,
                "threshold_value": args.threshold_value,
                "percentile_value": args.percentile,
                "min_area": args.min_area,
                "strong_closing": args.strong_closing
            }

            # Prediction CSV'de olasılık kolonları varsa onları da ekleyelim
            for optional_col in [
                "prob_fire",
                "prob_fire_ema",
                "prob_fire_ma",
                "decision_prob",
                "fire_probability",
                "smoothed_probability",
                "confidence",
                "alarm_state",
                "risk_score",
                "risk_score_norm"
            ]:
                if optional_col in predictions.columns:
                    result[optional_col] = row[optional_col]

            results.append(result)

        print(
            f"Frame {frame_idx}: {len(fire_regions)} fire region(s) exported"
        )

    thermal_cap.release()

    if rgb_cap is not None:
        rgb_cap.release()

    results_df = pd.DataFrame(results)

    output_csv = output_dir / "fire_regions.csv"
    results_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Done. Fire region CSV saved to: {output_csv}")
    print(f"Total exported fire regions: {len(results_df)}")


if __name__ == "__main__":
    main()
