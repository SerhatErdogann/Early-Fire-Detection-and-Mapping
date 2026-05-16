# src/segmentation/visualization.py

import cv2
import numpy as np


def draw_fire_regions_on_frame(frame, fire_regions):
    """
    Fire contour'larını frame üzerine çizer.
    """

    output = frame.copy()

    for region in fire_regions:
        contour = region["contour"]
        centroid_x = region["centroid_x"]
        centroid_y = region["centroid_y"]
        pixel_area = region["pixel_area"]

        cv2.drawContours(output, [contour], -1, (0, 0, 255), 2)

        cv2.circle(
            output,
            (centroid_x, centroid_y),
            5,
            (255, 255, 255),
            -1
        )

        cv2.putText(
            output,
            f"Area(px): {pixel_area:.0f}",
            (centroid_x + 8, centroid_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return output


def overlay_mask_on_frame(frame, mask, alpha=0.45):
    """
    Binary mask'i frame üzerine yarı saydam bindirir.
    """

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if len(mask.shape) == 2:
        mask_colored = np.zeros_like(frame)
        mask_colored[:, :, 2] = mask
    else:
        mask_colored = mask

    output = cv2.addWeighted(frame, 1.0, mask_colored, alpha, 0)

    return output