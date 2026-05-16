# src/segmentation/contours.py

import cv2


def extract_fire_regions(mask, min_contour_area=250):
    """
    Binary fire mask üzerinden ayrı yangın bölgelerini çıkarır.
    Her contour ayrı fire region olarak döner.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    fire_regions = []

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area < min_contour_area:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])

        x, y, w, h = cv2.boundingRect(contour)

        fire_regions.append({
            "region_id": idx + 1,
            "contour": contour,
            "pixel_area": float(area),
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "bbox": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            }
        })

    return fire_regions