# src/segmentation/region_merge.py

import cv2
import numpy as np


def bbox_distance(bbox_a, bbox_b):
    """
    İki bounding box arasındaki piksel mesafesini hesaplar.
    Box'lar çakışıyor veya temas ediyorsa 0 döner.
    """

    ax1 = bbox_a["x"]
    ay1 = bbox_a["y"]
    ax2 = bbox_a["x"] + bbox_a["width"]
    ay2 = bbox_a["y"] + bbox_a["height"]

    bx1 = bbox_b["x"]
    by1 = bbox_b["y"]
    bx2 = bbox_b["x"] + bbox_b["width"]
    by2 = bbox_b["y"] + bbox_b["height"]

    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)

    return (dx * dx + dy * dy) ** 0.5


def merge_close_fire_regions(fire_regions, image_shape, merge_distance_px=35):
    """
    Birbirine yakın yangın region'larını tek region olarak gruplar.

    Önemli:
    - Küçük yangınları silmez.
    - Uzak küçük yangınları ayrı bırakır.
    - Yakın parçaları tek region bilgisi altında toplar.
    - Çizim için contour'ları ayrı ayrı saklar.
    """

    if not fire_regions:
        return []

    n = len(fire_regions)
    visited = [False] * n
    merged_regions = []

    for i in range(n):
        if visited[i]:
            continue

        group_indices = []
        stack = [i]
        visited[i] = True

        while stack:
            current_idx = stack.pop()
            group_indices.append(current_idx)

            current_bbox = fire_regions[current_idx]["bbox"]

            for j in range(n):
                if visited[j]:
                    continue

                other_bbox = fire_regions[j]["bbox"]
                distance = bbox_distance(current_bbox, other_bbox)

                if distance <= merge_distance_px:
                    visited[j] = True
                    stack.append(j)

        group_regions = [fire_regions[idx] for idx in group_indices]

        merged_region = build_merged_region(
            group_regions,
            image_shape=image_shape,
            region_id=len(merged_regions) + 1
        )

        if merged_region is not None:
            merged_regions.append(merged_region)

    return merged_regions


def build_merged_region(regions, image_shape, region_id):
    """
    Yakın region grubunu tek mantıksal region'a çevirir.
    Çapraz çizgi oluşmaması için gerçek contour'ları ayrı saklar.
    """

    height, width = image_shape[:2]
    group_mask = np.zeros((height, width), dtype=np.uint8)

    original_contours = []

    for region in regions:
        contour = region["contour"]
        original_contours.append(contour)
        cv2.drawContours(group_mask, [contour], -1, 255, thickness=-1)

    contours, _ = cv2.findContours(
        group_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    all_points = np.vstack(contours)

    x, y, w, h = cv2.boundingRect(all_points)

    pixel_area = int(np.sum(group_mask > 0))

    moments = cv2.moments(group_mask, binaryImage=True)

    if moments["m00"] != 0:
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
    else:
        centroid_x = int(x + w / 2)
        centroid_y = int(y + h / 2)

    # Bu contour sadece güvenli fallback için.
    # Çizimde esas olarak "contours" kullanılacak.
    safe_contour = cv2.convexHull(all_points)

    return {
        "region_id": region_id,

        # Eski kod bozulmasın diye contour kalsın.
        # Ama artık çapraz çizgi üretmemesi için convex hull veriyoruz.
        "contour": safe_contour,

        # Asıl doğru çizim için gerçek sınırlar burada.
        "contours": contours,

        # İstersen orijinal parçaları da saklıyoruz.
        "original_contours": original_contours,

        "pixel_area": pixel_area,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "bbox": {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        }
    }