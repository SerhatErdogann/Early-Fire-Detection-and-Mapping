"""
Spatial stats from a soft map (segmentation prob or CAM), for GIS-style metrics.
"""
from __future__ import annotations

import numpy as np
import cv2


def stats_from_soft_map(cam: np.ndarray, thr_soft: float = 0.3, thr_bin: float = 0.5) -> dict:
    """
    cam: HxW float in [0,1] (normalized CAM or fire_mask_prob).
    Returns mass, soft/hard area fractions, peak intensity, connected components (4-conn).
    """
    m = np.asarray(cam, dtype=np.float64).ravel()
    h, w = cam.shape[:2]
    n = h * w
    fire_mass = float(m.sum())
    soft_area = float((cam >= thr_soft).mean())
    hard_area = float((cam >= thr_bin).mean())
    k = max(1, int(0.01 * n))
    # Faster than full sort: take top-k via partition.
    if m.size:
        topk = np.partition(m, m.size - k)[-k:]
        peak_intensity = float(np.mean(topk))
    else:
        peak_intensity = 0.0
    bin_mask = (cam >= thr_bin).astype(np.uint8)
    num_components, labels, areas, centroids = _connected_components(bin_mask)
    largest = float(max(areas)) if areas else 0.0
    largest_frac = largest / float(n) if n else 0.0
    cy, cx = (0.0, 0.0)
    if areas and int(np.argmax(areas)) < len(centroids):
        cy, cx = centroids[int(np.argmax(areas))]
    cx_norm = float(cx) / max(1, w - 1)
    cy_norm = float(cy) / max(1, h - 1)
    edge_density = _edge_density(bin_mask)
    return {
        "fire_mass": fire_mass / float(n),
        "fire_area_soft": soft_area,
        "fire_area_hard": hard_area,
        "peak_intensity": peak_intensity,
        "num_components": int(num_components),
        "largest_component_area": largest_frac,
        "centroid_x_norm": cx_norm,
        "centroid_y_norm": cy_norm,
        "edge_density": edge_density,
    }


def _edge_density(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    m = mask.astype(np.uint8)
    pad = np.pad(m, 1, mode="constant")
    interior = m > 0
    if not interior.any():
        return 0.0
    sh = (pad[1:-1, 2:] != pad[1:-1, :-2]) & interior
    sv = (pad[2:, 1:-1] != pad[:-2, 1:-1]) & interior
    return float((sh | sv).mean())


def _connected_components(mask: np.ndarray):
    try:
        n_labels, labels, stats, centroids_xy = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=4
        )
        if n_labels <= 1:
            return 0, labels.astype(np.int32), [], []
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64).tolist()
        centroids = [
            (float(centroids_xy[i, 1]), float(centroids_xy[i, 0]))
            for i in range(1, n_labels)
        ]
        return int(n_labels - 1), labels.astype(np.int32), areas, centroids
    except Exception:
        pass

    h, w = mask.shape
    labels = np.zeros_like(mask, dtype=np.int32)
    current = 0
    areas: list[int] = []
    centroids: list[tuple[float, float]] = []

    def neighbors(y, x):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or labels[y, x] > 0:
                continue
            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            sy, sx = 0, 0
            cnt = 0
            while stack:
                cy, cx = stack.pop()
                sy += cy
                sx += cx
                cnt += 1
                for ny, nx in neighbors(cy, cx):
                    if mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
            areas.append(cnt)
            centroids.append((sy / cnt, sx / cnt))

    return current, labels, areas, centroids
