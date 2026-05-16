# src/segmentation/mask_filter.py

import numpy as np


def filter_fire_regions(
    fire_regions,
    image_shape,
    max_region_area_ratio=0.20,
    max_bbox_width_ratio=0.85,
    max_bbox_height_ratio=0.60,
    min_aspect_ratio=0.15,
    max_aspect_ratio=8.0
):
    """
    Yangın gibi görünmeyen mask/contour bölgelerini eler.

    Amaç:
    - Çok büyük alanları elemek
    - Tüm frame'i yatay bant gibi kaplayan bölgeleri elemek
    - Aşırı ince/garip bölgeleri elemek
    """

    height, width = image_shape[:2]
    image_area = height * width

    filtered_regions = []

    for region in fire_regions:
        pixel_area = region["pixel_area"]
        bbox = region["bbox"]

        bbox_width = bbox["width"]
        bbox_height = bbox["height"]

        region_area_ratio = pixel_area / image_area
        bbox_width_ratio = bbox_width / width
        bbox_height_ratio = bbox_height / height

        if bbox_height == 0:
            continue

        aspect_ratio = bbox_width / bbox_height

        reject_reasons = []

        if region_area_ratio > max_region_area_ratio:
            reject_reasons.append("too_large_area")

        if bbox_width_ratio > max_bbox_width_ratio:
            reject_reasons.append("too_wide_bbox")

        if bbox_height_ratio > max_bbox_height_ratio:
            reject_reasons.append("too_tall_bbox")

        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            reject_reasons.append("bad_aspect_ratio")

        region["region_area_ratio"] = float(region_area_ratio)
        region["bbox_width_ratio"] = float(bbox_width_ratio)
        region["bbox_height_ratio"] = float(bbox_height_ratio)
        region["aspect_ratio"] = float(aspect_ratio)
        region["is_valid_fire_region"] = len(reject_reasons) == 0
        region["reject_reasons"] = ",".join(reject_reasons)

        if region["is_valid_fire_region"]:
            filtered_regions.append(region)

    return filtered_regions