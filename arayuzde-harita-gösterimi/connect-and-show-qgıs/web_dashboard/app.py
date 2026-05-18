from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import quote, urlparse

import requests
from flask import Flask, abort, jsonify, render_template, send_from_directory


for parent in Path(__file__).resolve().parents:
    if (parent / "env_utils.py").is_file():
        sys.path.insert(0, str(parent))
        break

from env_utils import load_project_env


load_project_env(__file__)

app = Flask(__name__)

GEOSERVER_URL = os.getenv("GEOSERVER_URL", "http://localhost:8080/geoserver").rstrip("/")
GEOSERVER_WORKSPACE = os.getenv("GEOSERVER_WORKSPACE", "fire_mapping")
GEOSERVER_WMS_URL = os.getenv(
    "GEOSERVER_WMS_URL",
    f"{GEOSERVER_URL}/{GEOSERVER_WORKSPACE}/wms",
)
LAYER_ACTIVE_FIRE = os.getenv("GEOSERVER_LAYER_ACTIVE_FIRE", f"{GEOSERVER_WORKSPACE}:active_fire_tracks")
LAYER_DRONE_FRAMES = os.getenv("GEOSERVER_LAYER_DRONE_FRAMES", f"{GEOSERVER_WORKSPACE}:drone_frame_points")
LAYER_FIRE_OBS = os.getenv("GEOSERVER_LAYER_FIRE_OBS", f"{GEOSERVER_WORKSPACE}:fire_observations")
GEOSERVER_TIMEOUT_S = float(os.getenv("GEOSERVER_TIMEOUT_S", "5"))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_FRAME_BASE_URL = os.getenv("DASHBOARD_FRAME_BASE_URL", "").rstrip("/")


def _wfs_features(type_name: str) -> list[dict]:
    url = f"{GEOSERVER_URL}/{GEOSERVER_WORKSPACE}/ows"
    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typeName": type_name,
        "outputFormat": "application/json",
    }
    response = requests.get(url, params=params, timeout=GEOSERVER_TIMEOUT_S)
    response.raise_for_status()
    payload = response.json()
    return payload.get("features") or []


def _prop(properties: dict, *names, default=None):
    for name in names:
        if name in properties and properties[name] is not None:
            return properties[name]
    return default


def _point_from_feature(feature: dict) -> tuple[float | None, float | None]:
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates")
    if geometry.get("type") == "Point" and isinstance(coordinates, list) and len(coordinates) >= 2:
        return float(coordinates[1]), float(coordinates[0])

    properties = feature.get("properties") or {}
    lat = _prop(properties, "latitude", "fire_latitude", "enlem")
    lon = _prop(properties, "longitude", "fire_longitude", "boylam")
    if lat is None or lon is None:
        return None, None
    return float(lat), float(lon)


def _float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_fire_frame(properties: dict) -> bool:
    prediction = str(_prop(properties, "prediction", default="")).lower()
    if prediction:
        return prediction == "fire"
    pred_fire = _prop(properties, "pred_fire", default=None)
    if pred_fire is not None:
        return str(pred_fire).lower() in {"1", "true", "fire", "yes"}
    prob = _float_or_none(_prop(properties, "fire_probability", "decision_prob", "last_probability"))
    return bool(prob is not None and prob >= 0.5)


def _is_url(value: str) -> bool:
    parsed = urlparse(str(value))
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _frame_dirs() -> list[Path]:
    configured = os.getenv(
        "DASHBOARD_FRAME_DIRS",
        "outputs/live_video_results/overlays;outputs/live_video_results/fire_frames",
    )
    dirs = []
    for raw_part in configured.split(";"):
        part = raw_part.strip()
        if not part:
            continue
        path = Path(part)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        dirs.append(path.resolve())
    return dirs


def _image_url_from_properties(properties: dict) -> tuple[str, str]:
    explicit_url = _prop(properties, "overlay_url", "image_url", "frame_url", "fire_frame_url", "photo_url")
    if explicit_url:
        explicit_url = str(explicit_url)
        if _is_url(explicit_url):
            return explicit_url, os.path.basename(urlparse(explicit_url).path)

    overlay_path = str(_prop(properties, "overlay_path", "fire_frame_path", "resim_adi", default="") or "")
    if _is_url(overlay_path):
        return overlay_path, os.path.basename(urlparse(overlay_path).path)

    filename = os.path.basename(overlay_path)
    if not filename:
        return "", ""
    if DASHBOARD_FRAME_BASE_URL:
        return f"{DASHBOARD_FRAME_BASE_URL}/{quote(filename)}", filename
    return f"/frames/{quote(filename)}", filename


@app.route("/")
def index():
    return render_template(
        "index.html",
        geoserver_wms_url=GEOSERVER_WMS_URL,
        layer_active_fire=LAYER_ACTIVE_FIRE,
        layer_drone_frames=LAYER_DRONE_FRAMES,
        layer_fire_obs=LAYER_FIRE_OBS,
    )


@app.route("/api/stats")
def stats():
    try:
        features = _wfs_features(LAYER_DRONE_FRAMES)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"GeoServer WFS error: {exc}"})

    total = len(features)
    fire_count = 0
    max_prob = 0.0
    points = []

    for feature in features:
        properties = feature.get("properties") or {}
        if _is_fire_frame(properties):
            fire_count += 1

        prob = _float_or_none(_prop(properties, "fire_probability", "decision_prob", "last_probability"))
        if prob is not None:
            max_prob = max(max_prob, prob)

        lat, lon = _point_from_feature(feature)
        if lat is not None and lon is not None:
            points.append((lat, lon))

    if total == 0 or not points:
        return jsonify({
            "status": "empty",
            "center": [39.0, 35.0],
            "stats": {"toplam": 0, "yangin": 0, "max_yuzde": 0},
        })

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    return jsonify({
        "status": "success",
        "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
        "center": [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0],
        "stats": {
            "toplam": total,
            "yangin": fire_count,
            "max_yuzde": round(max_prob * 100, 2),
        },
    })


@app.route("/frames/<path:filename>")
def serve_frame(filename):
    safe_name = os.path.basename(filename)
    if not safe_name:
        abort(404)
    for directory in _frame_dirs():
        candidate = directory / safe_name
        if candidate.is_file():
            return send_from_directory(directory, safe_name)
    abort(404)


@app.route("/api/fire-points")
def fire_points():
    try:
        features = _wfs_features(LAYER_ACTIVE_FIRE)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"GeoServer WFS error: {exc}"})

    normalized_features = []
    for feature in features:
        lat, lon = _point_from_feature(feature)
        if lat is None or lon is None:
            continue

        properties = feature.get("properties") or {}
        probability = _float_or_none(_prop(properties, "last_probability", "fire_probability", "decision_prob")) or 0.0
        video_time_s = _float_or_none(_prop(properties, "last_video_time_s", "video_time_s"))
        image_url, filename = _image_url_from_properties(properties)

        normalized_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "yangin_yuzdesi": round(probability * 100, 2),
                "zaman": f"{video_time_s:.2f}s" if video_time_s is not None else "",
                "resim_adi": filename,
                "image_url": image_url,
            },
        })

    return jsonify({"type": "FeatureCollection", "features": normalized_features})


if __name__ == "__main__":
    app.run(
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
    )
