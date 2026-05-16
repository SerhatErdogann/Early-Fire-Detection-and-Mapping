# src/static_server.py

from flask import Flask, send_from_directory, request, Response
from pathlib import Path
import requests

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]

OVERLAY_DIR = BASE_DIR / "outputs" / "live_video_results" / "overlays"
FIRE_FRAMES_DIR = BASE_DIR / "outputs" / "live_video_results" / "fire_frames"
MASKS_DIR = BASE_DIR / "outputs" / "live_video_results" / "masks"
STATIC_DIR = BASE_DIR / "static"


@app.route("/overlays/<path:filename>")
def serve_overlay(filename):
    return send_from_directory(OVERLAY_DIR, filename)


@app.route("/fire_frames/<path:filename>")
def serve_fire_frame(filename):
    return send_from_directory(FIRE_FRAMES_DIR, filename)


@app.route("/masks/<path:filename>")
def serve_mask(filename):
    return send_from_directory(MASKS_DIR, filename)


@app.route("/map")
def serve_map():
    return send_from_directory(STATIC_DIR, "test_fire_map.html")


@app.route("/geoserver_proxy")
def geoserver_proxy():
    """
    Browser CORS hatasını önlemek için GeoServer GetFeatureInfo isteklerini
    Flask üzerinden proxy'ler.
    """

    geoserver_url = request.args.get("url")

    if not geoserver_url:
        return {"error": "Missing url parameter"}, 400

    response = requests.get(geoserver_url)

    return Response(
        response.content,
        status=response.status_code,
        content_type=response.headers.get("Content-Type", "application/json")
    )


@app.route("/")
def index():
    return {
        "status": "ok",
        "overlay_dir": str(OVERLAY_DIR),
        "fire_frames_dir": str(FIRE_FRAMES_DIR),
        "masks_dir": str(MASKS_DIR),
        "map_url": "http://localhost:5000/map"
    }


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)