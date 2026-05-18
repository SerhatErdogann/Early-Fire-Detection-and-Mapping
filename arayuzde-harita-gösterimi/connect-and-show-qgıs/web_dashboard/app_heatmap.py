"""
Heatmap Demo - Mevcut sistemi bozmadan ayri portta (5001) calisan deneme sunucusu.
Orijinal app.py'ye dokunmaz.
"""
from flask import Flask, render_template, jsonify, send_from_directory
import psycopg2
import os
import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "env_utils.py").is_file():
        sys.path.insert(0, str(parent))
        break

from env_utils import load_project_env


load_project_env(__file__)

app = Flask(__name__)

DB_CONFIG = {
    "host": os.getenv("POSTGIS_HOST", "localhost"),
    "port": int(os.getenv("POSTGIS_PORT", "5432")),
    "database": os.getenv("POSTGIS_DB", "cografi_veritabani"),
    "user": os.getenv("POSTGIS_USER", "postgres"),
    "password": os.getenv("POSTGIS_PASSWORD", "postgres"),
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

@app.route('/')
def index():
    return render_template('index_heatmap.html')

@app.route('/frames/<filename>')
def serve_frame(filename):
    frames_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extracted_frames")
    return send_from_directory(frames_dir, filename)

@app.route('/api/stats')
def stats():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                COUNT(*) as toplam_kare,
                SUM(CASE WHEN yangin_var THEN 1 ELSE 0 END) as yangin_kare,
                MAX(yangin_yuzdesi) as max_yuzde
            FROM yangin_tahminleri
        """)
        toplam_kare, yangin_kare, max_yuzde = cur.fetchone()
        cur.execute("""
            SELECT 
                MIN(enlem) as min_lat, MAX(enlem) as max_lat,
                MIN(boylam) as min_lon, MAX(boylam) as max_lon
            FROM yangin_tahminleri
        """)
        min_lat, max_lat, min_lon, max_lon = cur.fetchone()
        cur.close()
        conn.close()

        if toplam_kare == 0 or min_lat is None:
            return jsonify({"status": "empty", "center": [39.0, 35.0], "stats": {"toplam": 0, "yangin": 0, "max_yuzde": 0}})

        return jsonify({
            "status": "success",
            "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
            "center": [(min_lat+max_lat)/2, (min_lon+max_lon)/2],
            "stats": {
                "toplam": toplam_kare,
                "yangin": yangin_kare,
                "max_yuzde": round((max_yuzde or 0) * 100, 2)
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/all-points')
def all_points():
    """Heatmap icin TUM noktalari (yangin olan+olmayan) yogunluk degeriyle dondurur."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT enlem, boylam, yangin_yuzdesi, zaman, resim_adi, yangin_var
            FROM yangin_tahminleri
            ORDER BY zaman
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        points = []
        for row in rows:
            points.append({
                "lat": float(row[0]),
                "lon": float(row[1]),
                "intensity": round(float(row[2]), 4),
                "zaman": row[3].strftime("%Y-%m-%d %H:%M:%S") if row[3] else "",
                "resim_adi": row[4],
                "yangin_var": row[5],
                "yuzde": round(float(row[2]) * 100, 2)
            })
        return jsonify(points)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    print("\n  HEATMAP DEMO -> http://127.0.0.1:5001\n")
    app.run(
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5001")),
    )
