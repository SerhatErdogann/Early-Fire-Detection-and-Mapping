from flask import Flask, render_template, jsonify
import psycopg2
import os

app = Flask(__name__)

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "cografi_veritabani",
    "user": "postgres",
    "password": "1313"
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

@app.route('/')
def index():
    # Sadece HTML sayfasini goster
    return render_template('index.html')

@app.route('/api/stats')
def stats():
    # Veritabanindan dinamik olarak bounding box ve istatistikleri cek
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Ozet istatistikler
        cur.execute("""
            SELECT 
                COUNT(*) as toplam_kare,
                SUM(CASE WHEN yangin_var THEN 1 ELSE 0 END) as yangin_kare,
                MAX(yangin_yuzdesi) as max_yuzde
            FROM yangin_tahminleri
        """)
        toplam_kare, yangin_kare, max_yuzde = cur.fetchone()
        
        # Bounding Box ve Merkez Koordinatlar
        cur.execute("""
            SELECT 
                MIN(enlem) as min_lat,
                MAX(enlem) as max_lat,
                MIN(boylam) as min_lon,
                MAX(boylam) as max_lon
            FROM yangin_tahminleri
        """)
        min_lat, max_lat, min_lon, max_lon = cur.fetchone()
        
        cur.close()
        conn.close()

        # Eger veri yoksa default koordinatlar (Turkiye vs) dondur
        if toplam_kare == 0 or min_lat is None:
            return jsonify({
                "status": "empty",
                "center": [39.0, 35.0],
                "stats": {"toplam": 0, "yangin": 0, "max_yuzde": 0}
            })

        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0

        return jsonify({
            "status": "success",
            "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
            "center": [center_lat, center_lon],
            "stats": {
                "toplam": toplam_kare,
                "yangin": yangin_kare,
                "max_yuzde": round((max_yuzde or 0) * 100, 2)
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

from flask import send_from_directory

@app.route('/frames/<filename>')
def serve_frame(filename):
    # Ana dizindeki extracted_frames klasorunu bul
    frames_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extracted_frames")
    return send_from_directory(frames_dir, filename)

@app.route('/api/fire-points')
def fire_points():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT enlem, boylam, yangin_yuzdesi, zaman, resim_adi
            FROM yangin_tahminleri
            WHERE yangin_var = True
        """)
        rows = cur.fetchall()
        
        features = []
        for row in rows:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row[1], row[0]] # lon, lat
                },
                "properties": {
                    "yangin_yuzdesi": round(row[2]*100, 2),
                    "zaman": row[3].strftime("%Y-%m-%d %H:%M:%S") if row[3] else "",
                    "resim_adi": row[4]
                }
            })
        cur.close()
        conn.close()
        return jsonify({
            "type": "FeatureCollection",
            "features": features
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # 5000 portunda calistir
    app.run(debug=True, host='0.0.0.0', port=5000)
