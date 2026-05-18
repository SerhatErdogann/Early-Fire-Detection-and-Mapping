import psycopg2
from psycopg2 import OperationalError
import os
import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "env_utils.py").is_file():
        sys.path.insert(0, str(parent))
        break

from env_utils import load_project_env


load_project_env(__file__)

def test_connection():
    db_config = {
        "host": os.getenv("POSTGIS_HOST", "localhost"),
        "port": int(os.getenv("POSTGIS_PORT", "5432")),
        "database": os.getenv("POSTGIS_DB", "cografi_veritabani"),
        "user": os.getenv("POSTGIS_USER", "postgres"),
        "password": os.getenv("POSTGIS_PASSWORD", "postgres"),
    }

    try:
        print("Veritabanina baglanilmaya calisiliyor...")
        conn = psycopg2.connect(**db_config)
        print("[BASARILI] Baglanti basarili!")
        
        # Tablo yoksa olusturalim (sonraki adimlar icin hazirlik)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS yangin_tahminleri (
                id SERIAL PRIMARY KEY,
                zaman TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                frame_idx INTEGER,
                saniye FLOAT,
                enlem FLOAT,
                boylam FLOAT,
                kamera_skoru FLOAT,
                bitki_skoru FLOAT,
                yangin_yuzdesi FLOAT,
                yangin_var BOOLEAN
            );
        """)
        conn.commit()
        print("[BASARILI] 'yangin_tahminleri' tablosu kontrol edildi/olusturuldu.")
        
        cur.close()
        conn.close()
        print("Baglanti kapatildi.")
        
    except OperationalError as e:
        print("[HATA] Baglanti hatasi:")
        print(e)

if __name__ == "__main__":
    test_connection()
