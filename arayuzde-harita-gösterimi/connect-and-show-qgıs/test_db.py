import psycopg2
from psycopg2 import OperationalError

def test_connection():
    db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "cografi_veritabani",
        "user": "postgres",
        "password": "1313"
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
