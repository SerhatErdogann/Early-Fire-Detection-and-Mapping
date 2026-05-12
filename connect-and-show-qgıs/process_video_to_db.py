import psycopg2
import pandas as pd
import cv2
import os
import math
import sys
import time

# Script'in kendi dizinini bul
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Ana proje dizini (bitki-verisi)
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Veritabani baglantisi
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "cografi_veritabani",
    "user": "postgres",
    "password": "1313"
}

# Dosya Yollari
VIDEO_PATH = os.path.join(BASE_DIR, "test-videos", "RGB-video.MP4")
CSV_GPS_PATH = os.path.join(BASE_DIR, "test-videos", "DJIFlightRecord_2019-01-09_[13-16-53].csv")
CSV_PRED_PATH = os.path.join(BASE_DIR, "model-destek-islemleri", "outputs", "video_predictions_scored.csv")
FRAMES_OUT_DIR = os.path.join(SCRIPT_DIR, "extracted_frames")

os.makedirs(FRAMES_OUT_DIR, exist_ok=True)

def db_hazirla():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # PostGIS eklentisini etkinlestir (Geom sutunu icin zorunlu)
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS yangin_tahminleri (
            id SERIAL PRIMARY KEY,
            enlem DOUBLE PRECISION,
            boylam DOUBLE PRECISION,
            yangin_var BOOLEAN,
            yangin_yuzdesi DOUBLE PRECISION,
            zaman TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            geom geometry(Point, 4326),
            resim_adi VARCHAR(255)
        );
    """)
    cur.execute("ALTER TABLE yangin_tahminleri ADD COLUMN IF NOT EXISTS resim_adi VARCHAR(255);")
    cur.execute("TRUNCATE TABLE yangin_tahminleri RESTART IDENTITY;")
    conn.commit()
    cur.close()
    conn.close()

def veri_isle_ve_kaydet():
    print("[1] Veriler yukleniyor...")
    
    # 1. DJI GPS CSV Okuma
    # Ilk 2 satir baslik/bilgi olabilir, pandas ile okurken skiprows gerekebilir.
    # Ucus kaydinda 1. satirda basliklar var, 2. satirda bosluk veya baska veri var.
    # Dikkatli okuyalim:
    df_gps = pd.read_csv(CSV_GPS_PATH, skiprows=1) 
    # Zaman sutununu sayisal saniyeye cevir
    if 'OSD.flyTime [s]' in df_gps.columns:
        df_gps['time_s'] = pd.to_numeric(df_gps['OSD.flyTime [s]'], errors='coerce')
    else:
        print("[HATA] GPS dosyasinda 'OSD.flyTime [s]' bulunamadi.")
        return

    # 2. Model Tahminleri CSV Okuma
    df_preds = pd.read_csv(CSV_PRED_PATH)
    prob_col = 'decision_prob' if 'decision_prob' in df_preds.columns else 'prob_fire'

    # 3. Video Okuma Hazirligi
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[HATA] Video acilamadi: {VIDEO_PATH}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0

    print(f"[2] Video isleniyor ({fps} FPS)... Saniyede 2 kare (2 FPS) cikarilacak.")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Saniyede 2 kare = 0.5 saniyede bir frame
    # 65 saniye = 130 kare
    hedef_saniyeler = [i * 0.5 for i in range(130)] # 0.0, 0.5, 1.0 ... 64.5
    
    # Ornek Bitki Skoru (Teksas konumu icin onceki analizden %54.2 cikmisti)
    BITKI_SKORU = 0.542
    MAX_ETKI = 0.05
    ESIK = 0.65

    kaydedilen_sayisi = 0

    for sn in hedef_saniyeler:
        frame_idx = int(sn * fps)
        
        # Kameradan o frame'i oku
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Fotograf olarak kaydet (cv2.imwrite Turkce karakterli yollarda sessizce coktugu icin alternatif yontem kullaniyoruz)
        resim_adi = f"frame_{frame_idx:04d}_{sn}s.jpg"
        resim_yolu = os.path.join(FRAMES_OUT_DIR, resim_adi)
        is_success, im_buf_arr = cv2.imencode(".jpg", frame)
        if is_success:
            im_buf_arr.tofile(resim_yolu)

        # O saniyeye en yakin GPS verisini bul
        fark = (df_gps['time_s'] - sn).abs()
        en_yakin_idx = fark.idxmin()
        en_yakin_gps = df_gps.loc[en_yakin_idx]
        
        enlem = en_yakin_gps.get('OSD.latitude', 0.0)
        boylam = en_yakin_gps.get('OSD.longitude', 0.0)

        # O frame icin Kamera Skoru bul (Birebir eslesmek yerine en yakin zamani/kareyi alalim)
        if 'timestamp_sec' in df_preds.columns:
            fark_pred = (df_preds['timestamp_sec'] - sn).abs()
            en_yakin_pred_idx = fark_pred.idxmin()
            kamera_skoru = float(df_preds.loc[en_yakin_pred_idx][prob_col])
        else:
            fark_pred = (df_preds['frame_idx'] - frame_idx).abs()
            en_yakin_pred_idx = fark_pred.idxmin()
            kamera_skoru = float(df_preds.loc[en_yakin_pred_idx][prob_col])

        # Yaniclik Entegrasyonu Hesabi
        modifiye_deger = ((BITKI_SKORU - 0.50) / 0.50) * MAX_ETKI
        yangin_yuzdesi = kamera_skoru + modifiye_deger
        yangin_yuzdesi = max(0.0, min(1.0, yangin_yuzdesi)) # 0-1 arasi sinirla
        
        yangin_var = bool(yangin_yuzdesi > ESIK)

        # Veritabanina Ekle (Sadece istenen 5 sutun + otomatik geom/zaman + resim_adi)
        cur.execute("""
            INSERT INTO yangin_tahminleri
            (enlem, boylam, yangin_var, yangin_yuzdesi, geom, resim_adi)
            VALUES (%s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s)
        """, (float(enlem), float(boylam), yangin_var, yangin_yuzdesi, float(boylam), float(enlem), resim_adi))

        kaydedilen_sayisi += 1
        conn.commit()
        
        if kaydedilen_sayisi % 10 == 0:
            print(f"  {kaydedilen_sayisi} kare islendi ve veritabanina kaydedildi...")
            
        time.sleep(1) # Gercek zamanli drone ucusu hissi vermek icin 1 saniye bekle
    cur.close()
    conn.close()
    cap.release()

    print(f"[BASARILI] Toplam {kaydedilen_sayisi} kayit veritabanina eklendi!")
    print(f"[BILGI] Cikarilan fotograflar suraya kaydedildi: {FRAMES_OUT_DIR}")

if __name__ == "__main__":
    db_hazirla()
    veri_isle_ve_kaydet()
