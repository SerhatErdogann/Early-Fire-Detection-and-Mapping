import psycopg2
import pandas as pd
import cv2
import numpy as np
import os
import math
import sys
import time
import torch
import joblib

sys.stdout.reconfigure(encoding='utf-8')

# Script'in kendi dizinini bul
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Ana proje dizini (bitki-verisi)
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# bitki-verisi dizinini Python path'e ekle (yerel models paketi icin)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.dual_branch import load_checkpoint, prep_rgb, prep_thermal

# ==============================================================
# YAPILANDIRMA (CONFIGURATION)
# ==============================================================

# Veritabani baglantisi
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "cografi_veritabani",
    "user": "postgres",
    "password": "1313"
}

# Dosya Yollari
RGB_VIDEO_PATH = os.path.join(BASE_DIR, "test-videos", "RGB-video.MP4")
THERMAL_VIDEO_PATH = os.path.join(BASE_DIR, "test-videos", "Thermal-video.MP4")
CSV_GPS_PATH = os.path.join(BASE_DIR, "test-videos", "Antalya_Orman_Ucusu.csv")
DUALBRANCH_MODEL_PATH = os.path.join(BASE_DIR, "models", "dual_branch.pt")
FUEL_SCORER_PATH = os.path.join(BASE_DIR, "models", "fuel_scorer_model.pkl")
FRAMES_OUT_DIR = os.path.join(SCRIPT_DIR, "extracted_frames")

os.makedirs(FRAMES_OUT_DIR, exist_ok=True)

# Sabitler
MAX_ETKI = 0.05   # Bitki skorunun kamera skoruna max etkisi (+-%5)
ESIK = 0.65        # Yangin var/yok esik degeri
KONUM_ESIK_KM = 1.5  # Fuel scorer'i yeniden hesaplamak icin minimum mesafe (km)

# ==============================================================
# YARDIMCI FONKSIYONLAR
# ==============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Iki GPS koordinati arasindaki mesafeyi km olarak hesaplar (Haversine formulu)."""
    R = 6371.0  # Dunyanin yaricapi (km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


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


def dualbranch_yukle():
    """DualBranch modelini yukler ve hazirlar."""
    print(f"[MODEL] DualBranch yukleniyor: {DUALBRANCH_MODEL_PATH}")
    model, mode, device, thr, temperature = load_checkpoint(DUALBRANCH_MODEL_PATH)
    model.eval()
    print(f"[MODEL] DualBranch hazir! Mode={mode}, Device={device}, Threshold={thr:.3f}, Temp={temperature:.3f}")
    return model, device, thr, temperature


def fuel_scorer_yukle():
    """Fuel scorer (bitki yaniclik) modelini yukler."""
    print(f"[MODEL] Fuel Scorer yukleniyor: {FUEL_SCORER_PATH}")
    model_data = joblib.load(FUEL_SCORER_PATH)
    model = model_data['model']
    features = model_data['features']
    print(f"[MODEL] Fuel Scorer hazir! Beklenen ozellikler: {features}")
    return model, features


def fuel_skoru_hesapla_gee(lat, lon, ucus_tarihi, fuel_model, fuel_features):
    """
    Google Earth Engine uzerinden NDVI, NDMI ve Land Cover cekip
    fuel scorer modeline vererek yaniclik skoru hesaplar.
    
    ucus_tarihi: datetime objesi (CSV'den dinamik olarak okunur)
    """
    try:
        import ee
        from datetime import timedelta
        ee.Initialize(project='bitirme-proje-494721')
        
        nokta = ee.Geometry.Point([lon, lat])
        # Tam o noktada veri yoksa diye etrafındaki 100 metrelik alanı (buffer) alıyoruz
        nokta_cevre = nokta.buffer(100)
        
        # Ucus tarihinden SADECE ONCEKI 2 ayi al (yangin sonrasi yanmis bitki verisini almamak icin)
        baslangic = (ucus_tarihi - timedelta(days=60)).strftime("%Y-%m-%d")
        bitis = ucus_tarihi.strftime("%Y-%m-%d")
        print(f"  [FUEL] GEE tarih araligi: {baslangic} - {bitis} (ucustan onceki 2 ay)")
        
        # Dynamic World - Land Cover
        dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
            .filterBounds(nokta_cevre) \
            .filterDate(baslangic, bitis)
        
        if dw.size().getInfo() > 0:
            dw_img = dw.sort('system:time_start', False).first()
            lc_data = dw_img.select('label').reduceRegion(
                reducer=ee.Reducer.mode(), # Çevredeki en yaygın bitki örtüsünü al
                geometry=nokta_cevre,
                scale=10
            ).getInfo()
            lc_kod = lc_data.get('label')
            if lc_kod is None:
                raise ValueError("100m çevrede Land Cover verisi bulunamadı.")
        else:
            raise ValueError("Belirtilen tarihte Land Cover görüntüsü bulunamadı.")
        
        # Sentinel-2 - NDVI & NDMI
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(nokta_cevre) \
            .filterDate(baslangic, bitis) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        
        if s2.size().getInfo() > 0:
            kompozit = s2.median()
            ndvi = kompozit.normalizedDifference(['B8', 'B4']).rename('NDVI')
            ndmi = kompozit.normalizedDifference(['B8', 'B11']).rename('NDMI')
            degerler = ndvi.addBands(ndmi).reduceRegion(
                reducer=ee.Reducer.mean(), # Çevrenin ortalama bitki yoğunluğu ve nemini al
                geometry=nokta_cevre,
                scale=10
            ).getInfo()
            ndvi_val = degerler.get('NDVI')
            ndmi_val = degerler.get('NDMI')
            if ndvi_val is None or ndmi_val is None:
                raise ValueError("100m çevrede NDVI veya NDMI verisi bulunamadı.")
        else:
            raise ValueError("Belirtilen tarihte bulutsuz Sentinel-2 görüntüsü bulunamadı.")
        
        # Fuel scorer modeline ver
        girdi = pd.DataFrame({'ndvi': [ndvi_val], 'ndmi': [ndmi_val]})
        girdi['land_cover'] = str(int(lc_kod))
        girdi = pd.get_dummies(girdi, columns=['land_cover'], prefix='LC')
        
        for f in fuel_features:
            if f not in girdi.columns:
                girdi[f] = 0
        girdi = girdi[fuel_features]
        
        skor = fuel_model.predict_proba(girdi)[0][1]
        
        print(f"  [FUEL] GEE Sorgusu tamamlandi: NDVI={ndvi_val:.4f}, NDMI={ndmi_val:.4f}, LC={lc_kod} -> Yaniclik=%{skor*100:.1f}")
        return skor
        
    except Exception as e:
        print(f"  [FUEL] GEE Hatasi: {e}")
        print(f"  [FUEL] Varsayilan skor kullanilacak (0.50)")
        return 0.50


def dualbranch_tahmin(model, device, temperature, rgb_frame, thermal_frame, size=384):
    """
    Tek bir RGB + Thermal frame ciftini DualBranch modeline verip
    yangin olasiligini (0-1 arasi) dondurur.
    """
    # Preprocessing
    rgb_arr, _ = prep_rgb(rgb_frame, size=size)
    th_arr, _ = prep_thermal(thermal_frame, size=size)
    
    # Fusion input: RGB (3ch) + Thermal (1ch) = 4ch
    x_np = np.concatenate([rgb_arr, th_arr], axis=0)
    x = torch.from_numpy(np.ascontiguousarray(x_np)).unsqueeze(0).float().to(device)
    
    with torch.inference_mode():
        logits = model(x)
        scores_cal = logits / max(1e-6, temperature)
        prob = torch.softmax(scores_cal, dim=1)[0, 1].item()
    
    return prob


# ==============================================================
# ANA ISLEM DONGUSU
# ==============================================================

def veri_isle_ve_kaydet():
    # 1. Modelleri yukle
    db_model, db_device, db_thr, db_temp = dualbranch_yukle()
    fuel_model, fuel_features = fuel_scorer_yukle()
    
    # 2. GPS verisini oku
    print("[1] GPS verisi yukleniyor...")
    df_gps = pd.read_csv(CSV_GPS_PATH, skiprows=1, low_memory=False)
    if 'OSD.flyTime [s]' in df_gps.columns:
        df_gps['time_s'] = pd.to_numeric(df_gps['OSD.flyTime [s]'], errors='coerce')
    else:
        print("[HATA] GPS dosyasinda 'OSD.flyTime [s]' bulunamadi.")
        return

    # 3. Ucus tarihini CSV'den dinamik olarak oku
    from datetime import datetime
    ucus_tarihi = None
    if 'CUSTOM.date [local]' in df_gps.columns:
        tarih_str = str(df_gps['CUSTOM.date [local]'].dropna().iloc[0])
        # DJI formati: "1/9/2019" veya "2019-01-09" olabilir
        for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"):
            try:
                ucus_tarihi = datetime.strptime(tarih_str.strip(), fmt)
                break
            except ValueError:
                continue
    
    if ucus_tarihi is None:
        # CSV dosya adindan tarihi cikar (DJIFlightRecord_2019-01-09_...)
        import re
        tarih_match = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(CSV_GPS_PATH))
        if tarih_match:
            ucus_tarihi = datetime.strptime(tarih_match.group(1), "%Y-%m-%d")
        else:
            print("[UYARI] Ucus tarihi bulunamadi, bugunun tarihi kullanilacak.")
            ucus_tarihi = datetime.now()
    
    print(f"[BILGI] Ucus tarihi: {ucus_tarihi.strftime('%Y-%m-%d')} (CSV'den dinamik okundu)")

    # 4. Videolari ac
    cap_rgb = cv2.VideoCapture(RGB_VIDEO_PATH)
    cap_thermal = cv2.VideoCapture(THERMAL_VIDEO_PATH)
    
    if not cap_rgb.isOpened():
        print(f"[HATA] RGB video acilamadi: {RGB_VIDEO_PATH}")
        return
    if not cap_thermal.isOpened():
        print(f"[HATA] Thermal video acilamadi: {THERMAL_VIDEO_PATH}")
        return
    
    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    # Video suresini dinamik olarak hesapla
    toplam_frame = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    video_suresi_sn = toplam_frame / fps
    
    # Saniyede 1 kare (1 FPS) - video suresi kadar
    hedef_fps = 1
    toplam_kare = int(video_suresi_sn * hedef_fps)
    hedef_saniyeler = [i * (1.0 / hedef_fps) for i in range(toplam_kare)]

    print(f"[2] Video isleniyor ({fps:.2f} FPS, {video_suresi_sn:.1f} saniye)...")
    print(f"    Saniyede {hedef_fps} kare -> Toplam {toplam_kare} kare cikarilacak.")
    print(f"[3] DualBranch ANLIK tahmin yapacak (onceden hesaplanmis CSV kullanilmayacak!)")
    print(f"[4] Fuel Scorer ilk koordinat icin GEE'den cekilecek, konum degismezse tekrar sorulmayacak.")
    print("=" * 70)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Fuel scorer cache degiskenleri
    son_fuel_lat = None
    son_fuel_lon = None
    son_fuel_skor = 0.50  # Varsayilan
    
    kaydedilen_sayisi = 0

    for sn in hedef_saniyeler:
        frame_idx = int(sn * fps)
        
        # --- RGB Frame Oku ---
        cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret_rgb, rgb_frame = cap_rgb.read()
        if not ret_rgb:
            print(f"  [UYARI] RGB frame {frame_idx} okunamadi, atlaniyor...")
            break
        
        # --- Thermal Frame Oku ---
        cap_thermal.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret_th, thermal_frame = cap_thermal.read()
        if not ret_th:
            print(f"  [UYARI] Thermal frame {frame_idx} okunamadi, atlaniyor...")
            break
            
        # --- Fotograf olarak kaydet ---
        resim_adi = f"frame_{frame_idx:04d}_{sn}s.jpg"
        resim_yolu = os.path.join(FRAMES_OUT_DIR, resim_adi)
        is_success, im_buf_arr = cv2.imencode(".jpg", rgb_frame)
        if is_success:
            im_buf_arr.tofile(resim_yolu)

        # --- GPS Koordinatini Bul ---
        fark = (df_gps['time_s'] - sn).abs()
        en_yakin_idx = fark.idxmin()
        en_yakin_gps = df_gps.loc[en_yakin_idx]
        enlem = float(en_yakin_gps.get('OSD.latitude', 0.0))
        boylam = float(en_yakin_gps.get('OSD.longitude', 0.0))

        # ============================================================
        # ADIM A: DUALBRANCH ANLIK TAHMIN (Her frame icin model calisir)
        # ============================================================
        kamera_skoru = dualbranch_tahmin(db_model, db_device, db_temp, rgb_frame, thermal_frame)

        # ============================================================
        # ADIM B: FUEL SCORER (Konum degismediyse cache'den al)
        # ============================================================
        if son_fuel_lat is None:
            # Ilk frame: GEE'den cek
            print(f"\n  [FUEL] Ilk koordinat icin GEE sorgulaniyor ({enlem:.6f}, {boylam:.6f})...")
            son_fuel_skor = fuel_skoru_hesapla_gee(enlem, boylam, ucus_tarihi, fuel_model, fuel_features)
            son_fuel_lat = enlem
            son_fuel_lon = boylam
        else:
            # Mesafeyi kontrol et
            mesafe = haversine_km(son_fuel_lat, son_fuel_lon, enlem, boylam)
            if mesafe > KONUM_ESIK_KM:
                print(f"\n  [FUEL] Konum {mesafe:.2f} km degisti! Yeni GEE sorgusu yapiliyor ({enlem:.6f}, {boylam:.6f})...")
                son_fuel_skor = fuel_skoru_hesapla_gee(enlem, boylam, ucus_tarihi, fuel_model, fuel_features)
                son_fuel_lat = enlem
                son_fuel_lon = boylam

        # ============================================================
        # ADIM C: ENTEGRE SKOR HESAPLA
        # ============================================================
        modifiye_deger = ((son_fuel_skor - 0.50) / 0.50) * MAX_ETKI
        yangin_yuzdesi = kamera_skoru + modifiye_deger
        yangin_yuzdesi = max(0.0, min(1.0, yangin_yuzdesi))
        
        yangin_var = bool(yangin_yuzdesi > ESIK)

        # ============================================================
        # ADIM D: VERITABANINA KAYDET
        # ============================================================
        cur.execute("""
            INSERT INTO yangin_tahminleri
            (enlem, boylam, yangin_var, yangin_yuzdesi, geom, resim_adi)
            VALUES (%s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s)
        """, (enlem, boylam, yangin_var, yangin_yuzdesi, boylam, enlem, resim_adi))

        kaydedilen_sayisi += 1
        conn.commit()
        
        # Konsola durum bildirimi
        durum = "YANGIN!" if yangin_var else "Temiz"
        print(f"  Frame {kaydedilen_sayisi:>3}/{len(hedef_saniyeler)} | t={sn:>5.1f}s | "
              f"Kamera={kamera_skoru*100:>5.1f}% | Fuel={son_fuel_skor*100:>5.1f}% | "
              f"Final={yangin_yuzdesi*100:>5.1f}% | {durum}")
            
        time.sleep(1)  # Gercek zamanli drone ucusu hissi vermek icin 1 saniye bekle

    cur.close()
    conn.close()
    cap_rgb.release()
    cap_thermal.release()

    print(f"\n[BASARILI] Toplam {kaydedilen_sayisi} kayit veritabanina eklendi!")
    print(f"[BILGI] Cikarilan fotograflar: {FRAMES_OUT_DIR}")
    print(f"[BILGI] DualBranch modeli HER FRAME icin anlik tahmin yapti.")
    print(f"[BILGI] Fuel scorer {KONUM_ESIK_KM} km esik ile cache'lendi.")


if __name__ == "__main__":
    db_hazirla()
    veri_isle_ve_kaydet()
