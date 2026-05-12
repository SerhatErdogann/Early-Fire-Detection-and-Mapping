import pandas as pd
import ee
import random
import os
import time
from datetime import datetime, timedelta

# Hatalari onlemek icin UTF-8 terminal destegi
import sys
sys.stdout.reconfigure(encoding='utf-8')

def initialize_gee():
    try:
        ee.Initialize(project='bitirme-proje-494721')
    except Exception as e:
        print("GEE kimlik dogrulama hatasi:", e)
        exit(1)

def get_gee_data(lat, lon, start_date_str, end_date_str):
    nokta = ee.Geometry.Point([lon, lat])
    
    # 1. Land Cover
    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterBounds(nokta).filterDate(start_date_str, end_date_str)
    try:
        dw_size = dw.size().getInfo()
        if dw_size > 0:
            goruntu = dw.sort('system:time_start', False).first()
            lc_val = goruntu.select('label').reduceRegion(reducer=ee.Reducer.first(), geometry=nokta, scale=10).getInfo().get('label')
        else:
            lc_val = None
    except Exception:
        lc_val = None
        
    # 2. NDVI & NDMI
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(nokta).filterDate(start_date_str, end_date_str).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    try:
        s2_size = s2.size().getInfo()
        if s2_size > 0:
            kompozit = s2.median()
            ndvi = kompozit.normalizedDifference(['B8', 'B4']).rename('NDVI')
            ndmi = kompozit.normalizedDifference(['B8', 'B11']).rename('NDMI')
            degerler = ndvi.addBands(ndmi).reduceRegion(reducer=ee.Reducer.first(), geometry=nokta, scale=10).getInfo()
            ndvi_val = degerler.get('NDVI')
            ndmi_val = degerler.get('NDMI')
        else:
            ndvi_val, ndmi_val = None, None
    except Exception:
        ndvi_val, ndmi_val = None, None

    return ndvi_val, ndmi_val, lc_val

def generate_dataset():
    initialize_gee()
    
    input_file = 'filtered_fires_2k.csv'
    output_file = 'temporal_ml_dataset.csv'
    
    df_input = pd.read_csv(input_file)
    
    # Kaldigimiz yeri bulmak icin
    islenmis_satirlar = 0
    if os.path.exists(output_file):
        df_out = pd.read_csv(output_file)
        islenmis_satirlar = len(df_out) // 2 # Her dongude 2 satir (pozitif+negatif) ekliyoruz
        print(f"\n[DIKKAT] {output_file} bulundu. Kaldigimiz yer: {islenmis_satirlar}. satirdan devam ediliyor...\n")
    else:
        # Yeni dosya basliklari yazdir
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("latitude,longitude,acq_date,ndvi,ndmi,land_cover,label\n")
            
    print(f"Toplam islenecek satir: {len(df_input)}")
    print("Islem basladi. Bu pencereyi kapatirsaniz, kod tekrar actiginizda kaldigi yerden devam eder.\n")
    
    for index, row in df_input.iterrows():
        if index < islenmis_satirlar:
            continue # Bu satiri zaten islemisiz
            
        lat = row['latitude']
        lon = row['longitude']
        acq_date_str = str(row['acq_date']) # YYYY-MM-DD
        
        # Zaman araligini hesapla (Tarih - 20 ile Tarih - 5)
        try:
            acq_date = datetime.strptime(acq_date_str, "%Y-%m-%d")
            start_date = (acq_date - timedelta(days=20)).strftime("%Y-%m-%d")
            end_date = (acq_date - timedelta(days=5)).strftime("%Y-%m-%d")
        except ValueError:
            print(f"[{index+1}] Tarih formati hatali: {acq_date_str}, atlandi.")
            continue
        
        # 1. POZITIF VERI (Kuru Yangin Zamani)
        p_ndvi, p_ndmi, p_lc = get_gee_data(lat, lon, start_date, end_date)
        
        # 2. NEGATIF VERI (Temporal Shifting - Zaman Kaydirma)
        # Ayni ormanda, ayni koordinatta, ama yaklasik 5-6 ay oncesi (Yagisli kis/bahar sezonu)
        n_start_date = (acq_date - timedelta(days=170)).strftime("%Y-%m-%d")
        n_end_date = (acq_date - timedelta(days=155)).strftime("%Y-%m-%d")
        
        # Negatif veri ayni koordinatta (lat, lon) cekiliyor
        n_ndvi, n_ndmi, n_lc = get_gee_data(lat, lon, n_start_date, n_end_date)
        
        # CSV'ye yaz (Eger None degillerse)
        with open(output_file, 'a', encoding='utf-8') as f:
            # Pozitif
            if p_ndvi is not None and p_ndmi is not None and p_lc is not None:
                f.write(f"{lat},{lon},{acq_date_str},{p_ndvi:.4f},{p_ndmi:.4f},{p_lc},1\n")
            
            # Negatif
            if n_ndvi is not None and n_ndmi is not None and n_lc is not None:
                f.write(f"{lat:.5f},{lon:.5f},{n_start_date},{n_ndvi:.4f},{n_ndmi:.4f},{n_lc},0\n")
                
        print(f"[{index+1}/{len(df_input)}] Islendi -> Pozitif: ({p_ndvi}, {p_ndmi}) | Negatif: ({n_ndvi}, {n_ndmi})")
        
        # API limitini asmamak icin ufak bekleme
        time.sleep(0.5) 

if __name__ == "__main__":
    generate_dataset()
