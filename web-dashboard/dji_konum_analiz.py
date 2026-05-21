import ee
import math
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# DJI UCUS KAYDI KONUM ANALIZI
# ============================================================

# DJI Flight Record'dan alinan koordinatlar
lat = 29.5771921690672
lon = -95.7542586759733
tarih = "2019-01-09"  # Ucus tarihi

# Lojistik Regresyon Katsayilari
INTERCEPT = -0.5907
NDVI_COEF = -1.3428
NDMI_COEF = -1.3042
LC_COEFS = {
    0: -2.7629, 1: 2.0896, 2: -0.0382, 3: -0.3532,
    4: 1.1267, 5: 1.6451, 6: -0.1594, 7: -0.2806, 8: -1.2702,
}

SINIF_ISIMLERI = {
    0: "Su (Water)",
    1: "Agac / Orman (Trees)",
    2: "Otlak (Grass)",
    3: "Su basmis bitki ortusu (Flooded vegetation)",
    4: "Tarim alani (Crops)",
    5: "Calilik (Shrub & Scrub)",
    6: "Yapi / Sehir (Built Area)",
    7: "Ciplak toprak (Bare ground)",
    8: "Kar ve Buz (Snow & Ice)",
}

def sigmoid(x):
    if x > 500: return 1.0
    if x < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-x))

# GEE Baglanti
ee.Initialize(project='bitirme-proje-494721')

nokta = ee.Geometry.Point([lon, lat])

# Tarih araligi (ucus tarihi etrafinda 30 gun)
baslangic = "2018-12-01"
bitis = "2019-02-15"

print("=" * 65)
print(" DJI UCUS KAYDI - CEVRE ANALIZI")
print("=" * 65)
print(f" Koordinat: {lat:.6f} N, {lon:.6f} W")
print(f" Ucus Tarihi: {tarih}")
print(f" Google Maps: https://www.google.com/maps?q={lat},{lon}&z=15")
print("=" * 65)

# DYNAMIC WORLD (Land Cover)
print("\n Arazi sinifi sorgulaniyoruz...")
dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
    .filterBounds(nokta) \
    .filterDate(baslangic, bitis)

lc_kod = None
if dw.size().getInfo() > 0:
    dw_img = dw.sort('system:time_start', False).first()
    lc_data = dw_img.select('label').reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=nokta,
        scale=10
    ).getInfo()
    lc_kod = lc_data.get('label')
    lc_adi = SINIF_ISIMLERI.get(lc_kod, "Bilinmeyen")
    print(f" Arazi Sinifi: LC_{lc_kod} -> {lc_adi}")
else:
    print(" Dynamic World goruntusi bulunamadi!")
    lc_kod = 2  # Default otlak

# SENTINEL-2 (NDVI & NDMI)
print(" Sentinel-2 sorgulaniyoruz...")
s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(nokta) \
    .filterDate(baslangic, bitis) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))

goruntu_sayisi = s2.size().getInfo()
if goruntu_sayisi > 0:
    kompozit = s2.median()
    ndvi = kompozit.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndmi = kompozit.normalizedDifference(['B8', 'B11']).rename('NDMI')
    
    degerler = ndvi.addBands(ndmi).reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=nokta,
        scale=10
    ).getInfo()
    
    ndvi_val = degerler.get('NDVI')
    ndmi_val = degerler.get('NDMI')
    
    if ndvi_val is not None and ndmi_val is not None:
        print(f" NDVI: {ndvi_val:.4f}")
        print(f" NDMI: {ndmi_val:.4f}")
        print(f" Sentinel-2 goruntu sayisi: {goruntu_sayisi}")
        
        # YANICLIK SKORU HESAPLA
        ham_skor = INTERCEPT + (NDVI_COEF * ndvi_val) + (NDMI_COEF * ndmi_val) + LC_COEFS[lc_kod]
        yuzde = sigmoid(ham_skor) * 100
        
        print("\n" + "=" * 65)
        print(" SONUCLAR")
        print("=" * 65)
        print(f" NDVI:            {ndvi_val:.4f}")
        print(f" NDMI:            {ndmi_val:.4f}")
        print(f" Land Cover:      LC_{lc_kod} ({lc_adi})")
        print(f" Yaniclik Skoru:  %{yuzde:.1f}")
        
        if yuzde > 70:
            print(f" Risk Seviyesi:   YUKSEK RISK")
        elif yuzde > 50:
            print(f" Risk Seviyesi:   ORTA RISK")
        elif yuzde > 30:
            print(f" Risk Seviyesi:   DUSUK RISK")
        else:
            print(f" Risk Seviyesi:   COK DUSUK RISK")
        print("=" * 65)
    else:
        print(" NDVI/NDMI alinamadi!")
else:
    print(" Sentinel-2 goruntusu bulunamadi!")
