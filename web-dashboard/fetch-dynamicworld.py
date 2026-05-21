import ee
import sys
sys.stdout.reconfigure(encoding='utf-8')

# -- Dynamic World Sinif Eslestirmesi --
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

# Earth Engine'e giris yap
ee.Initialize(project='bitirme-proje-494721')

# SCU Lightning Complex Fire koordinatlari (Agustos 2020)
boylam = -121.300
enlem = 37.430
nokta = ee.Geometry.Point([boylam, enlem])

print(f"Koordinat: {enlem} N, {boylam} E")
print(f"Yangin Tarihi: 16 Agustos 2020")
print("=" * 60)

# Incelenecek Donemler
donemler = {
    "YANGINDAN ONCE": {"baslangic": "2020-07-01", "bitis": "2020-08-15"},
    "YANGINDAN SONRA": {"baslangic": "2020-08-17", "bitis": "2020-10-15"}
}

for donem_adi, tarihler in donemler.items():
    baslangic = tarihler["baslangic"]
    bitis = tarihler["bitis"]
    
    print(f"\n[{donem_adi}] ({baslangic} - {bitis})")
    print("-" * 50)
    
    # DYNAMIC WORLD (Land Cover)
    dw_koleksiyonu = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
        .filterBounds(nokta) \
        .filterDate(baslangic, bitis)
    
    if dw_koleksiyonu.size().getInfo() > 0:
        dw_goruntu = dw_koleksiyonu.sort('system:time_start', False).first()
        arazi_verisi = dw_goruntu.select('label').reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=nokta,
            scale=10
        ).getInfo()
        
        sinif_kodu = arazi_verisi.get('label')
        sinif_adi = SINIF_ISIMLERI.get(sinif_kodu, "Bilinmeyen")
        print(f"  [Land Cover] Sinif Kodu: {sinif_kodu} -> {sinif_adi}")
    else:
        print("  [Land Cover] Goruntu bulunamadi.")

    # SENTINEL-2 (NDVI & NDMI)
    s2_koleksiyonu = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(nokta) \
        .filterDate(baslangic, bitis) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) # Sadece bulutsuz/az bulutlu olanlari al
    
    if s2_koleksiyonu.size().getInfo() > 0:
        # Medyan kompozit (bulut etkisini en aza indirmek icin)
        kompozit = s2_koleksiyonu.median()
        
        # NDVI ve NDMI hesapla
        ndvi = kompozit.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndmi = kompozit.normalizedDifference(['B8', 'B11']).rename('NDMI')
        
        degerler = ndvi.addBands(ndmi).reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=nokta,
            scale=10
        ).getInfo()
        
        ndvi_val = degerler.get('NDVI')
        ndmi_val = degerler.get('NDMI')
        
        # NDVI Yorumu (Yakici madde miktari / Fuel Load)
        if ndvi_val is None:
            ndvi_yorum = "Bilinmiyor"
        elif ndvi_val > 0.6:
            ndvi_yorum = "Cok yogun orman (Ekstrem yakit yuksek tepe yangini riski)"
        elif ndvi_val > 0.4:
            ndvi_yorum = "Orta yogunlukta orman (Yuksek yakit)"
        elif ndvi_val > 0.2:
            ndvi_yorum = "Calilik, otlak, seyrek orman (Kritik ince yakit - hizli yayilma)"
        elif ndvi_val > 0.1:
            ndvi_yorum = "Cok seyrek bitki / Toprak (Cok dusuk yakit)"
        else:
            ndvi_yorum = "Su, kar, ciplak kaya (Yakit yok)"
            
        # NDMI Yorumu (Yakici madde durumu / Fuel Moisture)
        if ndmi_val is None:
            ndmi_yorum = "Bilinmiyor"
        elif ndmi_val > 0.2:
            ndmi_yorum = "Yuksek nemli (Dusuk yangin riski)"
        elif ndmi_val > 0.0:
            ndmi_yorum = "Orta seviye nem (Orta yangin riski)"
        elif ndmi_val > -0.2:
            ndmi_yorum = "Kuru / Su stresi (Cok yuksek yangin riski)"
        else:
            ndmi_yorum = "Asiri kuru veya yanmis (Ekstrem risk)"
        
        print(f"  [NDVI]       Deger     : {ndvi_val:.4f} ({ndvi_yorum})")
        print(f"  [NDMI]       Deger     : {ndmi_val:.4f} ({ndmi_yorum})")
        
    else:
        print("  [Sentinel-2] Uygun goruntu bulunamadi.")

print("\n" + "=" * 60)