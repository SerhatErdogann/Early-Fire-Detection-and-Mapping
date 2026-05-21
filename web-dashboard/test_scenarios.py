import pandas as pd
import joblib
import warnings
import sys

# Uyarilari gizle
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import os

# Eğitilmiş Modeli Yükle (Dosya yolunu otomatik bul)
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'fuel_scorer_model.pkl')

model_data = joblib.load(model_path)
model = model_data['model']
expected_features = model_data['features']

def test_et(senaryo_adi, ndvi, ndmi, arazi_kodu):
    # Modelin beklediği formatta (hepsi 0 olan) boş bir satır oluştur
    input_data = {f: 0.0 for f in expected_features}
    
    # Manuel değerleri gir
    input_data['ndvi'] = ndvi
    input_data['ndmi'] = ndmi
    
    # Hangi arazi tipiyse sadece onun sütununu 1 yap (One-Hot Encoding)
    lc_column = f"LC_{arazi_kodu}"
    if lc_column in input_data:
        input_data[lc_column] = 1.0
        
    df = pd.DataFrame([input_data])
    
    # Modelden % cinsinden Yanıcılık İhtimalini (Skoru) iste
    # predict_proba() bize [0 olma ihtimali, 1 olma ihtimali] döner. Biz 1'i (Yangın) alıyoruz.
    skor = model.predict_proba(df)[0][1]
    
    print(f"\n>> {senaryo_adi}")
    print(f"   Girdiler: NDVI={ndvi}, NDMI={ndmi}, Arazi Sınıfı={arazi_kodu}")
    print(f"   🔥 MODELIN VERDIGI YANICILIK SKORU: %{skor*100:.1f}")
    print("-" * 60)

print("=" * 60)
print(" YAPAY ZEKA YANICILIK DENKLEMI SIMULASYON TESTI")
print("=" * 60)

# Dynamic World Sınıfları: 1=Ağaçlar, 4=Tarım Arazisi, 6=Şehir/Beton, 0=Su

test_et("Senaryo 1: Kupkuru ve Sık Bir Orman (En yüksek risk beklenir)", 
        ndvi=0.85, ndmi=-0.35, arazi_kodu=1)

test_et("Senaryo 2: Yağmur Yemiş, Islak Bir Orman (Düşük risk beklenir)", 
        ndvi=0.40, ndmi=0.45, arazi_kodu=1)

test_et("Senaryo 3: Hasat Sonrası Kurumuş Tarım Arazisi / Anız (Yüksek risk beklenir)", 
        ndvi=0.20, ndmi=-0.50, arazi_kodu=4)

test_et("Senaryo 4: Beton Şehir Merkezi (Sıfıra yakın risk beklenir)", 
        ndvi=0.05, ndmi=0.00, arazi_kodu=6)

test_et("Senaryo 5: Göl / Deniz Ortası (Sıfır risk beklenir)", 
        ndvi=-0.20, ndmi=0.80, arazi_kodu=0)
