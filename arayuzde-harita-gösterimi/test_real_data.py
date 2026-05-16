import pandas as pd
import joblib
import sys

# Terminalde UTF-8 sorunlarını çözmek için
sys.stdout.reconfigure(encoding='utf-8')

# 1. Modeli ve beklenen sütunları yükle
model_data = joblib.load('fuel_scorer_model.pkl')
model = model_data['model']
expected_features = model_data['features']

# 2. Gerçek veri setini yükle
df = pd.read_csv('final_ml_dataset.csv').dropna()

# One-hot encoding işlemini gerçek veri için yap (modelin formatına uydurmak için)
df_model_format = df.copy()
df_model_format['land_cover'] = df_model_format['land_cover'].astype(int).astype(str)
df_model_format = pd.get_dummies(df_model_format, columns=['land_cover'], prefix='LC')

# Modelin beklediği sütunların tam olduğundan emin ol (eksik varsa 0 koy)
for feature in expected_features:
    if feature not in df_model_format.columns:
        df_model_format[feature] = 0

# Sütun sırasını modelle aynı yap
X = df_model_format[expected_features]
y_gercek = df['label']
orijinal_bilgiler = df[['latitude', 'longitude', 'acq_date', 'ndvi', 'ndmi']] # Ekrana basmak icin

print("="*60)
print(" GERÇEK UYDU VERİLERİYLE MODEL TESTİ")
print("="*60)

# --- YANGIN OLAN GERCEK 3 NOKTA (LABEL = 1) ---
print("\n[🔥 GERCEKTEN YANGIN CIKAN BOLGELER]")
yangin_indexleri = df[df['label'] == 1].sample(3, random_state=42).index

for idx in yangin_indexleri:
    gercek_degerler = orijinal_bilgiler.loc[idx]
    model_girdisi = X.loc[[idx]]
    
    # Tahmin
    skor = model.predict_proba(model_girdisi)[0][1]
    
    print(f"\n-> Tarih/Konum: {gercek_degerler['acq_date']} | {gercek_degerler['latitude']:.2f}, {gercek_degerler['longitude']:.2f}")
    print(f"   Çevre: NDVI={gercek_degerler['ndvi']:.2f}, NDMI={gercek_degerler['ndmi']:.2f}")
    print(f"   GERÇEK DURUM: YANGIN VAR")
    print(f"   YAPAY ZEKA YANICILIK SKORU: %{skor*100:.1f}")

# --- YANGIN OLMAYAN GERCEK 3 NOKTA (LABEL = 0) ---
print("\n\n[🌲 YANGIN ÇIKMAYAN NORMAL BOLGELER]")
normal_indexler = df[df['label'] == 0].sample(3, random_state=42).index

for idx in normal_indexler:
    gercek_degerler = orijinal_bilgiler.loc[idx]
    model_girdisi = X.loc[[idx]]
    
    # Tahmin
    skor = model.predict_proba(model_girdisi)[0][1]
    
    print(f"\n-> Tarih/Konum: {gercek_degerler['acq_date']} | {gercek_degerler['latitude']:.2f}, {gercek_degerler['longitude']:.2f}")
    print(f"   Çevre: NDVI={gercek_degerler['ndvi']:.2f}, NDMI={gercek_degerler['ndmi']:.2f}")
    print(f"   GERÇEK DURUM: YANGIN YOK")
    print(f"   YAPAY ZEKA YANICILIK SKORU: %{skor*100:.1f}")

print("\n" + "="*60)
