import pandas as pd
import math
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

# Script'in kendi dizinini bul
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# DUAL BRANCH + YANICLIK SKORU ENTEGRASYON TESTI
# ============================================================

# --- 1) YANICLIK SKORU HESAPLA (GEE'den daha once cektigimiz veriler) ---
# FLAME 2 Bolge: Kaibab Ormani, Arizona
# Yangindan hemen once (Yaz 2021) verileri kullaniyoruz
NDVI = 0.4062
NDMI = -0.1605
LC = 5  # Calilik (Shrub & Scrub)

# Katsayilar
INTERCEPT = -0.5907
NDVI_COEF = -1.3428
NDMI_COEF = -1.3042
LC_COEFS = {
    0: -2.7629, 1: 2.0896, 2: -0.0382, 3: -0.3532,
    4: 1.1267, 5: 1.6451, 6: -0.1594, 7: -0.2806, 8: -1.2702,
}

def sigmoid(x):
    if x > 500: return 1.0
    if x < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-x))

ham_skor = INTERCEPT + (NDVI_COEF * NDVI) + (NDMI_COEF * NDMI) + LC_COEFS[LC]
yaniclik_yuzdesi = sigmoid(ham_skor)  # 0-1 arasi

print("=" * 75)
print(" DUAL BRANCH MODEL + YANICLIK SKORU ENTEGRASYON TESTI")
print("=" * 75)
print(f" Konum: Kaibab Ormani, Kuzey Arizona (35.879N, -111.856W)")
print(f" Cevre Verileri: NDVI={NDVI}, NDMI={NDMI}, Arazi=Calilik (LC_{LC})")
print(f" Yaniclik Skoru: %{yaniclik_yuzdesi*100:.1f}")
print("=" * 75)

# --- 2) DUAL BRANCH MODEL CIKTILARI (ilk 45 saniye) ---
df = pd.read_csv(os.path.join(SCRIPT_DIR, 'outputs', 'video_predictions_scored.csv'))
ilk45 = df[df['frame_idx'] <= 1350].copy()

prob_col = 'decision_prob' if 'decision_prob' in df.columns else 'prob_fire'

# --- 3) ENTEGRASYON FORMULU ---
# Final = (0.95 * Kamera) + (0.05 * Yaniclik)
KAMERA_AGIRLIK = 0.95
BITKI_AGIRLIK = 0.05

print(f"\n Agirliklar: Kamera=%{KAMERA_AGIRLIK*100:.0f} | Bitki Modeli=%{BITKI_AGIRLIK*100:.0f}")
print(f" Formula: Final = ({KAMERA_AGIRLIK} x Kamera) + ({BITKI_AGIRLIK} x Yaniclik)")

# --- 4) SONUC TABLOSU ---
print(f"\n{'Frame':>7} | {'Kamera':>8} | {'Yaniclik':>8} | {'Final':>8} | {'Fark':>6} | {'Karar':>8} | {'Etki':>20}")
print("-" * 85)

toplam_fark = 0
yangin_kamera = 0
yangin_final = 0
etki_pozitif = 0
etki_negatif = 0

for _, r in ilk45.iterrows():
    kamera = r[prob_col]
    
    # Entegre skor
    final = (KAMERA_AGIRLIK * kamera) + (BITKI_AGIRLIK * yaniclik_yuzdesi)
    
    fark = (final - kamera) * 100  # yuzde puan fark
    toplam_fark += abs(fark)
    
    # Karar (esik = 0.65)
    esik = 0.65
    kamera_karar = "YANGIN" if kamera >= esik else "-"
    final_karar = "YANGIN" if final >= esik else "-"
    
    if kamera >= esik:
        yangin_kamera += 1
    if final >= esik:
        yangin_final += 1
    
    # Etki analizi
    if final_karar != kamera_karar and final >= esik:
        etki = ">> YANGIN OLDU"
        etki_pozitif += 1
    elif final_karar != kamera_karar and final < esik:
        etki = "<< YANGIN SILINDI"
        etki_negatif += 1
    elif fark > 0.1:
        etki = "^ Risk artti"
    elif fark < -0.1:
        etki = "v Risk azaldi"
    else:
        etki = "~ Degisim yok"
    
    print(f"{int(r['frame_idx']):>7} | {kamera*100:>7.1f}% | {yaniclik_yuzdesi*100:>7.1f}% | {final*100:>7.1f}% | {fark:>+5.1f}% | {final_karar:>8} | {etki}")

# --- 5) OZET ---
print("\n" + "=" * 75)
print(" KARSILASTIRMA OZETI")
print("=" * 75)
print(f" Toplam frame (ilk 45 sn):     {len(ilk45)}")
print(f" Kamera 'YANGIN' dedigi:        {yangin_kamera} frame")
print(f" Entegre 'YANGIN' dedigi:       {yangin_final} frame")
print(f" Fark:                          {yangin_final - yangin_kamera:+d} frame")
print(f"")
print(f" Yaniclik skoru YANGINA ceviren: {etki_pozitif} frame (False Negative azaldi)")
print(f" Yaniclik skoru SILEN:           {etki_negatif} frame (False Positive azaldi)")
print(f" Ortalama etki:                  {toplam_fark/len(ilk45):.2f} yuzde puan")
print(f"")
print(f" Yaniclik Skoru: %{yaniclik_yuzdesi*100:.1f} (Calilik + Kuru)")
print(f" Bu alan YUKSEK yanici madde iceriyor -> Model destegi POZITIF yonde")
print("=" * 75)
