import pandas as pd
import math
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# ILK 45 SANIYE - DUZELTILMIS ENTEGRASYON
# Bitki Skoru %50 = nötr, >%50 = destek (+0 ile +5), <%50 = suphe (-0 ile -5)
# ============================================================

# Bitki Modeli Verileri (FLAME 2 bolge - GEE'den cekildi)
NDVI = 0.4062
NDMI = -0.1605
LC = 5  # Calilik

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
yaniclik = sigmoid(ham_skor)  # 0.672

# ============================================================
# YENI FORMULA:
# Bitki %50 = notr (0 etki)
# Bitki %100 = +5 puan
# Bitki %0 = -5 puan
# Modifiye = ((bitki - 0.50) / 0.50) * 0.05
# Final = Kamera + Modifiye  (0-1 arasinda clamp)
# ============================================================
MAX_ETKI = 0.05  # %5

def entegre_skor(kamera, bitki):
    modifiye = ((bitki - 0.50) / 0.50) * MAX_ETKI
    final = kamera + modifiye
    return max(0.0, min(1.0, final))  # 0-1 arasi tut

modifiye_deger = ((yaniclik - 0.50) / 0.50) * MAX_ETKI

ESIK = 0.65

# Video tahminlerini oku
df = pd.read_csv(os.path.join(SCRIPT_DIR, 'outputs', 'video_predictions_scored.csv'))
ilk45 = df[df['frame_idx'] <= 1350].copy()
prob_col = 'decision_prob' if 'decision_prob' in df.columns else 'prob_fire'

# Sonuc CSV olustur
sonuclar = []

print("=" * 90)
print("  ILK 45 SANIYE - DUZELTILMIS ENTEGRASYON")
print("=" * 90)
print(f"  Bitki Yaniclik Skoru: %{yaniclik*100:.1f}")
print(f"  Referans: %50 = notr")
print(f"  Sabit modifiye: {modifiye_deger*100:+.2f} puan (her frame'e eklenir)")
print(f"  Aciklama: Bitki %{yaniclik*100:.1f} > %50 -> DESTEK verir")
print("=" * 90)
print()
print("{:>6} | {:>10} {:>8} | {:>10} {:>8} | {:>7} {:>8}".format(
    "Frame", "ONCEKI", "Karar", "SONRAKI", "Karar", "Fark", "Degisim"))
print("-" * 90)

yangin_onceki = 0
yangin_sonraki = 0
karar_degisen = 0
degisen_listesi = []

for _, r in ilk45.iterrows():
    frame = int(r['frame_idx'])
    onceki = r[prob_col]
    sonraki = entegre_skor(onceki, yaniclik)
    
    fark = (sonraki - onceki) * 100
    
    karar_onceki = "YANGIN" if onceki >= ESIK else "YOK"
    karar_sonraki = "YANGIN" if sonraki >= ESIK else "YOK"
    
    if karar_onceki == "YANGIN":
        yangin_onceki += 1
    if karar_sonraki == "YANGIN":
        yangin_sonraki += 1
    
    if karar_onceki != karar_sonraki:
        degisim = "DEGISTI!"
        karar_degisen += 1
        degisen_listesi.append((frame, onceki*100, sonraki*100, karar_onceki, karar_sonraki))
    else:
        degisim = "-"
    
    print("{:>6} | {:>9.1f}% {:>8} | {:>9.1f}% {:>8} | {:>+6.2f}% {:>8}".format(
        frame,
        onceki * 100, karar_onceki,
        sonraki * 100, karar_sonraki,
        fark, degisim))
    
    sonuclar.append({
        "frame_idx": frame,
        "saniye": round(frame / 30, 1),
        "onceki_yuzde": round(onceki * 100, 2),
        "onceki_karar": karar_onceki,
        "sonraki_yuzde": round(sonraki * 100, 2),
        "sonraki_karar": karar_sonraki,
        "fark_puan": round(fark, 2),
        "karar_degisti": karar_onceki != karar_sonraki
    })

# Ozet
print()
print("=" * 90)
print("  OZET")
print("=" * 90)
print(f"  Toplam frame:              {len(ilk45)}")
print(f"  ONCEKI (Sadece Kamera):")
print(f"    Yangin dedigi:           {yangin_onceki} frame")
print(f"    Yangin demediyi:         {len(ilk45) - yangin_onceki} frame")
print(f"  SONRAKI (Kamera + Bitki):")
print(f"    Yangin dedigi:           {yangin_sonraki} frame")
print(f"    Yangin demediyi:         {len(ilk45) - yangin_sonraki} frame")
print(f"  KARAR DEGISEN FRAME:       {karar_degisen}")
print(f"")
print(f"  Bitki Yaniclik:            %{yaniclik*100:.1f} (>{50}% -> DESTEK)")
print(f"  Her frame'e eklenen:       {modifiye_deger*100:+.2f} yuzde puan")

if degisen_listesi:
    print(f"\n  KARAR DEGISEN FRAME'LER:")
    for f, o, s, ko, ks in degisen_listesi:
        print(f"    Frame {f}: {o:.1f}% ({ko}) -> {s:.1f}% ({ks})")

print("=" * 90)

# CSV kaydet
out_df = pd.DataFrame(sonuclar)
out_path = os.path.join(SCRIPT_DIR, 'outputs', 'ilk45sn_duzeltilmis.csv')
out_df.to_csv(out_path, index=False)
print(f"\n  CSV kaydedildi: {out_path}")
