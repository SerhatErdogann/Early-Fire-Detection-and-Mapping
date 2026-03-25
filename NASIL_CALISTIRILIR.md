# Nasıl Çalıştırılır?

Tüm komutları **proje klasörünün içinden** (flame_fire_project) çalıştırın.

---

## 1. Kurulum (bir kez)

```powershell
cd C:\Users\Vıctus\Desktop\bitirme\flame_fire_project

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

**Veri:** FLAME verisini `data/flame3/` altına koyun:
- `data/flame3/Fire/RGB/Corrected FOV/` ve `Fire/Thermal/Celsius TIFF/`
- `data/flame3/No Fire/RGB/Corrected FOV/` ve `No Fire/Thermal/Celsius TIFF/`
- İsteğe bağlı: `data/flame3/extra/RGB/` ve `extra/Thermal/`

---

## 2. Index oluştur (veri listesi)

```powershell
python src/01_build_index.py
```

Çıktı: `outputs/flame_index.csv`

---

## 3. Eğitim

**Hepsini eğit (rgb + thermal + fusion):**
```powershell
python src/02_train.py --mode all --epochs 20
```

**Sadece fusion:**
```powershell
python src/02_train.py --mode fusion --epochs 20
```

**Daha güçlü model (ResNet50):**
```powershell
python src/02_train.py --mode fusion --epochs 25 --backbone resnet50
```

Modeller `models/` klasörüne kaydedilir: `rgb.pt`, `thermal.pt`, `fusion.pt`

---

## 4. Video üzerinde tahmin

**Sadece RGB video:**
```powershell
python src/05_video_infer.py --rgb_video "videonun_yolu.mp4"
```

**RGB + termal (fusion):**
```powershell
python src/05_video_infer.py --rgb_video rgb.mp4 --th_video termal.mp4 --mode fusion
```

**Daha stabil sonuç (drone videoları için önerilen):**
```powershell
python src/05_video_infer.py --rgb_video video.mp4 --smooth_win 7 --ema_alpha 0.3 --tta
```

Çıktı: `outputs/video_predictions.csv`

---

## 5. Risk skoru ekle

Önce 4. adımı çalıştırıp `outputs/video_predictions.csv` oluşturmalısınız.

```powershell
python src/06_add_risk_score.py
```

Çıktı: `outputs/video_predictions_scored.csv`

---

## 6. Görsel demo (Gradio)

Tarayıcıda RGB + termal görsel yükleyip test etmek için:

```powershell
python src/03_app.py
```

Tarayıcıda açılan adresi (örn. http://127.0.0.1:7860) kullanın.

---

## Özet sıra

1. `python src/01_build_index.py`
2. `python src/02_train.py --mode all --epochs 20`
3. `python src/05_video_infer.py --rgb_video video.mp4`
4. `python src/06_add_risk_score.py`
5. (İsteğe bağlı) `python src/03_app.py`
