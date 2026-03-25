# 🔥 Proje Çalıştırma Adımları

Proje kök dizininde (`flame_fire_project`) çalıştır. Gerekirse sanal ortamı aktifleştir:  
`(.venv) PS C:\...\flame_fire_project>`

**Hangi kodda ne yapıyor, termal yoksa fusion mu RGB mi, eşik nasıl verilmeli:** Bunların hepsi **DOC.md** içinde açıklanıyor.

---

## 1) Veri hazırlığı

FLAME3 verisini şu yapıda yerleştir:

```
data/
  flame3/
    Fire/
      RGB/Corrected FOV/   (.JPG)
      Thermal/Celsius TIFF/ (.TIFF)
    No Fire/
      RGB/Corrected FOV/
      Thermal/Celsius TIFF/
    extra/                  (opsiyonel, no-fire ek veri)
      RGB/
      Thermal/
```

Index oluştur (RGB–Thermal eşleşmeleri `outputs/flame_index.csv`'e yazılır):

```bash
python src/01_build_index.py
```

Çıktı: `outputs/flame_index.csv`  
Kontrol: Konsolda toplam örnek ve label/source dağılımı görünür.

---

## 2) Model eğitimi

Index hazırsa eğitimi başlat (varsayılan: RGB, Thermal, Fusion hepsi):

```bash
python src/02_train.py
```

**İsteğe bağlı argümanlar:**

| Argüman | Açıklama | Varsayılan |
|---------|----------|------------|
| `--mode` | `rgb` \| `thermal` \| `fusion` \| `all` | `all` |
| `--loss_mode` | `sampler_ce` \| `focal_shuffle` \| `sampler_focal` | `sampler_focal` |
| `--epochs` | Epoch sayısı | 15 |
| `--patience` | Early stopping (val AP iyileşmezse) | 3 |
| `--extra_test_ratio` | Extra verisinin test oranı (drone no-fire) | 0.2 |
| `--bs` | Batch size | 16 |
| `--lr` | Öğrenme oranı | 1e-4 |

Örnek (sadece fusion, CE+sampler, 20 epoch):

```bash
python src/02_train.py --mode fusion --loss_mode sampler_ce --epochs 20
```

Çıktılar:
- `models/rgb.pt`, `models/thermal.pt`, `models/fusion.pt` (içinde `threshold` + `temperature`)

Her mod için val/test metrikleri, **extra_test FP_rate** ve kaydedilen en iyi checkpoint konsola yazılır.

---

## 3) Video üzerinde tahmin (inference)

Videolar `data/flame3/videos/` içinde (örn. `pair1_rgb.MP4`, `pair1_ir.MP4`). Diğer pair’ler için `pair2_rgb.MP4` / `pair2_ir.MP4` vb. kullan.

**Sadece RGB video:**

```bash
python src/05_video_infer.py --rgb_video "data/flame3/videos/pair1_rgb.MP4" --save_heatmaps
```

**RGB + Termal video (fusion model) – pair1:**

```bash
python src/05_video_infer.py --rgb_video "data/flame3/videos/pair1_rgb.MP4" --th_video "data/flame3/videos/pair1_ir.MP4" --save_heatmaps
```

**Termal var ama çok yanlış alarm veriyorsa → sadece RGB kullan (fusion yerine):**

```bash
python src/05_video_infer.py --rgb_video "data/flame3/videos/pair1_rgb.MP4" --th_video "data/flame3/videos/pair1_ir.MP4" --mode rgb --save_heatmaps
```

**İsteğe bağlı parametreler:**

| Parametre        | Açıklama                          | Örnek        |
|------------------|-----------------------------------|--------------|
| `--step`         | Kaç frame'de bir işlenecek        | `--step 3`   |
| `--smooth_win`   | Olasılık yumuşatma penceresi     | `--smooth_win 5` |
| `--override_thr`| Eşik (checkpoint yerine kullan)  | `--override_thr 0.65` |
| `--mode`        | Zorla mod: auto / rgb / thermal / fusion | `--mode rgb` (termal varken bile RGB) |
| `--out`          | Çıktı CSV yolu                   | `--out outputs/my_predictions.csv` |

Örnek (pair1 + eşik + yumuşatma). Eşik: kalibrasyon yaptıysan çıkan değer, yoksa videoda yangın yoksa **0.6–0.7** deneyin:

```bash
python src/05_video_infer.py --rgb_video "data/flame3/videos/pair1_rgb.MP4" --th_video "data/flame3/videos/pair1_ir.MP4" --override_thr 0.65 --smooth_win 3 --save_heatmaps
```

Çıktı: `outputs/video_predictions.csv` (ve `--save_heatmaps` ile `outputs/heatmaps/`).

**Keras .h5 model kullanacaksan (sadece RGB):**

TensorFlow yüklü olmalı (`pip install tensorflow`). Termal ve heatmap desteklenmez; çıktı CSV formatı aynıdır, 06 ve 07 aynen kullanılır.

```bash
python src/05_video_infer_h5.py --rgb_video "data/flame3/videos/pair1_rgb.MP4" --model_h5 "path/to/model.h5" --override_thr 0.5 --size 224
```

`--size`: model giriş boyutu (224 veya 384, modelin eğitimindeki gibi).

---

## 4) Risk skoru ve yangın olayı

Video tahminleri CSV'ye yazıldıktan sonra risk skoru ve `fire_event` ekle:

```bash
python src/06_add_risk_score.py
```

Varsayılan giriş: `outputs/video_predictions.csv`  
Çıktı: `outputs/video_predictions_scored.csv`  
İçerik: `risk_score`, `risk_score_norm`, `fire_run_len`, `fire_event` vb.

---

## 5) Sonuçları incelemek (Streamlit UI)

```bash
streamlit run src/07_ui.py
```

Tarayıcıda açılan arayüzde:
- CSV path: `outputs/video_predictions_scored.csv`
- Filtre: min prob, min risk, sıralama
- Frameleri tek tek açıp **Manuel Etiketleme** (yes_fire / no_fire) yapabilirsin; kayıtlar `outputs/manual_review.csv`'e yazılır.

---

## 6) Eşik kalibrasyonu (video bazlı)

Manuel etiketlediğin framelerle en iyi eşiği hesaplat:

```bash
python src/08_calibrate_threshold.py
```

Girdi: `outputs/manual_review.csv` (yes_fire / no_fire etiketli satırlar)  
Çıktı: Konsol + `outputs/calibration.json`  
Konsoldaki **Önerilen threshold (prob_fire)** değerini sonraki videolarda `--override_thr` ile kullan.

---

## 7) Opsiyonel: En riskli K frame

Sadece en yüksek risk skorlu K frame'i ayrı CSV'ye yazmak için:

```bash
python src/06b_top_frames.py
```

Varsayılan: `outputs/video_predictions_scored.csv` → `outputs/top_frames.csv` (K=50).

---

## Hızlı sıra özeti

| Sıra | Ne yapıyorsun           | Komut |
|------|-------------------------|--------|
| 1    | Veri index'i oluştur    | `python src/01_build_index.py` |
| 2    | Modelleri eğit          | `python src/02_train.py` |
| 3    | Videoya tahmin uygula   | `python src/05_video_infer.py --rgb_video ... [--th_video ...] --save_heatmaps` |
| 4    | Risk skoru ekle         | `python src/06_add_risk_score.py` |
| 5    | UI ile incele / etiketle| `streamlit run src/07_ui.py` |
| 6    | Eşik kalibre et         | `python src/08_calibrate_threshold.py` |
| 7    | (İsteğe bağlı) Top frame| `python src/06b_top_frames.py` |

---

## Sık kullanılan tam akış (ilk kurulum + video)

```bash
# 1. Index
python src/01_build_index.py

# 2. Eğitim
python src/02_train.py

# 3. Video (fusion) – pair1
python src/05_video_infer.py --rgb_video "data/flame3/videos/pair1_rgb.MP4" --th_video "data/flame3/videos/pair1_ir.MP4" --save_heatmaps

# 4. Risk skoru
python src/06_add_risk_score.py

# 5. İnceleme
streamlit run src/07_ui.py
```

Kalibrasyon sonrası tekrar video çalıştırmak için:

```bash
python src/05_video_infer.py --rgb_video "data/flame3/videos/pair1_rgb.MP4" --th_video "data/flame3/videos/pair1_ir.MP4" --override_thr 0.37 --smooth_win 3 --save_heatmaps
python src/06_add_risk_score.py
streamlit run src/07_ui.py
```
