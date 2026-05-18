# Nasıl Çalıştırılır?

Güncel aktif akış için önce `README.md` dosyasındaki conda ortamı ve aktif/legacy ayrımını takip edin. Bu dosyada eski demo komutları da bulunabilir; ana kaynak `src/`, `scripts/`, `tests/` ve kök `requirements*.txt` dosyalarıdır.

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
python src/01_build_master_index.py
```

Çıktı:
- `outputs/flame_index.csv` (legacy)
- `data/master_index.parquet` (önerilen)

---

## 3. Eğitim

Not:
- “Sıfırdan” çalıştırmak istiyorsanız, bu adım sonunda `models/*.pt` oluşur ve UI’dan video inference çalıştırabilirsiniz.
- `--epochs 1` sadece smoke test içindir; **gerçeğe yakın metrikler için 20+ epoch** önerilir.

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

## 6. Web arayüzü (Streamlit)

Tarayıcıda video çıktısını (risk/event) incelemek ve model metriklerini (eğitim/test) görmek için:

```powershell
streamlit run src/07_ui.py
```

Tarayıcıda açılan adresi (terminalde yazar) kullanın.

Streamlit içindeki akış (sade arayüz):
- **Hızlı Test**: video yükle → tahmin → "Yangın tespit edildi/edilmedi" verdict kartı + 4 özet metric (max güven, fire frame sayısı, event sayısı, ilk tespit zamanı) + olasılık/risk timeline.
  - Detaylı tablo, fire frame listesi ve manuel frame tarayıcı **"📊 Detaylı analiz"** açılır panelinin içindedir.
  - Yüklenen video diagnostics, çıktı dosya yolları ve ham JSON **"🛠️ Geliştirici / Debug bilgileri"** panelinin içindedir.
- **İnceleme (CSV)**: önceden üretilmiş `outputs/video_predictions_scored.csv` üzerinden filtre/sıralama/sayfalama.
- **Model Metrikleri**: `outputs/metrics_*.json` özet tablosu + ham JSON.
- **Video Eval (batch)**: `outputs/eval_summary.csv` görüntüleme.

> Noise / robustness testleri ayrı bir CLI modülündedir (`src/eval/robustness_eval.py`) ve UI'a karışmaz; aşağıdaki "Robustness değerlendirmesi" bölümüne bakın.

---

## 7. Veri sızıntısı denetimi

Yeni bir master index oluşturduktan sonra train/val/test arasında çakışma olup olmadığını kontrol edin:

```powershell
python scripts/check_leakage.py
```

Tüm bölme key'leri (`path_rgb`, `path_th`, `key`, `split_group`, video stem) üzerinden çapraz kontrol yapar; bir uyarı/uyumsuzluk varsa terminalde belirtir.

---

## 8. Robustness değerlendirmesi (offline, ayrı modül)

Eğitilmiş bir checkpoint'in giriş bozulmaları altındaki davranışını ölçmek için:

```powershell
python -m src.eval.robustness_eval `
  --ckpt models/dual_branch.pt `
  --csv data/master_index.parquet `
  --split test `
  --corruptions all `
  --out outputs/robustness_eval.csv
```

Varsayılan `--severities` değeri **1**’dir (ana raporlama ile uyumlu). Tam stres grid’i için `--severities 1,2,3` ekleyin.

Çıktı: `outputs/robustness_eval.csv` — her (corruption, severity) için `n, acc, bal_acc, precision, recall, F1, specificity, FPR, AUC, AP`.
"clean" satırı standart val/test ölçümüne eşittir, kontrol noktası olarak kullanılır.

Sadece bir korruption türü:

```powershell
python -m src.eval.robustness_eval --ckpt models/dual_branch.pt --csv data/master_index.parquet `
  --split test --corruptions gauss_noise_thermal --severities 1,2,3 `
  --out outputs/robustness_thermal_noise.csv
```

> Bu modül model'in sadece çıkarım çıkışını ölçer; eğitim verisini, dataset.py augmentation'larını veya threshold seçimini etkilemez. Gerçek inference akışı (UI / `src/05_video_infer.py` / `src/inference/video.py`) bu modüle bağlı değildir.

---

## Özet sıra

1. `python src/01_build_master_index.py`
2. `python src/02_train.py --mode all --epochs 20`
3. `streamlit run src/07_ui.py` (UI’dan video yükleyip “Hızlı Test” ile çalıştır)

Alternatif (CLI ile):
3. `python src/05_video_infer.py --rgb_video video.mp4`
4. `python src/06_add_risk_score.py`
5. `streamlit run src/07_ui.py`

---

## Performans için önerilen presetler

### A) Hız öncelikli (saha tarama)

```powershell
python src/05_video_infer.py --rgb_video "video.mp4" --step 8 --size 224 --fp16
```

Notlar:
- `--fp16` GPU varsa hızlandırır.
- CAM üretimi yoksa en hızlı çalışır.
- `--size 224` daha hızlı, doğruluk biraz düşebilir.

### B) Denge (önerilen günlük kullanım)

```powershell
python src/05_video_infer.py --rgb_video "video.mp4" --step 5 --adaptive-step --smooth_win 7 --ema_alpha 0.3 --tta --fp16
```

### C) Doğruluk/stabilite öncelikli

```powershell
python src/05_video_infer.py --rgb_video "video.mp4" --step 3 --adaptive-step --size 384 --smooth_win 9 --ema_alpha 0.35 --tta
```

### Fusion (RGB + termal)

```powershell
python src/05_video_infer.py --rgb_video "rgb.mp4" --th_video "termal.mp4" --mode fusion --smooth_win 7 --ema_alpha 0.3 --tta --fp16
```

---

## Tek seferde hızlı başlangıç

```powershell
cd C:\Users\Vıctus\Desktop\bitirme\flame_fire_project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/01_build_master_index.py
python src/02_train.py --mode all --epochs 20
python src/05_video_infer.py --rgb_video "video.mp4" --step 5 --smooth_win 7 --ema_alpha 0.3 --tta --fp16
python src/06_add_risk_score.py
streamlit run src/07_ui.py
```

---

## Ablasyonlar

Standart karşılaştırma eğitimleri için:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\ablations.ps1 -Ablation all -Epochs 20
```

Tek ablation çalıştırmak için:

```powershell
# RGB baseline
powershell -ExecutionPolicy Bypass -File scripts\ablations.ps1 -Ablation rgb

# Thermal baseline
powershell -ExecutionPolicy Bypass -File scripts\ablations.ps1 -Ablation thermal

# Early fusion (tek 4 kanallı encoder)
powershell -ExecutionPolicy Bypass -File scripts\ablations.ps1 -Ablation early_fusion

# Dual-branch fusion (varsayılan fusion yolu)
powershell -ExecutionPolicy Bypass -File scripts\ablations.ps1 -Ablation dual_branch

# Hard negative retrain (dual_branch_fusion -> val FP'lerini yeniden beslet)
powershell -ExecutionPolicy Bypass -File scripts\ablations.ps1 -Ablation hard_neg_retrain
```

Not: `--mode fusion` artık varsayılan olarak `dual_branch_fusion` model ailesini kullanır
(`rgb` ve `thermal` için ayrı backbone, son feature'lar concat edilip classifier'a gidiyor).
Eski davranışı istiyorsanız `--model_family early_fusion` ile açıkça belirtin.

## Benchmark ve Test

Benchmark JSON üretmek için:

```powershell
python src/05_video_infer.py --rgb_video "video.mp4" --step 5 --adaptive-step --fp16 --benchmark
```

Çıktı: `outputs/video_predictions.benchmark.json`

Hızlı test çalıştırma:

```powershell
pytest -q
```
