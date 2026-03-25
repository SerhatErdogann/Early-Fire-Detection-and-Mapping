# Hangi Kodda Ne Yapıyor + En İyi Kullanım

Bu dosyada her script’in görevi, **termal var/yokken ne kullanılacağı** ve **eşik önerisi** tek yerde toplanıyor.

---

## 1) Hangi kodda ne yapıyor?

| Dosya | Ne yapıyor |
|-------|------------|
| **01_build_index.py** | `data/flame3/` altındaki RGB ve Thermal görselleri eşleştirir (Fire, No Fire, extra). Çıktı: `outputs/flame_index.csv` (path_rgb, path_th, label, source). Eğitim ve test bu CSV’ye göre yapılır. |
| **02_train.py** | FLAME index’i okuyup **üç model** eğitir: **RGB** (3 kanal), **Thermal** (1 kanal), **Fusion** (4 kanal, RGB+Thermal). ResNet18, ImageNet ön-eğitim, Focal Loss + class weight, balanced sampler, val setinde en iyi F1 eşiğini bulup checkpoint’e yazar. Çıktı: `models/rgb.pt`, `models/thermal.pt`, `models/fusion.pt`. |
| **05_video_infer.py** | RGB (ve isteğe bağlı termal) video okuyup frame frame yangın olasılığı üretir. Grad-CAM ile heatmap çıkarır. Opsiyonel: `--override_thr`, `--smooth_win`, `--mode`. Çıktı: `outputs/video_predictions.csv`, istenirse `outputs/heatmaps/`. |
| **06_add_risk_score.py** | `video_predictions.csv`’e ağırlıklı risk skoru ekler (prob_fire + intensity_top10 + area_heat_gt_0_6). Ardışık yüksek olasılık sayısına göre `fire_event` (yangın olayı) bayrağı üretir. Çıktı: `outputs/video_predictions_scored.csv`. |
| **06b_top_frames.py** | Risk skoruna göre en yüksek K frame’i seçip ayrı CSV’ye yazar. Çıktı: `outputs/top_frames.csv`. |
| **07_ui.py** | Streamlit ile `video_predictions_scored.csv`’i açar; filtre/sıralama, frame bilgisi, heatmap gösterimi ve **manuel etiket** (yes_fire / no_fire) kaydı yapar. Kayıtlar `outputs/manual_review.csv`’e yazılır. |
| **08_calibrate_threshold.py** | `manual_review.csv`’deki yes_fire / no_fire etiketlerine göre en iyi **prob_fire** (ve isteğe bağlı risk_score) eşiğini F1’e göre hesaplar. Çıktı: konsol + `outputs/calibration.json`. |

---

## 2) Termal yoksa / termal varsa: Fusion mu, RGB mi?

- **Termal video yoksa**  
  Script zaten **sadece RGB** kullanır (`models/rgb.pt`). Fusion’a geçemez çünkü fusion 4 kanal (RGB+Thermal) ister.

- **Termal video varsa**  
  Varsayılan **fusion** kullanılır (RGB+Thermal birlikte, `models/fusion.pt`).  
  Ama senin gibi **videoda yangın yok, sadece sıcak arazi/güneş** varsa fusion çok false positive verebiliyor; termal her “sıcak” bölgeyi tetikleyebiliyor.

**Öneri:**

| Senaryo | Önerilen mod | Nasıl çalıştırılır |
|--------|---------------|---------------------|
| Termal video **yok** | RGB | `--rgb_video "..."` (--th_video verme) |
| Termal video **var**, sahne FLAME’e benzer / kontrollü | Fusion | `--rgb_video "..." --th_video "..."` (varsayılan) |
| Termal video **var**, fakat çok false positive (sıcak arazi, güneş) | **RGB zorla** | `--rgb_video "..." --th_video "..." --mode rgb` |

Yani: **Termal yoksa zaten RGB; termal varken de false positive çoksa `--mode rgb` ile RGB kullan.** Fusion’u sadece “gerçekten yangın + termal veri güvenilir” durumunda kullan.

---

## 3) Eşik (threshold) nasıl verilmeli?

- **Checkpoint’teki eşik**  
  Eğitimde val setinde F1’i en iyi yapan eşik kaydediliyor (`models/*.pt` içinde `threshold`). Hiçbir şey vermezsen inference bu eşiği kullanır.

- **Kendi videonda çok yanlış alarm varsa**  
  1) UI’da bir miktar frame’i **no_fire / yes_fire** etiketle.  
  2) `python src/08_calibrate_threshold.py` çalıştır.  
  3) Konsoldaki **Önerilen threshold (prob_fire)** değerini al (örn. 0.55).  
  4) Sonraki inference’ta: `--override_thr 0.55` kullan.

- **Pratik başlangıç değeri (kalibrasyon yapmadan)**  
  Videoda “neredeyse hiç yangın yok” biliyorsan, yanlış alarmı azaltmak için **0.6–0.7** deneyebilirsin:
  ```bash
  --override_thr 0.65
  ```
  Kalibrasyon yaptıktan sonra çıkan eşik daha doğru olur.

Özet: **Varsayılan = checkpoint eşiği. Yanlış alarm çoksa kalibrasyon yap veya geçici olarak 0.6–0.7 kullan.**

---

## 4) Eğitim seçenekleri (02_train.py)

- **Extra test seti:** Extra verisinin **%20'si** (varsayılan) GroupShuffleSplit ile **drone no-fire test** olarak ayrılır; her epoch sonunda **FP sayısı ve FP_rate** raporlanır. Gerçek dünya false positive'ı böyle izlenir.
- **Thermal uzantıya göre:** `.tif`/`.tiff` = radiometric (Celsius) → percentile 0..1. `.jpg`/`.png`/`.bmp` = görselleştirilmiş (viz8) → doğrudan /255. Böylece JPG thermal ile TIFF thermal aynı normalize edilmez; yanlış ipuçları azalır.
- **Loss/sampler:** `--loss_mode sampler_ce` (Sampler + CE), `focal_shuffle` (shuffle + Focal), `sampler_focal` (varsayılan). Overconfident olursa `sampler_ce` veya `focal_shuffle` dene.
- **Calibration:** Val logits ile **temperature scaling** yapılır; T checkpoint'e yazılır. Inference'da `prob = softmax(logits/T)` kullanılır; daha gerçekçi olasılıklar.

## 5) En iyi model / son adım mantığı

- **En iyi eğitim:** ResNet18, Focal veya CE+sampler, extra train/test bölünmesi, thermal source-based norm, temperature scaling, early stopping, LR schedule. Ek veri (drone no-fire frameleri) eklersen daha da iyileşir.
- **En son adımın mantığı:** Video → inference (RGB veya fusion, gerekirse `--mode rgb`) → risk skoru + fire_event → UI’da inceleme ve manuel etiket → kalibrasyon → aynı videoda veya yeni videoda `--override_thr` ile tekrar çalıştırma. Bu döngü hem eşiği doğrultuyor hem de ileride ek veri olarak kullanılabilir.
- **Termal yoksa fusion mu en iyi?** Hayır; termal yoksa **sadece RGB** kullanılır ve bu doğru davranış. Termal varken de false positive çoksa **fusion yerine RGB (`--mode rgb`)** kullanmak genelde daha mantıklı sonuç verir.

Bu dokümandaki öneriler RUN.md’deki komutlarla birlikte kullanılabilir; eşik ve mod seçimi için tek referans bu dosya olabilir.
