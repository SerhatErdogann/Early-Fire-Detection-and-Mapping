# Early Fire Detection and Mapping

RGB + termal görüntü füzyonu kullanan yangın/no-fire sınıflandırıcısı, video çıkarımı ve Streamlit tabanlı inceleme arayüzü.

## Bileşenler

- **Eğitim:** `src/02_train.py` — `src/training/trainer.py` üzerinde tek/çift dallı (dual-branch fusion) sınıflandırıcı eğitir, kalibrasyon ve threshold seçimi yapar, `outputs/metrics_*.json`'a sonuçları yazar.
- **Çıkarım:** `src/05_video_infer.py` (CLI) ve `src/inference/video.py` (modül) — video üzerinden frame-by-frame yangın olasılığı, EMA/TTA/sahne-değişikliği koruması, alarm durum makinesi.
- **Risk skoru:** `src/06_add_risk_score.py` (CLI) ve `src/risk/scoring.py` (modül) — pred CSV'sine zamansal/uzamsal risk özelliklerini ekler.
- **Olay (event) çıkarma:** `src/eval/event_extractor.py` — alarm durumlarından sürekli yangın olaylarını çıkarır.
- **Web arayüzü:** `src/07_ui.py` — Streamlit. Video yükle → tahmin → "yangın var/yok" verdict kartı + özet metrikler. Detaylı tablolar, ham JSON ve frame paneli "Detaylı analiz" / "Geliştirici" panellerinde gizlidir; debug/test/noise seçenekleri varsayılan olarak kapalıdır.
- **Veri sızıntısı kontrolü:** `scripts/check_leakage.py` — `master_index.parquet` üzerinde train/val/test bölmelerini path, key, split_group ve video stem üzerinden çapraz kontrol eder.
- **Robustness değerlendirmesi:** `src/eval/robustness_eval.py` — eğitilmiş bir checkpoint'i Gaussian noise / brightness-contrast / blur / thermal shift altında değerlendirir. Bu modül **sadece offline değerlendirme içindir**, gerçek çıkarım/UI akışına karışmaz.

## Hızlı Başlangıç

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src/01_build_master_index.py
python scripts/check_leakage.py            # veri sızıntısı denetimi
python src/02_train.py --mode fusion --epochs 25 --backbone resnet50
streamlit run src/07_ui.py
```

Robustness sweep (eğitim sonrası):

```powershell
python -m src.eval.robustness_eval `
  --ckpt models/dual_branch.pt `
  --csv data/master_index.parquet `
  --split test `
  --corruptions all `
  --severities 1,2,3 `
  --out outputs/robustness_eval.csv
```

Detaylı kullanım, preset'ler ve ablation komutları için: [`NASIL_CALISTIRILIR.md`](NASIL_CALISTIRILIR.md)

## Konfigürasyon

`config.py` tüm yolları (data, models, outputs) ve eğitim/çıkarım varsayılanlarını merkezi olarak tutar. Kaggle gibi read-only ortamlar için ortam değişkenleri ile override edilebilir (`FLAME_DATA_ROOT`, `FLAME_OUTPUTS_DIR`, `FLAME_MODELS_DIR`, `FLAME_MASTER_INDEX`, `FLAME_BINARY_ROOT`).

## Eğitim akışında nelere dikkat ediliyor

- **Sınıf dengesizliği:** `--loss_mode balanced_sampler` ile her batch yarı yarıya yangın/no-fire içerir; `WeightedRandomSampler` + source-aware ağırlıklar (`flame_video_nofire` arttırıldı, `cart_aux` azaltıldı) `cb_focal` kaybı ile birleştirilir.
- **Bölme & sızıntı:** Sahne/grup tabanlı bölme (`split_group`); ekstra (drone) no-fire için ayrı holdout. `scripts/check_leakage.py` her index güncellemesinden sonra çalıştırılmalıdır.
- **Eşik politikası (yanlış-negatif öncelikli):** Her epoch'ta validasyon üzerinde eşik taraması yapılır. Aday eşikler içinden **recall'ı en yüksek seviyenin %2 yakınında tutarken FPR'ı minimize eden** nokta seçilir; alarm eşiği `THRESHOLD_ALARM_MIN=0.25` ile alttan kıstırılır.
- **Metrik raporlama:** Her epoch için `acc, bal_acc, precision, recall, F1, AUC, AP, specificity, FPR, ECE, Brier, confusion matrix` ve kaynak (source) bazında dağılımlar yazdırılır + `outputs/metrics_*.json` dosyasına kaydedilir.
- **Checkpoint seçimi:** Varsayılan `--selection_metric f1_balacc` (`0.5*F1 + 0.5*BalAcc`); alternatif `--selection_metric realistic` ise `F1 + BalAcc + AP - 0.5*FPR` kompozit skoru kullanır.

## Test

```powershell
pytest -q
```
