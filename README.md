# Early Fire Detection and Mapping

RGB + termal görüntü füzyonu kullanan yangın/no-fire sınıflandırıcısı, video çıkarımı ve Streamlit tabanlı inceleme arayüzü.

## Bileşenler

- **Eğitim:** `src/02_train.py` → `src/training/trainer.py`. Dual-branch, gated / attention / mid fusion varyantları; `train_zscore` termal normalize; recall–FPR seçim metrikleri; experiment CSV loglama (`--experiment_log_csv`, `--experiment_name`).
- **Çıkarım:** `src/05_video_infer.py` (CLI) ve `src/inference/video.py` — RTSP/HTTP ve yerel dosya (`src/inference/capture_utils.py`); uzun videoda otomatik frame adımı; EMA + hareketli ortalama karışımı ve ardışık‑kare burst bayrağı.
- **Risk skoru:** `src/06_add_risk_score.py` (CLI) ve `src/risk/scoring.py` — pred CSV’ye zamansal/uzamsal risk.
- **Olay çıkarma:** `src/eval/event_extractor.py` — alarm sürekliliğinden event listesi.
- **Web arayüzü:** `src/07_ui.py` — Streamlit; dosya yükleme veya path/URI; doğru fusion/RGB checkpoint eşlemesi; temporal preset argümanları (blended MA/EMA, burst vb.).
- **Ablation:** `src/eval/ablation_eval.py` — fusion checkpoint üzerinde RGB/thermal sıfırlama ve koşullu metrikleri CSV’ye yazar (`outputs/ablation_suite.csv` için runner ile uyumludur).
- **Robustness:** `src/eval/robustness_eval.py` — corrupted input ile offline değerlendirme (**üretim akışına girmez**).
- **Leakage kontrol:** `scripts/check_leakage.py` — indeks bölmeleri üzerinden sızıntı denetimi.
- **Öncelikli deney süiti:** `scripts/run_priority_experiment_suite.py` — sıralı gated/attention/mid fusion eğitimleri, robustness/ablation + teşhis çıktıları.
- **Kaggle tam süit:** `scripts/run_kaggle_full_suite.py` — grid eğitim, tamamlanan `experiment_name` atlama, `logs/failed_runs.csv`, arşivlenmiş checkpoint (`models/by_experiment/`), otomatik `select_best`.
- **En iyi model:** `scripts/select_best_and_report.py` — `improve_results.csv` üzerinden seçim ve rapor (`suite_audit` satırları hariç); varsa `models/by_experiment/{slug}.pt` kopyalanır.

Notebook hücre akışı (Kaggle elle çalıştırma): `scripts/kaggle_notebook_cells_tr.md`.

## Hızlı başlangıç (yerel)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

python src/01_build_master_index.py
python scripts/check_leakage.py

python src/02_train.py --mode fusion --model_family dual_branch_fusion --epochs 25 --backbone resnet50

streamlit run src/07_ui.py
```

Gelişmiş kullanım ve preset’ler: [`NASIL_CALISTIRILIR.md`](NASIL_CALISTIRILIR.md).

## Konfigürasyon ve Kaggle kurulum

`config.py` yolları ve eğitim varsayılanlarını toplar. **Sadece okunabilir dataset** bağlandığında çıktılar ve checkpoint’lar `/kaggle/working` altında tutulmalıdır:

| Ortam değişkeni | Açıklama |
|-----------------|----------|
| `FLAME_DATA_ROOT` | Ham veri kökü (isteğe bağlı) |
| `FLAME_OUTPUTS_DIR` | Örn. `/kaggle/working/outputs` |
| `FLAME_MODELS_DIR` | Örn. `/kaggle/working/models` |
| `FLAME_MASTER_INDEX` | Örn. `/kaggle/working/data/master_index.parquet` |
| `FLAME_BINARY_ROOT` | Binary dataset kökü (isteğe bağlı) |

Tipik sıra:

1. Repoyu `/kaggle/working/code` altına klonlayın; `os.chdir("/kaggle/working/code")` veya terminalde cd.
2. Yukarıdaki `FLAME_*` değişkenlerini ayarlayın (Setup hücreleri için `scripts/kaggle_notebook_cells_tr.md`).
3. Grid eğitim + eval için:

```bash
cd /kaggle/working/code
python scripts/run_kaggle_full_suite.py \
  --code-root /kaggle/working/code \
  --working-root /kaggle/working \
  --master-index /kaggle/working/data/master_index.parquet \
  --improve-csv /kaggle/working/outputs/improve_results.csv \
  --epochs 25 --patience 5 --bs 8 --lr 2e-5
```

- Tamamlanan `experiment_name` tekrar **çalıştırılmaz** (`improve_results.csv` içinde geçerli `test_recall` varsa atlanır).
- Hatalı adımlar **`/kaggle/working/logs/failed_runs.csv`** içine yazılır; sıra bir sonraki deneyle devam eder.
- Her dual‑branch eğitiminden sonra **`models/by_experiment/{slug}.pt`** arşibi yazılır (üzerine yazılan `dual_branch.pt` ile metrik uyumu için).
- Robustness/ablation çıktıları `outputs/kaggle_eval_archive/` altında saklanır; son çalıştırma **`outputs/robustness_eval.csv`** ve **`outputs/ablation_suite.csv`** ile güncellenir.

Öncelikli mini süit dry-run:

```bash
python scripts/run_priority_experiment_suite.py --dry_run \
  --experiment_log_csv /kaggle/working/outputs/improve_results.csv \
  --csv /kaggle/working/data/master_index.parquet
```

## Eğitim

`src/02_train.py` ana bayrakları:

| Bayrak | Açıklama |
|--------|-----------|
| `--model_family` | `early_fusion`, `dual_branch_fusion`, `dual_branch_gated_fusion`, `dual_branch_attention_fusion`, `dual_branch_mid_fusion` |
| `--selection_metric` | `f1_balacc`, `realistic`, `recall_fpr` (recall≥0.98 sonra min FPR) |
| `--thermal_norm` | `percentile`, `minmax`, `uint16_div`, `train_zscore` |
| `--modal_dropout_p` | Füzyon modalite dropout olasılığı |
| `--thermal_lr_mult`, `--freeze_rgb_epochs` | Dual-branch ısıtma politikası |
| `--experiment_log_csv` / `--experiment_name` | `outputs/improve_results.csv` satır günlüğü |

## Video çıkarımı

```powershell
python src/05_video_infer.py --rgb_video path\to\rgb.mp4 --th_video path\to\thermal.mp4 `
  --prob-temporal-blend 0.25 --burst-min-frames 3 `
  --auto-step-long-video
```

- Yerel dosya, `http(s)://`, `rtsp://` URI’leri `capture_utils.open_video_capture` ile açılabilir (`--stream-buffer-reduce/--no-stream-buffer-reduce`).
- Çıktı CSV: `prob_fire_ma`, `prob_fire_ema`, `pred_fire_burst_consec`, `burst_run_len` dahil zaman istikrarı için alanlar.

## Streamlit arayüzü

```powershell
streamlit run src/07_ui.py
```

Checkpoint seçimi dropdown’dan yapılır; thermal yoksa RGB checkpoint’ına düşülür; path/URI metin kutuları büyük video yükü için alternatiftir.

## En iyi model seçimi

```powershell
python scripts/select_best_and_report.py ^
  --results_csv outputs/improve_results.csv ^
  --out_ckpt models/best_model.pt ^
  --out_md outputs/best_model_report.md
```

Kaggle’da yolları `/kaggle/working/outputs/...` ve `/kaggle/working/models/...` ile değiştirin. Süit sonunda `run_kaggle_full_suite.py` aynı adımı otomatik uygular ( `--no-select-best` ile kapatabilirsiniz).

## Eğitim akışında dikkat notları

- **Sınıf dengesi:** `--loss_mode balanced_sampler` + `WeightedRandomSampler`; `cb_focal` vb.
- **Bölme & sızıntı:** `split_group`; `flame_video_nofire` pair politikası README’deki özetle uyumlu. İndeks değişiminden sonra `scripts/check_leakage.py`.
- **Source-aware eşik:** Trainer val/test için kaynak bazlı eşik taraması ve JSON/checkpoint içi meta.
- **Augmentation:** Yalnızca **train** loader’da (RGB jitter/blur/erase; termal fotoğrafik + random patch). **Train’de** termal tensöre **Gaussian additive noise uygulanmaz** (temiz öğrenme yüzeyi).
- **Gaussian gürültü — sadece offline test:** `robustness_eval.py` (RGB/thermal Gaussian sweep) ve `ablation_eval.py` içindeki `*_gauss_noise` koşulları **val/test verisine forward sırasında** eklenir; trainer DataLoader’ına karışmaz.

## Robustness CLI (offline)

```powershell
python -m src.eval.robustness_eval `
  --ckpt models/dual_branch.pt `
  --csv data/master_index.parquet `
  --split test `
  --out outputs/robustness_eval.csv
```

## Ablation CLI (offline)

```powershell
python -m src.eval.ablation_eval `
  --ckpt models/dual_branch.pt `
  --csv data/master_index.parquet `
  --split test `
  --out outputs/ablation_suite.csv
```

## Test

```powershell
pytest -q
```
