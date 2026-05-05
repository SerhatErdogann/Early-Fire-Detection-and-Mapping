# Early Fire Detection and Mapping

RGB + termal görüntü füzyonu kullanan yangın/no-fire sınıflandırıcısı, video çıkarımı ve Streamlit tabanlı inceleme arayüzü.

## Bileşenler

- **Eğitim:** `src/02_train.py` — `src/training/trainer.py` üzerinde tek/çift dallı (dual-branch fusion) sınıflandırıcı eğitir, kalibrasyon ve threshold seçimi yapar, `outputs/metrics_*.json`'a sonuçları yazar.
- **Çıkarım:** `src/05_video_infer.py` (CLI) ve `src/inference/video.py` (modül) — video üzerinden frame-by-frame yangın olasılığı, EMA/TTA/sahne-değişikliği koruması, alarm durum makinesi.
- **Risk skoru:** `src/06_add_risk_score.py` (CLI) ve `src/risk/scoring.py` (modül) — pred CSV'sine zamansal/uzamsal risk özelliklerini ekler.
- **Olay (event) çıkarma:** `src/eval/event_extractor.py` — alarm durumlarından sürekli yangın olaylarını çıkarır.
- **Web arayüzü:** `src/07_ui.py` — Streamlit. Video yükle → çıkarım + risk + olay → timeline & frame paneli.

## Hızlı Başlangıç

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src/01_build_master_index.py
python src/02_train.py --mode fusion --epochs 25 --backbone resnet50
streamlit run src/07_ui.py
```

Detaylı kullanım, preset'ler ve ablation komutları için: [`NASIL_CALISTIRILIR.md`](NASIL_CALISTIRILIR.md)

## Konfigürasyon

`config.py` tüm yolları (data, models, outputs) ve eğitim/çıkarım varsayılanlarını merkezi olarak tutar. Kaggle gibi read-only ortamlar için ortam değişkenleri ile override edilebilir (`FLAME_DATA_ROOT`, `FLAME_OUTPUTS_DIR`, `FLAME_MODELS_DIR`, `FLAME_MASTER_INDEX`, `FLAME_BINARY_ROOT`).

## Test

```powershell
pytest -q
```
