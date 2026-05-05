# Early-Fire-Detection-and-Mapping

Bu repo için ana kullanım dokümanı:
- `NASIL_CALISTIRILIR.md` (kurulum, **sıfırdan eğitim**, video inference, eval ve Streamlit UI)

Rapor/tez özeti:
- `RAPOR_PROJE_OZETI.md`

## Hızlı Başlangıç

```powershell
python src/01_build_master_index.py
python src/02_train.py --mode fusion --epochs 25 --backbone resnet50
streamlit run src/07_ui.py
```