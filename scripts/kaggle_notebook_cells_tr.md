# Kaggle manuel Notebook — hücre akışı

Varsayılan: repo `/kaggle/working/code`, yazılabilir çıktılar `/kaggle/working/outputs` ve `/kaggle/working/models`.

Dataset’ini **Add Data** ile bağla; `master_index.parquet` dosyasına aşağıdaki `--master-path` ile işaret et.

---

## Hücre 1 — Setup

```python
import os
import sys
from pathlib import Path

CODE = Path("/kaggle/working/code")
WORK = Path("/kaggle/working")
OUT = WORK / "outputs"
MOD = WORK / "models"
DATA = WORK / "data"
LOG = WORK / "logs"

for d in (CODE, OUT, MOD, DATA, LOG):
    d.mkdir(parents=True, exist_ok=True)

os.chdir(CODE)
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

# config.py yazılabilir kökleri Kaggle’a yönlendirir (subprocess dahil).
os.environ["FLAME_OUTPUTS_DIR"] = str(OUT.resolve())
os.environ["FLAME_MODELS_DIR"] = str(MOD.resolve())
# Aşağıdaki yolu Kendi parquet yoluna göre güncelle (input veya working kopyası):
os.environ["FLAME_MASTER_INDEX"] = "/kaggle/working/data/master_index.parquet"

import torch

print("CWD:", os.getcwd())
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Harici index `input` altında ise Örnek:

```python
MASTER_SRC = Path("/kaggle/input/senin-datassetin/path/master_index.parquet")
DEST = DATA / "master_index.parquet"
if MASTER_SRC.is_file():
    import shutil

    shutil.copy2(MASTER_SRC, DEST)
    os.environ["FLAME_MASTER_INDEX"] = str(DEST)
```

Son kontrol:

```python
master = Path(os.environ["FLAME_MASTER_INDEX"])
assert master.is_file(), f"Eksik index: {master}"
print("master_index:", master, master.stat().st_size // 1024**2, "MB")
```

---

## Hücre 2 — Dry run (`run_priority_experiment_suite`)

```python
!python scripts/run_priority_experiment_suite.py --dry_run \
  --experiment_log_csv /kaggle/working/outputs/improve_results.csv \
  --csv /kaggle/working/data/master_index.parquet
```

`--csv` yolunu Özelleştirin (env `FLAME_MASTER_INDEX` ile aynı mantıkta).

---

## Hücre 3 — Kapsamlı deney süiti (tam runner)

Komutları kontrollü çalıştırır; **tamamlanan** `experiment_name` satırlarını atlar; hataları **durdurmaz**, `logs/failed_runs.csv` yazar.

```python
!python scripts/run_kaggle_full_suite.py \
  --code-root /kaggle/working/code \
  --working-root /kaggle/working \
  --master-index /kaggle/working/data/master_index.parquet \
  --improve-csv /kaggle/working/outputs/improve_results.csv \
  --epochs 25 \
  --patience 5 \
  --bs 8 \
  --lr 2e-5
```

Sadece bir alt küme için:

```python
!python scripts/run_kaggle_full_suite.py ... --only "kaggle_dbf_gated.*"
```

Listeyi görmek:

```python
!python scripts/run_kaggle_full_suite.py --list
```

`eval` veya son `select_best` istemezsen:

```python
# !python scripts/run_kaggle_full_suite.py ... --no-eval
# !python scripts/run_kaggle_full_suite.py ... --no-select-best
```

---

## Hücre 4 — Final seçim (isteğey bağlı tek başına)

Runner zaten `--no-select-best` verilmedikçe sonunda bunu yapar.

```bash
python scripts/select_best_and_report.py \
  --results_csv /kaggle/working/outputs/improve_results.csv \
  --copy_balanced_ckpt /kaggle/working/models/best_model.pt \
  --out_md /kaggle/working/outputs/best_model_report.md
```

---

## Hücre 5 — Son kontroller

```python
from pathlib import Path
import pandas as pd

out = Path("/kaggle/working/outputs")
md = out / "best_model_report.md"
if md.is_file():
    print(md.read_text(encoding="utf-8"))
else:
    print("Önce select_best çalıştırılmalı.")

csv_path = out / "improve_results.csv"
df = pd.read_csv(csv_path)
if "suite_audit" in df.columns:
    df = df[pd.to_numeric(df["suite_audit"], errors="coerce").fillna(0).astype(int) == 0]

cols = [
    "experiment_name",
    "model_family",
    "test_realistic_recall",
    "test_realistic_fpr",
    "test_realistic_f1",
    "val_realistic_f1",
]
cols = [c for c in cols if c in df.columns]

if cols:
    best10 = df.sort_values(
        by=["test_realistic_recall", "test_realistic_fpr", "test_realistic_f1"],
        ascending=[False, True, False],
        na_position="last",
    ).head(10)
    display(best10[cols])

# Son arşivlenen robustness/ablation (global kopyalar)
rob = out / "robustness_eval.csv"
ab = out / "ablation_suite.csv"
for p in (rob, ab):
    if p.is_file():
        d = pd.read_csv(p)
        print("\n===", p.name, "rows:", len(d), "===")
        display(d.head(12))

# Model dosya boyutları
mods = Path("/kaggle/working/models")
for p in sorted(mods.glob("**/*.pt")):
    mb = p.stat().st_size / (1024**2)
    print(f"{p}: {mb:.2f} MB")
```
