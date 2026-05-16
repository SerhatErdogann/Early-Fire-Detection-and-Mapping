#!/usr/bin/env python3
"""Audit model behaviour on sensitive sources (e.g. binary_root no-fire negatives).

Produces:
  - ``false_positive_frames.csv``: label=0, pred=1 rows on the filtered source
  - ``source_fpr_table.csv``: FPR/recall/confusion-style counts per ``source``
  - ``confusion_matrices.json``: sklearn-style confusion matrices per source
  - ``gallery/``: thumbnails for FP frames (PNG)

Requires the same preprocessing as training / rob_eval (checkpoint ``training_args``.
thermal_mu/sigma resolved via ``thermal_calibration.resolve_thermal_calibration_or_exit``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import confusion_matrix  # noqa: E402

from src.data import FlameDataset  # noqa: E402
from src.data.path_filter import filter_df_existing_paths  # noqa: E402
from src.eval.thermal_calibration import (  # noqa: E402
    resolve_thermal_calibration_or_exit,
    thermal_norm_from_checkpoint,
)
from src.models import make_classifier  # noqa: E402
from src.training.metrics import eval_probs  # noqa: E402


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["label"]).copy()
    if "path_th" not in d.columns and "path_thermal" in d.columns:
        d["path_th"] = d["path_thermal"]
    if "label_fire" not in d.columns:
        d["label_fire"] = d["label"].astype(int)
    return d


def _subset_split_sources(df: pd.DataFrame, split: str, sources: list[str]) -> pd.DataFrame:
    d = _prep_df(df)
    split = split.lower().strip()
    if split != "all" and "split" in d.columns:
        d = d[d["split"].astype(str).str.lower() == split]
    src_set = {s.strip().lower() for s in sources if s.strip()}
    if src_set:
        sx = d["source"].astype(str).str.lower()
        d = d[sx.isin(src_set)].copy()
    return d.reset_index(drop=True)


def _load_ck(pt: str, device: str) -> dict:
    try:
        return torch.load(pt, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(pt, map_location=device)


def _save_thumbnail(src_path: str, dest: Path, size: tuple[int, int] = (256, 256)) -> None:
    try:
        im = Image.open(src_path).convert("RGB")
        im.thumbnail(size)
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, format="PNG")
    except Exception:
        return


def run_audit(
    *,
    ckpt_path: Path,
    index_csv: Path,
    out_dir: Path,
    sources: list[str],
    split: str = "test",
    bs: int = 32,
    gallery_max: int = 80,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = _load_ck(str(ckpt_path), device)
    mf = str(ck.get("model_family", "dual_branch_fusion")).lower()
    backbone = str(ck.get("backbone", "resnet50"))
    size = int(ck.get("input_size", 384))
    mode = str(ck.get("mode", "fusion")).lower()
    temperature = float(ck.get("temperature", 1.0))
    thr = float(ck.get("threshold", 0.5))

    tn = thermal_norm_from_checkpoint(ck)
    mu, sigma = resolve_thermal_calibration_or_exit(
        ck=ck,
        thermal_norm=tn,
        cli_mu=None,
        cli_sigma=None,
        metrics_json=None,
        prog="python scripts/run_binary_root_audit.py",
    )

    if index_csv.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(index_csv)
    else:
        df = pd.read_csv(index_csv)

    subset = _subset_split_sources(df, split, sources)
    subset, dropped = filter_df_existing_paths(subset, mode=mode)
    if dropped:
        print(f"[audit] dropped missing paths: {dropped}")
    if len(subset) == 0:
        raise SystemExit("No rows after filter.")

    kw: dict = dict(mode=mode, size=size, train=False, thermal_norm=tn)
    if mu is not None:
        kw["thermal_mu"], kw["thermal_sigma"] = mu, float(sigma)

    ds = FlameDataset(subset, **kw)
    dl = DataLoader(ds, batch_size=int(bs), shuffle=False, num_workers=0)

    model = make_classifier(mf, backbone, mode, num_classes=2, pretrained=False)
    model.load_state_dict(ck["state"])
    model.to(device).eval()

    y_true, probs = eval_probs(model, dl, device, temperature=temperature)
    pred = (probs >= thr).astype(np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)
    tbl_rows = []
    cm_json: dict = {}

    for src in sorted(subset["source"].astype(str).unique()):
        m = subset["source"].astype(str).values == src
        yt = np.asarray(y_true[m], dtype=np.int64)
        yp = np.asarray(pred[m], dtype=np.int64)
        n = len(yt)
        if n == 0:
            continue
        # FPR definition on binary: negatives as class 0
        neg_m = yt == 0
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fpr = float(fp / max(1, int(neg_m.sum()))) if neg_m.any() else float("nan")
        rec = float(tp / max(1, int((yt == 1).sum()))) if int((yt == 1).sum()) else float("nan")
        tbl_rows.append(
            {"source": src, "n": n, "n_neg": int(neg_m.sum()), "n_pos": int((yt == 1).sum()), "FP": fp, "FPR": fpr, "TP": tp, "recall": rec}
        )
        try:
            cm_json[src] = confusion_matrix(yt, yp).tolist()
        except Exception:
            cm_json[src] = [[int(tn), int(fp)], [int(fn), int(tp)]]

    fps_rows = []
    for ii in np.where((y_true == 0) & (pred == 1))[0]:
        row = subset.iloc[int(ii)]
        fps_rows.append(
            {
                "source": row.get("source", ""),
                "path_rgb": row.get("path_rgb", ""),
                "path_th": row.get("path_th", ""),
                "label": 0,
                "prob_fire": float(probs[int(ii)]),
                "split": row.get("split", ""),
                "threshold": thr,
            }
        )

    pd.DataFrame(tbl_rows).sort_values(by="source").to_csv(out_dir / "source_fpr_table.csv", index=False)
    (out_dir / "confusion_matrices.json").write_text(json.dumps(cm_json, indent=2), encoding="utf-8")

    fps_df = pd.DataFrame(fps_rows)
    fps_df.to_csv(out_dir / "false_positive_frames.csv", index=False)
    gall = out_dir / "gallery"
    for ri, (_, r_) in enumerate(fps_df.head(int(gallery_max)).iterrows()):
        p_rgb = Path(str(r_["path_rgb"]))
        if p_rgb.is_file():
            _save_thumbnail(str(p_rgb), gall / f"fp_{ri:04d}_{p_rgb.stem[:40]}.png")

    meta = {
        "ckpt": str(ckpt_path),
        "index": str(index_csv),
        "split": split,
        "sources_requested": sources,
        "thermal_norm": tn,
        "threshold": thr,
        "temperature": temperature,
        "n_audited_rows": len(subset),
        "gallery_cap": gallery_max,
    }
    (out_dir / "audit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[audit] wrote under {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/binary_root_audit"))
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--sources",
        default="binary_root",
        help="Comma-separated source names included in audit (subset of index)",
    )
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--gallery_max", type=int, default=80)
    args = ap.parse_args()
    sources = [s.strip() for s in str(args.sources).split(",") if s.strip()]
    run_audit(
        ckpt_path=args.ckpt,
        index_csv=args.csv,
        out_dir=args.out_dir,
        sources=sources,
        split=args.split,
        bs=int(args.bs),
        gallery_max=int(args.gallery_max),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
