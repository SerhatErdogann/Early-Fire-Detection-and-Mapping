#!/usr/bin/env python3
"""Summarise experiment grid results and optionally copy checkpoints.

Reads ``outputs/improve_results.csv`` (typically produced on Kaggle) and writes
``best_model_report.md`` with three complementary picks — not a single winner:

- **best_recall_model** — maximise test recall on **protocol-realistic test** metrics.
- **best_low_false_alarm_model** — lowest test FPR among runs meeting realistic recall gate.
- **best_balanced_model** — blends val/test realistic F1 (deployment-friendly balance).

Operational scoring uses realistic-test columns only (:func:`operational_score_from_improve_realistic_row`).

If ``--copy_balanced_ckpt`` (default ``best_model.pt``), the balanced pick is copied.
When the CSV is missing (local-only dev machine), exits **0** and writes a short stub Markdown.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.eval_reporting import (
    balanced_realistic_rank_score,
    operational_score_from_improve_realistic_row,
)


def _flt_cell(row: pd.Series | dict, key: str) -> float:
    try:
        v = row.get(key) if hasattr(row, "get") else row[key]
        if v is None or v == "":
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _protocol_metrics_table(row: pd.Series) -> str:
    """Markdown table for realistic eval (soft gaussian_blur sev=1)."""
    spec = [
        ("Validation (realistic: soft gaussian_blur sev=1)", "val_realistic"),
        ("Test (realistic: soft gaussian_blur sev=1)", "test_realistic"),
    ]
    lines = [
        "| Protocol | F1 | Recall | FPR |",
        "|---|---:|---:|---:|",
    ]
    for label, pfx in spec:
        lines.append(
            f"| {label} | {_flt_cell(row, f'{pfx}_f1'):.4f} | "
            f"{_flt_cell(row, f'{pfx}_recall'):.4f} | {_flt_cell(row, f'{pfx}_fpr'):.4f} |"
        )
    lines.append("")
    lines.append(
        "_Realistic evaluation:_ **soft** ``gaussian_blur`` severity **1** (mild defocus / jitter), "
        "full tensor, **only** at val/test forward and in ``robustness_eval`` — not train-time noise.\n"
    )
    return "\n".join(lines)


def _slug_experiment_name(name: str) -> str:
    s = re.sub(r"[^\w\-.]", "_", str(name).strip())
    return (s[:120].strip("_") or "unnamed")


def _archived_ckpt_for_experiment(out_ckpt_str: str, experiment_name: str) -> Path | None:
    if not experiment_name or not str(out_ckpt_str).strip():
        return None
    p = Path(str(out_ckpt_str)).expanduser()
    cand = p.parent / "by_experiment" / f"{_slug_experiment_name(experiment_name)}.pt"
    return cand if cand.is_file() else None


def _resolve_ckpt_path(row: pd.Series) -> Path | None:
    ckpt_src = row.get("out_ckpt", "")
    exper = str(row.get("experiment_name", "") or "").strip()
    archived = _archived_ckpt_for_experiment(str(ckpt_src), exper)
    pick = archived if archived is not None else Path(str(ckpt_src))
    return pick if pick.is_file() else None


def _row_series_to_dict(sr: pd.Series) -> dict:
    return {k: sr.get(k) for k in sr.index}


def _format_pick(title: str, row: pd.Series, *, ckpt_hint: Path | None) -> list[str]:
    def _flt(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    lines = [
        f"### {title}\n\n",
        f"- **experiment_name**: `{row.get('experiment_name', '')}`\n",
        f"- **model_family**: `{row.get('model_family', '')}`\n",
        f"- **test_realistic** recall / FPR / F1: {_flt(row.get('test_realistic_recall')):.4f} / "
        f"{_flt(row.get('test_realistic_fpr')):.4f} / {_flt(row.get('test_realistic_f1')):.4f}\n",
    ]
    lines += ["\n", _protocol_metrics_table(row), "\n"]
    osc = operational_score_from_improve_realistic_row(_row_series_to_dict(row))
    if osc == osc:
        lines.append(f"- **deployment composite (realistic test + default calib prior)**: {osc:.4f}\n")
    lines += [
        f"- **checkpoint path**: `{row.get('out_ckpt', '')}`\n",
    ]
    if ckpt_hint is not None:
        lines.append(f"- **resolved file exists**: `{ckpt_hint}`\n")
    else:
        lines.append("- **resolved file exists**: _(not found locally — train on Kaggle or fix path)_\n")
    lines.append("\n")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_csv", default="outputs/improve_results.csv")
    ap.add_argument("--min_recall", type=float, default=0.98, help="Gate for low-FPR bucket (realistic test).")
    ap.add_argument("--baseline_family", default="dual_branch_fusion")
    ap.add_argument(
        "--copy_balanced_ckpt",
        default="best_model.pt",
        help="Copy best_balanced checkpoint to this path unless --no_copy_ckpt.",
    )
    ap.add_argument(
        "--no_copy_ckpt",
        action="store_true",
        help="Do not copy any checkpoint (report only).",
    )
    ap.add_argument("--out_md", default="best_model_report.md")
    ap.add_argument(
        "--exclude_experiment_substr",
        nargs="*",
        default=["kaggle_gated_anticollapse_safe_v1"],
        help="Substring match on experiment_name; matching rows excluded from all three buckets.",
    )
    args = ap.parse_args()

    p = Path(args.results_csv)
    md_path = Path(args.out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    stub = []
    stub.append("# Model seçim özeti (`improve_results.csv`)\n\n")
    if not p.is_file():
        stub.append(
            "_`outputs/improve_results.csv` bu ortamda yok — bu dosya tipik olarak Kaggle tam "
            "koşusu sonunda üretilir. Yerelde yalnızca script/UI geliştirilebilir._\n\n"
            "Kaggle’da sıra tamamlandığında yeniden çalıştırın:\n\n"
            "```bash\n"
            "python scripts/select_best_and_report.py \\\n"
            "  --results_csv /kaggle/working/outputs/improve_results.csv \\\n"
            "  --out_md best_model_report.md \\\n"
            "  --copy_balanced_ckpt best_model.pt\n"
            "```\n"
        )
        md_path.write_text("".join(stub), encoding="utf-8")
        print(f"[select_best] No results CSV — wrote stub {md_path} (exit 0)")
        return 0

    df = pd.read_csv(p)
    if "suite_audit" in df.columns:
        m = pd.to_numeric(df["suite_audit"], errors="coerce").fillna(0).astype(int) == 0
        df = df.loc[m].copy()
    req = [
        "test_realistic_recall",
        "test_realistic_fpr",
        "test_realistic_f1",
        "val_realistic_f1",
        "val_realistic_recall",
        "val_realistic_fpr",
    ]
    miss = [c for c in req if c not in df.columns]
    if miss:
        print(f"[select_best] Missing columns {miss}; abort.")
        md_path.write_text(
            "# best_model_report\n\n_Missing CSV columns — train must log protocol-realistic metrics._\n",
            encoding="utf-8",
        )
        return 3

    for c in req:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if args.exclude_experiment_substr:
        ex = df["experiment_name"].astype(str)
        mask = pd.Series(True, index=df.index)
        for sub in args.exclude_experiment_substr:
            sub = str(sub).strip()
            if not sub:
                continue
            mask &= ~ex.str.contains(re.escape(sub), case=False, regex=True)
        before = len(df)
        df = df.loc[mask].copy()
        if len(df) < before:
            print(f"[select_best] Excluded {before - len(df)} rows matching --exclude_experiment_substr")

    if len(df) == 0:
        md_path.write_text(
            "# best_model_report\n\n_No eligible rows after filters._\n",
            encoding="utf-8",
        )
        print("[select_best] Empty frame after filtering")
        return 0

    df["_opscore"] = df.apply(lambda r: operational_score_from_improve_realistic_row(r.to_dict()), axis=1)
    df["_bal_core"] = df.apply(lambda r: balanced_realistic_rank_score(r.to_dict()), axis=1)

    best_recall = df.sort_values(
        by=["test_realistic_recall", "test_realistic_fpr", "_opscore"],
        ascending=[False, True, False],
    ).iloc[0]

    hi = df[df["test_realistic_recall"] >= float(args.min_recall)].copy()
    if len(hi) == 0:
        hi = df.copy()
        low_fpr_note = (
            f"**Uyarı**: hiçbir satır ``test_realistic_recall>={float(args.min_recall):.3f}`` değildi; "
            "_en düşük yanlış alarm_ modeli tüm küme üzerinden seçildi.\n\n"
        )
    else:
        low_fpr_note = (
            f"**best_low_false_alarm_model** seçimi için önce ``test_realistic_recall>={float(args.min_recall):.3f}`` "
            f"olan **{len(hi)}** deney filtrelendi.\n\n"
        )

    best_low_fp = hi.sort_values(
        by=["test_realistic_fpr", "test_realistic_recall", "_opscore"],
        ascending=[True, False, False],
    ).iloc[0]

    cand_balanced = hi.copy() if len(hi) else df.copy()
    best_balanced = cand_balanced.sort_values(
        by=["_bal_core", "_opscore", "test_realistic_recall"],
        ascending=[False, False, False],
    ).iloc[0]

    ck_bal = _resolve_ckpt_path(best_balanced)
    ck_recall = _resolve_ckpt_path(best_recall)
    ck_lf = _resolve_ckpt_path(best_low_fp)

    out_dest = None
    if not bool(args.no_copy_ckpt) and str(args.copy_balanced_ckpt).strip():
        out_dest = Path(str(args.copy_balanced_ckpt).strip())
    if out_dest is not None and ck_bal is not None:
        shutil.copy2(ck_bal, out_dest)
        copy_note = f"**best_balanced_model** checkpoints kopyalandı → `{out_dest}` (from `{ck_bal}`)\n\n"
        print(f"[select_best] Copied balanced checkpoint → {out_dest}")
    elif out_dest is not None:
        copy_note = f"_Balanced checkpoint kopyalanamadı (`{out_dest}`) — yerel dosya yok._\n\n"
        print("[select_best] Skipped ckpt copy (resolved path missing)")
    else:
        copy_note = ""

    baseline_mask = df["model_family"].astype(str) == str(args.baseline_family)
    base = (
        df[baseline_mask].sort_values(by="test_realistic_recall", ascending=False).iloc[0]
        if baseline_mask.any()
        else df.iloc[0]
    )

    lines = ["# Experiment grid — üç model seçimi\n\n"]
    lines.append(
        "**Realistic evaluation:** Metrikler **çok hafif** ``gaussian_blur`` severity=1 ile ölçülür "
        "(operasyonel odak kayması / titreşim benzeri; yalnızca eval forward). Clean/stress ve "
        "çoklu bozunma grid’i yok.\n\n"
    )
    lines.append(
        "Tasarım hedefi **gerçek kullanımda güvenilir yangın uyarısı**dır (yüksek yakalama + düşük yanlış alarm). "
        "Aynı satır her üç başlıkta da kazanmak zorunda değildir.\n\n"
    )
    lines.append(low_fpr_note)
    lines.append(copy_note)

    lines.extend(_format_pick("best_recall_model", best_recall, ckpt_hint=ck_recall))
    lines.extend(_format_pick("best_low_false_alarm_model", best_low_fp, ckpt_hint=ck_lf))
    lines.extend(_format_pick("best_balanced_model (realistic val/test F1 blend)", best_balanced, ckpt_hint=ck_bal))

    lines.append("## Baseline karşılaştırma (referans)\n\n")
    lines.append(f"- **baseline row** (`model_family={args.baseline_family!r}`): `{base.get('experiment_name', '')}`\n")
    lines.append(
        f"  - test_realistic R / FPR: {float(base.get('test_realistic_recall', 0)):.4f} / "
        f"{float(base.get('test_realistic_fpr', 0)):.4f}\n"
    )
    lines.append("\n_Raporu Kaggle `improve_results.csv` dolduktan sonra yeniden üretin._\n")

    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"[select_best] Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
