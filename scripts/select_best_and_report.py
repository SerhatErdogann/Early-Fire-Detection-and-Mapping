#!/usr/bin/env python3
"""Pick ``best_model.pt`` from ``outputs/improve_results.csv`` and write ``best_model_report.md``.

Ranking matches training policy ``recall_fpr`` conceptually:

1. Among rows where ``test_recall`` >= MIN_REC (default 0.98), minimize ``test_false_positive_rate``.
2. Tie-break by higher ``test_bal_acc`` then ``test_f1``.

Baseline row is identified by ``model_family == dual_branch_fusion`` plus optional ``experiment_name``.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def _slug_experiment_name(name: str) -> str:
    import re

    s = re.sub(r"[^\w\-.]", "_", str(name).strip())
    return (s[:120].strip("_") or "unnamed")


def _archived_ckpt_for_experiment(out_ckpt_str: str, experiment_name: str) -> Path | None:
    """If ``models/by_experiment/{slug}.pt`` exists next to canonical ``out_ckpt``, use it."""
    if not experiment_name or not str(out_ckpt_str).strip():
        return None
    p = Path(str(out_ckpt_str)).expanduser()
    cand = p.parent / "by_experiment" / f"{_slug_experiment_name(experiment_name)}.pt"
    return cand if cand.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_csv", default="outputs/improve_results.csv")
    ap.add_argument("--min_recall", type=float, default=0.98)
    ap.add_argument("--baseline_family", default="dual_branch_fusion")
    ap.add_argument("--out_ckpt", default="best_model.pt")
    ap.add_argument("--out_md", default="best_model_report.md")
    args = ap.parse_args()
    p = Path(args.results_csv)
    if not p.exists():
        print(f"No results file: {p}")
        return 2
    df = pd.read_csv(p)
    if "suite_audit" in df.columns:
        m = pd.to_numeric(df["suite_audit"], errors="coerce").fillna(0).astype(int) == 0
        df = df.loc[m].copy()
    rcol, fpr_col = "test_recall", "test_false_positive_rate"
    bac, f1c = "test_bal_acc", "test_f1"
    thr_col = "threshold_saved"
    for c in (rcol, fpr_col, bac, f1c):
        if c not in df.columns:
            print(f"Missing column {c!r}; did training finish writing metrics rows?")
            return 3

    ok = df[df[rcol] >= float(args.min_recall)].copy()
    if len(ok) == 0:
        ok = df.copy()
        elig_note = (
            f"**Warning**: no rows with {rcol}>={args.min_recall}; used unfiltered rankings."
        )
    else:
        elig_note = f"Filtered `{rcol}>={float(args.min_recall):.3f}`: **{len(ok)} / {len(df)}** runs."

    ok = ok.sort_values(
        by=[fpr_col, bac, f1c, rcol],
        ascending=[True, False, False, False],
    )
    winner = ok.iloc[0]

    baseline_mask = df["model_family"].astype(str) == str(args.baseline_family)
    if baseline_mask.any():
        base = df[baseline_mask].sort_values(by=rcol, ascending=False).iloc[0]
    else:
        base = df.sort_values(by=rcol, ascending=False).iloc[0]

    def _flt(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    ckpt_src = winner.get("out_ckpt", "")
    exper = str(winner.get("experiment_name", "") or "").strip()
    archived = _archived_ckpt_for_experiment(str(ckpt_src), exper)
    ckpt_pick = archived if archived is not None else Path(str(ckpt_src))
    if not ckpt_pick.is_file():
        print(f"Checkpoint not found: {ckpt_pick}")
        return 4
    shutil.copy2(ckpt_pick, Path(args.out_ckpt))
    ckpt_display = str(archived) if archived is not None else str(ckpt_src)

    lines = []
    md_path = Path(args.out_md)
    lines.append("# Best fusion model summary\n")
    lines.append(elig_note + "\n\n")
    lines.append("## Winner\n")
    lines.append(f"- **experiment**: `{winner.get('experiment_name', '')}`\n")
    lines.append(f"- **model_family**: `{winner.get('model_family', '')}`\n")
    lines.append(f"- **checkpoints copied to**: `{args.out_ckpt}` (from `{ckpt_display}`)\n")
    lines.append(f"- **threshold_saved**: {_flt(winner.get(thr_col, float('nan')))}\n")
    lines.append(
        f"- **test recall / FPR / bal_acc / F1**: "
        f"{_flt(winner.get(rcol)):.4f} / {_flt(winner.get(fpr_col)):.4f} / "
        f"{_flt(winner.get(bac)):.4f} / {_flt(winner.get(f1c)):.4f}\n"
    )

    lines.append("\n## Why this row\n")
    lines.append(
        "Lowest test FPR among runs meeting the recall target; "
        "tie-break balanced accuracy → F1 (see sort order in script).\n"
    )

    lines.append("\n## vs baseline (`model_family` = ")
    lines.append(f"`{args.baseline_family}`)\n")
    lines.append(
        f"- baseline test R / FPR: {_flt(base.get(rcol)):.4f} / {_flt(base.get(fpr_col)):.4f}\n"
    )
    lines.append(
        f"- winner test R / FPR: {_flt(winner.get(rcol)):.4f} / {_flt(winner.get(fpr_col)):.4f}\n"
    )
    lines.append(
        f"- ΔFPR (negative is better): {_flt(winner.get(fpr_col)) - _flt(base.get(fpr_col)):.4f}\n"
    )

    lines.append("\n## Known weak spots\n")
    lines.append(
        f"- **worst_source_by_fpr (from metrics json)**: `{winner.get('worst_source_fpr', '')}`\n"
    )
    lines.append(
        f"- **worst_source_by_recall**: `{winner.get('worst_source_recall', '')}`\n"
    )
    lines.append(
        "\n_Regenerate after populating `outputs/improve_results.csv` from training runs._\n"
    )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {md_path} and copied checkpoint to {args.out_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
