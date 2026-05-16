#!/usr/bin/env python3
"""Run a long fusion experiment grid on Kaggle (or locally) with resume + failure log.

Features
--------
- Skip runs whose ``experiment_name`` already exists in ``improve_results.csv`` with a
  numeric ``test_realistic_recall`` (restart-safe; legacy rows may still expose ``test_recall``).
- After each successful train: copy checkpoint to ``models/by_experiment/{slug}.pt``
  so ``select_best_and_report.py`` can pick the *correct* weights (canonical
  ``dual_branch.pt`` / ``fusion.pt`` are overwritten each run).
- Run ``robustness_eval`` + ``ablation_eval`` per experiment; archive CSVs under
  ``outputs/kaggle_eval_archive/`` and refresh ``outputs/robustness_eval.csv`` /
  ``outputs/ablation_suite.csv`` with the latest run.
- On train/eval failure: append ``logs/failed_runs.csv`` and continue.
- Optionally run ``select_best_and_report.py`` at the end.

Typical Kaggle layout (set in your setup cell)::

    FLAME_OUTPUTS_DIR=/kaggle/working/outputs
    FLAME_MODELS_DIR=/kaggle/working/models
    FLAME_MASTER_INDEX=/kaggle/working/data/master_index.parquet   # or input path

Run from project root (e.g. ``cd /kaggle/working/code``)::

    python scripts/run_kaggle_full_suite.py \\
        --master-index /kaggle/working/data/master_index.parquet \\
        --code-root /kaggle/working/code \\
        --working-root /kaggle/working
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _slug_experiment_name(name: str) -> str:
    s = re.sub(r"[^\w\-.]", "_", str(name).strip())
    return (s[:120].strip("_") or "unnamed")


def _canonical_ckpt(models_dir: Path, model_family: str) -> Path:
    if str(model_family) == "early_fusion":
        return models_dir / "fusion.pt"
    return models_dir / "dual_branch.pt"


def _archive_checkpoint(models_dir: Path, src: Path, experiment_name: str) -> Path | None:
    if not src.is_file():
        return None
    dest_dir = models_dir / "by_experiment"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{_slug_experiment_name(experiment_name)}.pt"
    shutil.copy2(src, dest)
    return dest


def _is_experiment_completed(improve_csv: Path, experiment_name: str) -> bool:
    if not improve_csv.is_file():
        return False
    try:
        df = pd.read_csv(improve_csv)
    except Exception:
        return False
    if df.empty or "experiment_name" not in df.columns:
        return False
    if "suite_audit" in df.columns:
        m = pd.to_numeric(df["suite_audit"], errors="coerce").fillna(0).astype(int) == 0
        df = df.loc[m].copy()
    mask = df["experiment_name"].astype(str).str.strip() == experiment_name
    if not mask.any():
        return False
    sub = df.loc[mask]
    col = (
        "test_realistic_recall"
        if "test_realistic_recall" in sub.columns
        else ("test_recall" if "test_recall" in sub.columns else None)
    )
    if col is None:
        return False
    tr = pd.to_numeric(sub.get(col, float("nan")), errors="coerce")
    return bool(tr.notna().any())


def _append_fail_row(logs_dir: Path, row: dict) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / "failed_runs.csv"
    cols = sorted(row.keys())
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            w.writeheader()
        w.writerow(row)


def _append_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted(row.keys())
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            w.writeheader()
        w.writerow(row)


def _run_capture(cmd: list[str], cwd: Path, env: dict) -> tuple[int, str]:
    r = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
    )
    err = (r.stderr or "").strip()
    out = (r.stdout or "").strip()
    blob = err if len(err) >= len(out) else out
    if not blob:
        blob = err or out
    tail = blob[-3000:] if len(blob) > 3000 else blob
    return int(r.returncode), tail


def _experiments_catalog() -> list[dict]:
    """Ordered experiment grid (edit below for your notebook)."""
    fusion_common = ["--selection_metric", "recall_fpr", "--loss_mode", "balanced_sampler", "--loss_name", "cb_focal"]
    return [
        {
            "experiment_name": "kaggle_early_fusion_effnet",
            "model_family": "early_fusion",
            "extra_args": fusion_common.copy(),
            "notes": "Single-encoder 4-channel baseline",
        },
        {
            "experiment_name": "kaggle_dbf_baseline_rcfpr",
            "model_family": "dual_branch_fusion",
            "extra_args": fusion_common.copy(),
            "notes": "Dual-branch concat baseline + recall/FPR metric",
        },
        {
            "experiment_name": "kaggle_dbf_strengthened",
            "model_family": "dual_branch_fusion",
            "extra_args": fusion_common.copy()
            + [
                "--modal_dropout_p",
                "0.15",
                "--thermal_lr_mult",
                "1.25",
                "--freeze_rgb_epochs",
                "2",
                "--thermal_norm",
                "train_zscore",
            ],
            "notes": "Modal dropout + thermal LR + warmup + train_zscore",
        },
        {
            "experiment_name": "kaggle_dbf_attn",
            "model_family": "dual_branch_attention_fusion",
            "extra_args": fusion_common.copy()
            + [
                "--modal_dropout_p",
                "0.25",
                "--thermal_lr_mult",
                "1.25",
                "--freeze_rgb_epochs",
                "2",
                "--thermal_norm",
                "train_zscore",
            ],
            "notes": "Attention fusion variant",
        },
        {
            "experiment_name": "kaggle_dbf_mid_res50",
            "model_family": "dual_branch_mid_fusion",
            "extra_args": fusion_common.copy()
            + ["--backbone", "resnet50", "--modal_dropout_p", "0.18", "--thermal_lr_mult", "1.15", "--freeze_rgb_epochs", "2", "--thermal_norm", "train_zscore"],
            "notes": "Mid fusion heavier path",
        },
        {
            "experiment_name": "kaggle_dbf_gated_primary",
            "model_family": "dual_branch_gated_fusion",
            "extra_args": fusion_common.copy()
            + [
                "--modal_dropout_p",
                "0.25",
                "--thermal_lr_mult",
                "1.25",
                "--freeze_rgb_epochs",
                "3",
                "--thermal_norm",
                "train_zscore",
            ],
            "notes": "Gated fusion main recipe",
        },
        {
            "experiment_name": "kaggle_dbf_gated_md02_fr2",
            "model_family": "dual_branch_gated_fusion",
            "extra_args": fusion_common.copy()
            + [
                "--modal_dropout_p",
                "0.2",
                "--thermal_lr_mult",
                "1.2",
                "--freeze_rgb_epochs",
                "2",
                "--thermal_norm",
                "train_zscore",
            ],
            "notes": "Gated fusion variant A",
        },
        {
            "experiment_name": "kaggle_dbf_gated_md03_fr1",
            "model_family": "dual_branch_gated_fusion",
            "extra_args": fusion_common.copy()
            + [
                "--modal_dropout_p",
                "0.3",
                "--thermal_lr_mult",
                "1.3",
                "--freeze_rgb_epochs",
                "1",
                "--thermal_norm",
                "train_zscore",
            ],
            "notes": "Gated fusion variant B",
        },
        {
            "experiment_name": "kaggle_cmp_th_percentile",
            "model_family": "dual_branch_fusion",
            "extra_args": fusion_common.copy()
            + [
                "--thermal_norm",
                "percentile",
            ],
            "notes": "Thermal norm: percentile",
        },
        {
            "experiment_name": "kaggle_cmp_th_train_zscore",
            "model_family": "dual_branch_fusion",
            "extra_args": fusion_common.copy()
            + [
                "--thermal_norm",
                "train_zscore",
            ],
            "notes": "Thermal norm: train z-score remap",
        },
        {
            "experiment_name": "kaggle_cmp_th_minmax",
            "model_family": "dual_branch_fusion",
            "extra_args": fusion_common.copy()
            + [
                "--thermal_norm",
                "minmax",
            ],
            "notes": "Thermal norm: minmax",
        },
        {
            "experiment_name": "kaggle_cmp_loss_focal_shuffle",
            "model_family": "dual_branch_fusion",
            "extra_args": [
                "--selection_metric",
                "recall_fpr",
                "--loss_mode",
                "focal_shuffle",
                "--loss_name",
                "focal",
                "--thermal_norm",
                "train_zscore",
            ],
            "notes": "Loss: focal_shuffle + focal loss",
        },
        {
            "experiment_name": "kaggle_cmp_loss_sampler_focal_ce",
            "model_family": "dual_branch_fusion",
            "extra_args": ["--selection_metric", "recall_fpr", "--loss_mode", "sampler_focal", "--loss_name", "ce", "--thermal_norm", "train_zscore"],
            "notes": "Loss: sampler_ce-style path with sampler_focal mode + CE",
        },
        {
            "experiment_name": "kaggle_cmp_loss_sampler_ce_plain",
            "model_family": "dual_branch_fusion",
            "extra_args": ["--selection_metric", "recall_fpr", "--loss_mode", "sampler_ce", "--loss_name", "ce", "--thermal_norm", "train_zscore"],
            "notes": "Loss: weighted sampler + plain CE",
        },
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code-root", default=os.environ.get("KAGGLE_CODE_ROOT", str(Path(__file__).resolve().parents[1])))
    ap.add_argument("--working-root", default=os.environ.get("KAGGLE_WORKING_ROOT", "/kaggle/working"))
    ap.add_argument(
        "--master-index",
        default=os.environ.get("FLAME_MASTER_INDEX", ""),
        help="Path to master_index.parquet (or CSV). Required unless set in env.",
    )
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=8, help="Kaggle GPU memory–safe default.")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument(
        "--improve-csv",
        default="",
        help="Default: {working}/outputs/improve_results.csv relative to cwd, or FLAME_OUTPUTS_DIR.",
    )
    ap.add_argument("--logs-dir", default="", help="Default: {working-root}/logs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-skip-done", action="store_true")
    ap.add_argument("--no-eval", action="store_true", help="Skip robustness/ablation after each train.")
    ap.add_argument("--no-select-best", action="store_true")
    ap.add_argument("--only", default="", help="Regex; only experiments whose name matches.")
    ap.add_argument("--list", action="store_true", help="Print experiment grid and exit.")
    args = ap.parse_args()

    code_root = Path(args.code_root).resolve()
    work_root = Path(args.working_root).resolve()

    experiments = _experiments_catalog()
    if args.list:
        for ex in experiments:
            print(json.dumps({"experiment_name": ex["experiment_name"], "model_family": ex["model_family"], "notes": ex.get("notes", "")}))
        return 0

    master = str(args.master_index or "").strip()
    if not args.dry_run:
        if not master or not Path(master).expanduser().is_file():
            print(
                "Provide a valid master index via --master-index or FLAME_MASTER_INDEX.",
                file=sys.stderr,
            )
            return 2
        csv_arg = str(Path(master).expanduser().resolve())
    elif master and Path(master).expanduser().is_file():
        csv_arg = str(Path(master).expanduser().resolve())
    else:
        csv_arg = "/kaggle/working/data/master_index.parquet"
        print("[dry-run] master index unset; CSV path placeholder in printed commands only.")

    outs = Path(os.environ.get("FLAME_OUTPUTS_DIR") or (work_root / "outputs")).resolve()
    mods = Path(os.environ.get("FLAME_MODELS_DIR") or (work_root / "models")).resolve()
    outs.mkdir(parents=True, exist_ok=True)
    mods.mkdir(parents=True, exist_ok=True)

    improve_csv = Path(args.improve_csv).expanduser() if str(args.improve_csv).strip() else (outs / "improve_results.csv")
    logs_dir = Path(args.logs_dir).expanduser() if str(args.logs_dir).strip() else (work_root / "logs")

    manifest = outs / "kaggle_full_suite_manifest.csv"
    archive_dir = outs / "kaggle_eval_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    env_base.setdefault("FLAME_OUTPUTS_DIR", str(outs))
    env_base.setdefault("FLAME_MODELS_DIR", str(mods))
    if csv_arg and Path(csv_arg).expanduser().is_file():
        env_base.setdefault("FLAME_MASTER_INDEX", str(Path(csv_arg).expanduser().resolve()))

    only_re = re.compile(args.only) if str(args.only).strip() else None

    py = sys.executable
    for ex in experiments:
        ename = str(ex["experiment_name"])
        mf = str(ex["model_family"])
        if only_re is not None and not only_re.search(ename):
            continue

        if not args.no_skip_done and _is_experiment_completed(improve_csv, ename):
            print(f"[skip] done: {ename}")
            _append_manifest(
                manifest,
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "experiment_name": ename,
                    "status": "skipped_done",
                    "train_rc": "",
                    "robustness_rc": "",
                    "ablation_rc": "",
                    "note": "already in improve_results.csv",
                },
            )
            continue

        train_cmd = [
            py,
            str(code_root / "src" / "02_train.py"),
            "--mode",
            "fusion",
            "--model_family",
            mf,
            "--csv",
            csv_arg,
            "--epochs",
            str(int(args.epochs)),
            "--patience",
            str(int(args.patience)),
            "--bs",
            str(int(args.bs)),
            "--lr",
            str(float(args.lr)),
            "--experiment_log_csv",
            str(improve_csv),
            "--experiment_name",
            ename,
        ] + list(ex.get("extra_args") or [])

        if args.dry_run:
            print("[dry-run] +", " ".join(train_cmd))
            continue

        print("+", " ".join(train_cmd), flush=True)
        rc, blob = _run_capture(train_cmd, cwd=code_root, env=env_base)
        row_m = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_name": ename,
            "status": "ok" if rc == 0 else "train_failed",
            "train_rc": str(rc),
            "robustness_rc": "",
            "ablation_rc": "",
            "note": "",
        }

        ckpt_canon = _canonical_ckpt(mods, mf)

        rob_rc_s = ""
        ab_rc_s = ""

        if rc != 0:
            row_m["note"] = (blob[:500] + "…") if len(blob) > 500 else blob
            _append_fail_row(
                logs_dir,
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "experiment_name": ename,
                    "stage": "train",
                    "exit_code": str(rc),
                    "message": blob[:1800],
                    "command": json.dumps(train_cmd, ensure_ascii=False),
                },
            )
            _append_manifest(manifest, row_m | {"robustness_rc": "", "ablation_rc": "", "note": row_m["note"][:400]})
            continue

        arch = _archive_checkpoint(mods, ckpt_canon, ename)
        if arch:
            print(f"[archive] ckpt -> {arch}")

        if not args.no_eval:
            slug = _slug_experiment_name(ename)
            rob_out = archive_dir / f"robustness__{slug}.csv"
            ab_out = archive_dir / f"ablation__{slug}.csv"

            rob_cmd = [
                py,
                "-m",
                "src.eval.robustness_eval",
                "--ckpt",
                str(ckpt_canon),
                "--csv",
                csv_arg,
                "--split",
                "test",
                "--out",
                str(rob_out),
            ]
            ab_cmd = [
                py,
                "-m",
                "src.eval.ablation_eval",
                "--ckpt",
                str(ckpt_canon),
                "--csv",
                csv_arg,
                "--split",
                "test",
                "--out",
                str(ab_out),
            ]
            print("+", " ".join(rob_cmd), flush=True)
            rob_rc, rob_blob = _run_capture(rob_cmd, cwd=code_root, env=env_base)
            rob_rc_s = str(rob_rc)
            if rob_rc != 0:
                _append_fail_row(
                    logs_dir,
                    {
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "experiment_name": ename,
                        "stage": "robustness_eval",
                        "exit_code": str(rob_rc),
                        "message": rob_blob[:1800],
                        "command": json.dumps(rob_cmd, ensure_ascii=False),
                    },
                )

            print("+", " ".join(ab_cmd), flush=True)
            ab_rc, ab_blob = _run_capture(ab_cmd, cwd=code_root, env=env_base)
            ab_rc_s = str(ab_rc)
            if ab_rc != 0:
                _append_fail_row(
                    logs_dir,
                    {
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "experiment_name": ename,
                        "stage": "ablation_eval",
                        "exit_code": str(ab_rc),
                        "message": ab_blob[:1800],
                        "command": json.dumps(ab_cmd, ensure_ascii=False),
                    },
                )

            try:
                if rob_out.is_file():
                    shutil.copy2(rob_out, outs / "robustness_eval.csv")
                if ab_out.is_file():
                    shutil.copy2(ab_out, outs / "ablation_suite.csv")
            except Exception as exc:
                _append_fail_row(
                    logs_dir,
                    {
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "experiment_name": ename,
                        "stage": "eval_copy",
                        "exit_code": "-1",
                        "message": f"{type(exc).__name__}: {exc}",
                        "command": "",
                    },
                )

        row_m["robustness_rc"] = rob_rc_s
        row_m["ablation_rc"] = ab_rc_s
        _append_manifest(manifest, row_m)

    if args.dry_run:
        print("[dry-run] done (no subprocess).")
        return 0

    if args.no_select_best:
        print(f"[suite] manifest -> {manifest}")
        return 0

    sel_cmd = [
        py,
        str(code_root / "scripts" / "select_best_and_report.py"),
        "--results_csv",
        str(improve_csv),
        "--copy_balanced_ckpt",
        str(mods / "best_model.pt"),
        "--out_md",
        str(outs / "best_model_report.md"),
    ]
    print("+", " ".join(sel_cmd), flush=True)
    rc_sb, blob = _run_capture(sel_cmd, cwd=code_root, env=env_base)
    if rc_sb != 0:
        _append_fail_row(
            logs_dir,
            {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "experiment_name": "__select_best__",
                "stage": "select_best_and_report",
                "exit_code": str(rc_sb),
                "message": blob[:1800],
                "command": json.dumps(sel_cmd, ensure_ascii=False),
            },
        )
        return int(rc_sb)

    print(f"[suite] manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
