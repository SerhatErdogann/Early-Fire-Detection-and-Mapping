#!/usr/bin/env python3
"""Priority training grid + ablation + robustness + improve_results augmentation.

Runs (in order):
  1) dual_branch_gated_fusion — pri-1 hyperparameters
  2) dual_branch_attention_fusion — pri-2
  3) dual_branch_mid_fusion — optional if early runs miss test recall threshold

Adds ``suite_summary.json`` per run and merges diagnostics into experiment CSV paths.

Requirements: project root cwd, usable ``master_index.parquet``, GPU optional but slow otherwise.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime, timezone

from src.training.trainer import append_experiment_csv_row, sanitize_for_json

def run_cmd(cmd: list[str], cwd: Path) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(cwd))


def detect_modality_collapse(csv_path: Path) -> tuple[bool, str]:
    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    if df.empty:
        return False, "no_robustness_csv"
    try:
        clean = df[(df["corruption"] == "clean")]["recall"].iloc[0]
        r_rgb3 = df[(df["corruption"] == "gauss_noise_rgb") & (df["severity"] == 3)][
            "recall"
        ].iloc[0]
        rt3 = df[(df["corruption"] == "gauss_noise_thermal") & (df["severity"] == 3)][
            "recall"
        ].iloc[0]
    except Exception:
        return False, "parse_error"
    collapse = (float(r_rgb3) < 0.05) and (float(rt3) > float(clean) * 0.85)
    reason = ""
    if collapse:
        reason = (
            "RGB gauss_noise sev3 recall ~0 while thermal noise keeps clean-like recall "
            "(model likely RGB-dominated modality collapse)."
        )
    else:
        reason = (
            "No RGB-collapse signature (severity-3 RGB noise did not wipe recall "
            "with thermal unaffected)."
        )
    return collapse, reason


def _metrics_flat_for_improve_csv(
    mf: str, mode: str, metrics_json: Path, diag_extras: dict
) -> dict | None:
    """Build a DictWriter-stable row comparable to trainer ``experiment_log_csv`` rows."""
    if not metrics_json.is_file():
        return None
    try:
        lastm = json.loads(metrics_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    ta = lastm.get("training_args") if isinstance(lastm.get("training_args"), dict) else {}
    suffix = str(diag_extras.get("experiment_name_suffix") or f"{mf}_suite_audit")
    row: dict = {
        "ts_done": datetime.now(timezone.utc).isoformat(),
        "experiment_name": suffix,
        "suite_audit": 1,
        "csv_index": str(diag_extras.get("csv_index", "")),
        "model_family": mf,
        "mode": mode,
        "backbone": str(ta.get("backbone", lastm.get("backbone", ""))),
        "thermal_norm": str(ta.get("thermal_norm", "")),
        "thermal_init": str(ta.get("thermal_init", "")),
        "freeze_rgb_epochs": ta.get("freeze_rgb_epochs", ""),
        "thermal_lr_mult": ta.get("thermal_lr_mult", ""),
        "modal_dropout_p": ta.get("modal_dropout_p", ""),
        "selection_metric": str(ta.get("selection_metric", lastm.get("selection_metric", ""))),
        "loss_name": str(ta.get("loss_name", "")),
        "out_ckpt": str(diag_extras.get("out_ckpt_override", "")),
    }
    for split in ("val", "test"):
        if isinstance(lastm.get(split), dict):
            p = lastm[split]
            for k in (
                "acc",
                "bal_acc",
                "precision",
                "recall",
                "f1",
                "specificity",
                "false_positive_rate",
                "auc",
                "ap",
            ):
                if k in p:
                    val = p[k]
                    row[f"{split}_{k}"] = (
                        float(val) if not isinstance(val, (list, dict)) else json.dumps(val)
                    )
            if "cm" in p:
                row[f"{split}_cm"] = json.dumps(sanitize_for_json(p["cm"]))
    row["threshold_saved"] = float(lastm.get("threshold", float("nan")))
    ws = lastm.get("worst_source_by_fpr")
    row["worst_source_fpr"] = json.dumps(ws) if isinstance(ws, (dict, list)) else ws
    wr = lastm.get("worst_source_by_recall")
    row["worst_source_recall"] = json.dumps(wr) if isinstance(wr, (dict, list)) else wr
    _skip = {"experiment_name_suffix", "out_ckpt_override", "csv_index"}
    row.update({k: v for k, v in diag_extras.items() if k not in _skip})
    return row


def fusion_ablation_signals(ab_csv: Path) -> tuple[float, float, bool, str]:
    """Thermal-zero vs thermal-only proxy using same fusion ckpt."""
    if not ab_csv.exists():
        return float("nan"), float("nan"), False, "no_ablation_csv"
    df_raw = pd.read_csv(ab_csv)
    if df_raw.empty or "condition" not in df_raw.columns:
        return float("nan"), float("nan"), False, "ablation_empty_columns"
    df = df_raw.set_index("condition")
    try:
        r_full = float(df.loc["rgb_full_thermal_full", "recall"])
        r_tonly = float(df.loc["thermal_only_rgb_zero", "recall"])
        r_ronly = float(df.loc["rgb_only_thermal_zero", "recall"])
    except Exception:
        return float("nan"), float("nan"), False, "ablation_missing_rows"
    uses_th = abs(r_full - r_ronly) > 0.03
    note = []
    note.append(f"d_full_minus_r_rgb0={r_full - r_ronly:.4f}")
    note.append(f"d_full_minus_t_th0={r_full - r_tonly:.4f}")
    suspicious = uses_th is False and r_tonly > 0.5
    msg = "; ".join(note)
    if suspicious:
        msg += "; fusion may ignore thermal branch (scores similar when thermal zeroed)."
    return r_full, r_full - max(r_ronly, r_tonly), uses_th, msg


def diagnose_gated_ckpt(ckpt: Path, mf: str) -> dict:
    """Mean gate softmax on synthetic batch — no subprocess."""
    if mf != "dual_branch_gated_fusion" or not ckpt.is_file():
        return {"skipped": True}
    try:
        import torch

        from src.models import make_classifier

        try:
            d = torch.load(ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            d = torch.load(ckpt, map_location="cpu")
        mdl = make_classifier(
            str(d.get("model_family", mf)),
            str(d.get("backbone", "resnet50")),
            "fusion",
            pretrained=False,
        )
        mdl.load_state_dict(d["state"])
        mdl.eval()
        x = torch.rand(48, 4, 128, 128)
        with torch.no_grad():
            _, aux = mdl(x, return_aux=True)
        rg = float(aux["gate_rgb"].mean())
        tg = float(aux["gate_thermal"].mean())
        return {
            "mean_gate_rgb": rg,
            "mean_gate_thermal": tg,
            "warn_low_thermal": tg < 0.35,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def append_csv_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted(row.keys())
    new_file = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None, help="master_index parquet (default: auto)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--dry_run", action="store_true", help="Print commands without running.")
    ap.add_argument(
        "--experiment_log_csv",
        default=str(PROJECT_ROOT / "outputs/improve_results.csv"),
    )
    ap.add_argument(
        "--threshold_test_recall_mid",
        type=float,
        default=0.98,
        help="Run mid-fusion experiment if BOTH earlier runs dip below this on TEST recall.",
    )
    args = ap.parse_args()

    py = sys.executable
    cwd = PROJECT_ROOT

    try:
        from config import CKPT_DUAL_BRANCH, MASTER_INDEX_PARQUET
    except ImportError:
        CKPT_DUAL_BRANCH = PROJECT_ROOT / "models/dual_branch.pt"
        MASTER_INDEX_PARQUET = PROJECT_ROOT / "data/master_index.parquet"

    csv_path = args.csv or str(MASTER_INDEX_PARQUET)
    if not Path(csv_path).exists():
        print(f"Index missing: {csv_path}")
        return 2

    exp_csv = Path(args.experiment_log_csv)

    PRI1 = dict(
        name="pri1_gated",
        family="dual_branch_gated_fusion",
        extras=[
            "--modal_dropout_p",
            "0.25",
            "--thermal_lr_mult",
            "1.25",
            "--freeze_rgb_epochs",
            "3",
            "--thermal_norm",
            "train_zscore",
            "--loss_mode",
            "balanced_sampler",
            "--loss_name",
            "cb_focal",
            "--selection_metric",
            "recall_fpr",
            "--experiment_name",
            "pri1_gated",
        ],
    )
    PRI2 = dict(
        name="pri2_attention",
        family="dual_branch_attention_fusion",
        extras=[
            "--modal_dropout_p",
            "0.25",
            "--thermal_lr_mult",
            "1.25",
            "--freeze_rgb_epochs",
            "2",
            "--thermal_norm",
            "train_zscore",
            "--selection_metric",
            "recall_fpr",
            "--experiment_name",
            "pri2_attn",
            "--loss_mode",
            "balanced_sampler",
            "--loss_name",
            "cb_focal",
        ],
    )
    PRI3 = dict(
        name="pri3_mid",
        family="dual_branch_mid_fusion",
        extras=[
            "--backbone",
            "resnet50",
            "--modal_dropout_p",
            "0.2",
            "--thermal_lr_mult",
            "1.15",
            "--freeze_rgb_epochs",
            "2",
            "--thermal_norm",
            "train_zscore",
            "--selection_metric",
            "recall_fpr",
            "--experiment_name",
            "pri3_mid",
            "--loss_mode",
            "balanced_sampler",
            "--loss_name",
            "cb_focal",
        ],
    )

    recalls: dict[str, float] = {}

    def launch_train(spec: dict) -> int | None:
        cmd = [
            py,
            "src/02_train.py",
            "--mode",
            "fusion",
            "--model_family",
            spec["family"],
            "--csv",
            csv_path,
            "--epochs",
            str(int(args.epochs)),
            "--patience",
            str(int(args.patience)),
            "--bs",
            str(int(args.bs)),
            "--lr",
            str(float(args.lr)),
            "--experiment_log_csv",
            str(exp_csv),
        ]
        cmd += spec["extras"]
        if args.dry_run:
            print("[dry-run] +", " ".join(cmd))
            return None
        rc = run_cmd(cmd, cwd=cwd)
        mf_key = spec["family"]
        metrics_json = cwd / "outputs" / f"metrics_fusion_{mf_key}.json"
        tr = float("nan")
        if metrics_json.exists():
            try:
                mj = json.loads(metrics_json.read_text(encoding="utf-8"))
                tr = float(mj.get("test", {}).get("recall", float("nan")))
            except Exception:
                pass
        recalls[spec["name"]] = tr
        return int(rc)

    def collect_suite_diagnostics(spec: dict, rc_train: int) -> None:
        if args.dry_run:
            return
        mf = spec["family"]
        metrics_json = cwd / "outputs" / f"metrics_fusion_{mf}.json"
        tr = float(recalls.get(spec["name"], float("nan")))
        gated = diagnose_gated_ckpt(Path(str(CKPT_DUAL_BRANCH)), mf)
        gated_warn = ""
        if isinstance(gated, dict) and gated.get("warn_low_thermal"):
            gated_warn = "WARN: thermal gate mean < 0.35."
        collapse, cnote = detect_modality_collapse(PROJECT_ROOT / "outputs/robustness_eval.csv")
        collapse_flag = collapse
        cnote_full = (cnote + " " + gated_warn).strip()
        r_full, ddom, uses_th, ab_detail = fusion_ablation_signals(
            PROJECT_ROOT / "outputs/ablation_suite.csv"
        )
        row = {
            "suite_phase": spec["name"],
            "model_family": mf,
            "train_exit_code": int(rc_train),
            "test_recall_after_train": tr if tr == tr else "",
            "modality_collapse_flag": int(bool(collapse_flag)),
            "modality_collapse_notes": cnote_full[:800],
            "ablation_dom_delta": round(ddom, 6) if ddom == ddom else "",
            "fusion_thermal_use_ok": int(bool(uses_th)),
            "fusion_ablation_notes": str(ab_detail)[:800],
            "gated_diag_json": json.dumps(gated, ensure_ascii=False)[:900],
            "experiment_log_csv_used": str(exp_csv),
        }
        append_csv_row(PROJECT_ROOT / "outputs/experiment_suite_diagnostics.csv", row)
        (PROJECT_ROOT / f"outputs/suite_diag_{spec['name']}.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        merged = PROJECT_ROOT / "outputs/improve_results_suite_merged_hints.csv"
        row2 = dict(row)
        row2["metrics_json"] = str(metrics_json)
        append_csv_row(merged, row2)
        diag_for_csv = dict(row)
        en_suffix = ""
        if metrics_json.is_file():
            try:
                en_suffix = str(
                    json.loads(metrics_json.read_text(encoding="utf-8")).get(
                        "experiment_name", ""
                    )
                )
            except Exception:
                en_suffix = ""
        diag_for_csv["experiment_name_suffix"] = f"{(en_suffix or spec['name']).strip()}_suite_audit"
        diag_for_csv["csv_index"] = str(csv_path)
        diag_for_csv["out_ckpt_override"] = str(CKPT_DUAL_BRANCH)
        compat = _metrics_flat_for_improve_csv(mf, "fusion", metrics_json, diag_for_csv)
        if compat:
            append_experiment_csv_row(str(exp_csv), compat)
            print(f"[suite] appended suite audit row -> {exp_csv}")
        print(f"[suite] fusion_thermal_use_ok={uses_th} | {str(ab_detail)[:240]}")
        print(f"[suite] gated/thermal/ablation diagnostics -> outputs/suite_diag_{spec['name']}.json")

    def robustness_and_ablation() -> None:
        if args.dry_run:
            print("[dry-run] skip ablation+robustness")
            return
        ckpt = Path(str(CKPT_DUAL_BRANCH))
        # robustness sweep
        subprocess.run(
            [
                py,
                "-m",
                "src.eval.robustness_eval",
                "--ckpt",
                str(ckpt),
                "--csv",
                csv_path,
                "--split",
                "test",
                "--out",
                str(PROJECT_ROOT / "outputs/robustness_eval.csv"),
            ],
            cwd=str(cwd),
            check=False,
        )
        # ablation
        subprocess.run(
            [
                py,
                "-m",
                "src.eval.ablation_eval",
                "--ckpt",
                str(ckpt),
                "--csv",
                csv_path,
                "--split",
                "test",
                "--out",
                str(PROJECT_ROOT / "outputs/ablation_suite.csv"),
            ],
            cwd=str(cwd),
            check=False,
        )

    def run_phase(spec: dict) -> int | None:
        rc = launch_train(spec)
        robustness_and_ablation()
        if not args.dry_run:
            collect_suite_diagnostics(spec, int(rc) if rc is not None else -1)
        return rc

    if args.dry_run:
        for sp in (PRI1, PRI2, PRI3):
            launch_train(sp)
        print(
            "[dry-run] mid-fusion if both pri1 & pri2 test recall <",
            args.threshold_test_recall_mid,
        )
        return 0

    rc1 = run_phase(PRI1)
    rc2 = run_phase(PRI2)

    r1 = recalls.get("pri1_gated", float("nan"))
    r2 = recalls.get("pri2_attention", float("nan"))
    need_mid = False
    if r1 == r1 and r2 == r2:
        need_mid = (r1 < float(args.threshold_test_recall_mid)) and (
            r2 < float(args.threshold_test_recall_mid)
        )
    s1 = f"{r1:.4f}" if r1 == r1 else "nan"
    s2 = f"{r2:.4f}" if r2 == r2 else "nan"
    print(
        f"[suite] recalls pri1={s1} pri2={s2} threshold={args.threshold_test_recall_mid} -> need_mid={need_mid}"
    )

    rc3: int | None = None
    if need_mid:
        rc3 = run_phase(PRI3)

    summary = {"recalls_by_phase": recalls, "ran_mid": bool(need_mid), "codes": {}}
    if rc1 is not None:
        summary["codes"]["pri1"] = rc1
    if rc2 is not None:
        summary["codes"]["pri2"] = rc2
    if need_mid and rc3 is not None:
        summary["codes"]["pri3"] = rc3

    outp = PROJECT_ROOT / "outputs/suite_priority_run_summary.json"
    outp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
