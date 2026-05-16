"""
Generate human-readable profile comparison from evaluation summary CSV.

Usage:
    python src/eval/profile_comparison.py --input outputs/eval_summary.csv --output outputs/profile_comparison.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _safe_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else float("nan")


def _norm01(values: pd.Series, higher_is_better: bool) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    vmin = float(v.min()) if v.notna().any() else np.nan
    vmax = float(v.max()) if v.notna().any() else np.nan
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return pd.Series(0.5, index=v.index, dtype="float64")
    if abs(vmax - vmin) < 1e-12:
        return pd.Series(0.5, index=v.index, dtype="float64")
    out = (v - vmin) / (vmax - vmin)
    if not higher_is_better:
        out = 1.0 - out
    return out.fillna(0.5).astype(float)


def _aggregate_profiles(success_df: pd.DataFrame) -> pd.DataFrame:
    if success_df.empty:
        return pd.DataFrame(
            columns=[
                "profile",
                "runs",
                "video_duration_sec_mean",
                "event_count_mean",
                "false_alarms_per_hour_mean",
                "avg_event_duration_mean",
                "confirmed_coverage_ratio_mean",
                "pipeline_fps_processed_mean",
            ]
        )

    g = success_df.groupby("profile", dropna=False)
    out = g.apply(
        lambda x: pd.Series(
            {
                "runs": int(len(x)),
                "video_duration_sec_mean": _safe_mean(x["video_duration_sec"]),
                "event_count_mean": _safe_mean(x["event_count"]),
                "false_alarms_per_hour_mean": _safe_mean(x["false_alarms_per_hour"]),
                "avg_event_duration_mean": _safe_mean(x["avg_event_duration"]),
                "confirmed_coverage_ratio_mean": _safe_mean(x["confirmed_coverage_ratio"]),
                "pipeline_fps_processed_mean": _safe_mean(x["pipeline_fps_processed"]),
            }
        )
    ).reset_index()
    return out


def _pick_best_profiles(agg: pd.DataFrame) -> dict[str, str]:
    if agg.empty:
        return {"fastest": "", "safest": "", "balanced": "", "recommended_default": ""}

    fps = pd.to_numeric(agg["pipeline_fps_processed_mean"], errors="coerce")
    fa = pd.to_numeric(agg["false_alarms_per_hour_mean"], errors="coerce")
    speed_score = _norm01(fps, higher_is_better=True)
    safety_score = _norm01(fa, higher_is_better=False)
    balanced_score = 0.5 * speed_score + 0.5 * safety_score

    scored = agg.copy()
    scored["speed_score"] = speed_score
    scored["safety_score"] = safety_score
    scored["balanced_score"] = balanced_score

    fastest_idx = int(scored["pipeline_fps_processed_mean"].fillna(-1e9).idxmax())
    safest_idx = int(scored["false_alarms_per_hour_mean"].fillna(1e9).idxmin())
    balanced_idx = int(scored["balanced_score"].fillna(-1e9).idxmax())

    fastest = str(scored.loc[fastest_idx, "profile"])
    safest = str(scored.loc[safest_idx, "profile"])
    balanced = str(scored.loc[balanced_idx, "profile"])

    # Explainable default rule:
    # prefer safest only if materially safer and not too slow; otherwise balanced.
    fastest_fps = float(scored.loc[fastest_idx, "pipeline_fps_processed_mean"]) if fastest else np.nan
    safest_fa = float(scored.loc[safest_idx, "false_alarms_per_hour_mean"]) if safest else np.nan
    balanced_fa = float(scored.loc[balanced_idx, "false_alarms_per_hour_mean"]) if balanced else np.nan
    safest_fps = float(scored.loc[safest_idx, "pipeline_fps_processed_mean"]) if safest else np.nan

    recommend_safest = False
    if np.isfinite(safest_fa) and np.isfinite(balanced_fa) and np.isfinite(safest_fps) and np.isfinite(fastest_fps):
        materially_safer = safest_fa <= (balanced_fa * 0.8)
        not_too_slow = safest_fps >= (fastest_fps * 0.6)
        recommend_safest = materially_safer and not_too_slow

    recommended = safest if recommend_safest else balanced
    return {
        "fastest": fastest,
        "safest": safest,
        "balanced": balanced,
        "recommended_default": recommended,
    }


def _format_md(
    agg: pd.DataFrame,
    picks: dict[str, str],
    failed_df: pd.DataFrame,
    input_path: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Profile Comparison Report")
    lines.append("")
    lines.append(f"Source: `{input_path}`")
    lines.append("")

    if agg.empty:
        lines.append("No successful rows found. Ranking could not be computed.")
    else:
        lines.append("## Per-Profile Summary")
        lines.append("")
        lines.append(
            "| profile | runs | fps(mean) | false_alarms_per_hour(mean) | avg_event_duration(mean) | coverage(mean) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, r in agg.sort_values("profile").iterrows():
            lines.append(
                f"| {r['profile']} | {int(r['runs'])} | "
                f"{float(r['pipeline_fps_processed_mean']):.3f} | "
                f"{float(r['false_alarms_per_hour_mean']):.3f} | "
                f"{float(r['avg_event_duration_mean']):.3f} | "
                f"{float(r['confirmed_coverage_ratio_mean']):.4f} |"
            )
        lines.append("")
        lines.append("## Ranking")
        lines.append("")
        lines.append(f"- Fastest profile: `{picks['fastest']}`")
        lines.append(f"- Safest profile (lowest false alarms/hour): `{picks['safest']}`")
        lines.append(f"- Best balanced profile: `{picks['balanced']}`")
        lines.append(f"- Recommended default profile: `{picks['recommended_default']}`")

    lines.append("")
    lines.append("## Failed Runs")
    lines.append("")
    if failed_df.empty:
        lines.append("No failed runs.")
    else:
        lines.append(f"Failed rows: {len(failed_df)}")
        lines.append("")
        preview_cols = [c for c in ["video_name", "profile", "error_message"] if c in failed_df.columns]
        preview = failed_df[preview_cols].head(20)
        for _, r in preview.iterrows():
            lines.append(
                f"- video=`{r.get('video_name', '')}` profile=`{r.get('profile', '')}` "
                f"error=`{r.get('error_message', '')}`"
            )
    lines.append("")
    return "\n".join(lines)


def build_profile_comparison(input_csv: Path) -> dict[str, Any]:
    df = pd.read_csv(input_csv)
    if "status" not in df.columns:
        raise ValueError("Input summary must contain 'status' column.")
    if "profile" not in df.columns:
        raise ValueError("Input summary must contain 'profile' column.")

    for col in ["video_duration_sec", "event_count", "false_alarms_per_hour", "avg_event_duration", "confirmed_coverage_ratio", "pipeline_fps_processed"]:
        if col not in df.columns:
            df[col] = np.nan

    success_df = df[df["status"] == "ok"].copy()
    failed_df = df[df["status"] == "failed"].copy()
    agg = _aggregate_profiles(success_df)
    picks = _pick_best_profiles(agg)
    return {
        "aggregates": agg,
        "picks": picks,
        "failed_df": failed_df,
        "success_count": int(len(success_df)),
        "failed_count": int(len(failed_df)),
        "total_count": int(len(df)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build profile comparison report from eval summary CSV.")
    ap.add_argument("--input", required=True, help="Path to eval summary CSV.")
    ap.add_argument("--output", default="outputs/profile_comparison.md", help="Output markdown report path.")
    ap.add_argument("--json_output", default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    inp = Path(args.input)
    out_md = Path(args.output)
    out_json = Path(args.json_output) if args.json_output else None
    if not inp.exists():
        raise SystemExit(f"Input CSV not found: {inp}")

    result = build_profile_comparison(inp)
    agg = result["aggregates"]
    picks = result["picks"]
    failed_df = result["failed_df"]

    out_md.parent.mkdir(parents=True, exist_ok=True)
    md = _format_md(agg=agg, picks=picks, failed_df=failed_df, input_path=inp)
    out_md.write_text(md, encoding="utf-8")

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_csv": str(inp),
            "counts": {
                "total": int(result["total_count"]),
                "success": int(result["success_count"]),
                "failed": int(result["failed_count"]),
            },
            "picks": picks,
            "aggregates": agg.to_dict(orient="records"),
            "failed_runs": failed_df.to_dict(orient="records"),
        }
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Written markdown: {out_md}")
    if out_json is not None:
        print(f"Written json: {out_json}")


if __name__ == "__main__":
    main()
