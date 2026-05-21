"""Resolve thermal mu/sigma for eval when ``thermal_norm`` is train/global z-score.

Older checkpoints omit ``thermal_mu`` / ``thermal_sigma`` in ``training_args``; values
also appear in ``outputs/metrics_*.json`` after training fixes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def thermal_norm_needs_calibration(thermal_norm: str) -> bool:
    return (thermal_norm or "").strip().lower() in {"train_zscore", "global_zscore"}


def thermal_norm_from_checkpoint(ck: dict) -> str:
    ta = ck.get("training_args") if isinstance(ck.get("training_args"), dict) else {}
    tn = ta.get("thermal_norm") if isinstance(ta.get("thermal_norm"), str) else None
    if tn and str(tn).strip():
        return str(tn).strip()
    top = ck.get("thermal_norm")
    if isinstance(top, str) and top.strip():
        return top.strip()
    return "percentile"


def _default_metrics_path(ck: dict) -> Path:
    """Same naming convention as ``src.training.trainer`` when saving metrics JSON."""
    try:
        from config import OUTPUTS_DIR
    except Exception:  # pragma: no cover
        OUTPUTS_DIR = Path("outputs")
    mode = str(ck.get("mode") or "fusion").lower()
    mf_raw = ck.get("model_family")
    mf = str(mf_raw).lower().strip() if mf_raw else "dual_branch_gated_fusion"
    return Path(OUTPUTS_DIR) / f"metrics_{mode}_{mf}.json"


def _coerce_optional_float(val) -> float | None:
    if val is None or (isinstance(val, str) and not val.strip()):
        return None
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _read_mu_sigma_from_training_args(ck: dict) -> tuple[float | None, float | None]:
    ta = ck.get("training_args") if isinstance(ck.get("training_args"), dict) else {}
    return _coerce_optional_float(ta.get("thermal_mu")), _coerce_optional_float(ta.get("thermal_sigma"))


def _read_mu_sigma_from_metrics_json(path: Path) -> tuple[float | None, float | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _coerce_optional_float(payload.get("thermal_mu")), _coerce_optional_float(
        payload.get("thermal_sigma")
    )


def resolve_thermal_calibration_or_exit(
    *,
    ck: dict,
    thermal_norm: str,
    cli_mu: float | None,
    cli_sigma: float | None,
    metrics_json: str | Path | None,
    prog: str,
) -> tuple[float | None, float | None]:
    """
    Returns ``(thermal_mu, thermal_sigma)`` for ``FlameDataset``, or ``(None, None)``
    when ``thermal_norm`` does not require them.

    ``prog`` is a short replay hint, e.g. ``python -m src.eval.robustness_eval``.
    """
    norm = str(thermal_norm or "percentile").strip()
    if not thermal_norm_needs_calibration(norm):
        return None, None

    if cli_mu is not None and cli_sigma is not None:
        return float(cli_mu), float(cli_sigma)

    if cli_mu is not None or cli_sigma is not None:
        raise SystemExit(
            "train_zscore / global_zscore requires both --thermal_mu and --thermal_sigma when overriding."
        )

    mu, sigma = _read_mu_sigma_from_training_args(ck)
    if mu is not None and sigma is not None:
        return mu, sigma

    tried_paths: list[str] = []

    path_candidates: list[Path] = []
    if metrics_json is not None:
        path_candidates.append(Path(metrics_json))
    path_candidates.append(_default_metrics_path(ck))

    seen_resolved: set[str] = set()
    paths_to_scan: list[Path] = []
    for cand in path_candidates:
        try:
            key = str(cand.resolve())
        except Exception:
            key = str(cand)
        if key not in seen_resolved:
            seen_resolved.add(key)
            paths_to_scan.append(cand)

    for p in paths_to_scan:
        tried_paths.append(str(p))
        if not p.is_file():
            continue
        try:
            mu_j, sigma_j = _read_mu_sigma_from_metrics_json(p)
        except Exception as exc:  # pragma: no cover
            tried_paths.append(f"{p} (read error: {exc})")
            continue
        if mu_j is not None and sigma_j is not None:
            print(f"[eval] Loaded thermal_mu/thermal_sigma from metrics file: {p}")
            return mu_j, sigma_j
        tried_paths.append(f"{p} (missing thermal_mu or thermal_sigma)")

    msg = (
        f"thermal_norm={norm!r} requires thermal_mu and thermal_sigma, but they were not found "
        f"in checkpoint training_args nor in metrics JSON.\n\n"
        f"Tried: {', '.join(tried_paths) if tried_paths else '(no paths)'}\n\n"
        f"Fix options:\n"
        f"  1. Re-run training with this repo — metrics and checkpoint save mu/sigma automatically.\n"
        f"  2. Point to the correct metrics JSON:\n"
        f"       {prog} ... --metrics_json outputs/metrics_fusion_dual_branch_gated_fusion.json\n"
        f"  3. Pass values explicitly (from trainer log line \"thermal train_zscore\"):\n"
        f"       {prog} ... --thermal_mu <MU> --thermal_sigma <SIGMA>\n"
    )
    raise SystemExit(msg)
