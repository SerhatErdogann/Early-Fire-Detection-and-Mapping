import json
import runpy
import sys
from pathlib import Path

import pandas as pd


def test_risk_script_writes_metadata(tmp_path):
    inp = tmp_path / "video_predictions.csv"
    out = tmp_path / "video_predictions_scored.csv"
    pd.DataFrame(
        {
            "frame_idx": [0, 1, 2],
            "prob_fire": [0.2, 0.8, 0.9],
            "decision_prob": [0.25, 0.85, 0.88],
            "threshold_used": [0.6, 0.6, 0.6],
        }
    ).to_csv(inp, index=False)

    old_argv = sys.argv[:]
    try:
        sys.argv = ["06_add_risk_score.py", "--inp", str(inp), "--out", str(out)]
        runpy.run_path(str(Path("src") / "06_add_risk_score.py"), run_name="__main__")
    finally:
        sys.argv = old_argv

    meta_path = out.with_suffix(".risk_meta.json")
    assert out.exists()
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "probability_column_used" in meta
    assert "persistence_threshold" in meta
