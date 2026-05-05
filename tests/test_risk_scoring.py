import pandas as pd

from src.risk.scoring import build_risk_table


def test_risk_scoring_uses_decision_prob_and_persistence():
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2, 3],
            "decision_prob": [0.2, 0.8, 0.85, 0.3],
            "prob_fire": [0.1, 0.2, 0.2, 0.1],
            "largest_component_area": [0.0, 0.01, 0.03, 0.0],
            "peak_intensity": [0.1, 0.7, 0.8, 0.2],
            "mask_growth_rate": [0.0, 0.02, 0.01, -0.1],
        }
    )
    out, meta = build_risk_table(
        df,
        risk_weights={
            "prob_fire_cal": 0.4,
            "peak_intensity": 0.2,
            "largest_component_area": 0.2,
            "temporal_persistence": 0.1,
            "mask_growth_rate": 0.1,
        },
        persistence_win=3,
        persistence_thr=0.5,
    )
    assert meta["probability_column_used"] == "decision_prob"
    assert "temporal_persistence" in out.columns
    assert float(out.loc[2, "temporal_persistence"]) >= 0.66


def test_risk_output_has_explainability_columns():
    df = pd.DataFrame({"frame_idx": [0, 1], "prob_fire": [0.9, 0.1]})
    out, _ = build_risk_table(df, risk_weights={})
    for col in ("risk_reason", "confidence_band", "alarm_state", "risk_score_norm"):
        assert col in out.columns
