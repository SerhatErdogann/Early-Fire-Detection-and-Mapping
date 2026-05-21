# Legacy Code Notes

The active application code lives under `src/`, `scripts/`, `tests/`, `config.py`, and the root `requirements*.txt` files.

The following directories are retained for demos, experiments, or older integration attempts:

- `arayuzde-harita-gösterimi/`
- `model_ile_konumlu_çıktı/`
- `drone-haberlesmesi/`
- `project-showcase/`

Do not treat these directories as the source of truth for the current inference pipeline. They may use older model formats, separate database schemas, hard-coded demo paths, or optional dependencies.

Current source of truth:

- Video inference + GIS alarm feed: `src/05_video_infer.py`, `src/inference/video.py`, `src/inference/downstream_alarm_feed.py`
- Streamlit UI: `src/07_ui.py`, `src/ui/`
- Training: `src/02_train.py`, `models/dual_branch.pt` (`dual_branch_gated_fusion`)
- Fuel scoring: `src/risk/fuel_scorer.py`
- Risk scoring: `src/06_add_risk_score.py`, `src/risk/scoring.py`
- Tests: `tests/`
