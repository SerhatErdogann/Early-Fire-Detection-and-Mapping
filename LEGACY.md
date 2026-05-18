# Legacy Code Notes

The active application code lives under `src/`, `scripts/`, `tests/`, `config.py`, and the root `requirements*.txt` files.

The following directories are retained for demos, experiments, or older integration attempts:

- `arayuzde-harita-gösterimi/`
- `model_ile_konumlu_çıktı/`
- `drone-haberlesmesi/`
- `project-showcase/`

Do not treat these directories as the source of truth for the current inference pipeline. They may use older model formats, separate database schemas, hard-coded demo paths, or optional dependencies.

Current source of truth:

- Video inference: `src/05_video_infer.py`, `src/inference/video.py`
- Live/geospatial inference: `src/live_video_fire_pipeline.py`, `src/inference/unified_pipeline.py`
- Fuel scoring: `src/risk/fuel_scorer.py`
- Risk scoring: `src/06_add_risk_score.py`, `src/risk/scoring.py`
- Tests: `tests/`
