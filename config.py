"""
Central configuration for the wildfire detection project.
Inspired by Fire-Detection-UAV config-driven design.
"""
from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
FLAME_ROOT = DATA_ROOT / "flame3"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Index & data
FLAME_INDEX_CSV = OUTPUTS_DIR / "flame_index.csv"
MASTER_INDEX_PARQUET = DATA_ROOT / "master_index.parquet"
# Optional: extra roots to scan (e.g. extracted data.rar). Paths relative to cwd ok.
CUSTOM_DATA_SCAN_ROOTS: list = []
FLAME_BINARY_ROOT = FLAME_ROOT / "binary"
FLAME_BINARY_CSV = FLAME_BINARY_ROOT / "rgbt_multimodal_data.csv"
# Under flame3: folders that contain fire/ + no fire/ (or no_fire) each with rgb + thermal
FLAME_NESTED_SCAN = ["binary", "dataset"]

# Model checkpoints (relative to PROJECT_ROOT)
CKPT_RGB = MODELS_DIR / "rgb.pt"
CKPT_THERMAL = MODELS_DIR / "thermal.pt"
CKPT_FUSION = MODELS_DIR / "fusion.pt"
CKPT_DUAL_BRANCH = MODELS_DIR / "dual_branch.pt"

# ---------- Data ----------
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_INPUT_SIZE = 384
INPUT_SIZE_OPTIONS = (224, 384)

# ---------- Training (defaults) ----------
TRAIN_DEFAULT = {
    "batch_size": 16,
    "epochs": 20,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "patience": 4,
    "size": 384,
    "extra_test_ratio": 0.2,
    "val_split": 0.2,
    "flame_test_ratio": 0.1,
    "scheduler": "plateau",  # "plateau" | "cosine"
    "use_amp": True,
    "focal_gamma": 2.0,
}

# ---------- Inference ----------
INFERENCE_DEFAULT = {
    "smooth_window": 5,
    "ema_alpha": 0.3,
    "use_tta": False,
    "step_frames": 5,
    # Temporal / scene guards (video.py)
    "scene_thresh": 0.10,  # mean abs diff on normalized gray (0–1); raise if too sensitive
    "scene_conf_scale": 0.7,
    "hyst_high": 0.7,
    "hyst_low": 0.4,
    "persist_n": 5,
    "min_component_area": 0.01,
    "growth_downscale": 0.85,
    "kl_hist_thresh": 0.35,
    # Early-detection options (video.py)
    "early_detection": False,
    "early_threshold_shift": 0.15,
    "early_min_threshold": 0.25,
    "early_persist_n": 2,
    "small_fire_boost": 1.3,
    "small_fire_area_max": 0.02,
    "growth_upscale": 1.2,
    "texture_prob_max": 0.2,
    "texture_top10_min": 0.7,
    "enable_modal_agreement": False,
    "modal_agreement_min_corr": 0.2,
    "modal_agreement_penalty": 0.6,
}

# ---------- Risk score (06_add_risk_score) ----------
RISK_WEIGHTS = {
    "prob_fire": 0.60,
    "intensity_top10": 0.25,
    "area_heat_gt_0_6": 0.15,
}
# Mask/CAM-derived + temporal features (used when columns exist)
RISK_SCORE_WEIGHTS = {
    "prob_fire_cal": 0.35,
    "peak_intensity": 0.20,
    "largest_component_area": 0.20,
    "temporal_persistence": 0.15,
    "mask_growth_rate": 0.10,
}
FIRE_EVENT_THR = 0.85
FIRE_EVENT_MIN_RUN = 5
