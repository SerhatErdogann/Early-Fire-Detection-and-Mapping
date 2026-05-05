"""
Central configuration for the wildfire detection project.
Inspired by Fire-Detection-UAV config-driven design.

Environment variable overrides (all optional, all backwards-compatible):
- ``FLAME_DATA_ROOT``           : where the raw datasets live (read-only is fine)
- ``FLAME_OUTPUTS_DIR``         : writable outputs directory
- ``FLAME_MODELS_DIR``          : writable checkpoints directory
- ``FLAME_MASTER_INDEX``        : full path for ``master_index.parquet``
- ``FLAME_INDEX_CSV``           : full path for ``flame_index.csv``
- ``FLAME_CART_ROOT``           : optional separate CART RGB+thermal16 root (see ``01_build_master_index``)
- ``FLAME_BINARY_ROOT``         : standalone binary dataset root (typically ``train/``, ``val/``, ``test/`` splits)

This is what makes the project run on Kaggle, where ``/kaggle/input/...`` is
read-only and only ``/kaggle/working/...`` can be written.
"""
import os
from pathlib import Path


def _path_from_env(env_name: str, default: Path) -> Path:
    """Return ``Path(env)`` if the env var is set, otherwise the default."""
    raw = os.environ.get(env_name)
    return Path(raw).expanduser() if raw else default


# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = _path_from_env("FLAME_DATA_ROOT", PROJECT_ROOT / "data")
FLAME_ROOT = DATA_ROOT / "flame3"
FLAME_VIDEO_FRAMES_ROOT = DATA_ROOT / "flame_video_frames"
OUTPUTS_DIR = _path_from_env("FLAME_OUTPUTS_DIR", PROJECT_ROOT / "outputs")
MODELS_DIR = _path_from_env("FLAME_MODELS_DIR", PROJECT_ROOT / "models")

# Index & data. Master index defaults to DATA_ROOT but on read-only mounts
# (e.g. Kaggle inputs) the env var override is required so we can write it.
FLAME_INDEX_CSV = _path_from_env("FLAME_INDEX_CSV", OUTPUTS_DIR / "flame_index.csv")
MASTER_INDEX_PARQUET = _path_from_env("FLAME_MASTER_INDEX", DATA_ROOT / "master_index.parquet")
# Optional: extra roots to scan (e.g. extracted data.rar). Paths relative to cwd ok.
CUSTOM_DATA_SCAN_ROOTS: list = []

# Repo-level multimodal RGB+thermal binary dataset (preset train/val/test folders).
BINARY_ROOT = _path_from_env("FLAME_BINARY_ROOT", DATA_ROOT / "binary")
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
    # OOM-safe defaults (override via CLI flags in src/02_train.py)
    "batch_size": 8,
    "epochs": 20,
    "lr": 1e-4,
    # Slightly stronger L2 helps generalisation on RGB–thermal fusion (fewer val FP tails).
    "weight_decay": 0.02,
    "patience": 4,
    "size": 224,
    "extra_test_ratio": 0.2,
    "val_split": 0.2,
    "flame_test_ratio": 0.1,
    "scheduler": "plateau",  # "plateau" | "cosine"
    "use_amp": True,
    "focal_gamma": 2.0,
    # Oversample rows from flame_video_nofire in WeightedRandomSampler (reduces real-world FP).
    "flame_video_nofire_weight": 1.85,
    # DataLoader memory controls (Windows-friendly)
    "num_workers": 0,
    "prefetch_factor": 1,
    "persistent_workers": False,
    "pin_memory": False,
    # Dataset decoding controls
    # - percentile: robust but slower/heavier
    # - uint16_div: fast for 16-bit thermal (x/65535)
    "thermal_norm": "percentile",
}

# Fusion-specific training overrides. Applied by src/02_train.py when --mode fusion
# is used and the corresponding CLI flag was NOT set explicitly (sentinel=None).
FUSION_TRAIN_DEFAULT = {
    "lr": 5e-5,
    "patience": 6,
    "loss_name": "cb_focal",
    "loss_mode": "sampler_focal",
    "backbone": "efficientnet_b0",
    "size": 384,
    "model_family": "dual_branch_fusion",
    "dual_branch_backbone": "resnet50",
    # Slightly higher gamma on hard examples (less overconfident fire on ambiguous backgrounds).
    "focal_gamma": 2.25,
}

# ---------- Inference ----------
INFERENCE_DEFAULT = {
    "smooth_window": 7,
    "ema_alpha": 0.30,
    "use_tta": False,
    # More usable default: sparser + more sensitive alarm
    "step_frames": 12,
    # Temporal / scene guards (video.py)
    "scene_thresh": 0.10,  # mean abs diff on normalized gray (0–1); raise if too sensitive
    "scene_conf_scale": 0.7,
    "hyst_high": 0.60,
    "hyst_low": 0.40,
    "persist_n": 4,
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
    # Adaptive frame sampling
    "adaptive_min_step": 1,
    "adaptive_max_step": 12,
    "adaptive_low_motion": 0.03,
    "adaptive_high_risk": 0.65,
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

# Minimum safe alarm threshold. Applied by trainer to clamp threshold_alarm from below
# so the checkpoint's default alarm threshold never goes below this in production.
THRESHOLD_ALARM_MIN = 0.25
