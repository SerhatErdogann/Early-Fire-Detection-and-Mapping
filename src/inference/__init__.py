from .downstream_alarm_feed import (
    ALARM_FEED_COLUMNS,
    SCHEMA_VERSION,
    export_alarm_feed_bundle,
    load_alarm_feed_csv,
)


def __getattr__(name):
    if name == "load_checkpoint":
        from .model_loader import load_checkpoint

        return load_checkpoint
    if name in {"prep_rgb", "prep_thermal"}:
        from .preprocess import prep_rgb, prep_thermal

        return {"prep_rgb": prep_rgb, "prep_thermal": prep_thermal}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ALARM_FEED_COLUMNS",
    "SCHEMA_VERSION",
    "export_alarm_feed_bundle",
    "load_alarm_feed_csv",
    "load_checkpoint",
    "prep_rgb",
    "prep_thermal",
]
