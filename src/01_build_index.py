# Build dataset index (delegates to master builder: Parquet + legacy CSV).
# Run from project root: python src/01_build_index.py
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.build_master_index import build_master_index

try:
    from config import FLAME_INDEX_CSV, CUSTOM_DATA_SCAN_ROOTS
except ImportError:
    FLAME_INDEX_CSV = Path("outputs/flame_index.csv")
    CUSTOM_DATA_SCAN_ROOTS = []

if __name__ == "__main__":
    roots = [Path(p) for p in CUSTOM_DATA_SCAN_ROOTS if p]
    build_master_index(out_legacy_csv=FLAME_INDEX_CSV, scan_custom_roots=roots or None)
